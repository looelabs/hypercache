pub mod nfs;
pub mod s3;
pub mod striper;

use anyhow::Result;
use tracing::info;

use crate::config::Config;
use crate::metadata::MetadataStore;
use std::sync::Arc;
use crate::protocol::{ChunkDescriptor, StorageTier};
use crate::ram::RamMemoryPool;
use nfs::NfsPosixFetcher;
use s3::S3Fetcher;
use striper::Striper;

#[derive(Clone)]
pub struct IngestionEngine {
    s3_fetcher: S3Fetcher,
    nfs_fetcher: NfsPosixFetcher,
    striper: Striper,
    metadata: MetadataStore,
    chunk_size: usize,
    authoritative_backend: String,
}

impl IngestionEngine {
    pub fn new(config: Config, metadata: MetadataStore, ram_pool: Option<Arc<RamMemoryPool>>) -> Self {
        let chunk_size = config.chunk_size_bytes;
        let authoritative_backend = config.authoritative_backend.clone();
        let s3_fetcher = S3Fetcher::new(config.clone());
        let nfs_fetcher = NfsPosixFetcher::new(config.clone());
        let striper = Striper::new(config, metadata.clone(), ram_pool);
        Self {
            s3_fetcher,
            nfs_fetcher,
            striper,
            metadata,
            chunk_size,
            authoritative_backend,
        }
    }

    /// Ingests an authoritative object by streaming it in fixed 4 MiB chunks from S3 or NFS/POSIX
    /// and striping each chunk directly across RAM or raw NVMe blocks without full-object memory buffering.
    pub async fn ingest_object_full(
        &self,
        uri: &str,
        target_tier: Option<StorageTier>,
        pinned: bool,
        allow_synthetic: bool,
    ) -> Result<(u64, usize, Vec<ChunkDescriptor>)> {
        info!(
            "Beginning streaming ingestion for {} (target tier: {:?}, pinned: {})",
            uri, target_tier, pinned
        );

        let is_nfs = if uri.starts_with("file://")
            || uri.starts_with("nfs://")
            || uri.starts_with("hypercache://file/")
            || uri.starts_with("hypercache://nfs/")
            || uri.starts_with('/')
        {
            true
        } else if uri.starts_with("s3://")
            || uri.starts_with("hypercache://s3/")
            || uri.starts_with("http://")
            || uri.starts_with("https://")
        {
            false
        } else {
            self.authoritative_backend.to_lowercase() == "nfs"
        };

        let chunk_layout = Arc::new(tokio::sync::Mutex::new(Vec::<ChunkDescriptor>::new()));
        let striper = self.striper.clone();
        let uri_owned = uri.to_string();
        let layout_ref = Arc::clone(&chunk_layout);

        let stream_res = if is_nfs {
            self.nfs_fetcher
                .fetch_object_streaming(uri, self.chunk_size, allow_synthetic, |idx, slice| {
                    let striper_c = striper.clone();
                    let uri_c = uri_owned.clone();
                    let layout_c = Arc::clone(&layout_ref);
                    let chunk_vec = slice.to_vec();
                    async move {
                        let desc = striper_c
                            .stripe_single_chunk(&uri_c, idx, &chunk_vec, target_tier)
                            .await?;
                        layout_c.lock().await.push(desc);
                        Ok(())
                    }
                })
                .await
        } else {
            self.s3_fetcher
                .fetch_object_streaming(uri, self.chunk_size, allow_synthetic, |idx, slice| {
                    let striper_c = striper.clone();
                    let uri_c = uri_owned.clone();
                    let layout_c = Arc::clone(&layout_ref);
                    let chunk_vec = slice.to_vec();
                    async move {
                        let desc = striper_c
                            .stripe_single_chunk(&uri_c, idx, &chunk_vec, target_tier)
                            .await?;
                        layout_c.lock().await.push(desc);
                        Ok(())
                    }
                })
                .await
        };

        let object_size = match stream_res {
            Ok(sz) => sz,
            Err(e) => {
                // Reclaim any chunks allocated before failure
                let partial = chunk_layout.lock().await.clone();
                if !partial.is_empty() {
                    self.metadata.reclaim_chunks(&partial).await;
                }
                return Err(e);
            }
        };

        let final_layout = chunk_layout.lock().await.clone();

        self.metadata
            .save_layout_full(
                uri,
                object_size,
                self.chunk_size,
                final_layout.clone(),
                None,
                pinned,
            )
            .await?;

        let tier_name = if final_layout
            .first()
            .map_or(false, |c| c.tier == StorageTier::Ram)
        {
            "Tier 1 (RAM Shared Memory)"
        } else {
            "Tier 2 (Striped NVMe)"
        };

        info!(
            "Streaming ingestion completed for {}: {} bytes in {} chunks (target: {}, source: {}, pinned: {})",
            uri,
            object_size,
            final_layout.len(),
            tier_name,
            if is_nfs { "NFS/POSIX" } else { "S3" },
            pinned
        );

        Ok((object_size, self.chunk_size, final_layout))
    }

    pub async fn ingest_object_with_tier(
        &self,
        uri: &str,
        target_tier: Option<StorageTier>,
    ) -> Result<(u64, usize, Vec<ChunkDescriptor>)> {
        self.ingest_object_full(uri, target_tier, false, false).await
    }

    pub async fn ingest_object(&self, uri: &str) -> Result<(u64, usize, Vec<ChunkDescriptor>)> {
        self.ingest_object_with_tier(uri, None).await
    }

    pub fn striper(&self) -> &Striper {
        &self.striper
    }

    #[allow(dead_code)]
    pub fn s3_fetcher(&self) -> &S3Fetcher {
        &self.s3_fetcher
    }

    /// Reads chunk payload from physical storage back into a contiguous in-memory byte buffer
    pub async fn read_chunks_to_bytes(&self, layout: &[ChunkDescriptor]) -> Result<Vec<u8>> {
        use anyhow::Context;
        let total_size: usize = layout.iter().map(|c| c.length as usize).sum();
        let mut buffer = vec![0u8; total_size];
        let mut current_offset = 0;

        for chunk in layout {
            let chunk_len = chunk.length as usize;
            let slice = &mut buffer[current_offset..current_offset + chunk_len];

            if chunk.tier == StorageTier::Ram {
                let data = std::fs::read(&chunk.device)
                    .with_context(|| format!("Failed to read RAM shared memory chunk from {}", chunk.device))?;
                slice.copy_from_slice(&data[..chunk_len.min(data.len())]);
            } else {
                use std::fs::File;
                use std::os::unix::fs::FileExt;
                let file = File::open(&chunk.device)
                    .with_context(|| format!("Failed to open device {}", chunk.device))?;
                file.read_exact_at(slice, chunk.offset)
                    .with_context(|| format!("Failed to read chunk at offset {} on {}", chunk.offset, chunk.device))?;
            }

            current_offset += chunk_len;
        }

        Ok(buffer)
    }

    /// Flushes staged/committed data from local storage chunks up to authoritative S3 / NFS backend
    pub async fn flush_to_backend(&self, uri: &str, layout: &[ChunkDescriptor]) -> Result<()> {
        use anyhow::Context;
        let data = self.read_chunks_to_bytes(layout).await?;
        let is_nfs = if uri.starts_with("file://")
            || uri.starts_with("nfs://")
            || uri.starts_with("hypercache://file/")
            || uri.starts_with("hypercache://nfs/")
            || uri.starts_with('/')
        {
            true
        } else if uri.starts_with("s3://")
            || uri.starts_with("hypercache://s3/")
            || uri.starts_with("http://")
            || uri.starts_with("https://")
        {
            false
        } else {
            self.authoritative_backend.to_lowercase() == "nfs"
        };

        if is_nfs {
            let path = self.nfs_fetcher.parse_path(uri);
            if let Some(parent) = path.parent() {
                let _ = std::fs::create_dir_all(parent);
            }
            std::fs::write(&path, data)
                .with_context(|| format!("Failed to write authoritative data to NFS file {:?}", path))?;
        } else {
            self.s3_fetcher.upload_object(uri, &data).await?;
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_ingestion_engine_nfs_file_routing() {
        let test_dir = std::env::temp_dir().join(format!("hypercache_ingest_test_{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)));
        std::fs::create_dir_all(&test_dir).unwrap();

        let raw_dev = test_dir.join("dev0.raw");
        std::fs::write(&raw_dev, vec![0u8; 10 * 1024 * 1024]).unwrap();

        let nfs_source_file = test_dir.join("checkpoint.pt");
        let nfs_payload = b"AUTHORITATIVE_NFS_DATA_FOR_HYPERCACHE_INGESTION_TEST_STREAM";
        std::fs::write(&nfs_source_file, nfs_payload).unwrap();

        let config = Config {
            block_devices: vec![raw_dev.to_str().unwrap().to_string()],
            chunk_size_bytes: 4 * 1024 * 1024,
            device_start_offset_bytes: 4096,
            auto_create_mock_devices: false,
            ..Config::default()
        };

        let metadata = MetadataStore::new(10 * 1024 * 1024 * 1024, None, None);
        let engine = IngestionEngine::new(config, metadata.clone(), None);

        let uri = format!("file://{}", nfs_source_file.display());
        let (obj_size, chunk_size, chunks) = engine.ingest_object(&uri).await.unwrap();

        assert_eq!(obj_size, nfs_payload.len() as u64);
        assert_eq!(chunk_size, 4 * 1024 * 1024);
        assert_eq!(chunks.len(), 1);

        // Verify layout in metadata store
        let lookup = metadata.get_layout(&uri).await.unwrap().unwrap();
        assert_eq!(lookup.object_size, nfs_payload.len() as u64);
        assert_eq!(lookup.chunk_layout, chunks);

        let _ = std::fs::remove_dir_all(&test_dir);
    }
}
