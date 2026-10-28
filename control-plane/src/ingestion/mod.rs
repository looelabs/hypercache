pub mod nfs;
pub mod s3;
pub mod striper;
pub mod synthetic;

use anyhow::{Context, Result};
use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;
use tracing::info;

use crate::config::Config;
use crate::metadata::MetadataStore;
use crate::protocol::{ChunkDescriptor, StorageTier};
use crate::ram::RamMemoryPool;
use nfs::NfsPosixFetcher;
use s3::S3Fetcher;
use striper::Striper;

/// Authoritative backend an URI resolves to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    S3,
    Nfs,
}

/// Result of a completed ingest (or cache hit)
#[derive(Debug, Clone)]
pub struct IngestedObject {
    pub object_size: u64,
    pub chunk_size: usize,
    pub layout: Vec<ChunkDescriptor>,
    pub generation: u64,
}

#[derive(Clone)]
pub struct IngestionEngine {
    s3_fetcher: S3Fetcher,
    nfs_fetcher: NfsPosixFetcher,
    striper: Striper,
    metadata: MetadataStore,
    chunk_size: usize,
    authoritative_backend: String,
    allow_synthetic_globally: bool,
    multipart_threshold: u64,
    // Per-URI ingest locks: concurrent requests for the same object wait for one download
    inflight: Arc<std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
}

impl IngestionEngine {
    pub fn new(config: Config, metadata: MetadataStore, ram_pool: Option<Arc<RamMemoryPool>>) -> Self {
        metadata.set_reclaim_grace(std::time::Duration::from_secs(config.reclaim_grace_secs));
        let chunk_size = config.chunk_size_bytes;
        let authoritative_backend = config.authoritative_backend.clone();
        let allow_synthetic_globally = config.allow_synthetic_s3_fallback;
        let multipart_threshold = config.s3_multipart_threshold_bytes;
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
            allow_synthetic_globally,
            multipart_threshold,
            inflight: Arc::new(std::sync::Mutex::new(HashMap::new())),
        }
    }

    pub fn backend_for(&self, uri: &str) -> Backend {
        if uri.starts_with("file://")
            || uri.starts_with("nfs://")
            || uri.starts_with("hypercache://file/")
            || uri.starts_with("hypercache://nfs/")
            || uri.starts_with('/')
        {
            Backend::Nfs
        } else if uri.starts_with("s3://")
            || uri.starts_with("hypercache://s3/")
            || uri.starts_with("http://")
            || uri.starts_with("https://")
        {
            Backend::S3
        } else if self.authoritative_backend.eq_ignore_ascii_case("nfs") {
            Backend::Nfs
        } else {
            Backend::S3
        }
    }

    /// Synthetic payloads are served only when enabled globally, or when the client explicitly
    /// asks for them on a URI in the reserved benchmark namespace.
    pub fn synthetic_allowed(&self, uri: &str, requested: bool) -> bool {
        self.allow_synthetic_globally || (requested && uri.starts_with(synthetic::SYNTHETIC_URI_PREFIX))
    }

    /// Acquires the per-URI ingest lock. The map entry is removed when the last holder finishes.
    async fn lock_uri(&self, uri: &str) -> UriGuard {
        let lock = {
            let mut map = self.inflight.lock().unwrap();
            Arc::clone(map.entry(uri.to_string()).or_default())
        };
        let guard = Arc::clone(&lock).lock_owned().await;
        UriGuard {
            _guard: guard,
            lock,
            uri: uri.to_string(),
            map: Arc::clone(&self.inflight),
        }
    }

    /// Returns the cached layout if present; otherwise ingests it. Concurrent callers for the
    /// same URI share a single download. The bool is true for a cache hit.
    pub async fn get_or_ingest(
        &self,
        uri: &str,
        target_tier: Option<StorageTier>,
        pinned: bool,
        allow_synthetic: bool,
    ) -> Result<(IngestedObject, bool)> {
        if let Some(meta) = self.metadata.get_layout(uri).await? {
            return Ok((Self::from_meta(meta), true));
        }
        let _guard = self.lock_uri(uri).await;
        // Another caller may have completed the ingest while we waited
        if let Some(meta) = self.metadata.get_layout(uri).await? {
            return Ok((Self::from_meta(meta), true));
        }
        let obj = self.ingest_locked(uri, target_tier, pinned, allow_synthetic).await?;
        Ok((obj, false))
    }

    fn from_meta(meta: crate::metadata::StoredMetadata) -> IngestedObject {
        IngestedObject {
            object_size: meta.object_size,
            chunk_size: meta.chunk_size,
            layout: meta.chunk_layout,
            generation: meta.generation,
        }
    }

    /// Ingests (or re-ingests) an authoritative object by streaming it in fixed chunks from
    /// S3 or NFS/POSIX and striping each chunk across RAM or raw NVMe blocks.
    pub async fn ingest_object_full(
        &self,
        uri: &str,
        target_tier: Option<StorageTier>,
        pinned: bool,
        allow_synthetic: bool,
    ) -> Result<IngestedObject> {
        let _guard = self.lock_uri(uri).await;
        self.ingest_locked(uri, target_tier, pinned, allow_synthetic).await
    }

    async fn ingest_locked(
        &self,
        uri: &str,
        target_tier: Option<StorageTier>,
        pinned: bool,
        allow_synthetic: bool,
    ) -> Result<IngestedObject> {
        let allow_synthetic = self.synthetic_allowed(uri, allow_synthetic);
        let backend = self.backend_for(uri);
        info!(
            "Beginning streaming ingestion for {} (backend: {:?}, target tier: {:?}, pinned: {})",
            uri, backend, target_tier, pinned
        );

        // Chunks may complete out of order with parallel ranged GETs
        let chunks = Arc::new(tokio::sync::Mutex::new(BTreeMap::<usize, ChunkDescriptor>::new()));
        let on_chunk = {
            let striper = self.striper.clone();
            let uri_owned = uri.to_string();
            let chunks = Arc::clone(&chunks);
            move |idx: usize, data: Vec<u8>| {
                let striper = striper.clone();
                let uri_c = uri_owned.clone();
                let chunks = Arc::clone(&chunks);
                async move {
                    let desc = striper.stripe_single_chunk(&uri_c, idx, &data, target_tier).await?;
                    chunks.lock().await.insert(idx, desc);
                    Ok(())
                }
            }
        };

        let stream_res = match backend {
            Backend::Nfs => {
                self.nfs_fetcher
                    .fetch_object_streaming(uri, self.chunk_size, allow_synthetic, on_chunk)
                    .await
            }
            Backend::S3 => {
                self.s3_fetcher
                    .fetch_object_streaming(uri, self.chunk_size, allow_synthetic, on_chunk)
                    .await
            }
        };

        let collected: Vec<(usize, ChunkDescriptor)> = std::mem::take(&mut *chunks.lock().await).into_iter().collect();
        let final_layout: Vec<ChunkDescriptor> = collected.iter().map(|(_, c)| c.clone()).collect();

        let object_size = match stream_res {
            Ok(sz) => sz,
            Err(e) => {
                // Nothing was published: release every chunk written before the failure
                self.metadata.release_unpublished(&final_layout).await;
                return Err(e);
            }
        };

        let contiguous = collected.iter().enumerate().all(|(i, (idx, _))| i == *idx);
        let layout_bytes: u64 = final_layout.iter().map(|c| c.length).sum();
        if !contiguous || layout_bytes != object_size {
            self.metadata.release_unpublished(&final_layout).await;
            anyhow::bail!(
                "Ingest of {} produced an inconsistent layout ({} bytes in chunks, {} bytes streamed)",
                uri,
                layout_bytes,
                object_size
            );
        }

        // Durability barrier: data must be on stable storage before the layout is published
        if let Err(e) = self.striper.sync_chunks(&final_layout).await {
            self.metadata.release_unpublished(&final_layout).await;
            return Err(e);
        }

        let generation = self
            .metadata
            .save_layout_full(uri, object_size, self.chunk_size, final_layout.clone(), None, pinned)
            .await?;

        let tier_name = if final_layout.first().is_some_and(|c| c.tier == StorageTier::Ram) {
            "Tier 1 (RAM Shared Memory)"
        } else {
            "Tier 2 (Striped NVMe)"
        };

        info!(
            "Streaming ingestion completed for {}: {} bytes in {} chunks (target: {}, source: {:?}, pinned: {})",
            uri,
            object_size,
            final_layout.len(),
            tier_name,
            backend,
            pinned
        );

        Ok(IngestedObject {
            object_size,
            chunk_size: self.chunk_size,
            layout: final_layout,
            generation,
        })
    }

    pub async fn ingest_object_with_tier(&self, uri: &str, target_tier: Option<StorageTier>) -> Result<IngestedObject> {
        self.ingest_object_full(uri, target_tier, false, false).await
    }

    pub async fn ingest_object(&self, uri: &str) -> Result<IngestedObject> {
        self.ingest_object_with_tier(uri, None).await
    }

    pub fn striper(&self) -> &Striper {
        &self.striper
    }

    #[allow(dead_code)]
    pub fn s3_fetcher(&self) -> &S3Fetcher {
        &self.s3_fetcher
    }

    /// Reads one chunk's payload from its storage tier (blocking I/O off the async runtime).
    pub async fn read_chunk(&self, chunk: &ChunkDescriptor) -> Result<Vec<u8>> {
        let chunk = chunk.clone();
        tokio::task::spawn_blocking(move || read_chunk_blocking(&chunk, 0, chunk.length))
            .await
            .context("Chunk read task panicked")?
    }

    /// Reads `len` bytes starting at logical object offset `start` across a chunk layout.
    pub async fn read_range(&self, layout: &[ChunkDescriptor], start: u64, len: u64) -> Result<Vec<u8>> {
        let layout = layout.to_vec();
        tokio::task::spawn_blocking(move || -> Result<Vec<u8>> {
            let mut out = Vec::with_capacity(len as usize);
            let end = start + len;
            let mut chunk_start = 0u64;
            for chunk in &layout {
                let chunk_end = chunk_start + chunk.length;
                if chunk_end > start && chunk_start < end {
                    let from = start.max(chunk_start) - chunk_start;
                    let to = end.min(chunk_end) - chunk_start;
                    out.extend_from_slice(&read_chunk_blocking(chunk, from, to - from)?);
                }
                chunk_start = chunk_end;
                if chunk_start >= end {
                    break;
                }
            }
            if out.len() as u64 != len {
                anyhow::bail!("Range read returned {} bytes, expected {}", out.len(), len);
            }
            Ok(out)
        })
        .await
        .context("Range read task panicked")?
    }

    /// Reads chunk payload back into a contiguous buffer (small objects only; large objects
    /// are streamed with `read_chunk` / `read_range`).
    pub async fn read_chunks_to_bytes(&self, layout: &[ChunkDescriptor]) -> Result<Vec<u8>> {
        let total: u64 = layout.iter().map(|c| c.length).sum();
        self.read_range(layout, 0, total).await
    }

    /// Reads a staged layout back and returns its SHA-256 (hex) and per-chunk CRC32 values.
    pub async fn digest_layout(&self, layout: &[ChunkDescriptor]) -> Result<(String, Vec<u32>)> {
        use sha2::{Digest, Sha256};
        let mut hasher = Sha256::new();
        let mut crcs = Vec::with_capacity(layout.len());
        for chunk in layout {
            let data = self.read_chunk(chunk).await?;
            hasher.update(&data);
            crcs.push(crc32fast::hash(&data));
        }
        Ok((hex::encode(hasher.finalize()), crcs))
    }

    /// Flushes staged/committed data from local chunks to the authoritative S3 / NFS backend
    /// without buffering large objects in memory.
    pub async fn flush_to_backend(&self, uri: &str, layout: &[ChunkDescriptor]) -> Result<()> {
        let total: u64 = layout.iter().map(|c| c.length).sum();

        match self.backend_for(uri) {
            Backend::Nfs => {
                let path = self.nfs_fetcher.resolve_path(uri)?;
                if let Some(parent) = path.parent() {
                    std::fs::create_dir_all(parent)
                        .with_context(|| format!("Failed to create NFS directory {:?}", parent))?;
                }
                // Re-check after creating directories: a symlinked parent must not escape the roots
                let path = self.nfs_fetcher.resolve_path(uri)?;
                let tmp_path = path.with_file_name(format!(
                    ".{}.hypercache.{}.tmp",
                    path.file_name()
                        .map(|n| n.to_string_lossy().to_string())
                        .unwrap_or_default(),
                    std::process::id()
                ));
                let write_result: Result<()> = async {
                    let mut file = std::fs::File::create(&tmp_path)
                        .with_context(|| format!("Failed to create NFS temp file {:?}", tmp_path))?;
                    for chunk in layout {
                        let data = self.read_chunk(chunk).await?;
                        std::io::Write::write_all(&mut file, &data)?;
                    }
                    file.sync_all()?;
                    drop(file);
                    std::fs::rename(&tmp_path, &path)
                        .with_context(|| format!("Failed to move {:?} into place at {:?}", tmp_path, path))?;
                    Ok(())
                }
                .await;
                if write_result.is_err() {
                    let _ = std::fs::remove_file(&tmp_path);
                }
                write_result.with_context(|| format!("Failed to write authoritative data to NFS file {:?}", path))
            }
            Backend::S3 => {
                if total <= self.multipart_threshold {
                    let data = self.read_range(layout, 0, total).await?;
                    self.s3_fetcher.upload_object(uri, &data).await
                } else {
                    let part_size = S3Fetcher::multipart_part_size(total, self.chunk_size as u64);
                    self.s3_fetcher
                        .upload_object_multipart(uri, total, part_size, |start, len| {
                            self.read_range(layout, start, len)
                        })
                        .await
                }
            }
        }
    }
}

struct UriGuard {
    _guard: tokio::sync::OwnedMutexGuard<()>,
    lock: Arc<tokio::sync::Mutex<()>>,
    uri: String,
    map: Arc<std::sync::Mutex<HashMap<String, Arc<tokio::sync::Mutex<()>>>>>,
}

impl Drop for UriGuard {
    fn drop(&mut self) {
        let mut map = self.map.lock().unwrap();
        // Map + this guard's handle (+ the owned guard's internal handle) = no other waiters
        if Arc::strong_count(&self.lock) <= 3 {
            if let Some(existing) = map.get(&self.uri) {
                if Arc::ptr_eq(existing, &self.lock) {
                    map.remove(&self.uri);
                }
            }
        }
    }
}

/// Reads `len` bytes at `offset_in_chunk` of a chunk from RAM (shm file) or an NVMe device.
pub fn read_chunk_blocking(chunk: &ChunkDescriptor, offset_in_chunk: u64, len: u64) -> Result<Vec<u8>> {
    use std::os::unix::fs::FileExt;
    if offset_in_chunk + len > chunk.length {
        anyhow::bail!("Read past end of chunk on {}", chunk.device);
    }
    let file =
        std::fs::File::open(&chunk.device).with_context(|| format!("Failed to open chunk storage {}", chunk.device))?;
    let mut buf = vec![0u8; len as usize];
    let physical = if chunk.tier == StorageTier::Ram {
        0
    } else {
        chunk.offset
    } + offset_in_chunk;
    file.read_exact_at(&mut buf, physical)
        .with_context(|| format!("Failed to read chunk at offset {} on {}", physical, chunk.device))?;
    Ok(buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "hypercache_ingest_{}_{}",
            tag,
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn engine_for(dir: &std::path::Path, chunk_size: usize) -> (IngestionEngine, MetadataStore, String) {
        let raw_dev = dir.join("dev0.raw");
        std::fs::write(&raw_dev, vec![0u8; 32 * 1024 * 1024]).unwrap();
        let config = Config {
            block_devices: vec![raw_dev.to_str().unwrap().to_string()],
            chunk_size_bytes: chunk_size,
            device_start_offset_bytes: 4096,
            auto_create_mock_devices: false,
            nfs_mount_path: dir.to_string_lossy().to_string(),
            ..Config::default()
        };
        let metadata = MetadataStore::new(10 * 1024 * 1024 * 1024, None, None);
        let engine = IngestionEngine::new(config, metadata.clone(), None);
        (engine, metadata, raw_dev.to_string_lossy().to_string())
    }

    #[tokio::test]
    async fn test_ingestion_engine_nfs_file_routing() {
        let test_dir = temp_dir("routing");
        let (engine, metadata, _) = engine_for(&test_dir, 4 * 1024 * 1024);

        let nfs_source_file = test_dir.join("checkpoint.pt");
        let nfs_payload = b"AUTHORITATIVE_NFS_DATA_FOR_HYPERCACHE_INGESTION_TEST_STREAM";
        std::fs::write(&nfs_source_file, nfs_payload).unwrap();

        let uri = format!("file://{}", nfs_source_file.display());
        let obj = engine.ingest_object(&uri).await.unwrap();

        assert_eq!(obj.object_size, nfs_payload.len() as u64);
        assert_eq!(obj.chunk_size, 4 * 1024 * 1024);
        assert_eq!(obj.layout.len(), 1);
        assert_eq!(obj.layout[0].checksum, Some(crc32fast::hash(nfs_payload)));

        let lookup = metadata.get_layout(&uri).await.unwrap().unwrap();
        assert_eq!(lookup.object_size, nfs_payload.len() as u64);
        assert_eq!(lookup.chunk_layout, obj.layout);
        assert_eq!(lookup.generation, obj.generation);

        assert_eq!(engine.read_chunks_to_bytes(&obj.layout).await.unwrap(), nfs_payload);

        let _ = std::fs::remove_dir_all(&test_dir);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_locates_trigger_single_ingest() {
        let test_dir = temp_dir("singleflight");
        let (engine, metadata, _) = engine_for(&test_dir, 1024 * 1024);
        let src = test_dir.join("model.bin");
        let payload: Vec<u8> = (0..3 * 1024 * 1024 + 5).map(|i| (i % 253) as u8).collect();
        std::fs::write(&src, &payload).unwrap();
        let uri = format!("file://{}", src.display());

        let mut handles = Vec::new();
        for _ in 0..8 {
            let e = engine.clone();
            let u = uri.clone();
            handles.push(tokio::spawn(async move {
                e.get_or_ingest(&u, None, false, false).await.unwrap()
            }));
        }
        let mut misses = 0;
        let mut generations = std::collections::HashSet::new();
        for h in handles {
            let (obj, hit) = h.await.unwrap();
            if !hit {
                misses += 1;
            }
            generations.insert(obj.generation);
        }
        // Eight concurrent ranks: exactly one download, everyone sees the same layout
        assert_eq!(misses, 1);
        assert_eq!(generations.len(), 1);
        assert_eq!(metadata.list_entries().await.0.len(), 1);
        assert!(engine.inflight.lock().unwrap().is_empty());

        let _ = std::fs::remove_dir_all(&test_dir);
    }

    #[tokio::test]
    async fn test_synthetic_fallback_requires_explicit_opt_in() {
        let test_dir = temp_dir("synthetic");
        let (engine, _, _) = engine_for(&test_dir, 4 * 1024 * 1024);

        // A production-looking URI that happens to contain "models/benchmark/" must never get fake data
        let prod_uri = "hypercache://prod/models/benchmark/weights_1mb.safetensors";
        assert!(!engine.synthetic_allowed(prod_uri, true));
        assert!(engine.ingest_object_full(prod_uri, None, false, true).await.is_err());

        // The reserved benchmark namespace serves synthetic data only when the client asks for it
        let bench_uri = "hypercache://models/benchmark/nvme_weights_1mb.safetensors";
        assert!(!engine.synthetic_allowed(bench_uri, false));
        assert!(engine.synthetic_allowed(bench_uri, true));

        let _ = std::fs::remove_dir_all(&test_dir);
    }

    #[tokio::test]
    async fn test_flush_to_nfs_streams_and_confines_path() {
        let test_dir = temp_dir("flush");
        let (engine, _, _) = engine_for(&test_dir, 1024 * 1024);
        let data: Vec<u8> = (0..2 * 1024 * 1024 + 99).map(|i| (i % 241) as u8).collect();
        let layout = engine
            .striper()
            .stripe_bytes_to_tier("hypercache://ckpt", &data, Some(StorageTier::Nvme))
            .await
            .unwrap();

        let target = test_dir.join("out/ckpt.pt");
        engine
            .flush_to_backend(&format!("file://{}", target.display()), &layout)
            .await
            .unwrap();
        assert_eq!(std::fs::read(&target).unwrap(), data);

        // Write-through to a path outside the allowed roots is refused
        assert!(engine
            .flush_to_backend("file:///etc/hypercache_should_not_exist", &layout)
            .await
            .is_err());

        let (sha, crcs) = engine.digest_layout(&layout).await.unwrap();
        use sha2::Digest;
        assert_eq!(sha, hex::encode(sha2::Sha256::digest(&data)));
        assert_eq!(crcs.len(), 3);

        let _ = std::fs::remove_dir_all(&test_dir);
    }
}
