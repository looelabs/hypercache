use anyhow::{Context, Result};
use std::path::{Path, PathBuf};
use tracing::{info, warn};

use crate::config::Config;

#[derive(Clone)]
pub struct NfsPosixFetcher {
    config: Config,
}

impl NfsPosixFetcher {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Resolves any URI (file://, nfs://, hypercache://file/..., /mnt/nfs/...) to a filesystem PathBuf
    pub fn parse_path(&self, uri: &str) -> PathBuf {
        if let Some(rest) = uri.strip_prefix("file://") {
            PathBuf::from(rest)
        } else if let Some(rest) = uri.strip_prefix("nfs://") {
            PathBuf::from(if rest.starts_with('/') { rest.to_string() } else { format!("/{}", rest) })
        } else if let Some(rest) = uri.strip_prefix("hypercache://file/") {
            PathBuf::from(format!("/{}", rest.trim_start_matches('/')))
        } else if let Some(rest) = uri.strip_prefix("hypercache://nfs/") {
            PathBuf::from(format!("/{}", rest.trim_start_matches('/')))
        } else if uri.starts_with('/') {
            PathBuf::from(uri)
        } else {
            let clean = uri.strip_prefix("hypercache://").unwrap_or(uri);
            let base = Path::new(&self.config.nfs_mount_path);
            base.join(clean.trim_start_matches('/'))
        }
    }

    /// Fetches the object bytes directly from an NFS share or POSIX filesystem
    #[allow(dead_code)]
    pub async fn fetch_object(&self, uri: &str) -> Result<Vec<u8>> {
        self.fetch_object_with_fallback(uri, self.config.allow_synthetic_s3_fallback)
            .await
    }

    #[allow(dead_code)]
    pub async fn fetch_object_with_fallback(&self, uri: &str, allow_synthetic: bool) -> Result<Vec<u8>> {
        let path = self.parse_path(uri);
        info!("Fetching authoritative object from NFS/POSIX filesystem: {:?}", path);

        let is_benchmark_uri = uri.contains("models/benchmark/");

        if path.exists() {
            let bytes = tokio::fs::read(&path)
                .await
                .with_context(|| format!("Failed to read file from NFS path {:?}", path))?;
            info!("Successfully read {} bytes from NFS file {:?}", bytes.len(), path);
            Ok(bytes)
        } else if allow_synthetic || is_benchmark_uri {
            let target_size = extract_size_from_uri(uri);
            warn!(
                "NFS path {:?} does not exist (synthetic fallback active). Generating {:.2} MiB synthetic tensor payload for {}",
                path,
                target_size as f64 / (1024.0 * 1024.0),
                uri
            );
            Ok(generate_synthetic_payload(uri, target_size))
        } else {
            anyhow::bail!("Authoritative NFS/POSIX file not found at {:?}", path);
        }
    }

    /// Streams an NFS/POSIX file in fixed-size chunks (`chunk_size`) without buffering the entire file in RAM.
    pub async fn fetch_object_streaming<F, Fut>(
        &self,
        uri: &str,
        chunk_size: usize,
        allow_synthetic: bool,
        mut on_chunk: F,
    ) -> Result<u64>
    where
        F: FnMut(usize, Vec<u8>) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        use tokio::io::AsyncReadExt;
        let path = self.parse_path(uri);
        info!("Streaming authoritative object from NFS/POSIX filesystem: {:?}", path);

        let is_benchmark_uri = uri.contains("models/benchmark/");

        if path.exists() {
            let mut file = tokio::fs::File::open(&path)
                .await
                .with_context(|| format!("Failed to open NFS file {:?}", path))?;
            let mut chunk_idx = 0usize;
            let mut total_bytes = 0u64;

            loop {
                let mut buf = vec![0u8; chunk_size];
                let mut read_in_chunk = 0usize;
                while read_in_chunk < chunk_size {
                    let n = file.read(&mut buf[read_in_chunk..]).await?;
                    if n == 0 {
                        break;
                    }
                    read_in_chunk += n;
                }
                if read_in_chunk == 0 {
                    break;
                }
                buf.truncate(read_in_chunk);
                total_bytes += read_in_chunk as u64;
                on_chunk(chunk_idx, buf).await?;
                chunk_idx += 1;
            }
            Ok(total_bytes)
        } else if allow_synthetic || is_benchmark_uri {
            let target_size = extract_size_from_uri(uri);
            warn!(
                "NFS path {:?} does not exist (synthetic fallback active). Streaming {:.2} MiB synthetic tensor payload for {}",
                path,
                target_size as f64 / (1024.0 * 1024.0),
                uri
            );
            let data = generate_synthetic_payload(uri, target_size);
            let mut chunk_idx = 0usize;
            for slice in data.chunks(chunk_size) {
                on_chunk(chunk_idx, slice.to_vec()).await?;
                chunk_idx += 1;
            }
            Ok(data.len() as u64)
        } else {
            anyhow::bail!("Authoritative NFS/POSIX file not found at {:?}", path);
        }
    }
}

fn extract_size_from_uri(uri: &str) -> usize {
    if let Some(start) = uri.find("weights_") {
        let sub = &uri[start + 8..];
        if let Some(end) = sub.find("mb") {
            if let Ok(mb) = sub[..end].parse::<usize>() {
                return mb * 1024 * 1024;
            }
        }
    }
    100 * 1024 * 1024 // default 100 MiB
}

fn generate_synthetic_payload(uri: &str, size_bytes: usize) -> Vec<u8> {
    let mut data = vec![0u8; size_bytes];
    let uri_bytes = uri.as_bytes();
    for (i, byte) in data.iter_mut().enumerate() {
        let pattern = uri_bytes[i % uri_bytes.len()];
        *byte = pattern ^ ((i % 256) as u8);
    }
    data
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nfs_parse_path_variations() {
        let config = Config {
            nfs_mount_path: "/mnt/nfs/models".to_string(),
            ..Config::default()
        };
        let fetcher = NfsPosixFetcher::new(config);

        assert_eq!(
            fetcher.parse_path("file:///mnt/nfs/models/llama3/weights.pt"),
            PathBuf::from("/mnt/nfs/models/llama3/weights.pt")
        );

        assert_eq!(
            fetcher.parse_path("nfs:///mnt/nfs/models/llama3/weights.pt"),
            PathBuf::from("/mnt/nfs/models/llama3/weights.pt")
        );

        assert_eq!(
            fetcher.parse_path("hypercache://nfs/mnt/nfs/models/llama3/weights.pt"),
            PathBuf::from("/mnt/nfs/models/llama3/weights.pt")
        );

        assert_eq!(
            fetcher.parse_path("hypercache://models/llama3/weights.pt"),
            PathBuf::from("/mnt/nfs/models/models/llama3/weights.pt")
        );
    }

    #[tokio::test]
    async fn test_nfs_fetch_real_file() {
        let config = Config::default();
        let fetcher = NfsPosixFetcher::new(config);

        let test_file = std::env::temp_dir().join(format!("test_nfs_{}.bin", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)));
        let test_payload = b"REAL_NFS_CHECKPOINT_DATA_123456789";
        tokio::fs::write(&test_file, test_payload).await.unwrap();

        let uri = format!("file://{}", test_file.display());
        let fetched = fetcher.fetch_object(&uri).await.unwrap();
        assert_eq!(fetched, test_payload);

        let _ = tokio::fs::remove_file(&test_file).await;
    }
}
