use anyhow::{Context, Result};
use std::path::{Component, Path, PathBuf};
use tracing::{info, warn};

use super::synthetic::{extract_size_from_uri, generate_synthetic_payload};
use crate::config::Config;

#[derive(Clone)]
pub struct NfsPosixFetcher {
    config: Config,
}

impl NfsPosixFetcher {
    pub fn new(config: Config) -> Self {
        Self { config }
    }

    /// Resolves any URI (file://, nfs://, hypercache://file/..., /mnt/nfs/...) to a filesystem PathBuf.
    /// This is purely lexical; use [`Self::resolve_path`] before touching the filesystem.
    pub fn parse_path(&self, uri: &str) -> PathBuf {
        if let Some(rest) = uri.strip_prefix("file://") {
            PathBuf::from(rest)
        } else if let Some(rest) = uri.strip_prefix("nfs://") {
            PathBuf::from(if rest.starts_with('/') {
                rest.to_string()
            } else {
                format!("/{}", rest)
            })
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

    /// Directories the daemon may read from / write to: `nfs_mount_path` plus `nfs_allowed_paths`.
    fn allowed_roots(&self) -> Vec<PathBuf> {
        std::iter::once(self.config.nfs_mount_path.as_str())
            .chain(self.config.nfs_allowed_paths.iter().map(|s| s.as_str()))
            .filter(|s| !s.trim().is_empty())
            .filter_map(|root| std::fs::canonicalize(root).ok())
            .collect()
    }

    /// Resolves a URI to a path and confines it to the allowed roots. Rejects `..` components
    /// and symlinks that escape the roots (the nearest existing ancestor is canonicalized),
    /// so a URI like `file:///etc/shadow` cannot make the daemon read or write arbitrary files.
    pub fn resolve_path(&self, uri: &str) -> Result<PathBuf> {
        let path = self.parse_path(uri);
        if !path.is_absolute() {
            anyhow::bail!("NFS/POSIX path {:?} for '{}' is not absolute", path, uri);
        }
        if path.components().any(|c| matches!(c, Component::ParentDir)) {
            anyhow::bail!("NFS/POSIX path {:?} for '{}' contains '..' and was rejected", path, uri);
        }

        // Canonicalize the deepest existing ancestor and re-append the remaining components
        let mut existing = path.as_path();
        let mut remainder: Vec<&std::ffi::OsStr> = Vec::new();
        let canonical_base = loop {
            match std::fs::canonicalize(existing) {
                Ok(c) => break c,
                Err(_) => match (existing.parent(), existing.file_name()) {
                    (Some(parent), Some(name)) => {
                        remainder.push(name);
                        existing = parent;
                    }
                    _ => anyhow::bail!("Cannot resolve NFS/POSIX path {:?}", path),
                },
            }
        };
        let mut resolved = canonical_base;
        for part in remainder.iter().rev() {
            resolved.push(part);
        }

        let roots = self.allowed_roots();
        if roots.iter().any(|root| resolved.starts_with(root)) {
            Ok(resolved)
        } else {
            let missing: Vec<&str> = std::iter::once(self.config.nfs_mount_path.as_str())
                .chain(self.config.nfs_allowed_paths.iter().map(|s| s.as_str()))
                .filter(|s| !s.trim().is_empty() && !Path::new(s).exists())
                .collect();
            let hint = if missing.is_empty() {
                String::new()
            } else {
                format!(
                    " Note: {:?} does not exist on the daemon's filesystem (in a container, mount the share at that path).",
                    missing
                )
            };
            anyhow::bail!(
                "NFS/POSIX path {:?} for '{}' is outside the allowed roots (nfs_mount_path = {:?}, nfs_allowed_paths = {:?}). \
                 Add its directory to nfs_allowed_paths in the agent configuration to allow it.{}",
                path,
                uri,
                self.config.nfs_mount_path,
                self.config.nfs_allowed_paths,
                hint
            )
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
        let path = self.resolve_path(uri)?;
        info!("Fetching authoritative object from NFS/POSIX filesystem: {:?}", path);

        if path.exists() {
            let bytes = tokio::fs::read(&path)
                .await
                .with_context(|| format!("Failed to read file from NFS path {:?}", path))?;
            info!("Successfully read {} bytes from NFS file {:?}", bytes.len(), path);
            Ok(bytes)
        } else if allow_synthetic {
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
        on_chunk: F,
    ) -> Result<u64>
    where
        F: Fn(usize, Vec<u8>) -> Fut,
        Fut: std::future::Future<Output = Result<()>>,
    {
        use tokio::io::AsyncReadExt;
        let path = self.resolve_path(uri)?;
        info!("Streaming authoritative object from NFS/POSIX filesystem: {:?}", path);

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
        } else if allow_synthetic {
            let target_size = extract_size_from_uri(uri);
            warn!(
                "NFS path {:?} does not exist (synthetic fallback active). Streaming {:.2} MiB synthetic tensor payload for {}",
                path,
                target_size as f64 / (1024.0 * 1024.0),
                uri
            );
            let data = generate_synthetic_payload(uri, target_size);
            for (chunk_idx, slice) in data.chunks(chunk_size).enumerate() {
                on_chunk(chunk_idx, slice.to_vec()).await?;
            }
            Ok(data.len() as u64)
        } else {
            anyhow::bail!("Authoritative NFS/POSIX file not found at {:?}", path);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_root(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!(
            "hypercache_nfs_{}_{}",
            tag,
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

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
        let root = temp_root("fetch");
        let config = Config {
            nfs_mount_path: root.to_string_lossy().to_string(),
            ..Config::default()
        };
        let fetcher = NfsPosixFetcher::new(config);

        let test_file = root.join("ckpt.bin");
        let test_payload = b"REAL_NFS_CHECKPOINT_DATA_123456789";
        tokio::fs::write(&test_file, test_payload).await.unwrap();

        let uri = format!("file://{}", test_file.display());
        let fetched = fetcher.fetch_object(&uri).await.unwrap();
        assert_eq!(fetched, test_payload);

        let _ = std::fs::remove_dir_all(&root);
    }

    #[tokio::test]
    async fn test_paths_outside_allowed_roots_are_rejected() {
        let root = temp_root("confine");
        let outside = temp_root("outside");
        std::fs::write(outside.join("secret.txt"), b"secret").unwrap();
        std::fs::write(root.join("ok.bin"), b"ok").unwrap();

        let fetcher = NfsPosixFetcher::new(Config {
            nfs_mount_path: root.to_string_lossy().to_string(),
            ..Config::default()
        });

        assert!(fetcher.fetch_object("file:///etc/passwd").await.is_err());
        assert!(fetcher
            .fetch_object(&format!("file://{}/secret.txt", outside.display()))
            .await
            .is_err());
        // Parent-directory traversal out of the root is rejected lexically
        assert!(fetcher
            .resolve_path(&format!(
                "file://{}/../{}/secret.txt",
                root.display(),
                outside.file_name().unwrap().to_string_lossy()
            ))
            .is_err());
        // A symlink inside the root pointing outside it is rejected after canonicalization
        std::os::unix::fs::symlink(outside.join("secret.txt"), root.join("link.txt")).unwrap();
        assert!(fetcher
            .fetch_object(&format!("file://{}/link.txt", root.display()))
            .await
            .is_err());
        // Relative hypercache:// URIs resolve under the mount and are allowed, even before they exist
        assert!(fetcher.resolve_path("hypercache://ok.bin").is_ok());
        assert!(fetcher.resolve_path("hypercache://new/dir/file.bin").is_ok());
        assert_eq!(fetcher.fetch_object("hypercache://ok.bin").await.unwrap(), b"ok");

        // Explicitly allowed extra roots work
        let widened = NfsPosixFetcher::new(Config {
            nfs_mount_path: root.to_string_lossy().to_string(),
            nfs_allowed_paths: vec![outside.to_string_lossy().to_string()],
            ..Config::default()
        });
        assert_eq!(
            widened
                .fetch_object(&format!("file://{}/secret.txt", outside.display()))
                .await
                .unwrap(),
            b"secret"
        );

        let _ = std::fs::remove_dir_all(&root);
        let _ = std::fs::remove_dir_all(&outside);
    }
}
