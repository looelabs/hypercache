use anyhow::{Context, Result};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tracing::{debug, info, warn};

use crate::protocol::{ChunkDescriptor, StorageTier};

#[derive(Clone, Debug)]
pub struct RamMemoryPool {
    shm_path: PathBuf,
    max_capacity_bytes: u64,
    used_bytes: Arc<AtomicU64>,
    client_gid: Option<u32>,
}

impl RamMemoryPool {
    /// Initializes the Shared Memory RAM pool.
    /// Ensures the shared memory directory exists and scans current utilization.
    pub fn new(shm_path: PathBuf, max_capacity_bytes: u64) -> Self {
        if let Err(e) = fs::create_dir_all(&shm_path) {
            warn!("Failed to create shared memory directory at {:?}: {}", shm_path, e);
        }

        let pool = Self {
            shm_path,
            max_capacity_bytes,
            used_bytes: Arc::new(AtomicU64::new(0)),
            client_gid: None,
        };

        pool.recompute_used_bytes();
        pool
    }

    /// Shares the pool directory and every chunk file with `client_gid` (see `access`).
    pub fn with_client_gid(mut self, client_gid: Option<u32>) -> Self {
        self.client_gid = client_gid;
        if let Err(e) = crate::access::grant_dir(&self.shm_path, client_gid) {
            warn!(
                "Could not give group {:?} access to shared memory directory {:?}: {}",
                client_gid, self.shm_path, e
            );
        }
        self
    }

    /// Creates the empty file for a chunk a client is about to write (staged writes).
    /// Creating it here, rather than in the client, means a client that does not share this
    /// `/dev/shm` with the daemon fails loudly instead of writing into a private copy.
    pub fn create_staged_chunk(&self, path: &Path) -> Result<()> {
        OpenOptions::new()
            .write(true)
            .create(true)
            .truncate(true)
            .open(path)
            .with_context(|| format!("Failed to create staged shared memory file at {:?}", path))?;
        crate::access::grant_file_or_warn(path, self.client_gid);
        Ok(())
    }

    /// Recomputes used bytes by scanning existing .shm files in the shared memory directory.
    pub fn recompute_used_bytes(&self) -> u64 {
        let mut total: u64 = 0;
        if let Ok(entries) = fs::read_dir(&self.shm_path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() && p.extension().is_some_and(|ext| ext == "shm") {
                    if let Ok(meta) = fs::metadata(&p) {
                        total += meta.len();
                    }
                }
            }
        }
        self.used_bytes.store(total, Ordering::Relaxed);
        total
    }

    /// Allocates and writes a 4 MiB (or variable size) chunk directly into POSIX shared memory.
    /// Every call creates a new uniquely named file, so re-ingesting an object never truncates
    /// a file that readers of the previous layout may still have mapped.
    pub fn allocate_and_write_chunk(&self, chunk_id: &str, data: &[u8]) -> Result<ChunkDescriptor> {
        let chunk_len = data.len() as u64;

        // Reserve capacity atomically so concurrent writers cannot overshoot the pool limit
        let mut current_used = self.used_bytes.load(Ordering::Relaxed);
        loop {
            if current_used + chunk_len > self.max_capacity_bytes {
                anyhow::bail!(
                    "RAM cache capacity exceeded (used: {} B, requested: {} B, max: {} B)",
                    current_used,
                    chunk_len,
                    self.max_capacity_bytes
                );
            }
            match self.used_bytes.compare_exchange_weak(
                current_used,
                current_used + chunk_len,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => current_used = actual,
            }
        }

        let target_path = self.get_chunk_path(chunk_id);
        let write_result = (|| -> Result<()> {
            let mut file = OpenOptions::new()
                .read(true)
                .write(true)
                .create_new(true)
                .open(&target_path)
                .with_context(|| format!("Failed to create shared memory file at {:?}", target_path))?;
            file.write_all(data)
                .with_context(|| format!("Failed to write data to shared memory file at {:?}", target_path))?;
            crate::access::grant_file(&target_path, self.client_gid)
                .with_context(|| format!("Failed to share {:?} with client group", target_path))?;
            Ok(())
        })();

        if let Err(e) = write_result {
            let _ = fs::remove_file(&target_path);
            self.used_bytes.fetch_sub(chunk_len, Ordering::AcqRel);
            return Err(e);
        }

        debug!(
            "Allocated RAM shared memory chunk: {:?} ({} bytes)",
            target_path, chunk_len
        );

        Ok(ChunkDescriptor::new(
            StorageTier::Ram,
            target_path.to_string_lossy().to_string(),
            0,
            chunk_len,
        )
        .with_checksum(data))
    }

    /// Deletes a shared memory chunk file and decrements the used bytes counter.
    #[allow(dead_code)]
    pub fn delete_chunk(&self, chunk_path: &str) -> Result<()> {
        let path = Path::new(chunk_path);
        if path.exists() {
            if let Ok(metadata) = fs::metadata(path) {
                let size = metadata.len();
                let _ = fs::remove_file(path);
                self.used_bytes
                    .fetch_sub(size.min(self.used_bytes.load(Ordering::Relaxed)), Ordering::Relaxed);
                debug!("Removed RAM shared memory chunk: {:?} (freed {} bytes)", path, size);
            } else {
                let _ = fs::remove_file(path);
            }
        }
        Ok(())
    }

    /// Cleans up all `.shm` files inside the configured shared memory directory.
    pub fn cleanup_orphaned_chunks(&self) -> Result<()> {
        if !self.shm_path.exists() {
            return Ok(());
        }

        let mut freed_count = 0;
        let mut freed_bytes: u64 = 0;

        if let Ok(entries) = fs::read_dir(&self.shm_path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() && p.extension().is_some_and(|ext| ext == "shm") {
                    if let Ok(meta) = fs::metadata(&p) {
                        freed_bytes += meta.len();
                    }
                    let _ = fs::remove_file(&p);
                    freed_count += 1;
                }
            }
        }

        self.used_bytes.store(0, Ordering::Relaxed);

        if freed_count > 0 {
            info!(
                "Cleaned up {} stale shared memory chunk(s) ({:.2} MiB) in {:?}",
                freed_count,
                freed_bytes as f64 / (1024.0 * 1024.0),
                self.shm_path
            );
        }

        Ok(())
    }

    pub fn get_used_bytes(&self) -> u64 {
        self.used_bytes.load(Ordering::Relaxed)
    }

    pub fn get_capacity_bytes(&self) -> u64 {
        self.max_capacity_bytes
    }

    pub fn get_shm_path(&self) -> &Path {
        &self.shm_path
    }

    /// Returns a new, unique chunk file path for `chunk_id` (the URI-derived prefix keeps
    /// files recognizable; the suffix guarantees a fresh inode per allocation).
    pub fn get_chunk_path(&self, chunk_id: &str) -> PathBuf {
        static CHUNK_SEQ: AtomicU64 = AtomicU64::new(0);
        let sanitized: String = chunk_id
            .chars()
            .map(|c| {
                if c.is_ascii_alphanumeric() || c == '.' || c == '-' || c == '_' {
                    c
                } else {
                    '_'
                }
            })
            .collect();
        // Keep file names well under NAME_MAX (255) for long URIs
        let prefix: String = if sanitized.len() > 160 {
            sanitized[sanitized.len() - 160..].to_string()
        } else {
            sanitized
        };
        let file_name = format!(
            "{}.{}.{}.{}.shm",
            prefix,
            std::process::id(),
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0),
            CHUNK_SEQ.fetch_add(1, Ordering::Relaxed)
        );
        self.shm_path.join(file_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ram_pool_allocation_and_unlink() {
        let temp_shm = std::env::temp_dir().join(format!(
            "test_hypercache_shm_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        let pool = RamMemoryPool::new(temp_shm.clone(), 10 * 1024 * 1024);

        let test_data = vec![0x42u8; 1024 * 1024]; // 1 MiB
        let desc = pool.allocate_and_write_chunk("test_chunk_0", &test_data).unwrap();

        assert_eq!(desc.tier, StorageTier::Ram);
        assert_eq!(desc.length, 1024 * 1024);
        assert_eq!(desc.offset, 0);
        assert!(Path::new(&desc.device).exists());
        assert_eq!(pool.get_used_bytes(), 1024 * 1024);

        // Verify read back
        let read_back = fs::read(&desc.device).unwrap();
        assert_eq!(read_back, test_data);

        // Delete chunk
        pool.delete_chunk(&desc.device).unwrap();
        assert!(!Path::new(&desc.device).exists());
        assert_eq!(pool.get_used_bytes(), 0);

        let _ = fs::remove_dir_all(&temp_shm);
    }

    #[test]
    fn test_reallocating_same_chunk_id_never_reuses_file() {
        let temp_shm = std::env::temp_dir().join(format!(
            "test_hypercache_shm_unique_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        let pool = RamMemoryPool::new(temp_shm.clone(), 3 * 1024);

        let first = pool
            .allocate_and_write_chunk("s3://bucket/model.bin_chunk_0", &[1u8; 1024])
            .unwrap();
        let second = pool
            .allocate_and_write_chunk("s3://bucket/model.bin_chunk_0", &[2u8; 1024])
            .unwrap();
        assert_ne!(first.device, second.device);

        // Releasing the old generation must not touch the new generation's data
        pool.delete_chunk(&first.device).unwrap();
        assert_eq!(fs::read(&second.device).unwrap(), vec![2u8; 1024]);
        assert_eq!(second.checksum, Some(crc32fast::hash(&[2u8; 1024])));

        // Capacity is enforced across allocations
        pool.allocate_and_write_chunk("other", &[3u8; 2048]).unwrap();
        assert!(pool.allocate_and_write_chunk("overflow", &[4u8; 1]).is_err());
        assert_eq!(pool.get_used_bytes(), 3 * 1024);

        let _ = fs::remove_dir_all(&temp_shm);
    }
}
