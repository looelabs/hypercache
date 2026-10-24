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
        };

        pool.recompute_used_bytes();
        pool
    }

    /// Recomputes used bytes by scanning existing .shm files in the shared memory directory.
    pub fn recompute_used_bytes(&self) -> u64 {
        let mut total: u64 = 0;
        if let Ok(entries) = fs::read_dir(&self.shm_path) {
            for entry in entries.flatten() {
                let p = entry.path();
                if p.is_file() && p.extension().map_or(false, |ext| ext == "shm") {
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
    pub fn allocate_and_write_chunk(&self, chunk_id: &str, data: &[u8]) -> Result<ChunkDescriptor> {
        let chunk_len = data.len() as u64;
        let file_name = format!("{}.shm", chunk_id.replace('/', "_").replace(':', "_"));
        let target_path = self.shm_path.join(&file_name);

        let prev_len = if target_path.exists() {
            fs::metadata(&target_path).map(|m| m.len()).unwrap_or(0)
        } else {
            0
        };

        let current_used = self.used_bytes.load(Ordering::Relaxed);
        let net_new_bytes = if chunk_len > prev_len {
            chunk_len - prev_len
        } else {
            0
        };

        if current_used + net_new_bytes > self.max_capacity_bytes {
            anyhow::bail!(
                "RAM cache capacity exceeded (used: {} B, requested: {} B, max: {} B)",
                current_used,
                chunk_len,
                self.max_capacity_bytes
            );
        }

        let mut file = OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&target_path)
            .with_context(|| format!("Failed to create shared memory file at {:?}", target_path))?;

        file.write_all(data)
            .with_context(|| format!("Failed to write data to shared memory file at {:?}", target_path))?;
        file.sync_all()?;
        drop(file);

        if prev_len > 0 {
            self.used_bytes.fetch_sub(prev_len.min(self.used_bytes.load(Ordering::Relaxed)), Ordering::Relaxed);
        }
        self.used_bytes.fetch_add(chunk_len, Ordering::Relaxed);

        debug!(
            "Allocated RAM shared memory chunk: {:?} ({} bytes)",
            target_path, chunk_len
        );

        Ok(ChunkDescriptor {
            tier: StorageTier::Ram,
            device: target_path.to_string_lossy().to_string(),
            offset: 0,
            length: chunk_len,
        })
    }

    /// Deletes a shared memory chunk file and decrements the used bytes counter.
    #[allow(dead_code)]
    pub fn delete_chunk(&self, chunk_path: &str) -> Result<()> {
        let path = Path::new(chunk_path);
        if path.exists() {
            if let Ok(metadata) = fs::metadata(path) {
                let size = metadata.len();
                let _ = fs::remove_file(path);
                self.used_bytes.fetch_sub(size.min(self.used_bytes.load(Ordering::Relaxed)), Ordering::Relaxed);
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
                if p.is_file() && p.extension().map_or(false, |ext| ext == "shm") {
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

    pub fn get_chunk_path(&self, chunk_id: &str) -> PathBuf {
        let file_name = format!("{}.shm", chunk_id.replace('/', "_").replace(':', "_"));
        self.shm_path.join(&file_name)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ram_pool_allocation_and_unlink() {
        let temp_shm = std::env::temp_dir().join(format!("test_hypercache_shm_{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)));
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
}
