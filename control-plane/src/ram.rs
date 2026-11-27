use anyhow::{Context, Result};
use std::collections::HashMap;
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
    // Held for the daemon's lifetime once `lock_dir` succeeds
    dir_lock: Arc<std::sync::Mutex<Option<fs::File>>>,
}

/// Whether `name` has the shape `get_chunk_path` gives chunk files: `<id>.<pid>.<nanos>.<seq>.shm`.
fn is_pool_chunk_name(name: &str) -> bool {
    let Some(stem) = name.strip_suffix(".shm") else {
        return false;
    };
    let mut parts = stem.rsplitn(4, '.');
    let numeric = |p: Option<&str>| p.is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit()));
    // seq, nanos, pid, then a non-empty id prefix
    numeric(parts.next())
        && numeric(parts.next())
        && numeric(parts.next())
        && parts.next().is_some_and(|p| !p.is_empty())
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
            dir_lock: Arc::new(std::sync::Mutex::new(None)),
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

    /// Reserves `len` bytes of pool capacity, or fails without reserving anything when the
    /// pool would go over its limit. Atomic, so concurrent writers cannot overshoot.
    fn reserve(&self, len: u64) -> Result<()> {
        let mut current_used = self.used_bytes.load(Ordering::Relaxed);
        loop {
            if current_used + len > self.max_capacity_bytes {
                anyhow::bail!(
                    "RAM cache capacity exceeded (used: {} B, requested: {} B, max: {} B)",
                    current_used,
                    len,
                    self.max_capacity_bytes
                );
            }
            match self.used_bytes.compare_exchange_weak(
                current_used,
                current_used + len,
                Ordering::AcqRel,
                Ordering::Relaxed,
            ) {
                Ok(_) => return Ok(()),
                Err(actual) => current_used = actual,
            }
        }
    }

    /// Returns `len` reserved bytes to the pool (never below zero).
    fn unreserve(&self, len: u64) {
        let mut used = self.used_bytes.load(Ordering::Relaxed);
        while let Err(actual) =
            self.used_bytes
                .compare_exchange_weak(used, used.saturating_sub(len), Ordering::AcqRel, Ordering::Relaxed)
        {
            used = actual;
        }
    }

    /// Reserves `len` bytes and creates the empty file for a chunk a client is about to write
    /// (staged writes). The reservation counts the chunk's full length, not the file's current
    /// size, so a checkpoint being written can never take the pool over its limit.
    /// Creating the file here, rather than in the client, means a client that does not share
    /// this `/dev/shm` with the daemon fails loudly instead of writing into a private copy.
    pub fn allocate_staged_chunk(&self, chunk_id: &str, len: u64) -> Result<ChunkDescriptor> {
        self.reserve(len)?;
        let path = self.get_chunk_path(chunk_id);
        let created = OpenOptions::new()
            .write(true)
            .create_new(true)
            .open(&path)
            .with_context(|| format!("Failed to create staged shared memory file at {:?}", path));
        if let Err(e) = created {
            self.unreserve(len);
            return Err(e);
        }
        crate::access::grant_file_or_warn(&path, self.client_gid);
        Ok(ChunkDescriptor::new(
            StorageTier::Ram,
            path.to_string_lossy().to_string(),
            0,
            len,
        ))
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

        self.reserve(chunk_len)?;

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
            self.unreserve(chunk_len);
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

    /// Deletes a chunk file and returns its `length` (the chunk descriptor's length, which is
    /// what was reserved) to the pool. Nothing is returned when the file was already gone:
    /// `reconcile` only counts chunks whose file exists.
    pub fn delete_chunk(&self, chunk_path: &str, length: u64) -> Result<()> {
        match fs::remove_file(chunk_path) {
            Ok(()) => {
                self.unreserve(length);
                debug!(
                    "Removed RAM shared memory chunk: {} (freed {} bytes)",
                    chunk_path, length
                );
            }
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {}
            Err(e) => warn!("Failed to remove RAM shared memory chunk {}: {}", chunk_path, e),
        }
        Ok(())
    }

    /// Takes an exclusive lock on the pool directory for this process's lifetime. Returns false
    /// when another process (a second daemon pointed at the same directory) holds it.
    pub fn lock_dir(&self) -> bool {
        use fs2::FileExt;
        let mut held = self.dir_lock.lock().unwrap();
        if held.is_some() {
            return true;
        }
        let file = match OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(self.shm_path.join(".hypercache-pool.lock"))
        {
            Ok(f) => f,
            Err(e) => {
                warn!("Could not open the RAM pool lock file in {:?}: {}", self.shm_path, e);
                return false;
            }
        };
        if file.try_lock_exclusive().is_err() {
            return false;
        }
        *held = Some(file);
        true
    }

    /// Startup only, before anything is served: deletes every chunk file in the pool directory
    /// that `keep` (path to chunk length, from the restored index) does not list, such as chunks
    /// of an ingest a crash interrupted, then sets used bytes to the lengths of the kept chunks
    /// whose file exists. Returns the number of files and bytes deleted.
    pub fn reconcile(&self, keep: &HashMap<PathBuf, u64>) -> (usize, u64) {
        let mut removed = 0usize;
        let mut removed_bytes = 0u64;
        if let Ok(entries) = fs::read_dir(&self.shm_path) {
            for entry in entries.flatten() {
                let p = entry.path();
                let is_chunk = p.file_name().and_then(|n| n.to_str()).is_some_and(is_pool_chunk_name);
                if !is_chunk || keep.contains_key(&p) {
                    continue;
                }
                let Ok(meta) = fs::symlink_metadata(&p) else { continue };
                if !meta.is_file() {
                    continue;
                }
                match fs::remove_file(&p) {
                    Ok(()) => {
                        removed += 1;
                        removed_bytes += meta.len();
                    }
                    Err(e) => warn!("Failed to remove unreferenced RAM chunk {:?}: {}", p, e),
                }
            }
        }
        let used: u64 = keep.iter().filter(|(p, _)| p.is_file()).map(|(_, &len)| len).sum();
        self.used_bytes.store(used, Ordering::Relaxed);
        (removed, removed_bytes)
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
        pool.delete_chunk(&desc.device, desc.length).unwrap();
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
        pool.delete_chunk(&first.device, first.length).unwrap();
        assert_eq!(fs::read(&second.device).unwrap(), vec![2u8; 1024]);
        assert_eq!(second.checksum, Some(crc32fast::hash(&[2u8; 1024])));

        // Capacity is enforced across allocations
        pool.allocate_and_write_chunk("other", &[3u8; 2048]).unwrap();
        assert!(pool.allocate_and_write_chunk("overflow", &[4u8; 1]).is_err());
        assert_eq!(pool.get_used_bytes(), 3 * 1024);

        let _ = fs::remove_dir_all(&temp_shm);
    }

    fn temp_pool_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "test_hypercache_shm_{}_{}",
            tag,
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ))
    }

    #[test]
    fn test_staged_chunks_reserve_their_full_length() {
        let dir = temp_pool_dir("staged");
        let pool = RamMemoryPool::new(dir.clone(), 10 * 1024);

        let staged = pool.allocate_staged_chunk("ckpt_chunk_0", 8 * 1024).unwrap();
        assert_eq!(staged.tier, StorageTier::Ram);
        assert_eq!(staged.length, 8 * 1024);
        // The file is still empty, but its full length counts against the cap
        assert_eq!(fs::metadata(&staged.device).unwrap().len(), 0);
        assert_eq!(pool.get_used_bytes(), 8 * 1024);

        let files_before = fs::read_dir(&dir).unwrap().count();
        assert!(pool.allocate_staged_chunk("ckpt_chunk_1", 4 * 1024).is_err());
        assert!(pool.allocate_and_write_chunk("other", &[1u8; 4 * 1024]).is_err());
        assert_eq!(
            fs::read_dir(&dir).unwrap().count(),
            files_before,
            "a refused chunk leaves no file"
        );
        assert_eq!(pool.get_used_bytes(), 8 * 1024);

        // Freeing a chunk returns what it reserved, not the (short) file's size
        fs::write(&staged.device, [9u8; 100]).unwrap();
        pool.delete_chunk(&staged.device, staged.length).unwrap();
        assert_eq!(pool.get_used_bytes(), 0);
        // A second delete of the same chunk returns nothing
        pool.delete_chunk(&staged.device, staged.length).unwrap();
        assert_eq!(pool.get_used_bytes(), 0);
        pool.allocate_staged_chunk("ckpt_chunk_0", 10 * 1024).unwrap();

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_reconcile_removes_unreferenced_chunks_and_counts_kept_ones() {
        let dir = temp_pool_dir("reconcile");
        let pool = RamMemoryPool::new(dir.clone(), 1024 * 1024);
        let kept = pool.allocate_and_write_chunk("model_chunk_0", &[1u8; 3000]).unwrap();
        let orphan = pool.allocate_and_write_chunk("model_chunk_1", &[2u8; 5000]).unwrap();
        let staged = pool.allocate_staged_chunk("ckpt_chunk_0", 4096).unwrap();
        // Files that are not pool chunks are never touched
        let foreign = [dir.join("notes.shm"), dir.join("readme.txt"), dir.join("a.1.2.shm")];
        for f in &foreign {
            fs::write(f, b"x").unwrap();
        }

        // A fresh pool over the same directory (a restart) counts every .shm file
        drop(pool);
        let pool = RamMemoryPool::new(dir.clone(), 1024 * 1024);
        assert!(pool.get_used_bytes() >= 3000 + 5000);

        let gone = dir.join("model_chunk_9.1.2.3.shm");
        let keep: HashMap<PathBuf, u64> = [
            (PathBuf::from(&kept.device), kept.length),
            (PathBuf::from(&staged.device), staged.length),
            (gone, 4096),
        ]
        .into_iter()
        .collect();
        assert_eq!(pool.reconcile(&keep), (1, 5000));
        assert!(!Path::new(&orphan.device).exists());
        assert!(Path::new(&kept.device).exists() && Path::new(&staged.device).exists());
        for f in &foreign {
            assert!(f.exists(), "{:?} must be left alone", f);
        }
        // Kept chunks count at their reserved length; a referenced file that is gone counts nothing
        assert_eq!(pool.get_used_bytes(), 3000 + 4096);

        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_pool_dir_lock_is_exclusive() {
        let dir = temp_pool_dir("lock");
        let first = RamMemoryPool::new(dir.clone(), 1024);
        let second = RamMemoryPool::new(dir.clone(), 1024);
        assert!(first.lock_dir());
        assert!(first.lock_dir(), "taking it again in the same pool is fine");
        assert!(!second.lock_dir());
        drop(first);
        assert!(second.lock_dir());
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_pool_chunk_names() {
        let dir = temp_pool_dir("names");
        let pool = RamMemoryPool::new(dir.clone(), 1024);
        let path = pool.get_chunk_path("hypercache://s3/b/model.bin_chunk_3");
        assert!(is_pool_chunk_name(path.file_name().unwrap().to_str().unwrap()));
        for name in [
            "notes.shm",
            "a.1.2.shm",
            ".1.2.3.shm",
            "a.1.2.x.shm",
            "a.1.2.3.tmp",
            "a.1..3.shm",
        ] {
            assert!(!is_pool_chunk_name(name), "{}", name);
        }
        let _ = fs::remove_dir_all(&dir);
    }
}
