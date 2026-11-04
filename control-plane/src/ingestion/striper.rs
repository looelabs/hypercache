use anyhow::{Context, Result};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::sync::{Arc, Mutex};
use tracing::info;

use crate::config::Config;
use crate::metadata::MetadataStore;
use crate::protocol::{ChunkDescriptor, StorageTier};
use crate::ram::RamMemoryPool;

/// The drive an object's first chunk goes on. Each object starts on its own drive, so objects that
/// don't fill a whole stripe still spread evenly, and readers don't all start on the same drive.
/// Stable across restarts, so an append after a restart continues the object's pattern.
fn start_device(uri: &str, num_devices: usize) -> usize {
    // Multiply and shift so the choice comes from the hash's high bits
    ((crc32fast::hash(uri.as_bytes()) as u64 * num_devices as u64) >> 32) as usize
}

#[derive(Clone)]
pub struct Striper {
    config: Config,
    metadata: MetadataStore,
    ram_pool: Option<Arc<RamMemoryPool>>,
    // Write handles per device, reused across chunks instead of reopening per write
    device_files: Arc<Mutex<HashMap<String, Arc<File>>>>,
}

impl Striper {
    pub fn new(config: Config, metadata: MetadataStore, ram_pool: Option<Arc<RamMemoryPool>>) -> Self {
        Self {
            config,
            metadata,
            ram_pool,
            device_files: Arc::new(Mutex::new(HashMap::new())),
        }
    }

    pub fn metadata(&self) -> &MetadataStore {
        &self.metadata
    }

    /// Resolves the effective tier: an explicit request wins, otherwise RAM is used when
    /// enabled and the object fits, else NVMe.
    fn effective_tier(&self, target_tier: Option<StorageTier>, size_bytes: u64) -> StorageTier {
        match target_tier {
            Some(tier) => tier,
            None => match self.ram_pool {
                Some(ref pool)
                    if self.config.ram_cache_enabled
                        && pool.get_used_bytes() + size_bytes <= pool.get_capacity_bytes() =>
                {
                    StorageTier::Ram
                }
                _ => StorageTier::Nvme,
            },
        }
    }

    fn device_for_chunk(&self, uri: &str, chunk_idx: usize) -> Result<String> {
        let num_devices = self.config.block_devices.len();
        if num_devices == 0 {
            anyhow::bail!("No secondary raw NVMe block devices configured or discovered (OS root disk protected). Pass --tier ram to cache directly in Tier 1 Host RAM (/dev/shm) or configure secondary block devices in configs/hypercache-agent.toml.");
        }
        let start = start_device(uri, num_devices);
        Ok(self.config.block_devices[(start + chunk_idx) % num_devices].clone())
    }

    fn open_device(&self, device: &str) -> Result<Arc<File>> {
        let mut files = self.device_files.lock().unwrap();
        if let Some(f) = files.get(device) {
            return Ok(Arc::clone(f));
        }
        let f = Arc::new(
            OpenOptions::new()
                .write(true)
                .create(false)
                .open(device)
                .with_context(|| format!("Failed to open device {}", device))?,
        );
        files.insert(device.to_string(), Arc::clone(&f));
        Ok(f)
    }

    async fn write_at(&self, device: &str, offset: u64, data: Vec<u8>) -> Result<()> {
        use std::os::unix::fs::FileExt;
        let file = self.open_device(device)?;
        let dev = device.to_string();
        tokio::task::spawn_blocking(move || -> Result<()> {
            file.write_all_at(&data, offset)
                .with_context(|| format!("Failed to write chunk at offset {} on {}", offset, dev))
        })
        .await
        .context("Write chunk task panicked")?
    }

    /// Flushes device write caches for every NVMe device referenced by `chunks`. Must be
    /// called before a layout is published so a crash can never leave the manifest pointing
    /// at blocks whose data never reached stable storage.
    pub async fn sync_chunks(&self, chunks: &[ChunkDescriptor]) -> Result<()> {
        let mut devices: Vec<String> = chunks
            .iter()
            .filter(|c| c.tier != StorageTier::Ram)
            .map(|c| c.device.clone())
            .collect();
        devices.sort();
        devices.dedup();
        for dev in devices {
            let file = self.open_device(&dev)?;
            tokio::task::spawn_blocking(move || file.sync_data())
                .await
                .context("Device sync task panicked")?
                .with_context(|| format!("Failed to sync device {}", dev))?;
        }
        Ok(())
    }

    /// Stripes payload across either RAM (POSIX Shared Memory) or NVMe block devices.
    /// On any failure every chunk allocated by this call is released again.
    pub async fn stripe_bytes_to_tier(
        &self,
        uri: &str,
        data: &[u8],
        target_tier: Option<StorageTier>,
    ) -> Result<Vec<ChunkDescriptor>> {
        let chunk_size = self.config.chunk_size_bytes;
        let total_len = data.len();
        let num_chunks = total_len.div_ceil(chunk_size);

        info!(
            "Striping {} bytes into {} chunks (chunk_size: {} bytes)",
            total_len, num_chunks, chunk_size
        );

        let mut layout = Vec::with_capacity(num_chunks);
        for (chunk_idx, slice) in data.chunks(chunk_size).enumerate() {
            match self.stripe_single_chunk(uri, chunk_idx, slice, target_tier).await {
                Ok(desc) => layout.push(desc),
                Err(e) => {
                    self.metadata.release_unpublished(&layout).await;
                    return Err(e);
                }
            }
        }

        if let Err(e) = self.sync_chunks(&layout).await {
            self.metadata.release_unpublished(&layout).await;
            return Err(e);
        }
        Ok(layout)
    }

    /// Allocates chunk descriptors for incoming write without writing payload yet.
    pub async fn allocate_layout(
        &self,
        uri: &str,
        size_bytes: u64,
        target_tier: Option<StorageTier>,
    ) -> Result<Vec<ChunkDescriptor>> {
        self.allocate_layout_from_index(uri, 0, size_bytes, target_tier).await
    }

    /// Pre-allocates chunk descriptors starting at `start_chunk_idx` (used for streaming AppendWrite).
    pub async fn allocate_layout_from_index(
        &self,
        uri: &str,
        start_chunk_idx: usize,
        size_bytes: u64,
        target_tier: Option<StorageTier>,
    ) -> Result<Vec<ChunkDescriptor>> {
        let chunk_size = self.config.chunk_size_bytes as u64;
        let num_chunks = size_bytes.div_ceil(chunk_size) as usize;

        if self.effective_tier(target_tier, size_bytes) == StorageTier::Ram {
            let Some(ref pool) = self.ram_pool else {
                anyhow::bail!("RAM cache tier requested but RAM caching is disabled.");
            };
            let layout = (0..num_chunks)
                .map(|local_idx| {
                    let start = local_idx as u64 * chunk_size;
                    let chunk_len = (start + chunk_size).min(size_bytes) - start;
                    let chunk_id = format!("{}_chunk_{}", uri, start_chunk_idx + local_idx);
                    let path = pool.get_chunk_path(&chunk_id);
                    pool.create_staged_chunk(&path)?;
                    Ok(ChunkDescriptor::new(
                        StorageTier::Ram,
                        path.to_string_lossy().to_string(),
                        0,
                        chunk_len,
                    ))
                })
                .collect::<Result<Vec<_>>>()?;
            return Ok(layout);
        }

        let mut layout = Vec::with_capacity(num_chunks);
        for local_idx in 0..num_chunks {
            let start = local_idx as u64 * chunk_size;
            let chunk_len = (start + chunk_size).min(size_bytes) - start;
            let allocated = async {
                let device_path = self.device_for_chunk(uri, start_chunk_idx + local_idx)?;
                let offset = self
                    .metadata
                    .allocate_chunk_offset_for(
                        Some(uri),
                        &device_path,
                        chunk_len,
                        self.config.device_start_offset_bytes,
                    )
                    .await?;
                Ok::<_, anyhow::Error>(ChunkDescriptor::new(StorageTier::Nvme, device_path, offset, chunk_len))
            }
            .await;

            match allocated {
                Ok(desc) => layout.push(desc),
                Err(e) => {
                    self.metadata.release_unpublished(&layout).await;
                    return Err(e);
                }
            }
        }

        Ok(layout)
    }

    /// Writes a single chunk at `chunk_idx` directly to the target tier (or spills over from RAM to NVMe if full).
    /// The returned descriptor carries the chunk's CRC32. NVMe data is not yet fsynced; callers
    /// publish a layout only after `sync_chunks`.
    pub async fn stripe_single_chunk(
        &self,
        uri: &str,
        chunk_idx: usize,
        chunk_data: &[u8],
        target_tier: Option<StorageTier>,
    ) -> Result<ChunkDescriptor> {
        let chunk_len = chunk_data.len() as u64;

        if self.effective_tier(target_tier, chunk_len) == StorageTier::Ram {
            let Some(ref pool) = self.ram_pool else {
                anyhow::bail!("RAM cache tier requested but RAM caching is disabled.");
            };
            let chunk_id = format!("{}_chunk_{}", uri, chunk_idx);
            match pool.allocate_and_write_chunk(&chunk_id, chunk_data) {
                Ok(desc) => return Ok(desc),
                Err(e) if target_tier == Some(StorageTier::Ram) => {
                    return Err(e.context(format!(
                        "RAM cache capacity exceeded while writing chunk {} for {}",
                        chunk_idx, uri
                    )));
                }
                // Auto tier: fall through to NVMe spillover
                Err(_) => {}
            }
        }

        let device_path = self.device_for_chunk(uri, chunk_idx)?;
        let offset = self
            .metadata
            .allocate_chunk_offset_for(
                Some(uri),
                &device_path,
                chunk_len,
                self.config.device_start_offset_bytes,
            )
            .await?;
        let desc =
            ChunkDescriptor::new(StorageTier::Nvme, device_path.clone(), offset, chunk_len).with_checksum(chunk_data);

        if let Err(e) = self.write_at(&device_path, offset, chunk_data.to_vec()).await {
            self.metadata.release_unpublished(std::slice::from_ref(&desc)).await;
            return Err(e);
        }

        Ok(desc)
    }

    #[allow(dead_code)]
    pub async fn stripe_bytes(&self, data: &[u8]) -> Result<Vec<ChunkDescriptor>> {
        self.stripe_bytes_to_tier("unspecified", data, None).await
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_dir(tag: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!(
            "hypercache_striper_{}_{}",
            tag,
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    /// Bytes per drive for many objects of one size, as a fraction of an even share.
    fn spread(num_devices: usize, object_chunks: usize, last_chunk_fraction: f64) -> Vec<f64> {
        let objects = 8192;
        let mut bytes = vec![0.0; num_devices];
        for i in 0..objects {
            let uri = format!("hypercache://bench/m@00000000@0@{:08x}", i);
            let start = start_device(&uri, num_devices);
            for chunk in 0..object_chunks {
                let size = if chunk + 1 == object_chunks {
                    last_chunk_fraction
                } else {
                    1.0
                };
                bytes[(start + chunk) % num_devices] += size;
            }
        }
        let even = bytes.iter().sum::<f64>() / num_devices as f64;
        bytes.iter().map(|b| b / even).collect()
    }

    #[test]
    fn test_objects_spread_evenly_across_drives() {
        for num_devices in [2, 3, 4, 8] {
            // Under one chunk, 14 MiB in 4 MiB chunks, and exactly one stripe's worth
            for (chunks, last) in [(1, 0.5), (4, 0.5), (num_devices, 1.0)] {
                for share in spread(num_devices, chunks, last) {
                    assert!(
                        (0.9..=1.1).contains(&share),
                        "{} drives, {} chunks: share {}",
                        num_devices,
                        chunks,
                        share
                    );
                }
            }
        }
    }

    #[test]
    fn test_start_device_is_stable_and_in_range() {
        assert_eq!(start_device("hypercache://a", 1), 0);
        for num_devices in 1..=16 {
            for i in 0..1000 {
                let uri = format!("hypercache://{}", i);
                assert!(start_device(&uri, num_devices) < num_devices);
                assert_eq!(start_device(&uri, num_devices), start_device(&uri, num_devices));
            }
        }
    }

    #[tokio::test]
    async fn test_layout_keeps_stripe_order_from_the_objects_start_drive() {
        let config = Config {
            block_devices: (0..4).map(|i| format!("/tmp/hc-layout-{}.raw", i)).collect(),
            chunk_size_bytes: 1024 * 1024,
            ..Config::default()
        };
        let devices = config.block_devices.clone();
        let striper = Striper::new(config, MetadataStore::new(1 << 30, None, None), None);
        for uri in ["hypercache://a", "hypercache://b", "hypercache://c", "hypercache://d"] {
            let layout = striper
                .allocate_layout(uri, 6 * 1024 * 1024, Some(StorageTier::Nvme))
                .await
                .unwrap();
            let start = start_device(uri, 4);
            for (i, chunk) in layout.iter().enumerate() {
                assert_eq!(chunk.device, devices[(start + i) % 4]);
            }
            // Appending continues the same pattern
            let more = striper
                .allocate_layout_from_index(uri, 6, 1024 * 1024, Some(StorageTier::Nvme))
                .await
                .unwrap();
            assert_eq!(more[0].device, devices[(start + 6) % 4]);
        }
    }

    #[tokio::test]
    async fn test_failed_stripe_releases_allocated_extents() {
        let dir = temp_dir("rollback");
        let good = dir.join("good.raw");
        std::fs::write(&good, vec![0u8; 16 * 1024 * 1024]).unwrap();
        // Second device does not exist: writes to it fail
        let missing = dir.join("missing.raw");

        let config = Config {
            block_devices: vec![
                good.to_string_lossy().to_string(),
                missing.to_string_lossy().to_string(),
            ],
            chunk_size_bytes: 1024 * 1024,
            ..Config::default()
        };
        let metadata = MetadataStore::new(1 << 30, None, None);
        let striper = Striper::new(config, metadata.clone(), None);

        let data = vec![7u8; 3 * 1024 * 1024];
        assert!(striper
            .stripe_bytes_to_tier("hypercache://x", &data, Some(StorageTier::Nvme))
            .await
            .is_err());

        // Every extent allocated on the good device was returned: the next allocation starts at the beginning
        let off = metadata
            .allocate_chunk_offset(&good.to_string_lossy(), 1024 * 1024, 4096)
            .await
            .unwrap();
        assert_eq!(off, 4096);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_stripe_writes_checksummed_chunks() {
        let dir = temp_dir("crc");
        let dev = dir.join("dev.raw");
        std::fs::write(&dev, vec![0u8; 8 * 1024 * 1024]).unwrap();
        let config = Config {
            block_devices: vec![dev.to_string_lossy().to_string()],
            chunk_size_bytes: 1024 * 1024,
            ..Config::default()
        };
        let striper = Striper::new(config, MetadataStore::new(1 << 30, None, None), None);

        let data: Vec<u8> = (0..(2 * 1024 * 1024 + 17)).map(|i| (i % 251) as u8).collect();
        let layout = striper
            .stripe_bytes_to_tier("hypercache://y", &data, None)
            .await
            .unwrap();
        assert_eq!(layout.len(), 3);

        let raw = std::fs::read(&dev).unwrap();
        for (i, chunk) in layout.iter().enumerate() {
            let expected = &data[i * 1024 * 1024..((i + 1) * 1024 * 1024).min(data.len())];
            let on_disk = &raw[chunk.offset as usize..(chunk.offset + chunk.length) as usize];
            assert_eq!(on_disk, expected);
            assert_eq!(chunk.checksum, Some(crc32fast::hash(expected)));
        }

        let _ = std::fs::remove_dir_all(&dir);
    }
}
