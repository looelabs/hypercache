use anyhow::{Context, Result};
use std::fs::OpenOptions;
use std::sync::Arc;
use tracing::info;

use crate::config::Config;
use crate::metadata::MetadataStore;
use crate::protocol::{ChunkDescriptor, StorageTier};
use crate::ram::RamMemoryPool;

#[derive(Clone)]
pub struct Striper {
    config: Config,
    metadata: MetadataStore,
    ram_pool: Option<Arc<RamMemoryPool>>,
}

impl Striper {
    pub fn new(config: Config, metadata: MetadataStore, ram_pool: Option<Arc<RamMemoryPool>>) -> Self {
        Self {
            config,
            metadata,
            ram_pool,
        }
    }

    /// Stripes payload across either RAM (POSIX Shared Memory) or NVMe block devices.
    pub async fn stripe_bytes_to_tier(
        &self,
        uri: &str,
        data: &[u8],
        target_tier: Option<StorageTier>,
    ) -> Result<Vec<ChunkDescriptor>> {
        let chunk_size = self.config.chunk_size_bytes;
        let total_len = data.len();
        let num_chunks = (total_len + chunk_size - 1) / chunk_size;

        let effective_tier = match target_tier {
            Some(tier) => tier,
            None => {
                if let Some(ref pool) = self.ram_pool {
                    if self.config.ram_cache_enabled
                        && pool.get_used_bytes() + (total_len as u64) <= pool.get_capacity_bytes()
                    {
                        StorageTier::Ram
                    } else {
                        StorageTier::Nvme
                    }
                } else {
                    StorageTier::Nvme
                }
            }
        };

        if effective_tier == StorageTier::Ram {
            if let Some(ref pool) = self.ram_pool {
                info!(
                    "Striping {} bytes into {} RAM shared memory chunks (chunk_size: {} bytes)",
                    total_len, num_chunks, chunk_size
                );

                let mut layout = Vec::with_capacity(num_chunks);
                for chunk_idx in 0..num_chunks {
                    let start = chunk_idx * chunk_size;
                    let end = std::cmp::min(start + chunk_size, total_len);
                    let chunk_id = format!("{}_chunk_{}", uri, chunk_idx);
                    let slice = &data[start..end];
                    match pool.allocate_and_write_chunk(&chunk_id, slice) {
                        Ok(desc) => layout.push(desc),
                        Err(e) => {
                            for written in layout {
                                let _ = pool.delete_chunk(&written.device);
                            }
                            return Err(e);
                        }
                    }
                }
                return Ok(layout);
            } else {
                anyhow::bail!("RAM cache tier requested but RAM caching is disabled. Set ram_cache_enabled = true in configs/hypercache-agent.toml or pass --ram to enable Tier 1 Host RAM caching.");
            }
        }

        // Otherwise, stripe to NVMe block devices
        let num_devices = self.config.block_devices.len();
        if num_devices == 0 {
            anyhow::bail!("No secondary raw NVMe block devices configured or discovered (OS root disk protected). Pass --tier ram to cache directly in Tier 1 Host RAM (/dev/shm) or configure secondary block devices in configs/hypercache-agent.toml.");
        }

        let mut layout = Vec::with_capacity(num_chunks);
        info!(
            "Striping {} bytes into {} NVMe chunks (chunk_size: {} bytes) across {} devices",
            total_len, num_chunks, chunk_size, num_devices
        );

        let mut chunk_tasks = Vec::with_capacity(num_chunks);
        for chunk_idx in 0..num_chunks {
            let start = chunk_idx * chunk_size;
            let end = std::cmp::min(start + chunk_size, total_len);
            let chunk_len = (end - start) as u64;

            let device_path = self.config.block_devices[chunk_idx % num_devices].clone();
            let offset = self
                .metadata
                .allocate_chunk_offset(
                    &device_path,
                    chunk_len,
                    self.config.device_start_offset_bytes,
                )
                .await?;

            layout.push(ChunkDescriptor {
                tier: StorageTier::Nvme,
                device: device_path.clone(),
                offset,
                length: chunk_len,
            });

            chunk_tasks.push((device_path, offset, start, end));
        }

        let data_bytes = data.to_vec();
        tokio::task::spawn_blocking(move || -> Result<()> {
            use std::collections::HashMap;
            use std::os::unix::fs::FileExt;

            let mut files = HashMap::new();
            for (dev_path, offset, start, end) in chunk_tasks {
                let file = if let Some(f) = files.get_mut(&dev_path) {
                    f
                } else {
                    let f = OpenOptions::new()
                        .write(true)
                        .create(false)
                        .open(&dev_path)
                        .with_context(|| format!("Failed to open device {}", dev_path))?;
                    files.entry(dev_path.clone()).or_insert(f)
                };

                let slice = &data_bytes[start..end];
                file.write_all_at(slice, offset)
                    .with_context(|| format!("Failed to write chunk at offset {} on {}", offset, dev_path))?;
            }

            for (dev_path, file) in files {
                file.sync_all()
                    .with_context(|| format!("Failed to sync device {}", dev_path))?;
            }

            Ok(())
        })
        .await
        .with_context(|| "Write task panicked")??;

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
        let chunk_size = self.config.chunk_size_bytes;
        let num_chunks = ((size_bytes as usize) + chunk_size - 1) / chunk_size;

        let effective_tier = match target_tier {
            Some(tier) => tier,
            None => {
                if let Some(ref pool) = self.ram_pool {
                    if self.config.ram_cache_enabled
                        && pool.get_used_bytes() + size_bytes <= pool.get_capacity_bytes()
                    {
                        StorageTier::Ram
                    } else {
                        StorageTier::Nvme
                    }
                } else {
                    StorageTier::Nvme
                }
            }
        };

        if effective_tier == StorageTier::Ram {
            if let Some(ref pool) = self.ram_pool {
                let mut layout = Vec::with_capacity(num_chunks);
                for local_idx in 0..num_chunks {
                    let chunk_idx = start_chunk_idx + local_idx;
                    let start = (local_idx * chunk_size) as u64;
                    let end = std::cmp::min(start + (chunk_size as u64), size_bytes);
                    let chunk_len = end - start;
                    let chunk_id = format!("{}_chunk_{}", uri, chunk_idx);
                    let chunk_path = pool.get_chunk_path(&chunk_id);
                    layout.push(ChunkDescriptor {
                        tier: StorageTier::Ram,
                        device: chunk_path.to_string_lossy().to_string(),
                        offset: 0,
                        length: chunk_len,
                    });
                }
                return Ok(layout);
            } else {
                anyhow::bail!("RAM cache tier requested but RAM caching is disabled.");
            }
        }

        let num_devices = self.config.block_devices.len();
        if num_devices == 0 {
            anyhow::bail!("No secondary raw NVMe block devices configured or discovered.");
        }

        let mut layout = Vec::with_capacity(num_chunks);
        for local_idx in 0..num_chunks {
            let chunk_idx = start_chunk_idx + local_idx;
            let start = (local_idx * chunk_size) as u64;
            let end = std::cmp::min(start + (chunk_size as u64), size_bytes);
            let chunk_len = end - start;

            let device_path = self.config.block_devices[chunk_idx % num_devices].clone();
            let offset = self
                .metadata
                .allocate_chunk_offset(
                    &device_path,
                    chunk_len,
                    self.config.device_start_offset_bytes,
                )
                .await?;

            layout.push(ChunkDescriptor {
                tier: StorageTier::Nvme,
                device: device_path,
                offset,
                length: chunk_len,
            });
        }

        Ok(layout)
    }

    /// Writes a single chunk at `chunk_idx` directly to the target tier (or spills over from RAM to NVMe if full).
    pub async fn stripe_single_chunk(
        &self,
        uri: &str,
        chunk_idx: usize,
        chunk_data: &[u8],
        target_tier: Option<StorageTier>,
    ) -> Result<ChunkDescriptor> {
        let chunk_len = chunk_data.len() as u64;
        let effective_tier = match target_tier {
            Some(tier) => tier,
            None => {
                if let Some(ref pool) = self.ram_pool {
                    if self.config.ram_cache_enabled
                        && pool.get_used_bytes() + chunk_len <= pool.get_capacity_bytes()
                    {
                        StorageTier::Ram
                    } else {
                        StorageTier::Nvme
                    }
                } else {
                    StorageTier::Nvme
                }
            }
        };

        if effective_tier == StorageTier::Ram {
            if let Some(ref pool) = self.ram_pool {
                let chunk_id = format!("{}_chunk_{}", uri, chunk_idx);
                if let Ok(desc) = pool.allocate_and_write_chunk(&chunk_id, chunk_data) {
                    return Ok(desc);
                } else if target_tier == Some(StorageTier::Ram) {
                    anyhow::bail!("RAM cache capacity exceeded while writing chunk {} for {}", chunk_idx, uri);
                }
                // If target_tier was None (Auto), fall through seamlessly to NVMe spillover!
            } else {
                anyhow::bail!("RAM cache tier requested but RAM caching is disabled.");
            }
        }

        let num_devices = self.config.block_devices.len();
        if num_devices == 0 {
            anyhow::bail!("No secondary raw NVMe block devices configured or discovered (OS root disk protected). Pass --tier ram to cache directly in Tier 1 Host RAM (/dev/shm) or configure secondary block devices in configs/hypercache-agent.toml.");
        }

        let device_path = self.config.block_devices[chunk_idx % num_devices].clone();
        let offset = self
            .metadata
            .allocate_chunk_offset(
                &device_path,
                chunk_len,
                self.config.device_start_offset_bytes,
            )
            .await?;

        let data_vec = chunk_data.to_vec();
        let dev_clone = device_path.clone();
        tokio::task::spawn_blocking(move || -> Result<()> {
            use std::os::unix::fs::FileExt;
            let file = OpenOptions::new()
                .write(true)
                .create(false)
                .open(&dev_clone)
                .with_context(|| format!("Failed to open device {}", dev_clone))?;
            file.write_all_at(&data_vec, offset)
                .with_context(|| format!("Failed to write chunk at offset {} on {}", offset, dev_clone))?;
            Ok(())
        })
        .await
        .with_context(|| "Write chunk task panicked")??;

        Ok(ChunkDescriptor {
            tier: StorageTier::Nvme,
            device: device_path,
            offset,
            length: chunk_len,
        })
    }

    #[allow(dead_code)]
    pub async fn stripe_bytes(&self, data: &[u8]) -> Result<Vec<ChunkDescriptor>> {
        self.stripe_bytes_to_tier("unspecified", data, None).await
    }
}
