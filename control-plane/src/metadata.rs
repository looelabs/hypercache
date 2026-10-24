use anyhow::{Context, Result};
use moka::future::Cache;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::protocol::ChunkDescriptor;
use crate::telemetry::MetricsState;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredMetadata {
    pub object_size: u64,
    pub chunk_size: usize,
    pub chunk_layout: Vec<ChunkDescriptor>,
    pub created_at: i64,
    #[serde(default)]
    pub sync_status: Option<String>,
    #[serde(default)]
    pub pinned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DeviceFreePool {
    /// Freed (offset, length) extents available for immediate reuse
    pub free_extents: Vec<(u64, u64)>,
    /// Maximum physical capacity of the device in bytes (0 = unbounded/unknown)
    pub capacity_bytes: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataManifest {
    pub entries: HashMap<String, StoredMetadata>,
    pub local_alloc: HashMap<String, u64>,
    #[serde(default)]
    pub free_pools: HashMap<String, DeviceFreePool>,
    pub updated_at: i64,
}

#[derive(Clone)]
pub struct MetadataStore {
    // Highly concurrent W-TinyLFU in-memory async cache for hot lookups (< 5 us)
    cache: Cache<String, StoredMetadata>,
    // Thread-safe mirror of active entries for atomic persistence snapshots
    entries_mirror: Arc<RwLock<HashMap<String, StoredMetadata>>>,
    // Thread-safe physical NVMe device allocation pointers
    local_alloc: Arc<RwLock<HashMap<String, u64>>>,
    // Per-device free block extents and physical capacity bounds
    free_pools: Arc<RwLock<HashMap<String, DeviceFreePool>>>,
    // Configured max capacity in bytes
    pub max_capacity_bytes: u64,
    // Optional metrics state link
    metrics: Option<Arc<MetricsState>>,
    // Optional persistence path
    persistence_path: Option<PathBuf>,
    // Optional Tier 1 RAM memory pool link
    pub ram_pool: Arc<tokio::sync::RwLock<Option<Arc<crate::ram::RamMemoryPool>>>>,
}

impl MetadataStore {
    /// Initializes an ephemeral in-memory Moka cache with size-aware W-TinyLFU eviction.
    /// Capacity is weighted in Megabytes (MiB) based on physical NVMe storage capacity.
    pub fn new(
        max_capacity_bytes: u64,
        metrics: Option<Arc<MetricsState>>,
        persistence_path: Option<PathBuf>,
    ) -> Self {
        let max_capacity_mib = (max_capacity_bytes / (1024 * 1024)).max(100);

        info!(
            "Initializing embedded Moka metadata cache (W-TinyLFU eviction, max storage capacity: {:.2} GiB)...",
            max_capacity_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        let entries_mirror = Arc::new(RwLock::new(HashMap::new()));
        let local_alloc = Arc::new(RwLock::new(HashMap::new()));
        let free_pools = Arc::new(RwLock::new(HashMap::<String, DeviceFreePool>::new()));
        let ram_pool: Arc<tokio::sync::RwLock<Option<Arc<crate::ram::RamMemoryPool>>>> =
            Arc::new(tokio::sync::RwLock::new(None));

        let metrics_evict = metrics.clone();
        let mirror_evict = Arc::clone(&entries_mirror);
        let alloc_evict = Arc::clone(&local_alloc);
        let free_evict = Arc::clone(&free_pools);
        let ppath_evict = persistence_path.clone();
        let ram_pool_evict = Arc::clone(&ram_pool);

        let cache = Cache::builder()
            .weigher(|_key: &String, val: &StoredMetadata| -> u32 {
                // Pinned or dirty write-back entries get 0 eviction weight in Moka so W-TinyLFU
                // never evicts uncommitted checkpoints or pinned base models under capacity pressure.
                if val.pinned || Self::is_dirty_status(val.sync_status.as_deref()) {
                    return 0;
                }
                let mib = (val.object_size + 1048575) / (1024 * 1024);
                mib.max(1) as u32
            })
            .max_capacity(max_capacity_mib)
            .eviction_listener(move |key: Arc<String>, val: StoredMetadata, cause| {
                if cause == moka::notification::RemovalCause::Replaced {
                    return;
                }
                // Explicit invalidations are already handled synchronously by evict_entry / purge_all
                if cause == moka::notification::RemovalCause::Explicit {
                    return;
                }

                info!(
                    "Moka W-TinyLFU Eviction: URI='{}' ({:.2} MiB, {} chunks) evicted. Reason: {:?}",
                    key,
                    val.object_size as f64 / (1024.0 * 1024.0),
                    val.chunk_layout.len(),
                    cause
                );
                if let Some(ref m) = metrics_evict {
                    m.evictions_total.fetch_add(1, Ordering::Relaxed);
                    m.evicted_bytes_total
                        .fetch_add(val.object_size, Ordering::Relaxed);
                }

                let mirror_clone = Arc::clone(&mirror_evict);
                let alloc_clone = Arc::clone(&alloc_evict);
                let free_clone = Arc::clone(&free_evict);
                let key_str = (*key).clone();
                let ppath_clone = ppath_evict.clone();
                let pool_clone = Arc::clone(&ram_pool_evict);
                let chunks = val.chunk_layout;

                tokio::spawn(async move {
                    let maybe_pool = { pool_clone.read().await.clone() };
                    let mut nvme_reclaimed = Vec::new();
                    for chunk in chunks {
                        if chunk.tier == crate::protocol::StorageTier::Ram {
                            if let Some(ref pool) = maybe_pool {
                                let _ = pool.delete_chunk(&chunk.device);
                            } else {
                                let p = std::path::Path::new(&chunk.device);
                                if p.exists() {
                                    let _ = std::fs::remove_file(p);
                                }
                            }
                        } else {
                            nvme_reclaimed.push((chunk.device, chunk.offset, chunk.length));
                        }
                    }

                    if !nvme_reclaimed.is_empty() {
                        let mut pools = free_clone.write().await;
                        for (dev, off, len) in nvme_reclaimed {
                            let aligned_len = Self::align_up_4k(len);
                            pools
                                .entry(dev)
                                .or_default()
                                .free_extents
                                .push((off, aligned_len));
                        }
                    }

                    {
                        let mut m = mirror_clone.write().await;
                        m.remove(&key_str);
                    }
                    if let Some(ref path) = ppath_clone {
                        let _ = Self::write_manifest_to_disk(
                            &mirror_clone,
                            &alloc_clone,
                            &free_clone,
                            path,
                        )
                        .await;
                    }
                });
            })
            .build();

        Self {
            cache,
            entries_mirror,
            local_alloc,
            free_pools,
            max_capacity_bytes,
            metrics,
            persistence_path,
            ram_pool,
        }
    }

    #[inline]
    pub fn is_dirty_status(status: Option<&str>) -> bool {
        matches!(status, Some("STAGED") | Some("COMMITTED") | Some("SYNCING_BACKEND"))
    }

    #[inline]
    fn align_up_4k(val: u64) -> u64 {
        (val + 4095) & !4095
    }

    /// Registers physical device capacities so the block allocator never overflows a drive
    pub async fn register_device_capacities(&self, devices: &[crate::discovery::DiscoveredDevice]) {
        let mut pools = self.free_pools.write().await;
        for d in devices {
            let pool = pools.entry(d.path.clone()).or_default();
            pool.capacity_bytes = d.size_bytes;
        }
    }

    /// Associates an active RAM memory pool for automatic chunk eviction & purge tracking
    pub async fn set_ram_pool(&self, pool: Arc<crate::ram::RamMemoryPool>) {
        let mut guard = self.ram_pool.write().await;
        *guard = Some(pool);
    }

    /// Fast-path in-process metadata lookup (< 5 microseconds, zero network/TCP roundtrip)
    pub async fn get_layout(&self, uri: &str) -> Result<Option<StoredMetadata>> {
        Ok(self.cache.get(uri).await)
    }

    /// Inserts or updates physical chunk routing layout in the Moka cache & persistence tier
    pub async fn save_layout(
        &self,
        uri: &str,
        object_size: u64,
        chunk_size: usize,
        chunk_layout: Vec<ChunkDescriptor>,
    ) -> Result<()> {
        self.save_layout_full(uri, object_size, chunk_size, chunk_layout, None, false)
            .await
    }

    /// Inserts or updates physical chunk routing layout with explicit sync status
    pub async fn save_layout_with_status(
        &self,
        uri: &str,
        object_size: u64,
        chunk_size: usize,
        chunk_layout: Vec<ChunkDescriptor>,
        sync_status: Option<String>,
    ) -> Result<()> {
        self.save_layout_full(uri, object_size, chunk_size, chunk_layout, sync_status, false)
            .await
    }

    /// Inserts or updates physical chunk routing layout with explicit sync status and pin state
    pub async fn save_layout_full(
        &self,
        uri: &str,
        object_size: u64,
        chunk_size: usize,
        chunk_layout: Vec<ChunkDescriptor>,
        sync_status: Option<String>,
        pinned: bool,
    ) -> Result<()> {
        // If replacing an existing entry for the same URI, reclaim its old blocks first
        let old_meta = {
            let mut mirror = self.entries_mirror.write().await;
            mirror.remove(uri)
        };
        if let Some(prev) = old_meta {
            self.reclaim_chunks(&prev.chunk_layout).await;
        }

        let meta = StoredMetadata {
            object_size,
            chunk_size,
            chunk_layout,
            created_at: chrono::Utc::now().timestamp(),
            sync_status,
            pinned,
        };

        self.cache.insert(uri.to_string(), meta.clone()).await;

        {
            let mut mirror = self.entries_mirror.write().await;
            mirror.insert(uri.to_string(), meta);
        }

        self.update_metrics_counters().await;
        self.spawn_background_manifest_sync();

        Ok(())
    }

    async fn update_metrics_counters(&self) {
        if let Some(ref m) = self.metrics {
            let mirror = self.entries_mirror.read().await;
            let count = mirror.len() as u64;
            let total_bytes: u64 = mirror.values().map(|v| v.object_size).sum();
            m.cached_objects_count.store(count, Ordering::Relaxed);
            m.storage_used_bytes.store(total_bytes, Ordering::Relaxed);
        }
    }

    fn spawn_background_manifest_sync(&self) {
        if let Some(ref path) = self.persistence_path {
            let path_clone = path.clone();
            let mirror_clone = Arc::clone(&self.entries_mirror);
            let alloc_clone = Arc::clone(&self.local_alloc);
            let free_clone = Arc::clone(&self.free_pools);

            tokio::spawn(async move {
                let _ = Self::write_manifest_to_disk(
                    &mirror_clone,
                    &alloc_clone,
                    &free_clone,
                    &path_clone,
                )
                .await;
            });
        }
    }

    /// Updates the synchronization status (e.g. STAGED -> COMMITTED -> SYNCING_BACKEND -> SYNCED_BACKEND)
    pub async fn update_sync_status(&self, uri: &str, sync_status: &str) -> Result<()> {
        if let Some(mut meta) = self.cache.get(uri).await {
            meta.sync_status = Some(sync_status.to_string());
            self.cache.insert(uri.to_string(), meta.clone()).await;
            {
                let mut mirror = self.entries_mirror.write().await;
                mirror.insert(uri.to_string(), meta);
            }
            self.spawn_background_manifest_sync();
        }
        Ok(())
    }

    /// Appends newly allocated chunks to an existing staged object (or creates a new STAGED object)
    pub async fn append_chunks_to_layout(
        &self,
        uri: &str,
        additional_size: u64,
        chunk_size: usize,
        mut new_chunks: Vec<ChunkDescriptor>,
    ) -> Result<(u64, Vec<ChunkDescriptor>)> {
        let mut meta = if let Some(existing) = self.cache.get(uri).await {
            existing
        } else {
            StoredMetadata {
                object_size: 0,
                chunk_size,
                chunk_layout: Vec::new(),
                created_at: chrono::Utc::now().timestamp(),
                sync_status: Some("STAGED".to_string()),
                pinned: false,
            }
        };

        meta.object_size += additional_size;
        meta.chunk_layout.append(&mut new_chunks);
        meta.sync_status = Some("STAGED".to_string());

        let total_size = meta.object_size;
        let full_layout = meta.chunk_layout.clone();

        self.cache.insert(uri.to_string(), meta.clone()).await;
        {
            let mut mirror = self.entries_mirror.write().await;
            mirror.insert(uri.to_string(), meta);
        }

        self.update_metrics_counters().await;
        self.spawn_background_manifest_sync();

        Ok((total_size, full_layout))
    }

    /// Sets or clears the pin state on an existing cached object
    pub async fn set_pinned(&self, uri: &str, pinned: bool) -> Result<bool> {
        if let Some(mut meta) = self.cache.get(uri).await {
            meta.pinned = pinned;
            self.cache.insert(uri.to_string(), meta.clone()).await;
            {
                let mut mirror = self.entries_mirror.write().await;
                mirror.insert(uri.to_string(), meta);
            }
            self.spawn_background_manifest_sync();
            Ok(true)
        } else {
            Ok(false)
        }
    }

    /// Gets the synchronization status for an object
    #[allow(dead_code)]
    pub async fn get_sync_status(&self, uri: &str) -> Result<Option<String>> {
        if let Some(meta) = self.cache.get(uri).await {
            Ok(meta.sync_status)
        } else {
            Ok(None)
        }
    }

    /// Reclaims physical NVMe extents or RAM shared memory files for a slice of chunks
    pub async fn reclaim_chunks(&self, chunks: &[ChunkDescriptor]) {
        let maybe_pool = { self.ram_pool.read().await.clone() };
        let mut nvme_reclaimed = Vec::new();

        for chunk in chunks {
            if chunk.tier == crate::protocol::StorageTier::Ram {
                if let Some(ref pool) = maybe_pool {
                    let _ = pool.delete_chunk(&chunk.device);
                } else {
                    let p = std::path::Path::new(&chunk.device);
                    if p.exists() {
                        let _ = std::fs::remove_file(p);
                    }
                }
            } else {
                nvme_reclaimed.push((chunk.device.clone(), chunk.offset, chunk.length));
            }
        }

        if !nvme_reclaimed.is_empty() {
            let mut pools = self.free_pools.write().await;
            for (dev, off, len) in nvme_reclaimed {
                let aligned_len = Self::align_up_4k(len);
                let pool = pools.entry(dev).or_default();
                pool.free_extents.push((off, aligned_len));
            }
        }
    }

    /// Allocates next contiguous or reclaimed 4K-aligned block offset on a physical NVMe/SSD drive.
    /// Reuses freed blocks from evicted entries first, and enforces physical drive capacity limits.
    pub async fn allocate_chunk_offset(
        &self,
        device: &str,
        length: u64,
        initial_offset: u64,
    ) -> Result<u64> {
        let aligned_len = Self::align_up_4k(length.max(4096));

        // 1. Check free block pool for a first-fit reclaimed extent
        let cap_limit = {
            let mut pools = self.free_pools.write().await;
            let pool = pools.entry(device.to_string()).or_default();

            if let Some(idx) = pool
                .free_extents
                .iter()
                .position(|&(_off, free_len)| free_len >= aligned_len)
            {
                let (off, free_len) = pool.free_extents.swap_remove(idx);
                if free_len > aligned_len {
                    pool.free_extents
                        .push((off + aligned_len, free_len - aligned_len));
                }
                return Ok(off);
            }
            pool.capacity_bytes
        };

        // 2. Bump-allocate from frontier pointer with 4K sector alignment and capacity check
        let mut alloc = self.local_alloc.write().await;
        let current = alloc.entry(device.to_string()).or_insert(initial_offset);
        let aligned_start = Self::align_up_4k(*current);

        if cap_limit > 0 && aligned_start + aligned_len > cap_limit {
            anyhow::bail!(
                "NVMe device '{}' capacity exhausted (requested {} bytes at offset {}, device capacity: {} bytes)",
                device,
                aligned_len,
                aligned_start,
                cap_limit
            );
        }

        *current = aligned_start + aligned_len;
        Ok(aligned_start)
    }

    /// Atomically persists the metadata manifest and allocation pointers to NVMe/disk
    pub async fn flush_to_disk(&self) -> Result<()> {
        if let Some(ref path) = self.persistence_path {
            Self::write_manifest_to_disk(
                &self.entries_mirror,
                &self.local_alloc,
                &self.free_pools,
                path,
            )
            .await?;
        }
        Ok(())
    }

    /// Internal atomic writer (.tmp -> sync -> atomic rename) supporting both JSON and fast Bincode (.bin)
    async fn write_manifest_to_disk(
        mirror: &Arc<RwLock<HashMap<String, StoredMetadata>>>,
        alloc: &Arc<RwLock<HashMap<String, u64>>>,
        free_pools: &Arc<RwLock<HashMap<String, DeviceFreePool>>>,
        path: &Path,
    ) -> Result<()> {
        let entries = { mirror.read().await.clone() };
        let local_alloc = { alloc.read().await.clone() };
        let pools = { free_pools.read().await.clone() };

        let manifest = MetadataManifest {
            entries,
            local_alloc,
            free_pools: pools,
            updated_at: chrono::Utc::now().timestamp(),
        };

        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let is_binary = path.extension().map_or(false, |ext| ext == "bin" || ext == "bincode");
        let payload_bytes = if is_binary {
            bincode::serialize(&manifest).context("Failed to bincode-serialize metadata manifest")?
        } else {
            serde_json::to_vec(&manifest).context("Failed to JSON-serialize metadata manifest")?
        };

        let tmp_path = path.with_extension("tmp");
        let mut file = File::create(&tmp_path)
            .with_context(|| format!("Failed to create temporary manifest at {:?}", tmp_path))?;
        file.write_all(&payload_bytes)?;
        file.sync_all()?;
        drop(file);

        std::fs::rename(&tmp_path, path).with_context(|| {
            format!(
                "Failed to atomically rename manifest from {:?} to {:?}",
                tmp_path, path
            )
        })?;

        // Also write a companion binary index (.bin) if primary path is .json for <20ms fast warm-boot
        if !is_binary {
            if let Ok(bin_bytes) = bincode::serialize(&manifest) {
                let bin_path = path.with_extension("bin");
                let bin_tmp = path.with_extension("bin.tmp");
                if let Ok(mut bf) = File::create(&bin_tmp) {
                    if bf.write_all(&bin_bytes).is_ok() && bf.sync_all().is_ok() {
                        drop(bf);
                        let _ = std::fs::rename(&bin_tmp, &bin_path);
                    }
                }
            }
        }

        Ok(())
    }

    /// Hot-loads cached metadata manifest from disk on daemon warm-boot (tries fast binary index first, then JSON)
    pub async fn load_from_disk(
        path: &Path,
        max_capacity_bytes: u64,
        metrics: Option<Arc<MetricsState>>,
    ) -> Result<Option<(Self, usize, u64)>> {
        if !path.exists() {
            return Ok(None);
        }

        // Check if companion binary manifest exists and is at least as fresh as JSON
        let bin_path = path.with_extension("bin");
        let mut loaded_manifest: Option<MetadataManifest> = None;

        if path.extension().map_or(false, |e| e == "bin" || e == "bincode") {
            if let Ok(bytes) = std::fs::read(path) {
                if let Ok(m) = bincode::deserialize::<MetadataManifest>(&bytes) {
                    loaded_manifest = Some(m);
                }
            }
        } else {
            // Primary is JSON: try JSON first to ensure external edits or tests writing JSON are respected,
            // or if JSON is very large (>10 MiB), use companion .bin if timestamps match.
            let use_bin_fastpath = if let (Ok(j_meta), Ok(b_meta)) =
                (std::fs::metadata(path), std::fs::metadata(&bin_path))
            {
                j_meta.len() > 10 * 1024 * 1024
                    && b_meta.modified().ok() >= j_meta.modified().ok()
            } else {
                false
            };

            if use_bin_fastpath {
                if let Ok(bytes) = std::fs::read(&bin_path) {
                    if let Ok(m) = bincode::deserialize::<MetadataManifest>(&bytes) {
                        loaded_manifest = Some(m);
                    }
                }
            }

            if loaded_manifest.is_none() {
                match std::fs::read_to_string(path) {
                    Ok(content) => match serde_json::from_str::<MetadataManifest>(&content) {
                        Ok(m) => loaded_manifest = Some(m),
                        Err(e) => {
                            warn!(
                                "Corrupted metadata manifest at {:?}: {}. Starting cold.",
                                path, e
                            );
                            return Ok(None);
                        }
                    },
                    Err(e) => {
                        warn!(
                            "Failed to read existing metadata manifest at {:?}: {}. Starting cold.",
                            path, e
                        );
                        return Ok(None);
                    }
                }
            }
        }

        let manifest = match loaded_manifest {
            Some(m) => m,
            None => return Ok(None),
        };

        let store = Self::new(
            max_capacity_bytes,
            metrics.clone(),
            Some(path.to_path_buf()),
        );

        let mut total_bytes: u64 = 0;
        let count = manifest.entries.len();

        {
            let mut mirror = store.entries_mirror.write().await;
            for (uri, meta) in manifest.entries {
                total_bytes += meta.object_size;
                store.cache.insert(uri.clone(), meta.clone()).await;
                mirror.insert(uri, meta);
            }
        }

        {
            let mut alloc = store.local_alloc.write().await;
            *alloc = manifest.local_alloc;
        }

        {
            let mut pools = store.free_pools.write().await;
            *pools = manifest.free_pools;
        }

        if let Some(ref m) = metrics {
            m.cached_objects_count
                .store(count as u64, Ordering::Relaxed);
            m.storage_used_bytes.store(total_bytes, Ordering::Relaxed);
        }

        Ok(Some((store, count, total_bytes)))
    }

    /// Returns a list of all active cached objects for `hypercache ls`
    pub async fn list_entries(&self) -> (Vec<crate::protocol::CachedObjectSummary>, u64) {
        let mirror = self.entries_mirror.read().await;
        let mut list = Vec::new();
        let mut total_bytes: u64 = 0;

        for (uri, meta) in mirror.iter() {
            total_bytes += meta.object_size;
            let mut devices_set = std::collections::HashSet::new();
            for chunk in &meta.chunk_layout {
                devices_set.insert(chunk.device.clone());
            }
            let mut devices: Vec<String> = devices_set.into_iter().collect();
            devices.sort();

            let tier = meta
                .chunk_layout
                .first()
                .map_or(crate::protocol::StorageTier::Nvme, |c| c.tier);

            list.push(crate::protocol::CachedObjectSummary {
                uri: uri.clone(),
                object_size: meta.object_size,
                chunk_count: meta.chunk_layout.len(),
                created_at: meta.created_at,
                devices,
                tier,
                sync_status: meta.sync_status.clone(),
                pinned: meta.pinned,
            });
        }

        list.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        (list, total_bytes)
    }

    /// Evicts a specific cached item from Moka, mirror map, NVMe manifest, and reclaims its blocks
    pub async fn evict_entry(&self, uri: &str) -> Result<Option<u64>> {
        let removed_meta = {
            let mut mirror = self.entries_mirror.write().await;
            mirror.remove(uri)
        };

        if let Some(meta) = removed_meta {
            self.cache.invalidate(uri).await;
            self.reclaim_chunks(&meta.chunk_layout).await;

            if let Some(ref m) = self.metrics {
                m.evictions_total.fetch_add(1, Ordering::Relaxed);
                m.evicted_bytes_total
                    .fetch_add(meta.object_size, Ordering::Relaxed);
            }
            self.update_metrics_counters().await;
            self.flush_to_disk().await?;
            Ok(Some(meta.object_size))
        } else {
            Ok(None)
        }
    }

    /// Purges all cached items from Moka, mirror map, NVMe manifest, and resets allocation
    pub async fn purge_all(&self) -> Result<(usize, u64)> {
        let (count, total_bytes, all_chunks) = {
            let mut mirror = self.entries_mirror.write().await;
            let count = mirror.len();
            let total_bytes: u64 = mirror.values().map(|m| m.object_size).sum();
            let mut chunks = Vec::new();
            for meta in mirror.values() {
                chunks.extend(meta.chunk_layout.clone());
            }
            mirror.clear();
            (count, total_bytes, chunks)
        };

        // Clean up RAM shared memory chunk files
        let maybe_pool = { self.ram_pool.read().await.clone() };
        if let Some(ref pool) = maybe_pool {
            let _ = pool.cleanup_orphaned_chunks();
        } else {
            for chunk in all_chunks {
                if chunk.tier == crate::protocol::StorageTier::Ram {
                    let p = std::path::Path::new(&chunk.device);
                    if p.exists() {
                        let _ = std::fs::remove_file(p);
                    }
                }
            }
        }

        self.cache.invalidate_all();

        {
            let mut alloc = self.local_alloc.write().await;
            for val in alloc.values_mut() {
                *val = 4096; // reset to safety start offset
            }
        }

        {
            let mut pools = self.free_pools.write().await;
            for pool in pools.values_mut() {
                pool.free_extents.clear();
            }
        }

        if let Some(ref m) = self.metrics {
            m.cached_objects_count.store(0, Ordering::Relaxed);
            m.storage_used_bytes.store(0, Ordering::Relaxed);
        }

        self.flush_to_disk().await?;
        Ok((count, total_bytes))
    }

    /// Returns live cache statistics for diagnostics & monitoring
    #[allow(dead_code)]
    pub fn get_stats(&self) -> (u64, u64, u64) {
        let entry_count = self.cache.entry_count();
        let weighted_size_mib = self.cache.weighted_size();
        let weighted_size_bytes = weighted_size_mib * 1024 * 1024;
        (entry_count, weighted_size_bytes, self.max_capacity_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::StorageTier;

    #[tokio::test]
    async fn test_moka_metadata_hit_miss() {
        let store = MetadataStore::new(10 * 1024 * 1024 * 1024, None, None);

        let uri = "s3://models/llama3/weights.safetensors";
        // 1. Initial lookup -> Miss
        let miss = store.get_layout(uri).await.unwrap();
        assert!(miss.is_none());

        // 2. Save layout
        let chunks = vec![ChunkDescriptor {
            tier: StorageTier::Nvme,
            device: "/dev/nvme0n1".to_string(),
            offset: 4096,
            length: 4194304,
        }];
        store
            .save_layout(uri, 4194304, 4194304, chunks.clone())
            .await
            .unwrap();

        // 3. Second lookup -> Hit
        let hit = store.get_layout(uri).await.unwrap();
        assert!(hit.is_some());
        let meta = hit.unwrap();
        assert_eq!(meta.object_size, 4194304);
        assert_eq!(meta.chunk_layout, chunks);
    }

    #[tokio::test]
    async fn test_moka_weigher_and_allocation() {
        let store = MetadataStore::new(100 * 1024 * 1024, None, None);

        let off1 = store
            .allocate_chunk_offset("/dev/nvme0n1", 4194304, 4096)
            .await
            .unwrap();
        let off2 = store
            .allocate_chunk_offset("/dev/nvme0n1", 4194304, 4096)
            .await
            .unwrap();
        let off3 = store
            .allocate_chunk_offset("/dev/nvme1n1", 4194304, 4096)
            .await
            .unwrap();

        assert_eq!(off1, 4096);
        assert_eq!(off2, 4096 + 4194304);
        assert_eq!(off3, 4096);
    }

    #[tokio::test]
    async fn test_free_block_reclamation_and_capacity_bounds() {
        let store = MetadataStore::new(100 * 1024 * 1024, None, None);
        store
            .register_device_capacities(&[crate::discovery::DiscoveredDevice {
                path: "/dev/nvme0n1".to_string(),
                is_nvme: true,
                is_virtual: false,
                is_partition: false,
                is_rotational: false,
                size_bytes: 4096 + 2 * 4194304, // Exactly room for 2 chunks of 4 MiB after 4096 header
                model: None,
                existing_fs: None,
                numa_node: Some(0),
            }])
            .await;

        let off1 = store
            .allocate_chunk_offset("/dev/nvme0n1", 4194304, 4096)
            .await
            .unwrap();
        let off2 = store
            .allocate_chunk_offset("/dev/nvme0n1", 4194304, 4096)
            .await
            .unwrap();
        assert_eq!(off1, 4096);
        assert_eq!(off2, 4096 + 4194304);

        // 3rd allocation must fail because capacity is reached
        let off3 = store
            .allocate_chunk_offset("/dev/nvme0n1", 4194304, 4096)
            .await;
        assert!(off3.is_err());

        // Save layout for chunk 1 and then evict it
        store
            .save_layout(
                "hypercache://models/chunk1.bin",
                4194304,
                4194304,
                vec![ChunkDescriptor {
                    tier: StorageTier::Nvme,
                    device: "/dev/nvme0n1".to_string(),
                    offset: off1,
                    length: 4194304,
                }],
            )
            .await
            .unwrap();

        store
            .evict_entry("hypercache://models/chunk1.bin")
            .await
            .unwrap();

        // Now allocating 4 MiB again must succeed and reuse `off1` (4096)!
        let off_reused = store
            .allocate_chunk_offset("/dev/nvme0n1", 4194304, 4096)
            .await
            .unwrap();
        assert_eq!(off_reused, 4096);
    }

    #[tokio::test]
    async fn test_metadata_persistence_roundtrip() {
        let path = std::env::temp_dir().join(format!(
            "hypercache_test_{}.json",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));

        let store = MetadataStore::new(10 * 1024 * 1024 * 1024, None, Some(path.clone()));

        let uri1 = "hypercache://models/llama-3/weights.safetensors";
        let chunks1 = vec![
            ChunkDescriptor {
                tier: StorageTier::Nvme,
                device: "/dev/vdc".to_string(),
                offset: 4096,
                length: 4194304,
            },
            ChunkDescriptor {
                tier: StorageTier::Nvme,
                device: "/dev/vdd".to_string(),
                offset: 4096,
                length: 4194304,
            },
        ];
        store
            .save_layout(uri1, 8388608, 4194304, chunks1.clone())
            .await
            .unwrap();
        let _ = store
            .allocate_chunk_offset("/dev/vdc", 4194304, 4096)
            .await
            .unwrap();
        let _ = store
            .allocate_chunk_offset("/dev/vdd", 4194304, 4096)
            .await
            .unwrap();

        // Flush to disk explicitly
        store.flush_to_disk().await.unwrap();

        // Drop store and restore from disk
        drop(store);

        let loaded = MetadataStore::load_from_disk(&path, 10 * 1024 * 1024 * 1024, None)
            .await
            .unwrap();
        assert!(loaded.is_some());

        let (restored_store, count, total_bytes) = loaded.unwrap();
        assert_eq!(count, 1);
        assert_eq!(total_bytes, 8388608);

        // Verify restored hit
        let hit = restored_store.get_layout(uri1).await.unwrap();
        assert!(hit.is_some());
        let meta = hit.unwrap();
        assert_eq!(meta.object_size, 8388608);
        assert_eq!(meta.chunk_layout, chunks1);

        // Verify allocation pointer continued from last offset
        let next_off = restored_store
            .allocate_chunk_offset("/dev/vdc", 4194304, 4096)
            .await
            .unwrap();
        assert_eq!(next_off, 4096 + 4194304);

        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(path.with_extension("bin"));
    }

    #[tokio::test]
    async fn test_atomic_manifest_recovery_after_crash() {
        let path = std::env::temp_dir().join(format!(
            "hypercache_corrupt_{}.json",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));

        // Write corrupt JSON
        std::fs::write(&path, b"{ corrupted_invalid_json_data").unwrap();

        let loaded = MetadataStore::load_from_disk(&path, 10 * 1024 * 1024 * 1024, None)
            .await
            .unwrap();

        // Corrupted file gracefully ignored, starts fresh
        assert!(loaded.is_none());

        let _ = std::fs::remove_file(&path);
    }

    #[tokio::test]
    async fn test_list_evict_and_purge() {
        let store = MetadataStore::new(10 * 1024 * 1024 * 1024, None, None);

        let uri1 = "hypercache://models/llama-3/weights.safetensors";
        let uri2 = "hypercache://models/mistral/weights.safetensors";

        store
            .save_layout(
                uri1,
                4194304,
                4194304,
                vec![ChunkDescriptor {
                    tier: StorageTier::Nvme,
                    device: "/dev/vdc".to_string(),
                    offset: 4096,
                    length: 4194304,
                }],
            )
            .await
            .unwrap();
        store
            .save_layout(
                uri2,
                8388608,
                4194304,
                vec![ChunkDescriptor {
                    tier: StorageTier::Nvme,
                    device: "/dev/vdd".to_string(),
                    offset: 4096,
                    length: 8388608,
                }],
            )
            .await
            .unwrap();

        // 1. List entries
        let (list, total) = store.list_entries().await;
        assert_eq!(list.len(), 2);
        assert_eq!(total, 12582912);

        // 2. Evict uri1
        let freed = store.evict_entry(uri1).await.unwrap();
        assert_eq!(freed, Some(4194304));

        let (list2, total2) = store.list_entries().await;
        assert_eq!(list2.len(), 1);
        assert_eq!(total2, 8388608);

        // 3. Purge all
        let (purged_count, purged_freed) = store.purge_all().await.unwrap();
        assert_eq!(purged_count, 1);
        assert_eq!(purged_freed, 8388608);

        let (list3, total3) = store.list_entries().await;
        assert_eq!(list3.len(), 0);
        assert_eq!(total3, 0);
    }

    #[tokio::test]
    async fn test_pinned_and_dirty_eviction_protection() {
        // Max capacity = 12 MiB (3 x 4 MiB)
        let store = MetadataStore::new(12 * 1024 * 1024, None, None);

        let pinned_uri = "hypercache://models/base/llama3_pinned.safetensors";
        let dirty_uri = "hypercache://checkpoints/dirty_epoch_1.pt";

        // 1. Insert a pinned base model (8 MiB)
        store
            .save_layout_full(
                pinned_uri,
                8 * 1024 * 1024,
                4 * 1024 * 1024,
                vec![ChunkDescriptor {
                    tier: StorageTier::Nvme,
                    device: "/dev/vdc".to_string(),
                    offset: 4096,
                    length: 8 * 1024 * 1024,
                }],
                None,
                true,
            )
            .await
            .unwrap();

        // 2. Insert a dirty uncommitted write-back checkpoint (8 MiB, STAGED)
        store
            .save_layout_full(
                dirty_uri,
                8 * 1024 * 1024,
                4 * 1024 * 1024,
                vec![ChunkDescriptor {
                    tier: StorageTier::Nvme,
                    device: "/dev/vdd".to_string(),
                    offset: 4096,
                    length: 8 * 1024 * 1024,
                }],
                Some("STAGED".to_string()),
                false,
            )
            .await
            .unwrap();

        // 3. Flood the cache with 10 unpinned objects of 4 MiB each (40 MiB total >> 12 MiB capacity)
        for i in 0..10 {
            let u = format!("hypercache://ephemeral/item_{}.bin", i);
            store
                .save_layout(
                    &u,
                    4 * 1024 * 1024,
                    4 * 1024 * 1024,
                    vec![ChunkDescriptor {
                        tier: StorageTier::Nvme,
                        device: "/dev/vdc".to_string(),
                        offset: 100_000 + (i as u64) * 4194304,
                        length: 4 * 1024 * 1024,
                    }],
                )
                .await
                .unwrap();
        }
        store.cache.run_pending_tasks().await;

        // Pinned model and dirty STAGED checkpoint must still be present in Moka!
        let pinned_lookup = store.get_layout(pinned_uri).await.unwrap();
        assert!(pinned_lookup.is_some());
        assert!(pinned_lookup.unwrap().pinned);

        let dirty_lookup = store.get_layout(dirty_uri).await.unwrap();
        assert!(dirty_lookup.is_some());
    }
}
