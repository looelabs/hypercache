use anyhow::{Context, Result};
use moka::future::Cache;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, RwLock, Weak};
use std::time::{Duration, Instant};
use tokio::sync::Notify;
use tracing::{info, warn};

use crate::protocol::{ChunkDescriptor, StorageTier, SyncStatus};
use crate::telemetry::MetricsState;

/// Default grace period before freed NVMe extents become reusable. Clients resolve a layout
/// and then read raw device offsets directly, so extents must not be overwritten while a
/// reader that resolved the old layout may still be in flight.
pub const DEFAULT_RECLAIM_GRACE: Duration = Duration::from_secs(30);

/// Upper bound on objects evicted to satisfy a single allocation under capacity pressure.
const MAX_PRESSURE_EVICTIONS: usize = 64;

/// Coalescing delay for background manifest writes (bursts of mutations become one write).
const MANIFEST_WRITE_DEBOUNCE: Duration = Duration::from_millis(20);

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StoredMetadata {
    pub object_size: u64,
    pub chunk_size: usize,
    pub chunk_layout: Vec<ChunkDescriptor>,
    pub created_at: i64,
    #[serde(default)]
    pub sync_status: Option<SyncStatus>,
    #[serde(default)]
    pub pinned: bool,
    /// Monotonic layout generation, bumped on every re-ingest / re-stage of the same URI
    #[serde(default)]
    pub generation: u64,
}

impl StoredMetadata {
    pub fn is_dirty(&self) -> bool {
        self.sync_status.is_some_and(|s| s.is_dirty())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct DeviceFreePool {
    /// Freed (offset, length) extents available for reuse, sorted by offset and coalesced
    pub free_extents: Vec<(u64, u64)>,
    /// Maximum physical capacity of the device in bytes (0 = unbounded/unknown)
    pub capacity_bytes: u64,
}

impl DeviceFreePool {
    /// Inserts a freed extent, keeping the list sorted and merging adjacent or overlapping extents.
    pub fn insert_extent(&mut self, offset: u64, length: u64) {
        if length == 0 {
            return;
        }
        let idx = self.free_extents.partition_point(|&(o, _)| o < offset);
        self.free_extents.insert(idx, (offset, length));

        // Merge with following extents
        while idx + 1 < self.free_extents.len() {
            let (cur_off, cur_len) = self.free_extents[idx];
            let (next_off, next_len) = self.free_extents[idx + 1];
            if cur_off + cur_len >= next_off {
                let end = (cur_off + cur_len).max(next_off + next_len);
                self.free_extents[idx] = (cur_off, end - cur_off);
                self.free_extents.remove(idx + 1);
            } else {
                break;
            }
        }

        // Merge with preceding extent
        if idx > 0 {
            let (prev_off, prev_len) = self.free_extents[idx - 1];
            let (cur_off, cur_len) = self.free_extents[idx];
            if prev_off + prev_len >= cur_off {
                let end = (prev_off + prev_len).max(cur_off + cur_len);
                self.free_extents[idx - 1] = (prev_off, end - prev_off);
                self.free_extents.remove(idx);
            }
        }
    }

    /// Takes the lowest-offset free extent that fits `length` bytes.
    pub fn take_first_fit(&mut self, length: u64) -> Option<u64> {
        let idx = self.free_extents.iter().position(|&(_, free_len)| free_len >= length)?;
        let (off, free_len) = self.free_extents[idx];
        if free_len == length {
            self.free_extents.remove(idx);
        } else {
            self.free_extents[idx] = (off + length, free_len - length);
        }
        Some(off)
    }

    /// Re-sorts and coalesces extents (used for manifests written by older releases).
    pub fn normalize(&mut self) {
        let mut extents = std::mem::take(&mut self.free_extents);
        extents.sort_unstable();
        for (off, len) in extents {
            self.insert_extent(off, len);
        }
    }

    pub fn free_bytes(&self) -> u64 {
        self.free_extents.iter().map(|&(_, l)| l).sum()
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetadataManifest {
    pub entries: HashMap<String, StoredMetadata>,
    pub local_alloc: HashMap<String, u64>,
    #[serde(default)]
    pub free_pools: HashMap<String, DeviceFreePool>,
    pub updated_at: i64,
}

#[derive(Debug, Clone)]
struct QuarantinedExtent {
    released_at: Instant,
    device: String,
    offset: u64,
    length: u64,
}

struct Persister {
    path: PathBuf,
    notify: Arc<Notify>,
    write_lock: tokio::sync::Mutex<()>,
    writer_started: AtomicBool,
}

struct Inner {
    entries_mirror: Mutex<HashMap<String, StoredMetadata>>,
    local_alloc: Mutex<HashMap<String, u64>>,
    free_pools: Mutex<HashMap<String, DeviceFreePool>>,
    quarantine: Mutex<VecDeque<QuarantinedExtent>>,
    ram_pool: RwLock<Option<Arc<crate::ram::RamMemoryPool>>>,
    generation: AtomicU64,
    reclaim_grace_ms: AtomicU64,
    metrics: Option<Arc<MetricsState>>,
    persister: Option<Persister>,
}

#[derive(Clone)]
pub struct MetadataStore {
    // Highly concurrent W-TinyLFU in-memory async cache for hot lookups
    cache: Cache<String, StoredMetadata>,
    inner: Arc<Inner>,
    // Configured max capacity in bytes
    pub max_capacity_bytes: u64,
}

impl MetadataStore {
    /// Initializes an in-memory Moka cache with size-aware W-TinyLFU eviction.
    /// Capacity is weighted in Megabytes (MiB) based on physical storage capacity.
    pub fn new(max_capacity_bytes: u64, metrics: Option<Arc<MetricsState>>, persistence_path: Option<PathBuf>) -> Self {
        let max_capacity_mib = (max_capacity_bytes / (1024 * 1024)).max(100);

        info!(
            "Initializing embedded Moka metadata cache (W-TinyLFU eviction, max storage capacity: {:.2} GiB)...",
            max_capacity_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
        );

        let inner = Arc::new(Inner {
            entries_mirror: Mutex::new(HashMap::new()),
            local_alloc: Mutex::new(HashMap::new()),
            free_pools: Mutex::new(HashMap::new()),
            quarantine: Mutex::new(VecDeque::new()),
            ram_pool: RwLock::new(None),
            generation: AtomicU64::new(1),
            reclaim_grace_ms: AtomicU64::new(DEFAULT_RECLAIM_GRACE.as_millis() as u64),
            metrics,
            persister: persistence_path.map(|path| Persister {
                path,
                notify: Arc::new(Notify::new()),
                write_lock: tokio::sync::Mutex::new(()),
                writer_started: AtomicBool::new(false),
            }),
        });

        let listener_inner: Weak<Inner> = Arc::downgrade(&inner);

        let cache = Cache::builder()
            .weigher(|_key: &String, val: &StoredMetadata| -> u32 {
                // Pinned or dirty write-back entries get 0 eviction weight in Moka so W-TinyLFU
                // never evicts uncommitted checkpoints or pinned base models under capacity pressure.
                if val.pinned || val.is_dirty() {
                    return 0;
                }
                let mib = val.object_size.div_ceil(1024 * 1024);
                mib.clamp(1, u32::MAX as u64) as u32
            })
            .max_capacity(max_capacity_mib)
            .eviction_listener(move |key: Arc<String>, val: StoredMetadata, cause| {
                // Replacements and explicit invalidations are handled synchronously by the
                // code paths that triggered them (save_layout_full / evict_entry / purge_all).
                if cause == moka::notification::RemovalCause::Replaced
                    || cause == moka::notification::RemovalCause::Explicit
                {
                    return;
                }
                let Some(inner) = listener_inner.upgrade() else {
                    return;
                };

                // Only drop the mirror entry if it still refers to the evicted generation;
                // a concurrent re-ingest may already have installed a newer layout.
                let removed = {
                    let mut mirror = inner.entries_mirror.lock().unwrap();
                    match mirror.get(key.as_str()) {
                        Some(current) if current.generation == val.generation => {
                            mirror.remove(key.as_str());
                            true
                        }
                        Some(_) => false,
                        None => true,
                    }
                };
                if !removed {
                    return;
                }

                info!(
                    "Moka W-TinyLFU Eviction: URI='{}' ({:.2} MiB, {} chunks) evicted. Reason: {:?}",
                    key,
                    val.object_size as f64 / (1024.0 * 1024.0),
                    val.chunk_layout.len(),
                    cause
                );
                if let Some(ref m) = inner.metrics {
                    m.evictions_total.fetch_add(1, Ordering::Relaxed);
                    m.evicted_bytes_total.fetch_add(val.object_size, Ordering::Relaxed);
                }
                Self::quarantine_chunks_inner(&inner, &val.chunk_layout);
                Self::update_metrics_counters_inner(&inner);
                Self::mark_dirty_inner(&inner);
            })
            .build();

        Self {
            cache,
            inner,
            max_capacity_bytes,
        }
    }

    /// Overrides the grace period before freed NVMe extents are reused (0 = immediate reuse).
    pub fn set_reclaim_grace(&self, grace: Duration) {
        self.inner
            .reclaim_grace_ms
            .store(grace.as_millis() as u64, Ordering::Relaxed);
    }

    fn reclaim_grace(&self) -> Duration {
        Duration::from_millis(self.inner.reclaim_grace_ms.load(Ordering::Relaxed))
    }

    /// Returns a fresh, process-unique layout generation number.
    pub fn next_generation(&self) -> u64 {
        self.inner.generation.fetch_add(1, Ordering::Relaxed)
    }

    #[inline]
    fn align_up_4k(val: u64) -> u64 {
        (val + 4095) & !4095
    }

    /// Registers physical device capacities so the block allocator never overflows a drive
    pub async fn register_device_capacities(&self, devices: &[crate::discovery::DiscoveredDevice]) {
        let mut pools = self.inner.free_pools.lock().unwrap();
        for d in devices {
            let pool = pools.entry(d.path.clone()).or_default();
            pool.capacity_bytes = d.size_bytes;
        }
    }

    /// Associates an active RAM memory pool for automatic chunk eviction & purge tracking
    pub async fn set_ram_pool(&self, pool: Arc<crate::ram::RamMemoryPool>) {
        *self.inner.ram_pool.write().unwrap() = Some(pool);
    }

    /// Fast-path in-process metadata lookup (zero network/TCP roundtrip)
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
    ) -> Result<u64> {
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
        sync_status: Option<SyncStatus>,
    ) -> Result<u64> {
        self.save_layout_full(uri, object_size, chunk_size, chunk_layout, sync_status, false)
            .await
    }

    /// Inserts or updates a layout with explicit sync status and pin state.
    /// Returns the new layout generation. Blocks of a replaced layout are quarantined
    /// for the reclaim grace period before reuse.
    pub async fn save_layout_full(
        &self,
        uri: &str,
        object_size: u64,
        chunk_size: usize,
        chunk_layout: Vec<ChunkDescriptor>,
        sync_status: Option<SyncStatus>,
        pinned: bool,
    ) -> Result<u64> {
        let generation = self.next_generation();
        let meta = StoredMetadata {
            object_size,
            chunk_size,
            chunk_layout,
            created_at: chrono::Utc::now().timestamp(),
            sync_status,
            pinned,
            generation,
        };

        let old_meta = {
            let mut mirror = self.inner.entries_mirror.lock().unwrap();
            mirror.insert(uri.to_string(), meta.clone())
        };
        self.cache.insert(uri.to_string(), meta).await;

        if let Some(prev) = old_meta {
            self.quarantine_chunks(&prev.chunk_layout);
        }

        self.update_metrics_counters();
        self.mark_dirty();

        Ok(generation)
    }

    fn update_metrics_counters(&self) {
        Self::update_metrics_counters_inner(&self.inner);
    }

    fn update_metrics_counters_inner(inner: &Inner) {
        if let Some(ref m) = inner.metrics {
            let mirror = inner.entries_mirror.lock().unwrap();
            let count = mirror.len() as u64;
            let total_bytes: u64 = mirror.values().map(|v| v.object_size).sum();
            m.cached_objects_count.store(count, Ordering::Relaxed);
            m.storage_used_bytes.store(total_bytes, Ordering::Relaxed);
        }
    }

    fn mark_dirty(&self) {
        Self::mark_dirty_inner(&self.inner);
    }

    /// Signals the single background manifest writer. Bursts of mutations are coalesced
    /// into one atomic write, and writes are serialized so an older snapshot can never
    /// replace a newer one.
    fn mark_dirty_inner(inner: &Arc<Inner>) {
        let Some(ref persister) = inner.persister else {
            return;
        };
        if !persister.writer_started.load(Ordering::Acquire) {
            if let Ok(handle) = tokio::runtime::Handle::try_current() {
                if !persister.writer_started.swap(true, Ordering::AcqRel) {
                    let weak = Arc::downgrade(inner);
                    let notify = Arc::clone(&persister.notify);
                    handle.spawn(async move {
                        loop {
                            notify.notified().await;
                            tokio::time::sleep(MANIFEST_WRITE_DEBOUNCE).await;
                            let Some(inner) = weak.upgrade() else {
                                break;
                            };
                            if let Err(e) = Self::write_manifest(&inner).await {
                                warn!("Background metadata manifest write failed: {}", e);
                            }
                        }
                    });
                }
            }
        }
        persister.notify.notify_one();
    }

    /// Updates the synchronization status (e.g. STAGED -> COMMITTED -> SYNCING_BACKEND -> SYNCED_BACKEND)
    pub async fn update_sync_status(&self, uri: &str, sync_status: SyncStatus) -> Result<()> {
        self.update_sync_status_inner(uri, None, sync_status).await;
        Ok(())
    }

    /// Updates the sync status only if the object still has the given layout generation.
    /// Returns false if the object was replaced or removed in the meantime.
    pub async fn update_sync_status_if_generation(&self, uri: &str, generation: u64, sync_status: SyncStatus) -> bool {
        self.update_sync_status_inner(uri, Some(generation), sync_status).await
    }

    async fn update_sync_status_inner(&self, uri: &str, generation: Option<u64>, sync_status: SyncStatus) -> bool {
        let updated = {
            let mut mirror = self.inner.entries_mirror.lock().unwrap();
            match mirror.get_mut(uri) {
                Some(meta) if generation.is_none_or(|g| g == meta.generation) => {
                    meta.sync_status = Some(sync_status);
                    Some(meta.clone())
                }
                _ => None,
            }
        };
        match updated {
            Some(meta) => {
                self.cache.insert(uri.to_string(), meta).await;
                self.mark_dirty();
                true
            }
            None => false,
        }
    }

    /// Records per-chunk checksums for a layout generation (used after staged client writes are verified).
    pub async fn set_chunk_checksums(&self, uri: &str, generation: u64, checksums: &[u32]) -> bool {
        let updated = {
            let mut mirror = self.inner.entries_mirror.lock().unwrap();
            match mirror.get_mut(uri) {
                Some(meta) if meta.generation == generation && meta.chunk_layout.len() == checksums.len() => {
                    for (chunk, crc) in meta.chunk_layout.iter_mut().zip(checksums) {
                        chunk.checksum = Some(*crc);
                    }
                    Some(meta.clone())
                }
                _ => None,
            }
        };
        match updated {
            Some(meta) => {
                self.cache.insert(uri.to_string(), meta).await;
                self.mark_dirty();
                true
            }
            None => false,
        }
    }

    /// Appends newly allocated chunks to an existing staged object (or creates a new STAGED object)
    pub async fn append_chunks_to_layout(
        &self,
        uri: &str,
        additional_size: u64,
        chunk_size: usize,
        mut new_chunks: Vec<ChunkDescriptor>,
    ) -> Result<(u64, Vec<ChunkDescriptor>)> {
        let meta = {
            let mut mirror = self.inner.entries_mirror.lock().unwrap();
            let generation = self.inner.generation.fetch_add(1, Ordering::Relaxed);
            let meta = mirror.entry(uri.to_string()).or_insert_with(|| StoredMetadata {
                object_size: 0,
                chunk_size,
                chunk_layout: Vec::new(),
                created_at: chrono::Utc::now().timestamp(),
                sync_status: Some(SyncStatus::Staged),
                pinned: false,
                generation,
            });
            meta.object_size += additional_size;
            meta.chunk_layout.append(&mut new_chunks);
            meta.sync_status = Some(SyncStatus::Staged);
            meta.clone()
        };

        let total_size = meta.object_size;
        let full_layout = meta.chunk_layout.clone();
        self.cache.insert(uri.to_string(), meta).await;

        self.update_metrics_counters();
        self.mark_dirty();

        Ok((total_size, full_layout))
    }

    /// Sets or clears the pin state on an existing cached object
    pub async fn set_pinned(&self, uri: &str, pinned: bool) -> Result<bool> {
        let updated = {
            let mut mirror = self.inner.entries_mirror.lock().unwrap();
            mirror.get_mut(uri).map(|meta| {
                meta.pinned = pinned;
                meta.clone()
            })
        };
        match updated {
            Some(meta) => {
                self.cache.insert(uri.to_string(), meta).await;
                self.mark_dirty();
                Ok(true)
            }
            None => Ok(false),
        }
    }

    /// Gets the synchronization status for an object
    pub async fn get_sync_status(&self, uri: &str) -> Result<Option<SyncStatus>> {
        Ok(self.cache.get(uri).await.and_then(|m| m.sync_status))
    }

    /// Lists objects whose data has not reached the authoritative backend yet
    /// (re-queued for write-back on daemon boot).
    pub async fn list_dirty_entries(&self) -> Vec<(String, StoredMetadata)> {
        let mirror = self.inner.entries_mirror.lock().unwrap();
        let mut dirty: Vec<(String, StoredMetadata)> = mirror
            .iter()
            .filter(|(_, m)| {
                matches!(
                    m.sync_status,
                    Some(SyncStatus::Committed) | Some(SyncStatus::SyncingBackend) | Some(SyncStatus::SyncFailed)
                )
            })
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        dirty.sort_by_key(|(_, m)| m.created_at);
        dirty
    }

    /// Releases chunks of a layout that was published to clients: RAM files are unlinked
    /// (open readers keep their inode), NVMe extents enter the reclaim quarantine.
    pub async fn reclaim_chunks(&self, chunks: &[ChunkDescriptor]) {
        self.quarantine_chunks(chunks);
    }

    fn quarantine_chunks(&self, chunks: &[ChunkDescriptor]) {
        Self::quarantine_chunks_inner(&self.inner, chunks);
    }

    fn quarantine_chunks_inner(inner: &Inner, chunks: &[ChunkDescriptor]) {
        Self::delete_ram_chunks(inner, chunks);
        let now = Instant::now();
        let mut q = inner.quarantine.lock().unwrap();
        for chunk in chunks.iter().filter(|c| c.tier != StorageTier::Ram) {
            q.push_back(QuarantinedExtent {
                released_at: now,
                device: chunk.device.clone(),
                offset: chunk.offset,
                length: Self::align_up_4k(chunk.length.max(4096)),
            });
        }
    }

    /// Immediately returns chunks that were allocated but never published to any client
    /// (failed or aborted ingests). No grace period is needed because nobody can be reading them.
    pub async fn release_unpublished(&self, chunks: &[ChunkDescriptor]) {
        Self::delete_ram_chunks(&self.inner, chunks);
        let mut pools = self.inner.free_pools.lock().unwrap();
        let mut alloc = self.inner.local_alloc.lock().unwrap();
        for chunk in chunks.iter().filter(|c| c.tier != StorageTier::Ram) {
            Self::free_extent_locked(
                &mut pools,
                &mut alloc,
                &chunk.device,
                chunk.offset,
                Self::align_up_4k(chunk.length.max(4096)),
            );
        }
    }

    fn delete_ram_chunks(inner: &Inner, chunks: &[ChunkDescriptor]) {
        let maybe_pool = inner.ram_pool.read().unwrap().clone();
        for chunk in chunks.iter().filter(|c| c.tier == StorageTier::Ram) {
            if let Some(ref pool) = maybe_pool {
                let _ = pool.delete_chunk(&chunk.device);
            } else {
                let p = Path::new(&chunk.device);
                if p.exists() {
                    let _ = std::fs::remove_file(p);
                }
            }
        }
    }

    /// Returns an extent to the device free pool; if it ends at the bump frontier the
    /// frontier is pulled back instead so space at the tail is reused first.
    fn free_extent_locked(
        pools: &mut HashMap<String, DeviceFreePool>,
        alloc: &mut HashMap<String, u64>,
        device: &str,
        offset: u64,
        length: u64,
    ) {
        let pool = pools.entry(device.to_string()).or_default();
        pool.insert_extent(offset, length);
        if let Some(frontier) = alloc.get_mut(device) {
            while let Some(&(last_off, last_len)) = pool.free_extents.last() {
                if last_off + last_len == *frontier {
                    *frontier = last_off;
                    pool.free_extents.pop();
                } else {
                    break;
                }
            }
        }
    }

    /// Moves quarantined extents whose grace period expired into the free pools.
    /// With `force_device`, extents on that device are released regardless of age.
    fn drain_quarantine(&self, force_device: Option<&str>) -> usize {
        let grace = self.reclaim_grace();
        let now = Instant::now();
        let mut released = Vec::new();
        {
            let mut q = self.inner.quarantine.lock().unwrap();
            if let Some(dev) = force_device {
                let mut keep = VecDeque::with_capacity(q.len());
                for ext in q.drain(..) {
                    if ext.device == dev || now.duration_since(ext.released_at) >= grace {
                        released.push(ext);
                    } else {
                        keep.push_back(ext);
                    }
                }
                *q = keep;
            } else {
                while let Some(front) = q.front() {
                    if now.duration_since(front.released_at) >= grace {
                        released.push(q.pop_front().unwrap());
                    } else {
                        break;
                    }
                }
            }
        }
        if released.is_empty() {
            return 0;
        }
        let mut pools = self.inner.free_pools.lock().unwrap();
        let mut alloc = self.inner.local_alloc.lock().unwrap();
        for ext in &released {
            Self::free_extent_locked(&mut pools, &mut alloc, &ext.device, ext.offset, ext.length);
        }
        released.len()
    }

    fn try_allocate(&self, device: &str, aligned_len: u64, initial_offset: u64) -> Result<u64> {
        let mut pools = self.inner.free_pools.lock().unwrap();
        let pool = pools.entry(device.to_string()).or_default();
        if let Some(off) = pool.take_first_fit(aligned_len) {
            return Ok(off);
        }
        let cap_limit = pool.capacity_bytes;

        let mut alloc = self.inner.local_alloc.lock().unwrap();
        let current = alloc.entry(device.to_string()).or_insert(initial_offset);
        let aligned_start = Self::align_up_4k((*current).max(initial_offset));

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

    /// Allocates the next 4K-aligned block offset on a physical NVMe/SSD drive.
    pub async fn allocate_chunk_offset(&self, device: &str, length: u64, initial_offset: u64) -> Result<u64> {
        self.allocate_chunk_offset_for(None, device, length, initial_offset)
            .await
    }

    /// Allocates a block offset, reusing expired quarantined extents first. When the device
    /// is full, the oldest clean, unpinned objects with data on that device are evicted
    /// (never `exclude_uri`, which is the object currently being written).
    pub async fn allocate_chunk_offset_for(
        &self,
        exclude_uri: Option<&str>,
        device: &str,
        length: u64,
        initial_offset: u64,
    ) -> Result<u64> {
        let aligned_len = Self::align_up_4k(length.max(4096));
        self.drain_quarantine(None);

        let mut last_err = match self.try_allocate(device, aligned_len, initial_offset) {
            Ok(off) => return Ok(off),
            Err(e) => e,
        };

        for _ in 0..MAX_PRESSURE_EVICTIONS {
            let victim = {
                let mirror = self.inner.entries_mirror.lock().unwrap();
                mirror
                    .iter()
                    .filter(|(uri, m)| {
                        Some(uri.as_str()) != exclude_uri
                            && !m.pinned
                            && !m.is_dirty()
                            && m.chunk_layout
                                .iter()
                                .any(|c| c.tier != StorageTier::Ram && c.device == device)
                    })
                    .min_by_key(|(_, m)| (m.created_at, m.generation))
                    .map(|(uri, m)| (uri.clone(), m.generation))
            };

            match victim {
                Some((uri, generation)) => {
                    let removed = {
                        let mut mirror = self.inner.entries_mirror.lock().unwrap();
                        match mirror.get(&uri) {
                            Some(m) if m.generation == generation => mirror.remove(&uri),
                            _ => None,
                        }
                    };
                    if let Some(meta) = removed {
                        warn!(
                            "Capacity pressure on '{}': evicting '{}' ({:.2} MiB) to make room",
                            device,
                            uri,
                            meta.object_size as f64 / (1024.0 * 1024.0)
                        );
                        self.cache.invalidate(&uri).await;
                        if let Some(ref m) = self.inner.metrics {
                            m.evictions_total.fetch_add(1, Ordering::Relaxed);
                            m.evicted_bytes_total.fetch_add(meta.object_size, Ordering::Relaxed);
                        }
                        // Under pressure the evicted extents are released without a grace
                        // period; the oldest object is the least likely to have readers.
                        self.release_unpublished(&meta.chunk_layout).await;
                        self.update_metrics_counters();
                        self.mark_dirty();
                    }
                }
                None => {
                    if self.drain_quarantine(Some(device)) == 0 {
                        break;
                    }
                    warn!(
                        "Capacity pressure on '{}': releasing quarantined extents before grace period expiry",
                        device
                    );
                }
            }

            match self.try_allocate(device, aligned_len, initial_offset) {
                Ok(off) => return Ok(off),
                Err(e) => last_err = e,
            }
        }

        Err(last_err)
    }

    /// Atomically persists the metadata manifest and allocation pointers to disk
    pub async fn flush_to_disk(&self) -> Result<()> {
        Self::write_manifest(&self.inner).await
    }

    fn snapshot(inner: &Inner) -> MetadataManifest {
        let entries = inner.entries_mirror.lock().unwrap().clone();
        let mut free_pools = inner.free_pools.lock().unwrap().clone();
        let local_alloc = inner.local_alloc.lock().unwrap().clone();
        // Quarantined extents are persisted as free: after a daemon restart no reader can
        // still be holding a layout resolved by the previous process.
        for ext in inner.quarantine.lock().unwrap().iter() {
            free_pools
                .entry(ext.device.clone())
                .or_default()
                .insert_extent(ext.offset, ext.length);
        }
        MetadataManifest {
            entries,
            local_alloc,
            free_pools,
            updated_at: chrono::Utc::now().timestamp(),
        }
    }

    /// Serialized atomic writer: unique tmp file -> fsync -> rename -> fsync(dir).
    /// Also writes a companion bincode index (.bin) when the primary path is JSON.
    async fn write_manifest(inner: &Arc<Inner>) -> Result<()> {
        let Some(ref persister) = inner.persister else {
            return Ok(());
        };
        let _guard = persister.write_lock.lock().await;
        let manifest = Self::snapshot(inner);
        let path = persister.path.clone();
        tokio::task::spawn_blocking(move || Self::write_manifest_blocking(&manifest, &path))
            .await
            .context("Manifest writer task panicked")?
    }

    fn write_manifest_blocking(manifest: &MetadataManifest, path: &Path) -> Result<()> {
        if let Some(parent) = path.parent() {
            let _ = std::fs::create_dir_all(parent);
        }

        let is_binary = path.extension().is_some_and(|ext| ext == "bin" || ext == "bincode");
        let payload_bytes = if is_binary {
            bincode::serialize(manifest).context("Failed to bincode-serialize metadata manifest")?
        } else {
            serde_json::to_vec(manifest).context("Failed to JSON-serialize metadata manifest")?
        };
        atomic_write(path, &payload_bytes)?;

        if !is_binary {
            match bincode::serialize(manifest) {
                Ok(bin_bytes) => {
                    if let Err(e) = atomic_write(&path.with_extension("bin"), &bin_bytes) {
                        warn!("Failed to write companion binary manifest: {}", e);
                    }
                }
                Err(e) => warn!("Failed to bincode-serialize companion manifest: {}", e),
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

        let bin_path = path.with_extension("bin");
        let mut loaded_manifest: Option<MetadataManifest> = None;

        if path.extension().is_some_and(|e| e == "bin" || e == "bincode") {
            if let Ok(bytes) = std::fs::read(path) {
                if let Ok(m) = bincode::deserialize::<MetadataManifest>(&bytes) {
                    loaded_manifest = Some(m);
                }
            }
        } else {
            // Primary is JSON: JSON is authoritative. For very large manifests (>10 MiB) the companion
            // .bin is used when it is at least as fresh; a .bin from an older release that no longer
            // decodes falls back to JSON transparently.
            let use_bin_fastpath =
                if let (Ok(j_meta), Ok(b_meta)) = (std::fs::metadata(path), std::fs::metadata(&bin_path)) {
                    j_meta.len() > 10 * 1024 * 1024 && b_meta.modified().ok() >= j_meta.modified().ok()
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
                            warn!("Corrupted metadata manifest at {:?}: {}. Starting cold.", path, e);
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

        let store = Self::new(max_capacity_bytes, metrics.clone(), Some(path.to_path_buf()));

        let mut total_bytes: u64 = 0;
        let count = manifest.entries.len();
        let mut max_generation = 0u64;

        // Entries from releases without generations get unique ones assigned on load
        let mut entries: Vec<(String, StoredMetadata)> = manifest.entries.into_iter().collect();
        for (_, meta) in entries.iter() {
            max_generation = max_generation.max(meta.generation);
        }
        store.inner.generation.store(max_generation + 1, Ordering::Relaxed);
        for (_, meta) in entries.iter_mut() {
            if meta.generation == 0 {
                meta.generation = store.next_generation();
            }
        }

        {
            let mut mirror = store.inner.entries_mirror.lock().unwrap();
            for (uri, meta) in entries.iter() {
                total_bytes += meta.object_size;
                mirror.insert(uri.clone(), meta.clone());
            }
        }
        for (uri, meta) in entries {
            store.cache.insert(uri, meta).await;
        }

        *store.inner.local_alloc.lock().unwrap() = manifest.local_alloc;
        {
            let mut pools = manifest.free_pools;
            for pool in pools.values_mut() {
                pool.normalize();
            }
            *store.inner.free_pools.lock().unwrap() = pools;
        }

        if let Some(ref m) = metrics {
            m.cached_objects_count.store(count as u64, Ordering::Relaxed);
            m.storage_used_bytes.store(total_bytes, Ordering::Relaxed);
        }

        Ok(Some((store, count, total_bytes)))
    }

    /// Returns a list of all active cached objects for `hypercache ls`
    pub async fn list_entries(&self) -> (Vec<crate::protocol::CachedObjectSummary>, u64) {
        let mirror = self.inner.entries_mirror.lock().unwrap();
        let mut list = Vec::with_capacity(mirror.len());
        let mut total_bytes: u64 = 0;

        for (uri, meta) in mirror.iter() {
            total_bytes += meta.object_size;
            let mut devices: Vec<String> = meta
                .chunk_layout
                .iter()
                .map(|c| c.device.clone())
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            devices.sort();

            let tier = meta.chunk_layout.first().map_or(StorageTier::Nvme, |c| c.tier);

            list.push(crate::protocol::CachedObjectSummary {
                uri: uri.clone(),
                object_size: meta.object_size,
                chunk_count: meta.chunk_layout.len(),
                created_at: meta.created_at,
                devices,
                tier,
                sync_status: meta.sync_status,
                pinned: meta.pinned,
            });
        }

        list.sort_by_key(|a| std::cmp::Reverse(a.created_at));
        (list, total_bytes)
    }

    /// Evicts a specific cached item and quarantines its blocks
    pub async fn evict_entry(&self, uri: &str) -> Result<Option<u64>> {
        let removed_meta = self.inner.entries_mirror.lock().unwrap().remove(uri);

        if let Some(meta) = removed_meta {
            self.cache.invalidate(uri).await;
            self.quarantine_chunks(&meta.chunk_layout);

            if let Some(ref m) = self.inner.metrics {
                m.evictions_total.fetch_add(1, Ordering::Relaxed);
                m.evicted_bytes_total.fetch_add(meta.object_size, Ordering::Relaxed);
            }
            self.update_metrics_counters();
            self.flush_to_disk().await?;
            Ok(Some(meta.object_size))
        } else {
            Ok(None)
        }
    }

    /// Purges all cached items and resets allocation. Allocation restarts at each caller's
    /// configured `device_start_offset_bytes` (the frontier map is cleared, not reset to a constant).
    pub async fn purge_all(&self) -> Result<(usize, u64)> {
        let (count, total_bytes, all_chunks) = {
            let mut mirror = self.inner.entries_mirror.lock().unwrap();
            let count = mirror.len();
            let total_bytes: u64 = mirror.values().map(|m| m.object_size).sum();
            let chunks: Vec<ChunkDescriptor> = mirror.values().flat_map(|m| m.chunk_layout.iter().cloned()).collect();
            mirror.clear();
            (count, total_bytes, chunks)
        };

        let maybe_pool = self.inner.ram_pool.read().unwrap().clone();
        if let Some(ref pool) = maybe_pool {
            let _ = pool.cleanup_orphaned_chunks();
        } else {
            Self::delete_ram_chunks(&self.inner, &all_chunks);
        }

        self.cache.invalidate_all();

        self.inner.local_alloc.lock().unwrap().clear();
        for pool in self.inner.free_pools.lock().unwrap().values_mut() {
            pool.free_extents.clear();
        }
        self.inner.quarantine.lock().unwrap().clear();

        if let Some(ref m) = self.inner.metrics {
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

    #[cfg(test)]
    pub(crate) fn free_pool_snapshot(&self, device: &str) -> DeviceFreePool {
        self.inner
            .free_pools
            .lock()
            .unwrap()
            .get(device)
            .cloned()
            .unwrap_or_default()
    }

    #[cfg(test)]
    pub(crate) async fn run_pending_cache_tasks(&self) {
        self.cache.run_pending_tasks().await;
    }
}

/// Writes `bytes` to `path` atomically: unique temp file in the same directory, fsync, rename,
/// then fsync of the directory so the rename itself is durable.
pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    static TMP_COUNTER: AtomicU64 = AtomicU64::new(0);
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().to_string())
        .unwrap_or_else(|| "manifest".to_string());
    let tmp_name = format!(
        ".{}.{}.{}.tmp",
        file_name,
        std::process::id(),
        TMP_COUNTER.fetch_add(1, Ordering::Relaxed)
    );
    let tmp_path = path.with_file_name(tmp_name);

    let result = (|| -> Result<()> {
        let mut file = std::fs::File::create(&tmp_path)
            .with_context(|| format!("Failed to create temporary file at {:?}", tmp_path))?;
        file.write_all(bytes)?;
        file.sync_all()?;
        drop(file);
        std::fs::rename(&tmp_path, path)
            .with_context(|| format!("Failed to atomically rename {:?} to {:?}", tmp_path, path))?;
        if let Some(parent) = path.parent() {
            if let Ok(dir) = std::fs::File::open(parent) {
                let _ = dir.sync_all();
            }
        }
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&tmp_path);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::StorageTier;

    fn nvme(device: &str, offset: u64, length: u64) -> ChunkDescriptor {
        ChunkDescriptor::new(StorageTier::Nvme, device, offset, length)
    }

    fn temp_manifest_path(tag: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!(
            "hypercache_meta_{}_{}",
            tag,
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir.join("metadata.json")
    }

    fn test_device(path: &str, size_bytes: u64) -> crate::discovery::DiscoveredDevice {
        crate::discovery::DiscoveredDevice {
            path: path.to_string(),
            is_nvme: true,
            is_virtual: false,
            is_partition: false,
            is_rotational: false,
            size_bytes,
            model: None,
            existing_fs: None,
            numa_node: Some(0),
        }
    }

    #[tokio::test]
    async fn test_moka_metadata_hit_miss() {
        let store = MetadataStore::new(10 * 1024 * 1024 * 1024, None, None);

        let uri = "s3://models/llama3/weights.safetensors";
        assert!(store.get_layout(uri).await.unwrap().is_none());

        let chunks = vec![nvme("/dev/nvme0n1", 4096, 4194304)];
        let generation = store.save_layout(uri, 4194304, 4194304, chunks.clone()).await.unwrap();

        let meta = store.get_layout(uri).await.unwrap().unwrap();
        assert_eq!(meta.object_size, 4194304);
        assert_eq!(meta.chunk_layout, chunks);
        assert_eq!(meta.generation, generation);

        // Re-saving the same URI bumps the generation
        let generation2 = store.save_layout(uri, 4194304, 4194304, chunks).await.unwrap();
        assert!(generation2 > generation);
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
        store.set_reclaim_grace(Duration::ZERO);
        store
            .register_device_capacities(&[test_device("/dev/nvme0n1", 4096 + 2 * 4194304)])
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

        // 3rd allocation must fail: capacity reached and no evictable objects exist
        assert!(store
            .allocate_chunk_offset("/dev/nvme0n1", 4194304, 4096)
            .await
            .is_err());

        store
            .save_layout(
                "hypercache://models/chunk1.bin",
                4194304,
                4194304,
                vec![nvme("/dev/nvme0n1", off1, 4194304)],
            )
            .await
            .unwrap();
        store.evict_entry("hypercache://models/chunk1.bin").await.unwrap();

        // With a zero grace period the evicted extent is reused immediately
        let off_reused = store
            .allocate_chunk_offset("/dev/nvme0n1", 4194304, 4096)
            .await
            .unwrap();
        assert_eq!(off_reused, 4096);
    }

    #[tokio::test]
    async fn test_evicted_extents_quarantined_during_grace_period() {
        let store = MetadataStore::new(100 * 1024 * 1024, None, None);
        let uri = "hypercache://models/being_read.bin";

        let off = store.allocate_chunk_offset("/dev/q0", 4194304, 4096).await.unwrap();
        store
            .save_layout(uri, 4194304, 4194304, vec![nvme("/dev/q0", off, 4194304)])
            .await
            .unwrap();
        store.evict_entry(uri).await.unwrap();

        // A reader that resolved the old layout may still be reading: the extent must not be reused yet
        let next = store.allocate_chunk_offset("/dev/q0", 4194304, 4096).await.unwrap();
        assert_ne!(
            next, off,
            "quarantined extent was reused before the grace period expired"
        );

        // After the grace period the extent becomes reusable
        store.set_reclaim_grace(Duration::ZERO);
        let reused = store.allocate_chunk_offset("/dev/q0", 4194304, 4096).await.unwrap();
        assert_eq!(reused, off);
    }

    #[tokio::test]
    async fn test_replacing_layout_quarantines_old_extents() {
        let store = MetadataStore::new(100 * 1024 * 1024, None, None);
        let uri = "hypercache://models/reingested.bin";
        let off_a = store.allocate_chunk_offset("/dev/r0", 4096, 4096).await.unwrap();
        store
            .save_layout(uri, 4096, 4194304, vec![nvme("/dev/r0", off_a, 4096)])
            .await
            .unwrap();
        let off_b = store.allocate_chunk_offset("/dev/r0", 4096, 4096).await.unwrap();
        store
            .save_layout(uri, 4096, 4194304, vec![nvme("/dev/r0", off_b, 4096)])
            .await
            .unwrap();

        // Old layout's extent is quarantined, not handed out again immediately
        let off_c = store.allocate_chunk_offset("/dev/r0", 4096, 4096).await.unwrap();
        assert_ne!(off_c, off_a);
        assert_ne!(off_c, off_b);
    }

    #[test]
    fn test_free_pool_coalescing() {
        let mut pool = DeviceFreePool::default();
        pool.insert_extent(8192, 4096);
        pool.insert_extent(0, 4096);
        pool.insert_extent(4096, 4096);
        assert_eq!(pool.free_extents, vec![(0, 12288)]);

        pool.insert_extent(20480, 4096);
        assert_eq!(pool.free_extents, vec![(0, 12288), (20480, 4096)]);

        // Fragmented 4 KiB holes coalesce so a large allocation can be satisfied
        pool.insert_extent(12288, 8192);
        assert_eq!(pool.free_extents, vec![(0, 24576)]);
        assert_eq!(pool.take_first_fit(16384), Some(0));
        assert_eq!(pool.free_extents, vec![(16384, 8192)]);

        // Legacy unsorted / duplicated extents are normalized
        let mut legacy = DeviceFreePool {
            free_extents: vec![(8192, 4096), (0, 4096), (0, 4096), (4096, 4096)],
            capacity_bytes: 0,
        };
        legacy.normalize();
        assert_eq!(legacy.free_extents, vec![(0, 12288)]);
    }

    #[tokio::test]
    async fn test_freed_tail_extent_pulls_back_frontier() {
        let store = MetadataStore::new(100 * 1024 * 1024, None, None);
        let a = store.allocate_chunk_offset("/dev/t0", 4096, 4096).await.unwrap();
        let b = store.allocate_chunk_offset("/dev/t0", 4096, 4096).await.unwrap();
        store
            .release_unpublished(&[nvme("/dev/t0", b, 4096), nvme("/dev/t0", a, 4096)])
            .await;
        assert!(store.free_pool_snapshot("/dev/t0").free_extents.is_empty());
        assert_eq!(store.allocate_chunk_offset("/dev/t0", 4096, 4096).await.unwrap(), a);
    }

    #[tokio::test]
    async fn test_capacity_pressure_evicts_oldest_clean_object() {
        let store = MetadataStore::new(100 * 1024 * 1024, None, None);
        store
            .register_device_capacities(&[test_device("/dev/p0", 4096 + 2 * 4194304)])
            .await;

        let off1 = store.allocate_chunk_offset("/dev/p0", 4194304, 4096).await.unwrap();
        store
            .save_layout(
                "hypercache://old.bin",
                4194304,
                4194304,
                vec![nvme("/dev/p0", off1, 4194304)],
            )
            .await
            .unwrap();
        let off2 = store.allocate_chunk_offset("/dev/p0", 4194304, 4096).await.unwrap();
        store
            .save_layout_full(
                "hypercache://pinned.bin",
                4194304,
                4194304,
                vec![nvme("/dev/p0", off2, 4194304)],
                None,
                true,
            )
            .await
            .unwrap();

        // Device is full: allocating must evict the clean, unpinned object and reuse its blocks
        let off3 = store
            .allocate_chunk_offset_for(Some("hypercache://new.bin"), "/dev/p0", 4194304, 4096)
            .await
            .unwrap();
        assert_eq!(off3, off1);
        assert!(store.get_layout("hypercache://old.bin").await.unwrap().is_none());
        assert!(store.get_layout("hypercache://pinned.bin").await.unwrap().is_some());

        // Nothing evictable remains (pinned only): allocation fails with a clear error
        let err = store
            .allocate_chunk_offset_for(Some("hypercache://new.bin"), "/dev/p0", 4194304, 4096)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("capacity exhausted"));
    }

    #[tokio::test]
    async fn test_metadata_persistence_roundtrip() {
        let path = temp_manifest_path("roundtrip");
        let store = MetadataStore::new(10 * 1024 * 1024 * 1024, None, Some(path.clone()));

        let uri1 = "hypercache://models/llama-3/weights.safetensors";
        let chunks1 = vec![
            nvme("/dev/vdc", 4096, 4194304).with_checksum(b"abc"),
            nvme("/dev/vdd", 4096, 4194304),
        ];
        let generation = store
            .save_layout(uri1, 8388608, 4194304, chunks1.clone())
            .await
            .unwrap();
        store.allocate_chunk_offset("/dev/vdc", 4194304, 4096).await.unwrap();
        store.allocate_chunk_offset("/dev/vdd", 4194304, 4096).await.unwrap();

        store.flush_to_disk().await.unwrap();
        drop(store);

        let (restored_store, count, total_bytes) = MetadataStore::load_from_disk(&path, 10 * 1024 * 1024 * 1024, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(total_bytes, 8388608);

        let meta = restored_store.get_layout(uri1).await.unwrap().unwrap();
        assert_eq!(meta.object_size, 8388608);
        assert_eq!(meta.chunk_layout, chunks1);
        assert_eq!(meta.generation, generation);
        assert!(restored_store.next_generation() > generation);

        let next_off = restored_store
            .allocate_chunk_offset("/dev/vdc", 4194304, 4096)
            .await
            .unwrap();
        assert_eq!(next_off, 4096 + 4194304);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[tokio::test]
    async fn test_loads_manifest_from_previous_release() {
        // Manifest as written by v0.9.0: string sync_status, no generation, no checksums,
        // unsorted free extents
        let path = temp_manifest_path("legacy");
        let legacy = r#"{
            "entries": {
                "hypercache://ckpt/a.pt": {
                    "object_size": 4096, "chunk_size": 4194304,
                    "chunk_layout": [{"tier": "nvme", "device": "/dev/vdc", "offset": 4096, "length": 4096}],
                    "created_at": 1700000000, "sync_status": "COMMITTED", "pinned": false
                },
                "hypercache://models/b.bin": {
                    "object_size": 4096, "chunk_size": 4194304,
                    "chunk_layout": [{"device": "/dev/vdc", "offset": 8192, "length": 4096}],
                    "created_at": 1700000001
                }
            },
            "local_alloc": {"/dev/vdc": 20480},
            "free_pools": {"/dev/vdc": {"free_extents": [[16384, 4096], [12288, 4096]], "capacity_bytes": 0}},
            "updated_at": 1700000002
        }"#;
        std::fs::write(&path, legacy).unwrap();

        let (store, count, _) = MetadataStore::load_from_disk(&path, 1 << 30, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count, 2);
        let a = store.get_layout("hypercache://ckpt/a.pt").await.unwrap().unwrap();
        assert_eq!(a.sync_status, Some(SyncStatus::Committed));
        let b = store.get_layout("hypercache://models/b.bin").await.unwrap().unwrap();
        assert_ne!(a.generation, 0);
        assert_ne!(b.generation, 0);
        assert_ne!(a.generation, b.generation);

        // Committed-but-unsynced object is reported for write-back recovery
        let dirty = store.list_dirty_entries().await;
        assert_eq!(dirty.len(), 1);
        assert_eq!(dirty[0].0, "hypercache://ckpt/a.pt");

        // Legacy extents [12288..20480) adjoin the frontier and were coalesced
        assert_eq!(store.free_pool_snapshot("/dev/vdc").free_extents, vec![(12288, 8192)]);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn test_concurrent_mutations_produce_consistent_manifest() {
        let path = temp_manifest_path("concurrent");
        let store = MetadataStore::new(1 << 40, None, Some(path.clone()));

        let mut handles = Vec::new();
        for i in 0..200u64 {
            let s = store.clone();
            handles.push(tokio::spawn(async move {
                let uri = format!("hypercache://shards/{:04}.bin", i);
                s.save_layout(
                    &uri,
                    4096,
                    4194304,
                    vec![ChunkDescriptor::new(StorageTier::Nvme, "/dev/c0", 4096 * (i + 1), 4096)],
                )
                .await
                .unwrap();
            }));
        }
        for h in handles {
            h.await.unwrap();
        }
        store.flush_to_disk().await.unwrap();

        // Let any in-flight background write land, then verify the file is complete and valid
        tokio::time::sleep(Duration::from_millis(100)).await;
        let content = std::fs::read_to_string(&path).unwrap();
        let manifest: MetadataManifest = serde_json::from_str(&content).unwrap();
        assert_eq!(manifest.entries.len(), 200);

        // No temp files are left behind once the writer is idle (hold its lock while scanning)
        let persister = store.inner.persister.as_ref().unwrap();
        let _idle = persister.write_lock.lock().await;
        let leftovers: Vec<_> = std::fs::read_dir(path.parent().unwrap())
            .unwrap()
            .flatten()
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.ends_with(".tmp"))
            .collect();
        assert!(leftovers.is_empty(), "leftover temp files: {:?}", leftovers);

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[tokio::test]
    async fn test_background_writer_persists_without_explicit_flush() {
        let path = temp_manifest_path("background");
        let store = MetadataStore::new(1 << 30, None, Some(path.clone()));
        store
            .save_layout("hypercache://bg.bin", 4096, 4194304, vec![nvme("/dev/b0", 4096, 4096)])
            .await
            .unwrap();

        let mut found = false;
        for _ in 0..100 {
            tokio::time::sleep(Duration::from_millis(20)).await;
            if let Ok(content) = std::fs::read_to_string(&path) {
                if let Ok(m) = serde_json::from_str::<MetadataManifest>(&content) {
                    if m.entries.contains_key("hypercache://bg.bin") {
                        found = true;
                        break;
                    }
                }
            }
        }
        assert!(found, "background writer never persisted the manifest");
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[tokio::test]
    async fn test_atomic_manifest_recovery_after_crash() {
        let path = temp_manifest_path("corrupt");
        std::fs::write(&path, b"{ corrupted_invalid_json_data").unwrap();

        let loaded = MetadataStore::load_from_disk(&path, 10 * 1024 * 1024 * 1024, None)
            .await
            .unwrap();
        assert!(loaded.is_none());

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[tokio::test]
    async fn test_list_evict_and_purge() {
        let store = MetadataStore::new(10 * 1024 * 1024 * 1024, None, None);

        let uri1 = "hypercache://models/llama-3/weights.safetensors";
        let uri2 = "hypercache://models/mistral/weights.safetensors";

        store
            .save_layout(uri1, 4194304, 4194304, vec![nvme("/dev/vdc", 4096, 4194304)])
            .await
            .unwrap();
        store
            .save_layout(uri2, 8388608, 4194304, vec![nvme("/dev/vdd", 4096, 8388608)])
            .await
            .unwrap();

        let (list, total) = store.list_entries().await;
        assert_eq!(list.len(), 2);
        assert_eq!(total, 12582912);

        let freed = store.evict_entry(uri1).await.unwrap();
        assert_eq!(freed, Some(4194304));

        let (list2, total2) = store.list_entries().await;
        assert_eq!(list2.len(), 1);
        assert_eq!(total2, 8388608);

        let (purged_count, purged_freed) = store.purge_all().await.unwrap();
        assert_eq!(purged_count, 1);
        assert_eq!(purged_freed, 8388608);

        let (list3, total3) = store.list_entries().await;
        assert_eq!(list3.len(), 0);
        assert_eq!(total3, 0);
    }

    #[tokio::test]
    async fn test_purge_honors_configured_start_offset() {
        let store = MetadataStore::new(1 << 30, None, None);
        let custom_start = 1024 * 1024; // e.g. protecting a 1 MiB device header
        let first = store
            .allocate_chunk_offset("/dev/h0", 4096, custom_start)
            .await
            .unwrap();
        assert_eq!(first, custom_start);
        store
            .allocate_chunk_offset("/dev/h0", 4096, custom_start)
            .await
            .unwrap();

        store.purge_all().await.unwrap();

        // After purge allocation must restart at the configured offset, never below it
        let after = store
            .allocate_chunk_offset("/dev/h0", 4096, custom_start)
            .await
            .unwrap();
        assert_eq!(after, custom_start);
    }

    #[tokio::test]
    async fn test_pinned_and_dirty_eviction_protection() {
        // Max capacity = 12 MiB (3 x 4 MiB)
        let store = MetadataStore::new(12 * 1024 * 1024, None, None);

        let pinned_uri = "hypercache://models/base/llama3_pinned.safetensors";
        let dirty_uri = "hypercache://checkpoints/dirty_epoch_1.pt";
        let failed_uri = "hypercache://checkpoints/sync_failed_epoch_2.pt";

        store
            .save_layout_full(
                pinned_uri,
                8 * 1024 * 1024,
                4 * 1024 * 1024,
                vec![nvme("/dev/vdc", 4096, 8 * 1024 * 1024)],
                None,
                true,
            )
            .await
            .unwrap();
        store
            .save_layout_full(
                dirty_uri,
                8 * 1024 * 1024,
                4 * 1024 * 1024,
                vec![nvme("/dev/vdd", 4096, 8 * 1024 * 1024)],
                Some(SyncStatus::Staged),
                false,
            )
            .await
            .unwrap();
        // A checkpoint whose backend upload failed has not reached S3: it must survive pressure too
        store
            .save_layout_full(
                failed_uri,
                8 * 1024 * 1024,
                4 * 1024 * 1024,
                vec![nvme("/dev/vde", 4096, 8 * 1024 * 1024)],
                Some(SyncStatus::SyncFailed),
                false,
            )
            .await
            .unwrap();

        for i in 0..10 {
            let u = format!("hypercache://ephemeral/item_{}.bin", i);
            store
                .save_layout(
                    &u,
                    4 * 1024 * 1024,
                    4 * 1024 * 1024,
                    vec![nvme("/dev/vdc", 100_000 + (i as u64) * 4194304, 4 * 1024 * 1024)],
                )
                .await
                .unwrap();
        }
        store.run_pending_cache_tasks().await;

        let pinned_lookup = store.get_layout(pinned_uri).await.unwrap();
        assert!(pinned_lookup.is_some());
        assert!(pinned_lookup.unwrap().pinned);
        assert!(store.get_layout(dirty_uri).await.unwrap().is_some());
        assert!(store.get_layout(failed_uri).await.unwrap().is_some());
    }

    #[tokio::test]
    async fn test_sync_status_generation_guard() {
        let store = MetadataStore::new(1 << 30, None, None);
        let uri = "hypercache://ckpt/epoch.pt";
        let g1 = store
            .save_layout_with_status(
                uri,
                4096,
                4194304,
                vec![nvme("/dev/s0", 4096, 4096)],
                Some(SyncStatus::Committed),
            )
            .await
            .unwrap();
        // Object is re-staged by a newer checkpoint write
        let g2 = store
            .save_layout_with_status(
                uri,
                4096,
                4194304,
                vec![nvme("/dev/s0", 8192, 4096)],
                Some(SyncStatus::Staged),
            )
            .await
            .unwrap();

        // A late background flush for the old generation must not mark the new data as synced
        assert!(
            !store
                .update_sync_status_if_generation(uri, g1, SyncStatus::SyncedBackend)
                .await
        );
        assert_eq!(store.get_sync_status(uri).await.unwrap(), Some(SyncStatus::Staged));
        assert!(
            store
                .update_sync_status_if_generation(uri, g2, SyncStatus::Committed)
                .await
        );
        assert_eq!(store.get_sync_status(uri).await.unwrap(), Some(SyncStatus::Committed));
    }
}
