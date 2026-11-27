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
use tracing::{debug, error, info, warn};

use crate::protocol::{ChunkDescriptor, StorageTier, SyncStatus};
use crate::telemetry::MetricsState;

/// Default grace period before freed NVMe extents become reusable. Clients resolve a layout
/// and then read raw device offsets directly, so extents must not be overwritten while a
/// reader that resolved the old layout may still be in flight.
pub const DEFAULT_RECLAIM_GRACE: Duration = Duration::from_secs(30);

/// Upper bound on objects evicted to satisfy a single allocation under capacity pressure.
const MAX_PRESSURE_EVICTIONS: usize = 64;
/// Slack on top of the grace period when an allocation waits for quarantined extents, so an extent
/// quarantined at the start of the wait is certainly reusable before the wait gives up.
const PRESSURE_WAIT_MARGIN: Duration = Duration::from_millis(100);
/// Upper bound for `reclaim_reserve_percent`.
pub const MAX_RECLAIM_RESERVE_PERCENT: u64 = 50;

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
    /// Never holds a `LOCAL_ONLY` entry: those go in `local_only_entries`.
    pub entries: HashMap<String, StoredMetadata>,
    pub local_alloc: HashMap<String, u64>,
    #[serde(default)]
    pub free_pools: HashMap<String, DeviceFreePool>,
    pub updated_at: i64,
    /// Local-only objects, such as KV cache blocks. They are kept out of `entries`, and this field
    /// stays last, so a daemon from before `LOCAL_ONLY` (0.9.4 and older) can still load the
    /// manifest after a rollback: its JSON reader ignores an unknown field, and its bincode reader
    /// ignores trailing bytes. It then keeps every other object and simply doesn't see these.
    #[serde(default)]
    pub local_only_entries: HashMap<String, StoredMetadata>,
}

/// The manifest as 0.9.4 and older wrote it. A bincode manifest has no field names, so one written
/// before `local_only_entries` existed only decodes with this layout.
#[derive(Deserialize)]
struct ManifestWithoutLocalOnly {
    entries: HashMap<String, StoredMetadata>,
    local_alloc: HashMap<String, u64>,
    #[serde(default)]
    free_pools: HashMap<String, DeviceFreePool>,
    updated_at: i64,
}

impl From<ManifestWithoutLocalOnly> for MetadataManifest {
    fn from(old: ManifestWithoutLocalOnly) -> Self {
        MetadataManifest {
            entries: old.entries,
            local_alloc: old.local_alloc,
            free_pools: old.free_pools,
            updated_at: old.updated_at,
            local_only_entries: HashMap::new(),
        }
    }
}

/// Decodes a bincode manifest in the current layout, or in the layout from before
/// `local_only_entries`, so an upgrade never starts cold over a format change.
fn decode_bincode_manifest(bytes: &[u8]) -> Option<MetadataManifest> {
    bincode::deserialize::<MetadataManifest>(bytes).ok().or_else(|| {
        bincode::deserialize::<ManifestWithoutLocalOnly>(bytes)
            .ok()
            .map(Into::into)
    })
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
    /// Share of each device, in percent, kept free or on its way back through the quarantine, so a
    /// full device rarely makes a write wait for the grace period (0 = no reserve).
    reclaim_reserve_percent: AtomicU64,
    /// When each object was last looked up, in Unix milliseconds. Kept in memory only: after a
    /// restart an object counts as last used when it was created. Picks pressure-eviction victims.
    last_access: Mutex<HashMap<String, i64>>,
    /// When each STAGED object was last staged or appended to. In memory only: STAGED entries do
    /// not survive a restart. Used to expire writes that are never committed.
    staged_activity: Mutex<HashMap<String, Instant>>,
    metrics: Option<Arc<MetricsState>>,
    persister: Option<Persister>,
    /// True while `LOCAL_ONLY` commits may skip the device flush: always without a manifest (nothing
    /// survives a restart), otherwise only while the unflushed marker is in place.
    skip_local_only_flush: AtomicBool,
}

/// The marker that says `LOCAL_ONLY` data may not have reached stable storage. It exists while the
/// daemon runs and is removed only after a clean shutdown has flushed every device. A daemon that
/// finds it at startup did not stop cleanly, so it drops every `LOCAL_ONLY` entry instead of trusting
/// data that a power cut may have lost.
pub fn unflushed_marker_path(manifest_path: &Path) -> PathBuf {
    manifest_path.with_extension("local-only-unflushed")
}

fn fsync_parent_dir(path: &Path) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::File::open(parent)
            .and_then(|dir| dir.sync_all())
            .with_context(|| format!("Failed to sync directory {:?}", parent))?;
    }
    Ok(())
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

        let skip_local_only_flush = AtomicBool::new(persistence_path.is_none());
        let inner = Arc::new(Inner {
            entries_mirror: Mutex::new(HashMap::new()),
            local_alloc: Mutex::new(HashMap::new()),
            free_pools: Mutex::new(HashMap::new()),
            quarantine: Mutex::new(VecDeque::new()),
            ram_pool: RwLock::new(None),
            generation: AtomicU64::new(1),
            reclaim_grace_ms: AtomicU64::new(DEFAULT_RECLAIM_GRACE.as_millis() as u64),
            reclaim_reserve_percent: AtomicU64::new(0),
            last_access: Mutex::new(HashMap::new()),
            staged_activity: Mutex::new(HashMap::new()),
            metrics,
            persister: persistence_path.map(|path| Persister {
                path,
                notify: Arc::new(Notify::new()),
                write_lock: tokio::sync::Mutex::new(()),
                writer_started: AtomicBool::new(false),
            }),
            skip_local_only_flush,
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
                            inner.last_access.lock().unwrap().remove(key.as_str());
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

    /// Sets the share of each device kept free ahead of need, in percent (0 = none, at most 50).
    pub fn set_reclaim_reserve_percent(&self, percent: u64) {
        self.inner
            .reclaim_reserve_percent
            .store(percent.min(MAX_RECLAIM_RESERVE_PERCENT), Ordering::Relaxed);
    }

    fn touch(&self, uri: &str) {
        self.inner
            .last_access
            .lock()
            .unwrap()
            .insert(uri.to_string(), chrono::Utc::now().timestamp_millis());
    }

    fn forget_access(&self, uri: &str) {
        self.inner.last_access.lock().unwrap().remove(uri);
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

    /// Startup only, before anything is served: makes the RAM pool match the restored index.
    /// Chunk files no entry references (left by a crash mid-ingest, or by an object the manifest
    /// writer had not persisted yet) are deleted, and used bytes count only referenced chunks.
    /// Without this every crash shrinks the RAM tier until a full purge.
    pub fn reconcile_ram_pool(&self, pool: &crate::ram::RamMemoryPool) {
        if !pool.lock_dir() {
            warn!(
                "Another process holds the RAM pool directory {:?}; not removing unreferenced chunk files. \
                 Give each daemon its own ram_cache_shm_path.",
                pool.get_shm_path()
            );
            return;
        }
        let keep: HashMap<PathBuf, u64> = self
            .inner
            .entries_mirror
            .lock()
            .unwrap()
            .values()
            .flat_map(|m| m.chunk_layout.iter())
            .filter(|c| c.tier == StorageTier::Ram)
            .map(|c| (PathBuf::from(&c.device), c.length))
            .collect();
        let (removed, removed_bytes) = pool.reconcile(&keep);
        if removed > 0 {
            warn!(
                "Removed {} RAM chunk file(s) ({} bytes) in {:?} that no cached object references, \
                 left by an interrupted ingest or an unclean stop",
                removed,
                removed_bytes,
                pool.get_shm_path()
            );
        }
    }

    fn note_staged_activity(&self, uri: &str, staged: bool) {
        let mut activity = self.inner.staged_activity.lock().unwrap();
        if staged {
            activity.insert(uri.to_string(), Instant::now());
        } else {
            activity.remove(uri);
        }
    }

    /// The layout of `uri` for a reader: None while it is STAGED. A staged layout points at chunks
    /// a writer may not have filled yet, so it must never be served as cached data.
    pub async fn get_readable_layout(&self, uri: &str) -> Result<Option<StoredMetadata>> {
        Ok(self
            .get_layout(uri)
            .await?
            .filter(|m| m.sync_status != Some(SyncStatus::Staged)))
    }

    /// Removes STAGED objects with no stage or append activity for `max_idle`: their writer died or
    /// gave up without committing. RAM chunk files are deleted, NVMe extents go through the reclaim
    /// quarantine. Returns the URIs removed.
    pub async fn expire_stale_staged(&self, max_idle: Duration) -> Vec<String> {
        let now = Instant::now();
        let expired: Vec<(String, StoredMetadata)> = {
            let mut mirror = self.inner.entries_mirror.lock().unwrap();
            let mut activity = self.inner.staged_activity.lock().unwrap();
            activity.retain(|uri, _| {
                mirror
                    .get(uri)
                    .is_some_and(|m| m.sync_status == Some(SyncStatus::Staged))
            });
            let stale: Vec<String> = mirror
                .iter()
                .filter(|(_, m)| m.sync_status == Some(SyncStatus::Staged))
                .filter(|(uri, _)| {
                    let since = *activity.entry(uri.to_string()).or_insert(now);
                    now.duration_since(since) >= max_idle
                })
                .map(|(uri, _)| uri.clone())
                .collect();
            stale
                .into_iter()
                .filter_map(|uri| {
                    activity.remove(&uri);
                    mirror.remove(&uri).map(|m| (uri, m))
                })
                .collect()
        };
        for (uri, meta) in &expired {
            self.cache.invalidate(uri).await;
            self.forget_access(uri);
            self.quarantine_chunks(&meta.chunk_layout);
            warn!(
                "Removed staged write '{}' ({} bytes): not committed within {} s of its last stage or append",
                uri,
                meta.object_size,
                max_idle.as_secs()
            );
        }
        if !expired.is_empty() {
            self.update_metrics_counters();
            self.mark_dirty();
        }
        expired.into_iter().map(|(uri, _)| uri).collect()
    }

    /// Runs `expire_stale_staged` in the background (0 disables it).
    pub fn spawn_staged_write_reaper(&self, timeout: Duration) {
        if timeout.is_zero() {
            return;
        }
        let store = self.clone();
        let period = (timeout / 4).clamp(Duration::from_secs(1), Duration::from_secs(60));
        tokio::spawn(async move {
            let mut tick = tokio::time::interval(period);
            tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tick.tick().await;
                store.expire_stale_staged(timeout).await;
            }
        });
    }

    /// Fast-path in-process metadata lookup (zero network/TCP roundtrip)
    pub async fn get_layout(&self, uri: &str) -> Result<Option<StoredMetadata>> {
        let meta = self.cache.get(uri).await;
        if meta.is_some() {
            self.touch(uri);
        }
        Ok(meta)
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
        let mut meta = StoredMetadata {
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
            // A pin sticks when the object is re-ingested or overwritten: only `set_pinned` clears it.
            if mirror.get(uri).is_some_and(|prev| prev.pinned) {
                meta.pinned = true;
            }
            mirror.insert(uri.to_string(), meta.clone())
        };
        self.touch(uri);
        self.note_staged_activity(uri, sync_status == Some(SyncStatus::Staged));
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
        self.note_staged_activity(uri, true);
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
        let mut q = inner.quarantine.lock().unwrap();
        // Taken under the lock so the queue stays in release order, which drain_quarantine relies on.
        let now = Instant::now();
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

    /// Whether a `LOCAL_ONLY` commit may skip the device flush (see `unflushed_marker_path`).
    pub fn skip_local_only_flush(&self) -> bool {
        self.inner.skip_local_only_flush.load(Ordering::SeqCst)
    }

    /// Creates the unflushed marker durably, then lets `LOCAL_ONLY` commits skip the device flush.
    /// Must run before the daemon serves requests. On error the flag stays off and every commit keeps
    /// flushing.
    pub fn arm_unflushed_marker(&self) -> Result<()> {
        let Some(ref persister) = self.inner.persister else {
            return Ok(());
        };
        let marker = unflushed_marker_path(&persister.path);
        if let Some(parent) = marker.parent() {
            std::fs::create_dir_all(parent).with_context(|| format!("Failed to create {:?}", parent))?;
        }
        std::fs::File::create(&marker)
            .and_then(|f| f.sync_all())
            .with_context(|| format!("Failed to create {:?}", marker))?;
        fsync_parent_dir(&marker)?;
        self.inner.skip_local_only_flush.store(true, Ordering::SeqCst);
        Ok(())
    }

    /// First step of a clean shutdown: from here on every `LOCAL_ONLY` commit flushes again, so a
    /// device flush that starts after this call covers every commit that skipped one.
    pub fn stop_skipping_local_only_flush(&self) {
        self.inner.skip_local_only_flush.store(false, Ordering::SeqCst);
    }

    /// Last step of a clean shutdown, after every device was flushed and the manifest written.
    pub fn clear_unflushed_marker(&self) -> Result<()> {
        let Some(ref persister) = self.inner.persister else {
            return Ok(());
        };
        let marker = unflushed_marker_path(&persister.path);
        match std::fs::remove_file(&marker) {
            Ok(()) => fsync_parent_dir(&marker),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(e) => Err(e).with_context(|| format!("Failed to remove {:?}", marker)),
        }
    }

    /// Devices that hold or held NVMe allocations, for the shutdown flush.
    pub fn allocated_devices(&self) -> Vec<String> {
        let mut devices: Vec<String> = self.inner.local_alloc.lock().unwrap().keys().cloned().collect();
        devices.sort();
        devices
    }

    /// Whether any RAM chunk of `meta` has no file, or a file shorter than the chunk.
    fn ram_chunks_missing(meta: &StoredMetadata) -> bool {
        meta.chunk_layout
            .iter()
            .filter(|c| c.tier == StorageTier::Ram)
            .any(|c| std::fs::metadata(&c.device).map_or(true, |m| !m.is_file() || m.len() < c.length))
    }

    fn delete_ram_chunks(inner: &Inner, chunks: &[ChunkDescriptor]) {
        let maybe_pool = inner.ram_pool.read().unwrap().clone();
        for chunk in chunks.iter().filter(|c| c.tier == StorageTier::Ram) {
            if let Some(ref pool) = maybe_pool {
                let _ = pool.delete_chunk(&chunk.device, chunk.length);
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

    /// Moves quarantined extents whose grace period expired into the free pools. Extents are never
    /// released early, even under capacity pressure: readers resolve a layout and then read raw
    /// offsets with no lock, so early reuse would let them read another object's bytes.
    fn drain_quarantine(&self) -> usize {
        let grace = self.reclaim_grace();
        let now = Instant::now();
        let mut released = Vec::new();
        {
            let mut q = self.inner.quarantine.lock().unwrap();
            while let Some(front) = q.front() {
                if now.duration_since(front.released_at) >= grace {
                    released.push(q.pop_front().unwrap());
                } else {
                    break;
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

    /// Allocates a block offset, reusing expired quarantined extents first. When the device is full,
    /// the least recently used clean, unpinned objects with data on that device are evicted (never
    /// `exclude_uri`, which is the object currently being written). Their extents go through the
    /// reclaim quarantine like any other freed extent, so the caller waits until enough of it has
    /// served its grace period. It waits at most one grace period (plus a small margin): anything
    /// freed by then is reusable, so waiting longer can't help, and the allocation fails instead.
    ///
    /// After a successful allocation the device's reserve (`reclaim_reserve_percent`) is topped up
    /// by evicting ahead of need, so later writes find space that has already served its grace.
    pub async fn allocate_chunk_offset_for(
        &self,
        exclude_uri: Option<&str>,
        device: &str,
        length: u64,
        initial_offset: u64,
    ) -> Result<u64> {
        let offset = self
            .allocate_or_wait(exclude_uri, device, length, initial_offset)
            .await?;
        self.top_up_reserve(exclude_uri, device, initial_offset).await;
        Ok(offset)
    }

    async fn allocate_or_wait(
        &self,
        exclude_uri: Option<&str>,
        device: &str,
        length: u64,
        initial_offset: u64,
    ) -> Result<u64> {
        let aligned_len = Self::align_up_4k(length.max(4096));
        self.drain_quarantine();

        let mut last_err = match self.try_allocate(device, aligned_len, initial_offset) {
            Ok(off) => return Ok(off),
            Err(e) => e,
        };

        // A wait costs a whole grace period, so make it count: bring back the full reserve, not
        // just this one chunk.
        let target = aligned_len.max(self.reserve_bytes(device));
        let deadline = Instant::now() + self.reclaim_grace() + PRESSURE_WAIT_MARGIN;
        let mut evictions = 0;
        let mut waited = false;
        loop {
            // Make sure enough space is on its way back on this device before waiting for it.
            while self.quarantined_bytes_on(device) < target && evictions < MAX_PRESSURE_EVICTIONS {
                evictions += 1;
                if self
                    .evict_least_recent_for_pressure(exclude_uri, device)
                    .await
                    .is_none()
                {
                    break;
                }
            }

            let expiry = match self.next_quarantine_expiry_on(device) {
                Some(t) if t <= deadline => t,
                _ => break,
            };
            if !waited {
                waited = true;
                warn!(
                    "Capacity pressure on '{}': waiting up to {:.1} s for freed blocks to finish their reclaim grace period",
                    device,
                    deadline.saturating_duration_since(Instant::now()).as_secs_f64()
                );
            }
            tokio::time::sleep_until(tokio::time::Instant::from_std(expiry)).await;
            self.drain_quarantine();
            match self.try_allocate(device, aligned_len, initial_offset) {
                Ok(off) => return Ok(off),
                Err(e) => last_err = e,
            }
        }

        Err(last_err)
    }

    /// The reserve for `device` in bytes: `reclaim_reserve_percent` of its capacity (0 when the
    /// capacity is unknown).
    fn reserve_bytes(&self, device: &str) -> u64 {
        let percent = self.inner.reclaim_reserve_percent.load(Ordering::Relaxed);
        let capacity = self
            .inner
            .free_pools
            .lock()
            .unwrap()
            .get(device)
            .map_or(0, |p| p.capacity_bytes);
        capacity / 100 * percent
    }

    /// Bytes on `device` that can be allocated now: free extents plus the untouched space past the
    /// allocation frontier.
    fn free_bytes_on(&self, device: &str, initial_offset: u64) -> u64 {
        let pools = self.inner.free_pools.lock().unwrap();
        let Some(pool) = pools.get(device) else {
            return 0;
        };
        let in_extents: u64 = pool.free_extents.iter().map(|(_, len)| *len).sum();
        let frontier = self
            .inner
            .local_alloc
            .lock()
            .unwrap()
            .get(device)
            .copied()
            .unwrap_or(initial_offset);
        let frontier = Self::align_up_4k(frontier.max(initial_offset));
        in_extents + pool.capacity_bytes.saturating_sub(frontier)
    }

    /// Evicts least recently used objects from `device` until its free and quarantined space covers
    /// the reserve. Never waits: evicted space becomes reusable after the grace period.
    async fn top_up_reserve(&self, exclude_uri: Option<&str>, device: &str, initial_offset: u64) {
        let reserve = self.reserve_bytes(device);
        if reserve == 0 {
            return;
        }
        for _ in 0..MAX_PRESSURE_EVICTIONS {
            if self.free_bytes_on(device, initial_offset) + self.quarantined_bytes_on(device) >= reserve {
                return;
            }
            if self
                .evict_least_recent_for_pressure(exclude_uri, device)
                .await
                .is_none()
            {
                return;
            }
        }
    }

    /// Bytes on `device` waiting in the reclaim quarantine.
    fn quarantined_bytes_on(&self, device: &str) -> u64 {
        self.inner
            .quarantine
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.device == device)
            .map(|e| e.length)
            .sum()
    }

    /// When the oldest quarantined extent on `device` becomes reusable.
    fn next_quarantine_expiry_on(&self, device: &str) -> Option<Instant> {
        let grace = self.reclaim_grace();
        self.inner
            .quarantine
            .lock()
            .unwrap()
            .iter()
            .filter(|e| e.device == device)
            .map(|e| e.released_at + grace)
            .min()
    }

    /// Evicts the least recently used clean, unpinned object with data on `device` (by last lookup,
    /// or creation time if it hasn't been looked up since the daemon started) and quarantines its
    /// extents. Returns `None` when no such object exists, and `Some(false)` when the chosen object
    /// changed before it could be removed.
    async fn evict_least_recent_for_pressure(&self, exclude_uri: Option<&str>, device: &str) -> Option<bool> {
        let (uri, generation) = {
            let mirror = self.inner.entries_mirror.lock().unwrap();
            let last_access = self.inner.last_access.lock().unwrap();
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
                .min_by_key(|(uri, m)| {
                    let used = last_access
                        .get(uri.as_str())
                        .copied()
                        .unwrap_or(m.created_at.saturating_mul(1000));
                    (used, m.generation)
                })
                .map(|(uri, m)| (uri.clone(), m.generation))?
        };
        let removed = {
            let mut mirror = self.inner.entries_mirror.lock().unwrap();
            match mirror.get(&uri) {
                Some(m) if m.generation == generation => mirror.remove(&uri),
                _ => None,
            }
        };
        let meta = match removed {
            Some(meta) => meta,
            None => return Some(false),
        };
        info!(
            "Space on '{}': evicting least recently used '{}' ({:.2} MiB)",
            device,
            uri,
            meta.object_size as f64 / (1024.0 * 1024.0)
        );
        self.cache.invalidate(&uri).await;
        self.forget_access(&uri);
        if let Some(ref m) = self.inner.metrics {
            m.evictions_total.fetch_add(1, Ordering::Relaxed);
            m.evicted_bytes_total.fetch_add(meta.object_size, Ordering::Relaxed);
        }
        // A reader may have resolved this layout just before the eviction, so its extents serve
        // the grace period like any other freed extent.
        self.quarantine_chunks(&meta.chunk_layout);
        self.update_metrics_counters();
        self.mark_dirty();
        Some(true)
    }

    /// Atomically persists the metadata manifest and allocation pointers to disk
    pub async fn flush_to_disk(&self) -> Result<()> {
        Self::write_manifest(&self.inner).await
    }

    fn snapshot(inner: &Inner) -> MetadataManifest {
        let (local_only_entries, entries): (HashMap<_, _>, HashMap<_, _>) = inner
            .entries_mirror
            .lock()
            .unwrap()
            .iter()
            .map(|(uri, meta)| (uri.clone(), meta.clone()))
            .partition(|(_, meta)| meta.sync_status == Some(SyncStatus::LocalOnly));
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
            local_only_entries,
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
                if let Some(m) = decode_bincode_manifest(&bytes) {
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
                    if let Some(m) = decode_bincode_manifest(&bytes) {
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

        let mut manifest = match loaded_manifest {
            Some(m) => m,
            None => return Ok(None),
        };

        // The previous daemon did not stop cleanly, so LOCAL_ONLY data may never have reached the
        // drives: drop those entries and return their space. They become misses.
        if unflushed_marker_path(path).exists() {
            let dropped = std::mem::take(&mut manifest.local_only_entries);
            for pool in manifest.free_pools.values_mut() {
                pool.normalize();
            }
            let mut dropped_bytes = 0u64;
            for meta in dropped.values() {
                dropped_bytes += meta.object_size;
                for chunk in &meta.chunk_layout {
                    if chunk.tier == StorageTier::Ram {
                        let _ = std::fs::remove_file(&chunk.device);
                    } else {
                        Self::free_extent_locked(
                            &mut manifest.free_pools,
                            &mut manifest.local_alloc,
                            &chunk.device,
                            chunk.offset,
                            Self::align_up_4k(chunk.length.max(4096)),
                        );
                    }
                }
            }
            if !dropped.is_empty() {
                warn!(
                    "The previous daemon did not stop cleanly: dropped {} local-only object(s) ({} bytes), \
                     such as KV cache blocks, whose data may not have been flushed. They will be misses.",
                    dropped.len(),
                    dropped_bytes
                );
            }
        }

        // RAM-tier files do not survive a reboot (or anything else that empties the shared memory
        // directory). An entry whose RAM chunks are gone would fail every read and never re-ingest,
        // so drop it: it becomes a miss. That includes dirty entries: their data is gone, so a
        // write-back could never succeed. Each lost checkpoint is logged by name. STAGED entries are
        // dropped too: their writer belonged to the previous daemon's lifetime and can't commit now.
        let mut dropped_missing = 0usize;
        let mut dropped_staged = 0usize;
        for map in [&mut manifest.entries, &mut manifest.local_only_entries] {
            let lost: Vec<String> = map
                .iter()
                .filter(|(_, meta)| meta.sync_status == Some(SyncStatus::Staged) || Self::ram_chunks_missing(meta))
                .map(|(uri, _)| uri.clone())
                .collect();
            if !lost.is_empty() && dropped_missing == 0 {
                for pool in manifest.free_pools.values_mut() {
                    pool.normalize();
                }
            }
            for uri in lost {
                let Some(meta) = map.remove(&uri) else { continue };
                dropped_missing += 1;
                if meta.sync_status == Some(SyncStatus::Staged) {
                    dropped_staged += 1;
                    warn!(
                        "Dropped staged write '{}' ({} bytes): it was never committed before the last stop",
                        uri, meta.object_size
                    );
                } else if let Some(status) = meta.sync_status.filter(|s| s.is_dirty()) {
                    error!(
                        "Lost '{}' ({} bytes, status {}): its RAM-tier data was never written to the backend \
                         and its shared memory files are gone (for example after a reboot). It is removed \
                         from the cache and will not be written back.",
                        uri,
                        meta.object_size,
                        status.as_str()
                    );
                }
                debug!("Dropping {} on load: its RAM chunk files are missing or short", uri);
                for chunk in &meta.chunk_layout {
                    if chunk.tier == StorageTier::Ram {
                        let _ = std::fs::remove_file(&chunk.device);
                    } else {
                        // Nothing has been served yet, so the extents need no quarantine
                        Self::free_extent_locked(
                            &mut manifest.free_pools,
                            &mut manifest.local_alloc,
                            &chunk.device,
                            chunk.offset,
                            Self::align_up_4k(chunk.length.max(4096)),
                        );
                    }
                }
            }
        }
        if dropped_missing > dropped_staged {
            warn!(
                "Dropped {} cached object(s) whose RAM-tier files are gone (the shared memory directory \
                 was emptied, for example by a reboot). They will be misses and ingest again.",
                dropped_missing - dropped_staged
            );
        }

        let store = Self::new(max_capacity_bytes, metrics.clone(), Some(path.to_path_buf()));

        let mut total_bytes: u64 = 0;
        let count = manifest.entries.len() + manifest.local_only_entries.len();
        let mut max_generation = 0u64;

        // Entries from releases without generations get unique ones assigned on load
        let mut entries: Vec<(String, StoredMetadata)> = manifest
            .entries
            .into_iter()
            .chain(manifest.local_only_entries)
            .collect();
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
        if dropped_missing > 0 {
            // Rewrite the manifest without the dropped entries
            store.mark_dirty();
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
            self.forget_access(uri);
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
    /// Removes cached objects. Pinned objects are kept unless `include_pinned` is set. Returns the
    /// number of objects removed, their bytes, and the number of pinned objects kept.
    ///
    /// When nothing is kept, the allocator is reset as well (a full wipe). When pinned objects are
    /// kept their data stays where it is, so the removed objects' blocks go through the reclaim
    /// quarantine like any other eviction.
    pub async fn purge_all(&self, include_pinned: bool) -> Result<(usize, u64, usize)> {
        let kept_pinned = if include_pinned {
            0
        } else {
            self.inner
                .entries_mirror
                .lock()
                .unwrap()
                .values()
                .filter(|m| m.pinned)
                .count()
        };
        if kept_pinned > 0 {
            return self.purge_unpinned(kept_pinned).await;
        }

        let (count, total_bytes, all_chunks) = {
            let mut mirror = self.inner.entries_mirror.lock().unwrap();
            let count = mirror.len();
            let total_bytes: u64 = mirror.values().map(|m| m.object_size).sum();
            let chunks: Vec<ChunkDescriptor> = mirror.values().flat_map(|m| m.chunk_layout.iter().cloned()).collect();
            mirror.clear();
            self.inner.last_access.lock().unwrap().clear();
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
        Ok((count, total_bytes, 0))
    }

    async fn purge_unpinned(&self, kept_pinned: usize) -> Result<(usize, u64, usize)> {
        let removed: Vec<(String, StoredMetadata)> = {
            let mut mirror = self.inner.entries_mirror.lock().unwrap();
            let uris: Vec<String> = mirror
                .iter()
                .filter(|(_, m)| !m.pinned)
                .map(|(uri, _)| uri.clone())
                .collect();
            uris.into_iter()
                .filter_map(|uri| mirror.remove(&uri).map(|m| (uri, m)))
                .collect()
        };
        let mut total_bytes = 0u64;
        for (uri, meta) in &removed {
            total_bytes += meta.object_size;
            self.cache.invalidate(uri).await;
            self.forget_access(uri);
            self.quarantine_chunks(&meta.chunk_layout);
        }
        self.update_metrics_counters();
        self.flush_to_disk().await?;
        Ok((removed.len(), total_bytes, kept_pinned))
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
        let grace = Duration::from_millis(300);
        store.set_reclaim_grace(grace);
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

        // Device is full: allocating must evict the clean, unpinned object and reuse its blocks,
        // but only after they have served the grace period (a reader may still hold the layout).
        let started = Instant::now();
        let off3 = store
            .allocate_chunk_offset_for(Some("hypercache://new.bin"), "/dev/p0", 4194304, 4096)
            .await
            .unwrap();
        assert_eq!(off3, off1);
        assert!(
            started.elapsed() >= grace,
            "evicted extent reused after {:?}, before the {:?} grace period",
            started.elapsed(),
            grace
        );
        assert!(store.get_layout("hypercache://old.bin").await.unwrap().is_none());
        assert!(store.get_layout("hypercache://pinned.bin").await.unwrap().is_some());

        // Nothing evictable remains (pinned only): allocation fails with a clear error
        let err = store
            .allocate_chunk_offset_for(Some("hypercache://new.bin"), "/dev/p0", 4194304, 4096)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("capacity exhausted"));
    }

    /// Space that is still in quarantine is never handed out early, even when the device is full
    /// and nothing else can be evicted (issue #10).
    #[tokio::test]
    async fn test_capacity_pressure_waits_for_quarantined_extents() {
        let store = MetadataStore::new(100 * 1024 * 1024, None, None);
        let grace = Duration::from_millis(300);
        store.set_reclaim_grace(grace);
        store
            .register_device_capacities(&[test_device("/dev/w0", 4096 + 4194304)])
            .await;
        let off = store.allocate_chunk_offset("/dev/w0", 4194304, 4096).await.unwrap();
        store
            .save_layout(
                "hypercache://a.bin",
                4194304,
                4194304,
                vec![nvme("/dev/w0", off, 4194304)],
            )
            .await
            .unwrap();
        store.evict_entry("hypercache://a.bin").await.unwrap();

        // The only space is quarantined: the writer waits for it instead of reusing it at once.
        let started = Instant::now();
        let again = store.allocate_chunk_offset("/dev/w0", 4194304, 4096).await.unwrap();
        assert_eq!(again, off);
        assert!(started.elapsed() >= grace, "reused after {:?}", started.elapsed());
    }

    /// When even the space that frees up within the grace period can't fit the request, the
    /// allocation fails after at most about one grace period instead of reusing it early or hanging.
    #[tokio::test]
    async fn test_capacity_pressure_gives_up_after_one_grace_period() {
        let store = MetadataStore::new(100 * 1024 * 1024, None, None);
        let grace = Duration::from_millis(300);
        store.set_reclaim_grace(grace);
        store
            .register_device_capacities(&[test_device("/dev/g0", 4096 + 2 * 4194304)])
            .await;
        let off1 = store.allocate_chunk_offset("/dev/g0", 4194304, 4096).await.unwrap();
        store
            .save_layout(
                "hypercache://old.bin",
                4194304,
                4194304,
                vec![nvme("/dev/g0", off1, 4194304)],
            )
            .await
            .unwrap();
        let off2 = store.allocate_chunk_offset("/dev/g0", 4194304, 4096).await.unwrap();
        store
            .save_layout_full(
                "hypercache://pinned.bin",
                4194304,
                4194304,
                vec![nvme("/dev/g0", off2, 4194304)],
                None,
                true,
            )
            .await
            .unwrap();

        // 8 MiB can't fit next to the pinned object, even after old.bin's 4 MiB comes back.
        let started = Instant::now();
        let err = store
            .allocate_chunk_offset_for(Some("hypercache://big.bin"), "/dev/g0", 2 * 4194304, 4096)
            .await
            .unwrap_err();
        assert!(err.to_string().contains("capacity exhausted"));
        let took = started.elapsed();
        assert!(
            took >= grace,
            "gave up after {:?}, before waiting for the quarantine",
            took
        );
        assert!(took < grace * 3, "waited {:?}, far past the grace period", took);
        assert!(store.get_layout("hypercache://pinned.bin").await.unwrap().is_some());
    }

    /// Pressure eviction picks the object used longest ago, not the one created first.
    #[tokio::test]
    async fn test_capacity_pressure_evicts_least_recently_used() {
        let store = MetadataStore::new(100 * 1024 * 1024, None, None);
        store.set_reclaim_grace(Duration::ZERO);
        store
            .register_device_capacities(&[test_device("/dev/l0", 4096 + 2 * 4194304)])
            .await;
        let off_old = store.allocate_chunk_offset("/dev/l0", 4194304, 4096).await.unwrap();
        store
            .save_layout(
                "hypercache://busy-old.bin",
                4194304,
                4194304,
                vec![nvme("/dev/l0", off_old, 4194304)],
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        let off_new = store.allocate_chunk_offset("/dev/l0", 4194304, 4096).await.unwrap();
        store
            .save_layout(
                "hypercache://idle-new.bin",
                4194304,
                4194304,
                vec![nvme("/dev/l0", off_new, 4194304)],
            )
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(5)).await;
        // The older object is in use: it was looked up after the newer one was written.
        assert!(store.get_layout("hypercache://busy-old.bin").await.unwrap().is_some());

        let off = store
            .allocate_chunk_offset_for(Some("hypercache://x.bin"), "/dev/l0", 4194304, 4096)
            .await
            .unwrap();
        assert_eq!(off, off_new);
        assert!(store.get_layout("hypercache://idle-new.bin").await.unwrap().is_none());
        assert!(store.get_layout("hypercache://busy-old.bin").await.unwrap().is_some());
    }

    /// With a reserve, space is evicted ahead of need, so the write that fills the device returns
    /// at once and the next one finds space that has already served its grace period.
    #[tokio::test]
    async fn test_reserve_evicts_ahead_without_waiting() {
        let store = MetadataStore::new(100 * 1024 * 1024, None, None);
        let grace = Duration::from_millis(300);
        store.set_reclaim_grace(grace);
        store.set_reclaim_reserve_percent(20);
        let cap = 4096 + 4 * 4194304;
        store.register_device_capacities(&[test_device("/dev/v0", cap)]).await;
        for i in 0..3 {
            let off = store.allocate_chunk_offset("/dev/v0", 4194304, 4096).await.unwrap();
            store
                .save_layout(
                    &format!("hypercache://o{}.bin", i),
                    4194304,
                    4194304,
                    vec![nvme("/dev/v0", off, 4194304)],
                )
                .await
                .unwrap();
            tokio::time::sleep(Duration::from_millis(5)).await;
        }
        assert!(store.get_layout("hypercache://o0.bin").await.unwrap().is_some());
        tokio::time::sleep(Duration::from_millis(5)).await;
        assert!(store.get_layout("hypercache://o2.bin").await.unwrap().is_some());

        // This allocation fills the device: it returns at once, and the least recently used object
        // (o1) is evicted into quarantine to restore the reserve.
        let started = Instant::now();
        let off = store
            .allocate_chunk_offset_for(Some("hypercache://o3.bin"), "/dev/v0", 4194304, 4096)
            .await
            .unwrap();
        assert!(started.elapsed() < grace, "the reserve top-up must not wait");
        store
            .save_layout(
                "hypercache://o3.bin",
                4194304,
                4194304,
                vec![nvme("/dev/v0", off, 4194304)],
            )
            .await
            .unwrap();
        assert!(store.get_layout("hypercache://o1.bin").await.unwrap().is_none());
        assert!(store.get_layout("hypercache://o0.bin").await.unwrap().is_some());
        assert!(store.get_layout("hypercache://o2.bin").await.unwrap().is_some());

        // Once that space has served its grace period, the next write gets it without waiting.
        tokio::time::sleep(grace + Duration::from_millis(50)).await;
        let started = Instant::now();
        store
            .allocate_chunk_offset_for(Some("hypercache://o4.bin"), "/dev/v0", 4194304, 4096)
            .await
            .unwrap();
        assert!(started.elapsed() < Duration::from_millis(100));
    }

    #[tokio::test]
    async fn test_reserve_never_evicts_pinned_or_dirty_objects() {
        let store = MetadataStore::new(100 * 1024 * 1024, None, None);
        store.set_reclaim_reserve_percent(50);
        store
            .register_device_capacities(&[test_device("/dev/k0", 4096 + 2 * 4194304)])
            .await;
        let a = store.allocate_chunk_offset("/dev/k0", 4194304, 4096).await.unwrap();
        store
            .save_layout_full(
                "hypercache://pinned.bin",
                4194304,
                4194304,
                vec![nvme("/dev/k0", a, 4194304)],
                None,
                true,
            )
            .await
            .unwrap();
        let b = store.allocate_chunk_offset("/dev/k0", 4194304, 4096).await.unwrap();
        store
            .save_layout_full(
                "file:///ckpt/dirty.pt",
                4194304,
                4194304,
                vec![nvme("/dev/k0", b, 4194304)],
                Some(SyncStatus::Committed),
                false,
            )
            .await
            .unwrap();
        // The device is full and below its reserve, but nothing may be evicted.
        let _ = store.allocate_chunk_offset("/dev/k0", 4096, 4096).await;
        assert!(store.get_layout("hypercache://pinned.bin").await.unwrap().is_some());
        assert!(store.get_layout("file:///ckpt/dirty.pt").await.unwrap().is_some());
    }

    /// Purge keeps pinned objects (and their blocks) unless asked to include them.
    #[tokio::test]
    async fn test_purge_keeps_pinned_unless_included() {
        let store = MetadataStore::new(100 * 1024 * 1024, None, None);
        let p = store.allocate_chunk_offset("/dev/u0", 4194304, 4096).await.unwrap();
        store
            .save_layout_full(
                "hypercache://keep.bin",
                4194304,
                4194304,
                vec![nvme("/dev/u0", p, 4194304)],
                None,
                true,
            )
            .await
            .unwrap();
        let o = store.allocate_chunk_offset("/dev/u0", 4194304, 4096).await.unwrap();
        store
            .save_layout(
                "hypercache://drop.bin",
                4194304,
                4194304,
                vec![nvme("/dev/u0", o, 4194304)],
            )
            .await
            .unwrap();

        let (removed, bytes, kept) = store.purge_all(false).await.unwrap();
        assert_eq!((removed, bytes, kept), (1, 4194304, 1));
        assert!(store.get_layout("hypercache://keep.bin").await.unwrap().is_some());
        assert!(store.get_layout("hypercache://drop.bin").await.unwrap().is_none());
        // The pinned object's blocks are never handed out, and the removed object's blocks wait
        // out the grace period (default 30 s) instead of being reused at once.
        let next = store.allocate_chunk_offset("/dev/u0", 4194304, 4096).await.unwrap();
        assert_ne!(next, p);
        assert_ne!(next, o);

        let (removed, _, kept) = store.purge_all(true).await.unwrap();
        assert_eq!((removed, kept), (1, 0));
        assert!(store.get_layout("hypercache://keep.bin").await.unwrap().is_none());
    }

    /// A pin survives re-ingest and overwrite; only an explicit unpin clears it.
    #[tokio::test]
    async fn test_pin_sticks_across_reingest() {
        let store = MetadataStore::new(100 * 1024 * 1024, None, None);
        let uri = "hypercache://models/keep.bin";
        let a = store.allocate_chunk_offset("/dev/s0", 4096, 4096).await.unwrap();
        store
            .save_layout_full(uri, 4096, 4194304, vec![nvme("/dev/s0", a, 4096)], None, true)
            .await
            .unwrap();
        let b = store.allocate_chunk_offset("/dev/s0", 4096, 4096).await.unwrap();
        store
            .save_layout_full(uri, 4096, 4194304, vec![nvme("/dev/s0", b, 4096)], None, false)
            .await
            .unwrap();
        assert!(
            store.get_layout(uri).await.unwrap().unwrap().pinned,
            "re-ingest dropped the pin"
        );
        assert!(store.set_pinned(uri, false).await.unwrap());
        assert!(!store.get_layout(uri).await.unwrap().unwrap().pinned);
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

        // Normally ~40 ms, but each write syncs to disk, and under a loaded parallel test run one
        // took 3.2 s. This checks the write happens, not how fast, so allow 30 s.
        let mut found = false;
        for _ in 0..1500 {
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

        let (purged_count, purged_freed, kept) = store.purge_all(false).await.unwrap();
        assert_eq!(kept, 0);
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

        store.purge_all(false).await.unwrap();

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

    /// The manifest schema exactly as 0.9.4 reads it: a SyncStatus without LOCAL_ONLY, in the same
    /// variant order (bincode stores the index), and no `local_only_entries`.
    #[allow(dead_code)] // the fields mirror the old layout; only a few are checked
    mod v094 {
        use super::super::{ChunkDescriptor, DeviceFreePool};
        use serde::Deserialize;
        use std::collections::HashMap;

        #[derive(Debug, Deserialize, PartialEq)]
        pub enum SyncStatus {
            #[serde(rename = "STAGED")]
            Staged,
            #[serde(rename = "COMMITTED")]
            Committed,
            #[serde(rename = "SYNCING_BACKEND")]
            SyncingBackend,
            #[serde(rename = "SYNCED_BACKEND")]
            SyncedBackend,
            #[serde(rename = "SYNC_FAILED")]
            SyncFailed,
            #[serde(rename = "REPLICA_COMMITTED")]
            ReplicaCommitted,
        }

        #[derive(Debug, Deserialize)]
        pub struct StoredMetadata {
            pub object_size: u64,
            pub chunk_size: usize,
            pub chunk_layout: Vec<ChunkDescriptor>,
            pub created_at: i64,
            #[serde(default)]
            pub sync_status: Option<SyncStatus>,
            #[serde(default)]
            pub pinned: bool,
            #[serde(default)]
            pub generation: u64,
        }

        #[derive(Debug, Deserialize)]
        pub struct MetadataManifest {
            pub entries: HashMap<String, StoredMetadata>,
            pub local_alloc: HashMap<String, u64>,
            #[serde(default)]
            pub free_pools: HashMap<String, DeviceFreePool>,
            pub updated_at: i64,
        }
    }

    /// A chunk with a recorded CRC. Bincode can't decode a chunk without one, because
    /// `checksum` is skipped when None (a separate, existing issue), so these tests use CRCs.
    fn nvme_crc(device: &str, offset: u64, length: u64) -> ChunkDescriptor {
        let mut chunk = nvme(device, offset, length);
        chunk.checksum = Some(0x1234_5678);
        chunk
    }

    /// A store holding one object waiting for write-back and one local-only KV block.
    async fn store_with_dirty_and_local_only(path: &Path) -> MetadataStore {
        let store = MetadataStore::new(1 << 30, None, Some(path.to_path_buf()));
        store
            .save_layout_with_status(
                "hypercache://checkpoints/unflushed.bin",
                4096,
                4194304,
                vec![nvme_crc("/dev/r0", 4096, 4096)],
                Some(SyncStatus::Committed),
            )
            .await
            .unwrap();
        store
            .save_layout_with_status(
                "hypercache://kv_cache/ns/m/0/0/_/ab.kv",
                8192,
                4194304,
                vec![nvme_crc("/dev/r0", 8192, 8192)],
                Some(SyncStatus::LocalOnly),
            )
            .await
            .unwrap();
        store.flush_to_disk().await.unwrap();
        store
    }

    #[tokio::test]
    async fn test_rollback_to_0_9_4_keeps_unflushed_objects() {
        // A 0.9.4 daemon can't parse LOCAL_ONLY. If one were in `entries`, the whole manifest would
        // fail to load and the daemon would start cold, losing objects not yet flushed.
        let path = temp_manifest_path("rollback");
        let _store = store_with_dirty_and_local_only(&path).await;

        let json = std::fs::read_to_string(&path).unwrap();
        let old: v094::MetadataManifest = serde_json::from_str(&json).expect("0.9.4 must parse the JSON manifest");
        let unflushed = &old.entries["hypercache://checkpoints/unflushed.bin"];
        assert_eq!(unflushed.sync_status, Some(v094::SyncStatus::Committed));
        assert_eq!(old.entries.len(), 1, "local-only blocks are invisible to 0.9.4");

        let bin = std::fs::read(path.with_extension("bin")).unwrap();
        let old: v094::MetadataManifest = bincode::deserialize(&bin).expect("0.9.4 must decode the bincode manifest");
        assert_eq!(old.entries.len(), 1);
        assert!(old.entries.contains_key("hypercache://checkpoints/unflushed.bin"));
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    #[tokio::test]
    async fn test_local_only_entries_survive_a_restart() {
        let path = temp_manifest_path("localreload");
        let _store = store_with_dirty_and_local_only(&path).await;
        let manifest: MetadataManifest = serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert!(manifest
            .entries
            .values()
            .all(|m| m.sync_status != Some(SyncStatus::LocalOnly)));
        assert_eq!(manifest.local_only_entries.len(), 1);

        let (reloaded, count, _) = MetadataStore::load_from_disk(&path, 1 << 30, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count, 2);
        assert_eq!(
            reloaded
                .get_sync_status("hypercache://kv_cache/ns/m/0/0/_/ab.kv")
                .await
                .unwrap(),
            Some(SyncStatus::LocalOnly)
        );
        assert_eq!(
            reloaded
                .get_sync_status("hypercache://checkpoints/unflushed.bin")
                .await
                .unwrap(),
            Some(SyncStatus::Committed)
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A daemon that did not stop cleanly leaves the unflushed marker; the next start must drop
    /// every local-only object (its data may never have been flushed) and keep everything else.
    #[tokio::test]
    async fn test_unclean_restart_drops_local_only_entries() {
        let path = temp_manifest_path("unclean");
        let store = MetadataStore::new(1 << 30, None, Some(path.clone()));
        assert!(!store.skip_local_only_flush(), "must flush until the marker is armed");
        store.arm_unflushed_marker().unwrap();
        assert!(store.skip_local_only_flush());
        assert!(unflushed_marker_path(&path).exists());

        let kv_off = store.allocate_chunk_offset("/dev/r0", 8192, 4096).await.unwrap();
        let ck_off = store.allocate_chunk_offset("/dev/r0", 4096, 4096).await.unwrap();
        let kv = "hypercache://kv_cache/ns/m/0/0/_/ab.kv";
        let ck = "hypercache://checkpoints/step1.bin";
        store
            .save_layout_with_status(
                kv,
                8192,
                4194304,
                vec![nvme_crc("/dev/r0", kv_off, 8192)],
                Some(SyncStatus::LocalOnly),
            )
            .await
            .unwrap();
        store
            .save_layout_with_status(
                ck,
                4096,
                4194304,
                vec![nvme_crc("/dev/r0", ck_off, 4096)],
                Some(SyncStatus::Committed),
            )
            .await
            .unwrap();
        store.flush_to_disk().await.unwrap();
        drop(store); // crash: the marker stays

        let (reloaded, count, _) = MetadataStore::load_from_disk(&path, 1 << 30, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(reloaded.get_sync_status(kv).await.unwrap(), None);
        assert_eq!(reloaded.get_sync_status(ck).await.unwrap(), Some(SyncStatus::Committed));
        // The dropped object's space is free again
        assert_eq!(
            reloaded.allocate_chunk_offset("/dev/r0", 8192, 4096).await.unwrap(),
            kv_off
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// After a clean shutdown (devices flushed, manifest written, marker removed) local-only objects
    /// survive a restart.
    #[tokio::test]
    async fn test_clean_shutdown_keeps_local_only_entries() {
        let path = temp_manifest_path("cleanstop");
        let store = store_with_dirty_and_local_only(&path).await;
        store.arm_unflushed_marker().unwrap();
        store.stop_skipping_local_only_flush();
        assert!(!store.skip_local_only_flush());
        store.flush_to_disk().await.unwrap();
        store.clear_unflushed_marker().unwrap();
        assert!(!unflushed_marker_path(&path).exists());
        drop(store);

        let (reloaded, count, _) = MetadataStore::load_from_disk(&path, 1 << 30, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count, 2);
        assert_eq!(
            reloaded
                .get_sync_status("hypercache://kv_cache/ns/m/0/0/_/ab.kv")
                .await
                .unwrap(),
            Some(SyncStatus::LocalOnly)
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// Without a manifest nothing survives a restart, so local-only commits never need a flush.
    #[tokio::test]
    async fn test_no_persistence_skips_local_only_flush() {
        let store = MetadataStore::new(1 << 30, None, None);
        assert!(store.skip_local_only_flush());
        store.arm_unflushed_marker().unwrap();
        assert!(store.skip_local_only_flush());
    }

    #[tokio::test]
    async fn test_upgrade_reads_a_0_9_4_bincode_manifest() {
        // A bincode manifest has no field names, so one written by 0.9.4 lacks the new last field.
        // The upgraded daemon must still load it rather than start cold.
        #[derive(Serialize)]
        struct OldManifest {
            entries: HashMap<String, StoredMetadata>,
            local_alloc: HashMap<String, u64>,
            free_pools: HashMap<String, DeviceFreePool>,
            updated_at: i64,
        }
        let meta = StoredMetadata {
            object_size: 4096,
            chunk_size: 4194304,
            chunk_layout: vec![nvme_crc("/dev/u0", 4096, 4096)],
            created_at: 1,
            sync_status: Some(SyncStatus::SyncFailed),
            pinned: false,
            generation: 7,
        };
        let old = OldManifest {
            entries: HashMap::from([("hypercache://old.bin".to_string(), meta)]),
            local_alloc: HashMap::from([("/dev/u0".to_string(), 8192)]),
            free_pools: HashMap::new(),
            updated_at: 1,
        };
        let path = temp_manifest_path("upgradebin").with_extension("bin");
        std::fs::write(&path, bincode::serialize(&old).unwrap()).unwrap();
        assert!(bincode::deserialize::<MetadataManifest>(&std::fs::read(&path).unwrap()).is_err());

        let (store, count, _) = MetadataStore::load_from_disk(&path, 1 << 30, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(
            store.get_sync_status("hypercache://old.bin").await.unwrap(),
            Some(SyncStatus::SyncFailed)
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A RAM chunk backed by a real file of `file_len` bytes (None: no file, as after a reboot).
    fn ram_chunk(dir: &Path, name: &str, length: u64, file_len: Option<u64>) -> ChunkDescriptor {
        let path = dir.join(format!("{}.1.2.3.shm", name));
        if let Some(n) = file_len {
            std::fs::write(&path, vec![5u8; n as usize]).unwrap();
        }
        let mut chunk = ChunkDescriptor::new(StorageTier::Ram, path.to_str().unwrap(), 0, length);
        chunk.checksum = Some(0x1234_5678);
        chunk
    }

    /// After a reboot (or anything that empties the shared memory directory) entries whose RAM files
    /// are gone or short must load as misses, not as layouts that fail every read. That includes
    /// dirty entries: their data is gone, so they must not stay queued for write-back.
    #[tokio::test]
    async fn test_restart_drops_entries_whose_ram_files_are_gone() {
        let path = temp_manifest_path("ramgone");
        let shm = path.parent().unwrap().join("shm");
        std::fs::create_dir_all(&shm).unwrap();
        let store = MetadataStore::new(1 << 30, None, Some(path.clone()));
        let mixed_off = store.allocate_chunk_offset("/dev/r0", 8192, 4096).await.unwrap();

        let intact = "hypercache://s3/b/intact.bin";
        let gone = "hypercache://s3/b/gone.bin";
        let short = "hypercache://s3/b/short.bin";
        let mixed = "hypercache://s3/b/mixed.bin";
        let kv = "hypercache://kv_cache/ns/m/0/0/_/ab.kv";
        let dirty = "hypercache://checkpoints/unsynced.bin";
        let cases: Vec<(&str, Vec<ChunkDescriptor>, Option<SyncStatus>)> = vec![
            (intact, vec![ram_chunk(&shm, "intact", 4096, Some(4096))], None),
            (
                gone,
                vec![ram_chunk(&shm, "gone", 4096, None)],
                Some(SyncStatus::SyncedBackend),
            ),
            (short, vec![ram_chunk(&shm, "short", 4096, Some(100))], None),
            (
                mixed,
                vec![
                    ram_chunk(&shm, "mixed", 4096, Some(4096)),
                    ram_chunk(&shm, "mixed_b", 4096, None),
                    nvme_crc("/dev/r0", mixed_off, 8192),
                ],
                None,
            ),
            (kv, vec![ram_chunk(&shm, "kv", 4096, None)], Some(SyncStatus::LocalOnly)),
            (
                dirty,
                vec![ram_chunk(&shm, "dirty", 4096, None)],
                Some(SyncStatus::Committed),
            ),
        ];
        for (uri, chunks, status) in cases {
            let size = chunks.iter().map(|c| c.length).sum();
            store
                .save_layout_with_status(uri, size, 4194304, chunks, status)
                .await
                .unwrap();
        }
        store.flush_to_disk().await.unwrap();
        drop(store);

        let (reloaded, count, total_bytes) = MetadataStore::load_from_disk(&path, 1 << 30, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(total_bytes, 4096);
        assert!(reloaded.get_layout(intact).await.unwrap().is_some());
        assert!(
            reloaded.list_dirty_entries().await.is_empty(),
            "a lost checkpoint is still queued for write-back"
        );
        for uri in [gone, short, mixed, kv, dirty] {
            assert!(
                reloaded.get_layout(uri).await.unwrap().is_none(),
                "{} must be a miss",
                uri
            );
        }
        // The dropped entries' remaining RAM files are removed and their NVMe space is free again
        assert!(!shm.join("short.1.2.3.shm").exists());
        assert!(!shm.join("mixed.1.2.3.shm").exists());
        assert!(shm.join("intact.1.2.3.shm").exists());
        assert_eq!(
            reloaded.allocate_chunk_offset("/dev/r0", 8192, 4096).await.unwrap(),
            mixed_off
        );

        // The manifest is rewritten without them
        reloaded.flush_to_disk().await.unwrap();
        drop(reloaded);
        let (again, count, _) = MetadataStore::load_from_disk(&path, 1 << 30, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count, 1);
        drop(again);
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// At startup the RAM pool keeps only files the restored index references and counts only
    /// those, so a crash mid-ingest does not shrink the RAM tier.
    #[tokio::test]
    async fn test_reconcile_ram_pool_removes_files_no_entry_references() {
        let path = temp_manifest_path("ramreconcile");
        let shm = path.parent().unwrap().join("shm");
        std::fs::create_dir_all(&shm).unwrap();
        let store = MetadataStore::new(1 << 30, None, Some(path.clone()));
        let kept = ram_chunk(&shm, "kept", 4096, Some(4096));
        store
            .save_layout("hypercache://s3/b/kept.bin", 4096, 4194304, vec![kept.clone()])
            .await
            .unwrap();
        // Chunks of an ingest the crash interrupted: written, never published
        let orphans: Vec<_> = (0..3)
            .map(|i| ram_chunk(&shm, &format!("partial_chunk_{}", i), 4096, Some(4096)))
            .collect();

        let pool = Arc::new(crate::ram::RamMemoryPool::new(shm.clone(), 64 * 1024));
        assert_eq!(
            pool.get_used_bytes(),
            4 * 4096,
            "before the fix this was the starting point"
        );
        store.set_ram_pool(pool.clone()).await;
        store.reconcile_ram_pool(&pool);
        assert_eq!(pool.get_used_bytes(), 4096);
        assert!(Path::new(&kept.device).exists());
        for o in &orphans {
            assert!(!Path::new(&o.device).exists());
        }

        // A second daemon on the same directory must not delete the first one's files
        let stray = ram_chunk(&shm, "other_daemon", 4096, Some(4096));
        let second = crate::ram::RamMemoryPool::new(shm.clone(), 64 * 1024);
        MetadataStore::new(1 << 30, None, None).reconcile_ram_pool(&second);
        assert!(Path::new(&stray.device).exists());

        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A staged write nobody commits is removed once it has been idle for the timeout; committed
    /// objects and staged writes that are still being appended to stay.
    #[tokio::test]
    async fn test_idle_staged_writes_expire() {
        let path = temp_manifest_path("stagedexpire");
        let shm = path.parent().unwrap().join("shm");
        std::fs::create_dir_all(&shm).unwrap();
        let store = MetadataStore::new(1 << 30, None, Some(path.clone()));
        let pool = Arc::new(crate::ram::RamMemoryPool::new(shm.clone(), 1 << 20));
        store.set_ram_pool(pool.clone()).await;

        let abandoned = "hypercache://checkpoints/abandoned.pt";
        let active = "hypercache://checkpoints/active.pt";
        let done = "hypercache://checkpoints/done.pt";
        let ab_chunk = pool.allocate_staged_chunk("abandoned_chunk_0", 4096).unwrap();
        store
            .save_layout_with_status(
                abandoned,
                4096,
                4194304,
                vec![ab_chunk.clone()],
                Some(SyncStatus::Staged),
            )
            .await
            .unwrap();
        store
            .save_layout_with_status(
                active,
                4096,
                4194304,
                vec![nvme_crc("/dev/r0", 4096, 4096)],
                Some(SyncStatus::Staged),
            )
            .await
            .unwrap();
        store
            .save_layout_with_status(
                done,
                4096,
                4194304,
                vec![nvme_crc("/dev/r0", 8192, 4096)],
                Some(SyncStatus::Staged),
            )
            .await
            .unwrap();
        store.update_sync_status(done, SyncStatus::Committed).await.unwrap();
        assert_eq!(pool.get_used_bytes(), 4096);

        let idle = Duration::from_millis(1000);
        assert!(store.expire_stale_staged(idle).await.is_empty(), "nothing is idle yet");
        tokio::time::sleep(Duration::from_millis(600)).await;
        // The active writer appends more data: that counts as activity
        store
            .append_chunks_to_layout(active, 4096, 4194304, vec![nvme_crc("/dev/r0", 12288, 4096)])
            .await
            .unwrap();
        tokio::time::sleep(Duration::from_millis(500)).await;

        assert_eq!(store.expire_stale_staged(idle).await, vec![abandoned.to_string()]);
        assert!(store.get_layout(abandoned).await.unwrap().is_none());
        assert!(
            !Path::new(&ab_chunk.device).exists(),
            "the abandoned RAM chunk file is deleted"
        );
        assert_eq!(pool.get_used_bytes(), 0, "and its reservation returned");
        assert_eq!(store.get_sync_status(active).await.unwrap(), Some(SyncStatus::Staged));
        assert_eq!(store.get_sync_status(done).await.unwrap(), Some(SyncStatus::Committed));

        tokio::time::sleep(Duration::from_millis(1100)).await;
        assert_eq!(store.expire_stale_staged(idle).await, vec![active.to_string()]);
        assert!(store.get_layout(done).await.unwrap().is_some());
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }

    /// A staged write cannot be committed after a restart (its writer is gone), so it is dropped on
    /// load even when all of its chunks were written.
    #[tokio::test]
    async fn test_restart_drops_staged_writes() {
        let path = temp_manifest_path("stagedrestart");
        let shm = path.parent().unwrap().join("shm");
        std::fs::create_dir_all(&shm).unwrap();
        let store = MetadataStore::new(1 << 30, None, Some(path.clone()));
        let off = store.allocate_chunk_offset("/dev/r0", 4096, 4096).await.unwrap();
        let staged_ram = "hypercache://checkpoints/staged_ram.pt";
        let staged_nvme = "hypercache://checkpoints/staged_nvme.pt";
        let kept = "hypercache://checkpoints/committed.pt";
        let full = ram_chunk(&shm, "staged_full", 4096, Some(4096));
        store
            .save_layout_with_status(staged_ram, 4096, 4194304, vec![full.clone()], Some(SyncStatus::Staged))
            .await
            .unwrap();
        store
            .save_layout_with_status(
                staged_nvme,
                4096,
                4194304,
                vec![nvme_crc("/dev/r0", off, 4096)],
                Some(SyncStatus::Staged),
            )
            .await
            .unwrap();
        store
            .save_layout_with_status(
                kept,
                4096,
                4194304,
                vec![ram_chunk(&shm, "kept", 4096, Some(4096))],
                Some(SyncStatus::Committed),
            )
            .await
            .unwrap();
        store.flush_to_disk().await.unwrap();
        drop(store);

        let (reloaded, count, _) = MetadataStore::load_from_disk(&path, 1 << 30, None)
            .await
            .unwrap()
            .unwrap();
        assert_eq!(count, 1);
        assert!(reloaded.get_layout(staged_ram).await.unwrap().is_none());
        assert!(reloaded.get_layout(staged_nvme).await.unwrap().is_none());
        assert_eq!(
            reloaded.get_sync_status(kept).await.unwrap(),
            Some(SyncStatus::Committed)
        );
        assert!(!Path::new(&full.device).exists());
        assert_eq!(
            reloaded.allocate_chunk_offset("/dev/r0", 4096, 4096).await.unwrap(),
            off
        );
        let _ = std::fs::remove_dir_all(path.parent().unwrap());
    }
}
