use anyhow::{Context, Result};
use futures_util::{SinkExt, StreamExt};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::net::{UnixListener, UnixStream};
use tokio_util::codec::{Framed, LinesCodec, LinesCodecError};
use tracing::{debug, error, info, warn};

use crate::auth::with_bearer;
use crate::config::Config;
use crate::discovery::DiscoveredDevice;
use crate::ingestion::IngestionEngine;
use crate::metadata::MetadataStore;
use crate::protocol::{ChunkDescriptor, Request, Response, StorageTier, SyncStatus};
use crate::telemetry::{TelemetryCollector, TelemetryEvent};

/// Maximum size of one newline-delimited IPC request (protects the daemon from unbounded reads)
const MAX_REQUEST_BYTES: usize = 16 * 1024 * 1024;
/// Write-back attempts before an object is left in SYNC_FAILED (it stays protected from eviction)
const WRITEBACK_ATTEMPTS: u32 = 5;

pub struct Server {
    config: Config,
    metadata: MetadataStore,
    ingestion: IngestionEngine,
    telemetry: TelemetryCollector,
    devices: Vec<DiscoveredDevice>,
}

fn now_ts() -> String {
    chrono::Utc::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string()
}

/// Verifies the socket directory is owned by the daemon's user and not writable by others,
/// so another local user cannot pre-create it and swap in a rogue socket.
fn secure_socket_dir(dir: &Path) -> Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    fs::create_dir_all(dir).with_context(|| format!("Failed to create socket directory {:?}", dir))?;
    let meta = fs::metadata(dir)?;

    let probe = dir.join(format!(".owner_probe.{}", std::process::id()));
    fs::write(&probe, b"").with_context(|| format!("Socket directory {:?} is not writable", dir))?;
    let my_uid = fs::metadata(&probe).map(|m| m.uid());
    let _ = fs::remove_file(&probe);
    let my_uid = my_uid?;

    if meta.uid() != my_uid {
        anyhow::bail!(
            "Socket directory {:?} is owned by uid {} but the daemon runs as uid {}. Refusing to use it \
             (another user could replace the socket). Remove the directory or fix its ownership.",
            dir,
            meta.uid(),
            my_uid
        );
    }
    if meta.mode() & 0o022 != 0 {
        let mut perms = meta.permissions();
        perms.set_mode((meta.mode() & 0o7777) & !0o022);
        fs::set_permissions(dir, perms)?;
        warn!("Removed group/other write permission from socket directory {:?}", dir);
    }
    Ok(())
}

/// Resolves on SIGINT (Ctrl+C) or SIGTERM (`docker stop`, Kubernetes pod termination).
/// A container's PID 1 ignores SIGTERM unless it installs a handler, so without this the
/// runtime waits out its grace period, SIGKILLs the daemon and leaves a stale socket behind.
pub async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    match signal(SignalKind::terminate()) {
        Ok(mut term) => {
            tokio::select! {
                _ = tokio::signal::ctrl_c() => {}
                _ = term.recv() => {}
            }
        }
        Err(e) => {
            warn!(
                "Cannot install SIGTERM handler ({}); only Ctrl+C triggers a clean shutdown",
                e
            );
            let _ = tokio::signal::ctrl_c().await;
        }
    }
}

/// Checks that every RAM-tier chunk file holds at least the bytes its descriptor claims.
fn verify_ram_chunks_written(layout: &[ChunkDescriptor]) -> Result<()> {
    for chunk in layout.iter().filter(|c| c.tier == StorageTier::Ram) {
        let have = fs::metadata(&chunk.device).map(|m| m.len()).unwrap_or(0);
        if have < chunk.offset + chunk.length {
            anyhow::bail!(
                "RAM chunk {} holds {} of {} bytes. The client most likely does not share /dev/shm with the \
                 daemon (run the daemon container with --ipc=host or mount the host's /dev/shm into both)",
                chunk.device,
                have,
                chunk.offset + chunk.length
            );
        }
    }
    Ok(())
}

impl Server {
    pub fn new(
        config: Config,
        metadata: MetadataStore,
        ingestion: IngestionEngine,
        telemetry: TelemetryCollector,
        devices: Vec<DiscoveredDevice>,
    ) -> Self {
        Self {
            config,
            metadata,
            ingestion,
            telemetry,
            devices,
        }
    }

    /// Clean shutdown: stop skipping flushes for local-only commits, flush every device, write the
    /// manifest, and only then remove the unflushed marker. Any error leaves the marker in place, so
    /// the next start drops local-only entries instead of trusting them.
    pub async fn shutdown_cleanly(&self) -> anyhow::Result<()> {
        self.metadata.stop_skipping_local_only_flush();
        let devices = self.metadata.allocated_devices();
        self.ingestion.striper().sync_devices(&devices).await?;
        self.metadata.flush_to_disk().await?;
        self.metadata.clear_unflushed_marker()
    }

    pub fn metadata(&self) -> &MetadataStore {
        &self.metadata
    }

    pub async fn run(self: Arc<Self>) -> Result<()> {
        let sock_path = Path::new(&self.config.socket_path);

        if let Some(parent) = sock_path.parent() {
            secure_socket_dir(parent)?;
        }

        // Remove stale socket if exists (after verifying it is not currently active)
        if sock_path.exists() {
            if UnixStream::connect(sock_path).await.is_ok() {
                return Err(anyhow::anyhow!(
                    "HyperCache daemon is already running and active on socket {:?}",
                    sock_path
                ));
            }
            let _ = fs::remove_file(sock_path);
        }

        let listener =
            UnixListener::bind(sock_path).with_context(|| format!("Failed to bind UDS listener at {:?}", sock_path))?;

        crate::access::grant_file_or_warn(sock_path, self.config.client_gid);
        info!("HyperCache Control Plane listening on UDS: {:?}", sock_path);

        loop {
            match listener.accept().await {
                Ok((stream, _)) => {
                    let server = Arc::clone(&self);
                    tokio::spawn(async move {
                        if let Err(e) = server.handle_connection(stream).await {
                            debug!("Connection closed: {}", e);
                        }
                    });
                }
                Err(e) => {
                    error!("Error accepting UDS connection: {}", e);
                }
            }
        }
    }

    async fn handle_connection(&self, stream: UnixStream) -> Result<()> {
        let mut framed = Framed::new(stream, LinesCodec::new_with_max_length(MAX_REQUEST_BYTES));

        while let Some(line) = framed.next().await {
            let line = match line {
                Ok(l) => l,
                Err(LinesCodecError::MaxLineLengthExceeded) => {
                    let resp = Response::error(format!("Request exceeds the {} byte limit", MAX_REQUEST_BYTES));
                    let _ = framed.send(serde_json::to_string(&resp)?).await;
                    break;
                }
                Err(e) => return Err(e.into()),
            };

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let resp = self.handle_request_line(trimmed).await;
            framed.send(serde_json::to_string(&resp)?).await?;
        }

        Ok(())
    }

    async fn handle_request_line(&self, line: &str) -> Response {
        let t_start = Instant::now();
        let req: Result<Request, _> = serde_json::from_str(line);

        match req {
            Ok(Request::Ping) => Response::pong(),
            Ok(Request::GetDevices) => Response::devices(
                self.devices.clone(),
                self.config.ram_cache_enabled,
                self.config.ram_cache_capacity_bytes,
            ),
            Ok(Request::List) => {
                let (objects, total_bytes) = self.metadata.list_entries().await;
                debug!(
                    "Listing cached inventory: {} objects ({:.2} GiB)",
                    objects.len(),
                    total_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
                );
                Response::cached_list(objects, total_bytes, self.metadata.max_capacity_bytes)
            }
            Ok(Request::Evict { uri }) => match self.metadata.evict_entry(&uri).await {
                Ok(Some(freed)) => {
                    info!(
                        "Eviction executed for '{}': freed {:.2} GiB from cache index.",
                        uri,
                        freed as f64 / (1024.0 * 1024.0 * 1024.0)
                    );
                    Response::evicted(uri, freed)
                }
                Ok(None) => {
                    warn!("Eviction requested for '{}' but object was not found in cache.", uri);
                    Response::error(format!("Object '{}' not found in cache", uri))
                }
                Err(e) => {
                    error!("Eviction failed for '{}': {}", uri, e);
                    Response::error(format!("Eviction failed: {}", e))
                }
            },
            Ok(Request::Purge { include_pinned }) => match self.metadata.purge_all(include_pinned).await {
                Ok((count, freed, kept)) => {
                    if kept > 0 {
                        info!(
                            "Cache PURGE executed: cleared {} cached object(s) ({:.2} GiB), kept {} pinned object(s).",
                            count,
                            freed as f64 / (1024.0 * 1024.0 * 1024.0),
                            kept
                        );
                    } else {
                        info!(
                            "Cache PURGE/CLEAN executed: cleared {} cached model(s) ({:.2} GiB freed). Allocation restarts at offset {}.",
                            count,
                            freed as f64 / (1024.0 * 1024.0 * 1024.0),
                            self.config.device_start_offset_bytes
                        );
                    }
                    Response::purged(count, freed, kept)
                }
                Err(e) => {
                    error!("Purge failed: {}", e);
                    Response::error(format!("Purge failed: {}", e))
                }
            },
            Ok(Request::Locate {
                uri,
                auto_ingest,
                tier,
                pinned,
            }) => self.handle_locate(&uri, auto_ingest, tier, pinned, t_start).await,
            Ok(Request::Ingest {
                uri,
                tier,
                pinned,
                allow_synthetic,
            }) => self.handle_ingest(&uri, tier, pinned, allow_synthetic, t_start).await,
            Ok(Request::StageWrite {
                uri,
                size_bytes,
                tier,
                is_leader,
                auto_flush_s3,
                write_through: _,
                pinned,
            }) => {
                self.handle_stage_write(&uri, size_bytes, tier, is_leader, auto_flush_s3, pinned, t_start)
                    .await
            }
            Ok(Request::AppendWrite { uri, size_bytes, tier }) => {
                self.handle_append_write(&uri, size_bytes, tier).await
            }
            Ok(Request::Pin { uri, pinned }) => match self.metadata.set_pinned(&uri, pinned).await {
                Ok(true) => Response::pong(),
                Ok(false) => Response::error(format!("Object '{}' not found in cache", uri)),
                Err(e) => Response::error(format!("Pin update failed: {}", e)),
            },
            Ok(Request::CommitWrite {
                uri,
                sha256,
                write_through,
                local_only,
                chunk_checksums,
                generation,
            }) => {
                if local_only {
                    if write_through {
                        Response::error("local_only and write_through cannot be combined".to_string())
                    } else {
                        self.handle_commit_local_only(&uri, generation, sha256, chunk_checksums, t_start)
                            .await
                    }
                } else {
                    self.handle_commit_write(&uri, sha256, write_through, t_start).await
                }
            }
            Ok(Request::GetWriteStatus { uri }) => self.handle_get_write_status(&uri).await,
            Ok(Request::PreloadBatch { uris, tier }) => self.handle_preload_batch(uris, tier, t_start).await,
            Ok(Request::RecordTelemetry {
                event_type,
                uri,
                object_size_bytes,
                duration_us,
                bandwidth_mbps,
                client_id,
                chunk_count,
                device_targets,
                local_meta_us,
                remote_meta_us,
                local_ttfb_us,
                remote_ttfb_us,
                time_saved_ms,
                cpu_pct,
                status,
            }) => {
                self.telemetry.record(TelemetryEvent {
                    timestamp: now_ts(),
                    event_type,
                    uri,
                    object_size_bytes,
                    duration_us,
                    bandwidth_mbps,
                    client_id: client_id.unwrap_or_else(|| "python_client".to_string()),
                    chunk_count: chunk_count.unwrap_or(1),
                    device_targets: device_targets.unwrap_or_default(),
                    local_meta_us: local_meta_us.unwrap_or(0),
                    remote_meta_us: remote_meta_us.unwrap_or(0),
                    local_ttfb_us: local_ttfb_us.unwrap_or(0),
                    remote_ttfb_us: remote_ttfb_us.unwrap_or(0),
                    time_saved_ms: time_saved_ms.unwrap_or(0),
                    cpu_pct: cpu_pct.unwrap_or(0.0),
                    status: status.unwrap_or_else(|| "OK".to_string()),
                });
                Response::pong()
            }
            Err(e) => Response::error(format!("Malformed JSON request: {}", e)),
        }
    }

    /// Records a daemon-side event. Only values the daemon actually measured are filled in;
    /// remote / time-saved figures are left at 0 (unknown) rather than estimated.
    fn record_event(
        &self,
        event_type: &str,
        status: &str,
        uri: &str,
        object_size: u64,
        layout: &[ChunkDescriptor],
        t_start: Instant,
    ) {
        let duration_us = t_start.elapsed().as_micros() as u64;
        self.telemetry.record(TelemetryEvent {
            timestamp: now_ts(),
            event_type: event_type.to_string(),
            uri: uri.to_string(),
            object_size_bytes: object_size,
            duration_us,
            bandwidth_mbps: 0.0,
            client_id: "control_plane".to_string(),
            chunk_count: layout.len() as u32,
            device_targets: layout.iter().map(|c| c.device.clone()).collect(),
            local_meta_us: duration_us,
            remote_meta_us: 0,
            local_ttfb_us: 0,
            remote_ttfb_us: 0,
            time_saved_ms: 0,
            cpu_pct: 0.0,
            status: status.to_string(),
        });
    }

    async fn handle_locate(
        &self,
        uri: &str,
        auto_ingest: bool,
        tier: Option<crate::protocol::StorageTier>,
        pinned: bool,
        t_start: Instant,
    ) -> Response {
        match self.metadata.get_layout(uri).await {
            // Being written and not committed: its chunks may be unwritten, so it is a miss, with
            // the status saying why. No layout, and no auto-ingest, which would replace the
            // writer's layout.
            Ok(Some(meta)) if meta.sync_status == Some(SyncStatus::Staged) => {
                self.record_event("CACHE_MISS", "MISS", uri, 0, &[], t_start);
                // The generation lets the writer that staged it clean up only its own write
                Response::miss()
                    .with_sync_status(Some(SyncStatus::Staged))
                    .with_generation(meta.generation)
            }
            Ok(Some(meta)) => {
                if pinned && !meta.pinned {
                    let _ = self.metadata.set_pinned(uri, true).await;
                }
                self.record_event("CACHE_HIT", "HIT", uri, meta.object_size, &meta.chunk_layout, t_start);
                Response::hit(meta.object_size, meta.chunk_size, meta.chunk_layout)
                    .with_generation(meta.generation)
                    .with_sync_status(meta.sync_status)
            }
            Ok(None) => {
                self.record_event("CACHE_MISS", "MISS", uri, 0, &[], t_start);
                if !auto_ingest {
                    return Response::miss();
                }
                info!("Cache miss for {}, triggering auto-ingestion...", uri);
                // Concurrent locates for the same URI share one ingest
                match self.ingestion.get_or_ingest(uri, tier, pinned, false).await {
                    Ok((obj, hit)) => {
                        if !hit {
                            self.record_event("INGESTION", "INGESTED", uri, obj.object_size, &obj.layout, t_start);
                        }
                        Response::hit(obj.object_size, obj.chunk_size, obj.layout).with_generation(obj.generation)
                    }
                    Err(e) => {
                        error!("Ingestion error for {}: {:#}", uri, e);
                        Response::error(format!("Ingestion failed: {:#}", e))
                    }
                }
            }
            Err(e) => {
                error!("Metadata query error for {}: {}", uri, e);
                Response::error(format!("Metadata query error: {}", e))
            }
        }
    }

    async fn handle_ingest(
        &self,
        uri: &str,
        tier: Option<crate::protocol::StorageTier>,
        pinned: bool,
        allow_synthetic: bool,
        t_start: Instant,
    ) -> Response {
        match self
            .ingestion
            .ingest_object_full(uri, tier, pinned, allow_synthetic)
            .await
        {
            Ok(obj) => {
                self.record_event("INGESTION", "INGESTED", uri, obj.object_size, &obj.layout, t_start);
                Response::hit(obj.object_size, obj.chunk_size, obj.layout).with_generation(obj.generation)
            }
            Err(e) => {
                error!("Ingestion error for {}: {:#}", uri, e);
                Response::error(format!("Ingestion failed: {:#}", e))
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    async fn handle_stage_write(
        &self,
        uri: &str,
        size_bytes: u64,
        tier: Option<crate::protocol::StorageTier>,
        _is_leader: bool,
        _auto_flush_s3: bool,
        pinned: bool,
        _t_start: Instant,
    ) -> Response {
        let layout_res = if size_bytes == 0 {
            Ok(Vec::new())
        } else {
            self.ingestion.striper().allocate_layout(uri, size_bytes, tier).await
        };
        match layout_res {
            Ok(layout) => {
                let chunk_size = self.config.chunk_size_bytes;
                match self
                    .metadata
                    .save_layout_full(
                        uri,
                        size_bytes,
                        chunk_size,
                        layout.clone(),
                        Some(SyncStatus::Staged),
                        pinned,
                    )
                    .await
                {
                    Ok(generation) => {
                        info!(
                            "Staged write for '{}': {} bytes across {} chunk(s)",
                            uri,
                            size_bytes,
                            layout.len()
                        );
                        Response::write_staged(size_bytes, chunk_size, layout).with_generation(generation)
                    }
                    Err(e) => {
                        self.metadata.release_unpublished(&layout).await;
                        error!("Failed to save staged layout for {}: {}", uri, e);
                        Response::error(format!("Failed to stage metadata: {}", e))
                    }
                }
            }
            Err(e) => {
                error!("Allocation failed for stage write on {}: {:#}", uri, e);
                Response::error(format!("Allocation failed: {:#}", e))
            }
        }
    }

    async fn handle_append_write(
        &self,
        uri: &str,
        size_bytes: u64,
        tier: Option<crate::protocol::StorageTier>,
    ) -> Response {
        let start_chunk_idx = match self.metadata.get_layout(uri).await {
            Ok(Some(meta)) => meta.chunk_layout.len(),
            _ => 0,
        };

        match self
            .ingestion
            .striper()
            .allocate_layout_from_index(uri, start_chunk_idx, size_bytes, tier)
            .await
        {
            Ok(new_chunks) => {
                let chunk_size = self.config.chunk_size_bytes;
                match self
                    .metadata
                    .append_chunks_to_layout(uri, size_bytes, chunk_size, new_chunks.clone())
                    .await
                {
                    Ok((total_size, _full_layout)) => Response::write_staged(total_size, chunk_size, new_chunks),
                    Err(e) => {
                        self.metadata.release_unpublished(&new_chunks).await;
                        Response::error(format!("Failed to append staged layout: {}", e))
                    }
                }
            }
            Err(e) => Response::error(format!("Failed to allocate appended chunks: {:#}", e)),
        }
    }

    async fn handle_commit_write(
        &self,
        uri: &str,
        sha256: Option<String>,
        write_through: bool,
        t_start: Instant,
    ) -> Response {
        let meta = match self.metadata.get_layout(uri).await {
            Ok(Some(meta)) => meta,
            Ok(None) => return Response::error(format!("URI '{}' was not staged prior to commit", uri)),
            Err(e) => return Response::error(format!("Metadata lookup failed for commit: {}", e)),
        };
        let size = meta.object_size;
        let generation = meta.generation;
        let layout = meta.chunk_layout.clone();

        // A client that does not share /dev/shm with the daemon writes into its own private copy
        // and leaves the daemon's pre-created chunk files empty: refuse to commit that.
        if let Err(e) = verify_ram_chunks_written(&layout) {
            error!("Commit of '{}' rejected: {:#}", uri, e);
            return Response::error(format!("{:#}", e));
        }

        // The client wrote the chunks directly; flush them before declaring the write committed
        if let Err(e) = self.ingestion.striper().sync_chunks(&layout).await {
            return Response::error(format!("Failed to flush staged data to stable storage: {:#}", e));
        }

        // Verify end-to-end integrity when the client supplied a digest
        if let Some(expected) = sha256.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            match self.ingestion.digest_layout(&layout).await {
                Ok((actual, crcs)) => {
                    if !actual.eq_ignore_ascii_case(expected) {
                        error!(
                            "Commit of '{}' rejected: SHA-256 mismatch (expected {}, got {})",
                            uri, expected, actual
                        );
                        return Response::error(format!(
                            "SHA-256 mismatch for '{}': client reported {}, staged data hashes to {}",
                            uri, expected, actual
                        ));
                    }
                    self.metadata.set_chunk_checksums(uri, generation, &crcs).await;
                }
                Err(e) => return Response::error(format!("Failed to verify staged data for '{}': {:#}", uri, e)),
            }
        }

        if write_through || self.config.write_through_enabled {
            self.metadata
                .update_sync_status_if_generation(uri, generation, SyncStatus::SyncingBackend)
                .await;
            info!(
                "Executing synchronous write-through flush to authoritative backend for '{}' ({} bytes)...",
                uri, size
            );
            return match self.ingestion.flush_to_backend(uri, &layout).await {
                Ok(()) => {
                    self.metadata
                        .update_sync_status_if_generation(uri, generation, SyncStatus::SyncedBackend)
                        .await;
                    info!(
                        "Synchronous write-through flush succeeded for '{}' in {:.2} ms.",
                        uri,
                        t_start.elapsed().as_secs_f64() * 1000.0
                    );
                    Response::write_committed(uri.to_string(), size)
                }
                Err(e) => {
                    self.metadata
                        .update_sync_status_if_generation(uri, generation, SyncStatus::SyncFailed)
                        .await;
                    error!("Write-through flush failed for '{}': {:#}", uri, e);
                    Response::error(format!("Write-through flush to authoritative backend failed: {:#}", e))
                }
            };
        }

        self.metadata
            .update_sync_status_if_generation(uri, generation, SyncStatus::Committed)
            .await;
        info!(
            "Committed write for '{}': {} bytes in {:.2} ms. Asynchronous write-back active.",
            uri,
            size,
            t_start.elapsed().as_secs_f64() * 1000.0
        );
        self.spawn_writeback(uri.to_string(), generation, layout, true);
        Response::write_committed(uri.to_string(), size)
    }

    /// Commits a staged object that has no authoritative copy. Its data is synced to the cache
    /// device, but it is never flushed to a backend and stays evictable.
    ///
    /// Only a `STAGED` layout can be committed this way, so an object waiting for write-back can
    /// never lose its flush. With `generation`, the layout must be the one the caller staged.
    async fn handle_commit_local_only(
        &self,
        uri: &str,
        generation: Option<u64>,
        sha256: Option<String>,
        chunk_checksums: Option<Vec<u32>>,
        t_start: Instant,
    ) -> Response {
        let meta = match self.metadata.get_layout(uri).await {
            Ok(Some(meta)) => meta,
            Ok(None) => return Response::error(format!("URI '{}' was not staged prior to commit", uri)),
            Err(e) => return Response::error(format!("Metadata lookup failed for commit: {}", e)),
        };
        if let Some(staged) = generation {
            if staged != meta.generation {
                return Response::error(format!(
                    "'{}' was replaced since it was staged (staged generation {}, current {})",
                    uri, staged, meta.generation
                ));
            }
        }
        if meta.sync_status != Some(SyncStatus::Staged) {
            let status = meta.sync_status.map_or("none", SyncStatus::as_str);
            return Response::error(format!(
                "'{}' is {}, not STAGED: only a staged write can be committed as local-only",
                uri, status
            ));
        }
        if let Err(e) = verify_ram_chunks_written(&meta.chunk_layout) {
            error!("Commit of '{}' rejected: {:#}", uri, e);
            return Response::error(format!("{:#}", e));
        }
        // Local-only data (KV cache) can be recomputed, so it skips the device flush while the
        // unflushed marker guarantees that a restart after an unclean stop drops it instead of
        // trusting it (see `metadata::unflushed_marker_path`).
        if !self.metadata.skip_local_only_flush() {
            if let Err(e) = self.ingestion.striper().sync_chunks(&meta.chunk_layout).await {
                return Response::error(format!("Failed to flush staged data to stable storage: {:#}", e));
            }
        }
        // With a digest, the daemon hashes the staged data itself and records its own CRCs. Any
        // CRCs the client also sent must then agree with what is really on the device.
        let mut chunk_checksums = chunk_checksums;
        if let Some(expected) = sha256.as_deref().map(str::trim).filter(|s| !s.is_empty()) {
            match self.ingestion.digest_layout(&meta.chunk_layout).await {
                Ok((actual, crcs)) => {
                    if !actual.eq_ignore_ascii_case(expected) {
                        error!(
                            "Commit of '{}' rejected: SHA-256 mismatch (expected {}, got {})",
                            uri, expected, actual
                        );
                        return Response::error(format!(
                            "SHA-256 mismatch for '{}': client reported {}, staged data hashes to {}",
                            uri, expected, actual
                        ));
                    }
                    if chunk_checksums.as_ref().is_some_and(|client| *client != crcs) {
                        return Response::error(format!(
                            "Checksum mismatch for '{}': the client's chunk CRC32s differ from the staged data",
                            uri
                        ));
                    }
                    chunk_checksums = Some(crcs);
                }
                Err(e) => return Response::error(format!("Failed to verify staged data for '{}': {:#}", uri, e)),
            }
        }
        if let Some(crcs) = chunk_checksums {
            if !self.metadata.set_chunk_checksums(uri, meta.generation, &crcs).await {
                return Response::error(format!(
                    "Could not record {} checksum(s) for '{}': the layout changed or has a different chunk count",
                    crcs.len(),
                    uri
                ));
            }
        }
        if !self
            .metadata
            .update_sync_status_if_generation(uri, meta.generation, SyncStatus::LocalOnly)
            .await
        {
            return Response::error(format!("'{}' was replaced or evicted during commit", uri));
        }
        info!(
            "Committed local-only write for '{}': {} bytes in {:.2} ms.",
            uri,
            meta.object_size,
            t_start.elapsed().as_secs_f64() * 1000.0
        );
        // The reply says LOCAL_ONLY, so a client can tell this daemon really kept the object local.
        // An older daemon ignores local_only and replies COMMITTED.
        Response::write_committed(uri.to_string(), meta.object_size).with_sync_status(Some(SyncStatus::LocalOnly))
    }

    /// Flushes a committed object to the authoritative backend in the background, retrying
    /// with exponential backoff. Status updates are tied to the layout generation so a newer
    /// write of the same URI is never marked as synced by an older flush.
    fn spawn_writeback(&self, uri: String, generation: u64, layout: Vec<ChunkDescriptor>, replicate: bool) {
        let ingestion = self.ingestion.clone();
        let metadata = self.metadata.clone();
        let config = self.config.clone();

        tokio::spawn(async move {
            if replicate && config.peer_replication_factor >= 2 {
                if let Some(ref ctrl_url) = config.controller_url {
                    if let Err(e) = Self::replicate_to_peers(&config, ctrl_url, &ingestion, &uri, &layout).await {
                        warn!("Peer replication of '{}' failed: {:#}", uri, e);
                    }
                }
            }

            for attempt in 1..=WRITEBACK_ATTEMPTS {
                if !metadata
                    .update_sync_status_if_generation(&uri, generation, SyncStatus::SyncingBackend)
                    .await
                {
                    info!(
                        "Write-back of '{}' superseded by a newer write or eviction; stopping.",
                        uri
                    );
                    return;
                }
                match ingestion.flush_to_backend(&uri, &layout).await {
                    Ok(()) => {
                        info!("Successfully synced '{}' to authoritative backend in background.", uri);
                        metadata
                            .update_sync_status_if_generation(&uri, generation, SyncStatus::SyncedBackend)
                            .await;
                        return;
                    }
                    Err(e) => {
                        metadata
                            .update_sync_status_if_generation(&uri, generation, SyncStatus::SyncFailed)
                            .await;
                        if attempt == WRITEBACK_ATTEMPTS {
                            error!(
                                "Background sync of '{}' failed after {} attempts: {:#}. The object stays cached \
                                 (protected from eviction) and is retried on the next daemon start.",
                                uri, attempt, e
                            );
                            return;
                        }
                        let backoff = Duration::from_secs(2u64.pow(attempt));
                        warn!(
                            "Background sync of '{}' failed (attempt {}/{}): {:#}. Retrying in {:?}.",
                            uri, attempt, WRITEBACK_ATTEMPTS, e, backoff
                        );
                        tokio::time::sleep(backoff).await;
                    }
                }
            }
        });
    }

    /// Re-queues write-back for objects that were committed (or failed to sync) before the
    /// daemon stopped. Returns the number of objects re-queued.
    pub async fn resume_pending_writebacks(&self) -> usize {
        let pending = self.metadata.list_dirty_entries().await;
        for (uri, meta) in &pending {
            info!(
                "Resuming write-back of '{}' ({} bytes, status {:?}) after restart",
                uri, meta.object_size, meta.sync_status
            );
            self.spawn_writeback(uri.clone(), meta.generation, meta.chunk_layout.clone(), false);
        }
        pending.len()
    }

    /// Pushes a checkpoint replica to peer nodes chosen by rendezvous hashing (spreads load
    /// instead of always picking the alphabetically first peers). The body is streamed chunk by chunk.
    async fn replicate_to_peers(
        config: &Config,
        controller_url: &str,
        ingestion: &IngestionEngine,
        uri: &str,
        layout: &[ChunkDescriptor],
    ) -> Result<()> {
        let client = reqwest::Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .read_timeout(Duration::from_secs(120))
            .build()?;

        let nodes_url = format!("{}/api/v1/nodes", controller_url.trim_end_matches('/'));
        let resp = with_bearer(client.get(&nodes_url), config.cluster_auth_token.as_deref())
            .send()
            .await?;
        if !resp.status().is_success() {
            anyhow::bail!("Controller returned HTTP {} for node list", resp.status());
        }

        let summary: crate::controller::FleetSummary = resp.json().await?;
        let mut candidates: Vec<crate::controller::NodeState> = summary
            .nodes
            .into_iter()
            .filter(|n| n.status == crate::controller::NodeStatus::Online && n.registration.node_id != config.node_id)
            .collect();
        if candidates.is_empty() {
            return Ok(());
        }

        let needed = (config.peer_replication_factor - 1).min(candidates.len());
        let mut chosen = Vec::with_capacity(needed);
        for _ in 0..needed {
            let pick =
                crate::controller::assign_uri_rendezvous(uri, &candidates).map(|n| n.registration.node_id.clone());
            if let Some(id) = pick {
                let idx = candidates.iter().position(|n| n.registration.node_id == id).unwrap();
                chosen.push(candidates.remove(idx));
            }
        }

        let total: u64 = layout.iter().map(|c| c.length).sum();
        for peer in chosen {
            let replica_url = format!(
                "http://{}:{}/api/v1/agent/replica",
                peer.registration.address, peer.registration.http_port
            );
            let ing = ingestion.clone();
            let chunks = layout.to_vec();
            let body_stream = futures_util::stream::iter(chunks).then(move |chunk| {
                let ing = ing.clone();
                async move {
                    ing.read_chunk(&chunk)
                        .await
                        .map_err(|e| std::io::Error::other(e.to_string()))
                }
            });
            let req = client
                .post(&replica_url)
                .query(&[("uri", uri)])
                .header(reqwest::header::CONTENT_LENGTH, total)
                .body(reqwest::Body::wrap_stream(body_stream));
            match with_bearer(req, config.cluster_auth_token.as_deref()).send().await {
                Ok(r) if r.status().is_success() => info!(
                    "Replicated checkpoint '{}' ({} bytes) to peer node '{}'",
                    uri, total, peer.registration.node_id
                ),
                Ok(r) => warn!(
                    "Peer '{}' rejected replica of '{}' with HTTP {}",
                    peer.registration.node_id,
                    uri,
                    r.status()
                ),
                Err(e) => warn!(
                    "Failed to send replica of '{}' to '{}': {}",
                    uri, peer.registration.node_id, e
                ),
            }
        }

        Ok(())
    }

    async fn handle_get_write_status(&self, uri: &str) -> Response {
        match self.metadata.get_layout(uri).await {
            Ok(Some(meta)) => {
                let status = meta.sync_status.unwrap_or(SyncStatus::Committed);
                Response::write_status(status, status.progress_percent())
            }
            Ok(None) => Response::error(format!("URI '{}' not found", uri)),
            Err(e) => Response::error(format!("Metadata error: {}", e)),
        }
    }

    async fn handle_preload_batch(
        &self,
        uris: Vec<String>,
        tier: Option<crate::protocol::StorageTier>,
        t_start: Instant,
    ) -> Response {
        let count = uris.len();
        info!("Beginning batch preload for {} object(s)", count);

        let outcomes: Vec<(String, Result<u64>)> = futures_util::stream::iter(uris)
            .map(|uri| async move {
                let res = self
                    .ingestion
                    .get_or_ingest(&uri, tier, false, false)
                    .await
                    .map(|(obj, _)| obj.object_size);
                (uri, res)
            })
            .buffer_unordered(4)
            .collect()
            .await;

        let mut total_bytes = 0u64;
        let mut failed = Vec::new();
        for (uri, res) in outcomes {
            match res {
                Ok(size) => total_bytes += size,
                Err(e) => {
                    warn!("Failed to ingest '{}' in batch: {:#}", uri, e);
                    failed.push(uri);
                }
            }
        }
        failed.sort();
        let success_count = count - failed.len();

        info!(
            "Batch preload completed: {}/{} successful ({:.2} MiB in {} ms)",
            success_count,
            count,
            total_bytes as f64 / (1024.0 * 1024.0),
            t_start.elapsed().as_millis()
        );

        Response::batch_result(success_count, failed, total_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::DiscoveredDevice;
    use crate::protocol::StorageTier;

    #[tokio::test]
    async fn test_server_stage_write_and_commit_flow() {
        let test_dir = std::env::temp_dir().join(format!(
            "hypercache_srv_test_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&test_dir).unwrap();

        let raw_dev0 = test_dir.join("dev0.raw");
        let raw_dev1 = test_dir.join("dev1.raw");
        std::fs::write(&raw_dev0, vec![0u8; 10 * 1024 * 1024]).unwrap();
        std::fs::write(&raw_dev1, vec![0u8; 10 * 1024 * 1024]).unwrap();

        let config = Config {
            block_devices: vec![
                raw_dev0.to_str().unwrap().to_string(),
                raw_dev1.to_str().unwrap().to_string(),
            ],
            chunk_size_bytes: 4 * 1024 * 1024,
            device_start_offset_bytes: 4096,
            auto_create_mock_devices: false,
            ..Config::default()
        };

        let metadata = MetadataStore::new(10 * 1024 * 1024 * 1024, None, None);
        let ingestion = IngestionEngine::new(config.clone(), metadata.clone(), None);
        let telemetry = TelemetryCollector::new("127.0.0.1", 9110, false);
        let devices = vec![DiscoveredDevice {
            path: raw_dev0.to_str().unwrap().to_string(),
            is_nvme: true,
            is_virtual: false,
            is_partition: false,
            is_rotational: false,
            size_bytes: 10 * 1024 * 1024,
            model: Some("TestNVMe".to_string()),
            existing_fs: None,
            numa_node: Some(0),
        }];

        let server = Server::new(config, metadata, ingestion, telemetry, devices);

        let uri = "hypercache://checkpoints/epoch_1.pt";
        let t_start = Instant::now();

        // 1. Stage write 8 MiB (2 chunks of 4 MiB)
        let resp_stage = server
            .handle_stage_write(uri, 8388608, Some(StorageTier::Nvme), true, true, false, t_start)
            .await;
        assert_eq!(resp_stage.status, crate::protocol::ResponseStatus::WriteStaged);
        assert_eq!(resp_stage.object_size, Some(8388608));
        let layout = resp_stage.chunk_layout.unwrap();
        assert_eq!(layout.len(), 2);
        // The object's start drive depends on its URI, but its two chunks use both drives
        let mut used = [layout[0].device.as_str(), layout[1].device.as_str()];
        used.sort();
        assert_eq!(used, [raw_dev0.to_str().unwrap(), raw_dev1.to_str().unwrap()]);

        // 2. Commit write (write-back by default)
        let resp_commit = server.handle_commit_write(uri, None, false, t_start).await;
        assert_eq!(resp_commit.status, crate::protocol::ResponseStatus::WriteCommitted);
        assert_eq!(resp_commit.object_size, Some(8388608));

        // 3. Get write status
        let resp_status = server.handle_get_write_status(uri).await;
        assert_eq!(resp_status.status, crate::protocol::ResponseStatus::WriteStatus);

        // 4. Locate object -> should be Hit immediately
        let resp_locate = server.handle_locate(uri, false, None, false, t_start).await;
        assert_eq!(resp_locate.status, crate::protocol::ResponseStatus::Hit);
        assert_eq!(resp_locate.object_size, Some(8388608));

        let _ = std::fs::remove_dir_all(&test_dir);
    }

    #[tokio::test]
    async fn test_server_stage_write_and_commit_ram_flow() {
        let test_dir = std::env::temp_dir().join(format!(
            "hypercache_server_ram_test_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&test_dir).unwrap();
        let shm_dir = test_dir.join("shm");
        std::fs::create_dir_all(&shm_dir).unwrap();

        let config = Config {
            ram_cache_enabled: true,
            ram_cache_shm_path: shm_dir.to_str().unwrap().to_string(),
            ram_cache_capacity_bytes: Some(100 * 1024 * 1024),
            chunk_size_bytes: 4 * 1024 * 1024,
            ..Config::default()
        };

        let ram_pool = Arc::new(crate::ram::RamMemoryPool::new(
            shm_dir.clone(),
            config.ram_cache_capacity_bytes.unwrap_or(100 * 1024 * 1024),
        ));

        let metadata = MetadataStore::new(10 * 1024 * 1024 * 1024, None, None);
        let ingestion = IngestionEngine::new(config.clone(), metadata.clone(), Some(ram_pool));
        let telemetry = TelemetryCollector::new("127.0.0.1", 9110, false);

        let server = Server::new(config, metadata, ingestion, telemetry, vec![]);
        let uri = "hypercache://checkpoints/ram_epoch_1.pt";
        let t_start = Instant::now();

        // 1. Stage write to Tier 1 RAM (8 MiB = 2 chunks)
        let resp_stage = server
            .handle_stage_write(uri, 8388608, Some(StorageTier::Ram), true, true, false, t_start)
            .await;
        assert_eq!(resp_stage.status, crate::protocol::ResponseStatus::WriteStaged);
        assert_eq!(resp_stage.object_size, Some(8388608));
        let layout = resp_stage.chunk_layout.unwrap();
        assert_eq!(layout.len(), 2);
        assert_eq!(layout[0].tier, StorageTier::Ram);
        assert_eq!(layout[1].tier, StorageTier::Ram);

        // The daemon pre-creates the chunk files; a client that never writes them (e.g. one with
        // a private /dev/shm) must not get a successful commit
        for chunk in &layout {
            assert_eq!(std::fs::metadata(&chunk.device).unwrap().len(), 0);
        }
        let resp_unwritten = server.handle_commit_write(uri, None, false, t_start).await;
        assert_eq!(resp_unwritten.status, crate::protocol::ResponseStatus::Error);
        assert!(resp_unwritten
            .error_message
            .unwrap_or_default()
            .contains("does not share /dev/shm"));

        // 2. Client writes the chunks, then commits (write-back)
        for chunk in &layout {
            std::fs::write(&chunk.device, vec![7u8; chunk.length as usize]).unwrap();
        }
        let resp_commit = server.handle_commit_write(uri, None, false, t_start).await;
        assert_eq!(resp_commit.status, crate::protocol::ResponseStatus::WriteCommitted);

        // 3. Locate object -> Hit in RAM
        let resp_locate = server.handle_locate(uri, false, None, false, t_start).await;
        assert_eq!(resp_locate.status, crate::protocol::ResponseStatus::Hit);
        assert_eq!(resp_locate.object_size, Some(8388608));

        let _ = std::fs::remove_dir_all(&test_dir);
    }

    /// Staged checkpoints written into RAM count toward ram_cache_capacity_bytes from the moment
    /// they are staged, and give the space back when replaced or removed.
    #[tokio::test]
    async fn test_staged_ram_writes_respect_the_ram_cap() {
        let test_dir = std::env::temp_dir().join(format!(
            "hypercache_server_ram_cap_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        let shm_dir = test_dir.join("shm");
        std::fs::create_dir_all(&shm_dir).unwrap();
        let mib = 1024 * 1024u64;
        let config = Config {
            ram_cache_enabled: true,
            ram_cache_shm_path: shm_dir.to_str().unwrap().to_string(),
            ram_cache_capacity_bytes: Some(16 * mib),
            chunk_size_bytes: 4 * 1024 * 1024,
            block_devices: vec![],
            ..Config::default()
        };
        let ram_pool = Arc::new(crate::ram::RamMemoryPool::new(shm_dir.clone(), 16 * mib));
        let metadata = MetadataStore::new(10 * 1024 * 1024 * 1024, None, None);
        metadata.set_ram_pool(ram_pool.clone()).await;
        let ingestion = IngestionEngine::new(config.clone(), metadata.clone(), Some(ram_pool.clone()));
        let telemetry = TelemetryCollector::new("127.0.0.1", 9110, false);
        let server = Server::new(config, metadata.clone(), ingestion, telemetry, vec![]);
        let t = Instant::now();
        let files = || std::fs::read_dir(&shm_dir).unwrap().count();

        let a = "hypercache://checkpoints/a.pt";
        let staged = server
            .handle_stage_write(a, 12 * mib, Some(StorageTier::Ram), true, true, false, t)
            .await;
        assert_eq!(staged.status, crate::protocol::ResponseStatus::WriteStaged);
        assert_eq!(
            ram_pool.get_used_bytes(),
            12 * mib,
            "the staged layout reserves its full size"
        );

        // A second checkpoint that does not fit is refused and leaves nothing behind
        let files_before = files();
        let b = "hypercache://checkpoints/b.pt";
        let refused = server
            .handle_stage_write(b, 8 * mib, Some(StorageTier::Ram), true, true, false, t)
            .await;
        assert_eq!(refused.status, crate::protocol::ResponseStatus::Error);
        assert_eq!(files(), files_before);
        assert_eq!(ram_pool.get_used_bytes(), 12 * mib);
        assert!(metadata.get_layout(b).await.unwrap().is_none());
        // So is an ingest-style chunk write into the remaining space plus one byte
        assert!(ram_pool
            .allocate_and_write_chunk("x", &vec![0u8; 4 * mib as usize + 1])
            .is_err());

        // Re-staging the same URI smaller gives the old reservation back
        let restaged = server
            .handle_stage_write(a, 4 * mib, Some(StorageTier::Ram), true, true, false, t)
            .await;
        assert_eq!(restaged.status, crate::protocol::ResponseStatus::WriteStaged);
        assert_eq!(ram_pool.get_used_bytes(), 4 * mib);

        // Committing keeps the count; removing the object returns it
        let layout = restaged.chunk_layout.unwrap();
        for chunk in &layout {
            std::fs::write(&chunk.device, vec![7u8; chunk.length as usize]).unwrap();
        }
        let committed = server.handle_commit_write(a, None, false, t).await;
        assert_eq!(committed.status, crate::protocol::ResponseStatus::WriteCommitted);
        assert_eq!(ram_pool.get_used_bytes(), 4 * mib);
        metadata.evict_entry(a).await.unwrap();
        assert_eq!(ram_pool.get_used_bytes(), 0);
        assert_eq!(files(), 0);

        let _ = std::fs::remove_dir_all(&test_dir);
    }

    /// A staged object is being written: locate must not serve its layout (a reader would get
    /// unwritten chunks) nor ingest over it (that would replace the writer's layout).
    #[tokio::test]
    async fn test_staged_objects_are_never_served_or_ingested_over() {
        let (server, dir) = nvme_server("stagedread", 16, |_| {});
        let uri = "hypercache://checkpoints/in_progress.pt";
        let t = Instant::now();
        let staged = server
            .handle_stage_write(uri, 2 * 1024 * 1024, None, false, false, false, t)
            .await;
        assert_eq!(staged.status, crate::protocol::ResponseStatus::WriteStaged);

        for auto_ingest in [false, true] {
            let res = server.handle_locate(uri, auto_ingest, None, false, t).await;
            assert_eq!(res.status, crate::protocol::ResponseStatus::Miss);
            assert!(res.chunk_layout.is_none());
            assert_eq!(res.sync_status, Some(SyncStatus::Staged));
        }
        assert!(server.ingestion.get_or_ingest(uri, None, false, false).await.is_err());
        let meta = server.metadata().get_layout(uri).await.unwrap().unwrap();
        assert_eq!(
            meta.generation,
            staged.generation.unwrap(),
            "the writer's layout was replaced"
        );
        assert_eq!(meta.sync_status, Some(SyncStatus::Staged));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_server_write_through_flow() {
        let test_dir = std::env::temp_dir().join(format!(
            "hypercache_server_wt_test_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&test_dir).unwrap();
        let target_file = test_dir.join("authoritative.pt");

        let raw_dev = test_dir.join("dev0.raw");
        std::fs::write(&raw_dev, vec![0u8; 10 * 1024 * 1024]).unwrap();

        let config = Config {
            block_devices: vec![raw_dev.to_str().unwrap().to_string()],
            chunk_size_bytes: 4 * 1024 * 1024,
            device_start_offset_bytes: 4096,
            auto_create_mock_devices: false,
            nfs_mount_path: test_dir.to_string_lossy().to_string(),
            ..Config::default()
        };

        let metadata = MetadataStore::new(10 * 1024 * 1024 * 1024, None, None);
        let ingestion = IngestionEngine::new(config.clone(), metadata.clone(), None);
        let telemetry = TelemetryCollector::new("127.0.0.1", 9110, false);
        let devices = vec![DiscoveredDevice {
            path: raw_dev.to_str().unwrap().to_string(),
            is_nvme: true,
            is_virtual: false,
            is_partition: false,
            is_rotational: false,
            size_bytes: 10 * 1024 * 1024,
            model: Some("TestNVMe".to_string()),
            existing_fs: None,
            numa_node: Some(0),
        }];

        let server = Server::new(config, metadata.clone(), ingestion, telemetry, devices);
        let uri = format!("file://{}", target_file.display());
        let t_start = Instant::now();

        // 1. Stage write 4 MiB
        let resp_stage = server
            .handle_stage_write(&uri, 4194304, Some(StorageTier::Nvme), true, true, false, t_start)
            .await;
        assert_eq!(resp_stage.status, crate::protocol::ResponseStatus::WriteStaged);

        // 2. Commit write with write_through = true
        let resp_commit = server.handle_commit_write(&uri, None, true, t_start).await;
        assert_eq!(resp_commit.status, crate::protocol::ResponseStatus::WriteCommitted);

        // 3. Verify target authoritative file was created synchronously
        assert!(target_file.exists());
        assert_eq!(target_file.metadata().unwrap().len(), 4194304);

        // 4. Verify sync status is immediately SYNCED_BACKEND
        let status = metadata.get_sync_status(&uri).await.unwrap();
        assert_eq!(status, Some(SyncStatus::SyncedBackend));

        let _ = std::fs::remove_dir_all(&test_dir);
    }

    fn nvme_server(tag: &str, dev_mb: usize, extra: impl FnOnce(&mut Config)) -> (Server, std::path::PathBuf) {
        let test_dir = std::env::temp_dir().join(format!(
            "hypercache_srv_{}_{}",
            tag,
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&test_dir).unwrap();
        let raw_dev = test_dir.join("dev0.raw");
        std::fs::write(&raw_dev, vec![0u8; dev_mb * 1024 * 1024]).unwrap();
        let mut config = Config {
            block_devices: vec![raw_dev.to_str().unwrap().to_string()],
            chunk_size_bytes: 1024 * 1024,
            device_start_offset_bytes: 4096,
            auto_create_mock_devices: false,
            nfs_mount_path: test_dir.to_string_lossy().to_string(),
            ..Config::default()
        };
        extra(&mut config);
        let metadata = MetadataStore::new(10 * 1024 * 1024 * 1024, None, None);
        let ingestion = IngestionEngine::new(config.clone(), metadata.clone(), None);
        let telemetry = TelemetryCollector::new("127.0.0.1", 9110, false);
        (Server::new(config, metadata, ingestion, telemetry, vec![]), test_dir)
    }

    async fn write_staged(_server: &Server, layout: &[ChunkDescriptor], data: &[u8]) {
        use std::io::{Seek, SeekFrom, Write};
        let mut pos = 0usize;
        for c in layout {
            let mut f = std::fs::OpenOptions::new().write(true).open(&c.device).unwrap();
            f.seek(SeekFrom::Start(c.offset)).unwrap();
            f.write_all(&data[pos..pos + c.length as usize]).unwrap();
            pos += c.length as usize;
        }
    }

    #[tokio::test]
    async fn test_local_only_commit_is_never_flushed_and_stays_evictable() {
        let (server, dir) = nvme_server("local", 16, |config| {
            // A flush attempt would be visible: this path would appear under the NFS root.
            config.authoritative_backend = "nfs".to_string();
        });
        let uri = "hypercache://kv_cache/model/0/0/_/abc.kv";
        let data: Vec<u8> = (0..(2 * 1024 * 1024 + 123)).map(|i| (i % 251) as u8).collect();
        let t = Instant::now();

        let staged = server
            .handle_stage_write(uri, data.len() as u64, None, false, false, false, t)
            .await;
        let layout = staged.chunk_layout.unwrap();
        write_staged(&server, &layout, &data).await;

        // Before commit, locate is a miss with no layout, and reports the staged status
        let before = server.handle_locate(uri, false, None, false, t).await;
        assert_eq!(before.status, crate::protocol::ResponseStatus::Miss);
        assert!(before.chunk_layout.is_none());
        assert_eq!(before.sync_status, Some(SyncStatus::Staged));

        let mut pos = 0usize;
        let crcs: Vec<u32> = layout
            .iter()
            .map(|c| {
                let crc = crc32fast::hash(&data[pos..pos + c.length as usize]);
                pos += c.length as usize;
                crc
            })
            .collect();
        let committed = server
            .handle_commit_local_only(uri, staged.generation, None, Some(crcs.clone()), t)
            .await;
        assert_eq!(committed.status, crate::protocol::ResponseStatus::WriteCommitted);
        // The reply proves this daemon kept the object local; an older one would say COMMITTED
        assert_eq!(committed.sync_status, Some(SyncStatus::LocalOnly));

        let after = server.handle_locate(uri, false, None, false, t).await;
        assert_eq!(after.sync_status, Some(SyncStatus::LocalOnly));
        let recorded: Vec<u32> = after
            .chunk_layout
            .unwrap()
            .iter()
            .map(|c| c.checksum.unwrap())
            .collect();
        assert_eq!(recorded, crcs);

        let meta = server.metadata().get_layout(uri).await.unwrap().unwrap();
        assert!(!meta.is_dirty(), "local-only objects must stay evictable");
        assert!(server.metadata().list_dirty_entries().await.is_empty());

        tokio::time::sleep(Duration::from_millis(200)).await;
        let files: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().file_name())
            .collect();
        assert_eq!(
            files,
            vec![std::ffi::OsString::from("dev0.raw")],
            "no backend flush may run"
        );
        assert_eq!(
            server.metadata().get_sync_status(uri).await.unwrap(),
            Some(SyncStatus::LocalOnly)
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_local_only_commit_rejects_wrong_checksum_count_and_write_through() {
        let (server, dir) = nvme_server("localbad", 16, |_| {});
        let uri = "hypercache://kv_cache/model/0/0/_/bad.kv";
        let t = Instant::now();
        let staged = server.handle_stage_write(uri, 4096, None, false, false, false, t).await;
        write_staged(&server, &staged.chunk_layout.unwrap(), &[7u8; 4096]).await;

        let bad = server
            .handle_commit_local_only(uri, None, None, Some(vec![1, 2, 3]), t)
            .await;
        assert_eq!(bad.status, crate::protocol::ResponseStatus::Error);
        assert_eq!(
            server.metadata().get_sync_status(uri).await.unwrap(),
            Some(SyncStatus::Staged)
        );

        let line = format!(r#"{{"action":"commit_write","uri":"{uri}","local_only":true,"write_through":true}}"#);
        let conflict = server.handle_request_line(&line).await;
        assert_eq!(conflict.status, crate::protocol::ResponseStatus::Error);
        std::fs::remove_dir_all(&dir).ok();
    }

    fn crc_per_chunk(layout: &[ChunkDescriptor], data: &[u8]) -> Vec<u32> {
        let mut pos = 0usize;
        layout
            .iter()
            .map(|c| {
                let crc = crc32fast::hash(&data[pos..pos + c.length as usize]);
                pos += c.length as usize;
                crc
            })
            .collect()
    }

    fn local_commit_line(uri: &str, generation: Option<u64>, extra: &str) -> String {
        let generation = generation.map_or(String::new(), |g| format!(r#","generation":{g}"#));
        format!(r#"{{"action":"commit_write","uri":"{uri}","local_only":true{generation}{extra}}}"#)
    }

    #[tokio::test]
    async fn test_local_only_commit_refuses_a_layout_staged_by_someone_else() {
        // Two writers of one key: B re-stages while A is writing. A's commit must not publish B's
        // layout, whose data may not be written yet (durable before visible).
        let (server, dir) = nvme_server("localgen", 16, |_| {});
        let uri = "hypercache://kv_cache/model/0/0/_/gen.kv";
        let t = Instant::now();
        let data = vec![3u8; 8192];
        let a = server.handle_stage_write(uri, 8192, None, false, false, false, t).await;
        write_staged(&server, a.chunk_layout.as_ref().unwrap(), &data).await;
        let b = server.handle_stage_write(uri, 8192, None, false, false, false, t).await;
        assert_ne!(a.generation, b.generation);
        let crcs = crc_per_chunk(a.chunk_layout.as_ref().unwrap(), &data);

        let line = local_commit_line(uri, a.generation, &format!(r#","chunk_checksums":{crcs:?}"#));
        let stale = server.handle_request_line(&line).await;
        assert_eq!(stale.status, crate::protocol::ResponseStatus::Error);
        assert!(stale.error_message.unwrap().contains("replaced since it was staged"));
        let meta = server.metadata().get_layout(uri).await.unwrap().unwrap();
        assert_eq!(
            meta.sync_status,
            Some(SyncStatus::Staged),
            "B's layout stays unpublished"
        );
        assert_eq!(Some(meta.generation), b.generation);

        // B, the current writer, can still commit its own layout
        write_staged(&server, b.chunk_layout.as_ref().unwrap(), &data).await;
        let line = local_commit_line(uri, b.generation, &format!(r#","chunk_checksums":{crcs:?}"#));
        let ok = server.handle_request_line(&line).await;
        assert_eq!(ok.status, crate::protocol::ResponseStatus::WriteCommitted);
        assert_eq!(ok.sync_status, Some(SyncStatus::LocalOnly));
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_local_only_commit_never_turns_a_dirty_object_local() {
        // An object waiting for write-back must keep its flush: a local-only commit of it is refused
        let (server, dir) = nvme_server("localdirty", 16, |config| {
            config.authoritative_backend = "nfs".to_string();
        });
        let uri = "hypercache://checkpoints/step1.bin";
        let t = Instant::now();
        let staged = server.handle_stage_write(uri, 4096, None, false, false, false, t).await;
        write_staged(&server, &staged.chunk_layout.unwrap(), &[9u8; 4096]).await;
        let generation = staged.generation.unwrap();
        // Committed and still dirty, as it is while write-back runs or after it failed
        assert!(
            server
                .metadata()
                .update_sync_status_if_generation(uri, generation, SyncStatus::SyncFailed)
                .await
        );

        for line in [
            local_commit_line(uri, None, ""),
            local_commit_line(uri, Some(generation), ""),
        ] {
            let refused = server.handle_request_line(&line).await;
            assert_eq!(refused.status, crate::protocol::ResponseStatus::Error);
            assert!(refused.error_message.unwrap().contains("not STAGED"));
        }
        let meta = server.metadata().get_layout(uri).await.unwrap().unwrap();
        assert_eq!(meta.sync_status, Some(SyncStatus::SyncFailed));
        assert!(meta.is_dirty(), "the object must still be flushed and not evictable");
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_local_only_commit_verifies_a_requested_sha256() {
        use sha2::{Digest, Sha256};
        let (server, dir) = nvme_server("localsha", 16, |_| {});
        let t = Instant::now();
        let data: Vec<u8> = (0..(1024 * 1024 + 4096)).map(|i| (i % 253) as u8).collect();
        let digest = hex::encode(Sha256::digest(&data));

        // A torn write: the staged data isn't what the client meant to write
        let torn = "hypercache://kv_cache/model/0/0/_/torn.kv";
        let staged = server
            .handle_stage_write(torn, data.len() as u64, None, false, false, false, t)
            .await;
        let mut wrong = data.clone();
        wrong[100] ^= 0xff;
        write_staged(&server, &staged.chunk_layout.unwrap(), &wrong).await;
        let line = local_commit_line(torn, staged.generation, &format!(r#","sha256":"{digest}""#));
        let refused = server.handle_request_line(&line).await;
        assert_eq!(refused.status, crate::protocol::ResponseStatus::Error);
        assert!(refused.error_message.unwrap().contains("SHA-256 mismatch"));
        assert_eq!(
            server.metadata().get_sync_status(torn).await.unwrap(),
            Some(SyncStatus::Staged)
        );

        // Client CRCs that disagree with the device are refused when a digest is checked
        let good = "hypercache://kv_cache/model/0/0/_/good.kv";
        let staged = server
            .handle_stage_write(good, data.len() as u64, None, false, false, false, t)
            .await;
        let layout = staged.chunk_layout.unwrap();
        write_staged(&server, &layout, &data).await;
        let crcs = crc_per_chunk(&layout, &data);
        let bad_crcs: Vec<u32> = crcs.iter().map(|c| c ^ 1).collect();
        let line = local_commit_line(
            good,
            staged.generation,
            &format!(r#","sha256":"{digest}","chunk_checksums":{bad_crcs:?}"#),
        );
        let refused = server.handle_request_line(&line).await;
        assert_eq!(refused.status, crate::protocol::ResponseStatus::Error);
        assert!(refused.error_message.unwrap().contains("Checksum mismatch"));

        // The right digest commits and records the daemon's own CRCs
        let line = local_commit_line(good, staged.generation, &format!(r#","sha256":"{digest}""#));
        let ok = server.handle_request_line(&line).await;
        assert_eq!(ok.status, crate::protocol::ResponseStatus::WriteCommitted);
        let recorded: Vec<u32> = server
            .handle_locate(good, false, None, false, t)
            .await
            .chunk_layout
            .unwrap()
            .iter()
            .map(|c| c.checksum.unwrap())
            .collect();
        assert_eq!(recorded, crcs);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[tokio::test]
    async fn test_commit_verifies_sha256_and_records_checksums() {
        use sha2::{Digest, Sha256};
        let (server, dir) = nvme_server("sha", 16, |_| {});
        let uri = "hypercache://checkpoints/sha.pt";
        let data: Vec<u8> = (0..(2 * 1024 * 1024 + 123)).map(|i| (i % 251) as u8).collect();
        let t = Instant::now();

        let staged = server
            .handle_stage_write(uri, data.len() as u64, None, true, false, false, t)
            .await;
        let layout = staged.chunk_layout.unwrap();
        write_staged(&server, &layout, &data).await;

        // Wrong digest is rejected and the object is not marked committed
        let bad = server.handle_commit_write(uri, Some("00".repeat(32)), false, t).await;
        assert_eq!(bad.status, crate::protocol::ResponseStatus::Error);
        assert!(bad.error_message.unwrap().contains("SHA-256 mismatch"));
        assert_eq!(
            server.metadata().get_sync_status(uri).await.unwrap(),
            Some(SyncStatus::Staged)
        );

        // Correct digest commits and stores per-chunk CRC32s
        let digest = hex::encode(Sha256::digest(&data));
        // Use write-through to the NFS root so no background task outlives the test
        let nfs_uri = format!("file://{}/sha.pt", dir.display());
        let staged = server
            .handle_stage_write(&nfs_uri, data.len() as u64, None, true, false, false, t)
            .await;
        let layout = staged.chunk_layout.unwrap();
        write_staged(&server, &layout, &data).await;
        let ok = server
            .handle_commit_write(&nfs_uri, Some(digest.to_uppercase()), true, t)
            .await;
        assert_eq!(
            ok.status,
            crate::protocol::ResponseStatus::WriteCommitted,
            "{:?}",
            ok.error_message
        );
        let meta = server.metadata().get_layout(&nfs_uri).await.unwrap().unwrap();
        assert_eq!(meta.chunk_layout.len(), 3);
        for (i, c) in meta.chunk_layout.iter().enumerate() {
            let start = i * 1024 * 1024;
            let end = (start + c.length as usize).min(data.len());
            assert_eq!(c.checksum, Some(crc32fast::hash(&data[start..end])));
        }
        assert_eq!(std::fs::read(dir.join("sha.pt")).unwrap(), data);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_resume_pending_writebacks_after_restart() {
        let (server, dir) = nvme_server("resume", 16, |_| {});
        let uri = format!("file://{}/resumed.pt", dir.display());
        let data = vec![7u8; 1500 * 1024];
        let t = Instant::now();

        let staged = server
            .handle_stage_write(&uri, data.len() as u64, None, true, false, false, t)
            .await;
        write_staged(&server, &staged.chunk_layout.unwrap(), &data).await;
        // Simulate a crash right after commit: status COMMITTED, no flush happened yet
        server
            .metadata()
            .update_sync_status(&uri, SyncStatus::Committed)
            .await
            .unwrap();

        assert_eq!(server.resume_pending_writebacks().await, 1);
        let mut synced = false;
        for _ in 0..200 {
            if server.metadata().get_sync_status(&uri).await.unwrap() == Some(SyncStatus::SyncedBackend) {
                synced = true;
                break;
            }
            tokio::time::sleep(Duration::from_millis(20)).await;
        }
        assert!(synced, "write-back was not resumed");
        assert_eq!(std::fs::read(dir.join("resumed.pt")).unwrap(), data);
        assert_eq!(server.resume_pending_writebacks().await, 0);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_failed_write_through_marks_sync_failed() {
        let (server, dir) = nvme_server("wtfail", 16, |_| {});
        // Outside the NFS root: flush is refused, the object must stay dirty
        let uri = "file:///hypercache_forbidden_dir/x.pt";
        let t = Instant::now();
        let staged = server.handle_stage_write(uri, 4096, None, true, false, false, t).await;
        assert_eq!(staged.status, crate::protocol::ResponseStatus::WriteStaged);
        let resp = server.handle_commit_write(uri, None, true, t).await;
        assert_eq!(resp.status, crate::protocol::ResponseStatus::Error);
        assert_eq!(
            server.metadata().get_sync_status(uri).await.unwrap(),
            Some(SyncStatus::SyncFailed)
        );
        assert_eq!(server.metadata().list_dirty_entries().await.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_preload_batch_reports_failures() {
        let (server, dir) = nvme_server("batch", 16, |_| {});
        std::fs::write(dir.join("a.bin"), vec![1u8; 10_000]).unwrap();
        let uris = vec![
            format!("file://{}/a.bin", dir.display()),
            format!("file://{}/missing.bin", dir.display()),
        ];
        let resp = server.handle_preload_batch(uris.clone(), None, Instant::now()).await;
        assert_eq!(resp.ingested_count, Some(1));
        assert_eq!(resp.failed_uris, Some(vec![uris[1].clone()]));
        assert_eq!(resp.total_bytes_used, Some(10_000));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[tokio::test]
    async fn test_uds_round_trip_and_oversized_request() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        let (server, dir) = nvme_server("uds", 16, |_| {});
        let sock = dir.join("s").join("hc.sock");
        // Any group the test user belongs to works as client_gid; the dir's own gid is one
        let my_gid = std::fs::metadata(&dir).unwrap().gid();
        let server = Arc::new(Server {
            config: Config {
                socket_path: sock.to_string_lossy().to_string(),
                client_gid: Some(my_gid),
                ..server.config.clone()
            },
            ..Arc::try_unwrap(Arc::new(server)).ok().unwrap()
        });
        tokio::spawn(Arc::clone(&server).run());
        for _ in 0..100 {
            if sock.exists() {
                break;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }

        let stream = UnixStream::connect(&sock).await.unwrap();
        let (r, mut w) = stream.into_split();
        let mut lines = BufReader::new(r).lines();
        // Two pipelined requests in one write get two responses in order
        w.write_all(b"{\"action\":\"ping\"}\n\n{\"action\":\"list\"}\n")
            .await
            .unwrap();
        let first: Response = serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(first.status, crate::protocol::ResponseStatus::Pong);
        let second: Response = serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(second.status, crate::protocol::ResponseStatus::List);

        // client_gid opens the socket to the clients' group (connect needs write permission)
        let sock_meta = std::fs::metadata(&sock).unwrap();
        assert_eq!(sock_meta.gid(), my_gid);
        assert_eq!(sock_meta.permissions().mode() & 0o777, 0o660);

        w.write_all(b"not json\n").await.unwrap();
        let err: Response = serde_json::from_str(&lines.next_line().await.unwrap().unwrap()).unwrap();
        assert_eq!(err.status, crate::protocol::ResponseStatus::Error);

        // An unbounded line is cut off with an error instead of growing daemon memory
        let big = vec![b'x'; MAX_REQUEST_BYTES + 10];
        let _ = w.write_all(&big).await;
        let line = lines.next_line().await.unwrap().unwrap();
        let resp: Response = serde_json::from_str(&line).unwrap();
        assert!(resp.error_message.unwrap().contains("limit"));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn test_socket_dir_permissions_are_tightened() {
        use std::os::unix::fs::PermissionsExt;
        let dir = std::env::temp_dir().join(format!("hypercache_sockdir_{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::set_permissions(&dir, std::fs::Permissions::from_mode(0o777)).unwrap();
        secure_socket_dir(&dir).unwrap();
        let mode = std::fs::metadata(&dir).unwrap().permissions().mode() & 0o777;
        assert_eq!(mode & 0o022, 0);
        let _ = std::fs::remove_dir_all(&dir);
    }
}
