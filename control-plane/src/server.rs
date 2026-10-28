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
use crate::protocol::{ChunkDescriptor, Request, Response, SyncStatus};
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
            Ok(Request::Purge) => match self.metadata.purge_all().await {
                Ok((count, freed)) => {
                    info!(
                        "Cache PURGE/CLEAN executed: cleared {} cached model(s) ({:.2} GiB freed). Allocation restarts at offset {}.",
                        count,
                        freed as f64 / (1024.0 * 1024.0 * 1024.0),
                        self.config.device_start_offset_bytes
                    );
                    Response::purged(count, freed)
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
            }) => self.handle_commit_write(&uri, sha256, write_through, t_start).await,
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
            Ok(Some(meta)) => {
                if pinned && !meta.pinned {
                    let _ = self.metadata.set_pinned(uri, true).await;
                }
                self.record_event("CACHE_HIT", "HIT", uri, meta.object_size, &meta.chunk_layout, t_start);
                Response::hit(meta.object_size, meta.chunk_size, meta.chunk_layout).with_generation(meta.generation)
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
        assert_eq!(layout[0].device, raw_dev0.to_str().unwrap());
        assert_eq!(layout[1].device, raw_dev1.to_str().unwrap());

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

        // 2. Commit write (write-back)
        let resp_commit = server.handle_commit_write(uri, None, false, t_start).await;
        assert_eq!(resp_commit.status, crate::protocol::ResponseStatus::WriteCommitted);

        // 3. Locate object -> Hit in RAM
        let resp_locate = server.handle_locate(uri, false, None, false, t_start).await;
        assert_eq!(resp_locate.status, crate::protocol::ResponseStatus::Hit);
        assert_eq!(resp_locate.object_size, Some(8388608));

        let _ = std::fs::remove_dir_all(&test_dir);
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
        use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
        let (server, dir) = nvme_server("uds", 16, |_| {});
        let sock = dir.join("s").join("hc.sock");
        let server = Arc::new(Server {
            config: Config {
                socket_path: sock.to_string_lossy().to_string(),
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
