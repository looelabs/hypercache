use anyhow::{Context, Result};
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::net::{UnixListener, UnixStream};
use tracing::{debug, error, info, warn};

use crate::config::Config;
use crate::discovery::DiscoveredDevice;
use crate::ingestion::IngestionEngine;
use crate::metadata::MetadataStore;
use crate::protocol::{Request, Response};
use crate::telemetry::{TelemetryCollector, TelemetryEvent};

pub struct Server {
    config: Config,
    metadata: MetadataStore,
    ingestion: IngestionEngine,
    telemetry: TelemetryCollector,
    devices: Vec<DiscoveredDevice>,
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

        if let Some(parent) = sock_path.parent() {
            let _ = fs::create_dir_all(parent);
        }

        let listener = UnixListener::bind(sock_path)
            .with_context(|| format!("Failed to bind UDS listener at {:?}", sock_path))?;

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
        let (reader, mut writer) = stream.into_split();
        let mut buf_reader = BufReader::new(reader);
        let mut line = String::new();

        loop {
            line.clear();
            let bytes_read = buf_reader.read_line(&mut line).await?;
            if bytes_read == 0 {
                break; // EOF
            }

            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }

            let t_start = Instant::now();
            let req: Result<Request, _> = serde_json::from_str(trimmed);

            let resp = match req {
                Ok(Request::Ping) => Response::pong(),
                Ok(Request::GetDevices) => Response::devices(
                    self.devices.clone(),
                    self.config.ram_cache_enabled,
                    self.config.ram_cache_capacity_bytes,
                ),
                Ok(Request::List) => {
                    let (objects, total_bytes) = self.metadata.list_entries().await;
                    debug!("Listing cached inventory: {} objects ({:.2} GiB)", objects.len(), total_bytes as f64 / (1024.0 * 1024.0 * 1024.0));
                    Response::cached_list(objects, total_bytes, self.metadata.max_capacity_bytes)
                }
                Ok(Request::Evict { uri }) => {
                    match self.metadata.evict_entry(&uri).await {
                        Ok(Some(freed)) => {
                            info!("Eviction executed for '{}': freed {:.2} GiB from cache index.", uri, freed as f64 / (1024.0 * 1024.0 * 1024.0));
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
                    }
                }
                Ok(Request::Purge) => {
                    match self.metadata.purge_all().await {
                        Ok((count, freed)) => {
                            info!("Cache PURGE/CLEAN executed: cleared {} cached model(s) ({:.2} GiB freed). Reset drive allocation pointers to 4096.",
                                count, freed as f64 / (1024.0 * 1024.0 * 1024.0));
                            Response::purged(count, freed)
                        }
                        Err(e) => {
                            error!("Purge failed: {}", e);
                            Response::error(format!("Purge failed: {}", e))
                        }
                    }
                }
                Ok(Request::Locate { uri, auto_ingest, tier, pinned }) => {
                    self.handle_locate(&uri, auto_ingest, tier, pinned, t_start).await
                }
                Ok(Request::Ingest { uri, tier, pinned, allow_synthetic }) => {
                    self.handle_ingest(&uri, tier, pinned, allow_synthetic, t_start).await
                }
                Ok(Request::StageWrite { uri, size_bytes, tier, is_leader, auto_flush_s3, write_through: _, pinned }) => {
                    self.handle_stage_write(&uri, size_bytes, tier, is_leader, auto_flush_s3, pinned, t_start).await
                }
                Ok(Request::AppendWrite { uri, size_bytes, tier }) => {
                    self.handle_append_write(&uri, size_bytes, tier).await
                }
                Ok(Request::Pin { uri, pinned }) => {
                    match self.metadata.set_pinned(&uri, pinned).await {
                        Ok(true) => Response::pong(),
                        Ok(false) => Response::error(format!("Object '{}' not found in cache", uri)),
                        Err(e) => Response::error(format!("Pin update failed: {}", e)),
                    }
                }
                Ok(Request::CommitWrite { uri, sha256, write_through }) => {
                    self.handle_commit_write(&uri, sha256, write_through, t_start).await
                }
                Ok(Request::GetWriteStatus { uri }) => {
                    self.handle_get_write_status(&uri).await
                }
                Ok(Request::PreloadBatch { uris, tier }) => {
                    self.handle_preload_batch(uris, tier, t_start).await
                }
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
                        timestamp: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
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
                Err(e) => {
                    Response::error(format!("Malformed JSON request: {}", e))
                }
            };

            let mut out = serde_json::to_vec(&resp)?;
            out.push(b'\n');
            writer.write_all(&out).await?;
            writer.flush().await?;
        }

        Ok(())
    }

    async fn handle_locate(&self, uri: &str, auto_ingest: bool, tier: Option<crate::protocol::StorageTier>, pinned: bool, t_start: Instant) -> Response {
        let hit_result = self.metadata.get_layout(uri).await;

        match hit_result {
            Ok(Some(meta)) => {
                if pinned && !meta.pinned {
                    let _ = self.metadata.set_pinned(uri, true).await;
                }
                let duration_us = t_start.elapsed().as_micros() as u64;
                let targets = meta.chunk_layout.iter().map(|c| c.device.clone()).collect();

                self.telemetry.record(TelemetryEvent {
                    timestamp: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
                    event_type: "CACHE_HIT".to_string(),
                    uri: uri.to_string(),
                    object_size_bytes: meta.object_size,
                    duration_us,
                    bandwidth_mbps: 0.0,
                    client_id: "python_client".to_string(),
                    chunk_count: meta.chunk_layout.len() as u32,
                    device_targets: targets,
                    local_meta_us: duration_us,
                    remote_meta_us: 45000, // Estimated baseline
                    local_ttfb_us: duration_us + 100,
                    remote_ttfb_us: 60000,
                    time_saved_ms: 0,
                    cpu_pct: 0.2,
                    status: "HIT".to_string(),
                });

                Response::hit(meta.object_size, meta.chunk_size, meta.chunk_layout)
            }
            Ok(None) => {
                // Cache Miss
                let duration_us = t_start.elapsed().as_micros() as u64;
                self.telemetry.record(TelemetryEvent {
                    timestamp: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
                    event_type: "CACHE_MISS".to_string(),
                    uri: uri.to_string(),
                    object_size_bytes: 0,
                    duration_us,
                    bandwidth_mbps: 0.0,
                    client_id: "python_client".to_string(),
                    chunk_count: 0,
                    device_targets: vec![],
                    local_meta_us: duration_us,
                    remote_meta_us: 45000,
                    local_ttfb_us: 0,
                    remote_ttfb_us: 0,
                    time_saved_ms: 0,
                    cpu_pct: 0.0,
                    status: "MISS".to_string(),
                });

                // Auto ingest if requested
                if auto_ingest {
                    info!("Cache miss for {}, triggering auto-ingestion...", uri);
                    self.handle_ingest(uri, tier, pinned, false, t_start).await
                } else {
                    Response::miss()
                }
            }
            Err(e) => {
                error!("Metadata query error for {}: {}", uri, e);
                Response::error(format!("Metadata query error: {}", e))
            }
        }
    }

    async fn handle_ingest(&self, uri: &str, tier: Option<crate::protocol::StorageTier>, pinned: bool, allow_synthetic: bool, t_start: Instant) -> Response {
        match self.ingestion.ingest_object_full(uri, tier, pinned, allow_synthetic).await {
            Ok((size, chunk_size, layout)) => {
                let duration_us = t_start.elapsed().as_micros() as u64;
                let dur_sec = t_start.elapsed().as_secs_f64();
                let bw_mbps = if dur_sec > 0.0 {
                    (size as f64 / (1024.0 * 1024.0)) / dur_sec
                } else {
                    0.0
                };

                let targets = layout.iter().map(|c| c.device.clone()).collect();

                self.telemetry.record(TelemetryEvent {
                    timestamp: chrono::Utc::now().format("%Y-%m-%d %H:%M:%S%.3f").to_string(),
                    event_type: "INGESTION".to_string(),
                    uri: uri.to_string(),
                    object_size_bytes: size,
                    duration_us,
                    bandwidth_mbps: bw_mbps,
                    client_id: "control_plane".to_string(),
                    chunk_count: layout.len() as u32,
                    device_targets: targets,
                    local_meta_us: 200,
                    remote_meta_us: 45000,
                    local_ttfb_us: 300,
                    remote_ttfb_us: 55000,
                    time_saved_ms: 0,
                    cpu_pct: 0.2,
                    status: "INGESTED".to_string(),
                });

                Response::hit(size, chunk_size, layout)
            }
            Err(e) => {
                error!("Ingestion error for {}: {}", uri, e);
                Response::error(format!("Ingestion failed: {}", e))
            }
        }
    }

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
                if let Err(e) = self
                    .metadata
                    .save_layout_full(
                        uri,
                        size_bytes,
                        chunk_size,
                        layout.clone(),
                        Some("STAGED".to_string()),
                        pinned,
                    )
                    .await
                {
                    error!("Failed to save staged layout for {}: {}", uri, e);
                    return Response::error(format!("Failed to stage metadata: {}", e));
                }

                info!(
                    "Staged write for '{}': {} bytes across {} chunk(s)",
                    uri,
                    size_bytes,
                    layout.len()
                );
                Response::write_staged(size_bytes, chunk_size, layout)
            }
            Err(e) => {
                error!("Allocation failed for stage write on {}: {}", uri, e);
                Response::error(format!("Allocation failed: {}", e))
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
                    Ok((total_size, _full_layout)) => {
                        Response::write_staged(total_size, chunk_size, new_chunks)
                    }
                    Err(e) => Response::error(format!("Failed to append staged layout: {}", e)),
                }
            }
            Err(e) => Response::error(format!("Failed to allocate appended chunks: {}", e)),
        }
    }

    async fn handle_commit_write(&self, uri: &str, _sha256: Option<String>, write_through: bool, t_start: Instant) -> Response {
        match self.metadata.get_layout(uri).await {
            Ok(Some(meta)) => {
                let size = meta.object_size;
                let layout = meta.chunk_layout.clone();
                let is_write_through = write_through || self.config.write_through_enabled;

                if is_write_through {
                    let _ = self.metadata.update_sync_status(uri, "SYNCING_BACKEND").await;
                    info!(
                        "Executing synchronous write-through flush to authoritative backend for '{}' ({} bytes)...",
                        uri, size
                    );
                    match self.ingestion.flush_to_backend(uri, &layout).await {
                        Ok(()) => {
                            let _ = self.metadata.update_sync_status(uri, "SYNCED_BACKEND").await;
                            info!(
                                "Synchronous write-through flush succeeded for '{}' in {:.2} ms.",
                                uri,
                                t_start.elapsed().as_secs_f64() * 1000.0
                            );
                            Response::write_committed(uri.to_string(), size)
                        }
                        Err(e) => {
                            let _ = self.metadata.update_sync_status(uri, "SYNC_FAILED").await;
                            error!("Write-through flush failed for '{}': {}", uri, e);
                            Response::error(format!("Write-through flush to authoritative backend failed: {}", e))
                        }
                    }
                } else {
                    let _ = self.metadata.update_sync_status(uri, "COMMITTED").await;

                    info!(
                        "Committed write for '{}': {} bytes in {:.2} ms. Asynchronous NVMe write-back active.",
                        uri,
                        size,
                        t_start.elapsed().as_secs_f64() * 1000.0
                    );

                    // Spawn background Tokio task to flush to S3 / authoritative backend and optional k=2 peer replica
                    let ingestion_clone = self.ingestion.clone();
                    let metadata_clone = self.metadata.clone();
                    let config_clone = self.config.clone();
                    let uri_str = uri.to_string();

                    tokio::spawn(async move {
                        if config_clone.peer_replication_factor >= 2 {
                            if let Some(ref ctrl_url) = config_clone.controller_url {
                                let _ = Self::replicate_to_peers(
                                    &config_clone,
                                    ctrl_url,
                                    &ingestion_clone,
                                    &uri_str,
                                    &layout,
                                )
                                .await;
                            }
                        }

                        let _ = metadata_clone.update_sync_status(&uri_str, "SYNCING_BACKEND").await;
                        match ingestion_clone.flush_to_backend(&uri_str, &layout).await {
                            Ok(()) => {
                                info!("Successfully synced '{}' to authoritative backend in background.", uri_str);
                                let _ = metadata_clone.update_sync_status(&uri_str, "SYNCED_BACKEND").await;
                            }
                            Err(e) => {
                                warn!("Background sync to authoritative backend failed for '{}': {}", uri_str, e);
                                let _ = metadata_clone.update_sync_status(&uri_str, "SYNC_FAILED").await;
                            }
                        }
                    });

                    Response::write_committed(uri.to_string(), size)
                }
            }
            Ok(None) => Response::error(format!("URI '{}' was not staged prior to commit", uri)),
            Err(e) => Response::error(format!("Metadata lookup failed for commit: {}", e)),
        }
    }

    async fn replicate_to_peers(
        config: &Config,
        controller_url: &str,
        ingestion: &IngestionEngine,
        uri: &str,
        layout: &[crate::protocol::ChunkDescriptor],
    ) -> Result<()> {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;

        let nodes_url = format!("{}/api/v1/nodes", controller_url.trim_end_matches('/'));
        let mut req = client.get(&nodes_url);
        if let Some(ref tok) = config.cluster_auth_token {
            if !tok.trim().is_empty() {
                req = req.bearer_auth(tok.trim());
            }
        }

        let resp = req.send().await?;
        if !resp.status().is_success() {
            return Ok(());
        }

        let summary: crate::controller::FleetSummary = resp.json().await?;
        let candidate_peers: Vec<crate::controller::NodeState> = summary
            .nodes
            .into_iter()
            .filter(|n| n.status == "ONLINE" && n.registration.node_id != config.node_id)
            .collect();

        if candidate_peers.is_empty() {
            return Ok(());
        }

        let payload = ingestion.read_chunks_to_bytes(layout).await?;
        let needed_replicas = (config.peer_replication_factor - 1).min(candidate_peers.len());

        for peer in candidate_peers.iter().take(needed_replicas) {
            let replica_url = format!(
                "http://{}:{}/api/v1/agent/replica",
                peer.registration.address, peer.registration.http_port
            );
            let mut put_req = client
                .post(&replica_url)
                .query(&[("uri", uri)])
                .body(payload.clone());
            if let Some(ref tok) = config.cluster_auth_token {
                if !tok.trim().is_empty() {
                    put_req = put_req.bearer_auth(tok.trim());
                }
            }
            if let Ok(r) = put_req.send().await {
                if r.status().is_success() {
                    info!(
                        "Replicated checkpoint '{}' ({} bytes) to peer node '{}'",
                        uri,
                        payload.len(),
                        peer.registration.node_id
                    );
                }
            }
        }

        Ok(())
    }

    async fn handle_get_write_status(&self, uri: &str) -> Response {
        match self.metadata.get_layout(uri).await {
            Ok(Some(meta)) => {
                let status = meta.sync_status.unwrap_or_else(|| "COMMITTED".to_string());
                let progress = match status.as_str() {
                    "STAGED" => 25.0,
                    "COMMITTED" => 50.0,
                    "SYNCING_BACKEND" => 75.0,
                    "SYNCED_BACKEND" => 100.0,
                    _ => 100.0,
                };
                Response::write_status(status, progress)
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

        let mut total_bytes = 0u64;
        let mut success_count = 0;

        for uri in &uris {
            match self.ingestion.ingest_object_with_tier(uri, tier).await {
                Ok((size, _, _)) => {
                    total_bytes += size;
                    success_count += 1;
                }
                Err(e) => {
                    warn!("Failed to ingest '{}' in batch: {}", uri, e);
                }
            }
        }

        let dur_ms = t_start.elapsed().as_millis() as u64;
        info!(
            "Batch preload completed: {}/{} successful ({:.2} MiB in {} ms)",
            success_count,
            count,
            total_bytes as f64 / (1024.0 * 1024.0),
            dur_ms
        );

        Response::pong()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::DiscoveredDevice;
    use crate::protocol::StorageTier;

    #[tokio::test]
    async fn test_server_stage_write_and_commit_flow() {
        let test_dir = std::env::temp_dir().join(format!("hypercache_srv_test_{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)));
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
        let devices = vec![
            DiscoveredDevice {
                path: raw_dev0.to_str().unwrap().to_string(),
                is_nvme: true,
                is_virtual: false,
                is_partition: false,
                is_rotational: false,
                size_bytes: 10 * 1024 * 1024,
                model: Some("TestNVMe".to_string()),
                existing_fs: None,
                numa_node: Some(0),
            }
        ];

        let server = Server::new(config, metadata, ingestion, telemetry, devices);

        let uri = "hypercache://checkpoints/epoch_1.pt";
        let t_start = Instant::now();

        // 1. Stage write 8 MiB (2 chunks of 4 MiB)
        let resp_stage = server.handle_stage_write(uri, 8388608, Some(StorageTier::Nvme), true, true, false, t_start).await;
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
        let resp_stage = server.handle_stage_write(uri, 8388608, Some(StorageTier::Ram), true, true, false, t_start).await;
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
        let resp_stage = server.handle_stage_write(&uri, 4194304, Some(StorageTier::Nvme), true, true, false, t_start).await;
        assert_eq!(resp_stage.status, crate::protocol::ResponseStatus::WriteStaged);

        // 2. Commit write with write_through = true
        let resp_commit = server.handle_commit_write(&uri, None, true, t_start).await;
        assert_eq!(resp_commit.status, crate::protocol::ResponseStatus::WriteCommitted);

        // 3. Verify target authoritative file was created synchronously
        assert!(target_file.exists());
        assert_eq!(target_file.metadata().unwrap().len(), 4194304);

        // 4. Verify sync status is immediately SYNCED_BACKEND
        let status = metadata.get_sync_status(&uri).await.unwrap();
        assert_eq!(status, Some("SYNCED_BACKEND".to_string()));

        let _ = std::fs::remove_dir_all(&test_dir);
    }
}
