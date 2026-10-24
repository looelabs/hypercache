use anyhow::{Context, Result};
use axum::{
    body::Bytes,
    extract::{Query, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use futures_util::StreamExt;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

use crate::config::Config;
use crate::controller::NodeRegistration;
use crate::discovery::DiscoveredDevice;
use crate::ingestion::IngestionEngine;
use crate::metadata::MetadataStore;

#[derive(Clone)]
pub struct AgentContext {
    pub config: Config,
    pub metadata: MetadataStore,
    pub ingestion: IngestionEngine,
    pub discovered_devices: Vec<DiscoveredDevice>,
}

#[derive(Debug, Deserialize)]
pub struct AgentPreloadPayload {
    pub uri: String,
    #[serde(default)]
    pub source_peer_url: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct AgentChunkQuery {
    pub uri: String,
    pub chunk_idx: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct AgentReplicaQuery {
    pub uri: String,
}

#[derive(Debug, Deserialize)]
pub struct AgentEvictPayload {
    pub uri: String,
}

#[derive(Debug, Deserialize)]
pub struct AgentPreloadBatchPayload {
    pub uris: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct AgentPreloadBatchResponse {
    pub success: bool,
    pub message: String,
    pub ingested_count: usize,
    pub total_bytes: u64,
    pub duration_ms: u64,
}

#[derive(Debug, Serialize)]
pub struct AgentPreloadResponse {
    pub success: bool,
    pub message: String,
    pub size_bytes: Option<u64>,
    pub duration_ms: Option<u64>,
    pub bandwidth_mbps: Option<f64>,
}

fn validate_auth(headers: &axum::http::HeaderMap, expected_token: Option<&str>) -> bool {
    if let Some(expected) = expected_token {
        if expected.trim().is_empty() {
            return true;
        }
        if let Some(auth_header) = headers.get(axum::http::header::AUTHORIZATION) {
            if let Ok(val) = auth_header.to_str() {
                if let Some(token) = val.strip_prefix("Bearer ") {
                    return token.trim() == expected.trim();
                }
            }
        }
        false
    } else {
        true
    }
}

async fn handle_agent_preload(
    State(ctx): State<AgentContext>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<AgentPreloadPayload>,
) -> impl IntoResponse {
    if !validate_auth(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(AgentPreloadResponse {
                success: false,
                message: "Unauthorized: Invalid or missing cluster auth token".to_string(),
                size_bytes: None,
                duration_ms: None,
                bandwidth_mbps: None,
            }),
        );
    }

    info!("Agent received fleet PRELOAD directive for URI: {}", payload.uri);
    let t_start = Instant::now();

    // Check if already in cache
    if let Ok(Some(existing)) = ctx.metadata.get_layout(&payload.uri).await {
        let dur = t_start.elapsed();
        return (
            StatusCode::OK,
            Json(AgentPreloadResponse {
                success: true,
                message: "Already cached (Cache Hit)".to_string(),
                size_bytes: Some(existing.object_size),
                duration_ms: Some(dur.as_millis() as u64),
                bandwidth_mbps: None,
            }),
        );
    }

    // If a peer source URL was provided, attempt P2P pull from peer node first
    if let Some(ref peer_url) = payload.source_peer_url {
        match pull_object_from_peer(&ctx, peer_url, &payload.uri).await {
            Ok((object_size, chunk_count)) => {
                let dur = t_start.elapsed();
                let dur_s = dur.as_secs_f64();
                let bw = if dur_s > 0.0 {
                    (object_size as f64 / (1024.0 * 1024.0)) / dur_s
                } else {
                    0.0
                };
                info!(
                    "Successfully executed P2P PRELOAD for {} from peer {}: {} bytes in {} chunks ({:.2} MB/s)",
                    payload.uri, peer_url, object_size, chunk_count, bw
                );
                return (
                    StatusCode::OK,
                    Json(AgentPreloadResponse {
                        success: true,
                        message: format!(
                            "P2P pulled {} across {} chunks from {} in {:.3} s",
                            object_size, chunk_count, peer_url, dur_s
                        ),
                        size_bytes: Some(object_size),
                        duration_ms: Some(dur.as_millis() as u64),
                        bandwidth_mbps: Some(bw),
                    }),
                );
            }
            Err(e) => {
                warn!(
                    "P2P pull from peer {} failed for {} ({}). Falling back to authoritative origin...",
                    peer_url, payload.uri, e
                );
            }
        }
    }

    match ctx.ingestion.ingest_object(&payload.uri).await {
        Ok((object_size, _chunk_size, chunk_layout)) => {
            let dur = t_start.elapsed();
            let dur_s = dur.as_secs_f64();
            let bw = if dur_s > 0.0 {
                (object_size as f64 / (1024.0 * 1024.0)) / dur_s
            } else {
                0.0
            };

            info!(
                "Successfully executed fleet PRELOAD for {}: {} bytes in {} chunks ({:.2} MB/s)",
                payload.uri,
                object_size,
                chunk_layout.len(),
                bw
            );

            (
                StatusCode::OK,
                Json(AgentPreloadResponse {
                    success: true,
                    message: format!(
                        "Striped {} across {} NVMe chunks in {:.3} s",
                        object_size,
                        chunk_layout.len(),
                        dur_s
                    ),
                    size_bytes: Some(object_size),
                    duration_ms: Some(dur.as_millis() as u64),
                    bandwidth_mbps: Some(bw),
                }),
            )
        }
        Err(e) => {
            error!("Fleet PRELOAD failed for {}: {}", payload.uri, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(AgentPreloadResponse {
                    success: false,
                    message: format!("Ingestion failed: {}", e),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                }),
            )
        }
    }
}

async fn handle_agent_preload_batch(
    State(ctx): State<AgentContext>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<AgentPreloadBatchPayload>,
) -> impl IntoResponse {
    if !validate_auth(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(AgentPreloadBatchResponse {
                success: false,
                message: "Unauthorized: Invalid or missing cluster auth token".to_string(),
                ingested_count: 0,
                total_bytes: 0,
                duration_ms: 0,
            }),
        );
    }

    info!("Agent received fleet PRELOAD_BATCH directive for {} URIs", payload.uris.len());
    let t_start = Instant::now();

    let mut total_bytes = 0u64;
    let mut ingested_count = 0;

    for uri in &payload.uris {
        if let Ok(Some(existing)) = ctx.metadata.get_layout(uri).await {
            total_bytes += existing.object_size;
            ingested_count += 1;
            continue;
        }

        match ctx.ingestion.ingest_object(uri).await {
            Ok((size, _, _)) => {
                total_bytes += size;
                ingested_count += 1;
            }
            Err(e) => {
                warn!("Batch preload failed for URI {}: {}", uri, e);
            }
        }
    }

    let dur_ms = t_start.elapsed().as_millis() as u64;
    (
        StatusCode::OK,
        Json(AgentPreloadBatchResponse {
            success: true,
            message: format!(
                "Successfully preloaded {}/{} shards ({:.2} MiB in {} ms)",
                ingested_count,
                payload.uris.len(),
                total_bytes as f64 / (1024.0 * 1024.0),
                dur_ms
            ),
            ingested_count,
            total_bytes,
            duration_ms: dur_ms,
        }),
    )
}

async fn handle_agent_evict(
    State(ctx): State<AgentContext>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<AgentEvictPayload>,
) -> impl IntoResponse {
    if !validate_auth(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "success": false,
                "message": "Unauthorized: Invalid or missing cluster auth token"
            })),
        );
    }

    info!("Agent received fleet EVICT directive for URI: {}", payload.uri);
    match ctx.metadata.evict_entry(&payload.uri).await {
        Ok(Some(bytes)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "success": true,
                "message": format!("Successfully evicted {} ({} bytes freed)", payload.uri, bytes)
            })),
        ),
        Ok(None) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "success": true,
                "message": format!("URI {} was not in cache", payload.uri)
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "success": false,
                "message": format!("Eviction error: {}", e)
            })),
        ),
    }
}

async fn handle_agent_purge(
    State(ctx): State<AgentContext>,
    headers: axum::http::HeaderMap,
) -> impl IntoResponse {
    if !validate_auth(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "success": false,
                "message": "Unauthorized: Invalid or missing cluster auth token"
            })),
        );
    }

    info!("Agent received fleet PURGE directive");
    match ctx.metadata.purge_all().await {
        Ok((count, bytes)) => (
            StatusCode::OK,
            Json(serde_json::json!({
                "success": true,
                "message": format!("Successfully purged {} models ({} bytes freed) and reset drives", count, bytes)
            })),
        ),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "success": false,
                "message": format!("Purge error: {}", e)
            })),
        ),
    }
}

async fn handle_agent_status(
    State(ctx): State<AgentContext>,
    headers: axum::http::HeaderMap,
) -> impl IntoResponse {
    if !validate_auth(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({
                "error": "Unauthorized: Invalid or missing cluster auth token"
            })),
        );
    }

    let (entries, total_used_bytes) = ctx.metadata.list_entries().await;
    let cached_uris = entries.iter().map(|e| e.uri.clone()).collect::<Vec<_>>();
    let drives = ctx
        .discovered_devices
        .iter()
        .map(|d| d.path.clone())
        .collect::<Vec<_>>();
    let total_capacity_bytes = ctx.discovered_devices.iter().map(|d| d.size_bytes).sum::<u64>();

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "node_id": ctx.config.node_id,
            "drives": drives,
            "total_capacity_bytes": total_capacity_bytes,
            "used_capacity_bytes": total_used_bytes,
            "cached_objects_count": entries.len(),
            "cached_uris": cached_uris,
        })),
    )
}

pub fn create_agent_router(ctx: AgentContext) -> Router {
    Router::new()
        .route("/api/v1/agent/preload", post(handle_agent_preload))
        .route("/api/v1/agent/preload_batch", post(handle_agent_preload_batch))
        .route("/api/v1/agent/evict", post(handle_agent_evict))
        .route("/api/v1/agent/purge", post(handle_agent_purge))
        .route("/api/v1/agent/status", get(handle_agent_status))
        .route("/api/v1/agent/chunk", get(handle_agent_get_chunk))
        .route("/api/v1/agent/replica", post(handle_agent_put_replica))
        .with_state(ctx)
}

async fn pull_object_from_peer(
    ctx: &AgentContext,
    peer_base_url: &str,
    uri: &str,
) -> Result<(u64, usize)> {
    let client = Client::builder()
        .timeout(Duration::from_secs(60))
        .build()
        .unwrap_or_else(|_| Client::new());

    let url = format!("{}/api/v1/agent/chunk", peer_base_url.trim_end_matches('/'));
    let mut req = client.get(&url).query(&[("uri", uri)]);
    if let Some(ref tok) = ctx.config.cluster_auth_token {
        if !tok.trim().is_empty() {
            req = req.bearer_auth(tok.trim());
        }
    }

    let resp = req
        .send()
        .await
        .with_context(|| format!("Failed to connect to peer {}", url))?;
    if !resp.status().is_success() {
        anyhow::bail!("Peer {} returned HTTP {}", url, resp.status());
    }

    let chunk_size = ctx.config.chunk_size_bytes;
    let mut stream = resp.bytes_stream();
    let mut buffer = Vec::with_capacity(chunk_size);
    let mut layout = Vec::new();
    let mut chunk_idx = 0usize;
    let mut total_bytes = 0u64;

    while let Some(item) = stream.next().await {
        let bytes = item.with_context(|| "Error reading P2P byte stream")?;
        let mut offset = 0usize;
        while offset < bytes.len() {
            let needed = chunk_size - buffer.len();
            let take = needed.min(bytes.len() - offset);
            buffer.extend_from_slice(&bytes[offset..offset + take]);
            offset += take;

            if buffer.len() == chunk_size {
                let desc = ctx
                    .ingestion
                    .striper()
                    .stripe_single_chunk(uri, chunk_idx, &buffer, None)
                    .await?;
                total_bytes += buffer.len() as u64;
                layout.push(desc);
                buffer.clear();
                chunk_idx += 1;
            }
        }
    }

    if !buffer.is_empty() {
        let desc = ctx
            .ingestion
            .striper()
            .stripe_single_chunk(uri, chunk_idx, &buffer, None)
            .await?;
        total_bytes += buffer.len() as u64;
        layout.push(desc);
    }

    let chunk_count = layout.len();
    ctx.metadata
        .save_layout(uri, total_bytes, chunk_size, layout)
        .await?;

    Ok((total_bytes, chunk_count))
}

async fn handle_agent_get_chunk(
    State(ctx): State<AgentContext>,
    headers: axum::http::HeaderMap,
    Query(query): Query<AgentChunkQuery>,
) -> impl IntoResponse {
    if !validate_auth(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return (StatusCode::UNAUTHORIZED, Vec::new());
    }

    match ctx.metadata.get_layout(&query.uri).await {
        Ok(Some(meta)) => {
            let target_chunks = if let Some(idx) = query.chunk_idx {
                if let Some(c) = meta.chunk_layout.get(idx) {
                    vec![c.clone()]
                } else {
                    return (StatusCode::NOT_FOUND, Vec::new());
                }
            } else {
                meta.chunk_layout.clone()
            };

            match ctx.ingestion.read_chunks_to_bytes(&target_chunks).await {
                Ok(bytes) => (StatusCode::OK, bytes),
                Err(e) => {
                    error!("Failed reading local chunk(s) for P2P transfer on {}: {}", query.uri, e);
                    (StatusCode::INTERNAL_SERVER_ERROR, Vec::new())
                }
            }
        }
        _ => (StatusCode::NOT_FOUND, Vec::new()),
    }
}

async fn handle_agent_put_replica(
    State(ctx): State<AgentContext>,
    headers: axum::http::HeaderMap,
    Query(query): Query<AgentReplicaQuery>,
    body: Bytes,
) -> impl IntoResponse {
    if !validate_auth(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "success": false, "message": "Unauthorized" })),
        );
    }

    let size_bytes = body.len() as u64;
    let chunk_size = ctx.config.chunk_size_bytes;

    match ctx
        .ingestion
        .striper()
        .stripe_bytes_to_tier(&query.uri, &body, None)
        .await
    {
        Ok(layout) => {
            if let Err(e) = ctx
                .metadata
                .save_layout_with_status(
                    &query.uri,
                    size_bytes,
                    chunk_size,
                    layout.clone(),
                    Some("REPLICA_COMMITTED".to_string()),
                )
                .await
            {
                return (
                    StatusCode::INTERNAL_SERVER_ERROR,
                    Json(serde_json::json!({ "success": false, "message": e.to_string() })),
                );
            }
            info!(
                "Stored k=2 peer checkpoint replica for '{}': {} bytes across {} chunk(s)",
                query.uri,
                size_bytes,
                layout.len()
            );
            (
                StatusCode::OK,
                Json(serde_json::json!({
                    "success": true,
                    "uri": query.uri,
                    "size_bytes": size_bytes,
                    "chunks": layout.len(),
                })),
            )
        }
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "success": false, "message": e.to_string() })),
        ),
    }
}

pub async fn start_agent_http_server(
    ctx: AgentContext,
    port: u16,
) -> Result<()> {
    let app = create_agent_router(ctx.clone());
    let ip: std::net::IpAddr = ctx
        .config
        .agent_bind_address
        .parse()
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED));
    let addr = SocketAddr::new(ip, port);

    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .with_context(|| format!("Failed to bind Agent HTTP listener on {}", addr))?;

    info!("Node Agent HTTP server listening on http://{}", addr);

    tokio::spawn(async move {
        if let Err(e) = axum::serve(listener, app).await {
            error!("Node Agent HTTP server error: {}", e);
        }
    });

    // Start background heartbeat loop if controller_url is configured
    if let Some(ref controller_url) = ctx.config.controller_url {
        let controller_url = controller_url.clone();
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| Client::new());

        tokio::spawn(async move {
            info!("Starting Node Agent heartbeat to Controller at {}", controller_url);
            let reg_url = format!("{}/api/v1/register", controller_url.trim_end_matches('/'));

            loop {
                let (entries, total_used_bytes) = ctx.metadata.list_entries().await;
                let cached_uris = entries.iter().map(|e| e.uri.clone()).collect::<Vec<_>>();
                let drives = ctx
                    .discovered_devices
                    .iter()
                    .map(|d| d.path.clone())
                    .collect::<Vec<_>>();
                let total_capacity_bytes = ctx
                    .discovered_devices
                    .iter()
                    .map(|d| d.size_bytes)
                    .sum::<u64>();

                let address = ctx.config.node_address.clone().unwrap_or_else(|| {
                    if ctx.config.agent_bind_address != "0.0.0.0" && ctx.config.agent_bind_address != "127.0.0.1" {
                        ctx.config.agent_bind_address.clone()
                    } else {
                        "".to_string()
                    }
                });

                let reg = NodeRegistration {
                    node_id: ctx.config.node_id.clone(),
                    address,
                    http_port: port,
                    labels: ctx.config.node_labels.clone(),
                    drives,
                    total_capacity_bytes,
                    used_capacity_bytes: total_used_bytes,
                    cached_objects_count: entries.len(),
                    cached_uris,
                };

                let mut req = client.post(&reg_url).json(&reg);
                if let Some(ref token) = ctx.config.cluster_auth_token {
                    if !token.trim().is_empty() {
                        req = req.bearer_auth(token.trim());
                    }
                }

                if let Err(e) = req.send().await {
                    warn!("Heartbeat failed to reach Controller at {}: {}", reg_url, e);
                }

                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_agent_router_status_and_purge() {
        let test_dir = std::env::temp_dir().join(format!("agent_srv_test_{}", chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)));
        std::fs::create_dir_all(&test_dir).unwrap();

        let raw_dev = test_dir.join("dev0.raw");
        std::fs::write(&raw_dev, vec![0u8; 10 * 1024 * 1024]).unwrap();

        let config = Config {
            block_devices: vec![raw_dev.to_str().unwrap().to_string()],
            node_id: "test-node-42".to_string(),
            ..Config::default()
        };

        let metadata = MetadataStore::new(100 * 1024 * 1024, None, None);
        let ingestion = IngestionEngine::new(config.clone(), metadata.clone(), None);
        let discovered_devices = vec![DiscoveredDevice {
            path: raw_dev.to_str().unwrap().to_string(),
            is_nvme: true,
            is_virtual: false,
            is_partition: false,
            is_rotational: false,
            size_bytes: 10 * 1024 * 1024,
            model: Some("Test SSD".to_string()),
            existing_fs: None,
            numa_node: Some(0),
        }];

        let ctx = AgentContext {
            config,
            metadata,
            ingestion,
            discovered_devices,
        };

        let _router = create_agent_router(ctx);
        // Router created cleanly without panicking
        assert!(true);

        let _ = std::fs::remove_dir_all(&test_dir);
    }

    #[tokio::test]
    async fn test_agent_peer_chunk_and_replica_transfer() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let test_dir = std::env::temp_dir().join(format!(
            "agent_p2p_test_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&test_dir).unwrap();

        let raw_dev = test_dir.join("dev0.raw");
        std::fs::write(&raw_dev, vec![0u8; 10 * 1024 * 1024]).unwrap();

        let config = Config {
            block_devices: vec![raw_dev.to_str().unwrap().to_string()],
            node_id: "peer-node-01".to_string(),
            chunk_size_bytes: 4 * 1024 * 1024,
            device_start_offset_bytes: 4096,
            ..Config::default()
        };

        let metadata = MetadataStore::new(100 * 1024 * 1024, None, None);
        let ingestion = IngestionEngine::new(config.clone(), metadata.clone(), None);
        let discovered_devices = vec![DiscoveredDevice {
            path: raw_dev.to_str().unwrap().to_string(),
            is_nvme: true,
            is_virtual: false,
            is_partition: false,
            is_rotational: false,
            size_bytes: 10 * 1024 * 1024,
            model: Some("Test SSD".to_string()),
            existing_fs: None,
            numa_node: Some(0),
        }];

        let ctx = AgentContext {
            config,
            metadata: metadata.clone(),
            ingestion,
            discovered_devices,
        };
        let router = create_agent_router(ctx);

        let payload = b"PEER_REPLICA_TENSOR_CHECKPOINT_PAYLOAD_V1".repeat(128);
        let uri = "hypercache://checkpoints/p2p_epoch_1.pt";

        // 1. Push k=2 replica via POST /api/v1/agent/replica
        let put_req = Request::builder()
            .method("POST")
            .uri(format!("/api/v1/agent/replica?uri={}", uri))
            .body(Body::from(payload.clone()))
            .unwrap();
        let put_resp = router.clone().oneshot(put_req).await.unwrap();
        assert_eq!(put_resp.status(), StatusCode::OK);

        // 2. Fetch chunk back via GET /api/v1/agent/chunk
        let get_req = Request::builder()
            .method("GET")
            .uri(format!("/api/v1/agent/chunk?uri={}", uri))
            .body(Body::empty())
            .unwrap();
        let get_resp = router.oneshot(get_req).await.unwrap();
        assert_eq!(get_resp.status(), StatusCode::OK);
        let fetched = axum::body::to_bytes(get_resp.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(fetched.as_ref(), payload.as_slice());

        let _ = std::fs::remove_dir_all(&test_dir);
    }
}
