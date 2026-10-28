use anyhow::{Context, Result};
use axum::{
    body::Body,
    extract::{DefaultBodyLimit, Query, State},
    http::{header, HeaderValue, StatusCode},
    response::{IntoResponse, Response},
    routing::{get, post},
    Json, Router,
};
use futures_util::{StreamExt, TryStreamExt};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;
use std::time::{Duration, Instant};
use tracing::{error, info, warn};

use crate::auth::{validate_bearer, with_bearer};
use crate::config::Config;
use crate::controller::NodeRegistration;
use crate::discovery::DiscoveredDevice;
use crate::ingestion::IngestionEngine;
use crate::metadata::MetadataStore;
use crate::protocol::{ChunkDescriptor, SyncStatus};

/// Concurrent chunk fetches per object during a P2P pull
const P2P_PARALLEL_CHUNKS: usize = 8;
/// Concurrent object ingests per batch preload directive
const BATCH_PARALLEL_INGESTS: usize = 4;
/// Response header carrying a chunk's CRC32 on /api/v1/agent/chunk
const CHUNK_CRC_HEADER: &str = "x-hypercache-crc32";

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
pub struct AgentUriQuery {
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
    #[serde(default)]
    pub failed_uris: Vec<String>,
}

#[derive(Debug, Serialize)]
pub struct AgentPreloadResponse {
    pub success: bool,
    pub message: String,
    pub size_bytes: Option<u64>,
    pub duration_ms: Option<u64>,
    pub bandwidth_mbps: Option<f64>,
}

/// Chunk-level description of a cached object, served to peers before a P2P pull
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentLayoutResponse {
    pub uri: String,
    pub object_size: u64,
    pub chunk_size: usize,
    pub generation: u64,
    pub chunk_lengths: Vec<u64>,
    pub checksums: Vec<Option<u32>>,
}

fn unauthorized() -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(serde_json::json!({
            "success": false,
            "message": "Unauthorized: Invalid or missing cluster auth token"
        })),
    )
        .into_response()
}

fn mbps(bytes: u64, dur: Duration) -> f64 {
    let secs = dur.as_secs_f64();
    if secs > 0.0 {
        (bytes as f64 / (1024.0 * 1024.0)) / secs
    } else {
        0.0
    }
}

async fn handle_agent_preload(
    State(ctx): State<AgentContext>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<AgentPreloadPayload>,
) -> Response {
    if !validate_bearer(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return unauthorized();
    }

    info!("Agent received fleet PRELOAD directive for URI: {}", payload.uri);
    let t_start = Instant::now();

    if let Ok(Some(existing)) = ctx.metadata.get_layout(&payload.uri).await {
        return Json(AgentPreloadResponse {
            success: true,
            message: "Already cached (Cache Hit)".to_string(),
            size_bytes: Some(existing.object_size),
            duration_ms: Some(t_start.elapsed().as_millis() as u64),
            bandwidth_mbps: None,
        })
        .into_response();
    }

    if let Some(ref peer_url) = payload.source_peer_url {
        match pull_object_from_peer(&ctx, peer_url, &payload.uri).await {
            Ok((object_size, chunk_count)) => {
                let dur = t_start.elapsed();
                let bw = mbps(object_size, dur);
                info!(
                    "Successfully executed P2P PRELOAD for {} from peer {}: {} bytes in {} chunks ({:.2} MB/s)",
                    payload.uri, peer_url, object_size, chunk_count, bw
                );
                return Json(AgentPreloadResponse {
                    success: true,
                    message: format!(
                        "P2P pulled {} across {} chunks from {} in {:.3} s",
                        object_size,
                        chunk_count,
                        peer_url,
                        dur.as_secs_f64()
                    ),
                    size_bytes: Some(object_size),
                    duration_ms: Some(dur.as_millis() as u64),
                    bandwidth_mbps: Some(bw),
                })
                .into_response();
            }
            Err(e) => {
                warn!(
                    "P2P pull from peer {} failed for {} ({:#}). Falling back to authoritative origin...",
                    peer_url, payload.uri, e
                );
            }
        }
    }

    match ctx.ingestion.get_or_ingest(&payload.uri, None, false, false).await {
        Ok((obj, hit)) => {
            let dur = t_start.elapsed();
            let bw = mbps(obj.object_size, dur);
            info!(
                "Fleet PRELOAD for {}: {} bytes in {} chunks ({:.2} MB/s{})",
                payload.uri,
                obj.object_size,
                obj.layout.len(),
                bw,
                if hit { ", cache hit" } else { "" }
            );
            Json(AgentPreloadResponse {
                success: true,
                message: if hit {
                    "Already cached (Cache Hit)".to_string()
                } else {
                    format!(
                        "Striped {} across {} chunks in {:.3} s",
                        obj.object_size,
                        obj.layout.len(),
                        dur.as_secs_f64()
                    )
                },
                size_bytes: Some(obj.object_size),
                duration_ms: Some(dur.as_millis() as u64),
                bandwidth_mbps: if hit { None } else { Some(bw) },
            })
            .into_response()
        }
        Err(e) => {
            error!("Fleet PRELOAD failed for {}: {:#}", payload.uri, e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(AgentPreloadResponse {
                    success: false,
                    message: format!("Ingestion failed: {:#}", e),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                }),
            )
                .into_response()
        }
    }
}

async fn handle_agent_preload_batch(
    State(ctx): State<AgentContext>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<AgentPreloadBatchPayload>,
) -> Response {
    if !validate_bearer(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return unauthorized();
    }

    info!(
        "Agent received fleet PRELOAD_BATCH directive for {} URIs",
        payload.uris.len()
    );
    let t_start = Instant::now();

    let outcomes: Vec<(String, Result<u64>)> = futures_util::stream::iter(payload.uris.iter().cloned())
        .map(|uri| {
            let ctx = ctx.clone();
            async move {
                let res = ctx
                    .ingestion
                    .get_or_ingest(&uri, None, false, false)
                    .await
                    .map(|(obj, _)| obj.object_size);
                (uri, res)
            }
        })
        .buffer_unordered(BATCH_PARALLEL_INGESTS)
        .collect()
        .await;

    let mut total_bytes = 0u64;
    let mut ingested_count = 0;
    let mut failed_uris = Vec::new();
    for (uri, res) in outcomes {
        match res {
            Ok(size) => {
                total_bytes += size;
                ingested_count += 1;
            }
            Err(e) => {
                warn!("Batch preload failed for URI {}: {:#}", uri, e);
                failed_uris.push(uri);
            }
        }
    }
    failed_uris.sort();

    let dur_ms = t_start.elapsed().as_millis() as u64;
    let success = failed_uris.is_empty();
    let mut message = format!(
        "Preloaded {}/{} shards ({:.2} MiB in {} ms)",
        ingested_count,
        payload.uris.len(),
        total_bytes as f64 / (1024.0 * 1024.0),
        dur_ms
    );
    if !success {
        let shown: Vec<&str> = failed_uris.iter().take(5).map(|s| s.as_str()).collect();
        message.push_str(&format!(
            "; {} failed: {}{}",
            failed_uris.len(),
            shown.join(", "),
            if failed_uris.len() > shown.len() { ", ..." } else { "" }
        ));
    }

    Json(AgentPreloadBatchResponse {
        success,
        message,
        ingested_count,
        total_bytes,
        duration_ms: dur_ms,
        failed_uris,
    })
    .into_response()
}

async fn handle_agent_evict(
    State(ctx): State<AgentContext>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<AgentEvictPayload>,
) -> Response {
    if !validate_bearer(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return unauthorized();
    }

    info!("Agent received fleet EVICT directive for URI: {}", payload.uri);
    match ctx.metadata.evict_entry(&payload.uri).await {
        Ok(Some(bytes)) => Json(serde_json::json!({
            "success": true,
            "message": format!("Successfully evicted {} ({} bytes freed)", payload.uri, bytes)
        }))
        .into_response(),
        Ok(None) => Json(serde_json::json!({
            "success": true,
            "message": format!("URI {} was not in cache", payload.uri)
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "success": false,
                "message": format!("Eviction error: {}", e)
            })),
        )
            .into_response(),
    }
}

async fn handle_agent_purge(State(ctx): State<AgentContext>, headers: axum::http::HeaderMap) -> Response {
    if !validate_bearer(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return unauthorized();
    }

    info!("Agent received fleet PURGE directive");
    match ctx.metadata.purge_all().await {
        Ok((count, bytes)) => Json(serde_json::json!({
            "success": true,
            "message": format!("Successfully purged {} models ({} bytes freed) and reset drives", count, bytes)
        }))
        .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({
                "success": false,
                "message": format!("Purge error: {}", e)
            })),
        )
            .into_response(),
    }
}

async fn handle_agent_status(State(ctx): State<AgentContext>, headers: axum::http::HeaderMap) -> Response {
    if !validate_bearer(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return unauthorized();
    }

    let (entries, total_used_bytes) = ctx.metadata.list_entries().await;
    let cached_uris = entries.iter().map(|e| e.uri.clone()).collect::<Vec<_>>();
    let drives = ctx
        .discovered_devices
        .iter()
        .map(|d| d.path.clone())
        .collect::<Vec<_>>();
    let total_capacity_bytes = ctx.discovered_devices.iter().map(|d| d.size_bytes).sum::<u64>();

    Json(serde_json::json!({
        "node_id": ctx.config.node_id,
        "drives": drives,
        "total_capacity_bytes": total_capacity_bytes,
        "used_capacity_bytes": total_used_bytes,
        "cached_objects_count": entries.len(),
        "cached_uris": cached_uris,
        "agent_version": env!("CARGO_PKG_VERSION"),
    }))
    .into_response()
}

async fn handle_agent_get_layout(
    State(ctx): State<AgentContext>,
    headers: axum::http::HeaderMap,
    Query(query): Query<AgentUriQuery>,
) -> Response {
    if !validate_bearer(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return unauthorized();
    }
    match ctx.metadata.get_layout(&query.uri).await {
        Ok(Some(meta)) => Json(AgentLayoutResponse {
            uri: query.uri,
            object_size: meta.object_size,
            chunk_size: meta.chunk_size,
            generation: meta.generation,
            chunk_lengths: meta.chunk_layout.iter().map(|c| c.length).collect(),
            checksums: meta.chunk_layout.iter().map(|c| c.checksum).collect(),
        })
        .into_response(),
        _ => StatusCode::NOT_FOUND.into_response(),
    }
}

/// Serves one chunk (`chunk_idx`) or streams the whole object chunk by chunk, so serving a
/// multi-GB model never buffers it in memory.
async fn handle_agent_get_chunk(
    State(ctx): State<AgentContext>,
    headers: axum::http::HeaderMap,
    Query(query): Query<AgentChunkQuery>,
) -> Response {
    if !validate_bearer(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return StatusCode::UNAUTHORIZED.into_response();
    }

    let meta = match ctx.metadata.get_layout(&query.uri).await {
        Ok(Some(meta)) => meta,
        _ => return StatusCode::NOT_FOUND.into_response(),
    };

    if let Some(idx) = query.chunk_idx {
        let Some(chunk) = meta.chunk_layout.get(idx).cloned() else {
            return StatusCode::NOT_FOUND.into_response();
        };
        return match ctx.ingestion.read_chunk(&chunk).await {
            Ok(bytes) => {
                let mut resp = (StatusCode::OK, bytes).into_response();
                if let Some(crc) = chunk.checksum {
                    if let Ok(v) = HeaderValue::from_str(&crc.to_string()) {
                        resp.headers_mut().insert(CHUNK_CRC_HEADER, v);
                    }
                }
                resp
            }
            Err(e) => {
                error!(
                    "Failed reading chunk {} of {} for P2P transfer: {:#}",
                    idx, query.uri, e
                );
                StatusCode::INTERNAL_SERVER_ERROR.into_response()
            }
        };
    }

    let ingestion = ctx.ingestion.clone();
    let uri = query.uri.clone();
    let stream = futures_util::stream::iter(meta.chunk_layout).then(move |chunk| {
        let ingestion = ingestion.clone();
        let uri = uri.clone();
        async move {
            ingestion.read_chunk(&chunk).await.map_err(|e| {
                error!("Failed streaming chunk of {} for P2P transfer: {:#}", uri, e);
                std::io::Error::other(e.to_string())
            })
        }
    });
    let mut resp = Response::new(Body::from_stream(stream));
    if let Ok(v) = HeaderValue::from_str(&meta.object_size.to_string()) {
        resp.headers_mut().insert(header::CONTENT_LENGTH, v);
    }
    resp
}

/// Accepts a peer checkpoint replica as a streamed request body (any size), striping it
/// chunk by chunk as it arrives.
async fn handle_agent_put_replica(
    State(ctx): State<AgentContext>,
    headers: axum::http::HeaderMap,
    Query(query): Query<AgentUriQuery>,
    body: Body,
) -> Response {
    if !validate_bearer(&headers, ctx.config.cluster_auth_token.as_deref()) {
        return unauthorized();
    }

    let chunk_size = ctx.config.chunk_size_bytes;
    let striper = ctx.ingestion.striper().clone();
    let mut stream = body.into_data_stream();
    let mut buffer: Vec<u8> = Vec::with_capacity(chunk_size);
    let mut layout: Vec<ChunkDescriptor> = Vec::new();

    let result: Result<u64> = async {
        let mut total = 0u64;
        while let Some(item) = stream.next().await {
            let bytes = item.context("Error reading replica request body")?;
            let mut slice = &bytes[..];
            while !slice.is_empty() {
                let take = (chunk_size - buffer.len()).min(slice.len());
                buffer.extend_from_slice(&slice[..take]);
                slice = &slice[take..];
                if buffer.len() == chunk_size {
                    let desc = striper
                        .stripe_single_chunk(&query.uri, layout.len(), &buffer, None)
                        .await?;
                    total += buffer.len() as u64;
                    layout.push(desc);
                    buffer.clear();
                }
            }
        }
        if !buffer.is_empty() {
            let desc = striper
                .stripe_single_chunk(&query.uri, layout.len(), &buffer, None)
                .await?;
            total += buffer.len() as u64;
            layout.push(desc);
        }
        striper.sync_chunks(&layout).await?;
        Ok(total)
    }
    .await;

    let size_bytes = match result {
        Ok(n) => n,
        Err(e) => {
            ctx.metadata.release_unpublished(&layout).await;
            return (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "success": false, "message": format!("{:#}", e) })),
            )
                .into_response();
        }
    };

    let chunk_count = layout.len();
    if let Err(e) = ctx
        .metadata
        .save_layout_with_status(
            &query.uri,
            size_bytes,
            chunk_size,
            layout,
            Some(SyncStatus::ReplicaCommitted),
        )
        .await
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "success": false, "message": e.to_string() })),
        )
            .into_response();
    }
    info!(
        "Stored peer checkpoint replica for '{}': {} bytes across {} chunk(s)",
        query.uri, size_bytes, chunk_count
    );
    Json(serde_json::json!({
        "success": true,
        "uri": query.uri,
        "size_bytes": size_bytes,
        "chunks": chunk_count,
    }))
    .into_response()
}

pub fn create_agent_router(ctx: AgentContext) -> Router {
    Router::new()
        .route("/api/v1/agent/preload", post(handle_agent_preload))
        .route("/api/v1/agent/preload_batch", post(handle_agent_preload_batch))
        .route("/api/v1/agent/evict", post(handle_agent_evict))
        .route("/api/v1/agent/purge", post(handle_agent_purge))
        .route("/api/v1/agent/status", get(handle_agent_status))
        .route("/api/v1/agent/layout", get(handle_agent_get_layout))
        .route("/api/v1/agent/chunk", get(handle_agent_get_chunk))
        .route(
            "/api/v1/agent/replica",
            post(handle_agent_put_replica).layer(DefaultBodyLimit::disable()),
        )
        .with_state(ctx)
}

fn validate_peer_url(peer_base_url: &str) -> Result<reqwest::Url> {
    let url = reqwest::Url::parse(peer_base_url.trim_end_matches('/'))
        .with_context(|| format!("Invalid peer URL '{}'", peer_base_url))?;
    if url.scheme() != "http" && url.scheme() != "https" {
        anyhow::bail!("Peer URL '{}' must use http or https", peer_base_url);
    }
    if url.host_str().is_none() {
        anyhow::bail!("Peer URL '{}' has no host", peer_base_url);
    }
    Ok(url)
}

/// Pulls an object from a peer agent: fetches the peer's layout, then its chunks in
/// parallel, validating each chunk's length and CRC32 before the layout is published.
async fn pull_object_from_peer(ctx: &AgentContext, peer_base_url: &str, uri: &str) -> Result<(u64, usize)> {
    let base = validate_peer_url(peer_base_url)?;
    let base = base.as_str().trim_end_matches('/').to_string();
    let token = ctx.config.cluster_auth_token.clone();
    let client = Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .read_timeout(Duration::from_secs(60))
        .build()
        .unwrap_or_else(|_| Client::new());

    let layout_resp = with_bearer(
        client
            .get(format!("{}/api/v1/agent/layout", base))
            .query(&[("uri", uri)]),
        token.as_deref(),
    )
    .send()
    .await
    .with_context(|| format!("Failed to connect to peer {}", base))?;

    if layout_resp.status() == StatusCode::NOT_FOUND {
        anyhow::bail!("Peer {} does not have {} cached", base, uri);
    }
    if !layout_resp.status().is_success() {
        anyhow::bail!("Peer {} returned HTTP {} for layout", base, layout_resp.status());
    }
    let peer_layout: AgentLayoutResponse = layout_resp.json().await.context("Invalid peer layout response")?;
    let expected_total: u64 = peer_layout.chunk_lengths.iter().sum();
    if expected_total != peer_layout.object_size {
        anyhow::bail!("Peer layout for {} is inconsistent", uri);
    }

    let striper = ctx.ingestion.striper().clone();
    let chunk_url = format!("{}/api/v1/agent/chunk", base);
    // Every chunk written so far, so a failure part-way through can release them all
    let written = std::sync::Arc::new(tokio::sync::Mutex::new(Vec::<(usize, ChunkDescriptor)>::new()));
    let results: Result<Vec<()>> = futures_util::stream::iter(peer_layout.chunk_lengths.iter().copied().enumerate())
        .map(|(idx, expected_len)| {
            let client = client.clone();
            let striper = striper.clone();
            let chunk_url = chunk_url.clone();
            let token = token.clone();
            let expected_crc = peer_layout.checksums.get(idx).copied().flatten();
            let uri = uri.to_string();
            let written = std::sync::Arc::clone(&written);
            async move {
                let resp = with_bearer(
                    client
                        .get(&chunk_url)
                        .query(&[("uri", uri.as_str()), ("chunk_idx", &idx.to_string())]),
                    token.as_deref(),
                )
                .send()
                .await?;
                if !resp.status().is_success() {
                    anyhow::bail!("Peer returned HTTP {} for chunk {}", resp.status(), idx);
                }
                let data = resp.bytes().await?;
                if data.len() as u64 != expected_len {
                    anyhow::bail!(
                        "Chunk {} truncated: got {} bytes, expected {}",
                        idx,
                        data.len(),
                        expected_len
                    );
                }
                if let Some(crc) = expected_crc {
                    if crc32fast::hash(&data) != crc {
                        anyhow::bail!("Chunk {} failed CRC32 verification", idx);
                    }
                }
                let desc = striper.stripe_single_chunk(&uri, idx, &data, None).await?;
                written.lock().await.push((idx, desc));
                Ok(())
            }
        })
        .buffer_unordered(P2P_PARALLEL_CHUNKS)
        .try_collect()
        .await;

    let mut chunks = std::mem::take(&mut *written.lock().await);
    chunks.sort_by_key(|(i, _)| *i);
    let layout: Vec<ChunkDescriptor> = chunks.into_iter().map(|(_, c)| c).collect();

    if let Err(e) = results {
        ctx.metadata.release_unpublished(&layout).await;
        return Err(e);
    }

    if let Err(e) = striper.sync_chunks(&layout).await {
        ctx.metadata.release_unpublished(&layout).await;
        return Err(e);
    }

    let chunk_count = layout.len();
    ctx.metadata
        .save_layout(uri, peer_layout.object_size, ctx.config.chunk_size_bytes, layout)
        .await?;
    Ok((peer_layout.object_size, chunk_count))
}

/// Returns true for addresses reachable only from this host.
fn is_loopback_bind(addr: &str) -> bool {
    addr.parse::<std::net::IpAddr>()
        .map_or(addr == "localhost", |ip| ip.is_loopback())
}

pub async fn start_agent_http_server(ctx: AgentContext, port: u16) -> Result<()> {
    let token_configured = crate::auth::effective_token(ctx.config.cluster_auth_token.as_deref()).is_some();
    if !token_configured && !is_loopback_bind(&ctx.config.agent_bind_address) {
        anyhow::bail!(
            "Refusing to expose the Node Agent API on {}:{} without authentication. The agent API can purge \
             caches and write to raw NVMe devices. Set cluster_auth_token (or HYPERCACHE_AUTH_TOKEN), or bind \
             the agent to 127.0.0.1 with agent_bind_address.",
            ctx.config.agent_bind_address,
            port
        );
    }

    let app = create_agent_router(ctx.clone());
    let ip: std::net::IpAddr = ctx
        .config
        .agent_bind_address
        .parse()
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::LOCALHOST));
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

    if let Some(ref controller_url) = ctx.config.controller_url {
        let controller_url = controller_url.clone();
        let client = Client::builder()
            .timeout(Duration::from_secs(5))
            .build()
            .unwrap_or_else(|_| Client::new());

        tokio::spawn(async move {
            info!("Starting Node Agent heartbeat to Controller at {}", controller_url);
            let reg_url = format!("{}/api/v1/register", controller_url.trim_end_matches('/'));
            let mut last_sent_inventory: Option<u64> = None;
            let mut auth_warned = false;

            loop {
                let reg = build_registration(&ctx, port, &mut last_sent_inventory).await;
                let inventory_hash = reg.1;
                let req = with_bearer(
                    client.post(&reg_url).json(&reg.0),
                    ctx.config.cluster_auth_token.as_deref(),
                );

                match req.send().await {
                    Ok(resp) if resp.status().is_success() => {
                        auth_warned = false;
                        let body: serde_json::Value = resp.json().await.unwrap_or_default();
                        if body
                            .get("need_full_inventory")
                            .and_then(|v| v.as_bool())
                            .unwrap_or(false)
                        {
                            last_sent_inventory = None;
                        } else {
                            last_sent_inventory = Some(inventory_hash);
                        }
                    }
                    Ok(resp) if resp.status() == StatusCode::UNAUTHORIZED => {
                        last_sent_inventory = None;
                        if !auth_warned {
                            error!(
                                "Controller at {} rejected this node's registration (HTTP 401). Check that \
                                 cluster_auth_token matches the controller's token.",
                                reg_url
                            );
                            auth_warned = true;
                        }
                    }
                    Ok(resp) => {
                        last_sent_inventory = None;
                        warn!(
                            "Controller at {} rejected heartbeat with HTTP {}",
                            reg_url,
                            resp.status()
                        );
                    }
                    Err(e) => {
                        last_sent_inventory = None;
                        warn!("Heartbeat failed to reach Controller at {}: {}", reg_url, e);
                    }
                }

                tokio::time::sleep(Duration::from_secs(5)).await;
            }
        });
    }

    Ok(())
}

/// Builds a heartbeat registration. The cached URI list is only included when it changed
/// since the last acknowledged heartbeat (or the controller asked for it).
async fn build_registration(ctx: &AgentContext, port: u16, last_sent: &mut Option<u64>) -> (NodeRegistration, u64) {
    use std::hash::{Hash, Hasher};
    let (entries, total_used_bytes) = ctx.metadata.list_entries().await;
    let mut cached_uris = entries.iter().map(|e| e.uri.clone()).collect::<Vec<_>>();
    cached_uris.sort();
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    cached_uris.hash(&mut hasher);
    let inventory_hash = hasher.finish();
    let omit = *last_sent == Some(inventory_hash);

    let drives = ctx
        .discovered_devices
        .iter()
        .map(|d| d.path.clone())
        .collect::<Vec<_>>();
    let total_capacity_bytes = ctx.discovered_devices.iter().map(|d| d.size_bytes).sum::<u64>();

    let address = ctx.config.node_address.clone().unwrap_or_else(|| {
        if ctx.config.agent_bind_address != "0.0.0.0" && ctx.config.agent_bind_address != "127.0.0.1" {
            ctx.config.agent_bind_address.clone()
        } else {
            String::new()
        }
    });

    (
        NodeRegistration {
            node_id: ctx.config.node_id.clone(),
            address,
            http_port: port,
            labels: ctx.config.node_labels.clone(),
            drives,
            total_capacity_bytes,
            used_capacity_bytes: total_used_bytes,
            cached_objects_count: entries.len(),
            cached_uris: if omit { Vec::new() } else { cached_uris },
            cached_uris_omitted: omit,
            agent_version: Some(env!("CARGO_PKG_VERSION").to_string()),
        },
        inventory_hash,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::http::Request;
    use tower::ServiceExt;

    struct TestAgent {
        dir: std::path::PathBuf,
        ctx: AgentContext,
        router: Router,
    }

    impl Drop for TestAgent {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn test_agent(tag: &str, token: Option<&str>) -> TestAgent {
        let dir = std::env::temp_dir().join(format!(
            "agent_{}_{}",
            tag,
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let raw_dev = dir.join("dev0.raw");
        std::fs::write(&raw_dev, vec![0u8; 160 * 1024 * 1024]).unwrap();

        let config = Config {
            block_devices: vec![raw_dev.to_str().unwrap().to_string()],
            node_id: format!("{}-node", tag),
            chunk_size_bytes: 4 * 1024 * 1024,
            device_start_offset_bytes: 4096,
            cluster_auth_token: token.map(|t| t.to_string()),
            nfs_mount_path: dir.to_string_lossy().to_string(),
            ..Config::default()
        };
        let metadata = MetadataStore::new(1 << 30, None, None);
        let ingestion = IngestionEngine::new(config.clone(), metadata.clone(), None);
        let discovered_devices = vec![DiscoveredDevice {
            path: raw_dev.to_str().unwrap().to_string(),
            is_nvme: true,
            is_virtual: false,
            is_partition: false,
            is_rotational: false,
            size_bytes: 160 * 1024 * 1024,
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
        TestAgent {
            dir,
            router: create_agent_router(ctx.clone()),
            ctx,
        }
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn test_agent_status_and_purge_require_token() {
        let agent = test_agent("authz", Some("tok"));

        for (method, uri) in [
            ("GET", "/api/v1/agent/status"),
            ("POST", "/api/v1/agent/purge"),
            ("GET", "/api/v1/agent/chunk?uri=x"),
            ("GET", "/api/v1/agent/layout?uri=x"),
            ("POST", "/api/v1/agent/replica?uri=x"),
        ] {
            let req = Request::builder().method(method).uri(uri).body(Body::empty()).unwrap();
            let resp = agent.router.clone().oneshot(req).await.unwrap();
            assert_eq!(
                resp.status(),
                StatusCode::UNAUTHORIZED,
                "{} {} was not protected",
                method,
                uri
            );
        }

        agent
            .ctx
            .metadata
            .save_layout("hypercache://m.bin", 10, 4194304, vec![])
            .await
            .unwrap();

        let req = Request::builder()
            .method("GET")
            .uri("/api/v1/agent/status")
            .header("authorization", "Bearer tok")
            .body(Body::empty())
            .unwrap();
        let resp = agent.router.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let status = body_json(resp).await;
        assert_eq!(status["node_id"], "authz-node");
        assert_eq!(status["cached_objects_count"], 1);

        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/agent/purge")
            .header("authorization", "Bearer tok")
            .body(Body::empty())
            .unwrap();
        let resp = agent.router.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(agent.ctx.metadata.list_entries().await.0.len(), 0);
    }

    #[tokio::test]
    async fn test_agent_peer_chunk_and_replica_transfer() {
        let agent = test_agent("p2p", None);
        let payload = b"PEER_REPLICA_TENSOR_CHECKPOINT_PAYLOAD_V1".repeat(128);
        let uri = "hypercache://checkpoints/p2p_epoch_1.pt";

        let put_req = Request::builder()
            .method("POST")
            .uri(format!("/api/v1/agent/replica?uri={}", uri))
            .body(Body::from(payload.clone()))
            .unwrap();
        assert_eq!(
            agent.router.clone().oneshot(put_req).await.unwrap().status(),
            StatusCode::OK
        );

        let get_req = Request::builder()
            .method("GET")
            .uri(format!("/api/v1/agent/chunk?uri={}", uri))
            .body(Body::empty())
            .unwrap();
        let get_resp = agent.router.clone().oneshot(get_req).await.unwrap();
        assert_eq!(get_resp.status(), StatusCode::OK);
        let fetched = axum::body::to_bytes(get_resp.into_body(), usize::MAX).await.unwrap();
        assert_eq!(fetched.as_ref(), payload.as_slice());

        let meta = agent.ctx.metadata.get_layout(uri).await.unwrap().unwrap();
        assert_eq!(meta.sync_status, Some(SyncStatus::ReplicaCommitted));
    }

    #[tokio::test]
    async fn test_large_replica_upload_is_streamed_and_accepted() {
        // Well above axum's default 2 MiB body limit
        let agent = test_agent("bigreplica", None);
        let payload: Vec<u8> = (0..(64 * 1024 * 1024 + 123)).map(|i| (i % 249) as u8).collect();
        let uri = "hypercache://checkpoints/large_epoch.pt";

        let req = Request::builder()
            .method("POST")
            .uri(format!("/api/v1/agent/replica?uri={}", uri))
            .body(Body::from(payload.clone()))
            .unwrap();
        let resp = agent.router.clone().oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        let body = body_json(resp).await;
        assert_eq!(body["size_bytes"], payload.len() as u64);
        assert_eq!(body["chunks"], 17);

        // Layout endpoint reports per-chunk checksums
        let req = Request::builder()
            .method("GET")
            .uri(format!("/api/v1/agent/layout?uri={}", uri))
            .body(Body::empty())
            .unwrap();
        let layout: AgentLayoutResponse =
            serde_json::from_value(body_json(agent.router.clone().oneshot(req).await.unwrap()).await).unwrap();
        assert_eq!(layout.object_size, payload.len() as u64);
        assert_eq!(layout.chunk_lengths.len(), 17);
        assert!(layout.checksums.iter().all(|c| c.is_some()));

        // Single-chunk fetch carries the CRC header and matches the source bytes
        let req = Request::builder()
            .method("GET")
            .uri(format!("/api/v1/agent/chunk?uri={}&chunk_idx=16", uri))
            .body(Body::empty())
            .unwrap();
        let resp = agent.router.clone().oneshot(req).await.unwrap();
        let crc: u32 = resp.headers()[CHUNK_CRC_HEADER].to_str().unwrap().parse().unwrap();
        let tail = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        assert_eq!(tail.as_ref(), &payload[16 * 4 * 1024 * 1024..]);
        assert_eq!(crc, crc32fast::hash(&tail));
    }

    #[tokio::test]
    async fn test_p2p_pull_between_two_live_agents() {
        let source = test_agent("p2psrc", Some("fleet-tok"));
        let dest = test_agent("p2pdst", Some("fleet-tok"));

        let data: Vec<u8> = (0..(10 * 1024 * 1024 + 7)).map(|i| (i % 233) as u8).collect();
        let uri = "hypercache://models/p2p/weights.bin";
        let layout = source
            .ctx
            .ingestion
            .striper()
            .stripe_bytes_to_tier(uri, &data, None)
            .await
            .unwrap();
        source
            .ctx
            .metadata
            .save_layout(uri, data.len() as u64, 4 * 1024 * 1024, layout)
            .await
            .unwrap();

        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();
        let router = source.router.clone();
        tokio::spawn(async move { axum::serve(listener, router).await.unwrap() });

        let (size, chunks) = pull_object_from_peer(&dest.ctx, &format!("http://{}", addr), uri)
            .await
            .unwrap();
        assert_eq!(size, data.len() as u64);
        assert_eq!(chunks, 3);
        let meta = dest.ctx.metadata.get_layout(uri).await.unwrap().unwrap();
        assert_eq!(
            dest.ctx
                .ingestion
                .read_chunks_to_bytes(&meta.chunk_layout)
                .await
                .unwrap(),
            data
        );

        // Missing objects and bad schemes fail cleanly
        assert!(
            pull_object_from_peer(&dest.ctx, &format!("http://{}", addr), "hypercache://missing")
                .await
                .is_err()
        );
        assert!(pull_object_from_peer(&dest.ctx, "file:///etc/passwd", uri)
            .await
            .is_err());
    }

    #[tokio::test]
    async fn test_batch_preload_reports_failures() {
        let agent = test_agent("batch", None);
        let good = agent.dir.join("good.bin");
        std::fs::write(&good, b"payload").unwrap();
        let body = serde_json::json!({
            "uris": [format!("file://{}", good.display()), format!("file://{}/missing.bin", agent.dir.display())]
        });
        let req = Request::builder()
            .method("POST")
            .uri("/api/v1/agent/preload_batch")
            .header("content-type", "application/json")
            .body(Body::from(body.to_string()))
            .unwrap();
        let resp = body_json(agent.router.clone().oneshot(req).await.unwrap()).await;
        assert_eq!(resp["success"], false);
        assert_eq!(resp["ingested_count"], 1);
        assert_eq!(resp["failed_uris"].as_array().unwrap().len(), 1);
    }

    #[tokio::test]
    async fn test_agent_refuses_network_bind_without_token() {
        let mut agent = test_agent("bind", None);
        agent.ctx.config.agent_bind_address = "0.0.0.0".to_string();
        let err = start_agent_http_server(agent.ctx.clone(), 0).await.unwrap_err();
        assert!(err.to_string().contains("Refusing"));

        // Loopback without a token is allowed (local-only exposure)
        agent.ctx.config.agent_bind_address = "127.0.0.1".to_string();
        start_agent_http_server(agent.ctx.clone(), 0).await.unwrap();

        // Network bind with a token is allowed
        agent.ctx.config.agent_bind_address = "0.0.0.0".to_string();
        agent.ctx.config.cluster_auth_token = Some("tok".to_string());
        start_agent_http_server(agent.ctx.clone(), 0).await.unwrap();
    }

    #[tokio::test]
    async fn test_heartbeat_omits_unchanged_inventory() {
        let agent = test_agent("hb", None);
        agent
            .ctx
            .metadata
            .save_layout("hypercache://a", 1, 4194304, vec![])
            .await
            .unwrap();

        let mut last = None;
        let (reg1, hash1) = build_registration(&agent.ctx, 9115, &mut last).await;
        assert!(!reg1.cached_uris_omitted);
        assert_eq!(reg1.cached_uris, vec!["hypercache://a".to_string()]);

        last = Some(hash1);
        let (reg2, _) = build_registration(&agent.ctx, 9115, &mut last).await;
        assert!(reg2.cached_uris_omitted);
        assert!(reg2.cached_uris.is_empty());
        assert_eq!(reg2.cached_objects_count, 1);

        agent
            .ctx
            .metadata
            .save_layout("hypercache://b", 1, 4194304, vec![])
            .await
            .unwrap();
        let (reg3, _) = build_registration(&agent.ctx, 9115, &mut last).await;
        assert!(!reg3.cached_uris_omitted);
        assert_eq!(reg3.cached_uris.len(), 2);
    }
}
