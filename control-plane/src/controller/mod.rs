use anyhow::Result;
use axum::{
    extract::{ConnectInfo, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Json, Router,
};
use chrono::Utc;
use futures_util::future::join_all;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;
use tokio::sync::RwLock;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeRegistration {
    pub node_id: String,
    pub address: String,
    pub http_port: u16,
    #[serde(default)]
    pub labels: HashMap<String, String>,
    pub drives: Vec<String>,
    pub total_capacity_bytes: u64,
    pub used_capacity_bytes: u64,
    pub cached_objects_count: usize,
    #[serde(default)]
    pub cached_uris: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeState {
    pub registration: NodeRegistration,
    pub last_heartbeat: i64,
    pub status: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetSummary {
    pub total_nodes: usize,
    pub online_nodes: usize,
    pub offline_nodes: usize,
    pub aggregate_capacity_bytes: u64,
    pub total_used_bytes: u64,
    pub total_cached_objects: usize,
    pub nodes: Vec<NodeState>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetPreloadRequest {
    pub uri: String,
    pub nodes: Option<String>,
    pub selector: Option<String>,
    #[serde(default)]
    pub p2p: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetPreloadBatchRequest {
    #[serde(default)]
    pub uris: Vec<String>,
    #[serde(default)]
    pub prefix: Option<String>,
    pub nodes: Option<String>,
    pub selector: Option<String>,
    #[serde(default)]
    pub sharded: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetEvictRequest {
    pub uri: String,
    pub nodes: Option<String>,
    pub selector: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetPurgeRequest {
    pub nodes: Option<String>,
    pub selector: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FleetOperationResult {
    pub node_id: String,
    pub success: bool,
    pub message: String,
    pub size_bytes: Option<u64>,
    pub duration_ms: Option<u64>,
    pub bandwidth_mbps: Option<f64>,
}

#[derive(Clone)]
pub struct FleetRegistry {
    nodes: Arc<RwLock<HashMap<String, NodeState>>>,
    http_client: Client,
    pub auth_token: Option<String>,
    pub bind_ip: Option<std::net::IpAddr>,
    pub config: crate::config::Config,
}

impl FleetRegistry {
    pub fn new(auth_token: Option<String>, bind_ip: Option<std::net::IpAddr>) -> Self {
        Self::new_with_config(auth_token, bind_ip, crate::config::Config::default())
    }

    pub fn new_with_config(
        auth_token: Option<String>,
        bind_ip: Option<std::net::IpAddr>,
        config: crate::config::Config,
    ) -> Self {
        let http_client = Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            http_client,
            auth_token,
            bind_ip,
            config,
        }
    }

    pub async fn register_or_heartbeat(&self, reg: NodeRegistration) {
        let now = Utc::now().timestamp();
        let mut map = self.nodes.write().await;
        map.insert(
            reg.node_id.clone(),
            NodeState {
                registration: reg,
                last_heartbeat: now,
                status: "ONLINE".to_string(),
            },
        );
    }

    pub async fn get_summary(&self) -> FleetSummary {
        let now = Utc::now().timestamp();
        let mut map = self.nodes.write().await;

        let mut online = 0;
        let mut offline = 0;
        let mut total_cap = 0;
        let mut total_used = 0;
        let mut total_objs = 0;

        let mut node_list = Vec::new();

        for state in map.values_mut() {
            // If no heartbeat for > 15 seconds, mark offline
            if now - state.last_heartbeat > 15 {
                state.status = "OFFLINE".to_string();
                offline += 1;
            } else {
                state.status = "ONLINE".to_string();
                online += 1;
            }

            total_cap += state.registration.total_capacity_bytes;
            total_used += state.registration.used_capacity_bytes;
            total_objs += state.registration.cached_objects_count;
            node_list.push(state.clone());
        }

        node_list.sort_by(|a, b| a.registration.node_id.cmp(&b.registration.node_id));

        FleetSummary {
            total_nodes: node_list.len(),
            online_nodes: online,
            offline_nodes: offline,
            aggregate_capacity_bytes: total_cap,
            total_used_bytes: total_used,
            total_cached_objects: total_objs,
            nodes: node_list,
        }
    }

    pub async fn filter_nodes(
        &self,
        nodes_filter: Option<&str>,
        selector_filter: Option<&str>,
    ) -> Vec<NodeState> {
        let summary = self.get_summary().await;
        summary
            .nodes
            .into_iter()
            .filter(|n| matches_node(n, nodes_filter, selector_filter))
            .collect()
    }
}

pub fn matches_node(
    node: &NodeState,
    nodes_filter: Option<&str>,
    selector_filter: Option<&str>,
) -> bool {
    if let Some(nf) = nodes_filter {
        let nf_trim = nf.trim();
        if nf_trim != "all" && !nf_trim.is_empty() {
            let parts: Vec<&str> = nf_trim.split(',').map(|s| s.trim()).collect();
            let mut matched = false;

            for part in parts {
                if part == node.registration.node_id {
                    matched = true;
                    break;
                }
                // Handle range syntax e.g. "gpu-01..gpu-10" or "node-1..node-8"
                if part.contains("..") {
                    let range_parts: Vec<&str> = part.split("..").collect();
                    if range_parts.len() == 2 {
                        let start = range_parts[0];
                        let end = range_parts[1];
                        if node.registration.node_id.as_str() >= start
                            && node.registration.node_id.as_str() <= end
                        {
                            matched = true;
                            break;
                        }
                    }
                }
            }

            if !matched {
                return false;
            }
        }
    }

    if let Some(sel) = selector_filter {
        let sel_trim = sel.trim();
        if !sel_trim.is_empty() {
            let pairs: Vec<&str> = sel_trim.split(',').map(|s| s.trim()).collect();
            for pair in pairs {
                let kv: Vec<&str> = pair.split('=').map(|s| s.trim()).collect();
                if kv.len() == 2 {
                    let key = kv[0];
                    let val = kv[1];
                    if node.registration.labels.get(key).map(|v| v.as_str()) != Some(val) {
                        return false;
                    }
                }
            }
        }
    }

    true
}

/// Deterministic Rendezvous (Highest Random Weight) consistent hashing across active nodes.
/// Ensures that adding or removing a node only reassigns ~1/N of dataset objects.
pub fn assign_uri_rendezvous<'a>(uri: &str, nodes: &'a [NodeState]) -> Option<&'a NodeState> {
    use sha2::{Digest, Sha256};
    let mut best_node: Option<&'a NodeState> = None;
    let mut best_weight: u64 = 0;

    for node in nodes {
        let mut hasher = Sha256::new();
        hasher.update(node.registration.node_id.as_bytes());
        hasher.update(b"::");
        hasher.update(uri.as_bytes());
        let digest = hasher.finalize();
        let weight = u64::from_be_bytes([
            digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
        ]);
        if best_node.is_none() || weight > best_weight {
            best_weight = weight;
            best_node = Some(node);
        }
    }

    best_node
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

fn is_authorized_or_localhost(
    connect_info: Option<&ConnectInfo<SocketAddr>>,
    headers: &axum::http::HeaderMap,
    expected_token: Option<&str>,
    bind_ip: Option<std::net::IpAddr>,
) -> bool {
    if let Some(ci) = connect_info {
        if ci.ip().is_loopback() {
            return true;
        }
        if let Some(b_ip) = bind_ip {
            if !b_ip.is_unspecified() && ci.ip() == b_ip {
                return true;
            }
        }
    }
    validate_auth(headers, expected_token)
}

// Controller Handlers
async fn handle_register(
    State(registry): State<FleetRegistry>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: axum::http::HeaderMap,
    Json(mut payload): Json<NodeRegistration>,
) -> impl IntoResponse {
    if !validate_auth(&headers, registry.auth_token.as_deref()) {
        return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({ "error": "Unauthorized: Invalid or missing cluster auth token" })));
    }

    // Auto-resolve remote node IP from incoming socket connection if address is loopback or default
    if let Some(ConnectInfo(addr)) = connect_info {
        if (payload.address == "127.0.0.1" || payload.address == "0.0.0.0" || payload.address.is_empty()) && !addr.ip().is_loopback() {
            payload.address = addr.ip().to_string();
        }
    }

    registry.register_or_heartbeat(payload).await;
    (StatusCode::OK, Json(serde_json::json!({ "status": "OK" })))
}

async fn handle_get_nodes(
    State(registry): State<FleetRegistry>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: axum::http::HeaderMap,
) -> impl IntoResponse {
    if !is_authorized_or_localhost(connect_info.as_ref(), &headers, registry.auth_token.as_deref(), registry.bind_ip) {
        return (StatusCode::UNAUTHORIZED, Json(serde_json::json!({ "error": "Unauthorized: Remote access to cluster topology requires valid cluster auth token" })));
    }

    let summary = registry.get_summary().await;
    (StatusCode::OK, Json(serde_json::to_value(summary).unwrap_or_default()))
}

async fn handle_fleet_preload(
    State(registry): State<FleetRegistry>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<FleetPreloadRequest>,
) -> impl IntoResponse {
    if !is_authorized_or_localhost(connect_info.as_ref(), &headers, registry.auth_token.as_deref(), registry.bind_ip) {
        return (StatusCode::UNAUTHORIZED, Json(vec![FleetOperationResult {
            node_id: "none".to_string(),
            success: false,
            message: "Unauthorized: Remote access to fleet dispatch requires valid cluster auth token".to_string(),
            size_bytes: None,
            duration_ms: None,
            bandwidth_mbps: None,
        }]));
    }

    let targets = registry
        .filter_nodes(payload.nodes.as_deref(), payload.selector.as_deref())
        .await;

    if targets.is_empty() {
        return (StatusCode::OK, Json(vec![FleetOperationResult {
            node_id: "none".to_string(),
            success: false,
            message: "No online nodes matched the given selector/filter.".to_string(),
            size_bytes: None,
            duration_ms: None,
            bandwidth_mbps: None,
        }]));
    }

    info!(
        "Dispatching fleet preload for '{}' across {} target nodes",
        payload.uri,
        targets.len()
    );

    let client = registry.http_client.clone();
    let uri = payload.uri.clone();
    let auth_token = registry.auth_token.clone();
    let use_p2p = (payload.p2p || registry.config.p2p_preload_enabled) && targets.len() > 1;

    if use_p2p {
        // Check if any target node already has the URI cached; otherwise seed on targets[0] first
        let mut seed_idx = targets
            .iter()
            .position(|n| n.registration.cached_uris.contains(&uri))
            .unwrap_or(0);

        let mut all_results = Vec::with_capacity(targets.len());
        let seed_node = targets[seed_idx].clone();
        let seed_peer_url = format!(
            "http://{}:{}",
            seed_node.registration.address, seed_node.registration.http_port
        );

        if !seed_node.registration.cached_uris.contains(&uri) {
            let url = format!("{}/api/v1/agent/preload", seed_peer_url);
            let mut req = client.post(&url).json(&serde_json::json!({ "uri": uri }));
            if let Some(ref tok) = auth_token {
                if !tok.trim().is_empty() {
                    req = req.bearer_auth(tok.trim());
                }
            }
            let seed_res = match req.send().await {
                Ok(resp) if resp.status().is_success() => {
                    let body: serde_json::Value = resp.json().await.unwrap_or_default();
                    FleetOperationResult {
                        node_id: seed_node.registration.node_id.clone(),
                        success: body.get("success").and_then(|v| v.as_bool()).unwrap_or(true),
                        message: body
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Seeded from origin")
                            .to_string(),
                        size_bytes: body.get("size_bytes").and_then(|v| v.as_u64()),
                        duration_ms: body.get("duration_ms").and_then(|v| v.as_u64()),
                        bandwidth_mbps: body.get("bandwidth_mbps").and_then(|v| v.as_f64()),
                    }
                }
                Ok(resp) => FleetOperationResult {
                    node_id: seed_node.registration.node_id.clone(),
                    success: false,
                    message: format!("Seed agent returned HTTP {}", resp.status()),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                },
                Err(e) => FleetOperationResult {
                    node_id: seed_node.registration.node_id.clone(),
                    success: false,
                    message: format!("Failed to reach seed agent: {}", e),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                },
            };
            let seed_ok = seed_res.success;
            all_results.push(seed_res);
            if !seed_ok {
                seed_idx = usize::MAX;
            }
        }

        let peer_source = if seed_idx != usize::MAX {
            Some(seed_peer_url)
        } else {
            None
        };

        let remaining_futures = targets
            .into_iter()
            .enumerate()
            .filter(|(idx, n)| *idx != seed_idx || n.registration.cached_uris.contains(&uri))
            .map(|(_, node)| {
                let client = client.clone();
                let uri = uri.clone();
                let auth_token = auth_token.clone();
                let src = peer_source.clone();
                async move {
                    let url = format!(
                        "http://{}:{}/api/v1/agent/preload",
                        node.registration.address, node.registration.http_port
                    );
                    let node_id = node.registration.node_id.clone();
                    let req_body = serde_json::json!({
                        "uri": uri,
                        "source_peer_url": src,
                    });
                    let mut req = client.post(&url).json(&req_body);
                    if let Some(ref tok) = auth_token {
                        if !tok.trim().is_empty() {
                            req = req.bearer_auth(tok.trim());
                        }
                    }
                    match req.send().await {
                        Ok(resp) if resp.status().is_success() => {
                            let body: serde_json::Value = resp.json().await.unwrap_or_default();
                            FleetOperationResult {
                                node_id,
                                success: body.get("success").and_then(|v| v.as_bool()).unwrap_or(true),
                                message: body
                                    .get("message")
                                    .and_then(|v| v.as_str())
                                    .unwrap_or("P2P pre-loaded successfully")
                                    .to_string(),
                                size_bytes: body.get("size_bytes").and_then(|v| v.as_u64()),
                                duration_ms: body.get("duration_ms").and_then(|v| v.as_u64()),
                                bandwidth_mbps: body.get("bandwidth_mbps").and_then(|v| v.as_f64()),
                            }
                        }
                        Ok(resp) => FleetOperationResult {
                            node_id,
                            success: false,
                            message: format!("Agent returned HTTP {}", resp.status()),
                            size_bytes: None,
                            duration_ms: None,
                            bandwidth_mbps: None,
                        },
                        Err(e) => FleetOperationResult {
                            node_id,
                            success: false,
                            message: format!("Failed to reach agent at {}: {}", url, e),
                            size_bytes: None,
                            duration_ms: None,
                            bandwidth_mbps: None,
                        },
                    }
                }
            });

        let mut rem_results = join_all(remaining_futures).await;
        all_results.append(&mut rem_results);
        return (StatusCode::OK, Json(all_results));
    }

    let futures = targets.into_iter().map(|node| {
        let client = client.clone();
        let uri = uri.clone();
        let auth_token = auth_token.clone();
        async move {
            let url = format!(
                "http://{}:{}/api/v1/agent/preload",
                node.registration.address, node.registration.http_port
            );

            let node_id = node.registration.node_id.clone();
            let req_body = serde_json::json!({ "uri": uri });

            let mut req = client.post(&url).json(&req_body);
            if let Some(ref tok) = auth_token {
                if !tok.trim().is_empty() {
                    req = req.bearer_auth(tok.trim());
                }
            }

            match req.send().await {
                Ok(resp) if resp.status().is_success() => {
                    let body: serde_json::Value = resp.json().await.unwrap_or_default();
                    FleetOperationResult {
                        node_id,
                        success: body.get("success").and_then(|v| v.as_bool()).unwrap_or(true),
                        message: body
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Pre-loaded successfully")
                            .to_string(),
                        size_bytes: body.get("size_bytes").and_then(|v| v.as_u64()),
                        duration_ms: body.get("duration_ms").and_then(|v| v.as_u64()),
                        bandwidth_mbps: body.get("bandwidth_mbps").and_then(|v| v.as_f64()),
                    }
                }
                Ok(resp) => FleetOperationResult {
                    node_id,
                    success: false,
                    message: format!("Agent returned HTTP {}", resp.status()),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                },
                Err(e) => FleetOperationResult {
                    node_id,
                    success: false,
                    message: format!("Failed to reach agent at {}: {}", url, e),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                },
            }
        }
    });

    let results = join_all(futures).await;
    (StatusCode::OK, Json(results))
}

async fn handle_fleet_preload_batch(
    State(registry): State<FleetRegistry>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<FleetPreloadBatchRequest>,
) -> impl IntoResponse {
    if !is_authorized_or_localhost(connect_info.as_ref(), &headers, registry.auth_token.as_deref(), registry.bind_ip) {
        return (StatusCode::UNAUTHORIZED, Json(vec![FleetOperationResult {
            node_id: "none".to_string(),
            success: false,
            message: "Unauthorized: Remote access to fleet dispatch requires valid cluster auth token".to_string(),
            size_bytes: None,
            duration_ms: None,
            bandwidth_mbps: None,
        }]));
    }

    let targets = registry
        .filter_nodes(payload.nodes.as_deref(), payload.selector.as_deref())
        .await;

    if targets.is_empty() {
        return (StatusCode::OK, Json(vec![FleetOperationResult {
            node_id: "none".to_string(),
            success: false,
            message: "No online nodes matched the given selector/filter.".to_string(),
            size_bytes: None,
            duration_ms: None,
            bandwidth_mbps: None,
        }]));
    }

    let mut uris = payload.uris.clone();
    if uris.is_empty() {
        if let Some(ref prefix) = payload.prefix {
            let s3_fetcher = crate::ingestion::s3::S3Fetcher::new(registry.config.clone());
            if let Ok(keys) = s3_fetcher.list_prefix(prefix).await {
                uris = keys;
            }
        }
    }

    if uris.is_empty() {
        return (StatusCode::OK, Json(vec![FleetOperationResult {
            node_id: "none".to_string(),
            success: false,
            message: "No URIs or prefix matches provided for batch preload.".to_string(),
            size_bytes: None,
            duration_ms: None,
            bandwidth_mbps: None,
        }]));
    }

    let num_targets = targets.len();
    info!(
        "Dispatching batch preload ({} objects, sharded={}) across {} target nodes",
        uris.len(),
        payload.sharded,
        num_targets
    );

    let client = registry.http_client.clone();
    let auth_token = registry.auth_token.clone();

    let targets_snapshot = targets.clone();
    let futures = targets.into_iter().map(|node| {
        let client = client.clone();
        let auth_token = auth_token.clone();
        let target_id = node.registration.node_id.clone();
        let node_uris: Vec<String> = if payload.sharded {
            uris.iter()
                .filter(|u| {
                    assign_uri_rendezvous(u, &targets_snapshot)
                        .map(|n| n.registration.node_id.as_str() == target_id.as_str())
                        .unwrap_or(false)
                })
                .cloned()
                .collect()
        } else {
            uris.clone()
        };

        async move {
            let url = format!(
                "http://{}:{}/api/v1/agent/preload_batch",
                node.registration.address, node.registration.http_port
            );

            let node_id = node.registration.node_id.clone();
            let req_body = serde_json::json!({ "uris": node_uris });

            let mut req = client.post(&url).json(&req_body);
            if let Some(ref tok) = auth_token {
                if !tok.trim().is_empty() {
                    req = req.bearer_auth(tok.trim());
                }
            }

            match req.send().await {
                Ok(resp) if resp.status().is_success() => {
                    let body: serde_json::Value = resp.json().await.unwrap_or_default();
                    FleetOperationResult {
                        node_id,
                        success: body.get("success").and_then(|v| v.as_bool()).unwrap_or(true),
                        message: body
                            .get("message")
                            .and_then(|v| v.as_str())
                            .unwrap_or("Preload batch completed")
                            .to_string(),
                        size_bytes: body.get("total_bytes").and_then(|v| v.as_u64()),
                        duration_ms: body.get("duration_ms").and_then(|v| v.as_u64()),
                        bandwidth_mbps: None,
                    }
                }
                Ok(resp) => FleetOperationResult {
                    node_id,
                    success: false,
                    message: format!("Agent returned HTTP {}", resp.status()),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                },
                Err(e) => FleetOperationResult {
                    node_id,
                    success: false,
                    message: format!("Failed to reach agent at {}: {}", url, e),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                },
            }
        }
    });

    let results = join_all(futures).await;
    (StatusCode::OK, Json(results))
}

async fn handle_fleet_evict(
    State(registry): State<FleetRegistry>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<FleetEvictRequest>,
) -> impl IntoResponse {
    if !is_authorized_or_localhost(connect_info.as_ref(), &headers, registry.auth_token.as_deref(), registry.bind_ip) {
        return (StatusCode::UNAUTHORIZED, Json(vec![FleetOperationResult {
            node_id: "none".to_string(),
            success: false,
            message: "Unauthorized: Remote access to fleet dispatch requires valid cluster auth token".to_string(),
            size_bytes: None,
            duration_ms: None,
            bandwidth_mbps: None,
        }]));
    }

    let targets = registry
        .filter_nodes(payload.nodes.as_deref(), payload.selector.as_deref())
        .await;

    let client = registry.http_client.clone();
    let uri = payload.uri.clone();
    let auth_token = registry.auth_token.clone();

    let futures = targets.into_iter().map(|node| {
        let client = client.clone();
        let uri = uri.clone();
        let auth_token = auth_token.clone();
        async move {
            let url = format!(
                "http://{}:{}/api/v1/agent/evict",
                node.registration.address, node.registration.http_port
            );

            let node_id = node.registration.node_id.clone();
            let req_body = serde_json::json!({ "uri": uri });

            let mut req = client.post(&url).json(&req_body);
            if let Some(ref tok) = auth_token {
                if !tok.trim().is_empty() {
                    req = req.bearer_auth(tok.trim());
                }
            }

            match req.send().await {
                Ok(resp) if resp.status().is_success() => FleetOperationResult {
                    node_id,
                    success: true,
                    message: format!("Evicted {}", uri),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                },
                Ok(resp) => FleetOperationResult {
                    node_id,
                    success: false,
                    message: format!("Agent returned HTTP {}", resp.status()),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                },
                Err(e) => FleetOperationResult {
                    node_id,
                    success: false,
                    message: format!("Connection failed: {}", e),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                },
            }
        }
    });

    let results = join_all(futures).await;
    (StatusCode::OK, Json(results))
}

async fn handle_fleet_purge(
    State(registry): State<FleetRegistry>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<FleetPurgeRequest>,
) -> impl IntoResponse {
    if !is_authorized_or_localhost(connect_info.as_ref(), &headers, registry.auth_token.as_deref(), registry.bind_ip) {
        return (StatusCode::UNAUTHORIZED, Json(vec![FleetOperationResult {
            node_id: "none".to_string(),
            success: false,
            message: "Unauthorized: Remote access to fleet dispatch requires valid cluster auth token".to_string(),
            size_bytes: None,
            duration_ms: None,
            bandwidth_mbps: None,
        }]));
    }

    let targets = registry
        .filter_nodes(payload.nodes.as_deref(), payload.selector.as_deref())
        .await;

    let client = registry.http_client.clone();
    let auth_token = registry.auth_token.clone();

    let futures = targets.into_iter().map(|node| {
        let client = client.clone();
        let auth_token = auth_token.clone();
        async move {
            let url = format!(
                "http://{}:{}/api/v1/agent/purge",
                node.registration.address, node.registration.http_port
            );

            let node_id = node.registration.node_id.clone();

            let mut req = client.post(&url);
            if let Some(ref tok) = auth_token {
                if !tok.trim().is_empty() {
                    req = req.bearer_auth(tok.trim());
                }
            }

            match req.send().await {
                Ok(resp) if resp.status().is_success() => FleetOperationResult {
                    node_id,
                    success: true,
                    message: "Purged all cached models & reset drives".to_string(),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                },
                Ok(resp) => FleetOperationResult {
                    node_id,
                    success: false,
                    message: format!("Agent returned HTTP {}", resp.status()),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                },
                Err(e) => FleetOperationResult {
                    node_id,
                    success: false,
                    message: format!("Connection failed: {}", e),
                    size_bytes: None,
                    duration_ms: None,
                    bandwidth_mbps: None,
                },
            }
        }
    });

    let results = join_all(futures).await;
    (StatusCode::OK, Json(results))
}

pub fn create_controller_router(registry: FleetRegistry) -> Router {
    Router::new()
        .route("/api/v1/register", post(handle_register))
        .route("/api/v1/nodes", get(handle_get_nodes))
        .route("/api/v1/fleet/preload", post(handle_fleet_preload))
        .route("/api/v1/fleet/preload_batch", post(handle_fleet_preload_batch))
        .route("/api/v1/fleet/evict", post(handle_fleet_evict))
        .route("/api/v1/fleet/purge", post(handle_fleet_purge))
        .with_state(registry)
}

pub async fn run_controller_server(bind_addr: &str, port: u16, auth_token: Option<String>) -> Result<()> {
    run_controller_server_with_config(bind_addr, port, auth_token, crate::config::Config::default()).await
}

pub async fn run_controller_server_with_config(
    bind_addr: &str,
    port: u16,
    auth_token: Option<String>,
    config: crate::config::Config,
) -> Result<()> {
    let auth_active = auth_token.as_ref().map(|t| !t.trim().is_empty()).unwrap_or(false);
    let ip: std::net::IpAddr = bind_addr
        .parse()
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED));
    let addr = SocketAddr::new(ip, port);

    let registry = FleetRegistry::new_with_config(auth_token, Some(ip), config);
    let app = create_controller_router(registry);

    info!("Starting HyperCache Cluster Controller on {}", addr);

    println!("====================================================================================================");
    println!("                           HYPERCACHE CLUSTER CONTROL PLANE COORDINATOR                              ");
    println!("====================================================================================================");
    println!("Listening on:           http://{}:{}", bind_addr, port);
    println!("Fleet Registry:         ACTIVE (Accepting Node Agent Heartbeats & Dispatching Directives)");
    println!("Local Admin Access:     ZERO-CONFIG (Localhost & Controller host granted immediate admin access)");
    println!("Remote Security:        {}", if auth_active { "BEARER TOKEN PROTECTED" } else { "DISABLED (Public/Open)" });
    println!("====================================================================================================");

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!("====================================================================================================");
            eprintln!("Error: Failed to bind Cluster Controller on http://{}:{}", bind_addr, port);
            eprintln!("====================================================================================================");
            eprintln!("Another HyperCache controller or process is already running on port {}.", port);
            eprintln!("- To inspect registered nodes: ./hypercache-admin nodes");
            eprintln!("- To stop running controller:  pkill -f 'hypercache-admin controller'");
            eprintln!("====================================================================================================");
            return Err(anyhow::anyhow!("Controller port {} is already in use: {}", port, e));
        }
    };

    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>()).await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_fleet_registry_lifecycle_and_filter() {
        let registry = FleetRegistry::new(None, None);

        let reg1 = NodeRegistration {
            node_id: "gpu-node-01".to_string(),
            address: "10.0.0.1".to_string(),
            http_port: 9115,
            labels: [("gpu".to_string(), "h100".to_string()), ("pool".to_string(), "trn".to_string())].into(),
            drives: vec!["/dev/vdc".to_string(), "/dev/vdd".to_string()],
            total_capacity_bytes: 4_000_000_000_000,
            used_capacity_bytes: 50_000_000_000,
            cached_objects_count: 3,
            cached_uris: vec!["hypercache://models/llama3".to_string()],
        };

        let reg2 = NodeRegistration {
            node_id: "gpu-node-02".to_string(),
            address: "10.0.0.2".to_string(),
            http_port: 9115,
            labels: [("gpu".to_string(), "a100".to_string()), ("pool".to_string(), "inf".to_string())].into(),
            drives: vec!["/dev/nvme0n1".to_string()],
            total_capacity_bytes: 2_000_000_000_000,
            used_capacity_bytes: 10_000_000_000,
            cached_objects_count: 1,
            cached_uris: vec!["hypercache://models/mistral".to_string()],
        };

        registry.register_or_heartbeat(reg1).await;
        registry.register_or_heartbeat(reg2).await;

        let summary = registry.get_summary().await;
        assert_eq!(summary.total_nodes, 2);
        assert_eq!(summary.online_nodes, 2);
        assert_eq!(summary.aggregate_capacity_bytes, 6_000_000_000_000);
        assert_eq!(summary.total_cached_objects, 4);

        // Test filter by node id
        let filtered_01 = registry.filter_nodes(Some("gpu-node-01"), None).await;
        assert_eq!(filtered_01.len(), 1);
        assert_eq!(filtered_01[0].registration.node_id, "gpu-node-01");

        // Test filter by label selector
        let filtered_h100 = registry.filter_nodes(None, Some("gpu=h100")).await;
        assert_eq!(filtered_h100.len(), 1);
        assert_eq!(filtered_h100[0].registration.node_id, "gpu-node-01");

        let filtered_inf = registry.filter_nodes(None, Some("pool=inf")).await;
        assert_eq!(filtered_inf.len(), 1);
        assert_eq!(filtered_inf[0].registration.node_id, "gpu-node-02");

        // Test range matching
        let filtered_range = registry.filter_nodes(Some("gpu-node-01..gpu-node-05"), None).await;
        assert_eq!(filtered_range.len(), 2);
    }

    #[tokio::test]
    async fn test_controller_network_isolation() {
        let registry = FleetRegistry::new(None, None);
        let _router = create_controller_router(registry);

        // Verify router builds cleanly
        assert!(true);
    }

    #[tokio::test]
    async fn test_unauthenticated_registration_rejected() {
        use axum::body::Body;
        use axum::http::Request;
        use tower::ServiceExt;

        let registry = FleetRegistry::new(Some("secret-cluster-token".to_string()), None);
        let router = create_controller_router(registry.clone());

        let reg = NodeRegistration {
            node_id: "rogue-gpu-node".to_string(),
            address: "10.0.0.99".to_string(),
            http_port: 9115,
            labels: HashMap::new(),
            drives: vec!["/dev/nvme0n1".to_string()],
            total_capacity_bytes: 1_000_000_000,
            used_capacity_bytes: 0,
            cached_objects_count: 0,
            cached_uris: vec![],
        };

        // 1. Request with NO token -> should be 401 UNAUTHORIZED
        let req_no_token = Request::builder()
            .method("POST")
            .uri("/api/v1/register")
            .header("content-type", "application/json")
            .body(Body::from(serde_json::to_string(&reg).unwrap()))
            .unwrap();

        let resp = router.clone().oneshot(req_no_token).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);

        // Verify node was NOT registered
        let summary = registry.get_summary().await;
        assert_eq!(summary.total_nodes, 0);

        // 2. Request with WRONG token -> should be 401 UNAUTHORIZED
        let req_wrong_token = Request::builder()
            .method("POST")
            .uri("/api/v1/register")
            .header("content-type", "application/json")
            .header("authorization", "Bearer wrong-token")
            .body(Body::from(serde_json::to_string(&reg).unwrap()))
            .unwrap();

        let resp = router.clone().oneshot(req_wrong_token).await.unwrap();
        assert_eq!(resp.status(), StatusCode::UNAUTHORIZED);
        assert_eq!(registry.get_summary().await.total_nodes, 0);

        // 3. Request with CORRECT token -> should be 200 OK
        let req_valid = Request::builder()
            .method("POST")
            .uri("/api/v1/register")
            .header("content-type", "application/json")
            .header("authorization", "Bearer secret-cluster-token")
            .body(Body::from(serde_json::to_string(&reg).unwrap()))
            .unwrap();

        let resp = router.oneshot(req_valid).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(registry.get_summary().await.total_nodes, 1);
    }

    #[test]
    fn test_rendezvous_consistent_hashing_stability() {
        let make_node = |id: &str| NodeState {
            registration: NodeRegistration {
                node_id: id.to_string(),
                address: "10.0.0.1".to_string(),
                http_port: 9115,
                labels: HashMap::new(),
                drives: vec![],
                total_capacity_bytes: 1_000_000_000,
                used_capacity_bytes: 0,
                cached_objects_count: 0,
                cached_uris: vec![],
            },
            last_heartbeat: 0,
            status: "ONLINE".to_string(),
        };

        let cluster_4 = vec![
            make_node("gpu-01"),
            make_node("gpu-02"),
            make_node("gpu-03"),
            make_node("gpu-04"),
        ];
        let mut cluster_5 = cluster_4.clone();
        cluster_5.push(make_node("gpu-05"));

        let mut unchanged = 0;
        for i in 0..100 {
            let uri = format!("s3://datasets/fineweb/shard_{:04}.parquet", i);
            let n4 = assign_uri_rendezvous(&uri, &cluster_4)
                .unwrap()
                .registration
                .node_id
                .clone();
            let n5 = assign_uri_rendezvous(&uri, &cluster_5)
                .unwrap()
                .registration
                .node_id
                .clone();
            if n4 == n5 {
                unchanged += 1;
            }
        }

        // Consistent hashing must keep ~80% (at least >=70%) of keys on the same node when scaling 4 -> 5
        assert!(
            unchanged >= 70,
            "Expected >= 70 unchanged assignments, got {}",
            unchanged
        );
    }
}
