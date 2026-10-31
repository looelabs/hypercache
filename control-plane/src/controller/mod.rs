use anyhow::Result;
use axum::{
    extract::{ConnectInfo, State},
    http::StatusCode,
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
use tracing::{info, warn};

use crate::auth::{validate_bearer, with_bearer};

/// Seconds without a heartbeat before a node is reported OFFLINE
pub const OFFLINE_AFTER_SECS: i64 = 15;
/// Timeout for short directives (evict / purge); preloads have no overall timeout
const SHORT_DIRECTIVE_TIMEOUT: Duration = Duration::from_secs(120);

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
    /// True when the agent skipped `cached_uris` because its inventory is unchanged
    #[serde(default)]
    pub cached_uris_omitted: bool,
    #[serde(default)]
    pub agent_version: Option<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum NodeStatus {
    #[serde(rename = "ONLINE")]
    Online,
    #[serde(rename = "OFFLINE")]
    Offline,
}

impl std::fmt::Display for NodeStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // pad() so width specifiers like {:<7} work in table output
        f.pad(match self {
            NodeStatus::Online => "ONLINE",
            NodeStatus::Offline => "OFFLINE",
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeState {
    pub registration: NodeRegistration,
    pub last_heartbeat: i64,
    pub status: NodeStatus,
    /// Source IP of the last accepted heartbeat (used to detect node_id takeover)
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_ip: Option<String>,
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

impl FleetOperationResult {
    pub fn failure(node_id: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            node_id: node_id.into(),
            success: false,
            message: message.into(),
            size_bytes: None,
            duration_ms: None,
            bandwidth_mbps: None,
        }
    }

    /// Builds a result from an agent's JSON reply
    fn from_agent_body(node_id: String, body: &serde_json::Value, default_message: &str, size_field: &str) -> Self {
        Self {
            node_id,
            success: body.get("success").and_then(|v| v.as_bool()).unwrap_or(true),
            message: body
                .get("message")
                .and_then(|v| v.as_str())
                .unwrap_or(default_message)
                .to_string(),
            size_bytes: body.get(size_field).and_then(|v| v.as_u64()),
            duration_ms: body.get("duration_ms").and_then(|v| v.as_u64()),
            bandwidth_mbps: body.get("bandwidth_mbps").and_then(|v| v.as_f64()),
        }
    }
}

type FleetResponse = (StatusCode, Json<Vec<FleetOperationResult>>);

fn fleet_error(status: StatusCode, message: impl Into<String>) -> FleetResponse {
    (status, Json(vec![FleetOperationResult::failure("none", message)]))
}

/// Parsed node targeting expression
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NodeTarget {
    /// None = all nodes
    node_patterns: Option<Vec<NodePattern>>,
    labels: Vec<(String, String)>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum NodePattern {
    Exact(String),
    /// Same prefix, numeric suffix range inclusive (e.g. gpu-01..gpu-10)
    Range {
        prefix: String,
        start: u64,
        end: u64,
    },
}

/// Splits "gpu-07" into ("gpu-", 7). Returns None when there is no numeric suffix.
fn split_numeric_suffix(id: &str) -> Option<(&str, u64)> {
    let digits_start = id.rfind(|c: char| !c.is_ascii_digit()).map_or(0, |i| i + 1);
    if digits_start >= id.len() {
        return None;
    }
    id[digits_start..].parse().ok().map(|n| (&id[..digits_start], n))
}

impl NodeTarget {
    /// Parses `--nodes` and `--selector`. Malformed expressions are rejected rather than
    /// silently matching every node (a typo must never turn a purge into a fleet-wide purge).
    pub fn parse(nodes: Option<&str>, selector: Option<&str>) -> std::result::Result<Self, String> {
        let node_patterns = match nodes.map(str::trim).filter(|s| !s.is_empty()) {
            None => None,
            Some("all") | Some("*") => None,
            Some(list) => {
                let mut patterns = Vec::new();
                for part in list.split(',').map(str::trim) {
                    if part.is_empty() {
                        return Err(format!("Invalid --nodes '{}': empty node name", list));
                    }
                    if let Some((a, b)) = part.split_once("..") {
                        let (pa, sa) = split_numeric_suffix(a.trim())
                            .ok_or_else(|| format!("Invalid node range '{}': '{}' has no numeric suffix", part, a))?;
                        let (pb, sb) = split_numeric_suffix(b.trim())
                            .ok_or_else(|| format!("Invalid node range '{}': '{}' has no numeric suffix", part, b))?;
                        if pa != pb {
                            return Err(format!(
                                "Invalid node range '{}': prefixes '{}' and '{}' differ",
                                part, pa, pb
                            ));
                        }
                        if sa > sb {
                            return Err(format!("Invalid node range '{}': start is after end", part));
                        }
                        patterns.push(NodePattern::Range {
                            prefix: pa.to_string(),
                            start: sa,
                            end: sb,
                        });
                    } else {
                        patterns.push(NodePattern::Exact(part.to_string()));
                    }
                }
                Some(patterns)
            }
        };

        let mut labels = Vec::new();
        if let Some(sel) = selector.map(str::trim).filter(|s| !s.is_empty()) {
            for pair in sel.split(',').map(str::trim) {
                match pair.split_once('=') {
                    Some((k, v)) if !k.trim().is_empty() && !v.trim().is_empty() && !v.contains('=') => {
                        labels.push((k.trim().to_string(), v.trim().to_string()));
                    }
                    _ => {
                        return Err(format!(
                            "Invalid --selector '{}': expected comma-separated key=value pairs (e.g. gpu=h100,pool=training)",
                            sel
                        ))
                    }
                }
            }
        }

        Ok(Self { node_patterns, labels })
    }

    /// True when the expression targets every node (no node list and no label selector)
    pub fn is_unrestricted(&self) -> bool {
        self.node_patterns.is_none() && self.labels.is_empty()
    }

    pub fn matches(&self, node: &NodeState) -> bool {
        let id = node.registration.node_id.as_str();
        if let Some(ref patterns) = self.node_patterns {
            let hit = patterns.iter().any(|p| match p {
                NodePattern::Exact(name) => name == id,
                NodePattern::Range { prefix, start, end } => {
                    split_numeric_suffix(id).is_some_and(|(p, n)| p == prefix && n >= *start && n <= *end)
                }
            });
            if !hit {
                return false;
            }
        }
        self.labels
            .iter()
            .all(|(k, v)| node.registration.labels.get(k).map(|x| x.as_str()) == Some(v.as_str()))
    }
}

/// Backward-compatible helper: true if the node matches the given filters. Malformed
/// filters match nothing.
pub fn matches_node(node: &NodeState, nodes_filter: Option<&str>, selector_filter: Option<&str>) -> bool {
    NodeTarget::parse(nodes_filter, selector_filter).is_ok_and(|t| t.matches(node))
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
        // No overall timeout: a multi-GB preload legitimately runs for many minutes.
        // Dead nodes are excluded before dispatch and connects fail fast.
        let http_client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .tcp_keepalive(Duration::from_secs(30))
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

    /// Records a heartbeat. Returns Err(message) if another live node already holds this
    /// node_id from a different source address, and Ok(need_full_inventory) otherwise.
    pub async fn register_or_heartbeat_from(
        &self,
        mut reg: NodeRegistration,
        source_ip: Option<String>,
    ) -> std::result::Result<bool, String> {
        let now = Utc::now().timestamp();
        let mut map = self.nodes.write().await;
        let mut need_full_inventory = false;

        if let Some(existing) = map.get(&reg.node_id) {
            let alive = now - existing.last_heartbeat <= OFFLINE_AFTER_SECS;
            if alive && existing.source_ip.is_some() && source_ip.is_some() && existing.source_ip != source_ip {
                return Err(format!(
                    "node_id '{}' is already registered by a live node at {}; refusing registration from {}",
                    reg.node_id,
                    existing.source_ip.as_deref().unwrap_or("?"),
                    source_ip.as_deref().unwrap_or("?")
                ));
            }
            if reg.cached_uris_omitted {
                reg.cached_uris = existing.registration.cached_uris.clone();
            }
        } else if reg.cached_uris_omitted {
            need_full_inventory = true;
        }
        reg.cached_uris_omitted = false;

        map.insert(
            reg.node_id.clone(),
            NodeState {
                registration: reg,
                last_heartbeat: now,
                status: NodeStatus::Online,
                source_ip,
            },
        );
        Ok(need_full_inventory)
    }

    pub async fn register_or_heartbeat(&self, reg: NodeRegistration) {
        let _ = self.register_or_heartbeat_from(reg, None).await;
    }

    /// Returns the fleet summary, refreshing ONLINE/OFFLINE state and forgetting nodes
    /// that have been offline for longer than `node_gc_secs`.
    pub async fn get_summary(&self) -> FleetSummary {
        let now = Utc::now().timestamp();
        let gc_after = self.config.node_gc_secs as i64;
        let mut map = self.nodes.write().await;

        map.retain(|id, state| {
            let keep = gc_after <= 0 || now - state.last_heartbeat <= gc_after.max(OFFLINE_AFTER_SECS);
            if !keep {
                info!(
                    "Removing node '{}' from fleet registry (no heartbeat for {} s)",
                    id,
                    now - state.last_heartbeat
                );
            }
            keep
        });

        let mut summary = FleetSummary {
            total_nodes: 0,
            online_nodes: 0,
            offline_nodes: 0,
            aggregate_capacity_bytes: 0,
            total_used_bytes: 0,
            total_cached_objects: 0,
            nodes: Vec::with_capacity(map.len()),
        };

        for state in map.values_mut() {
            if now - state.last_heartbeat > OFFLINE_AFTER_SECS {
                state.status = NodeStatus::Offline;
                summary.offline_nodes += 1;
            } else {
                state.status = NodeStatus::Online;
                summary.online_nodes += 1;
            }
            summary.aggregate_capacity_bytes += state.registration.total_capacity_bytes;
            summary.total_used_bytes += state.registration.used_capacity_bytes;
            summary.total_cached_objects += state.registration.cached_objects_count;
            summary.nodes.push(state.clone());
        }

        summary
            .nodes
            .sort_by(|a, b| a.registration.node_id.cmp(&b.registration.node_id));
        summary.total_nodes = summary.nodes.len();
        summary
    }

    /// ONLINE nodes matching the target expression. Offline nodes are never dispatched to.
    pub async fn target_nodes(&self, target: &NodeTarget) -> Vec<NodeState> {
        self.get_summary()
            .await
            .nodes
            .into_iter()
            .filter(|n| n.status == NodeStatus::Online && target.matches(n))
            .collect()
    }

    /// Backward-compatible filter (malformed filters match nothing)
    pub async fn filter_nodes(&self, nodes_filter: Option<&str>, selector_filter: Option<&str>) -> Vec<NodeState> {
        match NodeTarget::parse(nodes_filter, selector_filter) {
            Ok(t) => self.target_nodes(&t).await,
            Err(_) => Vec::new(),
        }
    }

    /// Sends one directive per target node in parallel and collects per-node results.
    async fn dispatch<F>(
        &self,
        targets: Vec<NodeState>,
        path: &str,
        timeout: Option<Duration>,
        build_body: F,
        default_message: &str,
        size_field: &str,
    ) -> Vec<FleetOperationResult>
    where
        F: Fn(&NodeState) -> Option<serde_json::Value>,
    {
        let futures = targets.into_iter().map(|node| {
            let url = format!(
                "http://{}:{}{}",
                node.registration.address, node.registration.http_port, path
            );
            let mut req = self.http_client.post(&url);
            if let Some(body) = build_body(&node) {
                req = req.json(&body);
            }
            if let Some(t) = timeout {
                req = req.timeout(t);
            }
            let req = with_bearer(req, self.auth_token.as_deref());
            let node_id = node.registration.node_id.clone();
            async move {
                match req.send().await {
                    Ok(resp) => {
                        let status = resp.status();
                        let body: serde_json::Value = resp.json().await.unwrap_or_default();
                        let mut result =
                            FleetOperationResult::from_agent_body(node_id, &body, default_message, size_field);
                        if !status.is_success() {
                            result.success = false;
                            if body.get("message").is_none() {
                                result.message = format!("Agent returned HTTP {}", status);
                            }
                        }
                        result
                    }
                    Err(e) => {
                        FleetOperationResult::failure(node_id, format!("Failed to reach agent at {}: {}", url, e))
                    }
                }
            }
        });
        join_all(futures).await
    }
}

/// Deterministic Rendezvous (Highest Random Weight) consistent hashing across active nodes.
/// Ensures that adding or removing a node only reassigns ~1/N of dataset objects.
pub fn assign_uri_rendezvous<'a>(uri: &str, nodes: &'a [NodeState]) -> Option<&'a NodeState> {
    use sha2::{Digest, Sha256};
    nodes.iter().max_by_key(|node| {
        let mut hasher = Sha256::new();
        hasher.update(node.registration.node_id.as_bytes());
        hasher.update(b"::");
        hasher.update(uri.as_bytes());
        let digest = hasher.finalize();
        u64::from_be_bytes(digest[..8].try_into().unwrap())
    })
}

fn is_authorized_or_localhost(
    registry: &FleetRegistry,
    connect_info: Option<&ConnectInfo<SocketAddr>>,
    headers: &axum::http::HeaderMap,
) -> bool {
    if registry.config.allow_local_admin_bypass {
        if let Some(ci) = connect_info {
            if ci.ip().is_loopback() {
                return true;
            }
            if let Some(b_ip) = registry.bind_ip {
                if !b_ip.is_unspecified() && ci.ip() == b_ip {
                    return true;
                }
            }
        }
    }
    validate_bearer(headers, registry.auth_token.as_deref())
}

const UNAUTHORIZED_DISPATCH: &str = "Unauthorized: Remote access to fleet dispatch requires valid cluster auth token";

/// Parses targeting for a request; destructive operations must name their targets explicitly.
fn parse_target(
    nodes: Option<&str>,
    selector: Option<&str>,
    destructive: bool,
) -> std::result::Result<NodeTarget, FleetResponse> {
    let target = NodeTarget::parse(nodes, selector).map_err(|e| fleet_error(StatusCode::BAD_REQUEST, e))?;
    if destructive && nodes.is_none_or(|n| n.trim().is_empty()) && selector.is_none_or(|s| s.trim().is_empty()) {
        return Err(fleet_error(
            StatusCode::BAD_REQUEST,
            "Destructive fleet operations require an explicit target: pass nodes=\"all\", a node list, or a label selector.",
        ));
    }
    Ok(target)
}

// Controller Handlers
/// Picks the address the controller will dial for an agent. Agents bound to a wildcard or
/// loopback address advertise "" / "0.0.0.0" / "127.0.0.1"; the TCP source IP of the
/// registration is then the reachable address (including 127.0.0.1 on a single host).
/// IPv6 addresses are bracketed so they can be used directly in `http://{address}:{port}`.
pub fn resolve_agent_address(advertised: &str, source_ip: std::net::IpAddr) -> String {
    let advertised = advertised.trim();
    let unusable = advertised.is_empty() || advertised == "0.0.0.0" || advertised == "::" || advertised == "[::]";
    let loopback_from_remote =
        (advertised == "127.0.0.1" || advertised == "::1" || advertised == "[::1]") && !source_ip.is_loopback();
    if unusable || loopback_from_remote {
        match source_ip {
            std::net::IpAddr::V4(v4) => v4.to_string(),
            std::net::IpAddr::V6(v6) => format!("[{}]", v6),
        }
    } else if advertised.contains(':') && !advertised.starts_with('[') {
        format!("[{}]", advertised)
    } else {
        advertised.to_string()
    }
}

async fn handle_register(
    State(registry): State<FleetRegistry>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: axum::http::HeaderMap,
    Json(mut payload): Json<NodeRegistration>,
) -> (StatusCode, Json<serde_json::Value>) {
    if !validate_bearer(&headers, registry.auth_token.as_deref()) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "Unauthorized: Invalid or missing cluster auth token" })),
        );
    }

    let source_ip = connect_info.as_ref().map(|ConnectInfo(addr)| addr.ip().to_canonical());
    if let Some(ip) = source_ip {
        payload.address = resolve_agent_address(&payload.address, ip);
    }

    match registry
        .register_or_heartbeat_from(payload, source_ip.map(|ip| ip.to_string()))
        .await
    {
        Ok(need_full_inventory) => (
            StatusCode::OK,
            Json(serde_json::json!({ "status": "OK", "need_full_inventory": need_full_inventory })),
        ),
        Err(msg) => {
            warn!("{}", msg);
            (StatusCode::CONFLICT, Json(serde_json::json!({ "error": msg })))
        }
    }
}

async fn handle_get_nodes(
    State(registry): State<FleetRegistry>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: axum::http::HeaderMap,
) -> (StatusCode, Json<serde_json::Value>) {
    if !is_authorized_or_localhost(&registry, connect_info.as_ref(), &headers) {
        return (
            StatusCode::UNAUTHORIZED,
            Json(
                serde_json::json!({ "error": "Unauthorized: Remote access to cluster topology requires valid cluster auth token" }),
            ),
        );
    }

    let summary = registry.get_summary().await;
    (StatusCode::OK, Json(serde_json::to_value(summary).unwrap_or_default()))
}

async fn handle_fleet_preload(
    State(registry): State<FleetRegistry>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<FleetPreloadRequest>,
) -> FleetResponse {
    if !is_authorized_or_localhost(&registry, connect_info.as_ref(), &headers) {
        return fleet_error(StatusCode::UNAUTHORIZED, UNAUTHORIZED_DISPATCH);
    }
    let target = match parse_target(payload.nodes.as_deref(), payload.selector.as_deref(), false) {
        Ok(t) => t,
        Err(resp) => return resp,
    };
    let targets = registry.target_nodes(&target).await;
    if targets.is_empty() {
        return fleet_error(StatusCode::OK, "No online nodes matched the given selector/filter.");
    }

    info!(
        "Dispatching fleet preload for '{}' across {} target nodes",
        payload.uri,
        targets.len()
    );

    let uri = payload.uri.clone();
    let use_p2p = (payload.p2p || registry.config.p2p_preload_enabled) && targets.len() > 1;
    if !use_p2p {
        let results = registry
            .dispatch(
                targets,
                "/api/v1/agent/preload",
                None,
                |_| Some(serde_json::json!({ "uri": uri })),
                "Pre-loaded successfully",
                "size_bytes",
            )
            .await;
        return (StatusCode::OK, Json(results));
    }

    // P2P: one node pulls from origin (or already has it), the rest pull from that peer
    let seed_idx = targets
        .iter()
        .position(|n| n.registration.cached_uris.contains(&uri))
        .unwrap_or(0);
    let seed_node = targets[seed_idx].clone();
    let seed_already_cached = seed_node.registration.cached_uris.contains(&uri);
    let mut all_results = Vec::with_capacity(targets.len());

    let seed_ok = if seed_already_cached {
        true
    } else {
        let mut seed_result = registry
            .dispatch(
                vec![seed_node.clone()],
                "/api/v1/agent/preload",
                None,
                |_| Some(serde_json::json!({ "uri": uri })),
                "Seeded from origin",
                "size_bytes",
            )
            .await;
        let ok = seed_result.first().is_some_and(|r| r.success);
        all_results.append(&mut seed_result);
        ok
    };

    let peer_source = seed_ok.then(|| {
        format!(
            "http://{}:{}",
            seed_node.registration.address, seed_node.registration.http_port
        )
    });

    // Remaining nodes; an already-cached seed is still reported (as a cache hit)
    let remaining: Vec<NodeState> = targets
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| *idx != seed_idx || seed_already_cached)
        .map(|(_, n)| n)
        .collect();
    let seed_id = seed_node.registration.node_id.clone();
    let mut rest = registry
        .dispatch(
            remaining,
            "/api/v1/agent/preload",
            None,
            |n| {
                let src = if n.registration.node_id == seed_id {
                    None
                } else {
                    peer_source.clone()
                };
                Some(serde_json::json!({ "uri": uri, "source_peer_url": src }))
            },
            "P2P pre-loaded successfully",
            "size_bytes",
        )
        .await;
    all_results.append(&mut rest);
    (StatusCode::OK, Json(all_results))
}

async fn handle_fleet_preload_batch(
    State(registry): State<FleetRegistry>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<FleetPreloadBatchRequest>,
) -> FleetResponse {
    if !is_authorized_or_localhost(&registry, connect_info.as_ref(), &headers) {
        return fleet_error(StatusCode::UNAUTHORIZED, UNAUTHORIZED_DISPATCH);
    }
    let target = match parse_target(payload.nodes.as_deref(), payload.selector.as_deref(), false) {
        Ok(t) => t,
        Err(resp) => return resp,
    };
    let targets = registry.target_nodes(&target).await;
    if targets.is_empty() {
        return fleet_error(StatusCode::OK, "No online nodes matched the given selector/filter.");
    }

    let mut uris = payload.uris.clone();
    if uris.is_empty() {
        if let Some(ref prefix) = payload.prefix {
            let s3_fetcher = crate::ingestion::s3::S3Fetcher::new(registry.config.clone());
            match s3_fetcher.list_prefix(prefix).await {
                Ok(keys) => uris = keys,
                Err(e) => {
                    return fleet_error(
                        StatusCode::BAD_GATEWAY,
                        format!("Failed to list prefix '{}': {:#}", prefix, e),
                    );
                }
            }
        }
    }
    if uris.is_empty() {
        return fleet_error(StatusCode::OK, "No URIs or prefix matches provided for batch preload.");
    }

    info!(
        "Dispatching batch preload ({} objects, sharded={}) across {} target nodes",
        uris.len(),
        payload.sharded,
        targets.len()
    );

    let assignments: HashMap<String, Vec<String>> = if payload.sharded {
        let mut map: HashMap<String, Vec<String>> = HashMap::new();
        for u in &uris {
            if let Some(n) = assign_uri_rendezvous(u, &targets) {
                map.entry(n.registration.node_id.clone()).or_default().push(u.clone());
            }
        }
        map
    } else {
        HashMap::new()
    };

    let results = registry
        .dispatch(
            targets,
            "/api/v1/agent/preload_batch",
            None,
            |n| {
                let node_uris = if payload.sharded {
                    assignments.get(&n.registration.node_id).cloned().unwrap_or_default()
                } else {
                    uris.clone()
                };
                Some(serde_json::json!({ "uris": node_uris }))
            },
            "Preload batch completed",
            "total_bytes",
        )
        .await;
    (StatusCode::OK, Json(results))
}

async fn handle_fleet_evict(
    State(registry): State<FleetRegistry>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<FleetEvictRequest>,
) -> FleetResponse {
    if !is_authorized_or_localhost(&registry, connect_info.as_ref(), &headers) {
        return fleet_error(StatusCode::UNAUTHORIZED, UNAUTHORIZED_DISPATCH);
    }
    let target = match parse_target(payload.nodes.as_deref(), payload.selector.as_deref(), true) {
        Ok(t) => t,
        Err(resp) => return resp,
    };
    let targets = registry.target_nodes(&target).await;
    let uri = payload.uri.clone();
    let results = registry
        .dispatch(
            targets,
            "/api/v1/agent/evict",
            Some(SHORT_DIRECTIVE_TIMEOUT),
            |_| Some(serde_json::json!({ "uri": uri })),
            "Evicted",
            "size_bytes",
        )
        .await;
    (StatusCode::OK, Json(results))
}

async fn handle_fleet_purge(
    State(registry): State<FleetRegistry>,
    connect_info: Option<ConnectInfo<SocketAddr>>,
    headers: axum::http::HeaderMap,
    Json(payload): Json<FleetPurgeRequest>,
) -> FleetResponse {
    if !is_authorized_or_localhost(&registry, connect_info.as_ref(), &headers) {
        return fleet_error(StatusCode::UNAUTHORIZED, UNAUTHORIZED_DISPATCH);
    }
    let target = match parse_target(payload.nodes.as_deref(), payload.selector.as_deref(), true) {
        Ok(t) => t,
        Err(resp) => return resp,
    };
    let targets = registry.target_nodes(&target).await;
    let results = registry
        .dispatch(
            targets,
            "/api/v1/agent/purge",
            Some(SHORT_DIRECTIVE_TIMEOUT),
            |_| None,
            "Purged all cached models & reset drives",
            "size_bytes",
        )
        .await;
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
    let auth_active = crate::auth::effective_token(auth_token.as_deref()).is_some();
    let ip: std::net::IpAddr = bind_addr
        .parse()
        .unwrap_or(std::net::IpAddr::V4(std::net::Ipv4Addr::UNSPECIFIED));
    let addr = SocketAddr::new(ip, port);
    let local_bypass = config.allow_local_admin_bypass;

    let registry = FleetRegistry::new_with_config(auth_token, Some(ip), config);
    let app = create_controller_router(registry);

    info!("Starting HyperCache Cluster Controller on {}", addr);

    println!("====================================================================================================");
    println!("                           HYPERCACHE CLUSTER CONTROL PLANE COORDINATOR                              ");
    println!("====================================================================================================");
    println!("Listening on:           http://{}:{}", bind_addr, port);
    println!("Fleet Registry:         ACTIVE (Accepting Node Agent Heartbeats & Dispatching Directives)");
    if local_bypass {
        println!("Local Admin Access:     ZERO-CONFIG (Localhost & Controller host granted immediate admin access)");
    } else {
        println!("Local Admin Access:     TOKEN REQUIRED (allow_local_admin_bypass = false)");
    }
    println!(
        "Remote Security:        {}",
        if auth_active {
            "BEARER TOKEN PROTECTED"
        } else {
            "DISABLED (Public/Open)"
        }
    );
    println!("====================================================================================================");
    if !auth_active && !ip.is_loopback() {
        warn!(
            "Controller is listening on {} without cluster_auth_token: any host that can reach it can register nodes \
             and dispatch fleet operations. Set cluster_auth_token for multi-node deployments.",
            addr
        );
    }

    let listener = match tokio::net::TcpListener::bind(addr).await {
        Ok(l) => l,
        Err(e) => {
            eprintln!(
                "===================================================================================================="
            );
            eprintln!(
                "Error: Failed to bind Cluster Controller on http://{}:{}",
                bind_addr, port
            );
            eprintln!(
                "===================================================================================================="
            );
            eprintln!(
                "Another HyperCache controller or process is already running on port {}.",
                port
            );
            eprintln!("- To inspect registered nodes: ./hypercache-admin nodes");
            eprintln!("- To stop running controller:  pkill -f 'hypercache-admin controller'");
            eprintln!(
                "===================================================================================================="
            );
            return Err(anyhow::anyhow!("Controller port {} is already in use: {}", port, e));
        }
    };

    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(crate::server::shutdown_signal())
        .await?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt;

    fn reg(id: &str, labels: &[(&str, &str)]) -> NodeRegistration {
        NodeRegistration {
            node_id: id.to_string(),
            address: "10.0.0.1".to_string(),
            http_port: 9115,
            labels: labels.iter().map(|(k, v)| (k.to_string(), v.to_string())).collect(),
            drives: vec!["/dev/nvme0n1".to_string()],
            total_capacity_bytes: 1_000_000_000,
            used_capacity_bytes: 0,
            cached_objects_count: 0,
            cached_uris: vec![],
            cached_uris_omitted: false,
            agent_version: None,
        }
    }

    fn state(id: &str) -> NodeState {
        NodeState {
            registration: reg(id, &[]),
            last_heartbeat: 0,
            status: NodeStatus::Online,
            source_ip: None,
        }
    }

    #[tokio::test]
    async fn test_fleet_registry_lifecycle_and_filter() {
        let registry = FleetRegistry::new(None, None);

        let mut reg1 = reg("gpu-node-01", &[("gpu", "h100"), ("pool", "trn")]);
        reg1.total_capacity_bytes = 4_000_000_000_000;
        reg1.cached_objects_count = 3;
        let mut reg2 = reg("gpu-node-02", &[("gpu", "a100"), ("pool", "inf")]);
        reg2.total_capacity_bytes = 2_000_000_000_000;
        reg2.cached_objects_count = 1;

        registry.register_or_heartbeat(reg1).await;
        registry.register_or_heartbeat(reg2).await;

        let summary = registry.get_summary().await;
        assert_eq!(summary.total_nodes, 2);
        assert_eq!(summary.online_nodes, 2);
        assert_eq!(summary.aggregate_capacity_bytes, 6_000_000_000_000);
        assert_eq!(summary.total_cached_objects, 4);

        let filtered_01 = registry.filter_nodes(Some("gpu-node-01"), None).await;
        assert_eq!(filtered_01.len(), 1);
        assert_eq!(filtered_01[0].registration.node_id, "gpu-node-01");

        let filtered_h100 = registry.filter_nodes(None, Some("gpu=h100")).await;
        assert_eq!(filtered_h100.len(), 1);
        assert_eq!(filtered_h100[0].registration.node_id, "gpu-node-01");

        let filtered_inf = registry.filter_nodes(None, Some("pool=inf")).await;
        assert_eq!(filtered_inf.len(), 1);
        assert_eq!(filtered_inf[0].registration.node_id, "gpu-node-02");

        let filtered_range = registry.filter_nodes(Some("gpu-node-01..gpu-node-05"), None).await;
        assert_eq!(filtered_range.len(), 2);
    }

    #[test]
    fn test_numeric_node_ranges() {
        let t = NodeTarget::parse(Some("gpu-2..gpu-10"), None).unwrap();
        assert!(t.matches(&state("gpu-2")));
        assert!(t.matches(&state("gpu-7")));
        assert!(t.matches(&state("gpu-10")));
        assert!(!t.matches(&state("gpu-1")));
        assert!(!t.matches(&state("gpu-11")));
        assert!(!t.matches(&state("cpu-5")));

        // Lexicographic comparison would wrongly include gpu-10 in gpu-1..gpu-3
        let t = NodeTarget::parse(Some("gpu-1..gpu-3"), None).unwrap();
        assert!(!t.matches(&state("gpu-10")));
        assert!(t.matches(&state("gpu-3")));

        // Zero padding is compared numerically
        let t = NodeTarget::parse(Some("node-01..node-08"), None).unwrap();
        assert!(t.matches(&state("node-8")));
        assert!(t.matches(&state("node-05")));

        assert!(NodeTarget::parse(Some("gpu-a..gpu-b"), None).is_err());
        assert!(NodeTarget::parse(Some("gpu-1..cpu-3"), None).is_err());
        assert!(NodeTarget::parse(Some("gpu-5..gpu-2"), None).is_err());
    }

    #[test]
    fn test_malformed_selector_is_rejected_not_wildcard() {
        // "--selector gpu" (missing value) must never match the whole fleet
        assert!(NodeTarget::parse(None, Some("gpu")).is_err());
        assert!(NodeTarget::parse(None, Some("gpu=")).is_err());
        assert!(NodeTarget::parse(None, Some("=h100")).is_err());
        assert!(NodeTarget::parse(None, Some("gpu=h100,pool")).is_err());
        assert!(NodeTarget::parse(None, Some("a=b=c")).is_err());
        assert!(!matches_node(&state("n1"), None, Some("gpu")));

        let t = NodeTarget::parse(Some("all"), Some(" gpu = h100 , pool=trn ")).unwrap();
        assert_eq!(t.labels.len(), 2);
        assert!(!t.is_unrestricted());
        assert!(NodeTarget::parse(Some("all"), None).unwrap().is_unrestricted());
    }

    async fn post_json(
        router: &Router,
        uri: &str,
        body: serde_json::Value,
        token: Option<&str>,
    ) -> (StatusCode, serde_json::Value) {
        let mut req = Request::builder()
            .method("POST")
            .uri(uri)
            .header("content-type", "application/json");
        if let Some(t) = token {
            req = req.header("authorization", format!("Bearer {}", t));
        }
        let resp = router
            .clone()
            .oneshot(req.body(Body::from(body.to_string())).unwrap())
            .await
            .unwrap();
        let status = resp.status();
        let bytes = axum::body::to_bytes(resp.into_body(), usize::MAX).await.unwrap();
        (status, serde_json::from_slice(&bytes).unwrap_or_default())
    }

    #[tokio::test]
    async fn test_unauthenticated_registration_rejected() {
        let registry = FleetRegistry::new(Some("secret-cluster-token".to_string()), None);
        let router = create_controller_router(registry.clone());
        let body = serde_json::to_value(reg("rogue-gpu-node", &[])).unwrap();

        let (status, _) = post_json(&router, "/api/v1/register", body.clone(), None).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(registry.get_summary().await.total_nodes, 0);

        let (status, _) = post_json(&router, "/api/v1/register", body.clone(), Some("wrong-token")).await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        assert_eq!(registry.get_summary().await.total_nodes, 0);

        let (status, _) = post_json(&router, "/api/v1/register", body, Some("secret-cluster-token")).await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(registry.get_summary().await.total_nodes, 1);
    }

    #[tokio::test]
    async fn test_fleet_dispatch_requires_auth_without_connect_info() {
        // Without connection info (e.g. behind a proxy) no loopback bypass applies
        let registry = FleetRegistry::new(Some("tok".to_string()), None);
        let router = create_controller_router(registry);
        let (status, _) = post_json(
            &router,
            "/api/v1/fleet/purge",
            serde_json::json!({"nodes": "all"}),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::UNAUTHORIZED);
        let (status, _) = post_json(
            &router,
            "/api/v1/fleet/purge",
            serde_json::json!({"nodes": "all"}),
            Some("tok"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
    }

    #[test]
    fn test_local_bypass_can_be_disabled() {
        let loopback = ConnectInfo(SocketAddr::from(([127, 0, 0, 1], 5555)));
        let headers = axum::http::HeaderMap::new();

        let open = FleetRegistry::new(Some("tok".to_string()), None);
        assert!(is_authorized_or_localhost(&open, Some(&loopback), &headers));

        let config = crate::config::Config {
            allow_local_admin_bypass: false,
            ..crate::config::Config::default()
        };
        let strict = FleetRegistry::new_with_config(Some("tok".to_string()), None, config);
        assert!(!is_authorized_or_localhost(&strict, Some(&loopback), &headers));
    }

    #[tokio::test]
    async fn test_destructive_ops_reject_bad_or_missing_targets() {
        let registry = FleetRegistry::new(None, None);
        registry.register_or_heartbeat(reg("n1", &[("gpu", "h100")])).await;
        let router = create_controller_router(registry);

        let (status, body) = post_json(
            &router,
            "/api/v1/fleet/purge",
            serde_json::json!({"selector": "gpu"}),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
        assert!(body[0]["message"].as_str().unwrap().contains("selector"));

        let (status, _) = post_json(&router, "/api/v1/fleet/purge", serde_json::json!({}), None).await;
        assert_eq!(status, StatusCode::BAD_REQUEST);

        let (status, _) = post_json(
            &router,
            "/api/v1/fleet/evict",
            serde_json::json!({"uri": "x", "nodes": "gpu-9..gpu-1"}),
            None,
        )
        .await;
        assert_eq!(status, StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn test_offline_nodes_are_not_targeted_and_eventually_forgotten() {
        let config = crate::config::Config {
            node_gc_secs: 60,
            ..crate::config::Config::default()
        };
        let registry = FleetRegistry::new_with_config(None, None, config);
        registry.register_or_heartbeat(reg("alive", &[])).await;
        registry.register_or_heartbeat(reg("stale", &[])).await;
        registry.register_or_heartbeat(reg("dead", &[])).await;
        {
            let mut map = registry.nodes.write().await;
            map.get_mut("stale").unwrap().last_heartbeat -= 30;
            map.get_mut("dead").unwrap().last_heartbeat -= 3600;
        }

        let summary = registry.get_summary().await;
        assert_eq!(
            summary.total_nodes, 2,
            "node offline beyond node_gc_secs must be removed"
        );
        assert_eq!(summary.offline_nodes, 1);

        let targets = registry.filter_nodes(Some("all"), None).await;
        assert_eq!(targets.len(), 1);
        assert_eq!(targets[0].registration.node_id, "alive");
    }

    #[test]
    fn test_agent_address_resolution() {
        use std::net::IpAddr;
        let lo: IpAddr = "127.0.0.1".parse().unwrap();
        let remote: IpAddr = "10.1.2.3".parse().unwrap();
        let v6: IpAddr = "fd00::5".parse().unwrap();
        // Single-host cluster: agent bound to 127.0.0.1 / 0.0.0.0 must be dialed on loopback
        assert_eq!(resolve_agent_address("", lo), "127.0.0.1");
        assert_eq!(resolve_agent_address("0.0.0.0", lo), "127.0.0.1");
        assert_eq!(resolve_agent_address("127.0.0.1", lo), "127.0.0.1");
        // Remote agents are dialed on their source IP unless they advertise a real address
        assert_eq!(resolve_agent_address("", remote), "10.1.2.3");
        assert_eq!(resolve_agent_address("127.0.0.1", remote), "10.1.2.3");
        assert_eq!(resolve_agent_address("10.9.9.9", remote), "10.9.9.9");
        // IPv6 is bracketed for URL use
        assert_eq!(resolve_agent_address("", v6), "[fd00::5]");
        assert_eq!(resolve_agent_address("fd00::7", remote), "[fd00::7]");
        assert_eq!(resolve_agent_address("[fd00::7]", remote), "[fd00::7]");
    }

    #[tokio::test]
    async fn test_node_id_takeover_from_other_address_rejected() {
        let registry = FleetRegistry::new(None, None);
        assert!(registry
            .register_or_heartbeat_from(reg("gpu-01", &[]), Some("10.0.0.1".into()))
            .await
            .is_ok());
        assert!(registry
            .register_or_heartbeat_from(reg("gpu-01", &[]), Some("10.0.0.1".into()))
            .await
            .is_ok());
        assert!(registry
            .register_or_heartbeat_from(reg("gpu-01", &[]), Some("10.6.6.6".into()))
            .await
            .is_err());

        // Once the original node has gone offline, a replacement may take the id
        registry.nodes.write().await.get_mut("gpu-01").unwrap().last_heartbeat -= 60;
        assert!(registry
            .register_or_heartbeat_from(reg("gpu-01", &[]), Some("10.6.6.6".into()))
            .await
            .is_ok());
    }

    #[tokio::test]
    async fn test_omitted_inventory_keeps_previous_uris() {
        let registry = FleetRegistry::new(None, None);
        let mut first = reg("n1", &[]);
        first.cached_uris = vec!["s3://a".into()];
        assert_eq!(registry.register_or_heartbeat_from(first, None).await, Ok(false));

        let mut beat = reg("n1", &[]);
        beat.cached_uris_omitted = true;
        assert_eq!(registry.register_or_heartbeat_from(beat.clone(), None).await, Ok(false));
        assert_eq!(
            registry.get_summary().await.nodes[0].registration.cached_uris,
            vec!["s3://a".to_string()]
        );

        // An unknown node (e.g. after controller restart) is asked for its full inventory
        beat.node_id = "n2".into();
        assert_eq!(registry.register_or_heartbeat_from(beat, None).await, Ok(true));
    }

    #[tokio::test]
    async fn test_dispatch_reaches_live_agent_and_reports_failures() {
        // A minimal fake agent that records the auth header
        use std::sync::Mutex as StdMutex;
        let seen: Arc<StdMutex<Vec<String>>> = Arc::new(StdMutex::new(Vec::new()));
        let seen_c = Arc::clone(&seen);
        let app = Router::new().route(
            "/api/v1/agent/evict",
            post(move |headers: axum::http::HeaderMap| {
                let seen = Arc::clone(&seen_c);
                async move {
                    seen.lock().unwrap().push(
                        headers
                            .get("authorization")
                            .and_then(|v| v.to_str().ok())
                            .unwrap_or("")
                            .to_string(),
                    );
                    Json(serde_json::json!({"success": true, "message": "evicted"}))
                }
            }),
        );
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let port = listener.local_addr().unwrap().port();
        tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });

        let registry = FleetRegistry::new(Some("fleet-tok".to_string()), None);
        let mut live = reg("live", &[]);
        live.address = "127.0.0.1".into();
        live.http_port = port;
        let mut unreachable = reg("unreachable", &[]);
        unreachable.address = "127.0.0.1".into();
        unreachable.http_port = 1; // nothing listens here
        registry.register_or_heartbeat(live).await;
        registry.register_or_heartbeat(unreachable).await;

        let router = create_controller_router(registry);
        let (status, body) = post_json(
            &router,
            "/api/v1/fleet/evict",
            serde_json::json!({"uri": "s3://m", "nodes": "all"}),
            Some("fleet-tok"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let results: Vec<FleetOperationResult> = serde_json::from_value(body).unwrap();
        assert_eq!(results.len(), 2);
        let live_res = results.iter().find(|r| r.node_id == "live").unwrap();
        assert!(live_res.success);
        assert!(!results.iter().find(|r| r.node_id == "unreachable").unwrap().success);
        assert_eq!(seen.lock().unwrap().as_slice(), ["Bearer fleet-tok"]);
    }

    #[test]
    fn test_rendezvous_consistent_hashing_stability() {
        let cluster_4: Vec<NodeState> = ["gpu-01", "gpu-02", "gpu-03", "gpu-04"]
            .iter()
            .map(|i| state(i))
            .collect();
        let mut cluster_5 = cluster_4.clone();
        cluster_5.push(state("gpu-05"));

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
        assert!(
            unchanged >= 70,
            "Expected >= 70 unchanged assignments, got {}",
            unchanged
        );
    }

    #[test]
    fn test_node_state_wire_format_is_backward_compatible() {
        // FleetSummary JSON from a v0.9.0 controller (string status, no source_ip/agent fields)
        let legacy = r#"{"registration":{"node_id":"n","address":"1.2.3.4","http_port":9115,"drives":[],
            "total_capacity_bytes":1,"used_capacity_bytes":0,"cached_objects_count":0},
            "last_heartbeat":5,"status":"ONLINE"}"#;
        let s: NodeState = serde_json::from_str(legacy).unwrap();
        assert_eq!(s.status, NodeStatus::Online);
        let out = serde_json::to_string(&s).unwrap();
        assert!(out.contains("\"status\":\"ONLINE\""));
    }
}
