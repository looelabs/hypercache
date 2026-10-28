use anyhow::{Context, Result};
use reqwest::Client;
use std::time::{Duration, Instant};

use crate::controller::{
    FleetEvictRequest, FleetOperationResult, FleetPreloadRequest, FleetPurgeRequest, FleetSummary,
};

pub struct FleetClient {
    client: Client,
    controller_url: String,
    auth_token: Option<String>,
}

impl FleetClient {
    pub fn new(controller_url: String, auth_token: Option<String>) -> Self {
        // No total request timeout: fleet preloads of large models can legitimately run for many
        // minutes. The controller bounds per-node work; here we only bound connection setup.
        let client = Client::builder()
            .connect_timeout(Duration::from_secs(10))
            .build()
            .unwrap_or_else(|_| Client::new());

        Self {
            client,
            controller_url: controller_url.trim_end_matches('/').to_string(),
            auth_token,
        }
    }

    fn authorize(&self, req: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        crate::auth::with_bearer(req, self.auth_token.as_deref())
    }

    /// Turns a non-2xx controller response into a readable error (the controller returns
    /// `{"error": "..."}` or a list of per-node failures).
    async fn error_from(resp: reqwest::Response, url: &str) -> anyhow::Error {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        let detail = serde_json::from_str::<serde_json::Value>(&body)
            .ok()
            .and_then(|v| {
                v.get("error")
                    .or_else(|| v.get("message"))
                    .and_then(|m| m.as_str().map(str::to_string))
                    .or_else(|| {
                        v.as_array().map(|arr| {
                            arr.iter()
                                .filter_map(|r| r.get("message").and_then(|m| m.as_str()))
                                .collect::<Vec<_>>()
                                .join("; ")
                        })
                    })
            })
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| body.trim().to_string());
        if status == reqwest::StatusCode::UNAUTHORIZED {
            anyhow::anyhow!(
                "Controller at {} rejected the request (HTTP 401). Set --token or HYPERCACHE_AUTH_TOKEN to the cluster_auth_token. {}",
                url,
                detail
            )
        } else {
            anyhow::anyhow!("Controller at {} returned HTTP {}: {}", url, status, detail)
        }
    }

    async fn post_fleet<T: serde::Serialize>(
        &self,
        path: &str,
        payload: &T,
        what: &str,
    ) -> Result<Vec<FleetOperationResult>> {
        let url = format!("{}{}", self.controller_url, path);
        let resp = self
            .authorize(self.client.post(&url).json(payload))
            .send()
            .await
            .with_context(|| format!("Failed to dispatch {} to Controller at {}", what, url))?;
        if !resp.status().is_success() {
            return Err(Self::error_from(resp, &url).await);
        }
        resp.json()
            .await
            .with_context(|| format!("Controller returned an unreadable {} response", what))
    }

    pub async fn get_nodes(&self) -> Result<FleetSummary> {
        let url = format!("{}/api/v1/nodes", self.controller_url);
        let resp = self
            .authorize(self.client.get(&url))
            .send()
            .await
            .with_context(|| format!("Failed to connect to HyperCache Controller at {}", url))?;

        if !resp.status().is_success() {
            return Err(Self::error_from(resp, &url).await);
        }

        let summary: FleetSummary = resp.json().await?;
        Ok(summary)
    }

    pub async fn preload(
        &self,
        uri: &str,
        nodes: Option<String>,
        selector: Option<String>,
    ) -> Result<Vec<FleetOperationResult>> {
        let payload = FleetPreloadRequest {
            uri: uri.to_string(),
            nodes,
            selector,
            p2p: false,
        };
        self.post_fleet("/api/v1/fleet/preload", &payload, "preload").await
    }

    pub async fn preload_batch(
        &self,
        prefix: Option<String>,
        uris: Vec<String>,
        nodes: Option<String>,
        selector: Option<String>,
        sharded: bool,
    ) -> Result<Vec<FleetOperationResult>> {
        let payload = crate::controller::FleetPreloadBatchRequest {
            uris,
            prefix,
            nodes,
            selector,
            sharded,
        };
        self.post_fleet("/api/v1/fleet/preload_batch", &payload, "preload_batch")
            .await
    }

    pub async fn evict(
        &self,
        uri: &str,
        nodes: Option<String>,
        selector: Option<String>,
    ) -> Result<Vec<FleetOperationResult>> {
        let payload = FleetEvictRequest {
            uri: uri.to_string(),
            nodes,
            selector,
        };
        self.post_fleet("/api/v1/fleet/evict", &payload, "evict").await
    }

    pub async fn purge(&self, nodes: Option<String>, selector: Option<String>) -> Result<Vec<FleetOperationResult>> {
        let payload = FleetPurgeRequest { nodes, selector };
        self.post_fleet("/api/v1/fleet/purge", &payload, "purge").await
    }
}

pub async fn run_fleet_nodes(controller_url: &str, auth_token: Option<String>, json_output: bool) -> Result<()> {
    let client = FleetClient::new(controller_url.to_string(), auth_token);
    let summary = client.get_nodes().await?;

    if json_output {
        println!("{}", serde_json::to_string_pretty(&summary)?);
        return Ok(());
    }

    println!("====================================================================================================");
    println!("                               HYPERCACHE FLEET TOPOLOGY & INVENTORY                                ");
    println!("====================================================================================================");
    println!("Controller:  {}", controller_url);
    println!(
        "Fleet State: {}/{} Nodes Online | Aggregate Pool: {} | Active Footprint: {} ({} models)",
        summary.online_nodes,
        summary.total_nodes,
        format_bytes(summary.aggregate_capacity_bytes),
        format_bytes(summary.total_used_bytes),
        summary.total_cached_objects
    );
    println!("----------------------------------------------------------------------------------------------------");
    println!(
        "{:<20} | {:<7} | {:<20} | {:<24} | {:>10} | {:>14}",
        "NODE ID", "STATUS", "LABELS", "STRIPED NVMe DRIVES", "CAPACITY", "CACHED FOOTPRINT"
    );
    println!("----------------------------------------------------------------------------------------------------");

    if summary.nodes.is_empty() {
        println!("No active nodes registered with Controller. Start a daemon with controller_url configured.");
    } else {
        for node in &summary.nodes {
            let label_str = if node.registration.labels.is_empty() {
                "-".to_string()
            } else {
                node.registration
                    .labels
                    .iter()
                    .map(|(k, v)| format!("{}={}", k, v))
                    .collect::<Vec<_>>()
                    .join(",")
            };

            let drives_str = node.registration.drives.join(", ");
            let used_str = format!(
                "{} ({} obj)",
                format_bytes(node.registration.used_capacity_bytes),
                node.registration.cached_objects_count
            );

            println!(
                "{:<20} | {:<7} | {:<20} | {:<24} | {:>10} | {:>14}",
                node.registration.node_id,
                node.status,
                truncate_str(&label_str, 20),
                truncate_str(&drives_str, 24),
                format_bytes(node.registration.total_capacity_bytes),
                used_str
            );
        }
    }

    println!("====================================================================================================");
    Ok(())
}

pub async fn run_fleet_preload(
    controller_url: &str,
    auth_token: Option<String>,
    uri: &str,
    nodes: Option<String>,
    selector: Option<String>,
) -> Result<()> {
    let client = FleetClient::new(controller_url.to_string(), auth_token);
    println!("====================================================================================================");
    println!("                           HYPERCACHE FLEET-WIDE PROACTIVE PRE-LOAD                                 ");
    println!("====================================================================================================");
    println!("Controller:  {}", controller_url);
    println!("Target URI:  {}", uri);
    println!("Node Filter: {}", nodes.as_deref().unwrap_or("all"));
    if let Some(ref sel) = selector {
        println!("Selector:    {}", sel);
    }
    println!("----------------------------------------------------------------------------------------------------");
    println!("Dispatching parallel ingestion directives across target nodes...");

    let t_start = Instant::now();
    let results = client.preload(uri, nodes, selector).await?;
    let dur = t_start.elapsed();

    let mut successful = 0;
    let mut failed = 0;
    let mut total_bytes: u64 = 0;

    for res in &results {
        if res.success {
            successful += 1;
            let sz = res.size_bytes.unwrap_or(0);
            total_bytes += sz;
            let bw_str = res
                .bandwidth_mbps
                .map(|b| format!(" ({:.2} MB/s)", b))
                .unwrap_or_default();
            println!("  [OK] {:<20} -> {}{}", res.node_id, res.message, bw_str);
        } else {
            failed += 1;
            println!("  [FAIL] {:<18} -> {}", res.node_id, res.message);
        }
    }

    println!("----------------------------------------------------------------------------------------------------");
    println!(
        "Fleet Dispatch Completed in {:.2} s: {} succeeded, {} failed across {} nodes ({} aggregate NVMe data written).",
        dur.as_secs_f64(),
        successful,
        failed,
        results.len(),
        format_bytes(total_bytes)
    );
    println!("====================================================================================================");
    ensure_all_succeeded(&results, "preload")
}

pub async fn run_fleet_preload_batch(
    controller_url: &str,
    auth_token: Option<String>,
    prefix: Option<String>,
    uris: Vec<String>,
    nodes: Option<String>,
    selector: Option<String>,
    sharded: bool,
) -> Result<()> {
    let client = FleetClient::new(controller_url.to_string(), auth_token);
    println!("====================================================================================================");
    println!("                          HYPERCACHE FLEET BATCH PRELOAD & DATASET SHARDING                          ");
    println!("====================================================================================================");
    println!("Controller:  {}", controller_url);
    println!(
        "Sharded:     {}",
        if sharded {
            "ENABLED (Targeted modulo worker partition)"
        } else {
            "DISABLED (Full replication)"
        }
    );
    if let Some(ref p) = prefix {
        println!("Prefix:      {}", p);
    }
    if !uris.is_empty() {
        println!("Explicit:    {} URIs specified", uris.len());
    }
    println!("Target Nodes: {}", nodes.as_deref().unwrap_or("all"));
    println!("====================================================================================================");

    let t_start = Instant::now();
    let results = client.preload_batch(prefix, uris, nodes, selector, sharded).await?;

    let mut successful = 0;
    let mut failed = 0;
    let mut total_bytes: u64 = 0;

    for res in &results {
        if res.success {
            successful += 1;
            if let Some(bytes) = res.size_bytes {
                total_bytes += bytes;
            }
            println!("  [OK]   {:<20} -> {}", res.node_id, res.message);
        } else {
            failed += 1;
            println!("  [FAIL] {:<20} -> {}", res.node_id, res.message);
        }
    }

    let elapsed = t_start.elapsed().as_secs_f64();
    println!("====================================================================================================");
    println!(
        "Batch Preload Complete in {:.2} s ({} succeeded, {} failed out of {} total targets, aggregate {})",
        elapsed,
        successful,
        failed,
        results.len(),
        format_bytes(total_bytes)
    );
    println!("====================================================================================================");
    ensure_all_succeeded(&results, "preload_batch")
}

pub async fn run_fleet_evict(
    controller_url: &str,
    auth_token: Option<String>,
    uri: &str,
    nodes: Option<String>,
    selector: Option<String>,
) -> Result<()> {
    let client = FleetClient::new(controller_url.to_string(), auth_token);
    println!("Dispatching fleet eviction for '{}'...", uri);
    let results = client.evict(uri, nodes, selector).await?;

    for res in &results {
        if res.success {
            println!("  [OK]   {:<20} -> {}", res.node_id, res.message);
        } else {
            println!("  [FAIL] {:<20} -> {}", res.node_id, res.message);
        }
    }
    ensure_all_succeeded(&results, "evict")
}

pub async fn run_fleet_purge(
    controller_url: &str,
    auth_token: Option<String>,
    nodes: Option<String>,
    selector: Option<String>,
) -> Result<()> {
    let client = FleetClient::new(controller_url.to_string(), auth_token);
    println!("Dispatching fleet cache purge across target nodes...");
    let results = client.purge(nodes, selector).await?;

    for res in &results {
        if res.success {
            println!("  [OK]   {:<20} -> {}", res.node_id, res.message);
        } else {
            println!("  [FAIL] {:<20} -> {}", res.node_id, res.message);
        }
    }
    ensure_all_succeeded(&results, "purge")
}

/// Fleet commands exit non-zero when any targeted node failed (or none matched), so they can be
/// used in scripts and CI.
fn ensure_all_succeeded(results: &[FleetOperationResult], what: &str) -> Result<()> {
    let failed = results.iter().filter(|r| !r.success).count();
    if results.is_empty() {
        anyhow::bail!("Fleet {} matched no online nodes", what);
    }
    if failed > 0 {
        anyhow::bail!("Fleet {} failed on {} of {} node(s)", what, failed, results.len());
    }
    Ok(())
}

fn truncate_str(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len.saturating_sub(3)])
    }
}

fn format_bytes(bytes: u64) -> String {
    const KIB: u64 = 1024;
    const MIB: u64 = 1024 * KIB;
    const GIB: u64 = 1024 * MIB;
    const TIB: u64 = 1024 * GIB;

    if bytes >= TIB {
        format!("{:.2} TiB", bytes as f64 / TIB as f64)
    } else if bytes >= GIB {
        format!("{:.2} GiB", bytes as f64 / GIB as f64)
    } else if bytes >= MIB {
        format!("{:.2} MiB", bytes as f64 / MIB as f64)
    } else if bytes >= KIB {
        format!("{:.2} KiB", bytes as f64 / KIB as f64)
    } else {
        format!("{} B", bytes)
    }
}
