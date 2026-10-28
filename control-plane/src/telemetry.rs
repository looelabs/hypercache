use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::TcpListener;
use tracing::info;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryEvent {
    pub timestamp: String,
    pub event_type: String,
    pub uri: String,
    pub object_size_bytes: u64,
    pub duration_us: u64,
    pub bandwidth_mbps: f64,
    pub client_id: String,
    pub chunk_count: u32,
    #[serde(default)]
    pub device_targets: Vec<String>,
    #[serde(default)]
    pub local_meta_us: u64,
    #[serde(default)]
    pub remote_meta_us: u64,
    #[serde(default)]
    pub local_ttfb_us: u64,
    #[serde(default)]
    pub remote_ttfb_us: u64,
    #[serde(default)]
    pub time_saved_ms: u64,
    #[serde(default)]
    pub cpu_pct: f64,
    #[serde(default)]
    pub status: String,
}

pub struct MetricsState {
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub ingestions: AtomicU64,
    pub bytes_served: AtomicU64,
    pub last_bandwidth_x100: AtomicU64, // Stored as scaled integer (MB/s * 100)
    pub last_local_meta_us: AtomicU64,
    pub last_remote_meta_us: AtomicU64,
    pub last_local_ttfb_us: AtomicU64,
    pub last_remote_ttfb_us: AtomicU64,
    pub storage_capacity_bytes: AtomicU64,
    pub storage_used_bytes: AtomicU64,
    pub active_devices_count: AtomicU64,
    pub cached_objects_count: AtomicU64,
    pub evictions_total: AtomicU64,
    pub evicted_bytes_total: AtomicU64,
    pub s3_egress_bytes_saved: AtomicU64,
    pub gpu_idle_time_saved_ms: AtomicU64,
}

impl Default for MetricsState {
    fn default() -> Self {
        Self {
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
            ingestions: AtomicU64::new(0),
            bytes_served: AtomicU64::new(0),
            last_bandwidth_x100: AtomicU64::new(0),
            last_local_meta_us: AtomicU64::new(0),
            last_remote_meta_us: AtomicU64::new(0),
            last_local_ttfb_us: AtomicU64::new(0),
            last_remote_ttfb_us: AtomicU64::new(0),
            storage_capacity_bytes: AtomicU64::new(0),
            storage_used_bytes: AtomicU64::new(0),
            active_devices_count: AtomicU64::new(0),
            cached_objects_count: AtomicU64::new(0),
            evictions_total: AtomicU64::new(0),
            evicted_bytes_total: AtomicU64::new(0),
            s3_egress_bytes_saved: AtomicU64::new(0),
            gpu_idle_time_saved_ms: AtomicU64::new(0),
        }
    }
}

impl MetricsState {
    pub fn render_prometheus(&self) -> String {
        let hits = self.cache_hits.load(Ordering::Relaxed);
        let misses = self.cache_misses.load(Ordering::Relaxed);
        let ingestions = self.ingestions.load(Ordering::Relaxed);
        let bytes = self.bytes_served.load(Ordering::Relaxed);
        let bw = self.last_bandwidth_x100.load(Ordering::Relaxed) as f64 / 100.0;
        let local_meta = self.last_local_meta_us.load(Ordering::Relaxed);
        let remote_meta = self.last_remote_meta_us.load(Ordering::Relaxed);
        let local_ttfb = self.last_local_ttfb_us.load(Ordering::Relaxed);
        let remote_ttfb = self.last_remote_ttfb_us.load(Ordering::Relaxed);

        let capacity = self.storage_capacity_bytes.load(Ordering::Relaxed);
        let used = self.storage_used_bytes.load(Ordering::Relaxed);
        let active_devices = self.active_devices_count.load(Ordering::Relaxed);
        let cached_objects = self.cached_objects_count.load(Ordering::Relaxed);
        let evictions = self.evictions_total.load(Ordering::Relaxed);
        let evicted_bytes = self.evicted_bytes_total.load(Ordering::Relaxed);
        let egress_saved = self.s3_egress_bytes_saved.load(Ordering::Relaxed);
        let time_saved_s = self.gpu_idle_time_saved_ms.load(Ordering::Relaxed) as f64 / 1000.0;

        let util_pct = if capacity > 0 {
            (used as f64 / capacity as f64) * 100.0
        } else {
            0.0
        };

        let cost_saved_dollars = (egress_saved as f64 / (1024.0 * 1024.0 * 1024.0)) * 0.09;
        let total_requests = hits + misses;
        let hit_ratio = if total_requests > 0 {
            (hits as f64 / total_requests as f64) * 100.0
        } else {
            100.0
        };

        format!(
            "# HELP hypercache_cache_hits_total Total count of cache hits\n\
             # TYPE hypercache_cache_hits_total counter\n\
             hypercache_cache_hits_total {}\n\n\
             # HELP hypercache_cache_misses_total Total count of cache misses\n\
             # TYPE hypercache_cache_misses_total counter\n\
             hypercache_cache_misses_total {}\n\n\
             # HELP hypercache_ingestions_total Total count of S3 ingestions\n\
             # TYPE hypercache_ingestions_total counter\n\
             hypercache_ingestions_total {}\n\n\
             # HELP hypercache_bytes_served_total Total bytes served from local SSD/NVMe\n\
             # TYPE hypercache_bytes_served_total counter\n\
             hypercache_bytes_served_total {}\n\n\
             # HELP hypercache_hit_ratio_percent Live cache hit ratio percentage\n\
             # TYPE hypercache_hit_ratio_percent gauge\n\
             hypercache_hit_ratio_percent {:.2}\n\n\
             # HELP hypercache_bandwidth_mbps Most recent read throughput in MB/s\n\
             # TYPE hypercache_bandwidth_mbps gauge\n\
             hypercache_bandwidth_mbps {:.2}\n\n\
             # HELP hypercache_local_metadata_latency_us Local UDS metadata handshake latency in microseconds\n\
             # TYPE hypercache_local_metadata_latency_us gauge\n\
             hypercache_local_metadata_latency_us {}\n\n\
             # HELP hypercache_remote_metadata_latency_us Remote S3 metadata latency in microseconds\n\
             # TYPE hypercache_remote_metadata_latency_us gauge\n\
             hypercache_remote_metadata_latency_us {}\n\n\
             # HELP hypercache_local_ttfb_us Local SSD Time-To-First-Byte in microseconds\n\
             # TYPE hypercache_local_ttfb_us gauge\n\
             hypercache_local_ttfb_us {}\n\n\
             # HELP hypercache_remote_ttfb_us Remote S3 Time-To-First-Byte in microseconds\n\
             # TYPE hypercache_remote_ttfb_us gauge\n\
             hypercache_remote_ttfb_us {}\n\n\
             # HELP hypercache_storage_capacity_bytes Total capacity of local striped SSD pool in bytes\n\
             # TYPE hypercache_storage_capacity_bytes gauge\n\
             hypercache_storage_capacity_bytes {}\n\n\
             # HELP hypercache_storage_used_bytes Total cached storage currently used on SSD in bytes\n\
             # TYPE hypercache_storage_used_bytes gauge\n\
             hypercache_storage_used_bytes {}\n\n\
             # HELP hypercache_storage_utilization_percent Storage pool capacity utilization percentage\n\
             # TYPE hypercache_storage_utilization_percent gauge\n\
             hypercache_storage_utilization_percent {:.2}\n\n\
             # HELP hypercache_active_devices_count Number of active striped block devices in the pool\n\
             # TYPE hypercache_active_devices_count gauge\n\
             hypercache_active_devices_count {}\n\n\
             # HELP hypercache_cached_objects_count Number of active model objects cached in Moka\n\
             # TYPE hypercache_cached_objects_count gauge\n\
             hypercache_cached_objects_count {}\n\n\
             # HELP hypercache_evictions_total Total count of evicted models by W-TinyLFU\n\
             # TYPE hypercache_evictions_total counter\n\
             hypercache_evictions_total {}\n\n\
             # HELP hypercache_evicted_bytes_total Total bytes reclaimed by W-TinyLFU eviction\n\
             # TYPE hypercache_evicted_bytes_total counter\n\
             hypercache_evicted_bytes_total {}\n\n\
             # HELP hypercache_s3_egress_bytes_saved_total Total bytes saved from S3 egress by local cache hits\n\
             # TYPE hypercache_s3_egress_bytes_saved_total counter\n\
             hypercache_s3_egress_bytes_saved_total {}\n\n\
             # HELP hypercache_estimated_cost_savings_dollars Estimated dollar savings from S3 egress avoidance\n\
             # TYPE hypercache_estimated_cost_savings_dollars gauge\n\
             hypercache_estimated_cost_savings_dollars {:.4}\n\n\
             # HELP hypercache_gpu_idle_time_saved_seconds Cumulative GPU idle time saved waiting for network downloads\n\
             # TYPE hypercache_gpu_idle_time_saved_seconds counter\n\
             hypercache_gpu_idle_time_saved_seconds {:.3}\n",
            hits, misses, ingestions, bytes, hit_ratio, bw, local_meta, remote_meta,
            local_ttfb, remote_ttfb, capacity, used, util_pct, active_devices,
            cached_objects, evictions, evicted_bytes, egress_saved, cost_saved_dollars, time_saved_s
        )
    }
}

#[derive(Clone)]
pub struct TelemetryCollector {
    pub state: Arc<MetricsState>,
}

impl TelemetryCollector {
    pub fn new(bind_addr: &str, metrics_port: u16, enabled: bool) -> Self {
        let state = Arc::new(MetricsState::default());
        let state_for_metrics = Arc::clone(&state);
        let bind_str = format!("{}:{}", bind_addr, metrics_port);

        if enabled {
            // Spawn high-performance Prometheus HTTP metrics exporter
            tokio::spawn(async move {
                match TcpListener::bind(&bind_str).await {
                    Ok(listener) => {
                        info!(
                            "HyperCache Prometheus metrics exporter listening on http://{}/metrics",
                            bind_str
                        );
                        loop {
                            if let Ok((mut stream, _)) = listener.accept().await {
                                let state = Arc::clone(&state_for_metrics);
                                tokio::spawn(async move {
                                    let mut buf = [0u8; 1024];
                                    let _ = stream.read(&mut buf).await;
                                    let body = state.render_prometheus();
                                    let response = format!(
                                        "HTTP/1.1 200 OK\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: {}\r\nAccess-Control-Allow-Origin: *\r\nConnection: close\r\n\r\n{}",
                                        body.len(),
                                        body
                                    );
                                    let _ = stream.write_all(response.as_bytes()).await;
                                    let _ = stream.flush().await;
                                });
                            }
                        }
                    }
                    Err(e) => {
                        tracing::warn!("Could not bind Prometheus metrics on {}: {}", bind_str, e);
                    }
                }
            });
        }

        Self { state }
    }

    pub fn record(&self, event: TelemetryEvent) {
        match event.event_type.as_str() {
            "CACHE_HIT" => {
                self.state.cache_hits.fetch_add(1, Ordering::Relaxed);
                self.state
                    .bytes_served
                    .fetch_add(event.object_size_bytes, Ordering::Relaxed);
                self.state
                    .s3_egress_bytes_saved
                    .fetch_add(event.object_size_bytes, Ordering::Relaxed);
                if event.bandwidth_mbps > 0.0 {
                    self.state
                        .last_bandwidth_x100
                        .store((event.bandwidth_mbps * 100.0) as u64, Ordering::Relaxed);
                }
                if event.local_meta_us > 0 {
                    self.state
                        .last_local_meta_us
                        .store(event.local_meta_us, Ordering::Relaxed);
                }
                if event.remote_meta_us > 0 {
                    self.state
                        .last_remote_meta_us
                        .store(event.remote_meta_us, Ordering::Relaxed);
                }
                if event.local_ttfb_us > 0 {
                    self.state
                        .last_local_ttfb_us
                        .store(event.local_ttfb_us, Ordering::Relaxed);
                }
                if event.remote_ttfb_us > 0 {
                    self.state
                        .last_remote_ttfb_us
                        .store(event.remote_ttfb_us, Ordering::Relaxed);
                }
                if event.time_saved_ms > 0 {
                    self.state
                        .gpu_idle_time_saved_ms
                        .fetch_add(event.time_saved_ms, Ordering::Relaxed);
                }
            }
            "CACHE_MISS" => {
                self.state.cache_misses.fetch_add(1, Ordering::Relaxed);
            }
            "INGESTION" => {
                self.state.ingestions.fetch_add(1, Ordering::Relaxed);
                self.state
                    .storage_used_bytes
                    .fetch_add(event.object_size_bytes, Ordering::Relaxed);
                self.state.cached_objects_count.fetch_add(1, Ordering::Relaxed);
            }
            _ => {}
        }
    }
}
