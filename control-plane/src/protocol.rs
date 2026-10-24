use serde::{Deserialize, Serialize};

use crate::discovery::DiscoveredDevice;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum StorageTier {
    Ram,
    Nvme,
}

impl Default for StorageTier {
    fn default() -> Self {
        StorageTier::Nvme
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum Request {
    Locate {
        uri: String,
        #[serde(default)]
        auto_ingest: bool,
        #[serde(default)]
        tier: Option<StorageTier>,
        #[serde(default)]
        pinned: bool,
    },
    Ingest {
        uri: String,
        #[serde(default)]
        tier: Option<StorageTier>,
        #[serde(default)]
        pinned: bool,
        #[serde(default)]
        allow_synthetic: bool,
    },
    RecordTelemetry {
        event_type: String,
        uri: String,
        object_size_bytes: u64,
        duration_us: u64,
        bandwidth_mbps: f64,
        #[serde(default)]
        client_id: Option<String>,
        #[serde(default)]
        chunk_count: Option<u32>,
        #[serde(default)]
        device_targets: Option<Vec<String>>,
        #[serde(default)]
        local_meta_us: Option<u64>,
        #[serde(default)]
        remote_meta_us: Option<u64>,
        #[serde(default)]
        local_ttfb_us: Option<u64>,
        #[serde(default)]
        remote_ttfb_us: Option<u64>,
        #[serde(default)]
        time_saved_ms: Option<u64>,
        #[serde(default)]
        cpu_pct: Option<f64>,
        #[serde(default)]
        status: Option<String>,
    },
    GetDevices,
    List,
    Evict {
        uri: String,
    },
    Purge,
    Ping,
    StageWrite {
        uri: String,
        size_bytes: u64,
        #[serde(default)]
        tier: Option<StorageTier>,
        #[serde(default)]
        is_leader: bool,
        #[serde(default = "default_true")]
        auto_flush_s3: bool,
        #[serde(default)]
        write_through: bool,
        #[serde(default)]
        pinned: bool,
    },
    AppendWrite {
        uri: String,
        size_bytes: u64,
        #[serde(default)]
        tier: Option<StorageTier>,
    },
    CommitWrite {
        uri: String,
        #[serde(default)]
        sha256: Option<String>,
        #[serde(default)]
        write_through: bool,
    },
    Pin {
        uri: String,
        #[serde(default = "default_true")]
        pinned: bool,
    },
    GetWriteStatus {
        uri: String,
    },
    PreloadBatch {
        uris: Vec<String>,
        #[serde(default)]
        tier: Option<StorageTier>,
    },
}

fn default_true() -> bool {
    true
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ChunkDescriptor {
    #[serde(default)]
    pub tier: StorageTier,
    pub device: String,
    pub offset: u64,
    pub length: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CachedObjectSummary {
    pub uri: String,
    pub object_size: u64,
    pub chunk_count: usize,
    pub created_at: i64,
    pub devices: Vec<String>,
    #[serde(default)]
    pub tier: StorageTier,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_status: Option<String>,
    #[serde(default)]
    pub pinned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "UPPERCASE")]
pub enum ResponseStatus {
    Hit,
    Miss,
    Error,
    Pong,
    Devices,
    List,
    Evicted,
    Purged,
    WriteStaged,
    WriteCommitted,
    WriteStatus,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    pub status: ResponseStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_layout: Option<Vec<ChunkDescriptor>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub devices: Option<Vec<DiscoveredDevice>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub objects: Option<Vec<CachedObjectSummary>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub total_bytes_used: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_capacity_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub evicted_uri: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub freed_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub purged_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ram_cache_enabled: Option<bool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ram_capacity_bytes: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_status: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_progress_percent: Option<f64>,
}

impl Response {
    pub fn hit(object_size: u64, chunk_size: usize, chunk_layout: Vec<ChunkDescriptor>) -> Self {
        Self {
            status: ResponseStatus::Hit,
            object_size: Some(object_size),
            chunk_size: Some(chunk_size),
            chunk_layout: Some(chunk_layout),
            devices: None,
            objects: None,
            total_bytes_used: None,
            max_capacity_bytes: None,
            evicted_uri: None,
            freed_bytes: None,
            purged_count: None,
            error_message: None,
            ram_cache_enabled: None,
            ram_capacity_bytes: None,
            sync_status: None,
            sync_progress_percent: None,
        }
    }

    pub fn devices(devices: Vec<DiscoveredDevice>, ram_cache_enabled: bool, ram_capacity_bytes: Option<u64>) -> Self {
        Self {
            status: ResponseStatus::Devices,
            object_size: None,
            chunk_size: None,
            chunk_layout: None,
            devices: Some(devices),
            objects: None,
            total_bytes_used: None,
            max_capacity_bytes: None,
            evicted_uri: None,
            freed_bytes: None,
            purged_count: None,
            error_message: None,
            ram_cache_enabled: Some(ram_cache_enabled),
            ram_capacity_bytes,
            sync_status: None,
            sync_progress_percent: None,
        }
    }

    pub fn cached_list(
        objects: Vec<CachedObjectSummary>,
        total_bytes_used: u64,
        max_capacity_bytes: u64,
    ) -> Self {
        Self {
            status: ResponseStatus::List,
            object_size: None,
            chunk_size: None,
            chunk_layout: None,
            devices: None,
            objects: Some(objects),
            total_bytes_used: Some(total_bytes_used),
            max_capacity_bytes: Some(max_capacity_bytes),
            evicted_uri: None,
            freed_bytes: None,
            purged_count: None,
            error_message: None,
            ram_cache_enabled: None,
            ram_capacity_bytes: None,
            sync_status: None,
            sync_progress_percent: None,
        }
    }

    pub fn evicted(uri: String, freed_bytes: u64) -> Self {
        Self {
            status: ResponseStatus::Evicted,
            object_size: None,
            chunk_size: None,
            chunk_layout: None,
            devices: None,
            objects: None,
            total_bytes_used: None,
            max_capacity_bytes: None,
            evicted_uri: Some(uri),
            freed_bytes: Some(freed_bytes),
            purged_count: None,
            error_message: None,
            ram_cache_enabled: None,
            ram_capacity_bytes: None,
            sync_status: None,
            sync_progress_percent: None,
        }
    }

    pub fn purged(purged_count: usize, freed_bytes: u64) -> Self {
        Self {
            status: ResponseStatus::Purged,
            object_size: None,
            chunk_size: None,
            chunk_layout: None,
            devices: None,
            objects: None,
            total_bytes_used: None,
            max_capacity_bytes: None,
            evicted_uri: None,
            freed_bytes: Some(freed_bytes),
            purged_count: Some(purged_count),
            error_message: None,
            ram_cache_enabled: None,
            ram_capacity_bytes: None,
            sync_status: None,
            sync_progress_percent: None,
        }
    }

    pub fn miss() -> Self {
        Self {
            status: ResponseStatus::Miss,
            object_size: None,
            chunk_size: None,
            chunk_layout: None,
            devices: None,
            objects: None,
            total_bytes_used: None,
            max_capacity_bytes: None,
            evicted_uri: None,
            freed_bytes: None,
            purged_count: None,
            error_message: None,
            ram_cache_enabled: None,
            ram_capacity_bytes: None,
            sync_status: None,
            sync_progress_percent: None,
        }
    }

    pub fn pong() -> Self {
        Self {
            status: ResponseStatus::Pong,
            object_size: None,
            chunk_size: None,
            chunk_layout: None,
            devices: None,
            objects: None,
            total_bytes_used: None,
            max_capacity_bytes: None,
            evicted_uri: None,
            freed_bytes: None,
            purged_count: None,
            error_message: None,
            ram_cache_enabled: None,
            ram_capacity_bytes: None,
            sync_status: None,
            sync_progress_percent: None,
        }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            status: ResponseStatus::Error,
            object_size: None,
            chunk_size: None,
            chunk_layout: None,
            devices: None,
            objects: None,
            total_bytes_used: None,
            max_capacity_bytes: None,
            evicted_uri: None,
            freed_bytes: None,
            purged_count: None,
            error_message: Some(msg.into()),
            ram_cache_enabled: None,
            ram_capacity_bytes: None,
            sync_status: None,
            sync_progress_percent: None,
        }
    }

    pub fn write_staged(object_size: u64, chunk_size: usize, chunk_layout: Vec<ChunkDescriptor>) -> Self {
        Self {
            status: ResponseStatus::WriteStaged,
            object_size: Some(object_size),
            chunk_size: Some(chunk_size),
            chunk_layout: Some(chunk_layout),
            devices: None,
            objects: None,
            total_bytes_used: None,
            max_capacity_bytes: None,
            evicted_uri: None,
            freed_bytes: None,
            purged_count: None,
            error_message: None,
            ram_cache_enabled: None,
            ram_capacity_bytes: None,
            sync_status: Some("STAGED".to_string()),
            sync_progress_percent: Some(0.0),
        }
    }

    pub fn write_committed(uri: String, object_size: u64) -> Self {
        Self {
            status: ResponseStatus::WriteCommitted,
            object_size: Some(object_size),
            chunk_size: None,
            chunk_layout: None,
            devices: None,
            objects: None,
            total_bytes_used: None,
            max_capacity_bytes: None,
            evicted_uri: Some(uri),
            freed_bytes: None,
            purged_count: None,
            error_message: None,
            ram_cache_enabled: None,
            ram_capacity_bytes: None,
            sync_status: Some("COMMITTED".to_string()),
            sync_progress_percent: Some(100.0),
        }
    }

    pub fn write_status(sync_status: String, progress: f64) -> Self {
        Self {
            status: ResponseStatus::WriteStatus,
            object_size: None,
            chunk_size: None,
            chunk_layout: None,
            devices: None,
            objects: None,
            total_bytes_used: None,
            max_capacity_bytes: None,
            evicted_uri: None,
            freed_bytes: None,
            purged_count: None,
            error_message: None,
            ram_cache_enabled: None,
            ram_capacity_bytes: None,
            sync_status: Some(sync_status),
            sync_progress_percent: Some(progress),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_protocol_serialization() {
        let req = Request::Locate {
            uri: "hypercache://models/llama-3/weights.safetensors".to_string(),
            auto_ingest: true,
            tier: Some(StorageTier::Ram),
            pinned: false,
        };
        let json_str = serde_json::to_string(&req).unwrap();
        assert!(json_str.contains("locate"));
        assert!(json_str.contains("hypercache://models/llama-3/weights.safetensors"));
        assert!(json_str.contains("ram"));

        let resp = Response::hit(
            20971520,
            4194304,
            vec![
                ChunkDescriptor {
                    tier: StorageTier::Nvme,
                    device: "/tmp/nvme0.raw".to_string(),
                    offset: 4096,
                    length: 4194304,
                },
            ],
        );
        let resp_json = serde_json::to_string(&resp).unwrap();
        assert!(resp_json.contains("HIT"));
        assert!(resp_json.contains("20971520"));
        assert!(resp_json.contains("nvme"));
    }
}

