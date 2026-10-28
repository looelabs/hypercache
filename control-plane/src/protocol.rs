use serde::{Deserialize, Serialize};

use crate::discovery::DiscoveredDevice;

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "snake_case")]
pub enum StorageTier {
    Ram,
    #[default]
    Nvme,
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
pub struct ChunkDescriptor {
    #[serde(default)]
    pub tier: StorageTier,
    pub device: String,
    pub offset: u64,
    pub length: u64,
    /// CRC32 (IEEE, zlib-compatible) of the chunk payload, when known
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checksum: Option<u32>,
}

impl ChunkDescriptor {
    pub fn new(tier: StorageTier, device: impl Into<String>, offset: u64, length: u64) -> Self {
        Self {
            tier,
            device: device.into(),
            offset,
            length,
            checksum: None,
        }
    }

    pub fn with_checksum(mut self, data: &[u8]) -> Self {
        self.checksum = Some(crc32fast::hash(data));
        self
    }
}

/// Write-back / replication lifecycle of a cached object. Serialized as the legacy
/// upper-case strings so manifests and IPC clients from earlier releases stay compatible.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum SyncStatus {
    #[serde(rename = "STAGED")]
    Staged,
    #[serde(rename = "COMMITTED")]
    Committed,
    #[serde(rename = "SYNCING_BACKEND")]
    SyncingBackend,
    #[serde(rename = "SYNCED_BACKEND")]
    SyncedBackend,
    #[serde(rename = "SYNC_FAILED")]
    SyncFailed,
    #[serde(rename = "REPLICA_COMMITTED")]
    ReplicaCommitted,
}

impl SyncStatus {
    /// Dirty objects hold data that has not reached the authoritative backend yet and
    /// must never be evicted under capacity pressure.
    pub fn is_dirty(self) -> bool {
        matches!(
            self,
            SyncStatus::Staged | SyncStatus::Committed | SyncStatus::SyncingBackend | SyncStatus::SyncFailed
        )
    }

    pub fn as_str(self) -> &'static str {
        match self {
            SyncStatus::Staged => "STAGED",
            SyncStatus::Committed => "COMMITTED",
            SyncStatus::SyncingBackend => "SYNCING_BACKEND",
            SyncStatus::SyncedBackend => "SYNCED_BACKEND",
            SyncStatus::SyncFailed => "SYNC_FAILED",
            SyncStatus::ReplicaCommitted => "REPLICA_COMMITTED",
        }
    }

    pub fn progress_percent(self) -> f64 {
        match self {
            SyncStatus::Staged => 25.0,
            SyncStatus::Committed => 50.0,
            SyncStatus::SyncingBackend => 75.0,
            SyncStatus::SyncFailed => 0.0,
            SyncStatus::SyncedBackend | SyncStatus::ReplicaCommitted => 100.0,
        }
    }
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
    pub sync_status: Option<SyncStatus>,
    #[serde(default)]
    pub pinned: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Default)]
#[serde(rename_all = "UPPERCASE")]
pub enum ResponseStatus {
    Hit,
    Miss,
    #[default]
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

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Response {
    pub status: ResponseStatus,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub object_size: Option<u64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_size: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub chunk_layout: Option<Vec<ChunkDescriptor>>,
    /// Monotonic layout generation; changes whenever the object is re-ingested or re-staged
    #[serde(skip_serializing_if = "Option::is_none")]
    pub generation: Option<u64>,
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
    pub sync_status: Option<SyncStatus>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sync_progress_percent: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub ingested_count: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub failed_uris: Option<Vec<String>>,
}

impl Response {
    pub fn hit(object_size: u64, chunk_size: usize, chunk_layout: Vec<ChunkDescriptor>) -> Self {
        Self {
            status: ResponseStatus::Hit,
            object_size: Some(object_size),
            chunk_size: Some(chunk_size),
            chunk_layout: Some(chunk_layout),
            ..Default::default()
        }
    }

    pub fn with_generation(mut self, generation: u64) -> Self {
        self.generation = Some(generation);
        self
    }

    pub fn devices(devices: Vec<DiscoveredDevice>, ram_cache_enabled: bool, ram_capacity_bytes: Option<u64>) -> Self {
        Self {
            status: ResponseStatus::Devices,
            devices: Some(devices),
            ram_cache_enabled: Some(ram_cache_enabled),
            ram_capacity_bytes,
            ..Default::default()
        }
    }

    pub fn cached_list(objects: Vec<CachedObjectSummary>, total_bytes_used: u64, max_capacity_bytes: u64) -> Self {
        Self {
            status: ResponseStatus::List,
            objects: Some(objects),
            total_bytes_used: Some(total_bytes_used),
            max_capacity_bytes: Some(max_capacity_bytes),
            ..Default::default()
        }
    }

    pub fn evicted(uri: String, freed_bytes: u64) -> Self {
        Self {
            status: ResponseStatus::Evicted,
            evicted_uri: Some(uri),
            freed_bytes: Some(freed_bytes),
            ..Default::default()
        }
    }

    pub fn purged(purged_count: usize, freed_bytes: u64) -> Self {
        Self {
            status: ResponseStatus::Purged,
            freed_bytes: Some(freed_bytes),
            purged_count: Some(purged_count),
            ..Default::default()
        }
    }

    pub fn miss() -> Self {
        Self {
            status: ResponseStatus::Miss,
            ..Default::default()
        }
    }

    pub fn pong() -> Self {
        Self {
            status: ResponseStatus::Pong,
            ..Default::default()
        }
    }

    /// Batch preload summary. Status stays PONG for backward compatibility with existing clients.
    pub fn batch_result(ingested_count: usize, failed_uris: Vec<String>, total_bytes: u64) -> Self {
        Self {
            status: ResponseStatus::Pong,
            ingested_count: Some(ingested_count),
            failed_uris: Some(failed_uris),
            total_bytes_used: Some(total_bytes),
            ..Default::default()
        }
    }

    pub fn error(msg: impl Into<String>) -> Self {
        Self {
            status: ResponseStatus::Error,
            error_message: Some(msg.into()),
            ..Default::default()
        }
    }

    pub fn write_staged(object_size: u64, chunk_size: usize, chunk_layout: Vec<ChunkDescriptor>) -> Self {
        Self {
            status: ResponseStatus::WriteStaged,
            object_size: Some(object_size),
            chunk_size: Some(chunk_size),
            chunk_layout: Some(chunk_layout),
            sync_status: Some(SyncStatus::Staged),
            sync_progress_percent: Some(0.0),
            ..Default::default()
        }
    }

    pub fn write_committed(uri: String, object_size: u64) -> Self {
        Self {
            status: ResponseStatus::WriteCommitted,
            object_size: Some(object_size),
            evicted_uri: Some(uri),
            sync_status: Some(SyncStatus::Committed),
            sync_progress_percent: Some(100.0),
            ..Default::default()
        }
    }

    pub fn write_status(sync_status: SyncStatus, progress: f64) -> Self {
        Self {
            status: ResponseStatus::WriteStatus,
            sync_status: Some(sync_status),
            sync_progress_percent: Some(progress),
            ..Default::default()
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
            vec![ChunkDescriptor::new(StorageTier::Nvme, "/tmp/nvme0.raw", 4096, 4194304)],
        );
        let resp_json = serde_json::to_string(&resp).unwrap();
        assert!(resp_json.contains("HIT"));
        assert!(resp_json.contains("20971520"));
        assert!(resp_json.contains("nvme"));
    }
}
