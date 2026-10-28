use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::Path;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    #[serde(default = "default_socket_path")]
    pub socket_path: String,

    #[serde(default = "default_block_devices")]
    pub block_devices: Vec<String>,

    #[serde(default = "default_chunk_size")]
    pub chunk_size_bytes: usize,

    #[serde(default = "default_device_start_offset")]
    pub device_start_offset_bytes: u64,

    #[serde(default)]
    pub max_cache_capacity_bytes: Option<u64>,

    #[serde(default = "default_authoritative_backend")]
    pub authoritative_backend: String,

    #[serde(default = "default_nfs_mount_path")]
    pub nfs_mount_path: String,

    #[serde(default = "default_nfs_buffer_size")]
    pub nfs_read_buffer_bytes: usize,

    #[serde(default = "default_s3_endpoint")]
    pub s3_endpoint: String,

    #[serde(default = "default_s3_region")]
    pub s3_region: String,

    #[serde(default = "default_s3_bucket")]
    pub s3_bucket: String,

    #[serde(default = "default_s3_key")]
    pub s3_access_key: String,

    #[serde(default = "default_s3_key")]
    pub s3_secret_key: String,

    #[serde(default = "default_auto_create")]
    pub auto_create_mock_devices: bool,

    #[serde(default = "default_mock_device_size")]
    pub mock_device_size_bytes: u64,

    #[serde(default = "default_discovery_mode")]
    pub device_discovery_mode: String,

    #[serde(default = "default_mock_pattern")]
    pub dev_mock_device_pattern: String,

    #[serde(default = "default_mock_count")]
    pub dev_mock_auto_count: usize,

    #[serde(default = "default_allow_virtual")]
    pub allow_virtual_drives: bool,

    #[serde(default = "default_include_partitions")]
    pub include_partitions: bool,

    #[serde(default = "default_device_prefixes")]
    pub device_prefixes: Vec<String>,

    #[serde(default)]
    pub device_filter: String,

    #[serde(default = "default_exclude_root")]
    pub exclude_root_device: bool,

    #[serde(default = "default_exclude_mounted")]
    pub exclude_mounted_devices: bool,

    #[serde(default = "default_exclude_fs")]
    pub exclude_devices_with_existing_fs: bool,

    #[serde(default)]
    pub force_use_devices: Vec<String>,

    #[serde(default = "default_ram_cache_enabled")]
    pub ram_cache_enabled: bool,

    #[serde(default = "default_ram_cache_capacity")]
    pub ram_cache_capacity_bytes: Option<u64>,

    #[serde(default = "default_ram_cache_shm_path")]
    pub ram_cache_shm_path: String,

    #[serde(default)]
    pub ram_cache_hugepages: bool,

    #[serde(default = "default_ram_cache_hugepages_path")]
    pub ram_cache_hugepages_path: String,

    #[serde(default = "default_persistence_enabled")]
    pub metadata_persistence_enabled: bool,

    #[serde(default = "default_persistence_path")]
    pub metadata_persistence_path: String,

    #[serde(default)]
    pub controller_url: Option<String>,

    #[serde(default = "default_node_id")]
    pub node_id: String,

    #[serde(default = "default_node_address")]
    pub node_address: Option<String>,

    #[serde(default)]
    pub node_labels: std::collections::HashMap<String, String>,

    #[serde(default)]
    pub metrics_enabled: bool,

    #[serde(default = "default_metrics_bind_address")]
    pub metrics_bind_address: String,

    #[serde(default = "default_metrics_port")]
    pub metrics_port: u16,

    #[serde(default)]
    pub agent_enabled: bool,

    #[serde(default = "default_agent_bind_address")]
    pub agent_bind_address: String,

    #[serde(default = "default_agent_http_port")]
    pub agent_http_port: u16,

    #[serde(default = "default_controller_bind_address")]
    pub controller_bind_address: String,

    #[serde(default = "default_controller_port")]
    pub controller_port: u16,

    #[serde(default = "default_controller_admin_port")]
    pub controller_admin_port: u16,

    #[serde(default = "default_cluster_auth_token")]
    pub cluster_auth_token: Option<String>,

    #[serde(default)]
    pub write_through_enabled: bool,

    #[serde(default)]
    pub allow_synthetic_s3_fallback: bool,

    #[serde(default = "default_storage_backing_mode")]
    pub storage_backing_mode: String,

    #[serde(default)]
    pub filesystem_cache_paths: Vec<String>,

    #[serde(default = "default_filesystem_arena_size_bytes")]
    pub filesystem_arena_size_bytes: u64,

    #[serde(default = "default_peer_replication_factor")]
    pub peer_replication_factor: usize,

    #[serde(default)]
    pub p2p_preload_enabled: bool,

    /// Extra directories (besides `nfs_mount_path`) that file:// / nfs:// URIs may read or write
    #[serde(default)]
    pub nfs_allowed_paths: Vec<String>,

    /// Concurrent ranged GETs / multipart part uploads per S3 object (1 = single stream)
    #[serde(default = "default_s3_parallel_streams")]
    pub s3_parallel_streams: usize,

    /// Idle timeout for a single S3 socket read; there is no whole-request timeout
    #[serde(default = "default_s3_read_timeout_secs")]
    pub s3_read_timeout_secs: u64,

    /// Temporary-credential session token (STS / assumed roles); defaults to $AWS_SESSION_TOKEN
    #[serde(default = "default_s3_session_token")]
    pub s3_session_token: Option<String>,

    /// Objects larger than this are uploaded to S3 with multipart upload
    #[serde(default = "default_s3_multipart_threshold_bytes")]
    pub s3_multipart_threshold_bytes: u64,

    /// Grace period before blocks of evicted / replaced objects are reused, so in-flight
    /// readers of the old layout never observe overwritten data
    #[serde(default = "default_reclaim_grace_secs")]
    pub reclaim_grace_secs: u64,

    /// Controller only: let loopback / controller-host connections skip token authentication.
    /// Disable when the controller sits behind a local proxy or sidecar (all traffic then
    /// appears to come from 127.0.0.1).
    #[serde(default = "default_true")]
    pub allow_local_admin_bypass: bool,

    /// Controller only: seconds without heartbeat after which an OFFLINE node is forgotten
    #[serde(default = "default_node_gc_secs")]
    pub node_gc_secs: u64,
}

pub const PRIMARY_METADATA_PATH: &str = "/var/lib/hypercache/metadata.json";
pub const FALLBACK_METADATA_PATH: &str = "/tmp/hypercache/metadata.json";

pub const PRIMARY_SHM_PATH: &str = "/dev/shm/hypercache";
pub const FALLBACK_SHM_PATH: &str = "/tmp/hypercache/shm";

pub fn get_shm_path() -> std::path::PathBuf {
    if Path::new("/dev/shm").exists() && std::fs::create_dir_all(PRIMARY_SHM_PATH).is_ok() {
        std::path::PathBuf::from(PRIMARY_SHM_PATH)
    } else {
        let _ = std::fs::create_dir_all(FALLBACK_SHM_PATH);
        std::path::PathBuf::from(FALLBACK_SHM_PATH)
    }
}

pub fn get_metadata_persistence_path() -> std::path::PathBuf {
    if std::fs::create_dir_all("/var/lib/hypercache").is_ok() {
        std::path::PathBuf::from(PRIMARY_METADATA_PATH)
    } else {
        let _ = std::fs::create_dir_all("/tmp/hypercache");
        std::path::PathBuf::from(FALLBACK_METADATA_PATH)
    }
}

pub fn resolve_metadata_persistence_path(configured_path: &str) -> std::path::PathBuf {
    let p = Path::new(configured_path);
    if configured_path == PRIMARY_METADATA_PATH || configured_path == "/tmp/hypercache_metadata.json" {
        get_metadata_persistence_path()
    } else {
        if let Some(parent) = p.parent() {
            if std::fs::create_dir_all(parent).is_err() {
                let fallback = Path::new("/tmp/hypercache").join(p.file_name().unwrap_or_default());
                let _ = std::fs::create_dir_all("/tmp/hypercache");
                return fallback;
            }
        }
        p.to_path_buf()
    }
}

fn default_ram_cache_enabled() -> bool {
    false
}

fn default_ram_cache_capacity() -> Option<u64> {
    let shm_dir = get_shm_path();
    if let Ok(space) = fs2::total_space(&shm_dir) {
        if space > 0 {
            // Allocate 75% of /dev/shm capacity by default, with 8 GiB minimum
            let recommended = (space * 3) / 4;
            return Some(recommended.max(8 * 1024 * 1024 * 1024));
        }
    }
    Some(8 * 1024 * 1024 * 1024) // 8 GiB fallback
}

fn default_ram_cache_shm_path() -> String {
    get_shm_path().to_string_lossy().to_string()
}

fn default_persistence_enabled() -> bool {
    true
}

fn default_persistence_path() -> String {
    PRIMARY_METADATA_PATH.to_string()
}

fn default_discovery_mode() -> String {
    "auto".to_string()
}

fn default_mock_pattern() -> String {
    "/tmp/hypercache_nvme*.raw".to_string()
}

fn default_mock_count() -> usize {
    4
}

fn default_allow_virtual() -> bool {
    true
}

fn default_include_partitions() -> bool {
    true
}

fn default_device_prefixes() -> Vec<String> {
    vec![
        "nvme".to_string(),
        "vd".to_string(),
        "xvd".to_string(),
        "sd".to_string(),
    ]
}

fn default_exclude_root() -> bool {
    true
}

fn default_exclude_mounted() -> bool {
    true
}

fn default_exclude_fs() -> bool {
    true
}

pub const PRIMARY_SOCKET_PATH: &str = "/run/hypercache/hypercache.sock";
pub const FALLBACK_SOCKET_PATH: &str = "/tmp/hypercache/hypercache.sock";
pub const PRIMARY_LOCK_PATH: &str = "/run/hypercache/hypercache.lock";
pub const FALLBACK_LOCK_PATH: &str = "/tmp/hypercache/hypercache.lock";

/// Environment override for the daemon socket (isolated test instances, containers).
/// When set, the daemon lock file lives next to the socket.
pub const SOCKET_PATH_ENV: &str = "HYPERCACHE_SOCKET_PATH";

fn socket_path_override() -> Option<std::path::PathBuf> {
    std::env::var(SOCKET_PATH_ENV)
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from)
}

pub fn get_socket_path() -> std::path::PathBuf {
    if let Some(p) = socket_path_override() {
        if let Some(parent) = p.parent() {
            let _ = std::fs::create_dir_all(parent);
        }
        return p;
    }
    if std::fs::create_dir_all("/run/hypercache").is_ok() {
        let test_probe = "/run/hypercache/.probe_sock";
        if std::fs::write(test_probe, b"").is_ok() {
            let _ = std::fs::remove_file(test_probe);
            return std::path::PathBuf::from(PRIMARY_SOCKET_PATH);
        }
    }
    let _ = std::fs::create_dir_all("/tmp/hypercache");
    std::path::PathBuf::from(FALLBACK_SOCKET_PATH)
}

pub fn get_lock_path() -> std::path::PathBuf {
    if let Some(p) = socket_path_override() {
        return p.with_file_name("hypercache.lock");
    }
    if std::fs::create_dir_all("/run/hypercache").is_ok() {
        let test_probe = "/run/hypercache/.probe_lock";
        if std::fs::write(test_probe, b"").is_ok() {
            let _ = std::fs::remove_file(test_probe);
            return std::path::PathBuf::from(PRIMARY_LOCK_PATH);
        }
    }
    let _ = std::fs::create_dir_all("/tmp/hypercache");
    std::path::PathBuf::from(FALLBACK_LOCK_PATH)
}

pub fn probe_active_socket_path() -> std::path::PathBuf {
    if let Some(p) = socket_path_override() {
        return p;
    }
    if Path::new(PRIMARY_SOCKET_PATH).exists() {
        std::path::PathBuf::from(PRIMARY_SOCKET_PATH)
    } else if Path::new(FALLBACK_SOCKET_PATH).exists() {
        std::path::PathBuf::from(FALLBACK_SOCKET_PATH)
    } else if Path::new("/tmp/hypercache.sock").exists() {
        std::path::PathBuf::from("/tmp/hypercache.sock")
    } else {
        get_socket_path()
    }
}

fn default_socket_path() -> String {
    get_socket_path().to_string_lossy().to_string()
}

fn default_block_devices() -> Vec<String> {
    Vec::new()
}

fn default_chunk_size() -> usize {
    4 * 1024 * 1024 // 4 MiB
}

fn default_device_start_offset() -> u64 {
    4096 // 4 KB aligned
}

fn default_authoritative_backend() -> String {
    "s3".to_string()
}

fn default_nfs_mount_path() -> String {
    "/mnt/nfs".to_string()
}

fn default_nfs_buffer_size() -> usize {
    4 * 1024 * 1024 // 4 MiB buffer
}

fn default_s3_endpoint() -> String {
    "http://127.0.0.1:9002".to_string()
}

fn default_s3_region() -> String {
    "us-east-1".to_string()
}

fn default_s3_bucket() -> String {
    "hypercache-authoritative".to_string()
}

fn default_s3_key() -> String {
    String::new()
}

fn default_ram_cache_hugepages_path() -> String {
    "/dev/hugepages/hypercache".to_string()
}

fn default_node_id() -> String {
    std::env::var("HOSTNAME").unwrap_or_else(|_| "node-01".to_string())
}

fn default_node_address() -> Option<String> {
    std::env::var("HYPERCACHE_NODE_IP")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn default_metrics_bind_address() -> String {
    "127.0.0.1".to_string()
}

fn default_metrics_port() -> u16 {
    9110
}

fn default_agent_bind_address() -> String {
    "0.0.0.0".to_string()
}

fn default_agent_http_port() -> u16 {
    9115
}

fn default_controller_bind_address() -> String {
    "0.0.0.0".to_string()
}

fn default_controller_port() -> u16 {
    9120
}

fn default_controller_admin_port() -> u16 {
    9121
}

fn default_cluster_auth_token() -> Option<String> {
    std::env::var("HYPERCACHE_AUTH_TOKEN")
        .ok()
        .filter(|s| !s.trim().is_empty())
}

fn default_auto_create() -> bool {
    false
}

fn default_mock_device_size() -> u64 {
    100 * 1024 * 1024 // 100 MiB
}

fn default_storage_backing_mode() -> String {
    "auto".to_string()
}

fn default_filesystem_arena_size_bytes() -> u64 {
    10 * 1024 * 1024 * 1024 // 10 GiB default per filesystem directory arena
}

fn default_peer_replication_factor() -> usize {
    1
}

fn default_s3_parallel_streams() -> usize {
    8
}

fn default_s3_read_timeout_secs() -> u64 {
    60
}

fn default_s3_session_token() -> Option<String> {
    std::env::var("AWS_SESSION_TOKEN").ok().filter(|s| !s.trim().is_empty())
}

fn default_s3_multipart_threshold_bytes() -> u64 {
    64 * 1024 * 1024
}

fn default_reclaim_grace_secs() -> u64 {
    30
}

fn default_true() -> bool {
    true
}

fn default_node_gc_secs() -> u64 {
    600
}

pub fn get_ram_pool_path(config: &Config) -> std::path::PathBuf {
    if config.ram_cache_hugepages {
        let hp = Path::new(&config.ram_cache_hugepages_path);
        if std::fs::create_dir_all(hp).is_ok() {
            return hp.to_path_buf();
        }
        tracing::warn!(
            "Failed to initialize HugePages directory at {:?}; falling back to standard POSIX shared memory.",
            hp
        );
    }
    if !config.ram_cache_shm_path.is_empty() && config.ram_cache_shm_path != PRIMARY_SHM_PATH {
        let custom = Path::new(&config.ram_cache_shm_path);
        if std::fs::create_dir_all(custom).is_ok() {
            return custom.to_path_buf();
        }
    }
    get_shm_path()
}

impl Default for Config {
    fn default() -> Self {
        Self {
            socket_path: default_socket_path(),
            block_devices: default_block_devices(),
            chunk_size_bytes: default_chunk_size(),
            device_start_offset_bytes: default_device_start_offset(),
            max_cache_capacity_bytes: None,
            authoritative_backend: default_authoritative_backend(),
            nfs_mount_path: default_nfs_mount_path(),
            nfs_read_buffer_bytes: default_nfs_buffer_size(),
            s3_endpoint: default_s3_endpoint(),
            s3_region: default_s3_region(),
            s3_bucket: default_s3_bucket(),
            s3_access_key: default_s3_key(),
            s3_secret_key: default_s3_key(),
            auto_create_mock_devices: default_auto_create(),
            mock_device_size_bytes: default_mock_device_size(),
            device_discovery_mode: default_discovery_mode(),
            dev_mock_device_pattern: default_mock_pattern(),
            dev_mock_auto_count: default_mock_count(),
            allow_virtual_drives: default_allow_virtual(),
            include_partitions: default_include_partitions(),
            device_prefixes: default_device_prefixes(),
            device_filter: String::new(),
            exclude_root_device: default_exclude_root(),
            exclude_mounted_devices: default_exclude_mounted(),
            exclude_devices_with_existing_fs: default_exclude_fs(),
            force_use_devices: Vec::new(),
            ram_cache_enabled: default_ram_cache_enabled(),
            ram_cache_capacity_bytes: default_ram_cache_capacity(),
            ram_cache_shm_path: default_ram_cache_shm_path(),
            ram_cache_hugepages: false,
            ram_cache_hugepages_path: default_ram_cache_hugepages_path(),
            metadata_persistence_enabled: default_persistence_enabled(),
            metadata_persistence_path: default_persistence_path(),
            controller_url: None,
            node_id: default_node_id(),
            node_address: default_node_address(),
            node_labels: std::collections::HashMap::new(),
            metrics_enabled: false,
            metrics_bind_address: default_metrics_bind_address(),
            metrics_port: default_metrics_port(),
            agent_enabled: false,
            agent_bind_address: default_agent_bind_address(),
            agent_http_port: default_agent_http_port(),
            controller_bind_address: default_controller_bind_address(),
            controller_port: default_controller_port(),
            controller_admin_port: default_controller_admin_port(),
            cluster_auth_token: default_cluster_auth_token(),
            write_through_enabled: false,
            allow_synthetic_s3_fallback: false,
            storage_backing_mode: default_storage_backing_mode(),
            filesystem_cache_paths: Vec::new(),
            filesystem_arena_size_bytes: default_filesystem_arena_size_bytes(),
            peer_replication_factor: default_peer_replication_factor(),
            p2p_preload_enabled: false,
            nfs_allowed_paths: Vec::new(),
            s3_parallel_streams: default_s3_parallel_streams(),
            s3_read_timeout_secs: default_s3_read_timeout_secs(),
            s3_session_token: default_s3_session_token(),
            s3_multipart_threshold_bytes: default_s3_multipart_threshold_bytes(),
            reclaim_grace_secs: default_reclaim_grace_secs(),
            allow_local_admin_bypass: true,
            node_gc_secs: default_node_gc_secs(),
        }
    }
}

impl Config {
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let content = std::fs::read_to_string(path.as_ref())
            .with_context(|| format!("Failed to read config file at {:?}", path.as_ref()))?;
        Self::from_toml_str(&content)
    }

    pub fn from_toml_str(content: &str) -> Result<Self> {
        let preprocessed = preprocess_unquoted_env_vars(content);
        let mut config: Config = toml::from_str(&preprocessed).with_context(|| "Failed to parse TOML configuration")?;
        config.expand_and_apply_env_overrides();
        Ok(config)
    }

    pub fn expand_and_apply_env_overrides(&mut self) {
        self.socket_path = expand_env_vars(&self.socket_path);
        self.authoritative_backend = expand_env_vars(&self.authoritative_backend);
        self.nfs_mount_path = expand_env_vars(&self.nfs_mount_path);
        self.s3_endpoint = expand_env_vars(&self.s3_endpoint);
        self.s3_region = expand_env_vars(&self.s3_region);
        self.s3_bucket = expand_env_vars(&self.s3_bucket);
        self.s3_access_key = expand_env_vars(&self.s3_access_key);
        self.s3_secret_key = expand_env_vars(&self.s3_secret_key);
        self.node_id = expand_env_vars(&self.node_id);
        self.ram_cache_shm_path = expand_env_vars(&self.ram_cache_shm_path);
        self.ram_cache_hugepages_path = expand_env_vars(&self.ram_cache_hugepages_path);
        self.metadata_persistence_path = expand_env_vars(&self.metadata_persistence_path);
        self.nfs_allowed_paths = self.nfs_allowed_paths.iter().map(|p| expand_env_vars(p)).collect();
        if let Some(ref token) = self.s3_session_token {
            let expanded = expand_env_vars(token);
            self.s3_session_token = if expanded.trim().is_empty() {
                None
            } else {
                Some(expanded)
            };
        }

        if let Some(ref url) = self.controller_url {
            let expanded = expand_env_vars(url);
            self.controller_url = if expanded.trim().is_empty() {
                None
            } else {
                Some(expanded)
            };
        }
        if let Some(ref addr) = self.node_address {
            let expanded = expand_env_vars(addr);
            self.node_address = if expanded.trim().is_empty() {
                None
            } else {
                Some(expanded)
            };
        }
        if let Some(ref token) = self.cluster_auth_token {
            let expanded = expand_env_vars(token);
            self.cluster_auth_token = if expanded.trim().is_empty() {
                None
            } else {
                Some(expanded)
            };
        }

        self.block_devices = self.block_devices.iter().map(|d| expand_env_vars(d)).collect();
        self.filesystem_cache_paths = self.filesystem_cache_paths.iter().map(|p| expand_env_vars(p)).collect();

        // Standard environment variable fallbacks when fields are empty or unset
        if self.s3_access_key.is_empty() {
            if let Ok(val) = std::env::var("AWS_ACCESS_KEY_ID").or_else(|_| std::env::var("S3_ACCESS_KEY")) {
                if !val.trim().is_empty() {
                    self.s3_access_key = val;
                }
            }
        }
        if self.s3_secret_key.is_empty() {
            if let Ok(val) = std::env::var("AWS_SECRET_ACCESS_KEY").or_else(|_| std::env::var("S3_SECRET_KEY")) {
                if !val.trim().is_empty() {
                    self.s3_secret_key = val;
                }
            }
        }
        if self.s3_endpoint.is_empty() || self.s3_endpoint == default_s3_endpoint() {
            if let Ok(val) = std::env::var("S3_ENDPOINT") {
                if !val.trim().is_empty() {
                    self.s3_endpoint = val;
                }
            }
        }
        if self.s3_region.is_empty() || self.s3_region == default_s3_region() {
            if let Ok(val) = std::env::var("AWS_DEFAULT_REGION").or_else(|_| std::env::var("S3_REGION")) {
                if !val.trim().is_empty() {
                    self.s3_region = val;
                }
            }
        }
        if self.s3_bucket.is_empty() || self.s3_bucket == default_s3_bucket() {
            if let Ok(val) = std::env::var("S3_BUCKET") {
                if !val.trim().is_empty() {
                    self.s3_bucket = val;
                }
            }
        }
        if let Ok(val) = std::env::var("HYPERCACHE_SOCKET") {
            if !val.trim().is_empty() {
                self.socket_path = val;
            }
        }
        if self.controller_url.is_none() {
            if let Ok(val) = std::env::var("HYPERCACHE_CONTROLLER_URL") {
                if !val.trim().is_empty() {
                    self.controller_url = Some(val);
                }
            }
        }
        if self.s3_session_token.is_none() {
            if let Ok(val) = std::env::var("AWS_SESSION_TOKEN") {
                if !val.trim().is_empty() {
                    self.s3_session_token = Some(val);
                }
            }
        }
        if self.cluster_auth_token.is_none() {
            if let Ok(val) = std::env::var("HYPERCACHE_AUTH_TOKEN") {
                if !val.trim().is_empty() {
                    self.cluster_auth_token = Some(val);
                }
            }
        }
        if let Ok(val) = std::env::var("RAM_CACHE_ENABLED") {
            if val.eq_ignore_ascii_case("true") || val == "1" {
                self.ram_cache_enabled = true;
            }
        }
        if let Ok(val) = std::env::var("RAM_CAPACITY_BYTES") {
            if let Ok(parsed) = val.trim().parse::<u64>() {
                self.ram_cache_capacity_bytes = Some(parsed);
            }
        }
    }
}

/// Converts unquoted `$VAR` or `${VAR}` values in TOML assignments (e.g., `s3_access_key = $six_deep_access`)
/// into quoted TOML string literals (`s3_access_key = "$six_deep_access"`) prior to parsing.
fn preprocess_unquoted_env_vars(content: &str) -> String {
    let mut out = String::with_capacity(content.len() + 64);
    for line in content.lines() {
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') || !trimmed.contains('=') {
            out.push_str(line);
            out.push('\n');
            continue;
        }

        if let Some((lhs, rhs)) = line.split_once('=') {
            let rhs_trimmed = rhs.trim_start();
            if rhs_trimmed.starts_with('$') {
                // Separate value token from optional inline comment (# ...)
                let (val_part, comment_part) = match rhs_trimmed.find('#') {
                    Some(idx) => (&rhs_trimmed[..idx], Some(&rhs_trimmed[idx..])),
                    None => (rhs_trimmed, None),
                };
                let token = val_part.trim();
                if let Some(comment) = comment_part {
                    out.push_str(&format!("{}= \"{}\" {}\n", lhs, token, comment));
                } else {
                    out.push_str(&format!("{}= \"{}\"\n", lhs, token));
                }
                continue;
            }
        }

        out.push_str(line);
        out.push('\n');
    }
    out
}

/// Expands `$VAR` and `${VAR}` references within a string using the current process environment.
fn expand_env_vars(input: &str) -> String {
    if !input.contains('$') {
        return input.to_string();
    }

    let mut result = String::with_capacity(input.len());
    let chars: Vec<char> = input.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        if chars[i] == '$' && i + 1 < chars.len() {
            if chars[i + 1] == '{' {
                // ${VAR_NAME} syntax
                if let Some(end_offset) = chars[i + 2..].iter().position(|&c| c == '}') {
                    let var_name: String = chars[i + 2..i + 2 + end_offset].iter().collect();
                    if !var_name.is_empty() && var_name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_') {
                        match std::env::var(&var_name) {
                            Ok(val) => result.push_str(&val),
                            Err(_) => {
                                tracing::warn!(
                                    "Environment variable '{}' referenced in config is not set (if running with sudo, use 'sudo -E' or export the variable)",
                                    var_name
                                );
                            }
                        }
                        i += 2 + end_offset + 1;
                        continue;
                    }
                }
            } else if chars[i + 1].is_ascii_alphabetic() || chars[i + 1] == '_' {
                // $VAR_NAME syntax
                let start = i + 1;
                let mut end = start;
                while end < chars.len() && (chars[end].is_ascii_alphanumeric() || chars[end] == '_') {
                    end += 1;
                }
                let var_name: String = chars[start..end].iter().collect();
                match std::env::var(&var_name) {
                    Ok(val) => result.push_str(&val),
                    Err(_) => {
                        tracing::warn!(
                            "Environment variable '{}' referenced in config is not set (if running with sudo, use 'sudo -E' or export the variable)",
                            var_name
                        );
                    }
                }
                i = end;
                continue;
            }
        }

        result.push(chars[i]);
        i += 1;
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toml_unquoted_and_quoted_env_var_expansion() {
        std::env::set_var("hc_test_access_key", "AKIA_TEST_KEY_123");
        std::env::set_var("hc_test_secret_key", "SECRET_VALUE_456");
        std::env::set_var("hc_test_account", "acct99");

        let toml_data = r#"
s3_endpoint = "https://${hc_test_account}.r2.cloudflarestorage.com"
s3_access_key = $hc_test_access_key # unquoted env var
s3_secret_key = "$hc_test_secret_key"
"#;

        let cfg = Config::from_toml_str(toml_data).expect("should parse and expand env vars");
        assert_eq!(cfg.s3_access_key, "AKIA_TEST_KEY_123");
        assert_eq!(cfg.s3_secret_key, "SECRET_VALUE_456");
        assert_eq!(cfg.s3_endpoint, "https://acct99.r2.cloudflarestorage.com");

        std::env::remove_var("hc_test_access_key");
        std::env::remove_var("hc_test_secret_key");
        std::env::remove_var("hc_test_account");
    }
}
