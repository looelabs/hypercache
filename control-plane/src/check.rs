use anyhow::Result;
use fs2::FileExt;
use serde::{Deserialize, Serialize};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::path::{Path, PathBuf};
use std::time::Duration;

use crate::config::{self, Config};
use crate::discovery::DeviceDetector;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum CheckStatus {
    Pass,
    Warn,
    Fail,
}

impl std::fmt::Display for CheckStatus {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CheckStatus::Pass => write!(f, "PASS"),
            CheckStatus::Warn => write!(f, "WARN"),
            CheckStatus::Fail => write!(f, "FAIL"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckItem {
    pub category: String,
    pub name: String,
    pub status: CheckStatus,
    pub detail: String,
    pub remediation: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckReport {
    pub ready: bool,
    pub pass_count: usize,
    pub warn_count: usize,
    pub fail_count: usize,
    pub checks: Vec<CheckItem>,
}

/// Executes the comprehensive pre-flight host and configuration verification suite.
/// Reports whether a filesystem arena exists or can be provisioned with its configured size.
/// Returns the check item and, when the arena is usable, its size.
fn check_arena(config: &Config, raw_path: &str) -> (CheckItem, Option<u64>) {
    let arena = DeviceDetector::arena_file_for(raw_path);
    let size = config.filesystem_arena_size_bytes.max(DeviceDetector::MIN_ARENA_BYTES);
    let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
    let item = |status, detail: String, remediation: Option<&str>| CheckItem {
        category: "Storage".to_string(),
        name: format!("Arena {}", arena.display()),
        status,
        detail,
        remediation: remediation.map(str::to_string),
    };

    let existing = fs::metadata(&arena).ok();
    let length = existing.as_ref().map(|m| m.len()).unwrap_or(size);
    let missing = length.saturating_sub(DeviceDetector::allocated_bytes(&arena));
    // Free space on the filesystem that holds (or will hold) the arena
    let mut probe_dir = arena.parent().unwrap_or_else(|| Path::new("/")).to_path_buf();
    while !probe_dir.exists() {
        match probe_dir.parent() {
            Some(p) => probe_dir = p.to_path_buf(),
            None => break,
        }
    }
    let free = fs2::available_space(&probe_dir).unwrap_or(0);

    if let Some(meta) = &existing {
        if meta.permissions().readonly() || OpenOptions::new().read(true).write(true).open(&arena).is_err() {
            return (
                item(
                    CheckStatus::Fail,
                    "Arena file exists but is not readable/writable by this user".to_string(),
                    Some("Run as the daemon user (root) or fix the file's ownership"),
                ),
                None,
            );
        }
    }

    if config.filesystem_arena_preallocate && missing > free {
        return (
            item(
                CheckStatus::Fail,
                format!(
                    "Arena needs {:.2} GiB more on disk but only {:.2} GiB are free on {}; the daemon will refuse to start",
                    gib(missing),
                    gib(free),
                    probe_dir.display()
                ),
                Some("Lower filesystem_arena_size_bytes, free space, or set filesystem_arena_preallocate = false"),
            ),
            None,
        );
    }
    if !config.filesystem_arena_preallocate && missing > free {
        return (
            item(
                CheckStatus::Warn,
                format!(
                    "Sparse arena of {:.2} GiB, but only {:.2} GiB are free: ingests fail with ENOSPC once the disk fills",
                    gib(length),
                    gib(free)
                ),
                Some("Set filesystem_arena_preallocate = true to reserve the space at startup"),
            ),
            Some(length),
        );
    }
    let state = match (&existing, missing) {
        (Some(_), 0) => "exists, fully reserved".to_string(),
        (Some(_), m) => format!("exists, {:.2} GiB still sparse", gib(m)),
        (None, _) => "will be created at daemon start".to_string(),
    };
    (
        item(
            CheckStatus::Pass,
            format!("{:.2} GiB arena {} ({:.2} GiB free)", gib(length), state, gib(free)),
            None,
        ),
        Some(length),
    )
}

pub fn evaluate_host(
    config_path: Option<PathBuf>,
    explicit_devices: Option<Vec<String>>,
    enable_ram_flag: bool,
) -> Result<CheckReport> {
    let mut checks = Vec::new();

    // 1. Configuration loading and DMA alignment verification
    let mut config = match config_path {
        Some(ref p) => {
            if !p.exists() {
                checks.push(CheckItem {
                    category: "Config".to_string(),
                    name: "Configuration File".to_string(),
                    status: CheckStatus::Fail,
                    detail: format!("Specified configuration file {:?} does not exist", p),
                    remediation: Some("Provide a valid path via -c configs/hypercache-agent.toml".to_string()),
                });
                Config::default()
            } else {
                match Config::load_from_file(p) {
                    Ok(cfg) => {
                        checks.push(CheckItem {
                            category: "Config".to_string(),
                            name: "Configuration File".to_string(),
                            status: CheckStatus::Pass,
                            detail: format!("Loaded and validated TOML from {}", p.display()),
                            remediation: None,
                        });
                        cfg
                    }
                    Err(e) => {
                        checks.push(CheckItem {
                            category: "Config".to_string(),
                            name: "Configuration File".to_string(),
                            status: CheckStatus::Fail,
                            detail: format!("Failed to parse TOML at {}: {}", p.display(), e),
                            remediation: Some("Fix syntax errors in the TOML configuration file".to_string()),
                        });
                        Config::default()
                    }
                }
            }
        }
        None => {
            checks.push(CheckItem {
                category: "Config".to_string(),
                name: "Configuration File".to_string(),
                status: CheckStatus::Warn,
                detail: "No -c <config> specified; evaluating with default parameters".to_string(),
                remediation: Some("Pass -c configs/hypercache-agent.toml when starting the daemon".to_string()),
            });
            Config::default()
        }
    };

    if enable_ram_flag {
        config.ram_cache_enabled = true;
    }
    if let Some(devs) = explicit_devices {
        if !devs.is_empty() {
            config.device_discovery_mode = "manual".to_string();
            config.block_devices = devs;
        }
    }

    if config.chunk_size_bytes >= 4096 && config.chunk_size_bytes.is_multiple_of(4096) {
        checks.push(CheckItem {
            category: "Config".to_string(),
            name: "Direct I/O Alignment".to_string(),
            status: CheckStatus::Pass,
            detail: format!(
                "chunk_size_bytes ({} B) and device_start_offset_bytes ({} B) are 4K-aligned",
                config.chunk_size_bytes, config.device_start_offset_bytes
            ),
            remediation: None,
        });
    } else {
        checks.push(CheckItem {
            category: "Config".to_string(),
            name: "Direct I/O Alignment".to_string(),
            status: CheckStatus::Fail,
            detail: format!(
                "chunk_size_bytes ({} B) must be >= 4096 and a multiple of 4096 for O_DIRECT DMA",
                config.chunk_size_bytes
            ),
            remediation: Some(
                "Set chunk_size_bytes = 4194304 (4 MiB) and device_start_offset_bytes = 4096".to_string(),
            ),
        });
    }

    // 2. Host OS, CPU Concurrency & Hardware Acceleration Capabilities
    let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    let worker_threads = (cores * 4).clamp(16, 64);
    checks.push(CheckItem {
        category: "Host".to_string(),
        name: "CPU & Runtime Concurrency".to_string(),
        status: CheckStatus::Pass,
        detail: format!(
            "OS: {}/{}, {} hardware threads detected ({} async I/O worker threads)",
            std::env::consts::OS,
            std::env::consts::ARCH,
            cores,
            worker_threads
        ),
        remediation: None,
    });

    // Check GPUDirect Storage (nvidia-fs) / CUDA availability (informational/optional)
    let gds_available = Path::new("/proc/driver/nvidia-fs/stats").exists();
    let nvidia_dev = Path::new("/dev/nvidiactl").exists();
    if gds_available {
        checks.push(CheckItem {
            category: "Host".to_string(),
            name: "GPUDirect Storage (GDS)".to_string(),
            status: CheckStatus::Pass,
            detail: "nvidia-fs kernel driver detected (/proc/driver/nvidia-fs/stats)".to_string(),
            remediation: None,
        });
    } else if nvidia_dev {
        checks.push(CheckItem {
            category: "Host".to_string(),
            name: "GPUDirect Storage (GDS)".to_string(),
            status: CheckStatus::Pass,
            detail: "NVIDIA GPU detected (/dev/nvidiactl); host O_DIRECT / pinned DMA fallback ready".to_string(),
            remediation: None,
        });
    } else {
        checks.push(CheckItem {
            category: "Host".to_string(),
            name: "GPUDirect Storage (GDS)".to_string(),
            status: CheckStatus::Pass,
            detail: "No NVIDIA GDS driver detected; standard POSIX O_DIRECT / zero-copy mmap path active".to_string(),
            remediation: None,
        });
    }

    // 3. IPC Socket, Host Lockfile & Metadata Persistence Paths
    let socket_path = config::get_socket_path();
    let lock_path = config::get_lock_path();
    let ipc_dir = socket_path.parent().unwrap_or_else(|| Path::new("/tmp/hypercache"));

    match fs::create_dir_all(ipc_dir) {
        Ok(_) => {
            // Check if another daemon holds the exclusive lock
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(false)
                .open(&lock_path)
            {
                Ok(lock_file) => {
                    if lock_file.try_lock_exclusive().is_ok() {
                        let _ = lock_file.unlock();
                        checks.push(CheckItem {
                            category: "IPC".to_string(),
                            name: "Socket & Lockfile Directory".to_string(),
                            status: CheckStatus::Pass,
                            detail: format!(
                                "Writable IPC directory ({}) and lockfile ({}) is free",
                                ipc_dir.display(),
                                lock_path.display()
                            ),
                            remediation: None,
                        });
                    } else {
                        checks.push(CheckItem {
                            category: "IPC".to_string(),
                            name: "Socket & Lockfile Directory".to_string(),
                            status: CheckStatus::Warn,
                            detail: format!(
                                "Lockfile {} is currently held by a running HyperCache daemon",
                                lock_path.display()
                            ),
                            remediation: Some(
                                "Stop the existing daemon before starting a new instance (pkill hypercache)"
                                    .to_string(),
                            ),
                        });
                    }
                }
                Err(e) => {
                    checks.push(CheckItem {
                        category: "IPC".to_string(),
                        name: "Socket & Lockfile Directory".to_string(),
                        status: CheckStatus::Fail,
                        detail: format!("Cannot open or create lockfile {:?}: {}", lock_path, e),
                        remediation: Some(format!("Ensure write permissions on {}", ipc_dir.display())),
                    });
                }
            }
        }
        Err(e) => {
            checks.push(CheckItem {
                category: "IPC".to_string(),
                name: "Socket & Lockfile Directory".to_string(),
                status: CheckStatus::Fail,
                detail: format!("Failed to create IPC directory {:?}: {}", ipc_dir, e),
                remediation: Some("Verify permissions on /run/hypercache or /tmp/hypercache".to_string()),
            });
        }
    }

    if config.metadata_persistence_enabled {
        let meta_path = Path::new(&config.metadata_persistence_path);
        let parent = meta_path.parent().unwrap_or_else(|| Path::new("/var/lib/hypercache"));
        let writable = if fs::create_dir_all(parent).is_ok() {
            let probe = parent.join(".hypercache_write_probe");
            if OpenOptions::new()
                .write(true)
                .create(true)
                .truncate(true)
                .open(&probe)
                .is_ok()
            {
                let _ = fs::remove_file(&probe);
                true
            } else {
                false
            }
        } else {
            false
        };

        if writable {
            checks.push(CheckItem {
                category: "Persistence".to_string(),
                name: "NVMe Metadata Manifest Path".to_string(),
                status: CheckStatus::Pass,
                detail: format!("Manifest directory {} is writable", parent.display()),
                remediation: None,
            });
        } else {
            let fallback_dir = Path::new("/tmp/hypercache");
            if fs::create_dir_all(fallback_dir).is_ok() {
                checks.push(CheckItem {
                    category: "Persistence".to_string(),
                    name: "NVMe Metadata Manifest Path".to_string(),
                    status: CheckStatus::Warn,
                    detail: format!(
                        "Primary path {} is not writable by current user; fallback /tmp/hypercache/metadata.json will be used",
                        parent.display()
                    ),
                    remediation: Some(format!(
                        "Run as root or grant write permissions to {}",
                        parent.display()
                    )),
                });
            } else {
                checks.push(CheckItem {
                    category: "Persistence".to_string(),
                    name: "NVMe Metadata Manifest Path".to_string(),
                    status: CheckStatus::Fail,
                    detail: format!(
                        "Neither {} nor /tmp/hypercache is writable for metadata persistence",
                        parent.display()
                    ),
                    remediation: Some("Check filesystem permissions on /var/lib/hypercache or /tmp".to_string()),
                });
            }
        }
    }

    // 4. Storage Pool Verification (Tier 2 Block Devices / Filesystem Arenas & Tier 1 RAM)
    let mut readable_writable_devs = 0usize;
    let mut total_pool_bytes = 0u64;
    // Filesystem arenas are inspected, not provisioned: `check` must not create or reserve
    // a 100 GiB arena as a side effect
    let devices = if config.storage_backing_mode.eq_ignore_ascii_case("filesystem") {
        for raw_path in &config.filesystem_cache_paths {
            let (item, usable_bytes) = check_arena(&config, raw_path);
            if let Some(bytes) = usable_bytes {
                readable_writable_devs += 1;
                total_pool_bytes += bytes;
            }
            checks.push(item);
        }
        Vec::new()
    } else {
        DeviceDetector::detect_devices(&config)
    };
    let min_required_per_dev = config.device_start_offset_bytes + (config.chunk_size_bytes as u64);

    for dev in &devices {
        match OpenOptions::new().read(true).write(true).open(&dev.path) {
            Ok(_) => {
                if dev.size_bytes >= min_required_per_dev {
                    readable_writable_devs += 1;
                    total_pool_bytes += dev.size_bytes;
                } else {
                    checks.push(CheckItem {
                        category: "Storage".to_string(),
                        name: format!("Device {}", dev.path),
                        status: CheckStatus::Warn,
                        detail: format!(
                            "Device size ({} B) is smaller than minimum single chunk + offset ({} B)",
                            dev.size_bytes, min_required_per_dev
                        ),
                        remediation: Some("Provision a larger block device or arena".to_string()),
                    });
                }
            }
            Err(e) => {
                checks.push(CheckItem {
                    category: "Storage".to_string(),
                    name: format!("Device {}", dev.path),
                    status: CheckStatus::Fail,
                    detail: format!("Discovered device {} is not readable/writable: {}", dev.path, e),
                    remediation: Some(
                        "Run HyperCache with root/disk group permissions or use storage_backing_mode = \"filesystem\""
                            .to_string(),
                    ),
                });
            }
        }
    }

    if readable_writable_devs > 0 {
        checks.push(CheckItem {
            category: "Storage".to_string(),
            name: "Tier 2 Storage Pool (NVMe / Arena)".to_string(),
            status: CheckStatus::Pass,
            detail: format!(
                "{} accessible device(s)/arena(s) ready (Aggregate Capacity: {:.2} GiB)",
                readable_writable_devs,
                total_pool_bytes as f64 / (1024.0 * 1024.0 * 1024.0)
            ),
            remediation: None,
        });
    } else if config.ram_cache_enabled {
        checks.push(CheckItem {
            category: "Storage".to_string(),
            name: "Tier 2 Storage Pool (NVMe / Arena)".to_string(),
            status: CheckStatus::Warn,
            detail: "No Tier 2 block devices or arenas detected; operating in Tier 1 Host RAM-only mode".to_string(),
            remediation: Some(
                "Attach unformatted NVMe drives or configure filesystem_cache_paths for Tier 2 persistence".to_string(),
            ),
        });
    } else {
        checks.push(CheckItem {
            category: "Storage".to_string(),
            name: "Tier 2 Storage Pool (NVMe / Arena)".to_string(),
            status: CheckStatus::Fail,
            detail: "No accessible raw block devices or filesystem arenas found, and Tier 1 RAM cache is disabled".to_string(),
            remediation: Some(
                "Attach raw drives, configure filesystem_cache_paths with storage_backing_mode = \"filesystem\", or enable --ram / auto_create_mock_devices = true"
                    .to_string(),
            ),
        });
    }

    // 5. Tier 1 Host RAM / Shared Memory Probe
    let shm_dir = config::get_ram_pool_path(&config);
    match fs::create_dir_all(&shm_dir) {
        Ok(_) => {
            let probe_file = shm_dir.join(".hypercache_shm_probe");
            match OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&probe_file)
                .and_then(|mut f| {
                    f.write_all(&[0u8; 4096])?;
                    f.sync_all()
                }) {
                Ok(_) => {
                    let _ = fs::remove_file(&probe_file);
                    let avail = fs2::available_space(&shm_dir).unwrap_or(0);
                    let auto_cap = config.ram_cache_capacity_bytes.unwrap_or(8 * 1024 * 1024 * 1024);
                    let status_label = if config.ram_cache_enabled {
                        "ENABLED"
                    } else {
                        "READY (Disabled in config)"
                    };
                    checks.push(CheckItem {
                        category: "Memory".to_string(),
                        name: "Tier 1 Shared Memory Pool".to_string(),
                        status: CheckStatus::Pass,
                        detail: format!(
                            "{} at {} (Available FS Space: {:.2} GiB, Target Pool: {:.2} GiB)",
                            status_label,
                            shm_dir.display(),
                            avail as f64 / (1024.0 * 1024.0 * 1024.0),
                            auto_cap as f64 / (1024.0 * 1024.0 * 1024.0)
                        ),
                        remediation: None,
                    });
                }
                Err(e) => {
                    let st = if config.ram_cache_enabled {
                        CheckStatus::Fail
                    } else {
                        CheckStatus::Warn
                    };
                    checks.push(CheckItem {
                        category: "Memory".to_string(),
                        name: "Tier 1 Shared Memory Pool".to_string(),
                        status: st,
                        detail: format!("Shared memory probe failed at {}: {}", shm_dir.display(), e),
                        remediation: Some("Ensure /dev/shm or /tmp/hypercache/shm is mounted and writable".to_string()),
                    });
                }
            }
        }
        Err(e) => {
            let st = if config.ram_cache_enabled {
                CheckStatus::Fail
            } else {
                CheckStatus::Warn
            };
            checks.push(CheckItem {
                category: "Memory".to_string(),
                name: "Tier 1 Shared Memory Pool".to_string(),
                status: st,
                detail: format!("Cannot create shared memory directory {}: {}", shm_dir.display(), e),
                remediation: Some("Verify permissions on /dev/shm or /tmp".to_string()),
            });
        }
    }

    // 6. Authoritative Backend & Cluster Controller Reachability
    if config.authoritative_backend.eq_ignore_ascii_case("nfs") {
        let nfs_path = Path::new(&config.nfs_mount_path);
        if nfs_path.exists() && nfs_path.is_dir() {
            checks.push(CheckItem {
                category: "Backend".to_string(),
                name: "NFS Mount Path".to_string(),
                status: CheckStatus::Pass,
                detail: format!("NFS mount directory {} exists and is accessible", nfs_path.display()),
                remediation: None,
            });
        } else {
            checks.push(CheckItem {
                category: "Backend".to_string(),
                name: "NFS Mount Path".to_string(),
                status: CheckStatus::Warn,
                detail: format!("Configured nfs_mount_path ({}) does not exist yet", nfs_path.display()),
                remediation: Some(format!("Mount your shared POSIX filesystem at {}", nfs_path.display())),
            });
        }
    } else {
        match probe_http_endpoint(&config.s3_endpoint, Duration::from_millis(400)) {
            Ok(addr) => {
                checks.push(CheckItem {
                    category: "Backend".to_string(),
                    name: "S3 Authoritative Endpoint".to_string(),
                    status: CheckStatus::Pass,
                    detail: format!("S3 endpoint {} ({}) is reachable", config.s3_endpoint, addr),
                    remediation: None,
                });
            }
            Err(reason) => {
                checks.push(CheckItem {
                    category: "Backend".to_string(),
                    name: "S3 Authoritative Endpoint".to_string(),
                    status: CheckStatus::Warn,
                    detail: format!(
                        "S3 endpoint {} is not currently reachable ({}); local cache hits still work",
                        config.s3_endpoint, reason
                    ),
                    remediation: Some(
                        "Verify s3_endpoint URL and network connectivity before ingesting uncached S3 models"
                            .to_string(),
                    ),
                });
            }
        }
    }

    if let Some(ref ctrl_url) = config.controller_url {
        match probe_http_endpoint(ctrl_url, Duration::from_millis(400)) {
            Ok(addr) => {
                checks.push(CheckItem {
                    category: "Cluster".to_string(),
                    name: "Central Controller Connectivity".to_string(),
                    status: CheckStatus::Pass,
                    detail: format!("Controller {} ({}) is reachable", ctrl_url, addr),
                    remediation: None,
                });
            }
            Err(reason) => {
                checks.push(CheckItem {
                    category: "Cluster".to_string(),
                    name: "Central Controller Connectivity".to_string(),
                    status: CheckStatus::Warn,
                    detail: format!("Controller {} is not currently reachable ({})", ctrl_url, reason),
                    remediation: Some(
                        "Ensure ./hypercache-admin controller is running on the coordinator host".to_string(),
                    ),
                });
            }
        }
    }

    let pass_count = checks.iter().filter(|c| c.status == CheckStatus::Pass).count();
    let warn_count = checks.iter().filter(|c| c.status == CheckStatus::Warn).count();
    let fail_count = checks.iter().filter(|c| c.status == CheckStatus::Fail).count();

    Ok(CheckReport {
        ready: fail_count == 0,
        pass_count,
        warn_count,
        fail_count,
        checks,
    })
}

fn probe_http_endpoint(url: &str, timeout: Duration) -> std::result::Result<SocketAddr, String> {
    let trimmed = url.trim();
    let (scheme, rest) = if let Some(r) = trimmed.strip_prefix("https://") {
        ("https", r)
    } else if let Some(r) = trimmed.strip_prefix("http://") {
        ("http", r)
    } else {
        return Err("URL must start with http:// or https://".to_string());
    };

    let host_port = rest.split('/').next().unwrap_or("");
    if host_port.is_empty() {
        return Err("Missing host in URL".to_string());
    }

    let addr_str = if host_port.contains(':') {
        host_port.to_string()
    } else if scheme == "https" {
        format!("{}:443", host_port)
    } else {
        format!("{}:80", host_port)
    };

    let mut addrs = addr_str
        .to_socket_addrs()
        .map_err(|e| format!("DNS/address resolution failed: {}", e))?;

    let first_addr = addrs.next().ok_or_else(|| "No socket address resolved".to_string())?;

    TcpStream::connect_timeout(&first_addr, timeout).map_err(|e| format!("TCP connect timeout/refused: {}", e))?;

    Ok(first_addr)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_evaluate_host_with_ram_enabled_succeeds() {
        // An empty prefix list keeps discovery off the host's raw disks: a CI runner may expose an
        // unmounted disk the test user cannot open, which `check` correctly reports as FAIL.
        let temp_dir = std::env::temp_dir().join(format!("hc_check_ram_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();
        let cfg_path = temp_dir.join("ram_only.toml");
        std::fs::write(
            &cfg_path,
            format!(
                "device_prefixes = []\nmetadata_persistence_path = \"{}/metadata.json\"\n",
                temp_dir.display()
            ),
        )
        .unwrap();

        let report = evaluate_host(Some(cfg_path), None, true).expect("evaluate_host should succeed");
        let _ = std::fs::remove_dir_all(&temp_dir);
        assert!(
            report.ready,
            "Host check with RAM enabled should be ready even without raw block devices: {:?}",
            report.checks
        );
        assert_eq!(report.fail_count, 0);
        assert!(report.pass_count >= 4);
    }

    #[test]
    fn test_evaluate_host_with_missing_config_reports_failure() {
        let bad_path = PathBuf::from("/tmp/non_existent_hypercache_config_12345.toml");
        let report = evaluate_host(Some(bad_path), None, true).expect("evaluate_host should return report");
        assert!(!report.ready);
        assert!(report.fail_count >= 1);
        assert!(report
            .checks
            .iter()
            .any(|c| c.name == "Configuration File" && c.status == CheckStatus::Fail));
    }

    #[test]
    fn test_evaluate_host_with_filesystem_arena_config() {
        let temp_dir = std::env::temp_dir().join(format!("hc_check_arena_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        std::fs::create_dir_all(&temp_dir).unwrap();

        let cfg_path = temp_dir.join("test_agent.toml");
        let toml_content = format!(
            r#"
storage_backing_mode = "filesystem"
filesystem_cache_paths = ["{}"]
filesystem_arena_size_bytes = 16777216
chunk_size_bytes = 4194304
device_start_offset_bytes = 4096
metadata_persistence_enabled = true
metadata_persistence_path = "{}/metadata.json"
"#,
            temp_dir.display(),
            temp_dir.display()
        );
        std::fs::write(&cfg_path, toml_content).unwrap();

        let report = evaluate_host(Some(cfg_path), None, false).expect("evaluate_host should succeed");
        let _ = std::fs::remove_dir_all(&temp_dir);

        assert!(
            report.ready,
            "Host check with valid filesystem arena config should pass: {:?}",
            report.checks
        );
        assert_eq!(report.fail_count, 0);
    }
}
