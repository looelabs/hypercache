use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use tracing::{error, info, warn};

#[cfg(target_os = "linux")]
use std::collections::HashSet;
#[cfg(target_os = "linux")]
use std::fs::File;
#[cfg(target_os = "linux")]
use std::io::Read;

use crate::config::Config;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiscoveredDevice {
    pub path: String,
    pub is_nvme: bool,
    pub is_virtual: bool,
    pub is_partition: bool,
    pub is_rotational: bool,
    pub size_bytes: u64,
    pub model: Option<String>,
    pub existing_fs: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub numa_node: Option<i32>,
}

pub struct DeviceDetector;

impl DeviceDetector {
    /// Auto-discovers available physical NVMe, SSD, and virtual block devices (e.g. /dev/vdc, /dev/vdc1).
    /// Enforces strict mount-checking and existing filesystem signature detection to protect OS/data drives.
    pub fn detect_devices(config: &Config) -> Vec<DiscoveredDevice> {
        let mode = config.device_discovery_mode.to_lowercase();
        info!("Running block device discovery (mode: '{}')...", mode);

        let backing_mode = config.storage_backing_mode.to_lowercase();
        if backing_mode == "filesystem" {
            // Never fall back to raw disks here: a privileged pod sees the host's unmounted disks,
            // and striping onto them would overwrite whatever they hold.
            return match Self::try_provision_filesystem_arenas(config) {
                Ok(arenas) => arenas,
                Err(_) => {
                    error!(
                        "storage_backing_mode = \"filesystem\": arenas from filesystem_cache_paths = {:?} could \
                         not all be provisioned; raw block devices are not used in this mode",
                        config.filesystem_cache_paths
                    );
                    Vec::new()
                }
            };
        }

        if mode == "manual" && !config.block_devices.is_empty() {
            info!(
                "Manual device mode enabled. Using {} configured device(s).",
                config.block_devices.len()
            );
            return config
                .block_devices
                .iter()
                .map(|p| {
                    let dev_name = p.strip_prefix("/dev/").unwrap_or(p);
                    let sysfs_size_path = format!("/sys/class/block/{}/size", dev_name);
                    let real_size = if let Ok(size_str) = fs::read_to_string(&sysfs_size_path) {
                        size_str.trim().parse::<u64>().unwrap_or(0) * 512
                    } else if let Ok(meta) = fs::metadata(p) {
                        meta.len()
                    } else {
                        config.mock_device_size_bytes
                    };

                    let sysfs_model_path = format!("/sys/class/block/{}/device/model", dev_name);
                    let model = fs::read_to_string(&sysfs_model_path)
                        .map(|s| s.trim().to_string())
                        .ok()
                        .or_else(|| Some("Explicit Block Device".to_string()));

                    let sysfs_numa_path = format!("/sys/class/block/{}/device/numa_node", dev_name);
                    let numa_node = fs::read_to_string(&sysfs_numa_path)
                        .ok()
                        .and_then(|s| s.trim().parse::<i32>().ok())
                        .filter(|&n| n >= 0);

                    DiscoveredDevice {
                        path: p.clone(),
                        is_nvme: p.contains("nvme"),
                        is_virtual: p.contains("/dev/vd") || p.contains("/dev/xvd"),
                        is_partition: p.chars().last().map(|c| c.is_numeric()).unwrap_or(false) && !p.ends_with("n1"),
                        is_rotational: false,
                        size_bytes: if real_size > 0 {
                            real_size
                        } else {
                            config.mock_device_size_bytes
                        },
                        model,
                        existing_fs: None,
                        numa_node,
                    }
                })
                .collect();
        }

        // 1. Attempt Linux hardware and virtual drive detection
        #[cfg(target_os = "linux")]
        {
            let linux_devices = Self::detect_linux_devices(config);
            if !linux_devices.is_empty() {
                info!("------------------------------------------------------------");
                info!(
                    "Discovered {} Safe, Unmounted Block Device(s) for Striping Pool:",
                    linux_devices.len()
                );
                for dev in &linux_devices {
                    let type_str = if dev.is_nvme {
                        "NVMe"
                    } else if dev.is_virtual {
                        "Virtual Disk (VirtIO/Cloud)"
                    } else {
                        "SSD/Block Device"
                    };

                    let part_str = if dev.is_partition {
                        " [Partition]"
                    } else {
                        " [Whole Disk]"
                    };
                    let size_str = if dev.size_bytes >= 1024 * 1024 * 1024 * 1024 {
                        format!("{:.2} TiB", dev.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0 * 1024.0))
                    } else {
                        format!("{:.2} GiB", dev.size_bytes as f64 / (1024.0 * 1024.0 * 1024.0))
                    };

                    info!(
                        "  -> {:<16} | Type: {:<26}{} | Size: {:>9} | Model: {}",
                        dev.path,
                        type_str,
                        part_str,
                        size_str,
                        dev.model.as_deref().unwrap_or("Unknown")
                    );
                }
                info!("------------------------------------------------------------");
                return linux_devices;
            }
        }

        // 2. Check for filesystem-backed Direct I/O cache paths if configured (Kubernetes PVC / hostPath / XFS / ext4 GDS)
        if !config.filesystem_cache_paths.is_empty() && backing_mode != "raw_block" {
            if let Ok(arenas) = Self::try_provision_filesystem_arenas(config) {
                if !arenas.is_empty() {
                    return arenas;
                }
            }
        }

        // 3. Check for mock devices ONLY if explicitly enabled in configuration
        if config.auto_create_mock_devices {
            let pattern = &config.dev_mock_device_pattern;
            let mock_devices = Self::scan_mock_devices(pattern);
            if !mock_devices.is_empty() {
                info!(
                    "Discovered {} existing mock block device file(s) matching '{}':",
                    mock_devices.len(),
                    pattern
                );
                for d in &mock_devices {
                    info!("  -> {} ({:.2} MiB)", d.path, d.size_bytes as f64 / (1024.0 * 1024.0));
                }
                return mock_devices;
            }

            let count = config.dev_mock_auto_count.max(1);
            info!("No unmounted physical or virtual block drives found. Auto-provisioning {} mock NVMe device backing file(s)...", count);
            let mut created = Vec::new();
            for i in 0..count {
                let dev_path = format!("/tmp/hypercache_nvme{}.raw", i);
                if let Err(e) = Self::ensure_file_size(&dev_path, config.mock_device_size_bytes) {
                    warn!("Failed to create mock device {}: {}", dev_path, e);
                } else {
                    created.push(DiscoveredDevice {
                        path: dev_path,
                        is_nvme: true,
                        is_virtual: true,
                        is_partition: false,
                        is_rotational: false,
                        size_bytes: config.mock_device_size_bytes,
                        model: Some("HyperCache-Mock-NVMe".to_string()),
                        existing_fs: None,
                        numa_node: Some(0),
                    });
                }
            }
            return created;
        }

        info!("No secondary raw NVMe/block devices found (OS root drive protected).");
        Vec::new()
    }

    /// Comprehensive Linux block device scanner with mounted filesystem and superblock protection.
    #[cfg(target_os = "linux")]
    fn detect_linux_devices(config: &Config) -> Vec<DiscoveredDevice> {
        let mut devices = Vec::new();
        let mut excluded_by_partition_fs = HashSet::new();

        // 1. Gather all mounted devices & partition parents from /proc/mounts and /proc/swaps
        let mounted_set = if config.exclude_mounted_devices {
            Self::get_all_mounted_and_in_use_devices()
        } else {
            HashSet::new()
        };

        if !mounted_set.is_empty() {
            info!("Identified in-use / mounted device trees to exclude: {:?}", mounted_set);
        }

        // Scan all block entries in /sys/class/block
        let block_class = Path::new("/sys/class/block");
        if !block_class.exists() {
            return devices;
        }

        if let Ok(entries) = fs::read_dir(block_class) {
            for entry in entries.flatten() {
                let name = entry.file_name().to_string_lossy().to_string();

                // Check prefix match against configured prefixes (e.g. nvme, vd, xvd, sd)
                let matches_prefix = config.device_prefixes.iter().any(|prefix| name.starts_with(prefix));
                if !matches_prefix {
                    continue;
                }

                // Check custom filter if set (e.g. "vdc" or "vdc1" or "vd*")
                if !config.device_filter.is_empty() {
                    let filter = config.device_filter.trim_start_matches("/dev/");
                    if filter.ends_with('*') {
                        let prefix = filter.trim_end_matches('*');
                        if !name.starts_with(prefix) {
                            continue;
                        }
                    } else if name != filter {
                        continue;
                    }
                }

                let dev_path = format!("/dev/{}", name);

                // sysfs can list devices that have no node in this mount namespace (containers
                // and pods without a /dev mount see the host VM's disks); they are unusable here.
                if !Path::new(&dev_path).exists() {
                    tracing::debug!("Skipping {}: listed in sysfs but no device node exists", dev_path);
                    continue;
                }

                // 2. Check if this device or its parent disk is in the mounted set (e.g. vda1, vda14, vda)
                let base_disk = Self::extract_base_disk_name(&name);
                if mounted_set.contains(&name) || mounted_set.contains(&base_disk) || mounted_set.contains(&dev_path) {
                    info!(
                        "Excluding mounted / in-use device: {} (base disk: {})",
                        dev_path, base_disk
                    );
                    continue;
                }

                let dev_sys_path = entry.path();

                // Determine whether this is a partition or whole disk
                let is_partition = dev_sys_path.join("partition").exists()
                    || (name.starts_with("vd") && name.chars().last().unwrap_or(' ').is_numeric())
                    || (name.starts_with("sd") && name.chars().last().unwrap_or(' ').is_numeric())
                    || (name.contains("nvme") && name.contains('p'));

                if is_partition && !config.include_partitions {
                    continue;
                }

                let is_virtual = name.starts_with("vd") || name.starts_with("xvd");
                if is_virtual && !config.allow_virtual_drives {
                    continue;
                }

                let is_nvme = name.starts_with("nvme");

                // Check rotational flag if present (allow virtual disks regardless)
                let is_rotational = Self::check_rotational(&dev_sys_path);
                if is_rotational && !is_virtual {
                    continue; // Skip spinning HDDs
                }

                // 3. Filesystem Signature Detection: Check if unmounted disk/partition already has a formatted filesystem
                let existing_fs = Self::detect_filesystem_signature(&dev_path);
                if let Some(ref fs_type) = existing_fs {
                    let is_forced = config.force_use_devices.iter().any(|d| d == &dev_path || d == &name);
                    if config.exclude_devices_with_existing_fs && !is_forced {
                        info!(
                            "Excluding device {} because it contains existing '{}' filesystem signature (add to force_use_devices in configs/hypercache-agent.toml to override)",
                            dev_path, fs_type
                        );
                        if is_partition {
                            excluded_by_partition_fs.insert(base_disk.clone());
                        }
                        continue;
                    } else if is_forced {
                        warn!(
                            "Device {} has existing '{}' filesystem, but force_use_devices is set. Proceeding.",
                            dev_path, fs_type
                        );
                    }
                }

                let size_bytes = Self::read_sysfs_size(&dev_sys_path);
                let model = Self::read_sysfs_model(&dev_sys_path, &name);
                let numa_node = Self::extract_numa_node(&dev_sys_path);

                devices.push(DiscoveredDevice {
                    path: dev_path,
                    is_nvme,
                    is_virtual,
                    is_partition,
                    is_rotational,
                    size_bytes,
                    model,
                    existing_fs,
                    numa_node,
                });
            }
        }

        // Deduplicate parent vs partition: if a partition exists (e.g. vdc1), don't also grab parent vdc
        if config.include_partitions {
            let partition_bases: Vec<String> = devices
                .iter()
                .filter(|d| d.is_partition)
                .map(|d| {
                    let p = d.path.trim_start_matches("/dev/");
                    Self::extract_base_disk_name(p)
                })
                .collect();

            devices.retain(|d| {
                let base = d.path.trim_start_matches("/dev/");
                let is_forced = config.force_use_devices.iter().any(|f| f == &d.path || f == base);

                if !d.is_partition {
                    // If partition exists or partition was excluded due to existing fs, exclude parent disk
                    if partition_bases.contains(&base.to_string())
                        || (excluded_by_partition_fs.contains(base) && !is_forced)
                    {
                        return false;
                    }
                }
                true
            });
        }

        devices.sort_by(|a, b| a.path.cmp(&b.path));
        devices
    }

    /// Reads /proc/mounts, /proc/swaps, and sysfs holders to build a complete set of in-use device names & parent disks.
    #[cfg(target_os = "linux")]
    fn get_all_mounted_and_in_use_devices() -> HashSet<String> {
        let mut set = HashSet::new();

        // 1. Check /proc/mounts
        if let Ok(mounts) = fs::read_to_string("/proc/mounts") {
            for line in mounts.lines() {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if !parts.is_empty() && parts[0].starts_with("/dev/") {
                    let dev = parts[0].trim_start_matches("/dev/");
                    set.insert(dev.to_string());
                    set.insert(parts[0].to_string());
                    // Also add base disk
                    set.insert(Self::extract_base_disk_name(dev));
                }
            }
        }

        // 2. Check /proc/swaps
        if let Ok(swaps) = fs::read_to_string("/proc/swaps") {
            for line in swaps.lines().skip(1) {
                let parts: Vec<&str> = line.split_whitespace().collect();
                if !parts.is_empty() && parts[0].starts_with("/dev/") {
                    let dev = parts[0].trim_start_matches("/dev/");
                    set.insert(dev.to_string());
                    set.insert(parts[0].to_string());
                    set.insert(Self::extract_base_disk_name(dev));
                }
            }
        }

        // 3. Check sysfs holders (e.g. LVM, mdraid, dm-crypt)
        if let Ok(entries) = fs::read_dir("/sys/class/block") {
            for entry in entries.flatten() {
                let holders_dir = entry.path().join("holders");
                if holders_dir.exists() {
                    if let Ok(mut h_entries) = fs::read_dir(&holders_dir) {
                        if h_entries.next().is_some() {
                            let name = entry.file_name().to_string_lossy().to_string();
                            set.insert(name.clone());
                            set.insert(Self::extract_base_disk_name(&name));
                        }
                    }
                }
            }
        }

        set
    }

    /// Extracts the parent disk from a partition name (e.g. "vda1" -> "vda", "nvme0n1p1" -> "nvme0n1", "sda2" -> "sda")
    #[allow(dead_code)]
    pub fn extract_base_disk_name(name: &str) -> String {
        let clean = name.trim_start_matches("/dev/");
        if clean.contains("nvme") {
            if let Some((base, part)) = clean.rsplit_once('p') {
                if !part.is_empty() && part.chars().all(|c| c.is_numeric()) {
                    return base.to_string();
                }
            }
            return clean.to_string();
        }

        let mut non_digit_len = clean.len();
        while non_digit_len > 0 && clean.as_bytes()[non_digit_len - 1].is_ascii_digit() {
            non_digit_len -= 1;
        }

        if non_digit_len > 0 && non_digit_len < clean.len() {
            clean[..non_digit_len].to_string()
        } else {
            clean.to_string()
        }
    }

    /// Reads magic bytes from block device header / superblocks to identify existing filesystems.
    #[cfg(target_os = "linux")]
    fn detect_filesystem_signature(dev_path: &str) -> Option<String> {
        let mut file = match File::open(dev_path) {
            Ok(f) => f,
            Err(_) => return None,
        };

        let mut buf = vec![0u8; 131072]; // Read first 128 KiB
        if file.read_exact(&mut buf).is_err() {
            return None;
        }

        // 1. HyperCache Magic Header Check (Allowed!)
        if buf.starts_with(b"HYPERCACHE") {
            return None;
        }

        // 2. XFS Superblock (Offset 0: "XFSB")
        if &buf[0..4] == b"XFSB" {
            return Some("xfs".to_string());
        }

        // 3. ext2 / ext3 / ext4 Superblock (Offset 1080: 0xef53)
        if buf.len() > 1082 && buf[1080] == 0x53 && buf[1081] == 0xef {
            return Some("ext2/3/4".to_string());
        }

        // 4. Btrfs Superblock (Offset 65536 / 0x10000: "_BHRfS_M")
        if buf.len() >= 65536 + 8 && &buf[65536..65536 + 8] == b"_BHRfS_M" {
            return Some("btrfs".to_string());
        }

        // 5. VFAT / FAT Superblock (Offset 54: "FAT12" or "FAT16", Offset 82: "FAT32")
        if buf.len() > 90 && (&buf[54..59] == b"FAT12" || &buf[54..59] == b"FAT16" || &buf[82..87] == b"FAT32") {
            return Some("vfat/fat".to_string());
        }

        // 6. Linux Swap (Offset 4086: "SWAPSPACE2" or "SWAP-SPACE")
        if buf.len() > 4096 && (&buf[4086..4096] == b"SWAPSPACE2" || &buf[4086..4096] == b"SWAP-SPACE") {
            return Some("swap".to_string());
        }

        // 7. NTFS (Offset 3: "NTFS    ")
        if buf.len() > 11 && &buf[3..11] == b"NTFS    " {
            return Some("ntfs".to_string());
        }

        // 8. ISO9660 (Offset 32769: "CD001")
        if buf.len() >= 32774 && &buf[32769..32774] == b"CD001" {
            return Some("iso9660".to_string());
        }

        None
    }

    #[cfg(target_os = "linux")]
    fn check_rotational(dev_sys_path: &Path) -> bool {
        let rotational_file = dev_sys_path.join("queue/rotational");
        if rotational_file.exists() {
            fs::read_to_string(&rotational_file)
                .map(|s| s.trim() == "1")
                .unwrap_or(false)
        } else {
            false
        }
    }

    #[cfg(target_os = "linux")]
    fn read_sysfs_size(dev_sys_path: &Path) -> u64 {
        let size_path = dev_sys_path.join("size");
        if let Ok(s) = fs::read_to_string(size_path) {
            if let Ok(sectors) = s.trim().parse::<u64>() {
                return sectors * 512;
            }
        }
        0
    }

    #[cfg(target_os = "linux")]
    fn read_sysfs_model(dev_sys_path: &Path, name: &str) -> Option<String> {
        let model_path = dev_sys_path.join("device/model");
        if let Ok(s) = fs::read_to_string(model_path) {
            let trimmed = s.trim();
            if !trimmed.is_empty() {
                return Some(trimmed.to_string());
            }
        }
        if name.starts_with("vd") {
            Some("VirtIO Virtual Block Device".to_string())
        } else if name.starts_with("xvd") {
            Some("Xen/AWS Virtual Block Device".to_string())
        } else {
            None
        }
    }

    #[allow(dead_code)]
    pub fn extract_numa_node(dev_sys_path: &Path) -> Option<i32> {
        let direct_numa = dev_sys_path.join("device/numa_node");
        if direct_numa.exists() {
            if let Ok(content) = fs::read_to_string(&direct_numa) {
                if let Ok(node) = content.trim().parse::<i32>() {
                    if node >= 0 {
                        return Some(node);
                    }
                }
            }
        }
        let parent_numa = dev_sys_path.join("../device/numa_node");
        if parent_numa.exists() {
            if let Ok(content) = fs::read_to_string(&parent_numa) {
                if let Ok(node) = content.trim().parse::<i32>() {
                    if node >= 0 {
                        return Some(node);
                    }
                }
            }
        }
        None
    }

    fn scan_mock_devices(pattern: &str) -> Vec<DiscoveredDevice> {
        let mut results = Vec::new();
        let search_dir = Path::new(pattern).parent().unwrap_or_else(|| Path::new("/tmp"));
        let prefix = "hypercache_nvme";

        if let Ok(entries) = fs::read_dir(search_dir) {
            for entry in entries.flatten() {
                let file_name = entry.file_name().to_string_lossy().to_string();
                if file_name.starts_with(prefix) && file_name.ends_with(".raw") {
                    let full_path = entry.path().to_string_lossy().to_string();
                    let size_bytes = entry.metadata().map(|m| m.len()).unwrap_or(0);
                    results.push(DiscoveredDevice {
                        path: full_path,
                        is_nvme: true,
                        is_virtual: true,
                        is_partition: false,
                        is_rotational: false,
                        size_bytes,
                        model: Some("HyperCache-Mock-NVMe".to_string()),
                        existing_fs: None,
                        numa_node: Some(0),
                    });
                }
            }
        }

        results.sort_by(|a, b| a.path.cmp(&b.path));
        results
    }

    fn ensure_file_size(path_str: &str, size_bytes: u64) -> Result<()> {
        let path = Path::new(path_str);
        if let Some(parent) = path.parent() {
            let _ = fs::create_dir_all(parent);
        }
        if !path.exists() {
            let file = fs::File::create(path).with_context(|| format!("Failed to create file {:?}", path))?;
            file.set_len(size_bytes)
                .with_context(|| format!("Failed to set length for {:?}", path))?;
        }
        Ok(())
    }

    /// Minimum arena size; smaller configured sizes are rounded up.
    pub const MIN_ARENA_BYTES: u64 = 64 * 1024 * 1024;

    /// The arena file a `filesystem_cache_paths` entry resolves to: the path itself when it
    /// names a `.dat` / `.raw` file, otherwise `hypercache_arena.dat` inside that directory.
    pub fn arena_file_for(raw_path: &str) -> std::path::PathBuf {
        let p = Path::new(raw_path);
        if raw_path.ends_with(".dat") || raw_path.ends_with(".raw") {
            p.to_path_buf()
        } else {
            p.join("hypercache_arena.dat")
        }
    }

    /// Bytes on disk actually backing a file (sparse files report less than their length).
    pub fn allocated_bytes(path: &Path) -> u64 {
        use std::os::unix::fs::MetadataExt;
        fs::metadata(path).map(|m| m.blocks() * 512).unwrap_or(0)
    }

    /// Creates (or completes) an arena file of `size_bytes`. With `preallocate`, the space is
    /// reserved on disk now, so a filesystem that is too small fails here, at startup, instead
    /// of with write errors once ingests fill it. Existing arenas keep their length; only their
    /// holes are filled. Filesystems without fallocate support fall back to a sparse file.
    fn ensure_arena_file(path: &Path, size_bytes: u64, preallocate: bool) -> Result<()> {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).with_context(|| format!("Cannot create arena directory {:?}", parent))?;
        }
        let is_new = !path.exists();
        let target_len = if is_new { size_bytes } else { fs::metadata(path)?.len() };

        if preallocate {
            let missing = target_len.saturating_sub(Self::allocated_bytes(path));
            let dir = path.parent().unwrap_or_else(|| Path::new("."));
            let free = fs2::available_space(dir).unwrap_or(u64::MAX);
            if missing > free {
                anyhow::bail!(
                    "arena {:?} needs {} more bytes ({:.2} GiB) but only {} ({:.2} GiB) are free on that filesystem. \
                     Lower filesystem_arena_size_bytes, free space, or set filesystem_arena_preallocate = false \
                     to accept a sparse arena that can run out of space later",
                    path,
                    missing,
                    missing as f64 / GIB,
                    free,
                    free as f64 / GIB
                );
            }
        }

        let file = fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(false)
            .open(path)
            .with_context(|| format!("Failed to open arena file {:?}", path))?;

        let reserved = if preallocate {
            match reserve_space(&file, target_len) {
                Ok(()) => true,
                Err(e) if e.raw_os_error() == Some(libc::EOPNOTSUPP) => {
                    warn!(
                        "Filesystem holding {:?} cannot preallocate (fallocate unsupported); using a sparse arena",
                        path
                    );
                    false
                }
                Err(e) => {
                    if is_new {
                        let _ = fs::remove_file(path);
                    }
                    return Err(anyhow::Error::new(e)
                        .context(format!("Failed to reserve {} bytes for arena {:?}", target_len, path)));
                }
            }
        } else {
            false
        };
        if !reserved && file.metadata()?.len() < target_len {
            file.set_len(target_len)
                .with_context(|| format!("Failed to set length for {:?}", path))?;
        }
        Ok(())
    }

    /// Provisions Direct I/O arena files inside mounted XFS/ext4 directories for unprivileged
    /// Kubernetes pods and native GPUDirect Storage (cuFile). Every configured path must
    /// succeed; the error names each one that failed.
    pub fn try_provision_filesystem_arenas(config: &Config) -> Result<Vec<DiscoveredDevice>> {
        let mut arenas = Vec::new();
        let mut failures = Vec::new();
        let arena_size = config.filesystem_arena_size_bytes.max(Self::MIN_ARENA_BYTES);

        for raw_path in &config.filesystem_cache_paths {
            let arena_file = Self::arena_file_for(raw_path);
            let arena_str = arena_file.to_string_lossy().to_string();
            if let Err(e) = Self::ensure_arena_file(&arena_file, arena_size, config.filesystem_arena_preallocate) {
                error!(
                    "Failed to provision filesystem Direct I/O arena at {}: {:#}",
                    arena_str, e
                );
                failures.push(format!("{}: {:#}", arena_str, e));
                continue;
            }

            crate::access::grant_file_or_warn(&arena_file, config.client_gid);
            let actual_size = fs::metadata(&arena_file).map(|m| m.len()).unwrap_or(arena_size);
            let reserved = Self::allocated_bytes(&arena_file);

            info!(
                "Provisioned Filesystem Direct I/O Arena: {} ({:.2} GiB, {:.2} GiB reserved on disk)",
                arena_str,
                actual_size as f64 / GIB,
                reserved as f64 / GIB
            );

            arenas.push(DiscoveredDevice {
                path: arena_str,
                is_nvme: true,
                is_virtual: false,
                is_partition: false,
                is_rotational: false,
                size_bytes: actual_size,
                model: Some("Filesystem-DirectIO-Arena".to_string()),
                existing_fs: Some("ext4/xfs-arena".to_string()),
                numa_node: Some(0),
            });
        }

        if failures.is_empty() {
            Ok(arenas)
        } else {
            anyhow::bail!("{}", failures.join("; "))
        }
    }

    /// Like `detect_devices`, but a misconfigured storage pool is an error instead of an empty
    /// pool. The daemon uses this so it refuses to start rather than run without its storage.
    pub fn detect_devices_checked(config: &Config) -> Result<Vec<DiscoveredDevice>> {
        if config.storage_backing_mode.eq_ignore_ascii_case("filesystem") {
            if config.filesystem_cache_paths.is_empty() {
                warn!("storage_backing_mode = \"filesystem\" with no filesystem_cache_paths: no Tier 2 storage");
                return Ok(Vec::new());
            }
            return Self::try_provision_filesystem_arenas(config);
        }
        Ok(Self::detect_devices(config))
    }
}

const GIB: f64 = 1024.0 * 1024.0 * 1024.0;

/// Reserves disk blocks for the first `len` bytes of `file` and extends it to `len`.
#[cfg(target_os = "linux")]
fn reserve_space(file: &fs::File, len: u64) -> std::io::Result<()> {
    use std::os::unix::io::AsRawFd;
    // fallocate(2), not posix_fallocate(3): glibc emulates the latter by writing zeros on
    // filesystems without support, which takes minutes for a 100 GiB arena.
    let ret = unsafe { libc::fallocate(file.as_raw_fd(), 0, 0, len as libc::off_t) };
    if ret == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

#[cfg(not(target_os = "linux"))]
fn reserve_space(file: &fs::File, len: u64) -> std::io::Result<()> {
    fs2::FileExt::allocate(file, len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_base_disk_name() {
        assert_eq!(DeviceDetector::extract_base_disk_name("vda1"), "vda");
        assert_eq!(DeviceDetector::extract_base_disk_name("vda14"), "vda");
        assert_eq!(DeviceDetector::extract_base_disk_name("vda"), "vda");
        assert_eq!(DeviceDetector::extract_base_disk_name("vdc1"), "vdc");
        assert_eq!(DeviceDetector::extract_base_disk_name("vdc"), "vdc");
        assert_eq!(DeviceDetector::extract_base_disk_name("nvme0n1p1"), "nvme0n1");
        assert_eq!(DeviceDetector::extract_base_disk_name("nvme0n1p2"), "nvme0n1");
        assert_eq!(DeviceDetector::extract_base_disk_name("nvme0n1"), "nvme0n1");
        assert_eq!(DeviceDetector::extract_base_disk_name("sda2"), "sda");
    }

    #[test]
    fn test_numa_node_extraction() {
        let temp_dir = std::env::temp_dir().join(format!(
            "hypercache_numa_test_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        let dev_dir = temp_dir.join("device");
        std::fs::create_dir_all(&dev_dir).unwrap();

        let numa_file = dev_dir.join("numa_node");
        std::fs::write(&numa_file, "1\n").unwrap();

        let extracted = DeviceDetector::extract_numa_node(&temp_dir);
        assert_eq!(extracted, Some(1));

        // Test invalid/negative numa node (-1 means no affinity)
        std::fs::write(&numa_file, "-1\n").unwrap();
        let extracted_neg = DeviceDetector::extract_numa_node(&temp_dir);
        assert_eq!(extracted_neg, None);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_filesystem_arena_discovery() {
        let temp_dir = std::env::temp_dir().join(format!(
            "hypercache_fs_arena_test_{}",
            chrono::Utc::now().timestamp_nanos_opt().unwrap_or(0)
        ));
        let mount1 = temp_dir.join("nvme_mount_0");
        let mount2 = temp_dir.join("nvme_mount_1");

        let config = Config {
            storage_backing_mode: "filesystem".to_string(),
            filesystem_cache_paths: vec![
                mount1.to_string_lossy().to_string(),
                mount2.to_string_lossy().to_string(),
            ],
            filesystem_arena_size_bytes: 64 * 1024 * 1024,
            ..Config::default()
        };

        let devices = DeviceDetector::detect_devices(&config);
        assert_eq!(devices.len(), 2);
        assert!(devices[0].path.ends_with("hypercache_arena.dat"));
        assert!(devices[1].path.ends_with("hypercache_arena.dat"));
        assert_eq!(devices[0].size_bytes, 64 * 1024 * 1024);
        assert_eq!(devices[0].model.as_deref(), Some("Filesystem-DirectIO-Arena"));

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_arena_preallocation_and_capacity_errors() {
        let temp_dir = std::env::temp_dir().join(format!("hypercache_fs_prealloc_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        let size = 64 * 1024 * 1024;
        let config = |dir: &str, size: u64, preallocate: bool| Config {
            storage_backing_mode: "filesystem".to_string(),
            filesystem_cache_paths: vec![temp_dir.join(dir).to_string_lossy().to_string()],
            filesystem_arena_size_bytes: size,
            filesystem_arena_preallocate: preallocate,
            ..Config::default()
        };

        // Block counts are only asserted on Linux: APFS reserves preallocated space without
        // reporting it in st_blocks
        let blocks_reported = cfg!(target_os = "linux");

        // Default: the space is reserved on disk at provisioning time
        let reserved = DeviceDetector::detect_devices_checked(&config("reserved", size, true)).unwrap();
        assert_eq!(reserved.len(), 1);
        assert_eq!(reserved[0].size_bytes, size);
        if blocks_reported {
            assert!(DeviceDetector::allocated_bytes(Path::new(&reserved[0].path)) >= size);
        }

        // Opt-out keeps a sparse arena
        let sparse = DeviceDetector::detect_devices_checked(&config("sparse", size, false)).unwrap();
        assert!(DeviceDetector::allocated_bytes(Path::new(&sparse[0].path)) < size);
        // Re-provisioning with preallocation fills the existing arena's holes
        let filled = DeviceDetector::detect_devices_checked(&config("sparse", size, true)).unwrap();
        assert_eq!(filled[0].size_bytes, size);
        if blocks_reported {
            assert!(DeviceDetector::allocated_bytes(Path::new(&filled[0].path)) >= size);
        }

        // An arena larger than the filesystem's free space fails up front, names the numbers,
        // and leaves no partial file behind
        let free = fs2::available_space(&temp_dir).unwrap();
        let too_big = config("too_big", free.saturating_mul(2), true);
        let err = DeviceDetector::detect_devices_checked(&too_big)
            .unwrap_err()
            .to_string();
        assert!(err.contains("are free"), "{}", err);
        assert!(!temp_dir.join("too_big").join("hypercache_arena.dat").exists());
        assert!(DeviceDetector::detect_devices(&too_big).is_empty());
        // ... and `check` reports it as a failure without creating anything
        let report_cfg = temp_dir.join("too_big.toml");
        fs::write(
            &report_cfg,
            format!(
                "storage_backing_mode = \"filesystem\"\nfilesystem_cache_paths = [\"{}\"]\nfilesystem_arena_size_bytes = {}\nmetadata_persistence_path = \"{}/m.json\"\n",
                temp_dir.join("check_big").display(),
                free.saturating_mul(2),
                temp_dir.display()
            ),
        )
        .unwrap();
        let report = crate::check::evaluate_host(Some(report_cfg), None, false).unwrap();
        assert!(!report.ready);
        assert!(report
            .checks
            .iter()
            .any(|c| c.name.starts_with("Arena") && c.status == crate::check::CheckStatus::Fail));
        assert!(!temp_dir.join("check_big").exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_filesystem_mode_never_falls_back_to_raw_disks() {
        use std::os::unix::fs::{MetadataExt, PermissionsExt};
        let temp_dir = std::env::temp_dir().join(format!("hypercache_fs_nofallback_{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&temp_dir);
        fs::create_dir_all(&temp_dir).unwrap();
        // A regular file where the arena directory should be: provisioning fails
        let blocker = temp_dir.join("not_a_dir");
        fs::write(&blocker, b"x").unwrap();

        // Prefixes that match every disk name: a raw-scan fallback would pick them up
        let failing = Config {
            storage_backing_mode: "filesystem".to_string(),
            filesystem_cache_paths: vec![blocker.join("cache").to_string_lossy().to_string()],
            device_prefixes: vec!["nvme".into(), "vd".into(), "xvd".into(), "sd".into()],
            ..Config::default()
        };
        assert!(DeviceDetector::detect_devices(&failing).is_empty());
        let empty = Config {
            filesystem_cache_paths: vec![],
            ..failing.clone()
        };
        assert!(DeviceDetector::detect_devices(&empty).is_empty());

        // A provisioned arena is shared with client_gid
        let my_gid = fs::metadata(&blocker).unwrap().gid();
        let ok = Config {
            filesystem_cache_paths: vec![temp_dir.join("arena").to_string_lossy().to_string()],
            filesystem_arena_size_bytes: 64 * 1024 * 1024,
            client_gid: Some(my_gid),
            ..failing
        };
        let devices = DeviceDetector::detect_devices(&ok);
        assert_eq!(devices.len(), 1);
        let meta = fs::metadata(&devices[0].path).unwrap();
        assert_eq!(meta.gid(), my_gid);
        assert_eq!(meta.permissions().mode() & 0o777, 0o660);

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
