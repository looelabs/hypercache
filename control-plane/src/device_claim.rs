//! Ownership of raw drives.
//!
//! HyperCache stores data from `device_start_offset_bytes` (4 KiB by default) on and never writes
//! below it, so the first 4 KiB of a drive it claims hold an ownership marker. Auto discovery
//! uses only drives that carry the marker, that read as blank, or that the saved manifest already
//! lists (drives claimed before markers existed). A drive with any other data is refused with a
//! warning until the user erases it (`blkdiscard`, `hypercache prepare-drive`, or the
//! `zero_devices_with_data` config key).

use std::fs::{File, OpenOptions};
use std::io;
use std::os::unix::fs::{FileExt, FileTypeExt, OpenOptionsExt};
use std::path::Path;

/// Size of the marker block at the start of a claimed drive.
pub const MARKER_LEN: usize = 4096;
const MARKER_MAGIC: &[u8] = b"HYPERCACHE-DEVICE-V1\n";
/// How much of each end of a drive the blank check reads, and how much `zero_device` zeroes.
const EDGE_BYTES: u64 = 1024 * 1024;
/// Spread-out 4 KiB reads the blank check makes between the two ends.
const BLANK_SAMPLES: u64 = 64;

/// What the marker records about the drive when it was claimed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceMarker {
    pub serial: String,
    pub model: String,
    pub claimed_at: i64,
}

impl DeviceMarker {
    pub fn new(serial: Option<String>, model: Option<String>) -> Self {
        Self {
            serial: serial.unwrap_or_default(),
            model: model.unwrap_or_default(),
            claimed_at: chrono::Utc::now().timestamp(),
        }
    }

    /// The 4 KiB block: magic, `key=value` lines, zero padding, and a CRC32 of the rest at the end.
    pub fn encode(&self) -> Vec<u8> {
        let clean = |s: &str| s.replace(['\n', '\r', '\0'], " ");
        let body = format!(
            "serial={}\nmodel={}\nclaimed_at={}\n",
            clean(&self.serial),
            clean(&self.model),
            self.claimed_at
        );
        let mut block = vec![0u8; MARKER_LEN];
        let text_len = (MARKER_MAGIC.len() + body.len()).min(MARKER_LEN - 4);
        let mut text = MARKER_MAGIC.to_vec();
        text.extend_from_slice(body.as_bytes());
        block[..text_len].copy_from_slice(&text[..text_len]);
        let crc = crc32fast::hash(&block[..MARKER_LEN - 4]);
        block[MARKER_LEN - 4..].copy_from_slice(&crc.to_le_bytes());
        block
    }

    /// Some only for an intact marker: right magic and a matching CRC.
    pub fn decode(block: &[u8]) -> Option<Self> {
        if block.len() < MARKER_LEN || !block.starts_with(MARKER_MAGIC) {
            return None;
        }
        let crc = u32::from_le_bytes(block[MARKER_LEN - 4..MARKER_LEN].try_into().ok()?);
        if crc32fast::hash(&block[..MARKER_LEN - 4]) != crc {
            return None;
        }
        let text = &block[MARKER_MAGIC.len()..MARKER_LEN - 4];
        let end = text.iter().position(|&b| b == 0).unwrap_or(text.len());
        let text = std::str::from_utf8(&text[..end]).ok()?;
        let field = |key: &str| {
            text.lines()
                .find_map(|l| l.strip_prefix(key).and_then(|v| v.strip_prefix('=')))
                .unwrap_or("")
                .to_string()
        };
        Some(Self {
            serial: field("serial"),
            model: field("model"),
            claimed_at: field("claimed_at").parse().unwrap_or(0),
        })
    }
}

/// The marker on a drive, if it has an intact one.
pub fn read_marker(path: &str) -> io::Result<Option<DeviceMarker>> {
    let file = File::open(path)?;
    let mut block = vec![0u8; MARKER_LEN];
    match file.read_exact_at(&mut block, 0) {
        Ok(()) => Ok(DeviceMarker::decode(&block)),
        Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => Ok(None),
        Err(e) => Err(e),
    }
}

/// Writes the marker to the first 4 KiB and flushes it to the drive.
pub fn write_marker(path: &str, marker: &DeviceMarker) -> io::Result<()> {
    let file = OpenOptions::new().write(true).open(path)?;
    file.write_all_at(&marker.encode(), 0)?;
    file.sync_data()
}

/// Whether a drive reads as zeros: its first and last 1 MiB, and 64 4 KiB samples spread over the
/// rest. A new or discarded NVMe drive reads as zeros; a drive another program used does not.
pub fn looks_blank(path: &str, size: u64) -> io::Result<bool> {
    let file = File::open(path)?;
    let mut regions = vec![(0, EDGE_BYTES.min(size))];
    if size > EDGE_BYTES {
        regions.push((
            size - EDGE_BYTES.min(size - EDGE_BYTES),
            EDGE_BYTES.min(size - EDGE_BYTES),
        ));
    }
    if size > 2 * EDGE_BYTES {
        let span = size - 2 * EDGE_BYTES;
        for i in 0..BLANK_SAMPLES {
            let off = (EDGE_BYTES + span * i / BLANK_SAMPLES) & !4095;
            regions.push((off, 4096.min(size - off)));
        }
    }
    let mut buf = vec![0u8; EDGE_BYTES as usize];
    for (off, len) in regions {
        let chunk = &mut buf[..len as usize];
        file.read_exact_at(chunk, off)?;
        if chunk.iter().any(|&b| b != 0) {
            return Ok(false);
        }
    }
    Ok(true)
}

const BLKDISCARD: libc::c_ulong = 0x1277; // _IO(0x12, 119)
const BLKZEROOUT: libc::c_ulong = 0x127f; // _IO(0x12, 127)

/// Erases a drive before HyperCache claims it: discards every block (so the drive drops the old
/// data), then zeroes the first and last 1 MiB. A block device is opened with O_EXCL, which the
/// kernel refuses while anything has it mounted or held. A regular file (tests) has its ends
/// zeroed. This is not a secure erase.
pub fn zero_device(path: &str, size: u64) -> io::Result<()> {
    let is_block = std::fs::metadata(path)?.file_type().is_block_device();
    let mut opts = OpenOptions::new();
    opts.read(true).write(true);
    if is_block {
        opts.custom_flags(libc::O_EXCL);
    }
    let file = opts.open(path)?;
    let head = EDGE_BYTES.min(size);
    let tail_len = EDGE_BYTES.min(size - head);
    if is_block {
        use std::os::fd::AsRawFd;
        let fd = file.as_raw_fd();
        let range = |off: u64, len: u64| [off, len];
        let whole = range(0, size);
        // Discard can be unsupported (some virtual disks); the zeroed ends still make it blank
        // to the checks, and HyperCache tracks which blocks hold its data.
        // SAFETY: fd is open; the ioctl reads a [u64; 2] range that lives through the call.
        let rc = unsafe { libc::ioctl(fd, BLKDISCARD as _, whole.as_ptr()) };
        if rc != 0 {
            let e = io::Error::last_os_error();
            tracing::warn!(
                "Discarding {} failed ({}); zeroing its first and last 1 MiB only",
                path,
                e
            );
        }
        for r in [range(0, head), range(size - tail_len, tail_len)] {
            if r[1] == 0 {
                continue;
            }
            // SAFETY: as above
            if unsafe { libc::ioctl(fd, BLKZEROOUT as _, r.as_ptr()) } != 0 {
                write_zeros(&file, r[0], r[1])?;
            }
        }
    } else {
        write_zeros(&file, 0, head)?;
        write_zeros(&file, size - tail_len, tail_len)?;
    }
    file.sync_all()
}

fn write_zeros(file: &File, off: u64, len: u64) -> io::Result<()> {
    let zeros = vec![0u8; len as usize];
    file.write_all_at(&zeros, off)
}

/// Serial and model from sysfs (a partition reports its disk's).
pub fn device_identity(dev_path: &str) -> (Option<String>, Option<String>) {
    let name = dev_path.trim_start_matches("/dev/");
    let sys = Path::new("/sys/class/block").join(name);
    let device_dir = if sys.join("partition").exists() {
        sys.join("../device")
    } else {
        sys.join("device")
    };
    let read = |f: &str| {
        std::fs::read_to_string(device_dir.join(f))
            .ok()
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
    };
    (read("serial"), read("model"))
}

/// A stable `/dev/disk/by-id/...` name for a device, for messages (prefers `nvme-` names).
pub fn by_id_name(dev_path: &str) -> Option<String> {
    let target = std::fs::canonicalize(dev_path).ok()?;
    let mut names: Vec<String> = std::fs::read_dir("/dev/disk/by-id")
        .ok()?
        .flatten()
        .filter(|e| std::fs::canonicalize(e.path()).ok().as_ref() == Some(&target))
        .map(|e| e.path().to_string_lossy().to_string())
        .collect();
    names.sort_by_key(|n| (!n.contains("/nvme-"), n.contains("eui.") || n.contains("_1"), n.clone()));
    names.into_iter().next()
}

/// The devices `zero_devices_with_data` names, resolved to `/dev/...` paths. Only
/// `/dev/disk/by-id/` names are accepted: kernel names like `/dev/nvme0n1` can change between
/// boots, and the setting would then erase a different drive.
pub fn resolve_zero_targets(entries: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for entry in entries {
        if !entry.starts_with("/dev/disk/by-id/") {
            tracing::error!(
                "zero_devices_with_data: ignoring '{}': give drives by their /dev/disk/by-id/ name \
                 (kernel names like /dev/nvme0n1 can change between boots)",
                entry
            );
            continue;
        }
        match std::fs::canonicalize(entry) {
            Ok(p) => out.push(p.to_string_lossy().to_string()),
            Err(e) => tracing::warn!("zero_devices_with_data: '{}' does not resolve ({})", entry, e),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str, bytes: &[u8]) -> String {
        let path = std::env::temp_dir().join(format!("hc_claim_{}_{}", name, std::process::id()));
        std::fs::write(&path, bytes).unwrap();
        path.to_string_lossy().to_string()
    }

    #[test]
    fn marker_round_trips_and_rejects_damage() {
        let m = DeviceMarker::new(Some("S123\nX".into()), Some("Micron 7450".into()));
        let block = m.encode();
        assert_eq!(block.len(), MARKER_LEN);
        assert!(block.starts_with(b"HYPERCACHE"));
        let back = DeviceMarker::decode(&block).unwrap();
        assert_eq!(back.serial, "S123 X");
        assert_eq!(back.model, "Micron 7450");
        assert_eq!(back.claimed_at, m.claimed_at);

        let mut damaged = block.clone();
        damaged[100] ^= 1;
        assert_eq!(DeviceMarker::decode(&damaged), None);
        assert_eq!(DeviceMarker::decode(&vec![0u8; MARKER_LEN]), None);
    }

    #[test]
    fn marker_on_a_file() {
        let path = temp_file("marker", &vec![0u8; 64 * 1024]);
        assert_eq!(read_marker(&path).unwrap(), None);
        let m = DeviceMarker::new(None, None);
        write_marker(&path, &m).unwrap();
        assert_eq!(read_marker(&path).unwrap(), Some(m));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn blank_check_finds_data_at_the_ends_and_in_the_middle() {
        let size = 8 * 1024 * 1024;
        let path = temp_file("blank", &vec![0u8; size]);
        assert!(looks_blank(&path, size as u64).unwrap());
        for off in [0usize, size - 1, 1024 * 1024 + (size - 2 * 1024 * 1024) / 2] {
            let mut data = vec![0u8; size];
            data[off] = 7;
            std::fs::write(&path, &data).unwrap();
            assert!(!looks_blank(&path, size as u64).unwrap(), "missed data at {}", off);
        }
        std::fs::remove_file(&path).unwrap();

        let small = temp_file("blank_small", &[0u8; 8192]);
        assert!(looks_blank(&small, 8192).unwrap());
        std::fs::remove_file(&small).unwrap();
    }

    #[test]
    fn zeroing_makes_a_file_blank_at_the_ends() {
        let size = 4 * 1024 * 1024;
        let path = temp_file("zero", &vec![0xabu8; size]);
        zero_device(&path, size as u64).unwrap();
        let data = std::fs::read(&path).unwrap();
        assert!(data[..1024 * 1024].iter().all(|&b| b == 0));
        assert!(data[size - 1024 * 1024..].iter().all(|&b| b == 0));
        std::fs::remove_file(&path).unwrap();
    }

    #[test]
    fn zero_targets_must_be_by_id_names() {
        assert!(resolve_zero_targets(&["/dev/nvme0n1".to_string()]).is_empty());
        assert!(resolve_zero_targets(&["/dev/disk/by-id/hypercache-test-does-not-exist".to_string()]).is_empty());
    }
}
