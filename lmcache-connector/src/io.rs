//! Chunk reads and writes.
//!
//! O_DIRECT is used whenever the buffer, offset and length allow it, straight to and from the
//! caller's buffer. Writes use it on raw block devices only. Whatever O_DIRECT can't take, such as
//! a partial last block, goes through the page cache, so a write never fails because of it.
//! Callers must still commit, because the daemon's sync is what makes the data durable.

use crate::daemon::Chunk;
use std::alloc::{alloc_zeroed, dealloc, Layout};
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{self, Write};
#[cfg(target_os = "linux")]
use std::os::unix::fs::OpenOptionsExt;
use std::os::unix::fs::{FileExt, FileTypeExt};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering::Relaxed};
use std::sync::{Arc, RwLock};

const DEFAULT_BLOCK_SIZE: usize = 4096;
/// Alignment of the bounce buffer. A device with larger blocks simply skips O_DIRECT.
const BOUNCE_ALIGN: usize = 4096;

/// A chunk's data did not match its recorded CRC32.
#[derive(Debug)]
pub struct ChecksumMismatch {
    pub expected: u32,
    pub actual: u32,
}

#[derive(Debug)]
pub enum ReadError {
    Io(io::Error),
    Checksum(ChecksumMismatch),
}

impl From<io::Error> for ReadError {
    fn from(error: io::Error) -> Self {
        ReadError::Io(error)
    }
}

#[derive(Default)]
pub struct IoStats {
    /// Bytes moved by O_DIRECT straight to or from the caller's buffer
    pub direct_read_bytes: AtomicU64,
    pub direct_write_bytes: AtomicU64,
    /// Bytes moved by O_DIRECT through an aligned copy
    pub bounced_read_bytes: AtomicU64,
    pub bounced_write_bytes: AtomicU64,
    /// Bytes that went through the page cache
    pub buffered_read_bytes: AtomicU64,
    pub buffered_write_bytes: AtomicU64,
}

/// A page-aligned buffer owned by one worker, grown on demand.
pub struct BounceBuffer {
    ptr: *mut u8,
    capacity: usize,
}

// The pointer is owned, and only the worker holding the buffer uses it
unsafe impl Send for BounceBuffer {}

impl Default for BounceBuffer {
    fn default() -> Self {
        Self::new()
    }
}

impl BounceBuffer {
    pub fn new() -> Self {
        BounceBuffer {
            ptr: std::ptr::null_mut(),
            capacity: 0,
        }
    }

    fn slice(&mut self, length: usize) -> io::Result<&mut [u8]> {
        if length > self.capacity {
            self.release();
            let layout = Layout::from_size_align(length, BOUNCE_ALIGN).map_err(io::Error::other)?;
            let ptr = unsafe { alloc_zeroed(layout) };
            if ptr.is_null() {
                return Err(io::Error::from(io::ErrorKind::OutOfMemory));
            }
            self.ptr = ptr;
            self.capacity = length;
        }
        Ok(unsafe { std::slice::from_raw_parts_mut(self.ptr, length) })
    }

    fn release(&mut self) {
        if !self.ptr.is_null() {
            let layout = Layout::from_size_align(self.capacity, BOUNCE_ALIGN).expect("valid when allocated");
            unsafe { dealloc(self.ptr, layout) };
            self.ptr = std::ptr::null_mut();
            self.capacity = 0;
        }
    }
}

impl Drop for BounceBuffer {
    fn drop(&mut self) {
        self.release();
    }
}

struct ReadFds {
    buffered: File,
    direct: Option<File>,
    block_size: usize,
}

struct WriteFds {
    buffered: File,
    direct: Option<File>,
    block_size: usize,
}

/// Open device handles, shared by all workers. Failed opens aren't remembered.
struct Table<T>(RwLock<HashMap<String, Arc<T>>>);

impl<T> Table<T> {
    fn new() -> Self {
        Table(RwLock::new(HashMap::new()))
    }

    fn get(&self, path: &str, open: impl FnOnce() -> io::Result<T>) -> io::Result<Arc<T>> {
        if let Some(entry) = self.0.read().expect("table lock").get(path) {
            return Ok(entry.clone());
        }
        let opened = Arc::new(open()?);
        Ok(self
            .0
            .write()
            .expect("table lock")
            .entry(path.to_string())
            .or_insert(opened)
            .clone())
    }
}

pub struct ChunkIo {
    readers: Table<ReadFds>,
    writers: Table<WriteFds>,
    pub stats: IoStats,
}

impl Default for ChunkIo {
    fn default() -> Self {
        Self::new()
    }
}

impl ChunkIo {
    pub fn new() -> Self {
        ChunkIo {
            readers: Table::new(),
            writers: Table::new(),
            stats: IoStats::default(),
        }
    }

    /// Reads one whole chunk into `target` and verifies its CRC32.
    pub fn read_chunk(&self, chunk: &Chunk, target: &mut [u8], bounce: &mut BounceBuffer) -> Result<(), ReadError> {
        if target.len() as u64 != chunk.length {
            return Err(io::Error::other("chunk length does not match its buffer").into());
        }
        if chunk.is_shm() {
            File::open(&chunk.device)?.read_exact_at(target, chunk.offset)?;
            self.stats.buffered_read_bytes.fetch_add(target.len() as u64, Relaxed);
        } else {
            let fds = self.readers.get(&chunk.device, || open_for_read(&chunk.device))?;
            let mut filled = 0;
            if let Some(direct) = &fds.direct {
                filled = self.read_direct(direct, fds.block_size, chunk.offset, target, bounce);
            }
            if filled < target.len() {
                fds.buffered
                    .read_exact_at(&mut target[filled..], chunk.offset + filled as u64)?;
                self.stats
                    .buffered_read_bytes
                    .fetch_add((target.len() - filled) as u64, Relaxed);
            }
        }
        verify(chunk, target)
    }

    /// Returns how many leading bytes of `target` were filled. Any failure leaves the rest for
    /// the page cache path, which reports a real error if the data is unreadable.
    fn read_direct(
        &self,
        file: &File,
        block_size: usize,
        offset: u64,
        target: &mut [u8],
        bounce: &mut BounceBuffer,
    ) -> usize {
        if block_size > BOUNCE_ALIGN || !offset.is_multiple_of(block_size as u64) {
            return 0;
        }
        let length = target.len();
        let whole = length - length % block_size;
        let mut done = 0;

        if (target.as_ptr() as usize).is_multiple_of(block_size) {
            while done < whole {
                match file.read_at(&mut target[done..whole], offset + done as u64) {
                    Ok(count) if count > 0 => {
                        done += count - count % block_size;
                        if count % block_size != 0 {
                            break;
                        }
                    }
                    Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                    _ => break,
                }
            }
            self.stats.direct_read_bytes.fetch_add(done as u64, Relaxed);
            if done < whole {
                return done;
            }
        } else if whole > 0 {
            let Ok(aligned) = bounce.slice(whole) else { return 0 };
            match read_aligned(file, aligned, offset, block_size) {
                count if count == whole => target[..whole].copy_from_slice(aligned),
                count => {
                    target[..count].copy_from_slice(&aligned[..count]);
                    self.stats.bounced_read_bytes.fetch_add(count as u64, Relaxed);
                    return count;
                }
            }
            done = whole;
            self.stats.bounced_read_bytes.fetch_add(whole as u64, Relaxed);
        }

        // A partial last block. Allocations are rounded up to whole blocks, so reading all of it stays in bounds.
        if whole < length {
            let Ok(aligned) = bounce.slice(block_size) else {
                return done;
            };
            if read_aligned(file, aligned, offset + whole as u64, block_size) == block_size {
                target[whole..].copy_from_slice(&aligned[..length - whole]);
                done = length;
                self.stats
                    .bounced_read_bytes
                    .fetch_add((length - whole) as u64, Relaxed);
            }
        }
        done
    }

    /// Writes one whole chunk.
    pub fn write_chunk(&self, chunk: &Chunk, source: &[u8], bounce: &mut BounceBuffer) -> io::Result<()> {
        if source.len() as u64 != chunk.length {
            return Err(io::Error::other("chunk length does not match its buffer"));
        }
        if chunk.is_ram() {
            return write_ram_chunk(&chunk.device, source);
        }
        let fds = self.writers.get(&chunk.device, || open_for_write(&chunk.device))?;
        let mut written = 0;
        if let Some(direct) = &fds.direct {
            written = self.write_direct(direct, fds.block_size, chunk.offset, source, bounce)?;
        }
        if written < source.len() {
            // Retries short writes, for example when the disk is nearly full
            fds.buffered
                .write_all_at(&source[written..], chunk.offset + written as u64)?;
            self.stats
                .buffered_write_bytes
                .fetch_add((source.len() - written) as u64, Relaxed);
        }
        Ok(())
    }

    /// Writes as many whole blocks as O_DIRECT accepts and returns the bytes written. EINVAL means
    /// O_DIRECT refused the request, so the caller finishes through the page cache. Other errors,
    /// such as a failing drive, would fail there too and are reported.
    fn write_direct(
        &self,
        file: &File,
        block_size: usize,
        offset: u64,
        source: &[u8],
        bounce: &mut BounceBuffer,
    ) -> io::Result<usize> {
        let whole = source.len() - source.len() % block_size;
        if whole == 0 || block_size > BOUNCE_ALIGN || !offset.is_multiple_of(block_size as u64) {
            return Ok(0);
        }
        let aligned_source = (source.as_ptr() as usize).is_multiple_of(block_size);
        let data: &[u8] = if aligned_source {
            &source[..whole]
        } else {
            let copy = bounce.slice(whole)?;
            copy.copy_from_slice(&source[..whole]);
            copy
        };
        let mut written = 0;
        while written < whole {
            match file.write_at(&data[written..], offset + written as u64) {
                Ok(count) => {
                    written += count;
                    if count == 0 || count % block_size != 0 {
                        break;
                    }
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
                Err(e) if e.raw_os_error() == Some(libc::EINVAL) => break,
                Err(e) => return Err(e),
            }
        }
        let counter = if aligned_source {
            &self.stats.direct_write_bytes
        } else {
            &self.stats.bounced_write_bytes
        };
        counter.fetch_add(written as u64, Relaxed);
        Ok(written)
    }

    /// Pushes page-cache writes to disk. Runs at shutdown only: commits are what make data durable.
    pub fn sync(&self) {
        for fds in self.writers.0.read().expect("table lock").values() {
            let _ = fds.buffered.sync_all();
        }
    }
}

fn read_aligned(file: &File, buffer: &mut [u8], offset: u64, block_size: usize) -> usize {
    let mut done = 0;
    while done < buffer.len() {
        match file.read_at(&mut buffer[done..], offset + done as u64) {
            Ok(count) if count > 0 => {
                done += count;
                if count % block_size != 0 {
                    break;
                }
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            _ => break,
        }
    }
    done - done % block_size
}

fn verify(chunk: &Chunk, data: &[u8]) -> Result<(), ReadError> {
    let Some(expected) = chunk.checksum else {
        return Err(io::Error::other("chunk has no checksum").into());
    };
    let actual = crc32fast::hash(data);
    if actual != expected {
        return Err(ReadError::Checksum(ChecksumMismatch { expected, actual }));
    }
    Ok(())
}

fn write_ram_chunk(path: &str, data: &[u8]) -> io::Result<()> {
    // The daemon creates the directory. If it's missing, this process doesn't share /dev/shm with
    // the daemon, and a private copy would silently disappear.
    let parent = Path::new(path).parent().unwrap_or(Path::new("/"));
    if !parent.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "RAM chunk directory for {path} does not exist in this process; it must share /dev/shm with the daemon"
            ),
        ));
    }
    File::create(path)?.write_all(data)
}

fn open_for_read(path: &str) -> io::Result<ReadFds> {
    Ok(ReadFds {
        buffered: File::open(path)?,
        direct: open_direct(path, false),
        block_size: logical_block_size(path),
    })
}

fn open_for_write(path: &str) -> io::Result<WriteFds> {
    let buffered = OpenOptions::new().read(true).write(true).open(path)?;
    let is_block = buffered.metadata()?.file_type().is_block_device();
    let direct = if is_block { open_direct(path, true) } else { None };
    Ok(WriteFds {
        buffered,
        direct,
        block_size: logical_block_size(path),
    })
}

#[cfg(target_os = "linux")]
fn open_direct(path: &str, write: bool) -> Option<File> {
    OpenOptions::new()
        .read(true)
        .write(write)
        .custom_flags(libc::O_DIRECT)
        .open(path)
        .ok()
}

#[cfg(not(target_os = "linux"))]
fn open_direct(_path: &str, _write: bool) -> Option<File> {
    None
}

/// The O_DIRECT alignment for a device, from sysfs, or 4096 when it isn't known.
pub fn logical_block_size(path: &str) -> usize {
    let Ok(real) = std::fs::canonicalize(path) else {
        return DEFAULT_BLOCK_SIZE;
    };
    let Some(name) = real.file_name().and_then(|n| n.to_str()) else {
        return DEFAULT_BLOCK_SIZE;
    };
    let own = format!("/sys/class/block/{name}/queue/logical_block_size");
    // Partitions inherit the queue limits of their parent disk
    let parent = std::fs::canonicalize(format!("/sys/class/block/{name}"))
        .ok()
        .and_then(|p| p.parent().map(|parent| parent.join("queue/logical_block_size")));
    for candidate in [Some(PathBuf::from(own)), parent].into_iter().flatten() {
        if let Ok(text) = std::fs::read_to_string(&candidate) {
            if let Ok(value) = text.trim().parse::<usize>() {
                if value > 0 && value.is_power_of_two() {
                    return value.max(512);
                }
            }
        }
    }
    DEFAULT_BLOCK_SIZE
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_file(name: &str, size: usize) -> String {
        let path = std::env::temp_dir().join(format!("hc-io-{}-{name}", std::process::id()));
        std::fs::write(&path, vec![0u8; size]).unwrap();
        path.to_str().unwrap().to_string()
    }

    fn pattern(length: usize, seed: u8) -> Vec<u8> {
        (0..length)
            .map(|i| (i as u8).wrapping_mul(31).wrapping_add(seed))
            .collect()
    }

    fn chunk(device: &str, offset: u64, data: &[u8]) -> Chunk {
        Chunk {
            tier: "nvme".into(),
            device: device.into(),
            offset,
            length: data.len() as u64,
            checksum: Some(crc32fast::hash(data)),
        }
    }

    #[test]
    fn crc_matches_zlib() {
        // zlib.crc32(b"123456789") == 0xCBF43926, the value the daemon and Python client expect
        assert_eq!(crc32fast::hash(b"123456789"), 0xCBF4_3926);
    }

    #[test]
    fn round_trips_at_any_length_and_alignment() {
        let path = temp_file("roundtrip", 1 << 20);
        let io = ChunkIo::new();
        let mut bounce = BounceBuffer::new();
        for (length, target_shift) in [(4096, 0), (8192, 1), (4097, 0), (123, 0), (65536 + 5, 3)] {
            let data = pattern(length, length as u8);
            let c = chunk(&path, 4096, &data);
            io.write_chunk(&c, &data, &mut bounce).unwrap();
            let mut storage = vec![0u8; length + 8192];
            let start = (4096 - storage.as_ptr() as usize % 4096) % 4096 + target_shift;
            let target = &mut storage[start..start + length];
            io.read_chunk(&c, target, &mut bounce).unwrap();
            assert_eq!(target, &data[..], "length {length}");
        }
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn damaged_data_is_a_checksum_error() {
        let path = temp_file("damaged", 1 << 16);
        let io = ChunkIo::new();
        let mut bounce = BounceBuffer::new();
        let data = pattern(8192, 7);
        let c = chunk(&path, 0, &data);
        io.write_chunk(&c, &data, &mut bounce).unwrap();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .write_all_at(&[0xAA], 100)
            .unwrap();
        let mut target = vec![0u8; 8192];
        assert!(matches!(
            io.read_chunk(&c, &mut target, &mut bounce),
            Err(ReadError::Checksum(_))
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn a_chunk_without_a_checksum_never_verifies() {
        let path = temp_file("nosum", 8192);
        let io = ChunkIo::new();
        let mut c = chunk(&path, 0, &[0u8; 4096]);
        c.checksum = None;
        let mut target = vec![0u8; 4096];
        assert!(io.read_chunk(&c, &mut target, &mut BounceBuffer::new()).is_err());
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn reading_past_the_end_is_an_error() {
        let path = temp_file("short", 4096);
        let io = ChunkIo::new();
        let data = pattern(8192, 1);
        let c = chunk(&path, 0, &data);
        let mut target = vec![0u8; 8192];
        assert!(matches!(
            io.read_chunk(&c, &mut target, &mut BounceBuffer::new()),
            Err(ReadError::Io(_))
        ));
        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn wrong_buffer_length_is_refused() {
        let io = ChunkIo::new();
        let c = chunk("/nonexistent", 0, &[0u8; 4096]);
        assert!(io.write_chunk(&c, &[0u8; 100], &mut BounceBuffer::new()).is_err());
        let mut target = [0u8; 100];
        assert!(io.read_chunk(&c, &mut target, &mut BounceBuffer::new()).is_err());
    }

    #[test]
    fn ram_chunk_needs_the_shared_directory() {
        let io = ChunkIo::new();
        let mut c = chunk("/definitely/not/shared/x.shm", 0, &[1u8; 16]);
        c.tier = "ram".into();
        let error = io.write_chunk(&c, &[1u8; 16], &mut BounceBuffer::new()).unwrap_err();
        assert_eq!(error.kind(), io::ErrorKind::NotFound);
        assert!(!Path::new("/definitely").exists());
    }

    #[test]
    fn ram_chunk_round_trips() {
        let path = temp_file("ram.shm", 0);
        let io = ChunkIo::new();
        let data = pattern(5000, 9);
        let mut c = chunk(&path, 0, &data);
        c.tier = "ram".into();
        let mut bounce = BounceBuffer::new();
        io.write_chunk(&c, &data, &mut bounce).unwrap();
        let mut target = vec![0u8; 5000];
        io.read_chunk(&c, &mut target, &mut bounce).unwrap();
        assert_eq!(target, data);
        std::fs::remove_file(path).unwrap();
    }
}
