"""
HyperCache Zero-Copy Data Plane & GDS Reader Module.

Responsible for executing direct hardware DMA reads from raw NVMe block devices
(e.g., /dev/nvme0n1) into GPU VRAM (via kvikio.CuFile) or host memory.
"""

from __future__ import annotations

import logging
import mmap
import os
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger("hypercache.datapath")

# Attempt importing kvikio for GPUDirect Storage (GDS)
_KVIKIO_AVAILABLE = False
_CUDA_AVAILABLE = False

try:
    import kvikio
    import kvikio.cufile
    _KVIKIO_AVAILABLE = True
except (ImportError, Exception) as e:
    logger.debug(f"kvikio is not available: {e}. Falling back to DirectBlockReader.")

try:
    import torch
    if torch.cuda.is_available():
        _CUDA_AVAILABLE = True
except (ImportError, Exception):
    pass

# Opt-in end-to-end integrity check: when set, full-chunk reads are verified against the
# per-chunk CRC32 recorded by the daemon (adds one zlib.crc32 pass over the data).
VERIFY_CHECKSUMS_ENV = "HYPERCACHE_VERIFY_CHECKSUMS"
DEFAULT_BLOCK_SIZE = 4096


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


class ChecksumMismatchError(IOError):
    """Raised when chunk data does not match the CRC32 recorded at ingest time."""


def logical_block_size(device_path: str) -> int:
    """Returns the O_DIRECT alignment for a device (sysfs logical_block_size), or 4096."""
    try:
        name = os.path.basename(os.path.realpath(device_path))
        for candidate in (
            f"/sys/class/block/{name}/queue/logical_block_size",
            # Partitions inherit the queue limits of their parent disk
            os.path.join(os.path.dirname(os.path.realpath(f"/sys/class/block/{name}")), "queue", "logical_block_size"),
        ):
            if os.path.exists(candidate):
                with open(candidate) as f:
                    value = int(f.read().strip())
                if value > 0 and value & (value - 1) == 0:
                    return max(value, 512)
    except (OSError, ValueError):
        pass
    return DEFAULT_BLOCK_SIZE


@dataclass(frozen=True)
class ChunkLocation:
    """Represents the physical location of a striped chunk on a block device or shared memory."""
    device: str
    offset: int
    length: int
    tier: str = "nvme"
    # CRC32 (zlib.crc32) of the chunk bytes, when the daemon recorded one
    checksum: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> ChunkLocation:
        checksum = data.get("checksum")
        return cls(
            device=str(data["device"]),
            offset=int(data["offset"]),
            length=int(data["length"]),
            tier=str(data.get("tier", "nvme")).lower(),
            checksum=int(checksum) if checksum is not None else None,
        )

    @property
    def is_shm(self) -> bool:
        return self.tier == "ram" or self.device.endswith(".shm")

    def to_dict(self) -> dict:
        out = {
            "device": self.device,
            "offset": self.offset,
            "length": self.length,
            "tier": self.tier,
        }
        if self.checksum is not None:
            out["checksum"] = self.checksum
        return out


class DataReader:
    """Base interface for reading striped chunks from physical block devices."""

    def read_range(
        self,
        chunk_layout: List[ChunkLocation],
        start_offset: int,
        length: int,
    ) -> bytes:
        """Reads a byte range [start_offset, start_offset + length) from the striped layout."""
        raise NotImplementedError

    def read_into_gpu(
        self,
        chunk_layout: List[ChunkLocation],
        start_offset: int,
        length: int,
        gpu_buffer: Any,
    ) -> int:
        """Reads directly into GPU VRAM via GPUDirect Storage zero-copy DMA."""
        raise NotImplementedError

    def close(self) -> None:
        """Close all open hardware device handles."""
        pass


class DirectBlockReader(DataReader):
    """
    Direct Block Reader using os.pread on Linux block devices / mock backing files.
    Serves as the high-speed CPU reader and fallback/testing harness.
    Uses multi-threaded I/O submission to saturate SSD controller queue depth.
    """

    def __init__(self, max_workers: Optional[int] = None, verify_checksums: Optional[bool] = None):
        self._fd_cache: Dict[str, Tuple[Optional[int], int]] = {}
        self._block_sizes: Dict[str, int] = {}
        self._lock = threading.Lock()
        self._closed = False
        self._pid = os.getpid()
        self._local = threading.local()
        self.verify_checksums = _env_flag(VERIFY_CHECKSUMS_ENV) if verify_checksums is None else verify_checksums
        self._stats = {"direct_reads": 0, "direct_fallbacks": 0, "buffered_reads": 0, "shm_reads": 0}
        self._stats_lock = threading.Lock()
        if max_workers is None:
            max_workers = min(64, max(16, (os.cpu_count() or 4) * 4))
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _count(self, key: str) -> None:
        with self._stats_lock:
            self._stats[key] += 1

    def get_stats(self) -> Dict[str, int]:
        """Read path counters. A growing `direct_fallbacks` means O_DIRECT is not engaging."""
        with self._stats_lock:
            return dict(self._stats)

    def _bounce_buffer(self, size: int) -> mmap.mmap:
        # Anonymous mmaps are page-aligned, which satisfies O_DIRECT buffer alignment.
        # One buffer per worker thread, grown on demand.
        buf = getattr(self._local, "buf", None)
        if buf is None or len(buf) < size:
            if buf is not None:
                buf.close()
            buf = mmap.mmap(-1, size)
            self._local.buf = buf
        return buf

    def _block_size(self, device_path: str) -> int:
        bs = self._block_sizes.get(device_path)
        if bs is None:
            bs = logical_block_size(device_path)
            self._block_sizes[device_path] = bs
        return bs

    def _read_direct(self, fd_direct: int, dev_path: str, target_slice: memoryview, phys_offset: int) -> bool:
        """O_DIRECT read via an aligned bounce buffer. Returns False if the caller must fall back."""
        length = len(target_slice)
        bs = self._block_size(dev_path)
        aligned_start = phys_offset - (phys_offset % bs)
        head = phys_offset - aligned_start
        aligned_len = -(-(head + length) // bs) * bs
        buf = self._bounce_buffer(aligned_len)
        try:
            n = os.preadv(fd_direct, [memoryview(buf)[:aligned_len]], aligned_start)
        except OSError as e:
            logger.debug(f"O_DIRECT read failed on {dev_path} at {aligned_start}: {e}")
            return False
        # A short read is fine as long as it covers the requested bytes (e.g. at end of a file)
        if n < head + length:
            return False
        target_slice[:] = memoryview(buf)[head : head + length]
        return True

    def _get_fds(self, device_path: str) -> Tuple[Optional[int], int]:
        if device_path in self._fd_cache:
            return self._fd_cache[device_path]
        with self._lock:
            if device_path not in self._fd_cache:
                try:
                    fd_buffered = os.open(device_path, os.O_RDONLY)
                except OSError as e:
                    logger.error(f"Failed to open device {device_path}: {e}")
                    raise

                fd_direct = None
                if hasattr(os, "O_DIRECT"):
                    try:
                        fd_direct = os.open(device_path, os.O_RDONLY | os.O_DIRECT)
                    except OSError:
                        pass

                self._fd_cache[device_path] = (fd_direct, fd_buffered)
            return self._fd_cache[device_path]

    def read_range_into(
        self,
        chunk_layout: List[ChunkLocation],
        start_offset: int,
        length: int,
        target_buffer: Any,
        target_offset: int = 0,
    ) -> int:
        """Reads directly into a pre-allocated bytearray or memoryview with parallel Queue Depth submission."""
        if length <= 0 or not chunk_layout:
            return 0

        current_virtual_offset = 0
        view = memoryview(target_buffer)
        tasks = []

        for chunk in chunk_layout:
            chunk_start = current_virtual_offset
            chunk_end = current_virtual_offset + chunk.length
            current_virtual_offset = chunk_end

            # Check if this chunk overlaps with the requested range [start_offset, start_offset + length)
            range_end = start_offset + length
            if chunk_end <= start_offset or chunk_start >= range_end:
                continue

            # Calculate slice within this chunk
            overlap_start = max(start_offset, chunk_start)
            overlap_end = min(range_end, chunk_end)
            overlap_len = overlap_end - overlap_start

            offset_in_chunk = overlap_start - chunk_start
            physical_device_offset = chunk.offset + offset_in_chunk

            dest_pos = target_offset + (overlap_start - start_offset)
            tasks.append((chunk, overlap_len, physical_device_offset, dest_pos))

        if not tasks:
            return 0

        def _read_task(t):
            chunk, overlap_len, phys_offset, dest_pos = t
            dev_path = chunk.device
            target_slice = view[dest_pos : dest_pos + overlap_len]

            if chunk.is_shm:
                # Tier 1 chunks are files in /dev/shm: a positioned read copies straight from
                # the shared pages into the destination buffer.
                try:
                    fd = os.open(dev_path, os.O_RDONLY)
                    try:
                        n = os.preadv(fd, [target_slice], phys_offset) if hasattr(os, "preadv") else None
                        if n is None:
                            data = os.pread(fd, overlap_len, phys_offset)
                            n = len(data)
                            target_slice[:n] = data
                    finally:
                        os.close(fd)
                except OSError as e:
                    logger.error(f"Failed reading SHM chunk {dev_path}: {e}")
                    raise
                if n != overlap_len:
                    raise IOError(f"Short read from SHM chunk {dev_path}: expected {overlap_len}, got {n}")
                self._count("shm_reads")
                return self._verify(chunk, overlap_len, target_slice)

            fd_direct, fd_buffered = self._get_fds(dev_path)

            if fd_direct is not None and hasattr(os, "preadv"):
                if self._read_direct(fd_direct, dev_path, target_slice, phys_offset):
                    self._count("direct_reads")
                    return self._verify(chunk, overlap_len, target_slice)
                self._count("direct_fallbacks")

            self._count("buffered_reads")
            if hasattr(os, "preadv"):
                try:
                    n = os.preadv(fd_buffered, [target_slice], phys_offset)
                    if n == overlap_len:
                        return self._verify(chunk, overlap_len, target_slice)
                except OSError:
                    pass

            chunk_data = os.pread(fd_buffered, overlap_len, phys_offset)
            if len(chunk_data) != overlap_len:
                raise IOError(
                    f"Short read at offset {phys_offset}: "
                    f"expected {overlap_len} bytes, got {len(chunk_data)}"
                )
            target_slice[:] = chunk_data
            return self._verify(chunk, overlap_len, target_slice)

        if len(tasks) == 1:
            return _read_task(tasks[0])

        # Parallel chunk I/O across SSD queue depth
        results = list(self._executor.map(_read_task, tasks))
        return sum(results)

    def _verify(self, chunk: ChunkLocation, read_len: int, data: memoryview) -> int:
        # Only whole-chunk reads can be checked against the per-chunk CRC
        if self.verify_checksums and chunk.checksum is not None and read_len == chunk.length:
            actual = zlib.crc32(data) & 0xFFFFFFFF
            if actual != chunk.checksum:
                raise ChecksumMismatchError(
                    f"CRC32 mismatch for chunk at {chunk.device}+{chunk.offset}: "
                    f"expected {chunk.checksum:08x}, got {actual:08x}. The cached layout may be stale; "
                    f"locate the object again."
                )
        return read_len

    def read_range(
        self,
        chunk_layout: List[ChunkLocation],
        start_offset: int,
        length: int,
    ) -> Any:
        if length <= 0 or not chunk_layout:
            return b""

        # Allocate page-aligned zero-copy buffer for multi-megabyte reads
        if length >= 1024 * 1024:
            result = mmap.mmap(-1, length)
        else:
            result = bytearray(length)

        bytes_copied = self.read_range_into(chunk_layout, start_offset, length, result, 0)
        return result[:bytes_copied] if bytes_copied < length else result

    def read_into_gpu(
        self,
        chunk_layout: List[ChunkLocation],
        start_offset: int,
        length: int,
        gpu_buffer: Any,
    ) -> int:
        """Fallback when GDS is unavailable: host read followed by HtoD copy."""
        data = self.read_range(chunk_layout, start_offset, length)
        try:
            import torch
            if isinstance(gpu_buffer, torch.Tensor):
                # Copy host bytes into tensor
                host_tensor = torch.frombuffer(data, dtype=torch.uint8)
                gpu_buffer[:len(data)].copy_(host_tensor.to(gpu_buffer.device))
                return len(data)
        except Exception as e:
            logger.warning(f"Could not copy into GPU tensor directly: {e}")
        return len(data)

    def close(self) -> None:
        with self._lock:
            self._closed = True
            for fd_direct, fd_buffered in self._fd_cache.values():
                if fd_direct is not None:
                    try:
                        os.close(fd_direct)
                    except Exception:
                        pass
                try:
                    os.close(fd_buffered)
                except Exception:
                    pass
            self._fd_cache.clear()
        self._executor.shutdown(wait=False)


class GDSReader(DataReader):
    """
    GPUDirect Storage (GDS) Zero-Copy Reader.
    Uses kvikio.CuFile to execute direct DMA transfers from NVMe block devices to GPU VRAM.
    """

    def __init__(self):
        if not _KVIKIO_AVAILABLE:
            raise RuntimeError("kvikio is not installed. Cannot initialize GDSReader.")
        self._cufile_cache: Dict[str, Any] = {}
        self._fallback_reader = DirectBlockReader()

    def _get_cufile(self, device_path: str) -> Any:
        if device_path not in self._cufile_cache:
            # kvikio.CuFile opens the physical block device (e.g. /dev/nvme0n1)
            cf = kvikio.CuFile(device_path, flags="r")
            self._cufile_cache[device_path] = cf
        return self._cufile_cache[device_path]

    def read_range(
        self,
        chunk_layout: List[ChunkLocation],
        start_offset: int,
        length: int,
    ) -> bytes:
        # For CPU bytes requests, fallback to DirectBlockReader
        return self._fallback_reader.read_range(chunk_layout, start_offset, length)

    def read_into_gpu(
        self,
        chunk_layout: List[ChunkLocation],
        start_offset: int,
        length: int,
        gpu_buffer: Any,
    ) -> int:
        """
        Executes GPUDirect reads directly into the GPU memory buffer (CuPy or PyTorch tensor).
        Bypasses the host CPU entirely.
        """
        if length <= 0 or not chunk_layout:
            return 0

        current_virtual_offset = 0
        bytes_transferred = 0
        range_end = start_offset + length

        for chunk in chunk_layout:
            chunk_start = current_virtual_offset
            chunk_end = current_virtual_offset + chunk.length
            current_virtual_offset = chunk_end

            if chunk_end <= start_offset or chunk_start >= range_end:
                continue

            overlap_start = max(start_offset, chunk_start)
            overlap_end = min(range_end, chunk_end)
            overlap_len = overlap_end - overlap_start

            offset_in_chunk = overlap_start - chunk_start
            physical_device_offset = chunk.offset + offset_in_chunk
            dest_offset = overlap_start - start_offset

            cufile = self._get_cufile(chunk.device)

            # Extract GPU buffer pointer/slice
            # kvikio accepts GPU buffers (CuPy array, PyTorch CUDA tensor, or ctypes pointer)
            target_slice = gpu_buffer[dest_offset : dest_offset + overlap_len]
            read_bytes = cufile.pread(target_slice, size=overlap_len, file_offset=physical_device_offset)

            if read_bytes != overlap_len:
                raise IOError(
                    f"GDS pread mismatch on {chunk.device}: expected {overlap_len} bytes, read {read_bytes}"
                )

            bytes_transferred += read_bytes
            if bytes_transferred >= length:
                break

        return bytes_transferred

    def close(self) -> None:
        for cf in self._cufile_cache.values():
            try:
                cf.close()
            except Exception:
                pass
        self._cufile_cache.clear()
        self._fallback_reader.close()


_SHARED_READERS: Dict[bool, DataReader] = {}
_SHARED_READER_LOCK = threading.Lock()
_SHARED_READER_PID = os.getpid()


def get_data_reader(prefer_gds: bool = True) -> DataReader:
    """
    Factory to return the optimal shared DataReader based on hardware environment.
    Reuses open block-device file descriptors and DMA thread pools across requests.
    """
    global _SHARED_READER_PID
    with _SHARED_READER_LOCK:
        if _SHARED_READER_PID != os.getpid():
            # After fork() the parent's thread pools do not exist in the child and its file
            # descriptors / CUDA handles must not be shared: start fresh in this process.
            _SHARED_READERS.clear()
            _SHARED_READER_PID = os.getpid()
        existing = _SHARED_READERS.get(prefer_gds)
        if existing is not None and not getattr(existing, "_closed", False):
            return existing

        if prefer_gds and _KVIKIO_AVAILABLE and _CUDA_AVAILABLE:
            try:
                reader: DataReader = GDSReader()
                _SHARED_READERS[prefer_gds] = reader
                return reader
            except Exception as e:
                logger.warning(f"Failed to initialize GDSReader: {e}. Falling back to DirectBlockReader.")
        reader = DirectBlockReader()
        _SHARED_READERS[prefer_gds] = reader
        return reader


class DirectBlockWriter:
    """
    Direct Block Writer for write-back and write-through caching.
    Writes byte slices to striped physical NVMe block devices or POSIX shared memory chunks.
    """

    def __init__(self, max_workers: Optional[int] = None):
        self._fd_cache: Dict[str, int] = {}
        if max_workers is None:
            max_workers = min(64, max(16, (os.cpu_count() or 4) * 4))
        self._max_workers = max_workers
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def _get_write_fd(self, device_path: str) -> int:
        if device_path not in self._fd_cache:
            try:
                fd = os.open(device_path, os.O_RDWR)
                self._fd_cache[device_path] = fd
            except OSError as e:
                logger.error(f"Failed to open device {device_path} for write: {e}")
                raise
        return self._fd_cache[device_path]

    def _write_ram_chunk(self, target_path: str, chunk_data: bytes) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(target_path)), exist_ok=True)
        with open(target_path, "wb") as f:
            f.write(chunk_data)

    def write_chunks(
        self,
        chunk_layout: List[ChunkLocation],
        data: Union[bytes, bytearray, memoryview],
    ) -> int:
        """Writes data slices directly to physical block devices / shm files across the striped layout."""
        if not chunk_layout or not data:
            return 0

        view = memoryview(data)
        total_len = len(view)
        current_virtual_offset = 0
        tasks = []

        for chunk in chunk_layout:
            chunk_start = current_virtual_offset
            chunk_end = current_virtual_offset + chunk.length
            current_virtual_offset = chunk_end

            if chunk_start >= total_len:
                break

            slice_end = min(chunk_end, total_len)
            slice_bytes = bytes(view[chunk_start:slice_end])

            if chunk.tier == "ram":
                tasks.append(self._executor.submit(self._write_ram_chunk, chunk.device, slice_bytes))
            else:
                fd = self._get_write_fd(chunk.device)
                tasks.append(
                    self._executor.submit(os.pwrite, fd, slice_bytes, chunk.offset)
                )

        # Wait for all chunks to write
        for t in tasks:
            t.result()

        return total_len

    def close(self) -> None:
        for fd in self._fd_cache.values():
            try:
                os.fsync(fd)
                os.close(fd)
            except Exception:
                pass
        self._fd_cache.clear()
        self._executor.shutdown(wait=False)

