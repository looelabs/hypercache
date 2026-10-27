"""
HyperCache fsspec (AbstractFileSystem) Implementation.

Seamlessly intercepts hypercache:// URIs for PyTorch, Hugging Face, and Safetensors.
"""

from __future__ import annotations

import io
import logging
import os
from typing import Any, Dict, List, Optional

import fsspec
from fsspec.spec import AbstractBufferedFile, AbstractFileSystem

from hypercache.client import HyperCacheClient, LocateResult, StageWriteResult
from hypercache.datapath import DataReader, DirectBlockWriter, get_data_reader

logger = logging.getLogger("hypercache.fsspec")


class HyperCacheFile(io.RawIOBase):
    """
    Seekable, readable Python stream backed by physical striped NVMe block devices.
    """

    def __init__(
        self,
        fs: HyperCacheFileSystem,
        uri: str,
        locate_result: LocateResult,
        reader: DataReader,
        mode: str = "rb",
    ):
        super().__init__()
        self.fs = fs
        self.uri = uri
        self.locate_result = locate_result
        self.reader = reader
        self.mode = mode
        self._pos = 0
        self._size = locate_result.object_size
        self._closed = False

    def readable(self) -> bool:
        return "r" in self.mode and not self._closed

    def seekable(self) -> bool:
        return not self._closed

    def writable(self) -> bool:
        return False

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed HyperCacheFile")

        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = self._size + offset
        else:
            raise ValueError(f"Invalid whence argument: {whence}")

        if new_pos < 0:
            raise ValueError(f"Negative seek position: {new_pos}")

        self._pos = new_pos
        return self._pos

    def read(self, size: int = -1) -> bytes:
        if self._closed:
            raise ValueError("I/O operation on closed HyperCacheFile")

        if self._pos >= self._size:
            return b""

        if size is None or size < 0:
            bytes_to_read = self._size - self._pos
        else:
            bytes_to_read = min(size, self._size - self._pos)

        if bytes_to_read <= 0:
            return b""

        data = self.reader.read_range(
            self.locate_result.chunk_layout,
            start_offset=self._pos,
            length=bytes_to_read,
        )

        self._pos += len(data)
        return data

    def readinto(self, b: bytearray | memoryview) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed HyperCacheFile")

        if self._pos >= self._size:
            return 0

        bytes_to_read = min(len(b), self._size - self._pos)
        if bytes_to_read <= 0:
            return 0

        if hasattr(self.reader, "read_range_into"):
            n = self.reader.read_range_into(
                self.locate_result.chunk_layout,
                start_offset=self._pos,
                length=bytes_to_read,
                target_buffer=b,
                target_offset=0,
            )
        else:
            data = self.read(bytes_to_read)
            n = len(data)
            b[:n] = data

        self._pos += n
        return n

    def read_into_gpu(self, gpu_tensor_or_buffer: Any, offset: int = 0, length: Optional[int] = None) -> int:
        """
        Zero-copy GPUDirect read directly into GPU memory via kvikio.CuFile.
        """
        if length is None:
            length = self._size - offset

        return self.reader.read_into_gpu(
            self.locate_result.chunk_layout,
            start_offset=offset,
            length=length,
            gpu_buffer=gpu_tensor_or_buffer,
        )

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            try:
                self.fs._client.close()
            except Exception:
                pass
            super().close()

    def __enter__(self) -> HyperCacheFile:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class HyperCacheWriteFile(io.RawIOBase):
    """
    Writable Python stream backed by physical striped NVMe block devices or RAM,
    supporting streaming 4 MiB chunk flushes and asynchronous write-back caching to S3/NFS.
    Bounds host RAM buffer usage to 4 MiB even for multi-gigabyte checkpoints.
    """

    DEFAULT_CHUNK_SIZE = 4 * 1024 * 1024

    def __init__(
        self,
        fs: HyperCacheFileSystem,
        uri: str,
        client: HyperCacheClient,
        tier: Optional[str] = None,
        is_leader: bool = True,
        auto_flush_s3: bool = True,
        write_through: bool = False,
        pinned: bool = False,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        mode: str = "wb",
    ):
        super().__init__()
        self.fs = fs
        self.uri = uri
        self.client = client
        self.tier = tier
        self.is_leader = is_leader
        self.auto_flush_s3 = auto_flush_s3
        self.write_through = write_through
        self.pinned = pinned
        self.chunk_size = chunk_size
        self.mode = mode
        self._buffer = bytearray()
        self._flushed_bytes = 0
        self._flushed_chunks: List[Any] = []
        self._pos = 0
        self._staged_initial = False
        self._closed = False
        self._writer = DirectBlockWriter()

    def writable(self) -> bool:
        return not self._closed

    def readable(self) -> bool:
        return False

    def seekable(self) -> bool:
        return not self._closed

    def tell(self) -> int:
        return self._pos

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed HyperCacheWriteFile")
        total_len = self._flushed_bytes + len(self._buffer)
        if whence == io.SEEK_SET:
            new_pos = offset
        elif whence == io.SEEK_CUR:
            new_pos = self._pos + offset
        elif whence == io.SEEK_END:
            new_pos = total_len + offset
        else:
            raise ValueError(f"Invalid whence: {whence}")
        if new_pos < 0:
            raise ValueError(f"Negative seek offset: {new_pos}")
        self._pos = new_pos
        return self._pos

    def _flush_chunk_bytes(self, payload: bytes) -> None:
        size = len(payload)
        if size == 0:
            return
        if not self._staged_initial:
            stage_res = self.client.stage_write(
                uri=self.uri,
                size_bytes=size,
                tier=self.tier,
                is_leader=self.is_leader,
                auto_flush_s3=self.auto_flush_s3,
                write_through=self.write_through,
                pinned=self.pinned,
            )
            self._staged_initial = True
        else:
            stage_res = self.client.append_write(
                uri=self.uri,
                size_bytes=size,
                tier=self.tier,
            )
        if not stage_res.is_staged:
            raise IOError(f"Failed to stage/append write for {self.uri}: {stage_res.error_message}")
        self._writer.write_chunks(stage_res.chunk_layout, payload)
        self._flushed_chunks.extend(stage_res.chunk_layout)
        self._flushed_bytes += size

    def write(self, b: bytes | bytearray | memoryview) -> int:
        if self._closed:
            raise ValueError("I/O operation on closed HyperCacheWriteFile")
        view = memoryview(b)
        n = len(view)
        if n == 0:
            return 0

        # If seeking backwards into already-flushed chunks (e.g. ZIP header patch),
        # patch the physical chunk directly in-place.
        curr_pos = self._pos
        idx = 0
        if curr_pos < self._flushed_bytes:
            patch_end = min(curr_pos + n, self._flushed_bytes)
            patch_len = patch_end - curr_pos
            patch_bytes = bytes(view[:patch_len])
            v_off = 0
            for chunk in self._flushed_chunks:
                c_start = v_off
                c_end = v_off + chunk.length
                v_off = c_end
                if c_end <= curr_pos or c_start >= patch_end:
                    continue
                ov_start = max(curr_pos, c_start)
                ov_end = min(patch_end, c_end)
                ov_len = ov_end - ov_start
                sub_slice = patch_bytes[ov_start - curr_pos : ov_end - curr_pos]
                if chunk.tier == "ram":
                    with open(chunk.device, "r+b") as rf:
                        rf.seek(chunk.offset + (ov_start - c_start))
                        rf.write(sub_slice)
                else:
                    fd = self._writer._get_write_fd(chunk.device)
                    os.pwrite(fd, sub_slice, chunk.offset + (ov_start - c_start))
            idx += patch_len
            curr_pos += patch_len

        # Write remaining bytes into the active tail buffer
        if idx < n:
            rem = view[idx:]
            buf_offset = curr_pos - self._flushed_bytes
            if buf_offset > len(self._buffer):
                self._buffer.extend(b"\x00" * (buf_offset - len(self._buffer)))
            end_offset = buf_offset + len(rem)
            if end_offset <= len(self._buffer):
                self._buffer[buf_offset:end_offset] = rem
            elif buf_offset == len(self._buffer):
                self._buffer.extend(rem)
            else:
                self._buffer[buf_offset:] = rem[: len(self._buffer) - buf_offset]
                self._buffer.extend(rem[len(self._buffer) - buf_offset :])
            curr_pos += len(rem)

        self._pos = curr_pos

        # Stream full 4 MiB chunks to NVMe/RAM immediately so buffer stays bounded
        while len(self._buffer) >= self.chunk_size and self._pos == self._flushed_bytes + len(self._buffer):
            chunk_payload = bytes(self._buffer[: self.chunk_size])
            del self._buffer[: self.chunk_size]
            self._flush_chunk_bytes(chunk_payload)

        return n

    def flush(self) -> None:
        if self._closed:
            return
        while len(self._buffer) >= self.chunk_size and self._pos == self._flushed_bytes + len(self._buffer):
            chunk_payload = bytes(self._buffer[: self.chunk_size])
            del self._buffer[: self.chunk_size]
            self._flush_chunk_bytes(chunk_payload)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            if len(self._buffer) > 0:
                tail_payload = bytes(self._buffer)
                self._buffer.clear()
                self._flush_chunk_bytes(tail_payload)

            if self._flushed_bytes > 0:
                committed = self.client.commit_write(self.uri, write_through=self.write_through)
                if not committed:
                    logger.warning(f"Failed to confirm commit write for {self.uri}")

            self._writer.close()
            super().close()

    def __enter__(self) -> HyperCacheWriteFile:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


class HyperCacheFileSystem(AbstractFileSystem):
    """
    fsspec Filesystem handler for hypercache:// protocol.
    """

    protocol = "hypercache"
    cachable = False

    def __init__(
        self,
        socket_path: Optional[str] = None,
        prefer_gds: bool = True,
        auto_ingest: bool = True,
        **storage_options: Any,
    ):
        super().__init__(**storage_options)
        self.socket_path = socket_path
        self.prefer_gds = prefer_gds
        self.auto_ingest = auto_ingest
        self._client = HyperCacheClient(socket_path=socket_path)
        self._reader = get_data_reader(prefer_gds=prefer_gds)

    def _format_uri(self, path: str) -> str:
        if path.startswith("hypercache://"):
            return path
        if path.startswith("s3://") or path.startswith("file://") or path.startswith("nfs://"):
            return path
        if path.startswith("/"):
            return f"file://{path}"
        clean_path = path.lstrip("/")
        return f"hypercache://{clean_path}"

    def _open(
        self,
        path: str,
        mode: str = "rb",
        block_size: Optional[int] = None,
        autocommit: bool = True,
        cache_options: Optional[dict] = None,
        tier: Optional[str] = None,
        is_leader: bool = True,
        auto_flush_s3: bool = True,
        write_through: bool = False,
        pinned: bool = False,
        **kwargs: Any,
    ) -> Any:
        uri = self._format_uri(path)

        if "w" in mode or "a" in mode:
            return HyperCacheWriteFile(
                fs=self,
                uri=uri,
                client=self._client,
                tier=tier,
                is_leader=is_leader,
                auto_flush_s3=auto_flush_s3,
                write_through=write_through,
                pinned=pinned,
                mode=mode,
            )

        locate_result = self._client.locate(uri, auto_ingest=self.auto_ingest, tier=tier, pinned=pinned)

        if not locate_result.is_hit:
            raise FileNotFoundError(
                f"Could not locate object {uri} in HyperCache: {locate_result.error_message or 'Cache Miss'}"
            )

        return HyperCacheFile(
            fs=self,
            uri=uri,
            locate_result=locate_result,
            reader=self._reader,
            mode=mode,
        )

    def info(self, path: str, **kwargs: Any) -> Dict[str, Any]:
        uri = self._format_uri(path)
        res = self._client.locate(uri, auto_ingest=self.auto_ingest)
        if not res.is_hit:
            raise FileNotFoundError(f"{uri} not found in HyperCache")

        return {
            "name": path,
            "size": res.object_size,
            "type": "file",
            "chunk_count": len(res.chunk_layout),
        }

    def ls(self, path: str, detail: bool = False, **kwargs: Any) -> List[Any]:
        info = self.info(path)
        if detail:
            return [info]
        return [path]

    def size(self, path: str) -> int:
        return self.info(path)["size"]

    def exists(self, path: str, **kwargs: Any) -> bool:
        try:
            uri = self._format_uri(path)
            res = self._client.locate(uri, auto_ingest=False)
            return res.is_hit
        except Exception:
            return False


# Register with fsspec
fsspec.register_implementation("hypercache", HyperCacheFileSystem, clobber=True)
