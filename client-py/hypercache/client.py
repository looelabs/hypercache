"""
HyperCache UDS IPC Client.

Facilitates sub-millisecond metadata handshakes with the Rust Control Plane.
"""

from __future__ import annotations

import json
import logging
import os
import socket
import threading
from dataclasses import dataclass
from typing import List, Optional

from hypercache.datapath import ChunkLocation

logger = logging.getLogger("hypercache.client")

# Sentinel meaning "use the client's default request timeout"
_DEFAULT = object()

PRIMARY_SOCKET_PATH = "/run/hypercache/hypercache.sock"
FALLBACK_SOCKET_PATH = "/tmp/hypercache/hypercache.sock"
LEGACY_SOCKET_PATH = "/tmp/hypercache.sock"


def get_default_socket_path() -> str:
    """Discovers and returns the active HyperCache Unix Domain Socket path."""
    override = os.environ.get("HYPERCACHE_SOCKET_PATH", "").strip()
    if override:
        return override
    if os.path.exists(PRIMARY_SOCKET_PATH):
        return PRIMARY_SOCKET_PATH
    if os.path.exists(FALLBACK_SOCKET_PATH):
        return FALLBACK_SOCKET_PATH
    if os.path.exists(LEGACY_SOCKET_PATH):
        return LEGACY_SOCKET_PATH
    if os.path.exists("/run"):
        return PRIMARY_SOCKET_PATH
    return FALLBACK_SOCKET_PATH


DEFAULT_SOCKET_PATH = PRIMARY_SOCKET_PATH


@dataclass
class LocateResult:
    status: str
    object_size: int
    chunk_size: int
    chunk_layout: List[ChunkLocation]
    error_message: Optional[str] = None
    # Layout generation reported by the daemon (changes whenever the URI is re-ingested or rewritten)
    generation: Optional[int] = None

    @property
    def is_hit(self) -> bool:
        return self.status == "HIT"


@dataclass
class StageWriteResult:
    status: str
    object_size: int
    chunk_size: int
    chunk_layout: List[ChunkLocation]
    sync_status: str
    error_message: Optional[str] = None

    @property
    def is_staged(self) -> bool:
        return self.status in ("WRITESTAGED", "WRITE_STAGED")


@dataclass
class DiscoveredDeviceInfo:
    path: str
    is_nvme: bool
    is_virtual: bool
    is_partition: bool
    is_rotational: bool
    size_bytes: int
    model: Optional[str] = None
    existing_fs: Optional[str] = None
    numa_node: Optional[int] = None

    @classmethod
    def from_dict(cls, data: dict) -> DiscoveredDeviceInfo:
        return cls(
            path=str(data.get("path", "")),
            is_nvme=bool(data.get("is_nvme", False)),
            is_virtual=bool(data.get("is_virtual", False)),
            is_partition=bool(data.get("is_partition", False)),
            is_rotational=bool(data.get("is_rotational", False)),
            size_bytes=int(data.get("size_bytes", 0)),
            model=data.get("model"),
            existing_fs=data.get("existing_fs"),
            numa_node=data.get("numa_node"),
        )


class HyperCacheClient:
    """Client for querying the HyperCache Control Plane daemon via Unix Domain Socket.

    A client instance is safe to share between threads (requests are serialized on one
    connection) and across fork() (the child transparently opens its own connection).

    Args:
        socket_path: daemon socket; auto-discovered when omitted.
        timeout: per-request timeout in seconds for metadata calls.
        ingest_timeout: timeout for calls that may download a whole object from S3/NFS
            (locate with auto_ingest, ingest, preload). None waits indefinitely, since
            multi-GB ingests legitimately take minutes.
    """

    def __init__(
        self,
        socket_path: Optional[str] = None,
        timeout: float = 120.0,
        ingest_timeout: Optional[float] = None,
    ):
        self.socket_path = socket_path or get_default_socket_path()
        self.timeout = timeout
        self.ingest_timeout = ingest_timeout
        self._sock: Optional[socket.socket] = None
        self._rbuf = bytearray()
        self._lock = threading.RLock()
        self._pid = os.getpid()

    def _check_fork(self) -> None:
        # A socket inherited from the parent process is shared with it, so a response could
        # be read by the wrong process. Close this process's copy and reconnect lazily.
        if self._pid != os.getpid():
            if self._sock is not None:
                try:
                    self._sock.close()
                except Exception:
                    pass
            self._sock = None
            self._rbuf = bytearray()
            self._lock = threading.RLock()
            self._pid = os.getpid()

    def _connect(self) -> socket.socket:
        if self._sock is not None:
            return self._sock

        try:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(self.timeout)
            sock.connect(self.socket_path)
            self._sock = sock
            self._rbuf = bytearray()
            return sock
        except PermissionError as e:
            raise ConnectionError(
                f"Permission denied on HyperCache socket {self.socket_path} (uid {os.getuid()}). "
                "Set client_gid in the daemon configuration and run this process with that group."
            ) from e
        except Exception as e:
            logger.debug(f"Failed to connect to HyperCache daemon at {self.socket_path}: {e}")
            raise ConnectionError(
                f"HyperCache daemon unreachable at {self.socket_path}. Is the daemon running?"
            ) from e

    def _read_line(self, sock: socket.socket) -> bytes:
        while True:
            idx = self._rbuf.find(b"\n")
            if idx >= 0:
                line = bytes(self._rbuf[:idx])
                del self._rbuf[: idx + 1]
                return line
            chunk = sock.recv(65536)
            if not chunk:
                raise ConnectionResetError("Connection closed by server without response")
            self._rbuf.extend(chunk)

    def _send_request(self, payload: dict, timeout=_DEFAULT) -> dict:
        self._check_fork()
        with self._lock:
            sock = self._connect()
            data = (json.dumps(payload) + "\n").encode("utf-8")
            try:
                sock.settimeout(self.timeout if timeout is _DEFAULT else timeout)
                sock.sendall(data)
                while True:
                    line = self._read_line(sock)
                    if line.strip():
                        return json.loads(line.decode("utf-8"))
            except (OSError, ConnectionError, ValueError):
                # Timeouts, resets and garbled frames leave the stream in an unknown
                # position; drop the connection so the next request starts clean.
                self.close()
                raise

    def ping(self) -> bool:
        """Checks if the control plane is alive."""
        try:
            resp = self._send_request({"action": "ping"})
            return resp.get("status") == "PONG"
        except Exception:
            return False

    def get_devices(self) -> List[DiscoveredDeviceInfo]:
        """Queries the control plane for dynamically auto-discovered SSDs/NVMe devices."""
        resp = self._send_request({"action": "get_devices"})
        raw_devices = resp.get("devices", [])
        return [DiscoveredDeviceInfo.from_dict(d) for d in raw_devices]

    def locate(
        self,
        uri: str,
        auto_ingest: bool = True,
        tier: Optional[str] = None,
        pinned: bool = False,
    ) -> LocateResult:
        """
        Sends a locate request to the control plane for a given URI.
        Returns exact physical chunk routing layout.
        """
        req = {
            "action": "locate",
            "uri": uri,
            "auto_ingest": auto_ingest,
            "pinned": pinned,
        }
        if tier is not None:
            req["tier"] = tier.lower()

        resp = self._send_request(req, timeout=self.ingest_timeout if auto_ingest else _DEFAULT)
        status = resp.get("status", "ERROR")

        if status == "HIT":
            chunk_layout = [
                ChunkLocation.from_dict(c) for c in resp.get("chunk_layout", [])
            ]
            return LocateResult(
                status=status,
                object_size=resp.get("object_size", 0),
                chunk_size=resp.get("chunk_size", 4194304),
                chunk_layout=chunk_layout,
                generation=resp.get("generation"),
            )
        elif status == "MISS":
            return LocateResult(
                status=status,
                object_size=0,
                chunk_size=0,
                chunk_layout=[],
            )
        else:
            err = resp.get("error_message", "Unknown error from control plane")
            return LocateResult(
                status=status,
                object_size=0,
                chunk_size=0,
                chunk_layout=[],
                error_message=err,
            )

    def ingest(
        self,
        uri: str,
        tier: Optional[str] = None,
        pinned: bool = False,
    ) -> LocateResult:
        """Forces immediate ingestion and striping of an object into the SSD or RAM pool."""
        req = {
            "action": "ingest",
            "uri": uri,
            "pinned": pinned,
        }
        if tier is not None:
            req["tier"] = tier.lower()
        resp = self._send_request(req, timeout=self.ingest_timeout)
        status = resp.get("status", "ERROR")
        if status in ("HIT", "INGESTED"):
            chunk_layout = [
                ChunkLocation.from_dict(c) for c in resp.get("chunk_layout", [])
            ]
            return LocateResult(
                status="HIT",
                object_size=resp.get("object_size", 0),
                chunk_size=resp.get("chunk_size", 4194304),
                chunk_layout=chunk_layout,
                generation=resp.get("generation"),
            )
        else:
            err = resp.get("error_message", "Ingestion failed")
            return LocateResult(
                status="ERROR",
                object_size=0,
                chunk_size=0,
                chunk_layout=[],
                error_message=err,
            )

    def preload(self, uri: str, tier: Optional[str] = None, pinned: bool = False) -> LocateResult:
        """
        Proactively pre-loads / pre-warms a checkpoint from S3 or NFS into the local NVMe cache.
        """
        return self.ingest(uri, tier=tier, pinned=pinned)

    def pin(self, uri: str, pinned: bool = True) -> bool:
        """Pins or unpins an existing cached object so W-TinyLFU never evicts it."""
        resp = self._send_request({
            "action": "pin",
            "uri": uri,
            "pinned": pinned,
        })
        return resp.get("status") == "PONG"

    def evict(self, uri: str) -> bool:
        """Evicts a specific URI from the cache and reclaims its 4 MiB blocks."""
        resp = self._send_request({
            "action": "evict",
            "uri": uri,
        })
        return resp.get("status") in ("EVICTED", "PONG", "OK")

    def purge(self) -> bool:
        """Purges all cached objects from the daemon and resets block allocators."""
        resp = self._send_request({"action": "purge"})
        return resp.get("status") in ("PURGED", "PONG", "OK")

    def get_status(self) -> dict:
        """Queries live daemon metrics and pool utilization."""
        return self._send_request({"action": "get_status"})

    def stage_write(
        self,
        uri: str,
        size_bytes: int,
        tier: Optional[str] = None,
        is_leader: bool = False,
        auto_flush_s3: bool = True,
        write_through: bool = False,
        pinned: bool = False,
    ) -> StageWriteResult:
        """
        Stages an asynchronous write-back or synchronous write-through allocation in the control plane.
        Allocates physical chunk layouts across NVMe drives / RAM without blocking.
        """
        req = {
            "action": "stage_write",
            "uri": uri,
            "size_bytes": size_bytes,
            "is_leader": is_leader,
            "auto_flush_s3": auto_flush_s3,
            "write_through": write_through,
            "pinned": pinned,
        }
        if tier is not None:
            req["tier"] = tier.lower()

        resp = self._send_request(req)
        status = resp.get("status", "ERROR")

        if status in ("WRITESTAGED", "WRITE_STAGED"):
            chunk_layout = [
                ChunkLocation.from_dict(c) for c in resp.get("chunk_layout", [])
            ]
            return StageWriteResult(
                status=status,
                object_size=resp.get("object_size", size_bytes),
                chunk_size=resp.get("chunk_size", 4194304),
                chunk_layout=chunk_layout,
                sync_status=resp.get("sync_status", "STAGED"),
            )
        else:
            err = resp.get("error_message", "Failed to stage write")
            return StageWriteResult(
                status="ERROR",
                object_size=0,
                chunk_size=0,
                chunk_layout=[],
                sync_status="ERROR",
                error_message=err,
            )

    def append_write(
        self,
        uri: str,
        size_bytes: int,
        tier: Optional[str] = None,
    ) -> StageWriteResult:
        """
        Incrementally allocates and appends chunk descriptors for a streaming write.
        """
        req = {
            "action": "append_write",
            "uri": uri,
            "size_bytes": size_bytes,
        }
        if tier is not None:
            req["tier"] = tier.lower()

        resp = self._send_request(req)
        status = resp.get("status", "ERROR")
        if status in ("WRITESTAGED", "WRITE_STAGED"):
            chunk_layout = [
                ChunkLocation.from_dict(c) for c in resp.get("chunk_layout", [])
            ]
            return StageWriteResult(
                status=status,
                object_size=resp.get("object_size", size_bytes),
                chunk_size=resp.get("chunk_size", 4194304),
                chunk_layout=chunk_layout,
                sync_status=resp.get("sync_status", "STAGED"),
            )
        else:
            err = resp.get("error_message", "Failed to append write")
            return StageWriteResult(
                status="ERROR",
                object_size=0,
                chunk_size=0,
                chunk_layout=[],
                sync_status="ERROR",
                error_message=err,
            )

    def commit_write(
        self,
        uri: str,
        sha256: Optional[str] = None,
        write_through: bool = False,
    ) -> bool:
        """
        Commits a staged write in the control plane, triggering background sync to S3 (write-back)
        or synchronously waiting for authoritative backend flush (write-through).
        """
        req = {
            "action": "commit_write",
            "uri": uri,
            "write_through": write_through,
        }
        if sha256 is not None:
            req["sha256"] = sha256

        # Commit hashes the staged data (when sha256 is given) and, for write-through, uploads it
        resp = self._send_request(req, timeout=self.ingest_timeout)
        status = resp.get("status", "ERROR")
        if status not in ("WRITECOMMITTED", "WRITE_COMMITTED"):
            logger.error(f"Commit of '{uri}' failed: {resp.get('error_message', 'unknown error')}")
            return False
        return True

    def get_write_status(self, uri: str) -> dict:
        """
        Queries current sync status (STAGED, COMMITTED, SYNCING_BACKEND, SYNCED_BACKEND, SYNC_FAILED).
        """
        req = {
            "action": "get_write_status",
            "uri": uri,
        }
        resp = self._send_request(req)
        return {
            "status": resp.get("sync_status", "UNKNOWN"),
            "progress_percent": resp.get("sync_progress_percent", 0.0),
        }

    def record_telemetry(
        self,
        event_type: str,
        uri: str,
        object_size_bytes: int,
        duration_us: int,
        bandwidth_mbps: float,
        client_id: str = "python_benchmark",
        chunk_count: int = 1,
        local_meta_us: int = 0,
        remote_meta_us: int = 0,
        local_ttfb_us: int = 0,
        remote_ttfb_us: int = 0,
        time_saved_ms: int = 0,
        cpu_pct: float = 0.0,
        status: str = "OK",
    ) -> bool:
        """Emits telemetry metrics to the control plane daemon for Prometheus & Grafana."""
        req = {
            "action": "record_telemetry",
            "event_type": event_type,
            "uri": uri,
            "object_size_bytes": object_size_bytes,
            "duration_us": duration_us,
            "bandwidth_mbps": bandwidth_mbps,
            "client_id": client_id,
            "chunk_count": chunk_count,
            "local_meta_us": local_meta_us,
            "remote_meta_us": remote_meta_us,
            "local_ttfb_us": local_ttfb_us,
            "remote_ttfb_us": remote_ttfb_us,
            "time_saved_ms": time_saved_ms,
            "cpu_pct": cpu_pct,
            "status": status,
        }
        try:
            resp = self._send_request(req)
            return resp.get("status") == "PONG"
        except Exception:
            return False

    def close(self) -> None:
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        self._rbuf = bytearray()

    def __enter__(self) -> HyperCacheClient:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
