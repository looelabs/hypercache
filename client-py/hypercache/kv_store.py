"""
HyperCache KV-Cache Prefix Block Store (`HyperCacheKVStore`).

Provides a content-addressed key-value block interface over HyperCache's Tier-1
Host RAM (`/dev/shm` / HugePages) and Tier-2 Striped NVMe engine.

This is a building block for code that manages its own KV-cache blocks. For LMCache,
use the connector in `lmcache-connector/`. There is no NVIDIA Dynamo KVBM integration.

Blocks stay on this host and remain evictable, unless write_through=True asks for a
synchronous backend flush.

Every block is checked against per-chunk CRC32 values on read, and a block whose state or
checksums can't be confirmed is a miss: a wrong block is never returned.
"""

from __future__ import annotations

import hashlib
import logging
import zlib
from typing import Any, List, Optional, Sequence, Union

from hypercache.client import HyperCacheClient
from hypercache.datapath import ChunkLocation, DirectBlockWriter, get_data_reader

logger = logging.getLogger(__name__)

# States in which a block's data is complete and stays put: LOCAL_ONLY for normal blocks,
# SYNCED_BACKEND for write-through ones. Anything else (staged, being rewritten, unknown) is a miss.
_READABLE_STATES = ("LOCAL_ONLY", "SYNCED_BACKEND")


class HyperCacheKVStore:
    """
    Content-addressed KV-Cache Block Store backed by HyperCache Tier 1 (RAM) and Tier 2 (NVMe).
    """

    KV_URI_PREFIX = "hypercache://kv_cache"

    def __init__(
        self,
        socket_path: Optional[str] = None,
        default_tier: str = "ram",
        prefer_gds: bool = True,
        write_through: bool = False,
    ):
        self.socket_path = socket_path
        self.default_tier = default_tier
        self.prefer_gds = prefer_gds
        self.write_through = write_through
        self._client = HyperCacheClient(socket_path=socket_path)
        self._reader = get_data_reader(prefer_gds=prefer_gds)
        self._writer = DirectBlockWriter()

    @staticmethod
    def compute_prefix_hash(
        token_ids: Sequence[int],
        model_id: str = "default",
        layer_idx: Optional[int] = None,
    ) -> str:
        """
        Computes a deterministic SHA-256 content-addressed key for a token prefix chunk.
        """
        h = hashlib.sha256()
        h.update(model_id.encode("utf-8"))
        h.update(b"::")
        if layer_idx is not None:
            h.update(f"layer_{layer_idx}::".encode("utf-8"))
        h.update(",".join(str(t) for t in token_ids).encode("utf-8"))
        return h.hexdigest()

    def _make_uri(self, block_key: str) -> str:
        if block_key.startswith("hypercache://"):
            return block_key
        return f"{self.KV_URI_PREFIX}/{block_key}.kv"

    def put_kv_block(
        self,
        block_key: str,
        kv_data: Union[bytes, bytearray, memoryview, Any],
        tier: Optional[str] = None,
        pinned: bool = False,
    ) -> bool:
        """
        Stores a KV-cache tensor or byte block into HyperCache Tier-1 RAM or Tier-2 NVMe.
        """
        uri = self._make_uri(block_key)
        target_tier = tier or self.default_tier

        if isinstance(kv_data, (bytes, bytearray, memoryview)):
            raw_bytes = bytes(kv_data)
        else:
            try:
                import torch
                if isinstance(kv_data, torch.Tensor):
                    t_cpu = kv_data.detach().contiguous().cpu()
                    raw_bytes = (
                        t_cpu.numpy().tobytes()
                        if t_cpu.dtype != torch.bfloat16
                        else t_cpu.view(torch.uint16).numpy().tobytes()
                    )
                else:
                    raise TypeError(f"Unsupported kv_data type: {type(kv_data)}")
            except ImportError as e:
                raise TypeError("kv_data must be bytes-like when PyTorch is unavailable") from e

        stage_res = self._client.stage_write(
            uri=uri,
            size_bytes=len(raw_bytes),
            tier=target_tier,
            is_leader=False,
            auto_flush_s3=False,
            write_through=self.write_through,
            pinned=pinned,
        )
        if not stage_res.is_staged:
            return False

        layout = stage_res.chunk_layout
        try:
            if sum(c.length for c in layout) != len(raw_bytes):
                raise IOError(f"staged layout for {uri} does not match the block size")
            self._writer.write_chunks(layout, raw_bytes)
            if self.write_through:
                # The daemon verifies the digest and records per-chunk CRCs from it
                committed = self._client.commit_write(
                    uri, sha256=hashlib.sha256(raw_bytes).hexdigest(), write_through=True
                )
            else:
                committed = self._client.commit_write(
                    uri,
                    local_only=True,
                    chunk_checksums=_chunk_crcs(layout, raw_bytes),
                    generation=stage_res.generation,
                )
        except Exception:
            self._discard(uri)
            raise
        if not committed:
            self._discard(uri)
        return committed

    def _discard(self, uri: str) -> None:
        # A block that failed to store must not linger: it may be unwritten or not local-only
        try:
            self._client.evict(uri)
        except Exception as e:  # noqa: BLE001 - best effort; the block is unreadable anyway
            logger.warning(f"Could not evict failed KV block {uri}: {e}")

    def get_kv_block(
        self,
        block_key: str,
        out_tensor: Optional[Any] = None,
    ) -> Optional[Union[bytes, Any]]:
        """
        Retrieves a KV-cache block by key. If `out_tensor` (a pre-allocated CPU or CUDA
        PyTorch tensor) is provided, copies directly into `out_tensor` and returns it.
        Returns `None` on cache miss.
        """
        uri = self._make_uri(block_key)
        loc = self._client.locate(uri, auto_ingest=False)
        if not self._is_readable(loc):
            return None
        if out_tensor is not None and _nbytes(out_tensor) != loc.object_size:
            # A buffer of another size can't hold this block; leave it untouched
            return None

        # Read to host memory and check every chunk before anything reaches the caller, so a
        # stale or torn block is a miss rather than wrong data in out_tensor
        try:
            raw = bytes(self._reader.read_range(loc.chunk_layout, 0, loc.object_size))
        except (IOError, OSError) as e:
            logger.warning(f"KV block {uri} could not be read, treating it as a miss: {e}")
            return None
        if len(raw) != loc.object_size or _chunk_crcs(loc.chunk_layout, raw) != [
            c.checksum for c in loc.chunk_layout
        ]:
            logger.warning(f"KV block {uri} failed its checksum, treating it as a miss")
            return None

        if out_tensor is not None:
            import torch

            src = torch.frombuffer(bytearray(raw), dtype=torch.uint8)
            out_tensor.view(torch.uint8).reshape(-1).copy_(src)
            return out_tensor
        return raw

    def exists_kv_block(self, block_key: str) -> bool:
        uri = self._make_uri(block_key)
        return self._is_readable(self._client.locate(uri, auto_ingest=False))

    @staticmethod
    def _is_readable(loc: Any) -> bool:
        # Only a finished block whose every chunk has a recorded CRC can be verified. A staged
        # layout is published before its data is written, and an old daemon reports no state.
        layout = loc.chunk_layout or []
        return (
            loc.is_hit
            and loc.sync_status in _READABLE_STATES
            and bool(layout)
            and all(c.checksum is not None for c in layout)
            and sum(c.length for c in layout) == loc.object_size
        )

    def close(self) -> None:
        self._reader.close()
        self._writer.close()
        self._client.close()


def _chunk_crcs(layout: List[ChunkLocation], data: bytes) -> List[int]:
    """CRC32 of each chunk's slice of the block, in layout order."""
    view = memoryview(data)
    crcs = []
    position = 0
    for chunk in layout:
        crcs.append(zlib.crc32(view[position : position + chunk.length]) & 0xFFFFFFFF)
        position += chunk.length
    return crcs


def _nbytes(tensor: Any) -> int:
    return tensor.numel() * tensor.element_size()
