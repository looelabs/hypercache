"""
PyTorch Distributed Checkpoint (DCP / FSDP / DeepSpeed) Storage Adapter for HyperCache.

Allows multi-rank distributed training jobs to save and restore sharded checkpoints
(`shard_rank_00000.pt`, etc.) directly through HyperCache's Tier-1 RAM or Tier-2 NVMe
pool with asynchronous write-back to S3/NFS.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, List, Optional

from hypercache.fsspec_fs import HyperCacheFileSystem


class HyperCacheDCPStorageWriter:
    """
    Distributed Checkpoint StorageWriter backed by HyperCache.
    Each rank writes its own shard (`{base_uri}/rank_{rank:05d}.pt`) into local NVMe/RAM
    in milliseconds while background threads sync to authoritative storage.
    """

    def __init__(
        self,
        base_uri: str,
        rank: int = 0,
        world_size: int = 1,
        tier: Optional[str] = None,
        write_through: bool = False,
        socket_path: Optional[str] = None,
    ):
        self.base_uri = base_uri.rstrip("/")
        self.rank = rank
        self.world_size = world_size
        self.tier = tier
        self.write_through = write_through
        self.socket_path = socket_path
        self.fs = HyperCacheFileSystem(socket_path=socket_path, prefer_gds=False)

    def shard_uri(self, rank: Optional[int] = None) -> str:
        r = self.rank if rank is None else rank
        return f"{self.base_uri}/shard_rank_{r:05d}_of_{self.world_size:05d}.pt"

    def metadata_uri(self) -> str:
        return f"{self.base_uri}/.metadata.json"

    def save_rank_shard(
        self,
        state_dict: Dict[str, Any],
        extra_metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Saves the current rank's FSDP/DeepSpeed shard to HyperCache.
        If `rank == 0`, also writes the distributed checkpoint manifest `.metadata.json`.
        """
        uri = self.shard_uri(self.rank)
        is_leader = self.rank == 0

        try:
            import torch
            has_torch = True
        except ImportError:
            torch = None  # type: ignore
            has_torch = False

        with self.fs.open(
            uri,
            "wb",
            tier=self.tier,
            is_leader=is_leader,
            write_through=self.write_through,
        ) as wf:
            if has_torch:
                torch.save(state_dict, wf)
            else:
                payload = json.dumps(
                    {k: list(v) if isinstance(v, (bytes, bytearray)) else v for k, v in state_dict.items()}
                ).encode("utf-8")
                wf.write(payload)

        if is_leader:
            manifest = {
                "world_size": self.world_size,
                "shards": [self.shard_uri(r) for r in range(self.world_size)],
                "extra_metadata": extra_metadata or {},
            }
            meta_bytes = json.dumps(manifest, indent=2).encode("utf-8")
            with self.fs.open(
                self.metadata_uri(),
                "wb",
                tier=self.tier,
                is_leader=True,
                write_through=self.write_through,
            ) as mf:
                mf.write(meta_bytes)

        return uri


class HyperCacheDCPStorageReader:
    """
    Distributed Checkpoint StorageReader backed by HyperCache.
    Restores per-rank FSDP/DeepSpeed shards using zero-copy Direct Block DMA.
    """

    def __init__(
        self,
        base_uri: str,
        rank: int = 0,
        world_size: int = 1,
        socket_path: Optional[str] = None,
        prefer_gds: bool = True,
    ):
        self.base_uri = base_uri.rstrip("/")
        self.rank = rank
        self.world_size = world_size
        self.socket_path = socket_path
        self.fs = HyperCacheFileSystem(socket_path=socket_path, prefer_gds=prefer_gds)

    def shard_uri(self, rank: Optional[int] = None) -> str:
        r = self.rank if rank is None else rank
        return f"{self.base_uri}/shard_rank_{r:05d}_of_{self.world_size:05d}.pt"

    def metadata_uri(self) -> str:
        return f"{self.base_uri}/.metadata.json"

    def read_metadata(self) -> Dict[str, Any]:
        with self.fs.open(self.metadata_uri(), "rb") as mf:
            return json.loads(bytes(mf.read()).decode("utf-8"))

    def load_rank_shard(self, rank: Optional[int] = None, map_location: Any = "cpu") -> Dict[str, Any]:
        uri = self.shard_uri(rank)
        try:
            import torch
            has_torch = True
        except ImportError:
            torch = None  # type: ignore
            has_torch = False

        with self.fs.open(uri, "rb") as rf:
            if has_torch:
                return torch.load(rf, map_location=map_location, weights_only=False)
            else:
                raw = bytes(rf.read()).decode("utf-8")
                return json.loads(raw)
