"""
HyperCache KV-Cache Prefix Block Store (`HyperCacheKVStore`).

Provides a content-addressed key-value block interface over HyperCache's Tier-1
Host RAM (`/dev/shm` / HugePages) and Tier-2 Striped NVMe engine, enabling
plug-and-play KV-cache offloading for LMCache, vLLM prefix caching, and
NVIDIA Dynamo KVBM.
"""

from __future__ import annotations

import hashlib
from typing import Any, Optional, Sequence, Union

from hypercache.client import HyperCacheClient
from hypercache.datapath import DirectBlockWriter, get_data_reader


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

        self._writer.write_chunks(stage_res.chunk_layout, raw_bytes)
        return self._client.commit_write(uri, write_through=self.write_through)

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
        if not loc.is_hit:
            return None

        if out_tensor is not None:
            try:
                import torch
                if isinstance(out_tensor, torch.Tensor):
                    if out_tensor.is_cuda:
                        self._reader.read_into_gpu(
                            loc.chunk_layout,
                            start_offset=0,
                            length=loc.object_size,
                            gpu_buffer=out_tensor.view(torch.uint8).reshape(-1),
                        )
                    else:
                        raw = self._reader.read_range(loc.chunk_layout, 0, loc.object_size)
                        src = torch.frombuffer(bytearray(raw), dtype=out_tensor.dtype).reshape(out_tensor.shape)
                        out_tensor.copy_(src)
                    return out_tensor
            except ImportError:
                pass

        return bytes(self._reader.read_range(loc.chunk_layout, 0, loc.object_size))

    def exists_kv_block(self, block_key: str) -> bool:
        uri = self._make_uri(block_key)
        return self._client.locate(uri, auto_ingest=False).is_hit

    def close(self) -> None:
        self._reader.close()
        self._writer.close()
        self._client.close()
