"""
HyperCache Layer-by-Layer Pipelined Loader & Compute-I/O Overlap Engine.

Enables sub-100ms perceived cold starts for LLM inference by:
1. Immediately loading the Embedding & Layer 0 into GPU VRAM / Host RAM.
2. Beginning model forward pass execution without waiting for the full 70B model.
3. Concurrently prefetching Layer i+1 in a background worker while the GPU executes Layer i.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

from hypercache.client import HyperCacheClient, LocateResult
from hypercache.datapath import ChunkLocation, DataReader, get_data_reader

logger = logging.getLogger("hypercache.pipeline")


@dataclass
class LayerDescriptor:
    """Represents a logical layer within a serialized model checkpoint."""
    layer_idx: int
    name: str
    start_offset: int
    length: int


class HyperCachePipelinedLoader:
    """
    Pipelined, compute-overlapped model loader for streaming transformer weights.
    """

    def __init__(
        self,
        uri: str,
        socket_path: Optional[str] = None,
        layer_size_bytes: int = 64 * 1024 * 1024, # 64 MiB default per layer
        num_layers: Optional[int] = None,
        prefer_gds: bool = True,
        prefetch_ahead: int = 1,
    ):
        self.uri = uri
        self.layer_size_bytes = layer_size_bytes
        self.num_layers = num_layers
        self.prefer_gds = prefer_gds
        self.prefetch_ahead = max(1, prefetch_ahead)

        self._client = HyperCacheClient(socket_path=socket_path)
        self._reader: DataReader = get_data_reader(prefer_gds=prefer_gds)
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="HyperCachePipelinedPrefetch")

        # 1. Locate object
        self.locate_result: LocateResult = self._client.locate(self.uri, auto_ingest=True)
        if not self.locate_result.is_hit:
            raise FileNotFoundError(
                f"Could not locate {self.uri} in HyperCache: {self.locate_result.error_message or 'MISS'}"
            )

        self.total_size = self.locate_result.object_size
        self.chunk_layout = self.locate_result.chunk_layout

        # 2. Build layer descriptors
        self.layers: List[LayerDescriptor] = self._partition_layers()
        self._prefetch_futures: Dict[int, Future] = {}

    def _partition_layers(self) -> List[LayerDescriptor]:
        """Divides object byte range into sequential layer descriptors."""
        layers = []
        if self.num_layers is not None and self.num_layers > 0:
            size_per_layer = max(1, self.total_size // self.num_layers)
            for i in range(self.num_layers):
                start = i * size_per_layer
                length = (self.total_size - start) if i == self.num_layers - 1 else size_per_layer
                name = "embedding" if i == 0 else f"layer_{i-1}"
                layers.append(LayerDescriptor(layer_idx=i, name=name, start_offset=start, length=length))
        else:
            current_offset = 0
            idx = 0
            while current_offset < self.total_size:
                length = min(self.layer_size_bytes, self.total_size - current_offset)
                name = "embedding" if idx == 0 else f"layer_{idx-1}"
                layers.append(LayerDescriptor(layer_idx=idx, name=name, start_offset=current_offset, length=length))
                current_offset += length
                idx += 1
        return layers

    def _read_layer_bytes(self, layer: LayerDescriptor) -> bytes:
        """Reads layer bytes from local NVMe/RAM block pool."""
        return self._reader.read_range(
            self.chunk_layout,
            start_offset=layer.start_offset,
            length=layer.length,
        )

    def load_initial_layer(self) -> Tuple[LayerDescriptor, bytes]:
        """
        Synchronously loads Layer 0 / Embedding and triggers background prefetching for Layer 1.
        Unblocks the inference engine in ~20-50 ms.
        """
        if not self.layers:
            raise ValueError("No layers found in model.")

        first_layer = self.layers[0]
        data = self._read_layer_bytes(first_layer)

        # Trigger prefetching for subsequent layer(s)
        for i in range(1, min(1 + self.prefetch_ahead, len(self.layers))):
            self.trigger_prefetch(i)

        return first_layer, data

    def trigger_prefetch(self, layer_idx: int) -> None:
        """Schedules asynchronous background DMA fetch for layer_idx."""
        if layer_idx < len(self.layers) and layer_idx not in self._prefetch_futures:
            layer = self.layers[layer_idx]
            self._prefetch_futures[layer_idx] = self._executor.submit(self._read_layer_bytes, layer)

    def get_layer(self, layer_idx: int) -> Tuple[LayerDescriptor, bytes]:
        """
        Retrieves a layer payload, waiting on background prefetch if currently in flight.
        """
        if layer_idx >= len(self.layers):
            raise IndexError(f"Layer index {layer_idx} out of range (total: {len(self.layers)})")

        layer = self.layers[layer_idx]

        # Trigger prefetch for layer_idx + prefetch_ahead
        target_prefetch = layer_idx + self.prefetch_ahead
        if target_prefetch < len(self.layers):
            self.trigger_prefetch(target_prefetch)

        if layer_idx in self._prefetch_futures:
            future = self._prefetch_futures.pop(layer_idx)
            data = future.result()
        else:
            data = self._read_layer_bytes(layer)

        return layer, data

    def stream_layers(self) -> Generator[Tuple[LayerDescriptor, bytes], None, None]:
        """
        Sequential generator yielding (LayerDescriptor, bytes) while overlapping I/O.
        """
        # Prefetch initial layers
        for i in range(min(self.prefetch_ahead, len(self.layers))):
            self.trigger_prefetch(i)

        for i in range(len(self.layers)):
            yield self.get_layer(i)

    def close(self) -> None:
        self._executor.shutdown(wait=False)
        self._reader.close()
        self._client.close()

    def __enter__(self) -> HyperCachePipelinedLoader:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


def pipeline_load(
    uri: str,
    socket_path: Optional[str] = None,
    layer_size_bytes: int = 64 * 1024 * 1024,
    num_layers: Optional[int] = None,
) -> HyperCachePipelinedLoader:
    """Convenience helper to initialize a HyperCache pipelined streaming loader."""
    return HyperCachePipelinedLoader(
        uri=uri,
        socket_path=socket_path,
        layer_size_bytes=layer_size_bytes,
        num_layers=num_layers,
    )
