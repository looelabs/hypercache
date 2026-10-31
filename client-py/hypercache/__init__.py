"""
HyperCache: Host-Based Zero-Copy GPUDirect Storage Cache.
"""

from typing import Optional

from hypercache.client import DiscoveredDeviceInfo, HyperCacheClient, LocateResult
from hypercache.datapath import (
    ChunkLocation,
    DataReader,
    DirectBlockReader,
    GDSReader,
    get_data_reader,
)
from hypercache.fsspec_fs import HyperCacheFile, HyperCacheFileSystem
from hypercache.integrations import (
    HyperCacheDCPStorageReader,
    HyperCacheDCPStorageWriter,
    HyperCachevLLMWeightLoader,
    load_safetensors_state_dict,
    save_safetensors_state_dict,
)
from hypercache.kv_store import HyperCacheKVStore
from hypercache.pipeline import HyperCachePipelinedLoader, pipeline_load


def preload(
    uri: str,
    socket_path: Optional[str] = None,
    tier: Optional[str] = None,
    pinned: bool = False,
) -> LocateResult:
    """
    Convenience function to proactively pre-warm a checkpoint into the local NVMe cache.
    """
    client = HyperCacheClient(socket_path=socket_path)
    return client.preload(uri, tier=tier, pinned=pinned)


__all__ = [
    "HyperCacheClient",
    "DiscoveredDeviceInfo",
    "HyperCacheFileSystem",
    "HyperCacheFile",
    "ChunkLocation",
    "DataReader",
    "DirectBlockReader",
    "GDSReader",
    "get_data_reader",
    "preload",
    "HyperCachePipelinedLoader",
    "pipeline_load",
    "HyperCacheKVStore",
    "load_safetensors_state_dict",
    "save_safetensors_state_dict",
    "HyperCachevLLMWeightLoader",
    "HyperCacheDCPStorageWriter",
    "HyperCacheDCPStorageReader",
]

__version__ = "0.9.2"
