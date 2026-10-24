"""
HyperCache Framework & Ecosystem Integrations.

Provides native zero-copy loaders and checkpoint adapters for:
- Safetensors / Hugging Face / vLLM (`safetensors_loader`)
- PyTorch Distributed Checkpoint / FSDP / DeepSpeed (`dcp`)
"""

from hypercache.integrations.dcp import (
    HyperCacheDCPStorageReader,
    HyperCacheDCPStorageWriter,
)
from hypercache.integrations.safetensors_loader import (
    HyperCachevLLMWeightLoader,
    load_safetensors_state_dict,
    save_safetensors_state_dict,
)

__all__ = [
    "load_safetensors_state_dict",
    "save_safetensors_state_dict",
    "HyperCachevLLMWeightLoader",
    "HyperCacheDCPStorageWriter",
    "HyperCacheDCPStorageReader",
]
