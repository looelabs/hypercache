# HyperCache Python Client & Data Plane (`client-py`)

The `hypercache` Python package integrates transparently into AI training and inference pipelines (PyTorch, Hugging Face, Safetensors) to provide **zero-copy Direct Block DMA / GPUDirect Storage (GDS)** checkpoint loading.

---

## Key Features

- **Transparent `hypercache://` Protocol**: Seamless `fsspec` filesystem implementation.
- **Dual Direct / Buffered Handles**: Uses `O_DIRECT` with sector/page-aligned DMA buffers for maximum hardware read throughput (up to **~7 GB/s** per PCIe Gen 4 NVMe and **~14 GB/s** per PCIe Gen 5 NVMe), with automatic transparent fallback for arbitrary unaligned byte slices.
- **Sub-Millisecond Metadata**: Resolves physical drive offsets from the local HyperCache control plane daemon in **< 1 ms** over Unix Domain Sockets.
- **Zero-Copy GPU Direct**: Direct DMA transfers bypassing host CPU and Linux page cache.

---

## Installation

```bash
cd client-py
pip install -e .

# Optional GPU Direct Storage acceleration dependencies
pip install -e ".[gpu]"
```

---

## Usage Examples

### 1. PyTorch & Safetensors Loading
```python
import torch
import safetensors.torch
from hypercache.fsspec_fs import register_hypercache_fsspec

# Automatically registers hypercache:// protocol
register_hypercache_fsspec()

# Load tensors directly from striped NVMe storage into VRAM/Host memory
with open("hypercache://models/llama-3/weights.safetensors", "rb") as f:
    tensors = safetensors.torch.load(f.read())
```

### 2. Direct Python Client API
```python
from hypercache.client import HyperCacheClient
from hypercache.datapath import get_data_reader

# Connect to local Control Plane Daemon (auto-resolves /run/hypercache/hypercache.sock)
client = HyperCacheClient()

# Check daemon health & discovered drives
devices = client.get_devices()
print(f"Discovered {len(devices)} active striped devices.")

# Locate checkpoint
layout = client.locate("hypercache://models/llama-3/weights.safetensors", auto_ingest=True)
if layout.is_hit:
    reader = get_data_reader()
    data = reader.read_chunks(layout.chunk_layout, layout.object_size)
    print(f"Read {len(data)/(1024*1024):.2f} MiB at hardware line rate!")
```

---

## Running Unit Tests

```bash
pytest tests/
```
