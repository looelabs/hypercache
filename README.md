# HyperCache: Host-Local Storage & Memory Acceleration for AI Workloads

<img src="images/hypercache-logo.jpeg" alt="HyperCache Logo" width="400">


> **Looking for operator cluster setup instructions?** See the **[Getting Started Guide](GETTING_STARTED.md)** for a step-by-step walkthrough.  
> **Looking for deep systems internals?** See the **[Architecture Whitepaper](ARCHITECTURE.md)** for block layout, DMA mechanics, and subsystem design.

---

## What is HyperCache?

**HyperCache** is an open-source, host-local caching and I/O acceleration engine purpose-built for AI training and inference infrastructure. Written in **Rust** (control plane and cluster coordinator) and **Python** (zero-copy data plane and framework integrations), it runs directly on your compute nodes between remote storage (**Amazon S3, S3-compatible object stores, or NFS**) and your AI frameworks (**PyTorch, Safetensors, Hugging Face, vLLM**).

Instead of pulling multi-gigabyte model weights, training checkpoints, and tokenized datasets across the network every time a job starts, a container scales up, or a training step saves state, HyperCache pools each node's **local Host RAM (`/dev/shm` / HugePages)** and **local NVMe SSDs** into an autonomous, striped high-speed cache tier.

When an AI workload reads a model or writes a checkpoint using a `hypercache://` URI:
1. **Sub-Millisecond Local Resolution (< 1 ms)**: Your Python process queries the local HyperCache daemon over a Unix Domain Socket. The daemon resolves the exact 4 MiB chunk locations in under a millisecond using an embedded in-memory cache—with **zero external databases** (no Redis, etcd, PostgreSQL, or JVM required).
2. **Direct Hardware Streaming**: Your Python process streams chunks directly from local shared memory (`mmap`) or striped NVMe drives (`O_DIRECT` Direct Block DMA and NVIDIA GPUDirect Storage / `kvikio.CuFile`) straight into GPU VRAM or host memory—bypassing the network stack, the Linux kernel page cache, and redundant CPU memory copies.

```
+===================================================================================================+
|                        AI WORKLOAD (PyTorch / vLLM / Safetensors / FSDP)                          |
|                    torch.load("hypercache://s3/models/llama-3-70b/weights.pt")                    |
+=================================================+=================================================+
                                                  |
                  1. Locate Chunks (< 1 ms UDS)   |   2. Zero-Copy Direct Read / Write
                     +----------------------------+----------------------------+
                     |                                                         |
                     v                                                         v
+=========================================+               +=========================================+
|     HYPERCACHE LOCAL DAEMON (Rust)      |               |       LOCAL HARDWARE DATA PLANE         |
|                                         |               |                                         |
| - In-Memory Metadata (Moka W-TinyLFU)   |               | [ Tier 1: Host RAM (/dev/shm / Huge) ]  |
| - Instant Warm-Boot Manifest (< 20 ms)  |               |   -> 50 - 100+ GB/s DDR5 / CXL Stream   |
| - Auto Drive Discovery & Root Safety    |               |                                         |
| - Background Async S3 / NFS Sync        |               | [ Tier 2: Striped Local NVMe Pool ]     |
| - Peer-to-Peer (P2P) Chunk Exchange     |               |   -> Up to 7 GB/s per PCIe Gen 4 NVMe   |
+====================+====================+               |   -> Up to 14 GB/s per PCIe Gen 5 NVMe  |
                     |                                    +=========================================+
                     | 3. Cache Miss Ingest / Async Write-Back Flush
                     v
+===================================================================================================+
|                  AUTHORITATIVE REMOTE STORAGE (Amazon S3 / R2 / MinIO / Ceph / NFS)               |
+===================================================================================================+
```

---

## The Problem: Why AI Infrastructure Hits an I/O Wall

Modern GPUs (NVIDIA H100, H200, B200) can execute terabytes of tensor math per second, but feeding data to those GPUs from remote storage creates four major infrastructure bottlenecks:

1. **Slow Cold Starts & Auto-Scaling Delays**:
   When an inference server scales up to handle a traffic spike or a serverless GPU worker spins up, pulling a 70 GB to 140 GB model from S3 or NFS over a congested network typically takes **5 to 15+ minutes**. Expensive GPUs sit completely idle waiting for weights to download before serving a single token.
2. **Multi-Node "Checkpoint Storms"**:
   In distributed training or fleet rollouts, all 32, 64, or 128 GPU nodes request the exact same checkpoint from central storage at the exact same second ($64 \text{ nodes} \times 70 \text{ GB} = 4.48 \text{ TB}$ of simultaneous network traffic). This saturates Top-of-Rack (ToR) switches, hits S3 rate limits, and steals network bandwidth needed for GPU-to-GPU gradient synchronization (NCCL AllReduce over RoCE/InfiniBand).
3. **Training Stalls During `torch.save()`**:
   Saving multi-gigabyte checkpoints synchronously to remote object storage pauses the entire GPU training cluster at the end of every epoch or checkpoint interval.
4. **Physical Network Card (NIC) & OS Kernel Bottlenecks**:
   No matter how fast a central storage cluster is, a compute node cannot ingest data over the network faster than its physical NIC—and standard POSIX file reads copy gigabytes through the Linux kernel VFS and page cache before ever reaching the GPU.

---

## Theoretical Hardware Bandwidth: Local Bus vs. Network Storage

By shifting data delivery from the network fabric to **local PCIe lanes and host memory channels**, HyperCache unlocks the physical bandwidth already sitting inside every GPU server:

$$\text{Theoretical Max NIC Throughput (GB/s)} = \frac{\text{NIC Bandwidth (Gbps)}}{8}$$

| Data Path / Hardware Tier | Per-Device / Per-Link Theoretical Limit | Multi-Drive / Multi-Channel Node Throughput | Network Traffic Generated |
| :--- | :--- | :--- | :--- |
| **Remote S3 / Object Store (over 10 GbE)** | 1.25 GB/s wire limit | **~0.2 – 0.9 GB/s** (shared & rate-limited) | 100% over ToR switch |
| **Remote Storage (over 25 GbE NIC)** | 3.12 GB/s wire limit | **~1.5 – 2.2 GB/s** (after TCP/TLS overhead) | 100% over ToR switch |
| **Remote Storage (over 100 GbE NIC)** | 12.50 GB/s wire limit | **~8.5 – 10.5 GB/s** (contends with NCCL) | 100% over ToR switch |
| **HyperCache Tier 2: PCIe Gen 4 NVMe** | **~7.0 GB/s per NVMe SSD** | **~28 GB/s** (4x NVMe) / **~56 GB/s** (8x NVMe) | **0 Bytes (100% Local PCIe)** |
| **HyperCache Tier 2: PCIe Gen 5 NVMe** | **~14.0 GB/s per NVMe SSD** | **~56 GB/s** (4x NVMe) / **~112 GB/s** (8x NVMe) | **0 Bytes (100% Local PCIe)** |
| **HyperCache Tier 1: Host RAM (DDR5 / CXL)**| **50 – 100+ GB/s per stream** | **300 – 460+ GB/s** (8–12 channel DDR5 bus) | **0 Bytes (100% Memory Bus)** |

Because each node reads autonomously from its own local RAM and NVMe drives, cluster-wide read bandwidth scales linearly with the number of nodes—with zero central storage ceiling and zero interference with inter-GPU training traffic.

---

## What Can You Use HyperCache For?

HyperCache is designed for both **AI Inference** and **Distributed AI Training**, from a single GPU machine to multi-node clusters:

### 1. Fast LLM Inference Cold Starts & Serverless GPU Scaling
* **Pre-Warm Models Fleet-Wide**: Use `./hypercache-admin preload` to stage base models onto local NVMe or RAM across your cluster ahead of traffic spikes.
* **Layer-by-Layer Pipelined Loading (`HyperCachePipelinedLoader`)**: Instead of waiting for 100% of a model to load before starting inference, HyperCache loads Layer 0 in milliseconds so GPU forward-pass compute starts immediately while background DMA threads prefetch Layers $1, 2, \dots, N$ ahead of execution.
* **Zero-Copy Multi-Tenant Model Sharing (`MAP_SHARED`)**: If you run multiple inference worker processes or containers on the same machine, they map the exact same physical RAM pages in Tier 1 shared memory with **0 bytes of memory duplication**.

### 2. Sub-Millisecond Multi-LoRA Adapter Hot-Swapping
* Multi-tenant serving platforms often run one base model alongside hundreds of fine-tuned customer LoRA adapters (20 MB to 200 MB each).
* By pinning active LoRA adapters in **Tier 1 Host RAM (`/dev/shm` or HugePages)**, HyperCache swaps adapters at memory-bus speed in **under 1 millisecond** with zero network roundtrips.

### 3. Non-Blocking Training Checkpoints (`torch.save` & PyTorch FSDP / DCP)
* **Asynchronous Write-Back Caching**: When your training script calls `torch.save()` (or `HyperCacheDCPStorageWriter` for multi-rank FSDP), HyperCache stages the checkpoint to local RAM or striped NVMe at hardware line rate and **unblocks PyTorch immediately** so training continues without stalling. A background worker asynchronously uploads the checkpoint to S3 or NFS.
* **$k=2$ Peer Checkpoint Replication**: To protect against spot instance preemptions or node crashes before the background S3 upload finishes, HyperCache can replicate staged checkpoint chunks to a peer node in the cluster before unblocking the training loop (`peer_replication_factor = 2`).
* **Synchronous Write-Through Mode**: When strict remote durability is required (e.g., final epoch checkpoints), pass `write_through=True` to wait for remote S3/NFS confirmation before returning.

### 4. Sharded Dataset Caching for Distributed PyTorch DataLoaders
* For multi-node training, the Central Controller partitions tokenized dataset shards across worker nodes using **Rendezvous (Consistent) Hashing** (`./hypercache-admin preload-batch --prefix s3://bucket/shards/ --sharded`).
* Each node caches only its assigned $1/N$ slice of the dataset and streams batches into local PyTorch `DataLoader` workers at local NVMe line rate with zero network traffic. Because it uses consistent hashing, adding or removing a node only remaps $1/N$ of the shards instead of invalidating the entire cluster cache.

### 5. LLM KV-Cache Prefix Block Offloading (`HyperCacheKVStore`)
* Long-context LLM serving generates massive KV-cache tensors. Rather than recomputing shared prompt prefixes when GPU VRAM fills up, inference engines (and frameworks like LMCache or Dynamo KVBM) can use `HyperCacheKVStore` to offload content-addressed KV-cache blocks into Tier 1 Host RAM with automatic spillover to Tier 2 NVMe.

### 6. Peer-to-Peer (P2P) Fleet Distribution
* When preloading a large model across a multi-node cluster, you do not want every node pulling the full file from S3 simultaneously.
* With `p2p_preload_enabled = true`, each node pulls only $1/N$ of the model's 4 MiB chunks from S3 and exchanges the remaining chunks directly with peer nodes over the internal cluster network—cutting S3 egress cost and origin load by **$N\times$**.

---

## Key Capabilities at a Glance

| Capability | How HyperCache Works | Why It Matters to Engineers |
| :--- | :--- | :--- |
| **Dual-Tier Storage Engine** | Combines **Tier 1: Host RAM (`/dev/shm` / HugePages)** and **Tier 2: Striped Local NVMe** with automatic spillover when RAM fills up. | Delivers memory-bus speed for hot adapters/KV blocks and multi-terabyte NVMe capacity for 70B–405B+ weights. |
| **Flexible Storage Backing** | Supports **Raw Block Devices** (`/dev/nvme*`, `/dev/vd*`) AND **Filesystem Direct I/O Arenas** (`O_DIRECT` files on mounted ext4/XFS directories). | Run on bare-metal/VMs with zero filesystem overhead, or run inside unprivileged Kubernetes pods (`emptyDir` / PVC) with native NVIDIA GDS support. |
| **Hardware Safety Guards** | Automatically scans `/sys/class/block`, `/proc/mounts`, and filesystem magic superblocks (ext4, XFS, Btrfs, ZFS, Swap). | Unconditionally excludes your OS root drive and mounted partitions so system disks are never touched. |
| **Cache Pinning & Eviction Protection** | W-TinyLFU size-aware eviction with explicit `--pin` support and automatic lock protection for unsynced write-back checkpoints. | Guarantees active base models and in-flight training checkpoints are never evicted before syncing to S3/NFS. |
| **Zero External Dependencies** | Compiles into two native Rust binaries (`./hypercache` and `./hypercache-admin`) with embedded Moka metadata and atomic disk persistence. | No Docker, Redis, PostgreSQL, or Java runtime to deploy or maintain. Warm-boots multi-terabyte catalogs in `< 20 ms`. |
| **Secure by Default** | Single-node worker daemon opens **zero network TCP ports** by default (communicates only via local Unix Domain Socket). | Safe to run on any workstation or server; fleet API (`:9115`) and Prometheus metrics (`:9110`) are opt-in. |
| **CPU & GPU Agnostic** | Uses NVIDIA GPUDirect Storage (`kvikio.CuFile`) when available, and falls back to multi-threaded `O_DIRECT` + `mmap` on any Linux/macOS host. | Works on GPU clusters as well as CPU-only inference, preprocessing, or development machines. |

---

## Dual-Tier Storage Engine: Dynamic RAM & NVMe Placement

HyperCache unifies volatile memory (**Tier 1: Host RAM / CXL Shared Memory**) and persistent high-density flash (**Tier 2: Striped NVMe Pool**) into a single tiering hierarchy:

```
+===================================================================================+
|                                HYPERCACHE STRIPER                                 |
+─────────────────────────────────────────┬─────────────────────────────────────────+
                                          │ Auto-Tier Routing
               ┌──────────────────────────┴──────────────────────────┐
               │ Tier 1: Host RAM (/dev/shm or HugePages)            │ Tier 2: Striped NVMe
               ▼                                                     ▼
+=============================+                       +=============================+
|     TIER 1: HOST RAM        |                       |    TIER 2: STRIPED NVME     |
| - Speed: 50 - 100+ GB/s     |                       | - Gen 4: ~7 GB/s per drive  |
| - Latency: < 10 µs          |                       | - Gen 5: ~14 GB/s per drive |
| - Ideal for: Dynamic LoRA,  |                       | - Ideal for: 70B+ Weights,  |
|   KV-Cache, Hot Shards      |                       |   Checkpoints, Datasets     |
+=============================+                       +=============================+
```

When no explicit `--tier` flag is specified during ingestion, preloading, or write-back, the striper automatically selects the fastest available tier:

$$\text{Target Tier} = \begin{cases} \text{Tier 1 (RAM)}, & \text{if } \text{RAM is enabled and } (\text{Used} + \text{Size} \le \text{Capacity}) \\ \text{Tier 2 (NVMe)}, & \text{otherwise (seamless spillover to NVMe pool)} \end{cases}$$

---

## How to Build & Install HyperCache

HyperCache compiles into two native, zero-dependency Rust executables (`./hypercache` and `./hypercache-admin`) plus a lightweight Python package (`client-py`).

### Option A: Automated One-Step Install (`./install.sh`)

The repository includes an automated bootstrapper that installs system build tools, the Rust toolchain (`cargo`), a Python virtual environment (`.venv`), the `hypercache` Python client, optional NVIDIA GPUDirect Storage (`kvikio`) bindings if a GPU is detected, and compiles the release binaries:

```bash
git clone https://github.com/your-org/hypercache.git
cd hypercache

# Run full automated setup (Ubuntu, Debian, RHEL/Rocky, Arch, macOS)
./install.sh
```

### Option B: Manual Build (`./build.sh` + `pip`)

If you already have **Rust (1.75+)** and **Python (3.9+)** installed:

#### 1. Compile the Native Rust Binaries
```bash
# Compiles release binaries and copies ./hypercache and ./hypercache-admin to the repo root
./build.sh

# Or build directly with Cargo:
cargo build --release --manifest-path control-plane/Cargo.toml
cp control-plane/target/release/hypercache ./hypercache
cp control-plane/target/release/hypercache-admin ./hypercache-admin
```

* **`./hypercache` (Worker Node Daemon & Local CLI)**: Runs on each GPU/CPU compute node to manage local RAM and NVMe pools, serve local Python processes over Unix Domain Sockets, run host readiness checks, and export Prometheus metrics.
* **`./hypercache-admin` (Cluster Controller & Fleet CLI)**: Runs on the Controller host (or operator terminal) to track cluster topology and dispatch fleet-wide preload, dataset sharding, eviction, and purge commands.

#### 2. Install the Python Data Plane & Integrations
```bash
# Install into your active Python environment or virtualenv
pip install -e ./client-py

# Optional: On NVIDIA GPU nodes, install GPUDirect Storage (cuFile / KvikIO) support
pip install --extra-index-url https://pypi.nvidia.com kvikio-cu12 cupy-cuda12x
```

---

## How to Use HyperCache

You can run HyperCache in **Standalone Single-Node Mode** (on a single GPU server or workstation with zero network ports open) or in **Multi-Node Cluster Mode** (coordinated across a fleet of GPU nodes).

### Step 1: Run Pre-Flight Host Diagnostics (`./hypercache check`)

Before starting the daemon, verify your machine's storage drives, permissions, `O_DIRECT` alignment support, and shared memory capacity:

```bash
./hypercache -c configs/hypercache-agent.toml check
```

> If your machine does not have unmounted raw block drives (for example, on a laptop or inside a standard Kubernetes pod), set `storage_backing_mode = "filesystem"` and `filesystem_cache_paths = ["/path/to/nvme/dir"]` in `configs/hypercache-agent.toml` to use `O_DIRECT` arena files on your existing filesystem.

---

### Step 2: Start the HyperCache Daemon

#### Mode A: Standalone Single-Node Mode (Default)
On a single GPU server, start the worker daemon. By default, it opens **zero network TCP ports** and communicates exclusively over `/run/hypercache/hypercache.sock` (or `/tmp/hypercache/hypercache.sock`):

```bash
# Start worker daemon with NVMe pool + Tier 1 Host RAM cache enabled
./hypercache -c configs/hypercache-agent.toml start --ram

# Optional: Also enable Prometheus metrics on http://127.0.0.1:9110/metrics
./hypercache -c configs/hypercache-agent.toml start --ram --metrics
```

#### Mode B: Multi-Node Cluster Mode (Controller + Worker Nodes)
When managing multiple GPU nodes:

1. **On the Central Controller Host**:
   ```bash
   ./hypercache-admin -c configs/hypercache-controller.toml controller
   ```
2. **On Each GPU Worker Node** (with `controller_url` set in `configs/hypercache-agent.toml`):
   ```bash
   ./hypercache -c configs/hypercache-agent.toml start --ram --agent
   ```
3. **Manage the Fleet from the Controller**:
   ```bash
   # View all registered GPU nodes, drive pools, and capacities
   ./hypercache-admin nodes

   # Pre-warm a model across all nodes (or filter by label with --selector "gpu=h100")
   ./hypercache-admin preload hypercache://s3/my-bucket/models/llama-3-70b/weights.safetensors --nodes all

   # Partition and pre-load a sharded training dataset across the cluster
   ./hypercache-admin preload-batch --prefix "s3://my-bucket/datasets/shard_" --sharded --nodes all
   ```

---

### Step 3: Manage & Benchmark Local Cache from the CLI

On any worker node, use `./hypercache` to inspect drives, preload weights, or benchmark your hardware's DMA throughput:

```bash
# Inspect auto-discovered NVMe drives and NUMA topology
./hypercache inspect

# Pre-load a model into cache and pin it so it is never evicted
./hypercache preload hypercache://s3/my-bucket/models/llama-3-8b/weights.pt --pin

# List all cached objects, their storage tier (RAM / NVMe), sync status, and pin state
./hypercache ls

# Benchmark Tier 1 (Host RAM) and Tier 2 (Striped NVMe) DMA read throughput on your hardware
./hypercache bench --tier ram --size-mb 2500
./hypercache bench --tier nvme --size-mb 5000
```

---

### Step 4: Use HyperCache in Your Python AI Code

#### 1. Drop-In PyTorch Model Loading (`torch.load`)

Importing `hypercache` automatically registers the `hypercache://` protocol with Python's `fsspec`:

```python
import fsspec
import torch
import hypercache  # Auto-registers the hypercache:// protocol

# Streams weights from local striped NVMe/RAM cache (auto-fetches from S3 on first miss)
checkpoint_uri = "hypercache://s3/my-bucket/models/llama-3-8b/weights.pt"

with fsspec.open(checkpoint_uri, "rb") as f:
    state_dict = torch.load(f, map_location="cuda:0")

model.load_state_dict(state_dict)
```

#### 2. Native Safetensors & vLLM Zero-Copy Weight Loader

Stream `.safetensors` files directly into GPU or host tensors with optional selective tensor filtering and cache pinning (`pinned=True`):

```python
from hypercache import (
    HyperCachevLLMWeightLoader,
    load_safetensors_state_dict,
    save_safetensors_state_dict,
)

# Load all (or selected) tensors from a .safetensors shard and pin in local cache
state_dict = load_safetensors_state_dict(
    "hypercache://models/llama-3-70b/model-00001-of-00008.safetensors",
    device="cuda:0",
    pinned=True,
)

# Stream multi-shard weights directly into a vLLM model runner
vllm_loader = HyperCachevLLMWeightLoader(pin_base_weights=True)
for name, tensor in vllm_loader.iter_shard_tensors(shard_uris, device="cuda:0"):
    pass
```

#### 3. Training Checkpointing: Async Write-Back vs. Sync Write-Through

```python
import fsspec
import torch
import hypercache

checkpoint_uri = "hypercache://s3/my-bucket/checkpoints/epoch_1.pt"

# Mode A: Asynchronous Write-Back (Default - Maximum Training Speed)
# Stages to local RAM/NVMe at hardware line rate and unblocks GPUs immediately;
# background worker flushes to S3/NFS asynchronously.
with fsspec.open(checkpoint_uri, "wb") as f:
    torch.save(model.state_dict(), f)

# Mode B: Synchronous Write-Through (Strict Remote Durability)
# Blocks f.close() until confirmed written to both local cache AND remote S3/NFS.
with fsspec.open(checkpoint_uri, "wb", write_through=True) as f:
    torch.save(model.state_dict(), f)
```

#### 4. PyTorch Distributed Checkpoint (FSDP / DeepSpeed / DCP)

Save and restore per-rank distributed checkpoints at local hardware speed:

```python
from hypercache import HyperCacheDCPStorageReader, HyperCacheDCPStorageWriter

# Save per-rank FSDP shard to local NVMe/RAM with background S3 sync
writer = HyperCacheDCPStorageWriter(
    "hypercache://s3/my-bucket/checkpoints/step_1000",
    rank=rank,
    world_size=world_size,
)
writer.save_rank_shard(model.state_dict())

# Restore per-rank FSDP shard via Direct Block DMA
reader = HyperCacheDCPStorageReader(
    "hypercache://s3/my-bucket/checkpoints/step_1000",
    rank=rank,
    world_size=world_size,
)
state_dict = reader.load_rank_shard()
```

#### 5. Layer-by-Layer Pipelined Model Loader (Compute / I/O Overlap)

Start forward-pass compute on Layer 0 immediately while subsequent layers prefetch in the background:

```python
from hypercache.pipeline import HyperCachePipelinedLoader

loader = HyperCachePipelinedLoader(prefetch_depth=2)

# Layer 0 arrives immediately; Layers 1..N prefetch in background threads during GPU compute
for layer_idx, layer_bytes in loader.stream_layers(layer_uris):
    run_layer_forward_pass(layer_idx, layer_bytes)
```

#### 6. KV-Cache Prefix Block Store (LMCache / Dynamo KVBM Ready)

Offload and retrieve prompt prefix KV-cache blocks across Tier 1 Host RAM and Tier 2 NVMe:

```python
from hypercache import HyperCacheKVStore

kv_store = HyperCacheKVStore(default_tier="ram")
block_key = kv_store.compute_prefix_hash(
    token_ids=[128000, 9906, 1917],
    model_id="llama-3.1-70b",
    layer_idx=0,
)

# Store KV-cache block into Tier 1 Host RAM (/dev/shm or HugePages)
kv_store.put_kv_block(block_key, kv_tensor, tier="ram", pinned=True)

# Retrieve KV-cache block directly into pre-allocated GPU tensor
kv_store.get_kv_block(block_key, out_tensor=gpu_kv_buffer)
```

#### 7. Multi-Node Dataset Sharding & Distributed DataLoader

Stream consistent-hashed dataset shards directly into PyTorch `DataLoader` workers on each node:

```python
import fsspec
from torch.utils.data import DataLoader, Dataset
import hypercache

class ShardedTokenDataset(Dataset):
    def __init__(self, shard_uris):
        self.shard_uris = shard_uris

    def __getitem__(self, idx):
        with fsspec.open(self.shard_uris[idx], "rb") as f:
            return f.read()

dataset = ShardedTokenDataset(["hypercache://datasets/tokens/shard_000.bin", ...])
loader = DataLoader(dataset, batch_size=16, num_workers=4)
```

---

## Fleet & Node Architecture

```
+===================================================================================+
|                                CENTRAL CONTROLLER HOST                            |
|                                                                                   |
|  [ Operator / Admin CLI ]                                                         |
|   ./hypercache-admin nodes                                                        |
|   ./hypercache-admin preload <uri> --nodes all                                    |
|   ./hypercache-admin preload-batch --prefix <uri> --sharded --nodes all           |
|         │                                                                         |
|         │ (Local Loopback http://127.0.0.1:9120 - Zero Config Admin Access)       |
|         ▼                                                                         |
|  +-----------------------------------------------------------------------------+  |
|  | HyperCache Controller Daemon (Port 9120)                                    |  |
|  | - Fleet Registry: Aggregates active nodes, NVMe capacities, and health      |  |
|  | - Fleet Dispatcher: Coordinates parallel preload / P2P fill / evict / purge |  |
|  +-----------------------------------------------------------------------------+  |
+===================================================================================+
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │ Authenticated Directives (Bearer Token)       │ Node Heartbeats (5s)
                 ▼                                               ▼
+=============================================+ +=============================================+
|             GPU WORKER NODE 01              | |             GPU WORKER NODE 02              |
|                                             | |                                             |
|  +---------------------------------------+  | |  +---------------------------------------+  |
|  | Agent Fleet & P2P API (Port 9115)     |◄─┼─┼─►| Agent Fleet & P2P API (Port 9115)     |  |
|  | - Preload/evict + P2P chunk/replica   |  | |  | - Preload/evict + P2P chunk/replica   |  |
|  +---------------------------------------+  | |  +---------------------------------------+  |
|  +---------------------------------------+  | |  +---------------------------------------+  |
|  | HyperCache Worker Engine              |  | |  | HyperCache Worker Engine              |  |
|  | - Moka W-TinyLFU Metadata + Slab Pool |  | |  | - Moka W-TinyLFU Metadata + Slab Pool |  |
|  | - Tier 1 RAM (/dev/shm) & Tier 2 NVMe |  | |  | - Tier 1 RAM (/dev/shm) & Tier 2 NVMe |  |
|  | - Direct Block DMA / GDS to GPU VRAM  |  | |  | - Direct Block DMA / GDS to GPU VRAM  |  |
|  | - Prometheus Exporter (Port 9110)     |  | |  | - Prometheus Exporter (Port 9110)     |  |
|  +---------------------------------------+  | |  +---------------------------------------+  |
+=============================================+ +=============================================+
```

---

## Complete CLI Reference

### 1. GPU Worker Node Agent CLI (`./hypercache`)
Runs locally on each GPU or CPU compute node:

| Command | Action |
| :--- | :--- |
| `./hypercache -c configs/hypercache-agent.toml check` | Runs pre-flight host diagnostics (config, permissions, raw drives, `O_DIRECT`, RAM pool, sockets). Alias: `doctor`, `preflight`. |
| `./hypercache -c configs/hypercache-agent.toml start` | Starts worker daemon in **secure local mode** (zero network ports opened by default). |
| `./hypercache -c configs/hypercache-agent.toml start --ram` | Starts worker daemon with **Tier 1 Host RAM cache enabled**. |
| `./hypercache -c configs/hypercache-agent.toml start --metrics` | Starts worker daemon with Prometheus metrics exporter enabled (`:9110`). |
| `./hypercache -c configs/hypercache-agent.toml start --agent` | Explicitly enables the Node Agent Fleet & P2P HTTP API (`:9115`). |
| `./hypercache -c configs/hypercache-agent.toml start --devices /dev/nvme1n1,/dev/nvme2n1` | Starts the daemon targeting explicit block devices (`manual` mode). |
| `./hypercache ls` | Lists local cached checkpoints, sizes, chunk counts, tier (`RAM` / `NVMe`), sync status, and pin state. |
| `./hypercache preload <uri> [--tier ram\|nvme] [--pin]` | Pre-loads a model from S3/NFS and optionally pins it against eviction. |
| `./hypercache preload --file <path> [--pin]` | Batch pre-loads multiple checkpoints listed in a text file. |
| `./hypercache rm <uri>` | Evicts a specific object and reclaims its 4 MiB RAM/NVMe blocks into the free pool. |
| `./hypercache refresh <uri>` | Evicts and re-ingests fresh weights from authoritative S3 or NFS. |
| `./hypercache purge` | Clears all cached models from RAM & NVMe and resets block allocators. |
| `./hypercache bench --tier ram --size-mb 1000` | Benchmarks Tier 1 (Host RAM) read throughput over memory bus. |
| `./hypercache bench --tier nvme --size-mb 5000` | Benchmarks Tier 2 (NVMe) parallel DMA read throughput and verifies SHA-256 parity. |
| `./hypercache inspect` | Scans and displays discovered physical/virtual SSDs and NUMA node mappings. |
| `./hypercache status` | Queries running daemon for active drives, hit ratio, and health. |

---

### 2. Central Fleet Administration CLI (`./hypercache-admin`)
Runs on the Controller host (zero configuration required on `127.0.0.1` loopback):

| Command | Action |
| :--- | :--- |
| `./hypercache-admin -c configs/hypercache-controller.toml controller` | Starts the **Central Cluster Controller / Fleet Coordinator** daemon on port 9120. |
| `./hypercache-admin nodes` | Displays cluster topology dashboard of all registered nodes, drives, capacities, and health. |
| `./hypercache-admin preload <uri> --nodes all` | Dispatches parallel pre-load (or P2P chunk distribution) across all online nodes. |
| `./hypercache-admin preload <uri> --selector "gpu=h100"` | Dispatches pre-load targeting specific node labels. |
| `./hypercache-admin preload-batch --prefix <PREFIX> --nodes all --sharded` | Partitions and pre-loads a sharded dataset across cluster nodes using consistent hashing. |
| `./hypercache-admin evict <uri> --nodes all` | Dispatches parallel model eviction across all nodes. |
| `./hypercache-admin purge --nodes all` | Dispatches cache purge and block allocator reset across all nodes. |
| `./hypercache-admin status` | Shows aggregate fleet metrics (nodes online, total NVMe pool, cache utilization). |

---

## Configuration Reference

### Controller Host Configuration (`configs/hypercache-controller.toml`)

```toml
# Bind controller service to private cluster network IP or 0.0.0.0
controller_bind_address = "0.0.0.0"
controller_port = 9120

# Cluster Security Token (Secures remote node registrations & fleet dispatch)
# Can also be set via: export HYPERCACHE_AUTH_TOKEN="sec-cluster-secret-key"
cluster_auth_token = "sec-cluster-secret-key"

# Optional P2P Fleet Preload (1/N S3 origin pull + peer-to-peer chunk exchange)
p2p_preload_enabled = false
```

### GPU Compute Node Configuration (`configs/hypercache-agent.toml`)

```toml
# Node Identification & Targeting Labels
node_id = "gpu-node-01"

[node_labels]
gpu = "h100"
pool = "training"
zone = "us-east-1a"

# Point to Central Controller private IP (omit for standalone single-node mode)
controller_url = "http://10.0.0.10:9120"
cluster_auth_token = "sec-cluster-secret-key"

# GPU Node Agent Fleet API (OFF by default; auto-enabled when controller_url is set)
agent_enabled = true
agent_bind_address = "0.0.0.0"
agent_http_port = 9115

# Prometheus Metrics Exporter (OFF by default for zero network exposure)
metrics_enabled = false
metrics_bind_address = "127.0.0.1"
metrics_port = 9110

# Authoritative Storage Backends ("s3" or "nfs")
authoritative_backend = "s3"
s3_endpoint = "https://s3.us-east-1.amazonaws.com"
s3_region = "us-east-1"
s3_bucket = "model-weights-bucket"
s3_access_key = "YOUR_ACCESS_KEY"
s3_secret_key = "YOUR_SECRET_KEY"

# Storage Backing Mode:
# - "auto": Uses raw block devices if available, otherwise falls back to filesystem arenas
# - "raw_block": Stripes directly across unmounted NVMe drives (/dev/nvme*, /dev/vd*)
# - "filesystem": Uses pre-allocated O_DIRECT arena files on ext4/XFS (for Kubernetes / GDS)
storage_backing_mode = "auto"
filesystem_cache_paths = []
filesystem_arena_size_bytes = 10737418240 # 10 GiB per arena file

# NVMe Discovery & Striping
device_discovery_mode = "auto"
allow_virtual_drives = true
include_partitions = true
device_prefixes = ["nvme", "vd", "xvd", "sd"]
exclude_root_device = true
chunk_size_bytes = 4194304 # 4 MiB chunks

# Checkpoint Write Semantics & Peer Replication
write_through_enabled = false # false = async write-back, true = sync write-through
peer_replication_factor = 1   # Set to 2 for k=2 peer checkpoint replication on CommitWrite

# Metadata Persistence (Async atomic JSON + binary manifest for < 20 ms warm-boot recovery)
metadata_persistence_enabled = true
metadata_persistence_path = "/var/lib/hypercache/metadata.json"

# Tier 1 Host RAM / CXL Shared Memory Acceleration (/dev/shm or HugePages)
ram_cache_enabled = false
ram_cache_capacity_bytes = 34359738368 # 32 GiB default (or 75% of /dev/shm)
ram_cache_shm_path = "/dev/shm/hypercache"
ram_cache_hugepages = false
ram_cache_hugepages_path = "/dev/hugepages/hypercache"
```

---

## Prometheus Metrics & ROI Telemetry (`:9110/metrics`)

When `--metrics` (or `metrics_enabled = true`) is enabled, point Prometheus or Grafana to `http://<node-ip>:9110/metrics`:

```promql
# Real-Time Hardware Throughput & Latency
hypercache_bandwidth_mbps
hypercache_local_metadata_latency_us
hypercache_remote_metadata_latency_us
hypercache_local_ttfb_us
hypercache_hit_ratio_percent

# Cloud Egress Savings & GPU Idle Time ROI
hypercache_estimated_cost_savings_dollars
hypercache_s3_egress_bytes_saved_total
hypercache_gpu_idle_time_saved_seconds

# Storage Pool & Cache Lifecycle
hypercache_active_devices_count
hypercache_storage_utilization_percent
hypercache_storage_used_bytes
hypercache_cached_objects_count
hypercache_evictions_total
```
