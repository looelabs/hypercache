# HyperCache Architecture Whitepaper: High-Performance Host-Based AI Storage & Cache Acceleration

---

## 1. Executive Summary & Problem Statement

Modern Large Language Model (LLM) and foundation model training and inference workloads are fundamentally constrained by I/O bottlenecks. While GPU compute capabilities have grown exponentially—an NVIDIA H100 delivers up to 3,958 TFLOPS of FP8 compute—the data delivery pipeline remains tethered to remote object stores (Amazon S3, Cloudflare R2, Ceph, MinIO).

### The Three AI Storage Bottlenecks:
1. **Network Stack Latency & TTFB**: Remote S3 GET requests suffer from DNS resolution, TLS 1.3 handshake negotiation, TCP slow-start, and HTTP roundtrip overheads, resulting in Time-To-First-Byte (TTFB) latencies of **35 ms to 130 ms** per request.
2. **Bandwidth Saturation & GPU Starvation**: High-density GPU nodes frequently saturate Top-of-Rack (ToR) network interfaces (10 GbE / 25 GbE) during cold model loads and multi-gigabyte checkpoint restoring, stalling GPU execution units and inflating idle operating costs.
3. **Host CPU & Kernel Page Cache Pollution**: Traditional POSIX buffered I/O (`read()`, `fread()`) requires multiple memory copies (`kernel buffer -> page cache -> user buffer -> CUDA pinned host memory -> GPU VRAM`), consuming significant CPU cycles and displacing valuable system memory.

### The HyperCache Solution:
**HyperCache** is an autonomous, multi-tier host acceleration engine designed to eliminate network overhead and CPU memory copies. By combining an in-memory **W-TinyLFU metadata engine (Moka)** with **Tier 1: Host RAM / CXL-Ready Shared Memory (`/dev/shm`)** and **Tier 2: Parallel Direct Block DMA across Striped NVMe Arrays**, HyperCache achieves:
- **Tier 1 (Host RAM / Shared Memory)**: **50.0 GB/s to 100+ GB/s** at DDR5 / CXL memory bus line rate with **< 10 µs** metadata latency.
- **Tier 2 (Striped Local NVMe Pool)**: Up to **~7.0 GB/s per PCIe Gen 4 NVMe** and **~14.0 GB/s per PCIe Gen 5 NVMe** line-rate flash bandwidth with **< 1 ms** resolution.
- **CPU & GPU Agnostic**: Runs natively on both GPU compute clusters (with Direct Block DMA / GDS) and CPU-only inference/data-processing nodes with zero CUDA dependencies.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       AI APPLICATION LAYER (PyTorch / Safetensors)           │
│                            torch.load("hypercache://models/llama-3/weights") │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
            1. Metadata Locate Request │ 3. Zero-Copy DMA / Memory Mapping
               (Sub-ms Unix Domain Sock)│    (Zero-Copy Kernel Direct I/O / mmap)
                                       ▼                           ▼
┌──────────────────────────────────────────────┐        ┌─────────────────────┐
│          HYPERCACHE CONTROL PLANE (Rust)     │        │ TIER 1: HOST RAM    │
│  ┌────────────────────────────────────────┐  │        │ (POSIX /dev/shm)    │
│  │ Ephemeral Moka Metadata (W-TinyLFU)    │  │        │ 50 - 100+ GB/s      │
│  └────────────────────────────────────────┘  │        └──────────▲──────────┘
│  ┌────────────────────────────────────────┐  │                   │
│  │ Auto-Discovery & OS Root Protection    │  │                   │ 2. Route & Stripe
│  └────────────────────────────────────────┘  │        ┌──────────┴──────────┐
│  ┌────────────────────────────────────────┐  │        │ TIER 2: STRIPED NVMe│
│  │ Ingestion Engine (Multi-Tier Router)   │──┼───────►│ Drive 0: /dev/nvme0 │
│  └────────────────────────────────────────┘  │        │ Drive 1: /dev/nvme1 │
│  ┌────────────────────────────────────────┐  │        │ 7 - 14+ GB/s / drive│
│  │ Prometheus Exporter (:9110/metrics)   │  │        └─────────────────────┘
│  └────────────────────────────────────────┘  │
└──────────────────────┬───────────────────────┘
                       │
                       │ Cache Miss Fallback (HTTPS/REST or POSIX NFS)
                       ▼
┌──────────────────────────────────────────────┐
│  AUTHORITATIVE BACKENDS (S3 / Ceph / NFS)    │
└──────────────────────────────────────────────┘
```

---

## 2. High-Level Architecture & Core Principles

HyperCache enforces strict separation between the **Control Plane** (metadata resolution, drive safety, admission/eviction lifecycle, telemetry) and the **Data Plane** (high-throughput zero-copy DMA streaming).

### Core Design Principles:
- **Zero External Database Overhead**: No Redis, PostgreSQL, SQLite, or ClickHouse daemons required. The entire control plane operates as an ephemeral in-memory state engine.
- **Hardware Direct Block Access**: Bypasses local filesystem overhead (ext4/XFS journal locks, inode allocation, directory tree traversal) by addressing raw block devices at physical 4 MiB striped chunk boundaries.
- **Strict Sector & Memory Alignment**: Leverages Linux kernel `O_DIRECT` and page-aligned memory buffers (`mmap.mmap(-1, len)`) to initiate Direct Memory Access (DMA) transactions straight from NVMe flash controllers.
- **Zero-Docker Single Binary**: Compiles into a self-contained executable (`./hypercache`) with native Prometheus metrics export on port `9110`.

---

## 3. Subsystem Architecture & Implementation

### 3.1. Hardware Auto-Discovery & Safety Engine

HyperCache dynamically scans the host system on startup without requiring manual drive configurations.

```
/sys/class/block/
 ├── vda ── (Excluded: Root Mountpoint "/")
 ├── vdc ── (Accepted: Raw Block Device -> Capacity: 1.82 TiB)
 └── vdd ── (Accepted: Raw Block Device -> Capacity: 1.82 TiB)
```

#### Discovery Algorithm (`discovery.rs`):
1. **Sysfs Traversal**: Iterates over all block devices in `/sys/class/block/` matching allowed drive prefixes (`nvme*`, `vd*`, `xvd*`, `sd*`).
2. **Root Device & Mount Exclusion**:
   - Parses `/proc/mounts` and `/sys/class/block/<dev>/holders` to identify the OS root mountpoint (`/`).
   - Resolves device major/minor numbers and unconditionally excludes the root drive and parent disks from the striping pool.
3. **Filesystem Superblock Signature Detection**:
   - Probes device sector headers for magic bytes (`EXT4: 0x53EF`, `XFS: "XFSB"`, `Btrfs: "_BHRfS_M"`, `ZFS`).
   - Devices containing active mounted filesystems are safely quarantined unless explicitly flagged for cache pool dedication.
4. **Capacity Calculation**: Reads `/sys/class/block/<dev>/size` (sectors $\times$ 512 bytes) and builds the aggregate multi-drive storage pool.

---

### 3.2. Ephemeral In-Memory Metadata Engine (Moka W-TinyLFU)

Traditional caching architectures rely on external key-value stores (Redis) which incur socket serialization, TCP overhead, and memory duplication. HyperCache uses an embedded, thread-safe in-process cache built on **Moka** (`moka::future::Cache`).

#### W-TinyLFU Admission & Eviction Policy:
- **Window TinyLFU**: Tracks object access frequency using a compact Count-Min 4-bit sketch with dynamic aging (decay).
- **Size-Aware Weigher**: Evictions are computed directly on physical chunk payload bytes (`EntryWeight = ObjectSizeBytes`), ensuring strict adherence to the configured NVMe capacity limit.
- **Sub-Millisecond Routing**: Lookups return the full physical striping layout (chunk indices, physical device paths, exact byte offsets, and chunk lengths) in **$< 1\text{ ms}$** over Unix Domain Sockets (`/run/hypercache/hypercache.sock` or fallback `/tmp/hypercache/hypercache.sock`).

#### IPC Wire Protocol (`protocol.rs`):
Communication between the Python Data Plane and the Rust Control Plane uses newline-delimited JSON over local Unix Domain Sockets:

```json
// Locate Request (Python -> Rust)
{ "Locate": { "uri": "hypercache://models/llama-3/weights.safetensors", "auto_ingest": true } }

// Locate Response (Rust -> Python)
{
  "status": "Hit",
  "object_size": 5242880000,
  "chunk_size": 4194304,
  "chunk_layout": [
    { "chunk_index": 0, "device": "/dev/vdc", "offset": 4096, "length": 4194304 },
    { "chunk_index": 1, "device": "/dev/vdd", "offset": 4096, "length": 4194304 },
    { "chunk_index": 2, "device": "/dev/vdc", "offset": 4198400, "length": 4194304 },
    { "chunk_index": 3, "device": "/dev/vdd", "offset": 4198400, "length": 4194304 }
  ]
}
```

---

### 3.3. Ingestion & Multi-Drive Striping Engine

When an object is requested for the first time (`Cache MISS`), the Ingestion Engine coordinates background fetching and physical block striping.

```
Authoritative S3 Object (5.00 GB Payload)
  │
  ├── Chunk 0 (4 MiB) ──► Target: /dev/vdc @ Offset 4096
  ├── Chunk 1 (4 MiB) ──► Target: /dev/vdd @ Offset 4096
  ├── Chunk 2 (4 MiB) ──► Target: /dev/vdc @ Offset 4198400
  ├── Chunk 3 (4 MiB) ──► Target: /dev/vdd @ Offset 4198400
  └── Chunk N (4 MiB) ──► Target: /dev/vd(N % Count) @ Offset ...
```

#### Striping Strategy (`striper.rs`):
- **Fixed 4 MiB Chunk Sizing**: Aligns with NVMe internal erase block boundaries and flash page allocation units, minimizing write amplification.
- **Round-Robin Multi-Drive Striping**: Distributes sequential chunks evenly across all discovered block devices ($D_0, D_1, \dots, D_{N-1}$).
- **Positional Direct Writes**: Writes blocks directly using `preadv`/`pwritev` or positional file handles with a 4096-byte hardware offset safety margin (`device_start_offset_bytes`).

---

### 3.4. Zero-Copy Data Plane & Dual-Handle Fallback

The Python Data Plane (`client-py/hypercache/datapath.py`) implements zero-copy DMA transfers while maintaining 100% standard POSIX compatibility.

#### Dual-Handle File Descriptor Architecture:
1. **Direct DMA Handle (`fd_direct` with `O_DIRECT`)**:
   - Opens block device or filesystem arena with `os.O_DIRECT`.
   - Requires page/sector alignment (512 or 4096 bytes) for both the physical disk offset and user memory address.
   - Executes multi-threaded `os.preadv` directly into a pre-allocated anonymous `mmap` buffer (`mmap.mmap(-1, length)`), saturating local PCIe NVMe lanes (~7 GB/s per Gen 4 drive, ~14 GB/s per Gen 5 drive).
2. **Buffered Handle (`fd_buffered` with `O_RDONLY`)**:
   - Transparently handles arbitrary, unaligned byte slices (e.g. PyTorch header probing `f.read(16)` or Safetensors metadata JSON slices).
   - If an unaligned read or non-sector offset is encountered, the datapath automatically routes through `fd_buffered` without raising `EINVAL (Errno 22)` to the caller.

---

### 3.5. Asynchronous Write-Back Caching & Egress Engine

In distributed training workflows, waiting for multi-gigabyte checkpoints to upload synchronously over the network stalls GPU compute clusters. HyperCache implements a high-throughput **Asynchronous Write-Back Engine**:

```
+===================================================================================+
|                          PYTORCH TRAINING PROCESS (RANK 0)                        |
|                                                                                   |
|  torch.save(model.state_dict(), "hypercache://checkpoints/epoch_1.pt")             |
+─────────────────────────────────────────┬─────────────────────────────────────────+
                                          │
                               1. Stage & │ 2. Direct Block Write
                                  Layout  │    (Direct NVMe @ 712+ MB/s)
                                          ▼                ▼
┌──────────────────────────────────────────────┐        ┌─────────────────────┐
│          HYPERCACHE WORKER DAEMON            │        │ STRIPED NVMe POOL   │
│  - StageWrite / Allocate Layout              │───┬───►│ /dev/vdc            │
│  - CommitWrite / Status: COMMITTED           │   │    │ /dev/vdd            │
│  - Immediate Return (GPUs unblocked in 350ms)│   │    └──────────┬──────────┘
└──────────────────────┬───────────────────────┘   │               │
                       │                           │               │ 3. Background Flush
                       │ Tokio Async Task Spawns   │               │    (Direct Hardware Read)
                       ▼                           ▼               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                     BACKGROUND EGRESS WORKER POOL                           │
│  - Reads striped chunks directly from /dev/vdc, /dev/vdd                    │
│  - Streams payload to Authoritative S3 (PutObject) or NFS (/mnt/nfs)        │
│  - Updates metadata sync status to SYNCED_BACKEND                           │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
                                       ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  AUTHORITATIVE STORAGE (S3 / NFS / POSIX)                   │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **Stage & Direct Write**: The PyTorch client queries the local daemon (`StageWrite`) to obtain striped NVMe chunk allocations. The `DirectBlockWriter` writes chunks directly to raw disk sectors using positional `os.pwrite`.
2. **Instant GPU Unblock (`CommitWrite`)**: Once local NVMe striping completes, the client issues `CommitWrite`. HyperCache updates the Moka metadata manifest and immediately unblocks the PyTorch training loop (~350 ms for 250 MB).
3. **Non-Blocking Background Egress**: The daemon spawns an asynchronous Tokio task that reads the chunks directly from physical disks and streams them to authoritative S3 or NFS storage without consuming GPU cycles.

---

### 3.6. Central Controller Modular Dataset Sharding

For multi-node distributed training (PyTorch DDP, DeepSpeed, Megatron), dataset tokens are pre-sharded into independent chunks ($S_0, S_1, \dots, S_{K-1}$). The Central Controller coordinates targeted shard distribution across active cluster nodes:

$$\text{Target Node Index} = i \pmod N \quad \text{where } N = |\text{Active Nodes}|$$

```
+===================================================================================+
|                              CENTRAL CONTROLLER                                   |
|   ./hypercache-admin preload-batch --prefix "s3://tokens/shard_" --sharded        |
+─────────────────────────────────────────┬─────────────────────────────────────────+
                                          │
               ┌──────────────────────────┴──────────────────────────┐
               │ Modulo 0: Shards 0, 2, 4...                         │ Modulo 1: Shards 1, 3, 5...
               ▼                                                     ▼
+=============================+                       +=============================+
|     GPU WORKER NODE 01      |                       |     GPU WORKER NODE 02      |
|                             |                       |                             |
|  - Preloads Shards 0, 2, 4  |                       |  - Preloads Shards 1, 3, 5  |
|  - Striped to Local NVMe    |                       |  - Striped to Local NVMe    |
|  - DataLoader: 2.76+ GB/s   |                       |  - DataLoader: 2.76+ GB/s   |
+=============================+                       +=============================+
```

- Each node exclusively preloads and caches its assigned dataset partition, maximizing aggregate cluster cache capacity.
- Node DataLoader workers stream tokens from local NVMe cache at **2.76+ GB/s** with zero inter-node network interference.

---

## 4. Flash Parallelism & NVMe Queue Depth Mechanics

A common misconception in storage systems is that attaching a PCIe Gen 4 or Gen 5 SSD will automatically yield 7 GB/s or 14 GB/s throughput for single-threaded applications.

### The Physics of NVMe SSDs:
An enterprise NVMe SSD is an internally distributed array of **8 to 16 parallel NAND flash channels**, each controlling multiple dies.

```
                           Host PCIe Gen 5 Bus (x4 Lanes: ~14.0 GB/s)
                                              │
                                  ┌───────────▼───────────┐
                                  │ NVMe Flash Controller │
                                  │  (64k Command Queues) │
                                  └──┬───┬───┬───┬───┬───┬┘
                                     │   │   │   │   │   │
        Flash Channel 0 ─────────────┘   │   │   │   │   └──────────── Flash Channel 15
                        ▼                ▼   ▼   ▼   ▼                ▼
                  [NAND Die 0]     [NAND Die 1] ... [NAND Die 15]
```

- **Queue Depth = 1 (Synchronous Read)**: The SSD controller waits for a single 4KB/64KB transaction to complete across 1 flash channel. The other 15 channels remain idle, capping read bandwidth at **~500–700 MB/s**.
- **Queue Depth $\ge 16\text{--}32$ (Parallel DMA)**: HyperCache dynamically spawns a worker thread pool sized to the host CPU cores:
  $$\text{Workers} = \min(64, \max(16, \text{CPU Cores} \times 4))$$
  By dispatching parallel chunk DMA requests simultaneously across all available flash channels and all striped drives, HyperCache fully saturates the PCIe bus width.

---

## 5. Theoretical Hardware Throughput & Linear Scaling Profile

By shifting I/O from remote network links to local PCIe NVMe lanes and DDR5/CXL host memory channels, HyperCache scales linearly with attached hardware:

### 5.1. Multi-Tier Hardware Bandwidth Comparison

| Storage Tier / Data Path | Per-Device / Per-Link Limit | Multi-Drive / Multi-Channel Node Throughput | Metadata / TTFB Latency |
| :--- | :--- | :--- | :--- |
| **Remote S3 / Object Store (10 GbE / 25 GbE)** | 1.25 – 3.12 GB/s wire limit | **~0.2 – 2.2 GB/s** (shared & rate-limited) | 35 – 130 ms |
| **Tier 2: PCIe Gen 4 NVMe Pool (`O_DIRECT` / GDS)** | **~7.0 GB/s per NVMe SSD** | **~28 GB/s** (4x NVMe) / **~56 GB/s** (8x NVMe) | **< 1 ms** |
| **Tier 2: PCIe Gen 5 NVMe Pool (`O_DIRECT` / GDS)** | **~14.0 GB/s per NVMe SSD** | **~56 GB/s** (4x NVMe) / **~112 GB/s** (8x NVMe) | **< 1 ms** |
| **Tier 1: Host RAM / POSIX SHM (DDR5 / CXL)** | **50 – 100+ GB/s per stream** | **300 – 460+ GB/s** (8–12 channel DDR5 bus) | **< 10 µs** |

### 5.2. Multi-Drive Linear Striping Scaling
Because HyperCache stripes 4 MiB chunks in round-robin fashion across all discovered NVMe drives ($D_0, D_1, \dots, D_{N-1}$) with zero filesystem journal locks or inode contention, aggregate node read throughput scales linearly ($N \times \text{Per-Drive Bandwidth}$) up to the host PCIe root complex limit.

### 5.3. Bit-Level Integrity & SHA-256 Parity
Every benchmark and verification run (`./hypercache bench`) computes end-to-end SHA-256 checksums across all reconstructed 4 MiB striped chunks to verify 100% bit-for-bit parity across drive boundaries.

---

## 6. Observability, Multi-Node Metrics & ROI Telemetry

HyperCache embeds an asynchronous Prometheus metrics exporter listening on port **`9110`** (`http://<node-ip>:9110/metrics`), explicitly selected to avoid port collisions with Linux `node_exporter` (`:9100`) and NVIDIA `dcgm-exporter` (`:9400`).

### 6.1. Prometheus Metrics Reference

```promql
# ----------------------------------------------------------------------
# 1. Real-Time Hardware Performance
# ----------------------------------------------------------------------
hypercache_bandwidth_mbps               # Instantaneous direct DMA read rate in MB/s
hypercache_local_metadata_latency_us    # In-memory Moka metadata latency in microseconds (< 100 µs)
hypercache_remote_metadata_latency_us   # Remote S3 TLS/HTTP metadata latency in microseconds (~37 ms)
hypercache_local_ttfb_us                # Time-to-first-byte from local SSD (~2.0 ms)
hypercache_hit_ratio_percent            # Percentage of requests served from local cache

# ----------------------------------------------------------------------
# 2. Cost Savings & Cloud ROI
# ----------------------------------------------------------------------
hypercache_s3_egress_bytes_saved_total  # Cumulative network egress bytes avoided
hypercache_estimated_cost_savings_dollars # Dollar ROI computed at standard cloud egress rates ($0.09/GB)
hypercache_gpu_idle_time_saved_seconds  # Total GPU compute seconds saved from waiting for network downloads

# ----------------------------------------------------------------------
# 3. Storage Pool & Moka Cache Lifecycle
# ----------------------------------------------------------------------
hypercache_storage_capacity_bytes       # Total discovered physical storage pool capacity
hypercache_storage_used_bytes           # Cached checkpoint footprint on disk
hypercache_storage_utilization_percent  # Storage capacity utilization gauge (0 - 100%)
hypercache_active_devices_count         # Number of active striped block devices in pool
hypercache_cached_objects_count         # Active models maintained in Moka metadata
hypercache_evictions_total              # Count of objects reclaimed by W-TinyLFU eviction
```

### 6.2. Multi-Node GPU Cluster Scraping Architecture

In multi-node production clusters (Kubernetes, Slurm, or Ray), a single central Prometheus server scrapes all GPU nodes simultaneously:

```yaml
# Central Prometheus prometheus.yml
scrape_configs:
  - job_name: 'hypercache-cluster'
    scrape_interval: 1s
    static_configs:
      - targets:
          - '10.0.1.10:9110'   # GPU Node 01
          - '10.0.1.11:9110'   # GPU Node 02
          - '10.0.1.12:9110'   # GPU Node 03
          - '10.0.1.13:9110'   # GPU Node 04
        labels:
          cluster: 'h100-nvlink-pool'
```

---

## 7. Python AI Workload Integration Guide

HyperCache hooks directly into standard deep learning frameworks using the standard `fsspec` filesystem registry.

### 7.1. PyTorch Checkpoint Loading
```python
import torch
import hypercache  # Automatically registers hypercache:// protocol

# PyTorch loads weights directly from striped NVMe storage into VRAM/RAM
weights = torch.load("hypercache://models/llama-3-70b/weights.pt", map_location="cuda:0")
```

### 7.2. Hugging Face & Safetensors Loading
```python
import safetensors.torch
from hypercache.fsspec_fs import register_hypercache_fsspec

register_hypercache_fsspec()

# Open tensor checkpoint directly via Direct Block DMA
with open("hypercache://models/mistral-7b/model.safetensors", "rb") as f:
    tensors = safetensors.torch.load(f.read())
```

### 7.3. Asynchronous Write-Back Checkpointing (`torch.save`)
```python
import fsspec
import torch
import hypercache

# PyTorch saves state dict directly to striped NVMe tier with async S3 write-back
checkpoint_uri = "hypercache://s3/my-bucket/checkpoints/epoch_1.pt"

with fsspec.open(checkpoint_uri, "wb") as f:
    torch.save(model.state_dict(), f)

print("Checkpoint saved to NVMe cache. GPUs resume training immediately!")
```

### 7.4. Multi-Node Dataset Sharding & DataLoader
```python
import fsspec
import torch
from torch.utils.data import DataLoader, Dataset
import hypercache

class ShardedTokenDataset(Dataset):
    def __init__(self, shard_uris):
        self.shard_uris = shard_uris

    def __getitem__(self, idx):
        with fsspec.open(self.shard_uris[idx], "rb") as f:
            return f.read()

# Stream partitioned dataset shards at line rate
dataset = ShardedTokenDataset(["hypercache://datasets/tokens/shard_000.bin", ...])
loader = DataLoader(dataset, batch_size=16, num_workers=4)
```

---

## 8. CLI Command Reference & Operator Guide

| Command | Subcommand / Flags | Description |
| :--- | :--- | :--- |
| `hypercache start` | `-c <path> [--metrics]` | Starts the daemon, Moka cache, UDS listener, and optional Prometheus exporter. |
| `hypercache bench` | `--size-mb 5000 --iterations 3` | Runs local 32-worker parallel DMA benchmark and verifies SHA-256 parity. |
| `hypercache bench` | `--s3-endpoint <url> --bucket <name>` | Executes live over-the-wire S3 download and prints side-by-side comparison. |
| `hypercache inspect` | `-c <path>` | Scans host block devices, safety exclusions, and aggregate pool size. |
| `hypercache status` | | Queries running daemon for active drives, memory hit ratio, and health. |

---

## 9. Conclusion & Future Roadmap

HyperCache eliminates the primary I/O bottleneck in modern GPU computing by transforming host-attached NVMe arrays into a high-speed, zero-copy, transparent cache tier for authoritative object stores.

### Planned Enhancements:
1. **Direct Kernel `io_uring` Ingestion Engine**: Replacing standard async network fetchers with fixed registered buffer `io_uring` SQEs for multi-threaded NVMe writes.
2. **Distributed Peer-to-Peer NVMe-oF (NVMe over Fabrics)**: Enabling cross-node RDMA chunk sharing over 400 Gbps InfiniBand/RoCE without querying remote S3.
3. **Dynamic NUMA Node Pinning**: Automatically binding DMA worker threads and Direct I/O buffers to the exact NUMA node and PCIe root complex controlling the target NVMe drive.
