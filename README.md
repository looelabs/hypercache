# HyperCache: Host-Local Storage & Memory Acceleration for AI Workloads

<img src="images/hypercache-logo.jpeg" alt="HyperCache Logo" width="400">

> **Looking for operator cluster setup instructions?** See the **[Getting Started Guide](GETTING_STARTED.md)** for a step-by-step walkthrough.  
> **Looking for deep systems internals?** See the **[Architecture Whitepaper](ARCHITECTURE.md)** for block layout, DMA mechanics, and subsystem design.

---

## Table of Contents

1. [What is HyperCache?](#what-is-hypercache)
2. [The Problem: Why AI Infrastructure Hits an I/O Wall](#the-problem-why-ai-infrastructure-hits-an-io-wall)
3. [Theoretical Hardware Bandwidth: Local Bus vs. Network Storage](#theoretical-hardware-bandwidth-local-bus-vs-network-storage)
4. [What Can You Use HyperCache For?](#what-can-you-use-hypercache-for)
5. [Hardware & System Requirements](#hardware--system-requirements)
6. [How to Build & Install HyperCache](#how-to-build--install-hypercache)
7. [Configuration Guide (`hypercache-agent.toml` & `hypercache-controller.toml`)](#configuration-guide-hypercache-agenttoml--hypercache-controllertoml)
8. [How to Use HyperCache (Single-Node, Cluster & Laptop Dev)](#how-to-use-hypercache-single-node-cluster--laptop-dev)
9. [URI Addressing & Backend Routing](#uri-addressing--backend-routing)
10. [Python & PyTorch Integration Examples](#python--pytorch-integration-examples)
11. [Fleet & Node Architecture](#fleet--node-architecture)
12. [Security, Network Ports & Authentication Model](#security-network-ports--authentication-model)
13. [Host RAM Footprint & Sizing Calculator](#host-ram-footprint--sizing-calculator)
14. [Prometheus & Grafana Monitoring Stack](#prometheus--grafana-monitoring-stack)
15. [Complete CLI Reference](#complete-cli-reference)
16. [Running Tests & Verification Suites](#running-tests--verification-suites)

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

## Hardware & System Requirements

### 1. Storage Drives: Raw Unformatted Block Devices (`NVMe` Recommended, `SATA SSD` Supported)
* **Raw Drives Without a Filesystem**: For HyperCache's native Raw Block Striping mode (`storage_backing_mode = "raw_block"` or `"auto"`) to claim and stripe across local drives, the target drives **must be raw, unmounted block devices with no existing filesystem on them**.
  * HyperCache automatically inspects sector headers for existing filesystem superblocks (`ext4`, `XFS`, `Btrfs`, `ZFS`, and Linux `swap`). Any drive containing an existing filesystem or mounted partition is automatically skipped by the safety guard to prevent accidental data loss.
  * To prepare a previously formatted drive for raw block striping, unmount it and wipe its filesystem signature:
    ```bash
    sudo wipefs -a /dev/nvme1n1
    ```
* **NVMe SSDs vs. SATA SSDs**:
  * **PCIe Gen 4 / Gen 5 NVMe SSDs (Recommended)**: HyperCache is architected for multi-queue PCIe NVMe drives (`/dev/nvme*`), delivering up to **~7 GB/s per PCIe Gen 4 NVMe** and **~14 GB/s per PCIe Gen 5 NVMe** when striped across local drives.
  * **SATA SSDs (Supported at Lower Speeds)**: HyperCache can also discover and stripe across standard SATA SSDs (`/dev/sd*`). However, because SATA III is physically limited to a 6 Gbps interface (**~500–550 MB/s per drive**), SATA SSDs will not achieve NVMe-class throughput.

---

### 2. Supported Server Platforms & Reference Systems

HyperCache is hardware-vendor agnostic and runs on **any standard `x86_64` Linux server, GPU workstation, or cloud VM** equipped with local NVMe or SATA drives.

In production data centers, it is specifically designed to pool the high-density local NVMe bays (`8x` to `16x` `E1.S/L`, `E3.S/L`, `U.2`, or `M.2` drives) built into modern 8-GPU enterprise AI servers. Below are common reference platforms where HyperCache is deployed:

| Server Vendor | System Model | GPU Platform | Host CPU Architecture |
| :--- | :--- | :--- | :--- |
| **Supermicro (SMC)** | `SYS-821GE-TNHR` | NVIDIA HGX H100 / H200 (8-GPU) | Dual Intel Xeon (`x86_64`) |
| **Supermicro (SMC)** | `AS-8125GS-TNMR2` | NVIDIA HGX H100 / H200 (8-GPU) | Dual AMD EPYC (`x86_64`) |
| **Supermicro (SMC)** | `SYS-A22GA-NBRT` | NVIDIA HGX B200 / B300 (8-GPU) | Dual Intel Xeon (`x86_64`) |
| **Supermicro (SMC)** | `AS-A126GS-TNBR` | NVIDIA HGX B200 / B300 (8-GPU) | Dual AMD EPYC (`x86_64`) |
| **Dell PowerEdge** | `PowerEdge XE9680` | NVIDIA HGX H100 / H200 (8-GPU) | Dual Intel Xeon (`x86_64`) |
| **Dell PowerEdge** | `PowerEdge XE9780` | NVIDIA HGX B200 / B300 (8-GPU) | Dual Intel Xeon (`x86_64`) |
| **Dell PowerEdge** | `PowerEdge XE9785` | NVIDIA HGX B200 / B300 (8-GPU) | Dual AMD EPYC (`x86_64`) |

> **Note**: The systems listed above are representative examples, not an exclusive compatibility list. Any custom build, OEM server (ASUS, Gigabyte, Lenovo, HPE), or cloud GPU instance with attached local NVMe/SSD storage is supported.

---

### 3. CPU Architecture Support (`x86_64` Tested vs. `ARM64` Pending)

* **`x86_64` (Intel Xeon & AMD EPYC) — Tested & Supported**:
  HyperCache has been tested and validated on `x86_64` Linux servers and cloud instances.
* **`ARM64` (`aarch64` / NVIDIA Grace) — Untested / Pending Validation**:
  HyperCache has **not yet been tested on ARM systems**. Specifically, **NVIDIA GB200 / GB300 / NVL72** and **Vera Rubin** rack-scale architectures still require hardware testing and validation.

---

### 4. Operating System & Software Prerequisites
* **Operating System**: Linux (`x86_64`, Kernel 5.15+ recommended for optimal `O_DIRECT`, HugePages, and NUMA sysfs discovery).
* **Permissions**: Root privileges (`sudo`) or `CAP_SYS_RAWIO` + `disk` group read/write access on target `/dev/nvme*` block devices (verify anytime with `./hypercache check`).
* **Build & Runtime Toolchains**: Rust `1.75+` (`cargo`) and Python `3.9+` (`pip`, `venv`), both installed automatically if you run `./install.sh`.

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

## Configuration Guide (`hypercache-agent.toml` & `hypercache-controller.toml`)

All HyperCache configuration files live in the [`configs/`](configs/) directory and use strongly-typed TOML syntax. Passing `-c <path>` to `./hypercache` or `./hypercache-admin` loads the specified configuration file.

### Which Config File Should I Use?

| Configuration File | Where It Runs | Binary Used | Purpose |
| :--- | :--- | :--- | :--- |
| **[`configs/hypercache-agent.toml`](configs/hypercache-agent.toml)** | **Every GPU / CPU Worker Node** | `./hypercache` | Configures local storage discovery (raw NVMe or filesystem arenas), Tier 1 Host RAM caching, authoritative S3/NFS credentials, write-back/write-through behavior, and optional registration with a Central Controller. |
| **[`configs/hypercache-controller.toml`](configs/hypercache-controller.toml)** | **Central Controller / Head Node** *(Multi-node clusters only)* | `./hypercache-admin` | Configures the cluster coordinator bind IP/port (`9120`), shared Bearer token authentication (`cluster_auth_token`), and Peer-to-Peer fleet distribution (`p2p_preload_enabled`). Not needed for single-node setups. |
| **[`configs/hypercache.toml`](configs/hypercache.toml)** | **Local Dev / Laptop / CI** | `./hypercache` | All-in-one development config with `auto_create_mock_devices = true` (automatically creates 4x 100 MiB mock NVMe files in `/tmp/hypercache_nvme*.raw` so you can test on any laptop without dedicated SSDs). |

---

### 1. Choosing Your Storage Backing Mode (`configs/hypercache-agent.toml`)

HyperCache supports three ways to back your Tier 2 NVMe cache pool depending on your environment:

#### A. Raw Block Device Mode (`storage_backing_mode = "raw_block"` or `"auto"`)
Best for **bare-metal GPU servers or Cloud VMs** with dedicated, unmounted local NVMe drives (e.g., `/dev/nvme1n1`, `/dev/nvme2n1`, `/dev/vdc`). HyperCache automatically discovers unmounted drives, excludes your OS root disk (`/`), and stripes 4 MiB chunks directly across raw disk sectors with zero filesystem overhead.
```toml
storage_backing_mode = "auto"
device_discovery_mode = "auto" # Or set "manual" and block_devices = ["/dev/nvme1n1", "/dev/nvme2n1"]
exclude_root_device = true
```

#### B. Filesystem Direct I/O Arena Mode (`storage_backing_mode = "filesystem"`)
Best for **Kubernetes pods (`emptyDir` / `local-storage` PVCs), shared servers where drives are already mounted with ext4/XFS, or native NVIDIA GPUDirect Storage (`cuFile`)**. HyperCache pre-allocates aligned arena files inside the specified directories and opens them with `O_DIRECT`—requiring zero raw block privileges.
```toml
storage_backing_mode = "filesystem"
filesystem_cache_paths = ["/mnt/nvme0/hypercache", "/mnt/nvme1/hypercache"]
filesystem_arena_size_bytes = 107374182400 # 100 GiB arena capacity per directory
```

#### C. Local Laptop / CI Mock Mode (`auto_create_mock_devices = true`)
Best for **testing on a laptop or CI runner** with no extra disks.
```toml
auto_create_mock_devices = true
dev_mock_device_pattern = "/tmp/hypercache_nvme*.raw"
dev_mock_auto_count = 4
mock_device_size_bytes = 104857600 # 100 MiB per mock drive
```

---

### 2. Worker Node Agent Configuration (`configs/hypercache-agent.toml`)

Below is a complete annotated example of [`configs/hypercache-agent.toml`](configs/hypercache-agent.toml) showing the settings you configure on a GPU worker node:

```toml
# ==============================================================================
# 1. STORAGE POOL & TIERING
# ==============================================================================
storage_backing_mode = "auto"              # "auto", "raw_block", or "filesystem"
device_discovery_mode = "auto"             # "auto" (scan /sys/class/block) or "manual"
allow_virtual_drives = true                # Include cloud virtual NVMe disks (vd*, xvd*)
include_partitions = true                  # Include unmounted partitions
device_prefixes = ["nvme", "vd", "xvd", "sd"]
exclude_root_device = true                 # Always protect OS root disk (/)
chunk_size_bytes = 4194304                 # 4 MiB stripe chunk size
device_start_offset_bytes = 4096           # 4 KB alignment safety margin

# If using storage_backing_mode = "filesystem", specify mounted XFS/ext4 directories:
filesystem_cache_paths = []
filesystem_arena_size_bytes = 10737418240  # 10 GiB per arena file

# Tier 1 Host RAM / CXL Shared Memory Acceleration (/dev/shm or HugePages)
ram_cache_enabled = false                  # Set true (or pass --ram) for 50-100+ GB/s RAM tier
ram_cache_capacity_bytes = 34359738368     # 32 GiB (or auto-sizes to 75% of /dev/shm if omitted)
ram_cache_shm_path = "/dev/shm/hypercache"
ram_cache_hugepages = false                # Set true to back Tier 1 with /dev/hugepages
ram_cache_hugepages_path = "/dev/hugepages/hypercache"

# Dual-Tier Metadata Persistence (Async JSON + binary manifest for < 20 ms warm-boot recovery)
metadata_persistence_enabled = true
metadata_persistence_path = "/var/lib/hypercache/metadata.json"

# ==============================================================================
# 2. AUTHORITATIVE REMOTE STORAGE (S3 / NFS)
# ==============================================================================
authoritative_backend = "s3"               # Default backend: "s3" or "nfs"
s3_endpoint = "https://s3.us-east-1.amazonaws.com"
s3_region = "us-east-1"
s3_bucket = "model-weights-bucket"
s3_access_key = "YOUR_ACCESS_KEY"          # Or set via env: export AWS_ACCESS_KEY_ID="..."
s3_secret_key = "YOUR_SECRET_KEY"          # Or set via env: export AWS_SECRET_ACCESS_KEY="..."
allow_synthetic_s3_fallback = false        # Keep false in production so S3 errors fail fast

# Used when authoritative_backend = "nfs" or URI starts with file:// or nfs://
nfs_mount_path = "/mnt/nfs"

# Checkpoint Write Semantics & Fault Tolerance
write_through_enabled = false              # false = async Write-Back, true = sync Write-Through
peer_replication_factor = 1                # Set to 2 for k=2 peer replica before CommitWrite returns

# ==============================================================================
# 3. MULTI-NODE CLUSTER REGISTRATION (Optional - Omit for Single-Node Mode)
# ==============================================================================
node_id = "gpu-node-01"                    # Defaults to system hostname if omitted
# controller_url = "http://10.0.0.10:9120" # Uncomment to register with Central Controller
# cluster_auth_token = "your-cluster-token"# Or set via env: export HYPERCACHE_AUTH_TOKEN="..."

agent_enabled = false                      # Auto-enabled when controller_url is set (or pass --agent)
agent_bind_address = "0.0.0.0"
agent_http_port = 9115

[node_labels]                              # Optional labels for --selector targeting
gpu = "h100"
pool = "training"

# ==============================================================================
# 4. PROMETHEUS TELEMETRY (Optional - Off by Default)
# ==============================================================================
metrics_enabled = false                    # Set true (or pass --metrics) to expose :9110/metrics
metrics_bind_address = "127.0.0.1"
metrics_port = 9110
```

---

### 3. Central Controller Configuration (`configs/hypercache-controller.toml`)

If you are running a multi-node GPU cluster, deploy [`configs/hypercache-controller.toml`](configs/hypercache-controller.toml) on your head/coordinator node:

```toml
# IP address and port to bind the Central Controller service on
controller_bind_address = "0.0.0.0"
controller_port = 9120

# Shared secret Bearer token for cluster security
# Worker nodes must provide this token to register, and remote CLI calls require it.
# Local CLI calls from 127.0.0.1 on the controller host bypass authentication automatically.
# Can also be set via: export HYPERCACHE_AUTH_TOKEN="your-cluster-token"
cluster_auth_token = "your-cluster-token"

# Peer-to-Peer (P2P) Fleet Pre-Warming
# When true, fleet-wide preload pulls 1/N of chunks from S3 per node and distributes
# the remaining chunks peer-to-peer across worker nodes to eliminate S3 egress storms.
p2p_preload_enabled = false
```

---

### 4. Master Configuration Key & Environment Variable Reference

Every critical setting in `hypercache-agent.toml` and `hypercache-controller.toml` can also be overridden via CLI flags or environment variables:

| Config Key | Env Variable Override | Default | Description |
| :--- | :--- | :--- | :--- |
| `socket_path` | `HYPERCACHE_SOCKET` | `"/run/hypercache/hypercache.sock"` | Local Unix Domain Socket path (falls back to `/tmp/hypercache/hypercache.sock`). |
| `storage_backing_mode` | -- | `"auto"` | `"auto"`, `"raw_block"`, or `"filesystem"` (`O_DIRECT` arena files). |
| `filesystem_cache_paths` | -- | `[]` | List of mounted ext4/XFS directories for `O_DIRECT` arena files. |
| `filesystem_arena_size_bytes`| -- | `10737418240` (10 GiB) | Pre-allocated size per filesystem arena file in bytes. |
| `device_discovery_mode` | -- | `"auto"` | `"auto"` (scan `/sys/class/block`) or `"manual"` (use `block_devices` list). |
| `block_devices` | -- | `[]` | Explicit list of block devices when `device_discovery_mode = "manual"`. |
| `exclude_root_device` | -- | `true` | Unconditionally exclude OS root drive (`/`) from the storage pool. |
| `chunk_size_bytes` | -- | `4194304` (4 MiB) | Stripe chunk size in bytes across RAM and NVMe tiers. |
| `ram_cache_enabled` | `RAM_CACHE_ENABLED` | `false` | Enable Tier 1 Host RAM / CXL shared memory tier (`--ram`). |
| `ram_cache_capacity_bytes` | `RAM_CAPACITY_BYTES` | `34359738368` (32 GiB) | Maximum Tier 1 RAM cache size (or 75% of `/dev/shm` if omitted). |
| `ram_cache_hugepages` | -- | `false` | Use Linux `hugetlbfs` (`/dev/hugepages`) for Tier 1 memory blocks. |
| `authoritative_backend` | -- | `"s3"` | Default remote storage backend (`"s3"` or `"nfs"`). |
| `s3_endpoint` | `S3_ENDPOINT` | `"http://127.0.0.1:9002"`| S3 or S3-compatible endpoint URL. |
| `s3_region` | `AWS_DEFAULT_REGION` | `"us-east-1"` | S3 bucket region. |
| `s3_bucket` | `S3_BUCKET` | `"hypercache-authoritative"`| Default S3 bucket name when not specified in the URI. |
| `s3_access_key` | `AWS_ACCESS_KEY_ID` | `""` | S3 access key ID. |
| `s3_secret_key` | `AWS_SECRET_ACCESS_KEY` | `""` | S3 secret access key. |
| `nfs_mount_path` | -- | `"/mnt/nfs"` | Root mount path for NFS / POSIX shared filesystem backend. |
| `write_through_enabled` | -- | `false` | `false` = async Write-Back; `true` = synchronous Write-Through to S3/NFS. |
| `peer_replication_factor` | -- | `1` | Set to `2` to enable $k=2$ peer checkpoint replication on `CommitWrite`. |
| `controller_url` | `HYPERCACHE_CONTROLLER_URL`| `None` | URL of Central Controller (e.g. `"http://10.0.0.10:9120"`). |
| `cluster_auth_token` | `HYPERCACHE_AUTH_TOKEN` | `None` | Shared secret Bearer token for Controller <-> Worker Agent security. |
| `p2p_preload_enabled` | -- | `false` | Enable 1/N origin pull + peer-to-peer chunk exchange during fleet preload. |
| `metrics_enabled` | -- | `false` | Enable Prometheus HTTP exporter on `metrics_port` (`9110`). |

---

## How to Use HyperCache (Single-Node, Cluster & Laptop Dev)

### Step 1: Run Pre-Flight Host Diagnostics (`./hypercache check`)

Before starting the daemon, run `check` (or `doctor`) to verify your config syntax, drive permissions, `O_DIRECT` alignment support, and shared memory capacity:

```bash
./hypercache -c configs/hypercache-agent.toml check
```

If any check fails or warns, `./hypercache check` prints the exact remediation command (such as enabling `storage_backing_mode = "filesystem"` if your machine has no unmounted raw drives).

---

### Step 2: Start the HyperCache Daemon

#### Scenario A: Local Laptop / Workstation Testing (No Dedicated NVMe Required)
Use [`configs/hypercache.toml`](configs/hypercache.toml) (which automatically creates 4 mock NVMe files in `/tmp`):
```bash
./hypercache -c configs/hypercache.toml start --ram
```

#### Scenario B: Single-Node Production GPU Server
On a standalone GPU server, edit [`configs/hypercache-agent.toml`](configs/hypercache-agent.toml) with your S3/NFS settings and start the daemon. By default, it opens **zero network TCP ports** and listens only on the local Unix Domain Socket:
```bash
# Start worker daemon with NVMe pool + Tier 1 Host RAM cache enabled
./hypercache -c configs/hypercache-agent.toml start --ram

# Optional: Also enable Prometheus metrics on http://127.0.0.1:9110/metrics
./hypercache -c configs/hypercache-agent.toml start --ram --metrics
```

#### Scenario C: Multi-Node GPU Cluster (Central Controller + Worker Fleet)
1. **On the Central Controller Host**:
   ```bash
   ./hypercache-admin -c configs/hypercache-controller.toml controller
   ```
2. **On Each GPU Worker Node** (with `controller_url` and `cluster_auth_token` set in `configs/hypercache-agent.toml`):
   ```bash
   ./hypercache -c configs/hypercache-agent.toml start --ram --agent
   ```
3. **Orchestrate the Fleet from the Controller**:
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

## URI Addressing & Backend Routing

HyperCache routes requests automatically based on the URI scheme passed to Python (`fsspec.open`, `torch.load`) or the CLI (`./hypercache preload`):

| URI Format | Target Backend | How HyperCache Resolves It |
| :--- | :--- | :--- |
| `hypercache://s3/<bucket>/<key>` | **Amazon S3 / S3-Compatible** | Fetches/writes `<key>` in explicit `<bucket>` using `s3_endpoint` from `hypercache-agent.toml`. |
| `s3://<bucket>/<key>` | **Amazon S3 / S3-Compatible** | Standard S3 URI alias; routes identically to `hypercache://s3/<bucket>/<key>`. |
| `hypercache://<key>` | **Default Configured Backend** | Uses `authoritative_backend` (`"s3"` + default `s3_bucket`, or `"nfs"` + `nfs_mount_path`) from `hypercache-agent.toml`. |
| `hypercache://file/<abs/path>` | **POSIX / Shared NFS** | Reads/writes directly to `/<abs/path>` on the local or NFS-mounted filesystem. |
| `file:///<abs/path>` or `nfs://<rel/path>` | **POSIX / Shared NFS** | Reads/writes to `/<abs/path>` (or `<nfs_mount_path>/<rel/path>`). |

---

## Python & PyTorch Integration Examples

### 1. Drop-In PyTorch Model Loading (`torch.load`)

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

### 2. Native Safetensors & vLLM Zero-Copy Weight Loader

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

### 3. Training Checkpointing: Async Write-Back vs. Sync Write-Through

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

### 4. PyTorch Distributed Checkpoint (FSDP / DeepSpeed / DCP)

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

### 5. Layer-by-Layer Pipelined Model Loader (Compute / I/O Overlap)

Start forward-pass compute on Layer 0 immediately while subsequent layers prefetch in the background:

```python
from hypercache.pipeline import HyperCachePipelinedLoader

loader = HyperCachePipelinedLoader(prefetch_depth=2)

# Layer 0 arrives immediately; Layers 1..N prefetch in background threads during GPU compute
for layer_idx, layer_bytes in loader.stream_layers(layer_uris):
    run_layer_forward_pass(layer_idx, layer_bytes)
```

### 6. KV-Cache Prefix Block Store (LMCache / Dynamo KVBM Ready)

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

### 7. Multi-Node Dataset Sharding & Distributed DataLoader

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

## Security, Network Ports & Authentication Model

HyperCache is designed with a **zero-exposure default security posture**:

1. **Zero Open Ports in Single-Node Mode**:
   Running `./hypercache -c configs/hypercache-agent.toml start` without `controller_url` or `--metrics` opens **zero TCP network ports**. All communication between Python and the Rust daemon occurs over the local Unix Domain Socket (`/run/hypercache/hypercache.sock`).
2. **Zero-Config Localhost Admin**:
   Running `./hypercache-admin` directly on the Controller host (`127.0.0.1` / `::1`) automatically bypasses token checks so operators can run `./hypercache-admin nodes` or `preload` without flags.
3. **Bearer Token Authentication for Cluster Traffic**:
   When `cluster_auth_token` (or `HYPERCACHE_AUTH_TOKEN`) is set, all network requests between the Controller (`:9120`) and Worker Node Agents (`:9115`)—including heartbeats, preload directives, P2P chunk transfers, and $k=2$ checkpoint replication—require `Authorization: Bearer <token>`. Unauthorized requests are rejected with `HTTP 401 Unauthorized`.

| Component / Service | Default Port / Socket | Default State | Who Should Have Access |
| :--- | :--- | :--- | :--- |
| **Worker Local IPC** | `/run/hypercache/hypercache.sock` | **ON** (Local file socket) | Local processes on the compute node |
| **Prometheus Exporter** | TCP `:9110` (`127.0.0.1`) | **OFF** (Enable via `--metrics`) | Prometheus scraper / `localhost` |
| **Worker Agent Fleet & P2P API** | TCP `:9115` (`0.0.0.0`) | **OFF** (Auto-enabled with `controller_url`)| Central Controller & peer GPU worker nodes (Private VPC) |
| **Central Controller API** | TCP `:9120` (`0.0.0.0`) | **ON** when running `hypercache-admin controller` | GPU worker nodes & operator CLI (Private VPC) |

---

## Host RAM Footprint & Sizing Calculator

A common question from infrastructure engineers is: *"How much host system RAM does the HyperCache daemon itself consume?"*

When Tier 1 Host RAM caching is disabled (`ram_cache_enabled = false`), the Rust control plane and Moka metadata catalog have a tiny memory footprint—roughly **~125 MB of host RAM to index a 4 TB NVMe pool (~0.003% metadata-to-storage overhead)**:
* **Rust Control Plane Base RSS**: `~28 MB`
* **Moka W-TinyLFU + 4 MiB Chunk Routing Table (per 4 TB NVMe)**: `~96 MB` (`1,048,576 chunks x 80 bytes` + frequency sketch)
* **Transient Streaming Ring Buffers**: `~16 MB` (`4 concurrent streams x 4 MiB`)

You can calculate the exact memory footprint for your specific drive sizes, model counts, and multi-tenant container count using the included sizing tool:

```bash
python3 benchmarks/hypercache_ram_calculator.py --nvme-tb 16.0 --ram-tier-gb 32.0 --tenants 4
```

---

## Prometheus & Grafana Monitoring Stack

HyperCache includes a pre-configured **Prometheus + Grafana** observability stack in [`docker-compose.yml`](docker-compose.yml) and [`configs/grafana/`](configs/grafana/).

```bash
# 1. Start the worker daemon with Prometheus metrics enabled (:9110/metrics)
./hypercache -c configs/hypercache-agent.toml start --metrics

# 2. Launch the pre-provisioned Prometheus (:9090) and Grafana (:3000) dashboard stack
docker compose up -d
```

Exported Prometheus metrics on `http://<node-ip>:9110/metrics`:

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

## Running Tests & Verification Suites

```bash
# 1. Run the Rust Control Plane unit & integration test suite (27 tests)
cargo test --manifest-path control-plane/Cargo.toml

# 2. Run the Python Data Plane & framework integration test suite (21 tests)
.venv/bin/pytest client-py/tests

# 3. Run end-to-end PyTorch verification & endurance suites (with daemon running)
.venv/bin/python benchmarks/real_world_ai_test_suite.py
.venv/bin/python benchmarks/verify_writethrough_and_writeback.py
.venv/bin/python benchmarks/verify_advanced_optimizations.py
```
