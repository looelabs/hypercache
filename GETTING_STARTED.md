# HyperCache Quickstart & Operator Guide

HyperCache provides host-based caching and acceleration for distributed AI training and inference clusters. It eliminates model loading bottlenecks from S3 and NFS by caching weights across local striped NVMe/SSD pools and streaming directly into GPU memory via Direct Block DMA / GPUDirect Storage (GDS).

---

## System Architecture

```
+===================================================================================+
|                                CENTRAL CONTROLLER HOST                            |
|                                                                                   |
|  [ Operator / Admin CLI ]                                                         |
|   ./hypercache-admin nodes                                                        |
|   ./hypercache-admin preload <uri> --nodes all                                    |
|   ./hypercache-admin purge --nodes all                                            |
|         │                                                                         |
|         │ (Local Loopback http://127.0.0.1:9120 - Zero Config Admin Access)       |
|         ▼                                                                         |
|  +-----------------------------------------------------------------------------+  |
|  | HyperCache Controller Daemon (Port 9120)                                    |  |
|  | - Fleet Registry: Tracks active nodes, NVMe capacities, and health           |  |
|  | - Fleet Dispatcher: Coordinates parallel preload / evict / purge           |  |
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
|  | Agent Fleet API (Port 9115)           |  | |  | Agent Fleet API (Port 9115)           |  |
|  | - Receives preload/evict directives   |  | |  | - Receives preload/evict directives   |  |
|  +---------------------------------------+  | |  +---------------------------------------+  |
|  +---------------------------------------+  | |  +---------------------------------------+  |
|  | HyperCache Worker Engine              |  | |  | HyperCache Worker Engine              |  |
|  | - Moka W-TinyLFU Metadata in RAM      |  | |  | - Moka W-TinyLFU Metadata in RAM      |  |
|  | - Striped Raw NVMe Pool (/dev/nvme*)  |  | |  | - Striped Raw NVMe Pool (/dev/nvme*)  |  |
|  | - Direct Block DMA / GDS to GPU VRAM  |  | |  | - Direct Block DMA / GDS to GPU VRAM  |  |
|  | - Prometheus Exporter (Port 9110)     |  | |  | - Prometheus Exporter (Port 9110)     |  |
|  +---------------------------------------+  | |  +---------------------------------------+  |
+=============================================+ +=============================================+
```

---

## 1. Quick Installation & Build

Run on all machines (Controller and GPU worker nodes):

```bash
git clone https://github.com/looelabs/hypercache.git
cd hypercache

# Build native Linux release binaries (produces ./hypercache and ./hypercache-admin)
./build.sh
```

---

## 2. Setting Up the Central Controller Node

The Controller node maintains the cluster topology, aggregates storage capacity metrics across all worker nodes, and dispatches parallel preload/evict directives.

### Controller Configuration (`configs/hypercache-controller.toml` on Controller Host)

```toml
# Bind controller service to private cluster network IP or 0.0.0.0
controller_bind_address = "0.0.0.0"
controller_port = 9120

# Cluster Security Token for authenticating remote nodes and remote API access
# Can also be set via: export HYPERCACHE_AUTH_TOKEN="sec-cluster-secret-key"
cluster_auth_token = "sec-cluster-secret-key"
```

### Starting the Controller Daemon

```bash
# Start on default port 9120 (reads configs/hypercache-controller.toml)
./hypercache-admin -c configs/hypercache-controller.toml controller

# Or run in background
nohup ./hypercache-admin -c configs/hypercache-controller.toml controller > controller.log 2>&1 &
```

---

## 3. Setting Up GPU Worker Nodes

Each GPU node runs the `./hypercache` agent daemon. The agent automatically discovers striped NVMe drives, manages local RAM metadata, and reports heartbeats to the Controller.

### Worker Configuration (`configs/hypercache-agent.toml` on GPU Nodes)

```toml
# Node Identification & Targeting Labels
node_id = "gpu-node-01"

[node_labels]
gpu = "h100"
pool = "training"
zone = "us-east-1a"

# Point to Central Controller private IP
controller_url = "http://10.0.0.10:9120"
cluster_auth_token = "sec-cluster-secret-key"

# GPU Node Agent Fleet API (Receives preload/evict directives from Controller)
# OFF by default. Automatically enabled when controller_url is set, or set agent_enabled = true.
agent_enabled = true
agent_bind_address = "0.0.0.0"
agent_http_port = 9115

# Prometheus Metrics Exporter (OFF by default for maximum security)
# Set metrics_enabled = true (or pass --metrics) to expose Prometheus metrics.
metrics_enabled = false
metrics_bind_address = "127.0.0.1"
metrics_port = 9110

# Authoritative Storage Backends (S3 / NFS)
authoritative_backend = "s3"
s3_endpoint = "https://s3.us-east-1.amazonaws.com"
s3_region = "us-east-1"
s3_bucket = "model-weights-bucket"
s3_access_key = "YOUR_ACCESS_KEY"
s3_secret_key = "YOUR_SECRET_KEY"

# NVMe Discovery & Striping
device_discovery_mode = "auto"
include_partitions = true
device_prefixes = ["nvme", "sd"]
exclude_root_device = true
chunk_size_bytes = 4194304 # 4 MiB chunks

# Metadata Persistence (Async NVMe sync & instant warm-boot recovery)
metadata_persistence_enabled = true
metadata_persistence_path = "/var/lib/hypercache/metadata.json"

# Tier 1 Host RAM / CXL-Ready Shared Memory Acceleration (/dev/shm)
# Disabled by default. Set to true (or pass --ram) to enable Tier 1 Host RAM caching.
ram_cache_enabled = false
ram_cache_capacity_bytes = 34359738368 # 32 GiB default capacity (or auto-detected if omitted)
ram_cache_shm_path = "/dev/shm/hypercache"
```

### Starting the Worker Daemon on Compute Nodes (GPU or CPU)

```bash
# Standard startup (reads configs/hypercache-agent.toml)
./hypercache -c configs/hypercache-agent.toml start

# Start with Tier 1 Host RAM caching enabled
./hypercache -c configs/hypercache-agent.toml start --ram

# Optional: Enable Prometheus exporter via CLI flag
./hypercache -c configs/hypercache-agent.toml start --metrics

# Optional: Run in background
nohup ./hypercache -c configs/hypercache-agent.toml start > hypercache.log 2>&1 &
```

> **CPU & GPU Agnostic**: HyperCache runs on both GPU compute instances and CPU-only nodes with zero CUDA dependencies. To enable Tier 1 Host RAM (`/dev/shm`) acceleration delivering 50–100+ GB/s memory-bus streaming, set `ram_cache_enabled = true` in config or pass `--ram`.
> **Default Security Posture**: If run without a `controller_url` or `metrics_enabled`, `./hypercache -c <path> start` opens **ZERO network TCP ports**, communicating exclusively via the local Unix Domain Socket (`/run/hypercache/hypercache.sock` or fallback `/tmp/hypercache/hypercache.sock`).

---

## 4. Central Fleet Administration (`./hypercache-admin`)

All fleet operations are executed from the Controller machine using `./hypercache-admin`.

> **Zero-Config Local Admin**: When run on the Controller host, `./hypercache-admin` connects via `127.0.0.1` and requires **no flags, no tokens, and no IPs**.

### A. Inspect Fleet Topology & Storage Inventory
```bash
./hypercache-admin nodes
```

Output:
```
====================================================================================================
                               HYPERCACHE FLEET TOPOLOGY & INVENTORY                                
====================================================================================================
Controller:  http://127.0.0.1:9120
Fleet State: 8/8 Nodes Online | Aggregate Pool: 29.12 TiB | Active Footprint: 142.50 GiB (4 models)
----------------------------------------------------------------------------------------------------
NODE ID              | STATUS  | LABELS               | STRIPED NVMe DRIVES      |   CAPACITY | CACHED FOOTPRINT
----------------------------------------------------------------------------------------------------
gpu-node-01          | ONLINE  | gpu=h100,pool=train  | /dev/nvme1n1, /dev/nvme2n1 |   7.68 TiB |   35.62 GiB (1 obj)
gpu-node-02          | ONLINE  | gpu=h100,pool=train  | /dev/nvme1n1, /dev/nvme2n1 |   7.68 TiB |   35.62 GiB (1 obj)
gpu-node-03          | ONLINE  | gpu=h100,pool=train  | /dev/nvme1n1, /dev/nvme2n1 |   7.68 TiB |   35.62 GiB (1 obj)
...
====================================================================================================
```

### B. Proactive Pre-Loading / Pre-Warming Checkpoints
Warm up model weights across all GPU nodes before starting a training job:

```bash
# Preload on all nodes across the fleet
./hypercache-admin preload hypercache://s3/models/llama-3-70b/weights.safetensors --nodes all

# Preload only on specific nodes
./hypercache-admin preload hypercache://s3/models/llama-3-70b/weights.safetensors --nodes "gpu-node-01,gpu-node-02"

# Preload targeting specific node labels
./hypercache-admin preload hypercache://s3/models/llama-3-70b/weights.safetensors --selector "gpu=h100,pool=training"
```

### C. Evicting Specific Models Fleet-Wide
Reclaim NVMe storage space across all nodes when a model version is deprecated:

```bash
./hypercache-admin evict hypercache://s3/models/llama-2-7b/weights.safetensors --nodes all
```

### D. Purging All Caches Fleet-Wide
Clear all cached objects and reset NVMe drives across the entire cluster:

```bash
./hypercache-admin purge --nodes all
```

---

## 5. Local Worker CLI Operations (`./hypercache`)

Run directly on any individual GPU worker node:

| Task | Command |
| :--- | :--- |
| **Start Daemon** | `./hypercache -c configs/hypercache-agent.toml start` |
| **List Cached Models** | `./hypercache ls` (or `./hypercache ls --json`) |
| **Pre-load Model Locally** | `./hypercache preload hypercache://s3/models/llama-3/weights.safetensors` |
| **Batch Pre-load from File** | `./hypercache preload --file checkpoints.txt` |
| **Evict Model Locally** | `./hypercache rm hypercache://s3/models/llama-3/weights.safetensors` |
| **Refresh Model from Source**| `./hypercache refresh hypercache://s3/models/llama-3/weights.safetensors` |
| **Purge Local Cache** | `./hypercache purge` |
| **Inspect Local SSDs** | `./hypercache inspect` |
| **Check Local Health** | `./hypercache status` |
| **Run DMA Benchmark** | `./hypercache bench --size-mb 5000` |

---

## 6. PyTorch Integration

Install the Python client (`hypercache-io`) on your GPU worker nodes or inside your training/inference containers:
```bash
# Install from PyPI (exposes `import hypercache` in Python)
pip install hypercache-io

# Or install from the local repository
pip install -e ./client-py
```

Load checkpoints directly from striped NVMe into GPU memory:

```python
import fsspec
import torch
import hypercache  # Hooks hypercache:// protocol

# Load weights directly into GPU VRAM via Direct Block DMA
checkpoint_uri = "hypercache://s3/models/llama-3-8b/weights.pt"

with fsspec.open(checkpoint_uri, "rb") as f:
    state_dict = torch.load(f, map_location="cuda:0")

model.load_state_dict(state_dict)
```

---

## 7. Prometheus & Grafana Monitoring

HyperCache exports real-time metrics on port `9110` (`/metrics`).

### Prometheus Scrape Configuration (`prometheus.yml`)
```yaml
scrape_configs:
  - job_name: 'hypercache-workers'
    scrape_interval: 1s
    static_configs:
      - targets:
          - '10.0.0.11:9110'
          - '10.0.0.12:9110'
```

### Key Metrics Exported

- `hypercache_bandwidth_mbps`: Real-time DMA disk read throughput.
- `hypercache_hit_ratio_percent`: Cache hit percentage.
- `hypercache_storage_utilization_percent`: Striped NVMe pool utilization.
- `hypercache_storage_used_bytes`: Total cached bytes in the NVMe pool.
- `hypercache_s3_egress_bytes_saved_total`: Cumulative egress data avoided.
- `hypercache_gpu_idle_time_saved_seconds`: Cumulative GPU idle time saved.

---

## 8. Network Security & Production Hardening Checklist

1. **Private Subnets Only**: Ensure `controller_bind_address` and `agent_bind_address` are bound to your private cluster network (e.g. `10.0.0.x`), never public IPs.
2. **Bearer Token Authentication**: Set `cluster_auth_token` in `configs/hypercache-agent.toml` / `configs/hypercache-controller.toml` or `export HYPERCACHE_AUTH_TOKEN="your-cluster-secret"`.
3. **Firewall Rules (Security Groups)**:
   - Port `9120` (Controller): Open only to GPU Worker node IPs.
   - Port `9115` (Agent API): Open only to Controller IP.
   - Port `9110` (Prometheus): Open only to Prometheus scraper IP or `127.0.0.1`.
