# AGENTS.md: AI Developer & Contributor Guide for HyperCache

This document provides essential context, architectural principles, codebase structure, development standards, and testing procedures for AI agents and human engineers working on the **HyperCache** codebase.

---

## 1. Project Overview & Mission

HyperCache is a host-based caching and acceleration engine designed for high-performance AI training and inference clusters (PyTorch, Safetensors, Hugging Face). 

### Core Problems Solved
1. **Network Storage Bottlenecks**: Loading multi-gigabyte or terabyte model checkpoints from S3 or network-attached NFS during training job initialization or auto-scaling causes severe GPU idle time and high egress costs.
2. **Linux Page Cache & Host CPU Overhead**: Traditional file system reads pass data through the Linux kernel VFS, page cache, and host CPU memory before reaching GPU VRAM.
3. **Cluster-Wide Coordination Friction**: In multi-node GPU clusters, coordinating model caching across dozens of nodes requires a centralized, secure control plane with parallel dispatch.

### Core Solution
- **Striped Raw Block Acceleration**: Auto-discovers physical/virtual NVMe drives (`/dev/vdc`, `/dev/vdd`, `/dev/nvme*`) and stripes model weights into 4 MiB chunks directly across raw devices.
- **Direct Block DMA / GPUDirect Storage (GDS)**: Streams data directly from raw NVMe blocks or `O_DIRECT` filesystem arenas into GPU VRAM using Direct Block DMA (`O_DIRECT`, `kvikio.CuFile`), achieving hardware line-rate read throughput (~7 GB/s per PCIe Gen 4 NVMe, ~14 GB/s per PCIe Gen 5 NVMe).
- **Dual-Tier Metadata Architecture**: In-memory **Moka W-TinyLFU** cache for sub-millisecond route resolution, backed by an atomic NVMe persistence manifest for instant warm-boot recovery (<20 ms).
- **Dual Authoritative Backends**: Native streaming ingestion from both **Amazon S3 / S3-compatible object stores** and **NFS / Shared POSIX file systems**.
- **Centralized Cluster Controller & Coordinator**: Dedicated control plane (`./hypercache-admin`) for cluster topology discovery, aggregate NVMe capacity tracking, and parallel fleet-wide pre-loading (`preload`, `evict`, `purge`).

---

## 2. Strict Behavioral & Engineering Rules

When modifying or extending this codebase, you **MUST** adhere to the following rules:

1. **Strict Zero-Emoji Policy**:
   - Do **NOT** use any emojis in code, comments, docstrings, terminal outputs, CLI logs, unit tests, commit messages, or documentation.
2. **Zero Network Exposure by Default**:
   - The worker daemon (`./hypercache start`) must open **ZERO network TCP ports by default**.
   - By default, communication happens exclusively over the local Unix Domain Socket (`/run/hypercache/hypercache.sock` or fallback `/tmp/hypercache/hypercache.sock`).
   - Prometheus exporter (port 9110) and Node Agent Fleet API (port 9115) must remain **OFF** unless explicitly enabled via flags or configuration.
3. **Zero-Config Local Admin Ergonomics**:
   - Running `./hypercache-admin` on the Controller host (`127.0.0.1` / loopback) must work immediately with **zero flags, zero tokens, and zero IPs required**.
   - Remote access over the network requires Bearer token authentication (`cluster_auth_token` / `HYPERCACHE_AUTH_TOKEN`).
4. **Preserve Backward Compatibility**:
   - Maintain compatibility for existing Python bindings (`fsspec`, `torch.load`), IPC protocols, and TOML configuration keys.
5. **Full Test Verification**:
   - All Rust unit tests (`cargo test`) and Python unit tests (`pytest`) must pass cleanly before completing any task.

---

## 3. Workspace & Codebase Map

```
hypercache-poc/
├── AGENTS.md                   # This AI contributor and developer guide
├── README.md                   # Public project README and architectural summary
├── GETTING_STARTED.md          # Operator quickstart and cluster setup walkthrough
├── ARCHITECTURE.md             # In-depth systems design document
├── build.sh                    # Compiles native release binaries (./hypercache, ./hypercache-admin)
├── install.sh                  # Complete environment installer (Rust, Python .venv, PyTorch dependencies)
│
├── configs/
│   ├── hypercache-agent.toml   # Worker Node Agent configuration (drives, S3/NFS, cluster registration)
│   ├── hypercache-controller.toml # Central Controller configuration (network binding, auth token)
│   └── grafana/                # Grafana provisioning & dashboard templates
│
├── control-plane/              # High-performance Rust control plane & CLI suite
│   ├── Cargo.toml              # Dependencies: tokio, axum, moka, reqwest, tower, clap, serde
│   ├── src/
│   │   ├── lib.rs              # Library crate root exposing internal modules
│   │   ├── main.rs             # Binary: ./hypercache (Worker Node Daemon & Local CLI)
│   │   ├── bin/
│   │   │   └── admin.rs        # Binary: ./hypercache-admin (Cluster Administration Suite)
│   │   ├── config.rs           # Strongly typed configuration structs and TOML loader
│   │   ├── protocol.rs         # Binary/JSON IPC wire protocol over Unix Domain Socket
│   │   ├── server.rs           # Unix Domain Socket server listener & request dispatcher
│   │   ├── telemetry.rs        # Prometheus metrics collector and optional HTTP exporter
│   │   ├── agent_server.rs     # Worker Node Agent HTTP server (port 9115) & heartbeat loop
│   │   ├── fleet.rs            # CLI client functions for fleet administration commands
│   │   ├── controller/         # Central Cluster Controller & Fleet Coordinator module
│   │   │   └── mod.rs          # FleetRegistry, network security, and parallel dispatch handlers
│   │   ├── discovery/          # Raw block device discovery & safety exclusion engine
│   │   │   └── mod.rs          # Probes /dev, parses sysfs, excludes OS root and mounted FS
│   │   ├── metadata/           # Cache metadata store & NVMe persistence engine
│   │   │   └── mod.rs          # Moka W-TinyLFU cache, memory weighing, atomic JSON manifest
│   │   └── ingestion/          # Multi-backend data ingestion & NVMe striping engine
│   │       ├── mod.rs          # Stream router, chunk striping across drives
│   │       ├── s3.rs           # Streaming S3 client with parallel chunk writes
│   │       └── nfs.rs          # Zero-copy POSIX / NFS file reader
│
└── client-py/                  # Zero-copy Python AI client & framework integration
    ├── pyproject.toml          # Python package specification
    ├── hypercache/
    │   ├── __init__.py         # Package entry point, auto-registers fsspec hook
    │   ├── client.py           # Unix Domain Socket IPC client
    │   ├── datapath.py         # Direct Block DMA / GDS direct I/O reader
    │   ├── fsspec_fs.py        # AbstractFileSystem implementation for torch.load/save
    │   ├── kv_store.py         # Content-addressed KV-Cache Prefix Block Store
    │   ├── pipeline.py         # Double-buffered pipelined GPU tensor loader
    │   └── integrations/
    │       ├── safetensors_loader.py # Zero-copy Safetensors & vLLM weight loader
    │       └── dcp.py          # PyTorch Distributed Checkpoint (DCP) StorageReader/Writer
    └── tests/
        ├── test_client.py      # Unit tests for UDS protocol and client metadata parsing
        ├── test_fsspec.py      # Unit tests for fsspec file operations and streaming writes
        ├── test_integrations_and_kv.py # Unit tests for Safetensors/vLLM, DCP, and KV-Cache store
        └── test_pipelined_and_numa.py  # Unit tests for pipelined DMA and NUMA affinity
```

---

## 4. Key Architectural Subsystems

### A. Dynamic NVMe Auto-Discovery (`control-plane/src/discovery.rs`)
- Scans host block devices using sysfs (`/sys/block`) and `/dev`.
- Detects physical NVMe (`nvme*`), virtual disks (`vd*`, `xvd*`), and SCSI/SATA (`sd*`).
- **Safety Filters**: Automatically excludes the OS root partition drive, swap partitions, and active mounted filesystems to prevent data corruption.
- Mock fallback devices are **strictly disabled by default** (`auto_create_mock_devices = false`).

### B. Tier 1 Host RAM & CXL-Ready Acceleration (`control-plane/src/ram.rs`)
- Manages POSIX shared memory chunks in `/dev/shm/hypercache/` (with `/tmp/hypercache/shm/` fallback).
- Delivers **50–100+ GB/s** line-rate memory bus streaming for ephemeral AI workloads.
- Provides immediate drop-in compatibility for future CXL 2.0/3.0 Type 3 DAX appliances (`/dev/dax*`).
- Dynamically auto-sizes to **75% of `/dev/shm`** or configured `ram_cache_capacity_bytes`.

### C. Ingestion & Multi-Tier Striping Engine (`control-plane/src/ingestion/`)
- When a model URI is requested (`locate` with `auto_ingest=True` or `preload`):
  1. Identifies backend from URI: `s3://` or `hypercache://s3/...` routes to S3; `file:///`, `nfs://`, or `hypercache://file/...` routes to NFS.
  2. Identifies target tier (`--tier ram`, `--tier nvme`, or auto).
  3. Divides data into fixed **4 MiB chunks** (`chunk_size_bytes = 4194304`).
  4. Stripes chunks across POSIX shared memory or discovered NVMe drives in round-robin fashion.
  5. Records physical block offsets and device assignments in the Moka metadata manifest.

### D. In-Memory & Persistent Metadata (`control-plane/src/metadata.rs`)
- **Moka W-TinyLFU**: High-throughput concurrent cache managing object metadata in RAM. Evicts least-frequently-used objects when combined storage capacity (NVMe + RAM) is reached.
- **NVMe Persistence Manifest**: Flushes metadata asynchronously to `/var/lib/hypercache/metadata.json` (or fallback `/tmp/hypercache/metadata.json`). On daemon reboot, restores the entire cache catalog in **<20 ms** without re-reading raw disks.

### E. Direct Block DMA Datapath (`client-py/hypercache/datapath.py`)
- Reads raw block devices directly using `O_DIRECT` or `kvikio.CuFile` (GPUDirect Storage), and reads shared memory chunks using zero-copy `mmap`.
- Bypasses the OS kernel page cache and CPU memory copies.
- Computes chunk offsets: `device_start_offset_bytes + (stripe_index * chunk_size_bytes)`.

### F. Central Controller & Multi-Node Coordination (`control-plane/src/controller/`)
- **Controller Daemon (`./hypercache-admin controller`)**:
  - Listens on `controller_bind_address:controller_port` (default: `0.0.0.0:9120`).
  - Maintains `FleetRegistry`: In-memory thread-safe map of active nodes, last heartbeat timestamps, drive lists, aggregate capacities, and cached footprints.
  - Automatically identifies remote node IP addresses from incoming TCP connections (`ConnectInfo`).
  - Dispatches parallel HTTP requests (`futures_util::future::join_all`) to target worker nodes (`/api/v1/agent/preload`, `/api/v1/agent/evict`, `/api/v1/agent/purge`).
- **Worker Node Agent (`control-plane/src/agent_server.rs`)**:
  - Sends heartbeats to the Controller every **5 seconds** (`POST /api/v1/register`).
  - Listens on port `9115` for authenticated directives from the Controller.
  - Nodes that miss heartbeats for >15 seconds are marked `OFFLINE`.

---

## 5. Security & Authentication Model

1. **Local Loopback Zero-Config**:
   - Connections originating from `127.0.0.1` / `::1` or the Controller host's bound IP bypass authentication, granting instant access to `./hypercache-admin` on the Controller node.
2. **Cluster Security Token (`cluster_auth_token` / `HYPERCACHE_AUTH_TOKEN`)**:
   - When set, all remote network calls between the Controller and Worker Agents require `Authorization: Bearer <token>`.
   - Node registrations with missing or invalid tokens are rejected with `HTTP 401 Unauthorized`.
3. **Zero-Exposure Default Configuration**:
   - Single-node worker runs with `metrics_enabled = false` and `agent_enabled = false`, exposing no network ports.

---

## 6. Build, Test & Verification Procedures

### Compiling Release Binaries
```bash
# Using build script
./build.sh

# Or directly with Cargo
cargo build --release --manifest-path control-plane/Cargo.toml
cp control-plane/target/release/hypercache ./hypercache
cp control-plane/target/release/hypercache-admin ./hypercache-admin
```

### Running Rust Test Suite
```bash
cargo test --manifest-path control-plane/Cargo.toml
```
*Expected: All unit and integration tests pass (15+ tests covering metadata, persistence, ingestion, RAM memory pool, discovery, network isolation, and controller security).*

### Running Python Test Suite
```bash
.venv/bin/pytest client-py/tests
```
*Expected: All 10 client and fsspec tests pass.*

### Running Local Benchmark Verification
```bash
# Start background daemon on mock or real drives (explicit config path required)
./hypercache -c configs/hypercache-agent.toml start &
DAEMON_PID=$!

# Run 5 GB benchmark
./hypercache bench --size-mb 5000

# Terminate test daemon
kill $DAEMON_PID
```

---

## 7. Master Configuration Reference (`configs/hypercache-agent.toml` & `configs/hypercache-controller.toml`)

| Key | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `device_discovery_mode` | string | `"auto"` | `"auto"` (scan physical/virtual drives) or `"manual"` |
| `allow_virtual_drives` | bool | `true` | Allow virtualized block devices (`vd*`, `xvd*`) |
| `include_partitions` | bool | `true` | Include disk partitions (`vdc1`, `nvme0n1p1`) |
| `device_prefixes` | list | `["nvme", "vd", "xvd", "sd"]` | Allowed block device name prefixes |
| `exclude_root_device` | bool | `true` | Exclude OS root disk from storage pool |
| `chunk_size_bytes` | integer | `4194304` | Chunk striping size in bytes (4 MiB) |
| `device_start_offset_bytes`| integer | `4096` | Safety offset from start of block device (4 KB) |
| `metadata_persistence_enabled`| bool | `true` | Enable async NVMe metadata manifest sync |
| `metadata_persistence_path` | string | `"/var/lib/hypercache/metadata.json"` | Path to NVMe metadata manifest |
| `ram_cache_enabled` | bool | `false` | Enable Tier 1 Host RAM / CXL shared memory tier (off by default) |
| `ram_cache_capacity_bytes` | integer | `34359738368` | RAM cache capacity in bytes (32 GiB default or 75% of /dev/shm) |
| `ram_cache_shm_path` | string | `"/dev/shm/hypercache"` | POSIX shared memory directory path |
| `authoritative_backend` | string | `"s3"` | Default backend: `"s3"` or `"nfs"` |
| `s3_endpoint` | string | `"http://127.0.0.1:9002"` | Authoritative S3 endpoint URL |
| `s3_region` | string | `"us-east-1"` | S3 region |
| `s3_bucket` | string | `"hypercache-authoritative"` | S3 bucket name |
| `nfs_mount_path` | string | `"/mnt/nfs"` | POSIX/NFS root mount path |
| `metrics_enabled` | bool | `false` | Enable Prometheus HTTP exporter (off by default) |
| `metrics_bind_address` | string | `"127.0.0.1"` | IP to bind Prometheus metrics exporter on |
| `metrics_port` | integer | `9110` | Port for Prometheus metrics exporter |
| `agent_enabled` | bool | `false` | Enable Node Agent Fleet API (off by default) |
| `agent_bind_address` | string | `"0.0.0.0"` | IP to bind Node Agent Fleet API on |
| `agent_http_port` | integer | `9115` | Port for Node Agent Fleet API |
| `controller_url` | string | `None` | URL of Central Controller for auto-registration |
| `controller_bind_address`| string | `"0.0.0.0"` | IP to bind Central Controller service on |
| `controller_port` | integer | `9120` | Port for Central Controller service |
| `cluster_auth_token` | string | `None` | Shared secret Bearer token for cluster security |
| `node_id` | string | `"$HOSTNAME"` | Unique identifier for the worker node |
| `node_labels` | map | `{}` | Key-value labels for selective targeting |
| `storage_backing_mode` | string | `"auto"` | `"auto"`, `"raw_block"`, or `"filesystem"` (`O_DIRECT` arena files for K8s/GDS) |
| `filesystem_cache_paths` | list | `[]` | Mounted ext4/XFS directory paths for Direct I/O arena files |
| `filesystem_arena_size_bytes` | integer | `10737418240` | Size per filesystem arena file in bytes (10 GiB default) |
| `allow_synthetic_s3_fallback` | bool | `false` | Gate synthetic benchmark fallback on S3 error (`false` in production) |
| `peer_replication_factor` | integer | `1` | Peer checkpoint replication factor (`2` enables $k=2$ peer replica on `CommitWrite`) |
| `p2p_preload_enabled` | bool | `false` | Enable P2P fleet pre-warming (1 origin pull + peer chunk distribution) |
