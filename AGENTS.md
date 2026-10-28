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
├── CLAUDE.md                   # Claude Code entry point; imports this file
├── CHANGELOG.md                # Release notes per version
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
│   ├── Cargo.toml              # Dependencies: tokio, axum, moka, reqwest, tower, clap, serde, crc32fast
│   ├── Cargo.lock              # Committed: binaries need reproducible (--locked) builds
│   ├── rustfmt.toml            # max_width = 120; CI enforces cargo fmt --check
│   ├── src/
│   │   ├── lib.rs              # Library crate root; both binaries build on it
│   │   ├── main.rs             # Binary: ./hypercache (Worker Node Daemon & Local CLI)
│   │   ├── bin/
│   │   │   └── admin.rs        # Binary: ./hypercache-admin (Cluster Administration Suite)
│   │   ├── config.rs           # Strongly typed configuration structs and TOML loader
│   │   ├── protocol.rs         # JSON-lines IPC wire protocol over Unix Domain Socket (SyncStatus, ChunkDescriptor)
│   │   ├── server.rs           # UDS server (bounded requests), commit/sha256 verification, write-back retry & recovery
│   │   ├── auth.rs             # Constant-time Bearer token validation shared by controller and agent
│   │   ├── telemetry.rs        # Prometheus metrics collector and optional HTTP exporter
│   │   ├── agent_server.rs     # Worker Node Agent HTTP server (port 9115), streaming P2P/replica, heartbeat loop
│   │   ├── fleet.rs            # CLI client functions for fleet administration commands
│   │   ├── bench.rs            # `hypercache bench` end-to-end benchmark (explicit synthetic opt-in)
│   │   ├── check.rs            # `hypercache check` pre-flight host readiness checks
│   │   ├── discovery.rs        # Raw block device discovery & safety exclusion engine
│   │   ├── metadata.rs         # Moka W-TinyLFU cache, extent allocator + quarantine, generations, manifest writer
│   │   ├── ram.rs              # Tier 1 POSIX shared memory (/dev/shm) chunk pool
│   │   ├── controller/
│   │   │   └── mod.rs          # FleetRegistry, strict node targeting, network security, parallel dispatch
│   │   └── ingestion/
│   │       ├── mod.rs          # Backend router, single-flight ingest, digest, write-back flush
│   │       ├── striper.rs      # Chunk striping across devices with rollback, CRC32 and fdatasync
│   │       ├── s3.rs           # SigV4 S3 client: parallel ranged GET, multipart upload, ListObjectsV2
│   │       ├── nfs.rs          # POSIX / NFS reader with path confinement (nfs_allowed_paths)
│   │       └── synthetic.rs    # Synthetic benchmark payloads (opt-in only)
│   └── tests/
│       └── s3_live.rs          # #[ignore]d live S3 tests (run by scripts/verify_release.sh)
│
├── scripts/
│   ├── verify_release.sh       # Release verification harness (static, rust, python, e2e, s3, linux, image)
│   ├── e2e/hypercache_e2e.py   # End-to-end daemon / fleet / S3 scenarios against the real binaries
│   └── setup_mock_devices.sh   # Creates sparse mock block device files in /tmp
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
        ├── test_pipelined_and_numa.py  # Unit tests for pipelined DMA and NUMA affinity
        └── test_robustness.py  # Thread/fork safety, framing, O_DIRECT bounce buffers, CRC verification
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
   - Controlled by `allow_local_admin_bypass` (default `true`). It must be `false` wherever a local proxy (K8s sidecar, nginx, `kubectl port-forward`) can forward remote traffic; the bundled K8s manifest sets it to `false`.
2. **Cluster Security Token (`cluster_auth_token` / `HYPERCACHE_AUTH_TOKEN`)**:
   - When set, all remote network calls between the Controller and Worker Agents require `Authorization: Bearer <token>`.
   - Node registrations with missing or invalid tokens are rejected with `HTTP 401 Unauthorized`.
3. **Zero-Exposure Default Configuration**:
   - Single-node worker runs with `metrics_enabled = false` and `agent_enabled = false`, exposing no network ports.
   - The agent API refuses to bind a non-loopback address without `cluster_auth_token`.
4. **Local Data Access Confinement**:
   - `file://` / `nfs://` URIs are confined to `nfs_mount_path` and `nfs_allowed_paths` (canonicalized; `..` and symlink escapes rejected).
   - Synthetic data is never served unless `allow_synthetic_s3_fallback = true` or a request explicitly opts in for a `hypercache://models/benchmark/` URI.

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
*Expected: All unit and integration tests pass (70+ tests covering metadata, allocator, persistence, ingestion, RAM memory pool, discovery, auth, agent and controller security). The live S3 tests in `tests/s3_live.rs` are `#[ignore]`d and run by the verification harness.*

### Running Python Test Suite
```bash
.venv/bin/pytest client-py/tests
```
*Expected: All 32 client, datapath, fsspec and integration tests pass.*

### Running the Release Verification Harness
```bash
scripts/verify_release.sh            # static, rust, python, e2e, s3, linux, image
scripts/verify_release.sh --quick    # static, rust, python
```
*Expected: `RESULT: PASSED`. Docker-based stages (s3, linux, image) are skipped with a notice when Docker is unavailable. Run the full harness before tagging a release.*

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

## 7. Working on the Codebase: Invariants, Pitfalls & Workflows

### A. Data-Path Invariants (do not break these)
- **Never return freed extents straight to the free pool.** Evicted or replaced chunks go to the quarantine in `metadata.rs` and are reusable only after `reclaim_grace_secs`. Readers resolve a layout over IPC and then read raw offsets with no lock, so early reuse makes them silently read another object's bytes.
- **Every layout mutation bumps `generation`.** `locate` returns it. Write-back status updates must go through `update_sync_status_if_generation` with the generation they were started for, so an old flush cannot mark a newer commit as synced.
- **Go through single-flight ingest.** Any code path that may ingest a URI (UDS `locate`/`ingest`/`preload`, agent preload, batch) must call `IngestionEngine::get_or_ingest` / `ingest_object_full`, never a backend fetch directly.
- **Durable before visible.** Chunk data is `fdatasync`ed and its CRC32 recorded before `save_layout` publishes it. Do not publish a layout that points at unsynced data.
- **One manifest writer.** Mutations signal the coalescing writer in `metadata.rs`. Do not write `metadata.json` from anywhere else. `atomic_write` (unique tmp, fsync, rename, dir fsync) is the only correct way to replace it.
- **Dirty objects are never evictable.** `COMMITTED`, `SYNCING_BACKEND` and `SYNC_FAILED` weigh 0 for eviction. `resume_pending_writebacks` re-queues them at boot. A new `SyncStatus` variant must be classified in `SyncStatus::is_dirty` (`protocol.rs`).
- **Allocations roll back on failure.** If striping fails partway, every extent allocated in that call is returned.

### B. Compatibility Rules in Practice
- New fields on IPC or manifest structs (`ChunkDescriptor`, `StoredMetadata`, `Response`) must be `Option` or `#[serde(default)]`. Manifests and Python clients from 0.9.0 must keep loading.
- Enums on the wire use `#[serde(rename = "...")]` with the existing strings (`"ONLINE"`, `"SYNC_FAILED"`, ...). Do not rename them.
- Add new TOML keys with defaults. Never rename or remove a key. Document each new key in `configs/*.toml` and in the table in section 8.
- The Python client must keep working against an older daemon. Treat missing response fields as optional.

### C. Platform Pitfalls
- **macOS hosts do not compile the Linux-only code** (`#[cfg(target_os = "linux")]`: sysfs discovery, O_DIRECT, `fdatasync` paths). Clippy and test failures there show up only on Linux. Run `scripts/verify_release.sh --stages linux` (Docker) before calling a change done.
- **Clippy depends on the toolchain.** CI uses current stable. A local toolchain that is a few releases behind may miss new lints, so run `rustup update` if CI clippy fails but local clippy passes.
- **Containers expose host disks in sysfs without `/dev` nodes.** Discovery must skip them (it logs at debug). Do not turn that case into an error.
- **Agents bound to loopback or `0.0.0.0` advertise an unusable address.** The controller dials them through `resolve_agent_address` (falls back to the registration source IP, IPv6 bracketed). Use it for every new dispatch path.
- **Use `HYPERCACHE_SOCKET_PATH` to isolate test daemons.** It moves the socket and lock file for the daemon, the CLI and the Python client, so tests never touch `/run/hypercache` or another running daemon.
- **MinIO images are no longer pullable.** The live S3 tests use `versity/versitygw:v1.10.0`. `server_enforces_signatures` proves the server really checks SigV4. Keep it, or signing bugs pass silently.
- **Do not edit `scripts/verify_release.sh` while it is running.** Bash reads the script incrementally, and an edit mid-run causes bogus syntax errors.
- **The zero-emoji scanner checks itself.** Write emoji code point ranges as integers (`chr(0x1F300)`), never as literal characters or `\u` escapes that a tool might decode.
- **Docker buildx on macOS** can fail with a permission error on `~/.docker/buildx`. Run the image stage with `BUILDX_CONFIG=$(mktemp -d)` instead of changing the user's Docker config.

### D. Where Tests Go
| Change | Add a test in |
| :--- | :--- |
| Pure logic (allocator, selectors, auth, config parsing) | `#[cfg(test)]` module in the same `.rs` file |
| IPC behavior (request handling, commit, write-back) | `server.rs` tests (they run a real UDS round trip) |
| Anything a real S3 server must confirm (signing, pagination, multipart) | `control-plane/tests/s3_live.rs` (`#[ignore]`, gated on `HYPERCACHE_TEST_S3_*`) |
| Multi-process behavior (restart, SIGKILL, eviction pressure, fleet dispatch, auth over HTTP) | a check in `scripts/e2e/hypercache_e2e.py` (`scenario_daemon`, `scenario_fleet` or `scenario_s3`) |
| Python client, datapath, integrations | `client-py/tests/` (`test_robustness.py` for threading, fork, framing, O_DIRECT, CRC) |

A regression test should fail without the fix. When in doubt, revert the fix briefly and confirm that it does.

Run the E2E scenarios directly against release binaries:
```bash
cargo build --release --manifest-path control-plane/Cargo.toml
.venv/bin/python scripts/e2e/hypercache_e2e.py daemon fleet --keep   # add s3 with --s3-endpoint ...
```

### E. Release Checklist
1. Bump the version in `control-plane/Cargo.toml`, `client-py/pyproject.toml`, `client-py/hypercache/__init__.py` and the default tag in `.github/workflows/release.yml`. The static stage fails if they disagree. The binaries read their version from Cargo (`env!("CARGO_PKG_VERSION")`), so never hardcode version strings in Rust.
2. Run `cargo update` only on purpose. `Cargo.lock` is committed and every build uses `--locked`.
3. Add a section to `CHANGELOG.md`.
4. Run the full `scripts/verify_release.sh` until it prints `RESULT: PASSED`.
5. Leave committing, tagging and pushing to the maintainer unless they explicitly ask you to do it.

---

## 8. Master Configuration Reference (`configs/hypercache-agent.toml` & `configs/hypercache-controller.toml`)

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
| `nfs_allowed_paths` | list | `[]` | Extra roots `file://` / `nfs://` URIs may access (besides `nfs_mount_path`) |
| `reclaim_grace_secs` | integer | `30` | Grace period before freed extents are reused (protects in-flight readers) |
| `s3_parallel_streams` | integer | `8` | Parallel ranged GETs per object during S3 ingest |
| `s3_read_timeout_secs` | integer | `60` | Per-read idle timeout for S3 transfers (no total-time cap) |
| `s3_session_token` | string | `$AWS_SESSION_TOKEN` | Session token for temporary S3 credentials |
| `s3_multipart_threshold_bytes` | integer | `67108864` | Uploads above this size use S3 multipart upload |
| `allow_local_admin_bypass` | bool | `true` | Controller: loopback callers skip auth (set `false` behind proxies / in K8s) |
| `node_gc_secs` | integer | `600` | Controller: forget nodes silent for this long |
