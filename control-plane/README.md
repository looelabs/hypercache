# HyperCache Control Plane (`control-plane`)

Rust crate containing the HyperCache node daemon, cluster coordinator, and CLI binaries.

For full system requirements, configuration guides (`hypercache-agent.toml` and `hypercache-controller.toml`), and end-to-end usage documentation, see the root **[README.md](../README.md)** and **[ARCHITECTURE.md](../ARCHITECTURE.md)**.

---

## Binaries Built by This Crate

| Binary | Source Entrypoint | Description |
| :--- | :--- | :--- |
| `hypercache` | [`src/main.rs`](src/main.rs) | Worker node daemon and local CLI (`start`, `check`, `inspect`, `preload`, `ls`, `rm`, `refresh`, `purge`, `bench`, `status`). |
| `hypercache-admin` | [`src/bin/admin.rs`](src/bin/admin.rs) | Central Cluster Controller daemon and fleet administration CLI (`controller`, `nodes`, `preload`, `preload-batch`, `evict`, `purge`, `status`). |

---

## Crate Module Map (`src/`)

| File / Module | Responsibility |
| :--- | :--- |
| [`src/lib.rs`](src/lib.rs) | Library crate root exposing internal modules. |
| [`src/config.rs`](src/config.rs) | Strongly-typed TOML configuration structs and environment variable overrides. |
| [`src/check.rs`](src/check.rs) | Pre-flight host diagnostics (`hypercache check`) for permissions, raw drives, `O_DIRECT`, and sockets. |
| [`src/discovery.rs`](src/discovery.rs) | Linux `sysfs` block device auto-discovery, NUMA mapping, and OS root/filesystem safety exclusions. |
| [`src/metadata.rs`](src/metadata.rs) | Moka W-TinyLFU in-memory cache catalog, slab allocator, and async warm-boot manifest persistence. |
| [`src/ingestion/`](src/ingestion/) | Multi-backend ingestion router (`mod.rs`), 4 MiB chunk striper (`striper.rs`), S3 client (`s3.rs`), and NFS/POSIX reader (`nfs.rs`). |
| [`src/ram.rs`](src/ram.rs) | Optional POSIX shared memory (`/dev/shm`) and HugePages (`hugetlbfs`) block manager. |
| [`src/protocol.rs`](src/protocol.rs) | IPC request/response wire protocol over Unix Domain Sockets. |
| [`src/server.rs`](src/server.rs) | Local Unix Domain Socket server listener and request dispatcher. |
| [`src/agent_server.rs`](src/agent_server.rs) | Worker Node Agent HTTP API (`:9115`), controller heartbeat loop, and P2P chunk/replica transfer handlers. |
| [`src/controller/mod.rs`](src/controller/mod.rs) | Central Cluster Controller (`:9120`), `FleetRegistry`, consistent-hash sharding, and parallel node dispatch. |
| [`src/fleet.rs`](src/fleet.rs) | Client functions for `hypercache-admin` fleet management commands. |
| [`src/telemetry.rs`](src/telemetry.rs) | Prometheus metrics collector and optional HTTP exporter (`:9110/metrics`). |
| [`src/bench.rs`](src/bench.rs) | Multi-threaded hardware DMA benchmark runner and SHA-256 verification suite. |

---

## Building & Testing

```bash
# Compile release binaries
cargo build --release

# Run unit and integration tests
cargo test
```
