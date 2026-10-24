# HyperCache Control Plane Daemon (`control-plane`)

High-performance Rust daemon responsible for:
1. **Dynamic Storage Pool Auto-Discovery**: Automatically inspects host block devices (`/dev/nvme*`, `/dev/vd*`, `/dev/sd*`) via Linux sysfs while safely excluding the OS root filesystem.
2. **Ephemeral In-Memory Metadata Store**: High-concurrency **Moka W-TinyLFU cache** for sub-millisecond physical chunk offset lookups with zero external DB dependencies.
3. **Auto-Ingestion Engine**: Fetches cache misses from authoritative cloud storage (S3 / Spaces / Ceph) and stripes data into 4 MiB chunks across discovered NVMe/SSDs.
4. **Native Prometheus Exporter**: High-throughput telemetry endpoint on `:9110/metrics` providing real-time hardware DMA bandwidth, Moka latencies, S3 egress savings, and GPU idle time ROI metrics.

---

## CLI Subcommands

```bash
# Build the native release binary
cargo build --release

# Start daemon and Prometheus exporter on :9110
./target/release/hypercache start

# Start on custom metrics port
./target/release/hypercache start --metrics-port 9110

# Run native 32-worker parallel verification benchmark
./target/release/hypercache bench --size-mb 5000

# Run live S3 comparison benchmark
./target/release/hypercache bench --size-mb 5000 \
  --s3-endpoint "https://s3.us-east-1.amazonaws.com" \
  --bucket "my-model-bucket"

# Inspect auto-discovered physical SSDs and pool capacity
./target/release/hypercache inspect

# Query live daemon status and active drives
./target/release/hypercache status
```

---

## Unit Tests

```bash
cargo test
```
