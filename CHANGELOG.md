# Changelog

## 0.9.1

Hardening release: data integrity under concurrency and crashes, network and filesystem security,
S3 throughput, and a release verification harness. Wire protocol, Python APIs and TOML keys stay
backward compatible: manifests and clients from 0.9.0 keep working, and every config key is
additive.

### Data integrity
- Freed blocks are quarantined for `reclaim_grace_secs` (default 30) before reuse, so a reader that
  resolved a layout just before an eviction or re-ingest never reads another object's bytes.
- `locate` returns a layout `generation`. Write-back status updates are tied to the generation, so
  an older flush can never mark a newer write as synced.
- Concurrent `locate(auto_ingest)` / preload calls for the same URI share one ingest (single-flight)
  instead of downloading the object N times.
- A single coalescing manifest writer: unique temp files, fsync of the file and its directory, no
  more torn or out-of-order manifests under concurrent mutations.
- Chunk data is fsynced (`fdatasync`) before a layout is published, and every chunk records a CRC32.
  The Python client can verify reads (`HYPERCACHE_VERIFY_CHECKSUMS=1`).
- `commit_write` verifies the client-supplied SHA-256 against the staged data and rejects a mismatch.
- `SYNC_FAILED` objects stay protected from eviction. Background write-back retries with backoff,
  and committed but unflushed checkpoints are re-queued when the daemon restarts.
- `purge` restarts allocation at the configured `device_start_offset_bytes` instead of a fixed 4096.
- Free-extent allocator: coalescing, rollback of partial allocations on failure, and eviction of
  clean objects under capacity pressure instead of "pool exhausted" errors.
- RAM-tier chunk files get unique names, so re-ingesting a RAM object can no longer delete the new
  copy's data.

### Security
- The agent API refuses to bind a non-loopback address without `cluster_auth_token`.
- `file://` / `nfs://` URIs are confined to `nfs_mount_path` and the new `nfs_allowed_paths`
  (canonicalized; `..` and symlink escapes rejected).
- New controller key `allow_local_admin_bypass` (default `true`, unchanged behavior). It is set to
  `false` in the Kubernetes manifest, because sidecars and port-forwards make remote traffic look local.
- Constant-time bearer token comparison, shared by controller and agent.
- A live node's `node_id` cannot be taken over from a different source IP (HTTP 409).
- Peer URLs for P2P pulls are validated before the agent sends its token to them.
- Synthetic benchmark data is served only when globally enabled or explicitly requested for a
  `hypercache://models/benchmark/` URI. Production keys containing that substring are no longer affected.
- The UDS socket directory must be owned by the daemon user, and group/other write permission is removed.
- IPC requests are capped at 16 MiB per line.
- Kubernetes: a `NetworkPolicy` restricts the agent API to the controller and peer agents.

### Fleet controller
- Fleet operations target ONLINE nodes only. Nodes silent for `node_gc_secs` are forgotten.
- Strict node targeting: a malformed `--selector` is rejected (it previously matched every node,
  so a typo on `purge` wiped the fleet). Destructive API calls require an explicit target.
- Numeric node ranges (`gpu-2..gpu-10`) compare numerically.
- Single-host clusters work: agents bound to loopback/wildcard addresses are dialed on their
  registration source IP (previously dispatched to `http://:port`). IPv6 addresses are bracketed.
- No total request timeout for long preloads (connection timeouts remain).
- Batch preload reports real per-URI failures. The heartbeat omits an unchanged inventory and logs 401s.
- `hypercache-admin` fleet commands and `hypercache preload/rm/purge/refresh` exit non-zero on failure.

### Performance
- S3 ingest uses HEAD plus parallel ranged GETs (`s3_parallel_streams`, default 8) with an idle read
  timeout instead of a 300 s whole-request timeout.
- S3 uploads above `s3_multipart_threshold_bytes` (64 MiB) use multipart upload streamed from
  chunks (lifts the 5 GiB single-PUT limit). ListObjectsV2 pagination past 1000 keys.
- One SigV4 signer with proper URI encoding and `x-amz-security-token` (STS / IRSA / Pod Identity).
- P2P chunk transfer and checkpoint replicas stream chunk by chunk (no whole-object buffering, and
  no 2 MB request body limit).
- Python O_DIRECT reads use aligned bounce buffers (sysfs logical block size), so direct I/O actually
  engages. Fallbacks are counted in `DirectBlockReader.get_stats()`.

### Python client
- `HyperCacheClient` is thread-safe and fork-safe, buffers responses correctly, and uses
  `ingest_timeout` (default: no timeout) for calls that may download a whole object.
- Fixed `NameError` (`logger` undefined) when the daemon was unreachable.
- RAM-tier detection uses the chunk tier instead of a path substring. The shared reader is
  recreated after `fork()`.
- `HYPERCACHE_SOCKET_PATH` overrides the socket path (daemon, CLI and client).

### Telemetry
- Removed fabricated constants (estimated remote latency, CPU percentage, TTFB offsets). Unknown
  values are reported as 0.

### Build, CI and release
- New `scripts/verify_release.sh`: static checks, Rust and Python suites, end-to-end daemon / fleet /
  live S3 scenarios on the release binaries, a clean Linux container run, and a Docker image smoke test.
- New `.github/workflows/ci.yml` runs the harness on every push and pull request.
- `Cargo.lock` is committed, and builds use `--locked`. The crate is `rustfmt`-formatted
  (`max_width = 120`) and clippy-clean with `-D warnings`.
- Version 0.9.1 across Cargo, PyPI package and binaries. The binaries read their version from Cargo.
