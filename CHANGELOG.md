# Changelog

## 0.9.3

Raw-block access for non-root clients and up-front arena sizing, tested on bare-metal Linux,
Docker and Kubernetes.

### Fixes
- Filesystem arenas are reserved on disk at startup (`fallocate`) instead of created sparse. An arena
  that does not fit its filesystem stops the daemon immediately with the required and free bytes,
  instead of surfacing later as failed ingests once the disk fills. Existing sparse arenas are filled
  in place. New key `filesystem_arena_preallocate` (default `true`; `false` keeps sparse arenas).
- `hypercache check` inspects filesystem arenas without creating or reserving them, and reports an
  arena that will not fit as FAIL.
- In raw-block mode, non-root clients could not open the NVMe devices. With `client_gid` set, the
  daemon now shares the devices it claims (`0660`, client group) at every start.
- CI: the e2e script hardcoded the version (`0.9.1`) and failed on the 0.9.2 bump. It now reads
  `Cargo.toml`, and the static stage rejects literal versions in scripts and tests. The YAML check
  reports SKIP instead of PASS when PyYAML is missing.

### Docs
- README: raw-block mode with non-root clients (udev rule, Docker `--device`, Kubernetes device
  policy, security note) and arena sizing. Manifest and config template comments updated.

### Tests
- `scripts/e2e/bare_metal_access.sh` (new, `e2e` stage on Linux as root / passwordless sudo, and in
  the privileged `linux` stage): raw loop device shared with `client_gid`, non-root O_DIRECT reads
  and raw-device checkpoints, refusal outside the group, `check` and startup refusal for an oversized
  arena on a 128 MiB ext4 filesystem, full reservation, sparse opt-out, SIGTERM.
- `container_host_access.sh`: raw-block mode with a privileged daemon and a `--device` client
  container; container exits when the arena does not fit a tmpfs.
- `k8s_smoke.sh`: raw-block mode as documented (privileged agent, host `/dev`) with a privileged
  non-root training pod, the unprivileged-pod limitation, and an oversized arena keeping the agent
  from starting.
- Rust: arena preallocation, capacity errors, sparse opt-out, `check` without side effects.

## 0.9.2

Containers and Kubernetes: a daemon running in a container (as root) is now usable from the host
and from other pods by unprivileged users, and stops cleanly when the runtime asks it to.

### Fixes
- The daemon and the controller now shut down cleanly on SIGTERM. As a container's PID 1 they ignored
  it, so `docker stop` and pod deletion waited out the grace period (10 s / 30 s), SIGKILLed the
  daemon and left a stale socket behind.
- RAM-tier checkpoints written by a client with its own `/dev/shm` (any client outside the daemon's
  container) were silently lost: the client created a private copy and the commit still succeeded.
  The daemon now pre-creates staged RAM chunk files, the client never creates them, and
  `commit_write` rejects RAM chunks that were never written.
- `storage_backing_mode = "filesystem"` no longer falls back to scanning raw block devices when no
  arena can be provisioned. In a privileged pod, that scan could select the host's unmounted disks.
- A non-root client that cannot open the socket now gets an error naming `client_gid` instead of
  "Is the daemon running?". NFS confinement errors say when `nfs_mount_path` does not exist inside
  the daemon's filesystem (share not mounted into the container).

### New
- `client_gid` (agent): the numeric group that local clients run with. The socket, filesystem
  arenas and RAM-tier files (and the RAM directory, setgid) are shared with it read/write.
  Unset keeps the previous behavior (daemon user only).
- Kubernetes manifest: `client_gid = 2000`, readiness/liveness probes for the agent
  (`hypercache status`) and the controller (TCP), RAM-tier hostPath instructions, and an example
  training pod with the required mounts and `supplementalGroups`.
- README: running the daemon in Docker with host access.
- `scripts/e2e/container_host_access.sh` (in the `image` stage, now also run by CI) and
  `scripts/e2e/k8s_smoke.sh` (new `k8s` stage, minikube).

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
