# Changelog

## Unreleased

Faster stores for KV cache and for many small commits.

### Changed
- KV cache blocks (`LOCAL_ONLY` commits, used by the LMCache connector and `HyperCacheKVStore`) no
  longer wait for a device flush. A lost block only costs a recompute, so instead of flushing every
  commit the daemon keeps a marker file next to the manifest while it runs. A clean shutdown
  (SIGINT or SIGTERM) flushes every device and removes it; a start that finds it (after a crash,
  `kill -9` or power cut) drops every KV block, which become misses. Checkpoints and other objects
  still flush before their commit returns.
- Concurrent commits share device flushes (group commit). Each commit still returns only after a
  flush that started after it arrived, so their durability is unchanged.
- LMCache L2 benchmark on one Gen4 drive (EPYC 7443, LMCache 0.5.7.post1, connector 0.10.0), stores
  in GB/s, 0.10.0 -> this release: 2 MiB blocks 2.03 -> 3.18, 6 MiB 2.72 -> 3.25, 14 MiB 2.98 -> 3.28,
  level with LMCache's `raw_block` (3.18-3.24) and `fs_native` (3.10-3.23), which never flush.

### Upgrading
- No configuration change is needed. KV blocks stored by 0.10.0 are kept after the upgrade.
- KV blocks no longer survive a crash or power cut, only a clean stop.
- Before rolling back to 0.10.0 after a crash, start this release once and stop it cleanly: 0.10.0
  ignores the marker and would keep KV blocks whose data may not have been flushed.

## 0.10.0

Adds the Rust LMCache connector, which keeps LMCache's KV cache on HyperCache's NVMe and RAM tiers,
published to PyPI as `hypercache-lmcache-native`.

### Upgrading
- `HyperCacheKVStore` no longer uploads blocks to the backend by default: commits are `LOCAL_ONLY`.
  Pass `write_through=True` to keep uploading.
- The daemon and the connector must both be 0.10.0 or later: a daemon without local-only commits
  makes every connector store fail and every lookup miss.
- Rolling back the daemon to 0.9.4 keeps all objects except the local-only ones; see the manifest
  note below.

### Added
- `lmcache-connector/`: an L2 connector for LMCache multiprocess mode, written in Rust and loaded
  through LMCache's `native_plugin` type. Install with `pip install hypercache-lmcache-native`. Tested on LMCache 0.5.5,
  0.5.6 and 0.5.7.post1 and vLLM 0.31.0 and 0.33.0. At startup it warns when LMCache's compiled GPU kernels
  (`lmcache.cuda_ops`) don't load, as in the `vllm/vllm-openai:v0.33.0` image: LMCache's torch
  fallback changed model output after a cache hit. `check_lmcache_gpu_ops()` runs the same check.
- `commit_write` accepts `local_only`, `chunk_checksums` and `generation`. A `LOCAL_ONLY` object is
  never flushed to a backend and stays evictable. A local-only commit is refused unless the object is
  still `STAGED` with the `generation` its writer was given, and a `sha256`, when sent, is verified.
  The reply's `sync_status` confirms `LOCAL_ONLY`. All three fields are optional on the wire; older
  clients are unaffected.
- **Rolling back:** local-only objects are saved in a separate part of the metadata manifest that
  older daemons skip, so rolling back to 0.9.4 keeps every other object, including ones not yet
  flushed. The older daemon doesn't know the local-only objects, so their space isn't reused until
  the arena is reset. Upgrading reads 0.9.4 manifests as before.
- The connector is published to PyPI as `hypercache-lmcache-native` (manylinux_2_28 x86_64 wheels for
  CPython 3.10 to 3.13, and an sdist), at the same version as HyperCache. `release.yml` publishes it
  with PyPI trusted publishing, and CI builds the same wheels on every pull request.
- `locate` reports `sync_status` for HITs, so a reader can refuse an object that is still staged.
- `DirectBlockWriter(direct_on_block_devices=True)` writes to raw block devices with O_DIRECT, falling
  back to the page cache for anything it can't take. Off by default.
- `benchmarks/pipelined_ttft.py`, `benchmarks/write_concurrency_probe.py`, `benchmarks/raw_write_probe.py` and
  `benchmarks/results/lmcache_and_ttft.md`.
- `benchmarks/lmcache_connector_direct.py`, `benchmarks/results/lmcache_rust_connector.md`,
  `benchmarks/results/lmcache_rust_connector_xeon.md` and
  `benchmarks/results/lmcache_rust_connector_nvme.md`.


### Fixes
- `HyperCacheKVStore` commits are now `LOCAL_ONLY` unless `write_through` is set. Previously every
  commit started a backend upload of prompt-derived data, ignoring `auto_flush_s3=False`, and a failed
  upload left the block dirty and impossible to evict. Reads now verify a CRC32 on every chunk and
  return a block only when it is `LOCAL_ONLY` (or `SYNCED_BACKEND` for write-through); anything else
  is a miss, so a block evicted and reused during a read is never returned. A store that fails is
  evicted.
- `HyperCacheClient.commit_write(local_only=True)` returns False unless the daemon confirms
  `LOCAL_ONLY`, and `stage_write` results carry the layout `generation`.
- `DirectBlockWriter.write_chunks` waits for every chunk write to finish before raising an error.
- `DirectBlockWriter` retries short writes instead of ignoring them, no longer copies each chunk, and
  no longer leaks a file descriptor when two threads open the same device.

### Docs
- Corrected the pipelined loader claims. It does not shorten time to first token by itself: on a
  15 GB model the change was between -3.4% and +1.1%, and the first layer took 880 ms, not milliseconds.
- Fixed the pipelined loader README example, which did not match the API.
- Removed the claims of LMCache and Dynamo KVBM readiness. LMCache now has a real adapter; Dynamo
  KVBM has no integration.

## 0.9.4

Faster loads and stores for objects that don't fill a whole stripe.

### Fixes
- Each object now starts on its own drive, chosen from a hash of its URI. Previously every object
  started on the first drive. Objects that don't fill a whole stripe left the last drives underused,
  and objects smaller than one chunk all landed on the first drive, however many drives were in the
  pool. Measured through LMCache's L2 benchmark on 4 Micron 7450 Gen4 drives, 32 workers, median of
  3 runs: 2 MiB objects load at 24609 MB/s (was 6321) and store at 15390 MB/s (was 4962), and 14 MiB
  objects load at 24884 MB/s (was 20740) and store at 18549 MB/s (was 16129).
- Upgrading needs no migration. Objects already stored keep their placement and stay readable, and
  objects stored after the upgrade stay readable if you roll back. Existing objects aren't rebalanced.
- CI: the bare-metal e2e script loaded its client from the checkout, which uid 1000 can't read on a
  GitHub runner. It now runs the client from a world-readable work dir.
- Release: `release.yml` now stops before building or publishing if the release tag doesn't match
  the version files.

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
