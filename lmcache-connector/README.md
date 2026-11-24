# HyperCache connector for LMCache (Rust)

Lets an LMCache MP-mode server keep KV-cache chunks on HyperCache's NVMe and RAM tiers. LMCache loads it through its `native_plugin` adapter type. Below LMCache's own wrapper, all the work runs in Rust worker threads without the Python interpreter lock.

## Install

From PyPI, in the environment that runs the LMCache server (Linux x86_64, CPython 3.10 to 3.13), from the first HyperCache release that publishes it:

```sh
pip install hypercache-lmcache-native
```

Use the same version as the HyperCache daemon; the two are released together.

## Build

To build it yourself, you need a Rust toolchain and maturin. Build against the Python that runs LMCache:

```sh
cd lmcache-connector
maturin build --release -i python3.12
pip install target/wheels/hypercache_lmcache_native-*.whl
```

## Quick start with vLLM

This is the setup the vLLM results (`benchmarks/results/lmcache_rust_connector_vllm.md`) ran with: vLLM 0.33.0, LMCache 0.5.7.post1 in multiprocess (MP) mode, and HyperCache on one NVMe drive. In MP mode, vLLM sends KV cache to a separate LMCache server process. That server keeps it in memory (L1) and hands it to this connector (L2), which stores it on the drive through the HyperCache daemon.

**1. Start the daemon on a dedicated drive.** The daemon writes to the raw device, so use a drive with nothing on it: no filesystem, not mounted. Save as `kv.toml`:

```toml
device_discovery_mode = "manual"
block_devices = ["/dev/nvme1n1"]
include_partitions = false
exclude_root_device = true
storage_backing_mode = "raw_block"
chunk_size_bytes = 4194304
device_start_offset_bytes = 4096
metadata_persistence_enabled = true    # keep the KV cache across clean daemon restarts
agent_enabled = false
metrics_enabled = false
ram_cache_enabled = false
# KV blocks are local-only and never flushed. The backend defaults to S3, so point it at an
# address that answers nothing: nothing can then be uploaded by mistake.
authoritative_backend = "s3"
s3_endpoint = "http://127.0.0.1:9"
s3_region = "us-east-1"
s3_bucket = "unused"
allow_synthetic_s3_fallback = false
```

```sh
sudo hypercache -c kv.toml start      # socket: /run/hypercache/hypercache.sock
```

**2. Install LMCache and the connector where the LMCache server runs.** Use LMCache 0.5.7.post1. The `vllm/vllm-openai:v0.33.0` image ships 0.5.6, whose GPU kernels don't load there (see below), so install 0.5.7.post1 over it. Install the connector as in [Install](#install), then check the kernels:

```sh
pip install "lmcache==0.5.7.post1"
pip install hypercache-lmcache-native
python3 -c "import hypercache_lmcache_native as h; print(h.check_lmcache_gpu_ops())"   # must print None
python3 -c "import hypercache_lmcache_native as h; print(h.check_lmcache_version())"   # None on a tested LMCache
```

**3. Start the LMCache server with the connector as its L2.** Pick a `namespace` that names the model and its revision (see [Use](#use)):

```sh
lmcache server --host 127.0.0.1 --port 18555 --l1-size-gb 6 --eviction-policy LRU --chunk-size 256 \
  --l2-adapter '{"type": "native_plugin", "module_path": "hypercache_lmcache_native",
                 "class_name": "HyperCacheNativeConnector", "max_capacity_gb": 0,
                 "adapter_params": {"namespace": "qwen2.5-7b-awq-b250375", "tier": "nvme"}}'
```

**4. Start vLLM pointed at that server:**

```sh
vllm serve Qwen/Qwen2.5-7B-Instruct-AWQ --kv-transfer-config \
  '{"kv_connector": "LMCacheMPConnector", "kv_role": "kv_both",
    "kv_connector_extra_config": {"lmcache.mp.host": "127.0.0.1", "lmcache.mp.port": 18555}}'
```

**5. Check it works.** Send the same long prompt twice: the second first-token time should drop sharply. Then restart the LMCache server and vLLM, leave the daemon running, and send it again. LMCache's `Prefetch request completed` log line should count the keys as L2, which means they came from the drive. `scripts/e2e/lmcache_vllm_e2e.py` automates this and compares the answers. The connector prints any store or checksum failure to the LMCache server's stderr.

**In containers:** the LMCache server's container needs the daemon's socket directory mounted (`-v /run/hypercache:/run/hypercache`) and the drive itself (`--device /dev/nvme1n1`), because the connector reads and writes the drive directly. Without `--device`, every store fails and LMCache quietly runs on L1 alone, so check its log for failed L2 stores. With `tier: "ram"`, it needs the host's `/dev/shm` instead.

**Troubleshooting:**

| Symptom | Cause |
|---|---|
| Every store fails with "no local-only commit support" | The daemon is older than the connector. Upgrade the daemon. |
| Every store fails with a permission or "No such file" error on the drive | The process can't open the drive: add `--device`, or add its user to the daemon's `client_gid`. |
| Warning "cuda_ops compiled extension not found" | LMCache doesn't match the installed torch. Its fallback can store wrong KV data: fix the install, then change `namespace`. |
| Hits but no speedup after a restart | L1 is too small for the load to fit (LMCache only evicts L1 down to 80%), or the prompt is shorter than one 256-token chunk. |

## Use

```json
{
  "type": "native_plugin",
  "module_path": "hypercache_lmcache_native",
  "class_name": "HyperCacheNativeConnector",
  "adapter_params": {
    "namespace": "qwen2.5-7b-awq-rev1234",
    "socket_path": "/run/hypercache/hypercache.sock",
    "tier": "nvme",
    "num_workers": 32
  },
  "max_capacity_gb": 0
}
```

| Setting | Meaning |
|---|---|
| `namespace` | Required. Must change when the model weights, dtype or engine settings change. LMCache's key has a model name but no revision, so without this a weight change could return wrong KV data. |
| `socket_path` | Daemon socket. Defaults to the same search order as the Python client. |
| `tier` | `nvme` (default) or `ram`. For `ram`, LMCache must share `/dev/shm` with the daemon. |
| `num_workers` | Objects read or written at once, per direction (default 32). Each object is read or written front to back, one chunk at a time. Keep 32 on enterprise drives; set 4 on consumer or DRAM-less drives. See [Tuning](#tuning). |
| `timeout_s` | Daemon request timeout in seconds (default 30). Commits get at least 300. |
| `max_capacity_gb` | Keep at 0. The daemon owns capacity and evicts under pressure. |

## Tuning

`num_workers` is how many objects are written at once and how many are read at once. The right number depends on the drive:

- **Enterprise NVMe drives (with their own DRAM): keep the default 32.** On 4 Micron 7450 drives, 32 workers loaded fastest: 25.0 GB/s for 14 MiB blocks, against 22.5 with 16 and 16.1 with 8.
- **Consumer or DRAM-less drives: set `"num_workers": 4`.** These drives keep little of their address map on board. With many objects written at once, the data lands interleaved on flash, and with many read at once the reads turn random, so their latency jumps (from about 2 to 13 ms per request on the drive below). On one Samsung MZVL8256HFJD (Gen4 x4, no DRAM), LMCache `bench l2` in GB/s:

| Block | Loads, 4 workers | Loads, 32 (default) | Stores, 4 workers | Stores, 32 |
|---|---|---|---|---|
| 2 MiB | 4.93 | 1.62 | 3.09 | 3.16 |
| 6 MiB | 4.29 | 1.76 | 3.26 | 3.14 |
| 14 MiB | 4.65 | 2.13 | 3.27 | 3.12 |

  With vLLM on that drive, 4 workers brought the first token of a 14k-token prompt loaded from the drive from 0.65 s to 0.41 s ([results](../benchmarks/results/lmcache_rust_connector_vllm.md#worker-count)).

  Both directions need the lower count: writing with 4 and reading with 32, or the other way round, loaded at only 1.6-2.5 GB/s. With two such drives, 4 to 6 workers were fastest.
- **Not sure?** Run `lmcache bench l2` with 4, 8, 16 and 32 workers on your drives and keep the fastest. `nvme id-ctrl /dev/nvmeX | grep hmpre` prints a nonzero value on a DRAM-less drive that borrows host memory instead.

Full results, hardware and versions: [`benchmarks/results/lmcache_connector_workers.md`](../benchmarks/results/lmcache_connector_workers.md).

## Check LMCache's GPU kernels first

LMCache copies KV data between the GPU and its memory tier with compiled kernels in `lmcache.cuda_ops`. If they don't load, LMCache falls back to plain torch ops and logs a single warning:

```
lmcache.cuda_ops compiled extension not found; CudaDeviceOps stays on the torch baseline for all ops.
```

With that fallback, answers at temperature 0 changed after a cache hit on 24 or 25 of 25 test prompts, with or without HyperCache. LMCache [#6673](https://github.com/LMCache/LMCache/issues/6673) traces it to the fallback's copy not waiting for the GPU kernel that fills it, so a store can save stale KV data. HyperCache then keeps exactly those bytes, so **anything stored while the fallback was active may be wrong, and stays wrong after the fix**: change `namespace` once the kernels load. It happens in the `vllm/vllm-openai:v0.33.0` image, whose LMCache 0.5.6 was built for torch 2.13 while the image has 2.14. See `benchmarks/results/lmcache_rust_connector_xeon.md`.

Check before going live, in the environment that runs the LMCache server, with a GPU visible:

```sh
python3 -c "import hypercache_lmcache_native as h; print(h.check_lmcache_gpu_ops())"
```

It prints `None` when the kernels load, and the reason when they don't. The connector runs the same check at startup and prints a warning to stderr on a problem. It does not stop LMCache from starting. Fix it with an LMCache build that matches the installed torch. vLLM 0.31.0 with LMCache 0.5.5 is a tested pair whose answers matched. In the vLLM 0.33.0 image, LMCache 0.5.7.post1 from PyPI loads its kernels, and with batch invariance cached answers matched cold answers on 25 of 25. The connector's safety and failure tests pass on 0.5.7.post1, and answers from KV loaded off the drive matched no-cache answers on 9 of 9 prompts (`benchmarks/results/lmcache_rust_connector_vllm.md`).

## What isn't supported

- `list_l2_keys` and LMCache's IsolatedLRU quotas.
- Versions other than LMCache 0.5.5, 0.5.6 and 0.5.7.post1, which are the ones tested (`hypercache_lmcache_native.TESTED_LMCACHE_VERSIONS`). On any other version, including dev, rc and later post releases, the connector prints a warning to stderr at startup and `check_lmcache_version()` returns the reason. It still starts, since LMCache's plugin interface may or may not have changed. A local build tag such as `0.5.7.post1+cu128` counts as its release.
- A daemon without `local_only` commits. Stores fail before writing any data, and lookups miss.

## Behaviour

- Any doubt about a block is a miss. Lookup only accepts committed local-only blocks with a checksum for every chunk, and load verifies every chunk before reporting success.
- If the daemon restarts, the connector reconnects by itself. A request that couldn't be sent is resent once, and one the daemon may have received is never resent, so it reports a failure.
- The daemon does not flush the drive's write cache for KV blocks, because a lost block only costs a recompute. A clean stop (SIGINT or SIGTERM) flushes the drives and keeps every block. After a crash, `kill -9` or power cut, the next start drops every KV block, so they are misses rather than data that may never have reached the drive. Checkpoints and other objects still flush on every commit.
- Blocks are never flushed to a backend. Against an older daemon, stores fail before any data is written and lookups miss. Support is checked again on every new connection, because a restarted daemon may be a different version, and a commit only counts when the daemon confirms `LOCAL_ONLY`.
- A commit names the layout this connector staged, so if another writer replaced the block in between, the daemon refuses it and the store fails.
- Two stores of the same block in one process run one after the other, and each reports its real outcome. A block already stored counts only if its size and layout match.
- In a process forked from the one that opened the connector, closing it does nothing: the workers belong to the parent.
- A block that fails its checksum is evicted, unless it was replaced during the read.
- When a drive is full, the daemon evicts older blocks to make room, but their space only becomes reusable after `reclaim_grace_secs` (default 30), so a store can wait that long. The daemon keeps `reclaim_reserve_percent` (default 5) of each drive free ahead of need, so this is rare. When it does happen with the default `timeout_s` of 30 the store may time out and fail; that costs a recompute, never wrong data.
- On raw block devices, writes use O_DIRECT. Page-aligned L1 buffers (`--l1-align-bytes 4096`) go to and from the drive with no copy. A partial last block goes through the page cache on writes and an aligned copy on reads.

## LMCache's wrapper

LMCache's `NativeConnectorL2Adapter` wrapper sits between LMCache and this connector, which leads to some behaviour worth knowing.

- **Locks.** The wrapper tracks lookup locks itself and doesn't check them before a delete, so a locked block can be deleted. The later load is a miss, never wrong data. LMCache only calls delete from global eviction, which `max_capacity_gb: 0` turns off.
- **Status.** `report_status` doesn't include this connector's counters. Checksum failures are logged to stderr, and `adapter._client.stats()` returns the counters.
- **Store accounting.** The wrapper counts bytes per key it hasn't seen in this process, so after a restart a store of an existing block reports its bytes even though nothing was written.
- **Partial store failures.** A store task with any failed key reports zero bytes, even if other keys in it were stored.

## Test

```sh
cargo test                                   # no Python, LMCache or GPU needed
scripts/e2e/lmcache_l2_ci.sh --python python3   # 30 checks against real LMCache on a loop device
python3 scripts/e2e/lmcache_l2_e2e.py --socket /run/hypercache/hypercache.sock
python3 scripts/e2e/lmcache_vllm_e2e.py --fillers 40   # against a running vLLM + LMCache
```

`lmcache_l2_ci.sh` needs LMCache, torch (CPU is fine) and the connector wheel installed for that Python, and sudo for `losetup`; CI runs it on every pull request. `lmcache_l2_e2e.py` runs the same checks against a daemon you already started, on a real device, and damages one stored chunk on purpose. `lmcache_vllm_e2e.py` checks that answers loaded from L2 match cold ones. Results are in `benchmarks/results/lmcache_rust_connector.md`, `lmcache_rust_connector_xeon.md` and `lmcache_rust_connector_nvme.md`.
