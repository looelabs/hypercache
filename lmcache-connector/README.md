# HyperCache connector for LMCache (Rust)

Lets an LMCache MP-mode server keep KV-cache chunks on HyperCache's NVMe and RAM tiers. LMCache loads it through its `native_plugin` adapter type. Below LMCache's own wrapper, all the work runs in Rust worker threads without the Python interpreter lock.

## Build

Needs a Rust toolchain and maturin. Build against the Python that runs LMCache:

```sh
cd lmcache-connector
maturin build --release -i python3.12
pip install target/wheels/hypercache_lmcache_native-*.whl
```

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
| `num_workers` | Objects read or written at once, per direction (default 32). Each object is read one chunk at a time, so this sets the queue depth. On one drive, 8 loaded fastest. On 4 drives, 32 did. |
| `timeout_s` | Daemon request timeout in seconds (default 30). Commits get at least 300. |
| `max_capacity_gb` | Keep at 0. The daemon owns capacity and evicts under pressure. |

## What isn't supported

- `list_l2_keys` and LMCache's IsolatedLRU quotas.
- Versions other than LMCache 0.5.5 and 0.5.6, which are the ones tested. Nothing in the connector enforces that yet.
- A daemon without `local_only` commits. Stores fail before writing any data, and lookups miss.

## Behaviour

- Any doubt about a block is a miss. Lookup only accepts committed local-only blocks with a checksum for every chunk, and load verifies every chunk before reporting success.
- If the daemon restarts, the connector reconnects by itself. A request that couldn't be sent is resent once, and one the daemon may have received is never resent, so it reports a failure.
- Blocks are never flushed to a backend. Against an older daemon, stores fail before any data is written and lookups miss.
- A block that fails its checksum is evicted, unless it was replaced during the read.
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
python3 scripts/e2e/lmcache_l2_e2e.py --socket /run/hc/hc.sock
```

The second needs a running daemon on a real device, and damages one stored chunk on purpose. Results are in `benchmarks/results/lmcache_rust_connector.md`, `lmcache_rust_connector_xeon.md` and `lmcache_rust_connector_nvme.md`.
