# Rust LMCache connector results

Measured 2026-11-02 on one small server. These results show behaviour and relative cost on this machine. They are not throughput claims for production hardware.

**Hardware:** Ryzen 5 5600GT (6 cores), 31 GB RAM, RTX 4060 Ti 16 GB, Intel SSDPEKNU512GZ Gen3 x2 (raw, wiped before each run).

**Software:** Linux 6.8.0, LMCache 0.5.6 (benchmarks), LMCache 0.5.5 (vLLM run), vLLM 0.31.0, Python 3.12.3, rustc 1.99.1, HyperCache 0.9.3 plus this change.

**Method:** `lmcache bench l2`, 14 MiB objects (one 256-token Qwen2.5-7B chunk), 8 keys per task, L1 aligned to 4 KiB unless noted, order shuffled, median of 3 runs. CPU is user plus system time for the whole bench process, including the bench's own work, and excludes the daemon. Raw data is in `lmcache_l2_raw/rust-connector/`.

## On the drive

4 tasks in flight, 20 rounds plus 2 warmup. All 1,920 stores and 1,920 loads succeeded in every row, with no timeouts. MB/s, with the range across runs.

| Connector | Workers | Store | Load | Lookup p50, 32 keys | CPU user + sys |
|---|---|---|---|---|---|
| Rust | 4 | 1353 (1350-1355) | 1498 (1498-1501) | 0.52 ms | 15 + 4 s |
| Rust | 8 | 1353 (1345-1354) | 1547 (1536-1548) | 0.50 ms | 15 + 4 s |
| Rust | 16 | 1349 (1348-1353) | 1547 (1546-1549) | 0.52 ms | 15 + 4 s |
| LMCache `raw_block` | 8 | 1367 (1363-1368) | 1536 (1533-1538) | 0.24 ms | 14 + 4 s |
| Rust, unaligned L1 | 8 | 1340 (1337-1341) | 1544 (1541-1548) | 0.53 ms | 19 + 4 s |

- **The drive sets the limit here.** Compared with `raw_block`, Rust stores are 1.0% slower and loads 0.7% faster, while Rust also checks a CRC32 on every chunk.
- **Lookups take about twice as long as `raw_block`'s,** which doesn't ask a daemon.
- **CPU:** Rust adds about 1 s of user time over `raw_block` for this workload.
- **Unaligned L1 buffers** cost Rust about 1% on stores and 4 s of user time, so prefer 4 KiB alignment.
- **4 workers load 3% slower than 8** here, because a worker reads one object's chunks in turn. For tuning on other drives, see `lmcache_connector_workers.md`.

## On the RAM tier, with no drive limit

18 runs at 4 in flight (12 rounds plus 2 warmup) and 12 runs at 16 in flight (3 rounds plus 1 warmup). All 1,152 stores and 1,152 loads per row succeeded.

| Connector | Workers | In flight | Store | Load | Lookup p50 | CPU user + sys |
|---|---|---|---|---|---|---|
| Rust | 4 | 4 | 5404 | 6451 | 0.52 ms | 14 + 8 s |
| Rust | 8 | 4 | 6238 | 6445 | 0.55 ms | 16 + 13 s |
| Rust | 16 | 4 | 6244 | 6386 | 0.52 ms | 17 + 15 s |
| Rust | 16 | 16 | 6392 | 6454 | 1.82 ms | 20 + 19 s |
| Rust | 32 | 16 | 6354 | 6439 | 1.90 ms | 20 + 19 s |

The connector doesn't scale past about 6.4 GB/s here, so these results hit a limit outside it:

- **It isn't LMCache.** `benchmarks/lmcache_connector_direct.py`, which calls the connector with LMCache out of the path, gives the same result: store 5.5-6.5 GB/s and load 6.4-6.7 GB/s at 4 to 32 workers.
- **It isn't the checksum.** One thread computes CRC32 at about 28 GB/s, on cached data.
- **Memory is the likely cause.** With no connector or daemon involved, reading tmpfs reaches 12 GB/s on 1 thread and only 9-10 GB/s on 4 to 12 threads, and memcpy falls from 36 GB/s on 1 thread to 16 GB/s on 12. Those tests repeat over 4 MiB buffers, so they're a best case for this machine.

This means these results can't show how the connector behaves above about 6.5 GB/s. That needs a machine with more memory bandwidth.

## With vLLM

Qwen2.5-7B-Instruct-AWQ, a prompt that fills 37 chunks, one run, on LMCache 0.5.5 with the raw drive.

| | Rust connector |
|---|---|
| Cold (compute) | 3.549 s |
| Hit in L1 | 0.199 s |
| Hit on NVMe, after flushing L1 | 0.594 s |
| Reading 37 chunks from the drive | 377 ms |
| Output identical across all three | yes |

## Safety checks

`scripts/e2e/lmcache_l2_e2e.py` passed 34 of 34 on LMCache 0.5.5 and on 0.5.6. The checks include a flipped byte on disk becoming a miss, no-copy O_DIRECT, partial last blocks, reading the same data through the page cache, loading each other's blocks in both directions, a block deleted while locked, and concurrent tasks returning the right bytes. `cargo test` in `lmcache-connector/` has 31 tests and runs without Python.
