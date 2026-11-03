# Rust LMCache connector results

Note: the Python adapter compared here was later replaced by this connector. It remains in git history at commit 5108dcc.

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
| Python | 4 | 1312 (1311-1314) | 1533 (1531-1534) | 1.30 ms | 22 + 4 s |
| Python | 8 | 1314 (1313-1315) | 1540 (1532-1542) | 1.29 ms | 21 + 4 s |
| LMCache `raw_block` | 8 | 1367 (1363-1368) | 1536 (1533-1538) | 0.24 ms | 14 + 4 s |
| Rust, unaligned L1 | 8 | 1340 (1337-1341) | 1544 (1541-1548) | 0.53 ms | 19 + 4 s |
| Python, unaligned L1 | 4 | 1058 (1052-1063) | 1524 (1523-1525) | 1.28 ms | 23 + 13 s |

- **The drive sets the limit here.** At 8 workers, Rust stores are 3.0% faster than Python's and loads 0.5% faster. Compared with `raw_block`, Rust stores are 1.0% slower and loads 0.7% faster, while Rust also checks a CRC32 on every chunk.
- **Lookups are 2.6 times faster than Python's,** but still about twice `raw_block`'s, which doesn't ask a daemon.
- **CPU:** Rust adds about 1 s of user time over `raw_block` for this workload, and Python adds about 8 s.
- **Unaligned L1 buffers:** Rust stores are 27% faster than Python's, though Python ran 4 workers here and Rust 8. Prefer 4 KiB alignment either way, because it saves 4 s of user time on the Rust side.
- **Rust at 4 workers loads 2.3% slower than Python at 4 workers.** A Rust worker reads one object's chunks in turn, while Python reads them in parallel. Use 8 or more workers.

## On the RAM tier, with no drive limit

18 runs at 4 in flight (12 rounds plus 2 warmup) and 12 runs at 16 in flight (3 rounds plus 1 warmup). All 1,152 stores and 1,152 loads per row succeeded.

| Connector | Workers | In flight | Store | Load | Lookup p50 | CPU user + sys |
|---|---|---|---|---|---|---|
| Rust | 4 | 4 | 5404 | 6451 | 0.52 ms | 14 + 8 s |
| Rust | 8 | 4 | 6238 | 6445 | 0.55 ms | 16 + 13 s |
| Rust | 16 | 4 | 6244 | 6386 | 0.52 ms | 17 + 15 s |
| Rust | 16 | 16 | 6392 | 6454 | 1.82 ms | 20 + 19 s |
| Rust | 32 | 16 | 6354 | 6439 | 1.90 ms | 20 + 19 s |
| Python | 4 | 4 | 5248 | 5729 | 1.25 ms | 17 + 8 s |
| Python | 8 | 4 | 5748 | 6239 | 1.26 ms | 18 + 12 s |
| Python | 16 | 4 | 5461 | 6096 | 1.26 ms | 18 + 12 s |
| Python | 16 | 16 | 5718 | 6283 | 5.00 ms | 20 + 15 s |
| Python | 32 | 16 | 5650 | 6194 | 5.11 ms | 20 + 15 s |

Best against best, Rust stores are about 11% faster than Python's and loads about 3% faster. At 16 in flight, Rust's 32-key lookups take 1.8 ms against Python's 5 ms.

Neither connector scales past about 6.4 GB/s here, so this comparison also hits a limit outside the connector:

- **It isn't LMCache.** `benchmarks/lmcache_connector_direct.py`, which calls the connector with LMCache out of the path, gives the same result: store 5.5-6.5 GB/s and load 6.4-6.7 GB/s at 4 to 32 workers.
- **It isn't the checksum.** One thread computes CRC32 at about 28 GB/s, on cached data.
- **Memory is the likely cause.** With no connector or daemon involved, reading tmpfs reaches 12 GB/s on 1 thread and only 9-10 GB/s on 4 to 12 threads, and memcpy falls from 36 GB/s on 1 thread to 16 GB/s on 12. Those tests repeat over 4 MiB buffers, so they're a best case for this machine.

This means these results can't show how the connector behaves above about 6.5 GB/s. That needs a machine with more memory bandwidth.

## With vLLM

Qwen2.5-7B-Instruct-AWQ, a prompt that fills 37 chunks, one run, on LMCache 0.5.5 with the raw drive.

| | Rust connector | Python adapter (earlier run) |
|---|---|---|
| Cold (compute) | 3.549 s | 3.558 s |
| Hit in L1 | 0.199 s | 0.201 s |
| Hit on NVMe, after flushing L1 | 0.594 s | 0.619 s |
| Reading 37 chunks from the drive | 377 ms | 402 ms |
| Output identical across all three | yes | yes |

## Safety checks

`scripts/e2e/lmcache_l2_e2e.py --connector rust` passed 34 of 34 on LMCache 0.5.5 and on 0.5.6. That covers the 18 checks the Python adapter passes, plus: no-copy O_DIRECT, partial last blocks, reading the same data through the page cache, loading each other's blocks in both directions, a block deleted while locked, and concurrent tasks returning the right bytes. The Python adapter still passes 18 of 18 on both versions. `cargo test` in `lmcache-connector/` has 31 tests and runs without Python.
