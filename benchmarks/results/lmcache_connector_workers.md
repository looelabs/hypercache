# LMCache connector: worker count on DRAM-less drives

**Hardware:** AMD EPYC 7443 (24 cores), 46 GB RAM, Samsung MZVL8256HFJD (256 GB, Gen4 x4, no DRAM cache, uses a 64 MiB host memory buffer), discarded before each run. Two-drive runs add a Samsung MZVL8256HEJD (same type, about half as fast), so that pair is not identical and the slower drive sets the pace.

**Software:** HyperCache 0.10.1 with the metadata manifest on, LMCache 0.5.7.post1, Python 3.12, torch 2.15.0+cpu. Connector 0.10.0 for the worker sweeps; test builds of 0.10.1 for the per-drive cap and the split store/load counts, which aren't shipped.

**Method:** `lmcache bench l2`, 8 keys per task, 16 tasks in flight, L1 aligned to 4 KiB, 2 warmup rounds, rounds scaled to 8 x 14 MiB, median of 3 runs. Every run checks that all keys were stored and loaded, and verifies the data. `iostat -x` ran during the two-drive sweep (including its one-drive comparison runs) and the later runs. 14 MiB is one 256-token Qwen2.5-7B chunk.

## Summary

- **On this drive, 4 workers loaded 2.2 to 3.0 times as fast as the default 32**, and 40 to 120% faster than LMCache's `raw_block` adapter. Stores were the same (3.1-3.3 GB/s).
- **Both directions need the lower count.** Each worker writes or reads one object front to back. With many objects written at once, the data lands interleaved on flash; with many read at once, the reads turn random. Writing with 4 and reading with 32, or the other way round, loaded at only 1.6-2.5 GB/s.
- **The drive's latency is what collapses.** With 8 or more workers, its read latency rises from about 2 to 13 ms per request (20 ms on the slower drive).
- **Enterprise drives behave the other way.** On 4 Micron 7450 drives, 32 workers loaded fastest (25.0 GB/s, against 16.1 with 8; [results](lmcache_rust_connector_nvme.md#worker-count)). So the default stays 32, and the advice for DRAM-less drives is `"num_workers": 4`.
- **Limiting reads per drive doesn't replace it.** A prototype cap on chunk reads in flight per drive, with 32 workers, reached only about half the 4-worker speed.

## Workers, one drive

Loads in GB/s, `raw_block` (posix, 8 workers) for reference.

| Block | 1 | 2 | 4 | 6 | 8 | 16 | 32 | `raw_block` |
|---|---|---|---|---|---|---|---|---|
| 2 MiB | 2.30 | 3.38 | **4.93** | 4.32 | 4.15 | 1.56 | 1.62 | 3.53 |
| 6 MiB | 2.94 | 4.38 | **4.29** | 3.32 | 1.76 | 1.71 | 1.76 | 2.10 |
| 14 MiB | 2.85 | 4.69 | **4.65** | 2.60 | 1.73 | 2.12 | 2.13 | 2.10 |

Stores at 4 workers: 3.09, 3.26 and 3.27 GB/s at 2, 6 and 14 MiB; at 32: 3.16, 3.14 and 3.12.

## Workers, two drives (not identical)

| Block | 2 | 4 | 6 | 8 | 12 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| 2 MiB | 3.64 | 4.71 | **5.47** | 5.13 | 2.46 | 2.13 | 2.03 |
| 6 MiB | 3.88 | 5.24 | **5.44** | 3.10 | 2.08 | 2.04 | 1.99 |
| 14 MiB | 4.32 | **6.19** | 4.95 | 2.01 | 2.12 | 2.37 | 2.22 |

Stores: 2.8 GB/s from 4 workers up, capped by the slower drive.

## Store and load workers separately (test build)

Loads in GB/s; stores were 3.1-3.3 GB/s in every cell. Only 4 workers on both sides reaches full speed.

| Block | Store 4, load 4 | Store 4, load 32 | Store 32, load 4 | Store 32, load 32 |
|---|---|---|---|---|
| 2 MiB | **4.67** | 1.63 | 2.48 | 1.60 |
| 14 MiB | **4.65** | 2.22 | 2.49 | 2.18 |

## What the drive does

`iostat -x` during 14 MiB loads. Each 4 MiB chunk read reaches the drive as 32 requests of 128 KiB.

| Setup | Drive | MB/s | Read latency | Queue depth |
|---|---|---|---|---|
| 1 drive, 4 workers | faster | 2987 | 2.2 ms | 58 |
| 1 drive, 8 workers | faster | 1464 | 13.2 ms | 155 |
| 2 drives, 4 workers | faster / slower | 2118 / 2121 | 0.7 / 2.0 ms | 12 / 38 |
| 2 drives, 8 workers | faster / slower | 875 / 874 | 0.8 / 20.5 ms | 6 / 148 |

## Per-drive read cap (test build, not shipped)

32 workers with at most N chunk reads in flight per drive, loads in GB/s.

| Block | One drive, no cap | Cap 2 | Cap 4 | Cap 6 | Two drives, no cap | Cap 2 | Cap 4 |
|---|---|---|---|---|---|---|---|
| 2 MiB | 1.62 | 2.31 | 2.37 | 2.33 | 2.03 | 3.24 | 3.20 |
| 6 MiB | 1.76 | 2.44 | 2.46 | 2.25 | 1.99 | 3.31 | 3.26 |
| 14 MiB | 2.13 | 2.47 | 2.48 | 2.04 | 2.22 | 3.27 | 3.24 |

## Setting it

In LMCache's `adapter_params`, `"num_workers": 4` on consumer or DRAM-less drives; leave the default 32 on enterprise drives. To check a drive: `nvme id-ctrl /dev/nvmeX | grep hmpre` prints a nonzero value on a DRAM-less drive that borrows host memory.
