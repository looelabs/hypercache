# LMCache adapter and pipelined loader results

Measured 2026-11-02 on one small server. These results show behaviour and relative cost on this machine. They are not throughput claims for production hardware.

**Hardware:** Ryzen 5 5600GT, 31 GB RAM, RTX 4060 Ti 16 GB (PCIe Gen3 x8), Crucial P3 Gen3 x4 (ext4 on LVM), Intel SSDPEKNU512GZ Gen3 x2 (raw, about 1.4 GB/s direct read measured with dd).

**Software:** Linux 6.8.0, vLLM 0.31.0, LMCache 0.5.5, torch 2.13.0+cu130, transformers 5.17.0, HyperCache 0.9.3 plus this change. The contract checks also passed on LMCache 0.5.6.

## Pipelined loader: time to first token

`benchmarks/pipelined_ttft.py` with Qwen2.5-7B-Instruct (bfloat16, 15.2 GB, 30 layer groups), on the raw Intel disk, `prefetch_ahead=2`. Medians of 3 runs, with raw data in `pipelined_ttft.json`.

| Prompt tokens | First layer | T_io | T_compute | Eager TTFT | Pipelined TTFT | Saving | Limit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 5 | 880 ms | 15.06 s | 0.06 s | 15.14 s | 15.66 s | -0.52 s (-3.4%) | 0.06 s |
| 2048 | 878 ms | 15.16 s | 0.70 s | 15.88 s | 16.07 s | -0.20 s (-1.2%) | 0.70 s |
| 4096 | 881 ms | 15.21 s | 1.46 s | 16.67 s | 16.49 s | +0.18 s (+1.1%) | 1.46 s |

- Both modes produced the same next token in every run.
- An 8192-token prompt ran out of VRAM with 15 GB of weights loaded.
- The first layer took 880 ms because it is the 1 GB embedding table. The cause of pipelined being slower on short prompts was not profiled.
- The limit is `min(T_io, T_compute)`. Faster storage shrinks T_io, so a long prompt on much faster storage could save a larger share. That case was not measured.

## LMCache L2 adapter on the root disk: `lmcache bench l2`

Same Crucial disk for all three, HyperCache in filesystem arena mode with 8 workers. 14 MiB objects, 8 keys per submit, 4 in flight, 5 rounds plus 1 warm-up. Median of 3 runs, range in brackets. Page cache dropped before each run.

| Adapter | Store MB/s | Load MB/s |
|---|---:|---:|
| HyperCache | 1464 (1214-1465) | 3302 (3141-3355) |
| `fs_native`, O_DIRECT | 3003 (3003-3005) | 3405 (3395-3433) |
| `fs_native`, default | 5025 (5004-5063) | 7580 (7548-7758) |

- HyperCache verifies a CRC32 on every load chunk, and syncs and computes CRCs before reporting a store. The store gap was not profiled.
- The default `fs_native` row includes the page cache, so it does not measure the disk.
- Lookup of 32 keys: 1.3 ms for HyperCache, 0.1 ms for `fs_native`.

## LMCache L2 adapter on the raw drive

The adapter now writes to raw devices with O_DIRECT. Medians of 3 runs in shuffled order, on the same Intel drive, wiped before each run, on LMCache 0.5.6. Same settings as the tables below. Raw JSON is in `lmcache_l2_raw/lmcache-0.5.6-odirect/`.

| Adapter | Workers | L1 alignment | Store MB/s | Load MB/s |
|---|---:|---:|---:|---:|
| LMCache `raw_block` | 8 store, 8 load | 4096 | 1367 (1362-1368) | 1538 (1533-1538) |
| HyperCache | 2 | 4096 | 1310 (1309-1312) | 1452 (1451-1452) |
| HyperCache | 4 (default) | 4096 | 1313 (1313-1314) | 1532 (1531-1532) |
| HyperCache | 8 | 4096 | 1314 (1311-1315) | 1532 (1531-1537) |
| HyperCache | 16 | 4096 | 1310 (1309-1311) | 1533 (1528-1541) |
| HyperCache | 4 | 1 (unaligned buffers) | 1071 (1066-1072) | 1527 (1525-1533) |

- **Against `raw_block`:** at the default 4 workers, loads are within 0.4% and stores about 4% slower. HyperCache also verifies a CRC32 per chunk on every load.
- **Against the page-cache path:** at 4 workers, store rose from 1098 to 1313 MB/s and load from 1390 to 1532 MB/s. At 8 workers, load rose from 1144 to 1532 MB/s.
- **Worker count no longer lowers load speed.**
- **Unaligned L1 buffers** cost about 18% on stores because of the extra copy. LMCache's default L1 alignment is 4096, so the aligned rows are the expected case.
- These loads are near this drive's limit of about 1.6 GB/s, so a faster drive could show differences this one hides.
- `raw_block` with io_uring was not measured, because it could not start in this container.

### Before O_DIRECT

These runs used the page-cache write path. They are kept because they show the problem that led to the change.

All rows use the same Intel drive, which was discarded with `blkdiscard` and had caches dropped before every run. 14 MiB objects, 8 keys per submit, 4 in flight, 20 rounds plus 2 warm-up. Median of 3 runs, range in brackets. Every store and load succeeded. Raw JSON is in `lmcache_l2_raw/`.

**LMCache 0.5.6:**

| Adapter | Mode | Workers | Store MB/s | Load MB/s |
|---|---|---:|---:|---:|
| LMCache `raw_block` | Raw | 2 store, 4 load (default) | 1327 (1324-1363) | 1534 (1531-1537) |
| LMCache `raw_block` | Raw | 8 store, 8 load | 1366 (1361-1369) | 1534 (1532-1535) |
| HyperCache | Raw | 2 | 906 (905-907) | 1423 (1408-1426) |
| HyperCache | Raw | 4 | 1098 (1095-1098) | 1390 (1375-1394) |
| HyperCache | Raw | 8 | 1127 (1125-1131) | 1144 (1143-1144) |
| LMCache `fs_native`, O_DIRECT | ext4 | 4 (default) | 1327 (1325-1332) | 1513 (1508-1513) |

**LMCache 0.5.5:**

| Adapter | Mode | Workers | Store MB/s | Load MB/s |
|---|---|---:|---:|---:|
| HyperCache | Filesystem arena on ext4 | 8 | 1064 (1061-1082) | 1500 (1496-1501) |
| LMCache `fs_native`, O_DIRECT | ext4 | 8 | 1343 (1340-1345) | 1514 (1511-1515) |

- **HyperCache raw with 4 workers against `raw_block`:** loads are about 9% slower and stores about 20% slower. With 8 workers, loads are about 25% slower.
- **HyperCache's raw mode is not faster than its arena mode here.** At 8 workers the arena loaded at 1500 MB/s against 1144 for raw. Single runs at 4 workers gave 1445 against about 1390. The arena stored more slowly in both.
- **Not measured:** `raw_block` with io_uring failed to start in this container ("Operation not permitted").
- **Setup differences:** `raw_block` ran with a fixed 14 MiB slot and periodic metadata checkpoints turned off, which may favour it. HyperCache computes and verifies a CRC32 per chunk; a direct probe put verification at about 3% of read throughput. A search of `raw_block`'s source found CRC checks only on its metadata checkpoints, not on KV data. Durability was not tested for either adapter.

### Why more workers lowered raw load speed

`benchmarks/write_concurrency_probe.py` writes through HyperCache's client at a chosen parallelism, then reads everything back, with no LMCache involved:

| Objects written in parallel | Write MB/s | Read MB/s (4, 8 or 16 parallel readers) |
|---:|---:|---:|
| 1 | 812-815 | 1574-1617 |
| 8 | 1179 | 1225-1251 |
| 16 | 1194 | 1133-1153 |

Read parallelism made almost no difference. Write parallelism did.

`benchmarks/raw_write_probe.py` then tested the write method directly, on a freshly wiped drive for each of 3 runs, in shuffled order, with CRC checks on during reads. Medians, 512 objects of 14 MiB. Raw data is in `raw_write_probe.jsonl`.

| Write method | Objects in parallel | Write MB/s | Read MB/s | Commit median |
|---|---:|---:|---:|---:|
| Page cache, shared chunk pool (today's path) | 1 | 817 | 1616 | 10 ms |
| Page cache, shared chunk pool (today's path) | 4 | 1114 | 1420 | 39 ms |
| Page cache, shared chunk pool (today's path) | 8 | 1189 | 1214 | 77 ms |
| Page cache, shared chunk pool (today's path) | 16 | 1215 | 1134 | 158 ms |
| Page cache, chunks in order | 8 | 1089 | 1175 | 81 ms |
| Page cache, one write per object | 8 | 1090 | 1183 | 80 ms |
| O_DIRECT, shared chunk pool | 8 | 1363 | 1604 | 20 ms |
| O_DIRECT, chunks in order | 8 | 1394 | 1611 | 13 ms |
| O_DIRECT, one write per object | 4 | 1387 | 1605 | 16 ms |
| O_DIRECT, one write per object | 8 | 1392 | 1610 | 31 ms |
| O_DIRECT, one write per object | 16 | 1394 | 1616 | 58 ms |

- **Writing through the page cache is what lowers later read speed.** With O_DIRECT, reads hold at about 1610 MB/s at every parallelism tested.
- **Chunk order is not the cause.** Writing chunks in order did not help through the page cache, and O_DIRECT did not need it.
- **O_DIRECT also improves writes,** from 1189 to about 1390 MB/s at 8 parallel objects.
- Why page-cache writeback has this effect on this drive was not investigated, so it may differ on other drives. These are probe numbers without LMCache.

## Real serving path: vLLM, LMCache, adapter

`scripts/e2e/lmcache_vllm_e2e.py` with Qwen2.5-7B-Instruct-AWQ, a prompt of about 9,500 tokens, 3 GB L1, HyperCache on the raw Intel disk.

| Request | TTFT | Served from |
|---|---:|---|
| Cold | 3.55 s | Computed |
| Repeat | 0.20 s | L1 |
| After filling L1 with other prompts | 0.62 s | NVMe: 37 of 37 chunks in 402 ms |

- These figures are with O_DIRECT writes. The NVMe case is from one run; before O_DIRECT, one run gave 0.68 s and 459 ms.
- The cold and repeat figures matched across every run.
- Generated text was identical across all three requests.
- With a 1 GB L1, the NVMe load could not get L1 buffers (LMCache only evicts down to 80%), so vLLM recomputed. The result was correct but slow, so size L1 above about 5x the largest expected load.

## Safety checks

`scripts/e2e/lmcache_l2_e2e.py` passed 18 of 18 on LMCache 0.5.5 and 0.5.6, with the real daemon on the raw disk. This includes a flipped byte on disk, which became a miss and was removed, and the same extents rewritten alternately through the page cache and O_DIRECT, with both read paths seeing the latest data each time. The daemon was pointed at an S3 endpoint with nothing listening, and logged no flush attempts across these runs.
