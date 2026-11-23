# Python client: pipelined loader and raw write path

Measured 2026-11-02 on one small server. These results show behaviour and relative cost on this machine. They are not throughput claims for production hardware.

**Hardware:** Ryzen 5 5600GT, 31 GB RAM, RTX 4060 Ti 16 GB (PCIe Gen3 x8), Intel SSDPEKNU512GZ Gen3 x2 (raw, about 1.4 GB/s direct read measured with dd).

**Software:** Linux 6.8.0, torch 2.13.0+cu130, transformers 5.17.0, HyperCache 0.9.3 plus this change. No LMCache.

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

## Raw drives: page cache against O_DIRECT

With many objects written at once through the page cache, later reads from the raw drive got slower. These probes found why, with no LMCache involved. They led the Python client to write raw devices with O_DIRECT.

`benchmarks/write_concurrency_probe.py` (since removed; in git history before commit 86c1e47) wrote through HyperCache's client at a chosen parallelism, then read everything back:

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
