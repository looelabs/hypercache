# Rust connector with vLLM on NVMe

Measured 2026-11-22 and 23. These are end-to-end tests: a real model on a GPU, with LMCache storing
its KV cache through the Rust connector to a drive. KV cache is the attention state the model
computes for a prompt. Loading it instead of recomputing it is what shortens the time to the first
output token (TTFT). Raw data is in `lmcache_l2_raw/vllm-4060ti/`.

**Hardware:** AMD EPYC 7443 (24 cores), 46 GB DDR4, NVIDIA RTX 4060 Ti 16 GB (PCIe Gen4 x8), one
Samsung MZVL8256HFJD drive (256 GB, Gen4 x4, no DRAM cache), discarded before each run.

**Software:** HyperCache at `aad6452` with the RAM tier off, `vllm/vllm-openai:v0.33.0` (vLLM 0.33.0,
torch 2.14.0+cu130) with LMCache 0.5.7.post1 from PyPI in place of the image's 0.5.6, LMCache in
multiprocess mode with 256-token chunks, Qwen2.5-7B-Instruct-AWQ (revision `b250375`),
`--gpu-memory-utilization 0.6`, `--max-model-len 16000`. LMCache's compiled GPU kernels loaded in
every run (no fallback warnings). With the image's own LMCache 0.5.6 they don't, and its fallback can
store wrong KV data; see the connector README.

## Safety

Before the timed runs, the connector's safety script passed 30 of 30 checks on one drive and on two.
The 19 failure tests passed with no wrong results: daemon restarts and kills, a frozen daemon,
corrupted chunks, deletes during loads and lookups, and eviction under pressure. Under that
pressure, 16 of about 4,600 stores failed to commit; each was a miss, never wrong data. No media or
PCIe errors were logged, and the drives stayed under 47 °C.

## First-token time, one request at a time

Three prompts at each length, median of the three, 32 output tokens at temperature 0, vLLM's own
prefix cache off so every hit comes from LMCache. "From the drive" means LMCache and vLLM restarted
with an empty LMCache memory cache (L1) and the page cache dropped, so every load came from the drive
(LMCache's log showed 0 loads from L1).

| Setup | 2k tokens | 8k tokens | 14k tokens |
|---|---|---|---|
| No cache | 0.638 s | 2.869 s | 5.470 s |
| LMCache + Rust, first send (stores to the drive) | 0.644 s | 2.871 s | 5.476 s |
| From LMCache L1 (memory) | 0.103 s | 0.169 s | 0.229 s |
| **From the drive** | **0.160 s (4.0x)** | **0.450 s (6.4x)** | **0.649 s (8.4x)** |

Storing costs almost nothing on the first send. The output text matched the no-cache answer on 9 of
9 prompts in every phase, including from the drive.

At 14k tokens the drive adds about 0.42 s over L1 for about 770 MB of KV, about 1.8 GB/s. That is
this drive's limit, not the connector's or LMCache's (see [Load speed](#load-speed)). Also, a second
send right after a load from the drive loaded from the drive again: LMCache 0.5.7.post1 didn't keep
those chunks in L1.

## Load speed

Measured 2026-11-23 on the same machine, HyperCache at `a5bfbc9`, page cache dropped before each read.

- **The drive is the limit.** On data that has been written, fio reads this drive at 2.1 to 3.1 GB/s
  with 4 MiB random reads from 8 or 32 threads, 4.1 GB/s with 32 reads in flight from one thread,
  and 5 to 6 GB/s sequentially. On discarded blocks it reports about 7 GB/s, but the drive
  answers those without reading flash, so that number doesn't apply to stored KV.
- **The connector reaches that limit.** Loading 55 blocks of 14 MiB (one 14k-token prompt) with the
  connector alone, no LMCache, ran at 1.9 to 2.6 GB/s with 8, 32 or 64 workers, aligned or
  unaligned buffers. The 55 daemon lookups took 2.7 ms in total.
- **Placement isn't the cause.** Most blocks are already stored contiguously (51 of 55 when stored 32
  at a time), and reading each block in one request instead of 4 MiB chunks was no faster.

A drive with DRAM, or a data-center drive, should load faster. That isn't measured yet.

## Two drives

The same single-request test, run again in one session on one drive and then striped over two:
the drive above and a Samsung MZVL8256HEJD (256 GB, Gen4 x4, no DRAM cache, slower). The daemon
places a block's 4 MiB chunks on alternating drives. HyperCache at `a5bfbc9`, everything else as
above. Raw data: `ttft_2026-11-23_one_drive.jsonl` and `ttft_2026-11-23_two_drives.jsonl`.

| From the drive(s) | 2k tokens | 8k tokens | 14k tokens |
|---|---|---|---|
| No cache | 0.637 s | 2.871 s | 5.470 s |
| One drive | 0.159 s | 0.446 s | 0.644 s |
| **Two drives** | **0.127 s** | **0.384 s** | **0.598 s** |

Two drives cut the first-token time by 7 to 20%. LMCache's load of the 55 blocks at 14k tokens fell
from about 418 ms to about 370 ms. The gain is small because each drive is limited as above and the
second drive is the slower model. Output text matched the no-cache answer on 9 of 9 prompts in every
phase on both setups, no store failed, no media errors were logged, and the drives stayed under 48 °C.

## Many requests at once

LMCache's own benchmark, `lmcache bench engine --workload long-doc-qa`: 22 documents of 10,000
tokens, 3 questions about each in random order (66 requests per level), 32 output tokens, a 12 GB
KV working set against a 6 GB LMCache L1, warm-up on. TTFT p50 / p99; all 66 requests succeeded in
every cell:

| Requests at once | No cache | vLLM prefix cache | LMCache L1 only (6 GB) | **LMCache L1 + Rust on the drive** |
|---|---|---|---|---|
| 1 | 3.64 / 3.67 s | 3.65 / 3.68 s | 3.64 / 3.68 s | **0.50 / 0.62 s** |
| 4 | 5.88 / 12.36 s | 3.83 / 9.36 s | 3.83 / 10.60 s | **0.64 / 0.91 s** |
| 8 | 19.76 / 27.49 s | 15.11 / 23.15 s | 15.23 / 25.06 s | **1.40 / 2.09 s** |
| 16 | 50.31 / 58.15 s | 38.38 / 47.93 s | 37.20 / 46.26 s | **3.60 / 4.27 s** |

With the drive, every request hit, against about a third with L1 alone, and most hits loaded
entirely from the drive (at 16 at once: 76 from the drive, 7 from L1, 5 from both, counting the 22
warm-up sends). No store failed. Against no cache, the median is 7.3 times shorter at 1 request,
9.1 at 4, and 14 at 8 and 16. The GPU's own KV space at `0.6` holds only about 5 documents, which is
why vLLM's prefix cache and a 6 GB L1 help so little here.

Limits of this test: one run per cell, setups in a fixed order (no cache, prefix, L1, Rust), and
what limits Rust at 16 at once (GPU queueing or the load path) isn't measured. The documents are
repeated filler tokens, so output text wasn't compared; the single-request test above covers that.

**Not done yet:** LMCache's `raw_block` adapter under the same tests, faster drives, and longer documents.
