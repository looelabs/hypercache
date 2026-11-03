# Rust LMCache connector results: Xeon VM, 4 network disks

Note: the Python adapter compared here was later replaced by this connector. It remains in git history at commit 5108dcc.

Measured 2026-11-03 on a rented cloud VM. Raw data is in `lmcache_l2_raw/rust-connector-xeon/`. Earlier results from a small server are in `lmcache_rust_connector.md`.

**Hardware:** Xeon 6776P (KVM guest, 24 vCPUs), 214 GB RAM, RTX PRO 6000 Blackwell (97,887 MiB), 4 virtio network disks of 2.3-2.4 TiB, sold as capped at 2 GB/s each.

**Software:** Linux 6.11.0, LMCache 0.5.6, vLLM 0.33.0, Python 3.12.3, rustc 1.99.1, HyperCache 0.9.3 plus this change.

**Method:** `lmcache bench l2`, 14 MiB objects (one 256-token Qwen2.5-7B chunk), 8 keys per task, L1 aligned to 4 KiB, order shuffled, median of 3 runs with the range in brackets. Every run was checked for the expected key count; all succeeded with no timeouts. CPU is user plus system time for the whole bench process, including the bench's own work, and excludes the daemon. The disks can't discard, so runs overwrite earlier data instead of starting wiped.

## What the VM can do without HyperCache

fio, 4 MiB blocks, queue depth 16 per drive, 3 runs of 20 s. The drives can't be tested in isolation from the VM's limits, and the disks' provider wasn't asked about them.

| Drives | Read MB/s | Write MB/s |
|---|---|---|
| 1 | 2157-2182 | 993-1478 |
| 2 | 4303-4338 | 1933-2418 |
| 3 | 5495-6019 | 2080-2219 |
| 4 | 5688-5899 | 1988-2240 |

- **Four disks don't reach 8 GB/s.** Reads stop near 6 GB/s from 3 drives on, and writes stop near 2.2 GB/s from 2 drives on. Latency rises as they flatten and fio's CPU use stays low, which points to a shared limit on the VM, though that isn't confirmed.
- **Memory is not a limit here.** tmpfs reads reach 97-99 GB/s on 12 and 24 threads, against 9-10 GB/s on the earlier server. memcpy reaches 130 GB/s.

## RAM tier, no drive limit

16 tasks in flight, 50 rounds plus 2 warmup, about 94 GB per phase.

| Connector | Workers | Store MB/s | Load MB/s | Lookup p50 | CPU user + sys |
|---|---|---|---|---|---|
| Rust | 8 | 12169 [12166-12773] | 22366 [22092-22445] | 3.77 ms | 40 + 77 s |
| Rust | 16 | 16785 [16446-16809] | 33696 [33690-33959] | 3.80 ms | 49 + 103 s |
| Rust | 32 | 19354 [19169-19579] | 39468 [39343-39471] | 4.17 ms | 57 + 130 s |
| Rust | 64 | 19507 [19310-19687] | 39756 [39692-39764] | 4.62 ms | 58 + 131 s |
| Python | 8 | 3945 [3781-4166] | 9438 [9239-9475] | 9.64 ms | 131 + 105 s |
| Python | 16 | 3444 [3306-3454] | 6267 [6158-6945] | 10.05 ms | 145 + 121 s |
| Python | 32 | 3374 [3218-3445] | 6626 [6526-6638] | 10.69 ms | 146 + 120 s |

- Best against best, Rust stores 4.9 times as fast and loads 4.2 times as fast.
- **Python gets slower with more workers,** which fits contention on the interpreter lock, though that wasn't profiled. Rust gets faster up to 32 workers, then flattens.
- What stops Rust near 19.5 and 39.8 GB/s wasn't identified. A load phase at this speed lasts only about 2.4 s.

## On the drives

16 tasks in flight unless noted.

| Drives | Connector | Workers | Store MB/s | Load MB/s | Lookup p50 | CPU user + sys |
|---|---|---|---|---|---|---|
| 4 | Rust | 16 | 1811 [1804-1951] | 6018 [5529-6175] | 3.98 ms | 40 + 10 s |
| 4 | Rust | 32 | 1748 [1700-1768] | 6418 [5869-7177] | 4.41 ms | 41 + 11 s |
| 4 | Rust | 64 | 1663 [1619-1704] | 5839 [5643-6978] | 3.22 ms | 43 + 10 s |
| 4 | Rust | 128 | 1675 [1606-1681] | 5728 [5583-6485] | 3.89 ms | 42 + 11 s |
| 4 | Python | 8 | 1777 [1707-1850] | 2526 [2310-2646] | 8.66 ms | 133 + 15 s |
| 4 | Python | 64 | 1496 [1452-1593] | 2758 [2567-3288] | 11.48 ms | 151 + 27 s |
| 2 | Rust | 16 | 1698 [1638-1827] | 4037 [3989-4054] | 4.35 ms | 27 + 8 s |
| 2 | Rust | 32 | 1826 [1744-1964] | 4049 [4017-4081] | 3.09 ms | 29 + 9 s |
| 2 | Rust | 64 | 1842 [1779-1865] | 4047 [4034-4050] | 3.92 ms | 29 + 9 s |
| 2 | Python | 16 | 1545 [1487-1589] | 2952 [2692-3108] | 9.02 ms | 89 + 15 s |
| 2 | Python | 32 | 1492 [1380-1526] | 2774 [2456-3032] | 9.03 ms | 94 + 19 s |
| 1 | Rust | 8 | 1061 [1027-1393] | 2145 [2082-2165] | 2.98 ms | 19 + 7 s |
| 1 | Rust | 16 | 1022 [988-1111] | 2085 [2067-2092] | 3.07 ms | 19 + 8 s |
| 1 | Rust | 32 | 953 [950-996] | 2087 [2079-2089] | 3.99 ms | 18 + 7 s |
| 1 | Python | 8 | 1008 [955-1043] | 2234 [2164-2257] | 7.09 ms | 48 + 8 s |
| 1 | Python | 16 | 1113 [1047-1214] | 2157 [2123-2165] | 6.04 ms | 50 + 8 s |
| 1 | `raw_block` | 8 | 1132 [1070-1169] | 2101 [2091-2107] | 0.60 ms | 14 + 5 s |
| 1 | `raw_block` | 16 | 1039 [1019-1073] | 2110 [2090-2112] | 0.66 ms | 14 + 5 s |
| 1 | `raw_block` | 32 | 1011 [1000-1040] | 2082 [2082-2110] | 0.66 ms | 13 + 5 s |
| 1, 4 in flight | Rust | 8 | 1078 [1063-1146] | 2377 [2372-2378] | 2.01 ms | 18 + 4 s |
| 1, 4 in flight | Python | 8 | 1106 [958-1138] | 2292 [2290-2308] | 3.98 ms | 46 + 5 s |
| 1, 4 in flight | `raw_block` | 8 | 1050 [1011-1135] | 2425 [2425-2442] | 0.21 ms | 14 + 3 s |

- **On 4 drives, Rust loads about 2.3 times as fast as Python and reaches the VM's read limit.** Python stays near 2.5-3 GB/s on both 2 and 4 drives, so it stops scaling past 2 drives.
- **Rust uses about a third of Python's CPU** on 4 drives for the same data.
- **On one drive, all three connectors tie on throughput,** because the drive sets the limit. `raw_block` handles one device only, so it has no 2- or 4-drive rows.
- **Stores are held back by the VM's write limit,** so they say little about the connectors. Rust leads on 2 drives by about 19%, and the 4-drive results overlap.
- **Lookups:** Rust is 2 to 3 times as fast as Python, but `raw_block`, which doesn't ask a daemon, is 5 to 10 times as fast as Rust.

## With vLLM

Qwen2.5-7B-Instruct-AWQ, revision b250375, 4 drives, a prompt that fills 37 chunks, LMCache L1 of 3 GB. Rust ran 32 workers and Python 8. Time to first token, 3 runs.

| | Rust connector | Python adapter |
|---|---|---|
| Cold (compute) | 0.565-0.572 s | 0.563-0.571 s |
| Repeat straight after, L1 | 0.119-0.127 s | 0.297-0.317 s |
| After flushing L1, from the drives | 0.182-0.207 s | 0.301-0.331 s |
| Reading 37 chunks from the drives, all 6 runs | 89-113 ms | 195-249 ms |

Why the repeat in L1 differs by connector wasn't investigated. This GPU computes the prompt quickly, so the cache saves far less time than on the earlier server, where a cold run took 3.5 s.

## Open issue: output sometimes differs after a cache hit

The same prompt at temperature 0 didn't always return the same text. Runs where any answer differed:

| Setup | Runs that differed |
|---|---|
| No cache, 6 requests | 0 of 2 |
| vLLM's own prefix cache only, 6 requests | 0 of 2 |
| LMCache L1 only, no HyperCache, 6 requests | 1 of 2 |
| LMCache with Rust connector, first test | 2 of 3 |
| LMCache with Python adapter, first test | 2 of 3 |
| LMCache with Rust connector, second test, 5 requests | 1 of 3 |
| LMCache with Python adapter, second test, 5 requests | 0 of 3 |

- **It happens with no HyperCache in the path,** and it isn't specific to either connector. No checksum errors appeared in the second test's logs.
- In the two runs of the second test and the L1-only test that differed, the first request gave the usual answer and every cached request gave the same other answer. Both answers read normally: "...and storage location on one of 3..." against "...and stored on one of 37...".
- **Cause not found.** Rounding differences between a full computation and a cached one could explain it, but that isn't shown, and nor is a bug in LMCache or vLLM ruled out. The earlier server, on vLLM 0.31.0 and LMCache 0.5.5, gave identical output in the runs recorded, which were few.

## Safety checks

`scripts/e2e/lmcache_l2_e2e.py` on 4 drives, on LMCache 0.5.6: Rust 34 of 34, Python 18 of 18. Both were run twice because the second run, meant for 0.5.5, also loaded 0.5.6 from this image. `cargo test` in `lmcache-connector/` passed 31 of 31. The 0.5.5 pass comes from the earlier server.

## Runs thrown away

The first full run of 87 was invalid: a script bug overwrote the in-flight setting, so most runs used 1 task in flight. It is not included. A shorter first pass, with one run per setup, is kept in `scout/`.
