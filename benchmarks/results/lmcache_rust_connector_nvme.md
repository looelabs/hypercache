# Rust LMCache connector results: bare metal, 4 NVMe drives

Measured 2026-11-04 on a rented bare-metal node with no GPU. Raw data is in `lmcache_l2_raw/rust-connector-epyc/`. This page covers storage only: no model ran, so it says nothing about first-token time or output.

**Hardware:** AMD EPYC 9474F (48 cores, 1 socket, 1 NUMA node), 755 GB DDR5-4800 on 8 channels, 4 Micron 7450 3.84 TB drives (firmware E2MU300, Gen4 x4, 4 KiB blocks), each discarded before a run.

**Software:** Linux 6.8.0, LMCache 0.5.6, Python 3.12, rustc 1.99.1, HyperCache 0.9.3 plus this branch. The safety script passed 30 of 30 on LMCache 0.5.5 and 0.5.6 on these drives.

**Method:** `lmcache bench l2`, 8 keys per task, L1 aligned to 4 KiB, 2 warmup rounds, order shuffled, median of 3 runs with the range in brackets. Every run checks that all keys were stored and loaded. 14 MiB is one 256-token Qwen2.5-7B chunk. "Old" is the daemon before the stripe fix (each object starts on the first drive), and "fixed" is after it.

## Summary

- **Striping scales linearly with the fix.** 14 MiB objects load at 90% of what the raw drives can do.
- **The stripe fix matters most for small objects.** 2 MiB objects load 3.9 times as fast, because they used to land on one drive.
- **On 4 drives, the connector matches the best `raw_block` result on RAID 0, and holds it.** `raw_block` on RAID 0 reached the same speed in 2 of 6 runs and got stuck near 15.2 GB/s in the other 4. The cause isn't known, and could be our array setup.
- **No load returned wrong bytes** across crashes, restarts, corruption, eviction and a 30-minute soak.
- **Lookups are slower than `raw_block`'s**, at about 2.6 ms against 0.9-1.0 ms on one drive.
- **Full-speed writes cut reads hard.** Most of that comes from the drives themselves. See [mixed reads and writes](#mixed-reads-and-writes).

## What the drives can do without HyperCache

fio, 4 MiB blocks, queue depth 32 per drive, 3 runs. Results were tight and linear.

| Drives | Read MB/s | Write MB/s |
|---|---|---|
| 1 | 6911 | 5398 |
| 2 | 13822 | 10833 |
| 4 | 27643 | 21576 |

## Stripe fix, old vs fixed

4 drives unless noted, 64 objects in flight. Old and fixed ran in the same session.

| Objects, workers | Old load | Fixed load | Old store | Fixed store |
|---|---|---|---|---|
| 2 MiB, 32 | 6321 [6231-6350] | 24609 [24604-24695] | 4962 [4920-4989] | 15390 [15278-15441] |
| 14 MiB, 32 | 20740 [20707-20887] | 24884 [24619-24888] | 16129 [15977-16261] | 18549 [18248-18842] |
| 14 MiB, 128 | 18713 [18622-18781] | 23050 [23016-23173] | 14962 [14519-14983] | 18686 [18202-18893] |
| 16 MiB, 32 | 23587 [23536-23790] | 24646 [24641-24768] | 18144 [18119-18349] | 18759 [18717-18802] |
| 14 MiB, 32, 2 drives | 10526 [10390-10541] | 12071 [12014-12217] | 8745 [8687-8764] | 9608 [9497-9717] |

Per-drive peak read during a 14 MiB run, from iostat:

| Drive | Old MB/s | Fixed MB/s |
|---|---|---|
| 1 | 5077 | 5183 |
| 2 | 5072 | 5173 |
| 3 | 5065 | 5172 |
| 4 | 2532 | 5175 |

## Worker count

Fixed daemon, 4 drives, 14 MiB, 64 in flight.

| Workers | Load MB/s | Store MB/s |
|---|---|---|
| 8 | 16097 [16003-16171] | 17744 [17706-17795] |
| 16 | 22513 [22504-22522] | 18671 [18591-18721] |
| 32 | 24959 [24918-24983] | 18811 [18734-18981] |
| 64 | 23357 [23186-23485] | 18586 [18286-18634] |
| 128 | 23050 [23016-23173] | 18686 [18202-18893] |

The 8, 16 and 32 rows were one batch, and the 64 and 128 rows were another, whose 32-worker result was 24884. On 4 drives, 32 is the best count tested, so it stays the default. On one drive, 8 is better (below), which is about 8 workers per drive. Other drive counts weren't swept.

## One drive against raw_block

14 MiB, 16 in flight. `raw_block` here is LMCache's own adapter, which uses one device.

| Adapter, workers | Store MB/s | Load MB/s | Lookup ms |
|---|---|---|---|
| Rust, 8 | 4954 [4918-4996] | 6458 [6454-6475] | 2.64 |
| Rust, 32 | 4957 [4941-5005] | 5917 [5889-5920] | 2.62 |
| `raw_block` posix, 8 | 4984 [4928-5006] | 6163 [6147-6169] | 0.90 |
| `raw_block` posix, 32 | 4913 [4906-4984] | 5878 [5837-5887] | 1.00 |
| `raw_block` io_uring, 8 | 4968 [4928-4980] | 5976 [5831-5980] | 0.95 |
| `raw_block` io_uring, 32 | 4942 [4915-4985] | 5830 [5714-5915] | 0.97 |

With 4 in flight, 8 workers: Rust stores at 4917 and loads at 6328 MB/s with 0.72 ms lookups, and `raw_block` io_uring stores at 4880 and loads at 5927 MB/s with 0.27 ms lookups. LMCache logged that io_uring fixed buffers weren't in use in this setup, because no stable L1 range was eligible.

These one-drive runs used the old daemon. With one drive, every object starts on that drive either way.

## Multiple drives against raw_block on RAID 0

14 MiB, 64 in flight. `raw_block` can only use one device, so to give it all the drives we built Linux md RAID 0 with a 4 MiB chunk, and used io_uring.

| Setup | Load MB/s | Store MB/s | Lookup ms |
|---|---|---|---|
| Rust, fixed, 4 drives, 32 workers | 24884 [24619-24888] | 18549 [18248-18842] | 10.5 |
| `raw_block` RAID 0, 4 drives, 32 workers | 23042 [15192-24721] | 16855 [16645-18534] | 3.9 |
| `raw_block` RAID 0, 4 drives, 64 workers | 15236 [15163-15324] | 16812 [16479-16940] | 4.0 |
| Rust, fixed, 2 drives, 32 workers | 12071 [12014-12217] | 9608 [9497-9717] | 10.3 |
| `raw_block` RAID 0, 2 drives, 64 workers | 11548 [11439-11606] | 9810 [9805-9831] | 3.9 |
| `raw_block` RAID 0, 2 drives, 32 workers | 10727 [10677-10794] | 9669 [9649-9701] | 3.9 |

Don't read the stuck 15.2 GB/s runs as a general fault of RAID 0 or `raw_block`. It could come from either of those or from our array settings, and we didn't look for the cause. The comparison shows what we measured, without tuning `raw_block`'s setup.

## Mixed reads and writes

No LMCache in the path. 4 drives, 14 MiB, one connector per direction with 32 workers and 64 in flight, 30 s each, 3 rounds, new keys for writes. Reads are in random order.

| Writes limited to | Read MB/s | Write MB/s | Read batch p50 / p99 ms |
|---|---|---|---|
| Reads only | 25420 | none | 36.9 / 43.5 |
| 1 GB/s | 24181 | 1000 | 37.8 / 50.2 |
| 4 GB/s | 19099 | 4001 | 49.5 / 59.9 |
| No limit | 6478 | 12163 | 144.6 / 187.4 |
| Writes only | none | 20048 | not applicable |

The same mix on the drives with fio and no HyperCache, as a rough guide. Its queue depths differ (32 for reads, 16 for writes, per drive).

| Writes limited to | Read MB/s | Write MB/s |
|---|---|---|
| Reads only | 27650 | none |
| 4.2 GB/s | 20637 | 4194 |
| No limit | 10715 | 10663 |

- **The drives account for most of the drop.** Reads fall by about 25% with 4 GB/s of writes and about 60% with unlimited writes, with no HyperCache involved. Adding an fsync every 4 writes made no difference in fio.
- **At full write speed, the connector leans further towards writes than fio did:** reads at 6478 against 10715, and writes at 12163 against 10663. The cause isn't known, and different queue depths could explain part of it.

## Failure tests

Rule: a load is a miss or the right bytes. Each object carries its ID, a generation and its position every 4 KiB, so stale, misplaced or half-replaced data is caught. Results are in `failure/` and `failure_final/`.

| Test | Result |
|---|---|
| `kill -9` during loads, with and without saved metadata | No hangs or wrong bytes, and it recovers after restart |
| `kill -9` during stores, with and without saved metadata | Same. With saved metadata, 1402 of 1436 acknowledged objects came back and the rest were misses |
| Restart with saved metadata, clean or `kill -9`, straight after storing | All 512 objects came back |
| Blocks freed and reused by other objects, then a crash, grace period 0 or 1 s | Deleted objects miss, live objects are right |
| One 4 KiB block corrupted on disk | Exactly one object missed |
| 1 GiB corrupted on disk | 294 objects missed, 294 checksum failures, none wrong |
| Storing again after corruption | All 512 objects read back right |
| 20 GiB capacity limit under load, 80 GiB stored twice | No wrong bytes. 38 of about 11,500 stores failed at commit, because the object was evicted first. They were reported as failures |
| Full 20 GiB device under load, 80 GiB stored twice | No wrong bytes, no failed stores |

`failure_final/` is the final connector, which reconnects after a daemon restart. The corruption and eviction rows are from `failure/`, before that change, which doesn't touch those paths. In `failure/`, each worker's first store after a restart fails with a broken pipe, which is what the change fixed.

## Upgrading

- **Connector branch, old daemon to fixed and back:** all 512 objects from each version read back right after the upgrade and the rollback, with a clean stop and with `kill -9`.
- **Main's daemon to main plus the fix and back, with the Python client:** 300 objects of 2, 14 and 16 MiB, compared byte for byte, all right at every step, both ways of stopping. Results are in `main_upgrade/`.

Objects keep the placement they were stored with, so existing objects aren't rebalanced.

## Soak

30 minutes, 8192 objects of 14 MiB, 4 drives. Reads ran as fast as they could while 2 GB/s of writes deleted and stored the same objects. Every 8th batch of reads was checked.

| Measure | Result |
|---|---|
| Loads | 2,585,477 |
| Loads checked byte for byte | 323,029, none wrong |
| Misses | 427 (0.017%), which is expected when an object is being replaced |
| Read speed per minute | 20874-21497 MB/s, median 21352 |
| Write speed per minute | 2001 MB/s throughout |
| Read batch latency | p50 42 ms, worst minute's p99 58.5 ms |
| Daemon memory | 24 MB at the start, 40 MB at most |
| Hottest drive reading | 33 C |
| Stores failed, checksum failures | 0, 0 |
| Kernel NVMe errors, media errors | 0, 0 |

## Not covered

- **Real vLLM:** first-token time, output comparison and concurrent traffic need a GPU.
- **RAM tier:** set aside for now. A short sweep loaded at about 55 GB/s, and the limit wasn't investigated.
- **Other hardware:** results at this drive speed may not carry over to consumer drives.
