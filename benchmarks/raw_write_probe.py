#!/usr/bin/env python3
"""
Finds which part of the raw write path lowers later read speed. No LMCache involved.

Writes objects through the real daemon protocol using one write strategy, then reads them all
back with the real DirectBlockReader, CRC checks on. A failed CRC stops the run, so a strategy
that writes wrong data can't report a good number. The strategies live here, not in the product.

  pool     each chunk goes to a shared thread pool, so chunks can land out of order (today's path)
  inorder  each object writes its chunks one after another
  merged   each object issues one write across its adjacent chunks

Each runs through the page cache (buffered) or with O_DIRECT. Needs a running daemon on a raw device.

    python3 benchmarks/raw_write_probe.py --socket /run/hc/hc.sock --mode inorder --direct --parallel 8
"""

import argparse
import json
import mmap
import os
import statistics
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

from hypercache.client import HyperCacheClient
from hypercache.datapath import DirectBlockReader, DirectBlockWriter

OBJECT_SIZE = 14 * 1024 * 1024
BLOCK = 4096


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--mode", required=True, choices=["pool", "inorder", "merged"])
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--objects", type=int, default=512)
    args = parser.parse_args()

    source = mmap.mmap(-1, OBJECT_SIZE)  # page-aligned, as O_DIRECT requires
    source.write(os.urandom(OBJECT_SIZE))
    view = memoryview(source)

    clients = threading.local()
    fds = {}
    fd_lock = threading.Lock()
    product_writer = DirectBlockWriter(max_workers=8)
    chunk_pool = ThreadPoolExecutor(8)
    flags = os.O_RDWR | (os.O_DIRECT if args.direct else 0)

    def client():
        if not hasattr(clients, "client"):
            clients.client = HyperCacheClient(socket_path=args.socket, timeout=120)
        return clients.client

    def device_fd(path):
        with fd_lock:
            if path not in fds:
                fds[path] = os.open(path, flags)
            return fds[path]

    def pwrite_all(fd, data, offset):
        while len(data) > 0:
            written = os.pwrite(fd, data, offset)
            data = data[written:]
            offset += written

    def write_object(layout):
        if args.mode == "pool" and not args.direct:
            product_writer.write_chunks(layout, view)  # exactly today's path
            return
        spans, position = [], 0
        for chunk in layout:
            spans.append((chunk.device, chunk.offset, position, chunk.length))
            position += chunk.length
        if args.mode == "merged":
            merged = [spans[0]]
            for device, offset, start, length in spans[1:]:
                d, o, s, n = merged[-1]
                if d == device and o + n == offset:
                    merged[-1] = (d, o, s, n + length)
                else:
                    merged.append((device, offset, start, length))
            spans = merged
        work = [(device_fd(d), view[s : s + n], o) for d, o, s, n in spans]
        if args.mode == "pool":
            for future in [chunk_pool.submit(pwrite_all, *item) for item in work]:
                future.result()
        else:
            for item in work:
                pwrite_all(*item)

    layouts = [None] * args.objects
    commit_seconds = [0.0] * args.objects

    def put(index):
        c = client()
        uri = f"hypercache://kv_cache/probe/{index}.kv"
        staged = c.stage_write(uri, OBJECT_SIZE, tier="nvme", auto_flush_s3=False)
        if not staged.is_staged:
            raise IOError(staged.error_message)
        write_object(staged.chunk_layout)
        crcs, position = [], 0
        for chunk in staged.chunk_layout:
            crcs.append(zlib.crc32(view[position : position + chunk.length]) & 0xFFFFFFFF)
            position += chunk.length
        start = time.perf_counter()
        if not c.commit_write(uri, local_only=True, chunk_checksums=crcs):
            raise IOError("commit failed")
        commit_seconds[index] = time.perf_counter() - start
        layouts[index] = c.locate(uri, auto_ingest=False).chunk_layout

    total_mb = args.objects * OBJECT_SIZE / 1e6
    start = time.perf_counter()
    with ThreadPoolExecutor(args.parallel) as pool:
        list(pool.map(put, range(args.objects)))
    write_mbps = total_mb / (time.perf_counter() - start)

    reader = DirectBlockReader(max_workers=8, verify_checksums=True)
    buffers = [memoryview(mmap.mmap(-1, OBJECT_SIZE)) for _ in range(8)]
    slots = iter(buffers)
    slot_lock = threading.Lock()
    local = threading.local()

    def get(layout):
        if not hasattr(local, "buffer"):
            with slot_lock:
                local.buffer = next(slots)
        if reader.read_range_into(layout, 0, OBJECT_SIZE, local.buffer) != OBJECT_SIZE:
            raise IOError("short read")

    read_runs = []
    for _ in range(3):
        start = time.perf_counter()
        with ThreadPoolExecutor(8) as pool:
            list(pool.map(get, layouts))
        read_runs.append(total_mb / (time.perf_counter() - start))
        local = threading.local()
        slots = iter(buffers)
    stats = reader.get_stats()
    reader.close()

    print(
        json.dumps(
            {
                "mode": args.mode,
                "direct": args.direct,
                "parallel": args.parallel,
                "write_mbps": round(write_mbps),
                "read_mbps": round(statistics.median(read_runs)),
                "commit_ms_median": round(1000 * statistics.median(commit_seconds), 1),
                "commit_ms_p99": round(1000 * sorted(commit_seconds)[int(0.99 * args.objects)], 1),
                "reader_fallbacks": stats["direct_fallbacks"] + stats["buffered_reads"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
