#!/usr/bin/env python3
"""
Measures the Rust LMCache connector with LMCache out of the path, to find what limits it.

Calls the connector the way LMCache's wrapper does, with page-aligned buffers. Needs a running
daemon, and the hypercache_lmcache_native package.

    python3 benchmarks/lmcache_connector_direct.py --socket /run/hc/hc.sock --tier ram
"""

import argparse
import json
import mmap
import os
import select
import time

from hypercache_lmcache_native import HyperCacheNativeConnector

OBJECT_SIZE = 14 * 1024 * 1024
OBJECTS = 64
BATCH = 8
MIB = 1024 * 1024


def make_buffers():
    seed = os.urandom(MIB)
    buffers = []
    for _ in range(OBJECTS):
        view = memoryview(mmap.mmap(-1, OBJECT_SIZE))
        for offset in range(0, OBJECT_SIZE, MIB):
            view[offset : offset + MIB] = seed
        buffers.append(view)
    return buffers


def wait(connector, future_ids):
    pending = set(future_ids)
    all_ok = True
    while pending:
        select.select([connector.event_fd()], [], [], 1)
        for future_id, ok, _error, _results in connector.drain_completions():
            all_ok = all_ok and ok
            pending.discard(future_id)
    return all_ok


def timed(connector, submit, keys, buffers, in_flight_keys):
    start = time.perf_counter()
    future_ids = []
    all_ok = True
    for i in range(0, OBJECTS, BATCH):
        if len(future_ids) * BATCH >= in_flight_keys:
            all_ok = wait(connector, future_ids) and all_ok
            future_ids = []
        future_ids.append(submit(keys[i : i + BATCH], buffers[i : i + BATCH]))
    all_ok = wait(connector, future_ids) and all_ok
    return OBJECTS * OBJECT_SIZE / 1e6 / (time.perf_counter() - start), all_ok


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--tier", default="nvme", choices=["nvme", "ram"])
    args = parser.parse_args()

    buffers = make_buffers()
    keys = [f"m@00000000@0@{i:04x}" for i in range(OBJECTS)]
    for workers, in_flight_keys in [(4, 64), (8, 64), (16, 64), (32, 64), (16, 8)]:
        for repeat in range(2):
            namespace = f"direct-{os.getpid()}-{workers}-{in_flight_keys}-{repeat}"
            connector = HyperCacheNativeConnector(namespace, args.socket, args.tier, workers)
            store, store_ok = timed(connector, connector.submit_batch_set, keys, buffers, in_flight_keys)
            load, load_ok = timed(connector, connector.submit_batch_get, keys, buffers, in_flight_keys)
            print(
                json.dumps(
                    {
                        "workers": workers,
                        "in_flight_keys": in_flight_keys,
                        "store_mbps": round(store),
                        "load_mbps": round(load),
                        "ok": store_ok and load_ok,
                    }
                ),
                flush=True,
            )
            wait(connector, [connector.submit_batch_delete(keys)])
            connector.close()


if __name__ == "__main__":
    main()
