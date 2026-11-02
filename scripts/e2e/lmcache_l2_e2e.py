#!/usr/bin/env python3
"""
End-to-end check of the LMCache L2 adapter: real LMCache, real daemon, real device.

Loads the adapter through LMCache's plugin loader, then checks round trips, restart behaviour,
misses and corruption handling. Needs a running daemon that can write to its device, and read
and write access to that device, because the corruption test damages one stored chunk.

    python3 scripts/e2e/lmcache_l2_e2e.py --socket /run/hc-l2/hc.sock
"""

import argparse
import hashlib
import json
import mmap
import os
import select
import stat
import sys
import time

import torch  # noqa: F401  # must precede lmcache_native
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.l2_adapters import create_l2_adapter
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface
from lmcache.v1.distributed.l2_adapters.plugin_l2_adapter import PluginL2AdapterConfig

from hypercache.client import HyperCacheClient
from hypercache.datapath import DirectBlockReader, DirectBlockWriter


class Obj:
    """A buffer with the attribute the adapter and LMCache's file adapter use."""

    def __init__(self, data):
        self.byte_array = memoryview(bytearray(data))


def make_adapter(socket_path, namespace):
    config = PluginL2AdapterConfig.from_dict(
        {
            "type": "plugin",
            "module_path": "hypercache.integrations.lmcache_l2",
            "class_name": "HyperCacheL2Adapter",
            "adapter_params": {"namespace": namespace, "socket_path": socket_path, "tier": "nvme"},
        }
    )
    adapter = create_l2_adapter(config)
    assert isinstance(adapter, L2AdapterInterface)
    return adapter


def key(n, salt=""):
    return ObjectKey(hashlib.sha256(str(n).encode()).digest(), "Qwen/test-model", 0, 0, salt)


def block(n, size):
    seed = hashlib.sha256(str(n).encode()).digest()
    return (seed * (size // len(seed) + 1))[:size]


def wait_store(adapter, task_id):
    deadline = time.time() + 60
    while time.time() < deadline:
        select.select([adapter.get_store_event_fd()], [], [], 0.1)
        done = adapter.pop_completed_store_tasks()
        if task_id in done:
            return int(done[task_id])
    raise TimeoutError("store")


def wait_bits(query, fd, task_id, count):
    deadline = time.time() + 60
    while time.time() < deadline:
        select.select([fd], [], [], 0.1)
        bitmap = query(task_id)
        if bitmap is not None:
            return [bitmap.test(i) for i in range(count)]
    raise TimeoutError("query")


def lookup(adapter, keys):
    task_id = adapter.submit_lookup_and_lock_task(keys, {})
    return wait_bits(adapter.query_lookup_and_lock_result, adapter.get_lookup_and_lock_event_fd(), task_id, len(keys))


def load(adapter, keys, objects):
    task_id = adapter.submit_load_task(keys, objects)
    return wait_bits(adapter.query_load_result, adapter.get_load_event_fd(), task_id, len(keys))


def client_device(socket_path, adapter, stored_key):
    client = HyperCacheClient(socket_path=socket_path)
    try:
        return client.locate(adapter._uri(stored_key), auto_ingest=False).chunk_layout[0].device
    finally:
        client.close()


def mixed_writes_are_coherent(client):
    """Writes the same extents through each path in turn. Every read must see the latest write."""
    uri = f"hypercache://kv_cache/e2e-mixed-{os.getpid()}/object.kv"
    size = 6 * 1024 * 1024 + 123  # two chunks, and a last chunk that isn't a whole number of blocks
    layout = client.stage_write(uri, size, tier="nvme", auto_flush_s3=False).chunk_layout
    buffered = DirectBlockWriter(max_workers=4)
    direct = DirectBlockWriter(max_workers=4, direct_on_block_devices=True)
    direct_reader = DirectBlockReader(max_workers=4, verify_checksums=False)
    plain = os.open(layout[0].device, os.O_RDONLY)
    aligned = mmap.mmap(-1, size)
    view = memoryview(aligned)

    def read_both():
        through_direct = bytearray(size)
        direct_reader.read_range_into(layout, 0, size, through_direct)
        through_cache, position = bytearray(size), 0
        for chunk in layout:
            through_cache[position : position + chunk.length] = os.pread(plain, chunk.length, chunk.offset)
            position += chunk.length
        return bytes(through_direct), bytes(through_cache)

    try:
        for round_number, writer in enumerate([buffered, direct, buffered, direct, direct, buffered]):
            data = block(1000 + round_number, size)
            view[:] = data
            writer.write_chunks(layout, view)
            if read_both() != (data, data):
                return False
        return True
    finally:
        view.release()
        os.close(plain)
        direct_reader.close()
        direct.close()
        buffered.close()
        client.evict(uri)


def check(name, condition):
    print(f"{'PASS' if condition else 'FAIL'}  {name}", flush=True)
    return bool(condition)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--size-mb", type=int, default=14, help="object size; 14 MiB is one 256-token Qwen2.5-7B chunk")
    args = parser.parse_args()

    size = args.size_mb * 1024 * 1024
    namespace = f"e2e-{os.getpid()}"
    results = []
    adapter = make_adapter(args.socket, namespace)
    keys = [key(i) for i in range(8)]

    stored = wait_store(adapter, adapter.submit_store_task(keys, [Obj(block(i, size)) for i in range(8)]))
    results.append(check("store reports every byte", stored == 8 * size))
    if stat.S_ISBLK(os.stat(client_device(args.socket, adapter, keys[0])).st_mode):
        results.append(check("raw device writes use O_DIRECT", any(fd is not None for fd in adapter._writer._direct_fds.values())))
    results.append(check("repeat store writes nothing", wait_store(adapter, adapter.submit_store_task(keys, [Obj(block(i, size)) for i in range(8)])) == 0))

    results.append(check("lookup finds stored keys only", lookup(adapter, keys + [key(99)]) == [True] * 8 + [False]))
    targets = [Obj(bytes(size)) for _ in keys]
    results.append(check("load succeeds", load(adapter, keys, targets) == [True] * 8))
    results.append(check("bytes match", all(bytes(t.byte_array) == block(i, size) for i, t in enumerate(targets))))
    adapter.submit_unlock(keys)

    results.append(check("other salt misses", lookup(adapter, [key(0, salt="tenant-b")]) == [False]))
    other = make_adapter(args.socket, namespace + "-other")
    results.append(check("other namespace misses", lookup(other, keys[:1]) == [False]))
    other.close()

    client = HyperCacheClient(socket_path=args.socket)
    location = client.locate(adapter._uri(keys[0]), auto_ingest=False)
    results.append(check("block is LOCAL_ONLY", location.sync_status == "LOCAL_ONLY"))
    results.append(check("every chunk has a checksum", all(c.checksum is not None for c in location.chunk_layout)))

    # Damage one byte on the device, as a bad sector or a stale layout would
    chunk = location.chunk_layout[-1]
    fd = os.open(chunk.device, os.O_RDWR)
    try:
        position = chunk.offset + chunk.length // 2
        original = os.pread(fd, 1, position)
        os.pwrite(fd, bytes([original[0] ^ 0xFF]), position)
        os.fsync(fd)
    finally:
        os.close(fd)
    damaged = [Obj(bytes(size))]
    results.append(check("corrupt block is a miss", load(adapter, keys[:1], damaged) == [False]))
    results.append(check("corrupt block is removed", lookup(adapter, keys[:1]) == [False]))
    results.append(check("other blocks still load", load(adapter, keys[1:2], [Obj(bytes(size))]) == [True]))
    adapter.submit_unlock(keys[1:2])
    results.append(check("status counts the checksum failure", adapter.report_status()["checksum_failures"] == 1))

    wrong = load(adapter, keys[1:2], [Obj(bytes(size - 4096))])
    results.append(check("wrong buffer size is a miss", wrong == [False]))

    results.append(check("page cache and O_DIRECT stay coherent on one device", mixed_writes_are_coherent(client)))

    adapter.delete(keys)
    adapter.submit_unlock(keys)
    adapter.delete(keys)
    results.append(check("delete removes blocks", lookup(adapter, keys) == [False] * 8))
    results.append(check("status is healthy", adapter.report_status()["is_healthy"] is True))
    adapter.close()
    client.close()

    passed = sum(results)
    print(json.dumps({"passed": passed, "total": len(results)}))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
