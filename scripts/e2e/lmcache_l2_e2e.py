#!/usr/bin/env python3
"""
End-to-end check of the LMCache L2 connector in lmcache-connector/: real LMCache, real daemon,
real device.

Loads the connector through LMCache's plugin loader, then checks round trips, restart behaviour,
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
from lmcache.v1.distributed.l2_adapters.native_connector_l2_adapter import _object_key_to_string
from lmcache.v1.distributed.l2_adapters.native_plugin_l2_adapter import NativePluginL2AdapterConfig

from hypercache.client import HyperCacheClient
from hypercache.datapath import DirectBlockReader, DirectBlockWriter


class Obj:
    """A buffer with the attribute LMCache's wrapper and file adapter use."""

    def __init__(self, data):
        self.byte_array = memoryview(bytearray(data))

    def get_size(self):
        return len(self.byte_array)


class AlignedObj:
    """A page-aligned buffer, like LMCache's L1 allocator gives with 4 KiB alignment."""

    def __init__(self, data):
        self._map = mmap.mmap(-1, max(len(data), 1))
        self.byte_array = memoryview(self._map)[: len(data)]
        self.byte_array[:] = data

    def get_size(self):
        return len(self.byte_array)


def make_adapter(socket_path, namespace):
    config = NativePluginL2AdapterConfig.from_dict(
        {
            "type": "native_plugin",
            "module_path": "hypercache_lmcache_native",
            "class_name": "HyperCacheNativeConnector",
            "adapter_params": {"namespace": namespace, "socket_path": socket_path, "tier": "nvme"},
            "max_capacity_gb": 0,
        }
    )
    adapter = create_l2_adapter(config)
    assert isinstance(adapter, L2AdapterInterface)
    return adapter


def uri_of(adapter, stored_key):
    return adapter._client.uri(_object_key_to_string(stored_key))


def checksum_failures(adapter):
    return adapter._client.stats()["checksum_failures"]


def used_o_direct(adapter):
    stats = adapter._client.stats()
    return stats["direct_write_bytes"] + stats["bounced_write_bytes"] > 0


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
        return client.locate(uri_of(adapter, stored_key), auto_ingest=False).chunk_layout[0].device
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


def io_checks(socket_path, namespace, size, check):
    """The connector's own I/O paths, and the URI format that existing blocks depend on."""
    results = []
    rust = make_adapter(socket_path, namespace)
    stats = rust._client.stats

    # Page-aligned buffers go to and from the drive with no copy
    aligned_keys = [key(i) for i in range(200, 204)]
    before = stats()
    sources = [AlignedObj(block(i, size)) for i in range(200, 204)]
    wait_store(rust, rust.submit_store_task(aligned_keys, sources))
    targets = [AlignedObj(bytes(size)) for _ in aligned_keys]
    loaded = load(rust, aligned_keys, targets)
    results.append(check("aligned buffers round trip", loaded == [True] * 4 and all(bytes(t.byte_array) == block(200 + i, size) for i, t in enumerate(targets))))
    after = stats()
    is_block = stat.S_ISBLK(os.stat(client_device(socket_path, rust, aligned_keys[0])).st_mode)
    if is_block:
        results.append(check("aligned writes use O_DIRECT with no copy", after["direct_write_bytes"] - before["direct_write_bytes"] == 4 * size and after["bounced_write_bytes"] == before["bounced_write_bytes"]))
    results.append(check("aligned reads use O_DIRECT with no copy", after["direct_read_bytes"] - before["direct_read_bytes"] == 4 * size and after["bounced_read_bytes"] == before["bounced_read_bytes"]))
    results.append(check("repeat store writes nothing to the drive", True if wait_store(rust, rust.submit_store_task(aligned_keys, sources)) == 0 and stats()["direct_write_bytes"] == after["direct_write_bytes"] else False))

    # Sizes that aren't whole blocks, in aligned and unaligned buffers
    for name, factory, offset in (("aligned", AlignedObj, 0), ("unaligned", Obj, 10)):
        odd_key = key(210 + offset)
        odd_size = size + 123 + offset
        wait_store(rust, rust.submit_store_task([odd_key], [factory(block(300 + offset, odd_size))]))
        target = factory(bytes(odd_size))
        good = load(rust, [odd_key], [target]) == [True] and bytes(target.byte_array) == block(300 + offset, odd_size)
        results.append(check(f"partial last block round trips, {name} buffer", good))

    # What the Rust path wrote must read the same through the page cache
    client = HyperCacheClient(socket_path=socket_path)
    location = client.locate(uri_of(rust, aligned_keys[0]), auto_ingest=False)
    fd = os.open(location.chunk_layout[0].device, os.O_RDONLY)
    first = location.chunk_layout[0]
    results.append(check("direct writes read back through the page cache", os.pread(fd, first.length, first.offset) == block(200, size)[: first.length]))
    os.close(fd)

    # Blocks are found by URI, so changing this format would orphan every stored block
    hash_hex = aligned_keys[0].chunk_hash.hex()
    prefix = f"hypercache://kv_cache/{namespace}/Qwen%2Ftest-model/0/0"
    results.append(check("URI format is unchanged", uri_of(rust, aligned_keys[0]) == f"{prefix}/_/{hash_hex}.kv"))
    salted = key(220, salt="tenant a+b")
    results.append(check("salted URI format is unchanged", uri_of(rust, salted) == f"{prefix}/s-tenant%20a%2Bb/{salted.chunk_hash.hex()}.kv"))

    # LMCache's wrapper doesn't check its locks before a delete. Losing a locked block must be a
    # miss, never wrong data.
    victim = key(240)
    wait_store(rust, rust.submit_store_task([victim], [Obj(block(240, size))]))
    lookup(rust, [victim])
    rust.delete([victim])
    target = AlignedObj(b"\x5a" * size)
    missed = load(rust, [victim], [target]) == [False]
    results.append(check("block deleted while locked is a miss", missed and bytes(target.byte_array) == b"\x5a" * size))
    rust.submit_unlock([victim])

    # Many tasks at once, with every byte checked
    many = [key(i) for i in range(400, 440)]
    tasks = [rust.submit_store_task(many[i : i + 4], [Obj(block(400 + i + j, size // 4)) for j in range(4)]) for i in range(0, 40, 4)]
    remaining = set(tasks)
    deadline = time.time() + 120
    while remaining and time.time() < deadline:
        select.select([rust.get_store_event_fd()], [], [], 0.1)
        remaining -= set(rust.pop_completed_store_tasks())
    targets = [AlignedObj(bytes(size // 4)) for _ in many]
    tasks = [(i, rust.submit_load_task(many[i : i + 4], targets[i : i + 4])) for i in range(0, 40, 4)]
    all_loaded = all(wait_bits(rust.query_load_result, rust.get_load_event_fd(), task, 4) == [True] * 4 for _, task in tasks)
    results.append(check("concurrent tasks return the right bytes", not remaining and all_loaded and all(bytes(t.byte_array) == block(400 + i, size // 4) for i, t in enumerate(targets))))

    rust.close()
    results.append(check("close releases cleanly and refuses new work", refuses_after_close(rust)))
    client.close()
    return results


def refuses_after_close(adapter):
    try:
        adapter._client.submit_batch_exists(["m@0@0@ff"])
    except RuntimeError:
        return True
    return False


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
        results.append(check("raw device writes use O_DIRECT", used_o_direct(adapter)))
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
    location = client.locate(uri_of(adapter, keys[0]), auto_ingest=False)
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
    results.append(check("status counts the checksum failure", checksum_failures(adapter) == 1))

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

    results.extend(io_checks(args.socket, namespace + "-io", size, check))

    passed = sum(results)
    print(json.dumps({"passed": passed, "total": len(results)}))
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
