"""
Unit tests for the LMCache L2 adapter, using a fake daemon on a real socket and a real data file.

LMCache only ships Linux wheels, so its modules are stubbed where it isn't installed.
scripts/e2e/lmcache_l2_e2e.py covers the real LMCache and the real daemon.
"""

import importlib.util
import json
import os
import select
import shutil
import socket
import sys
import tempfile
import threading
import time
import types
import unittest
from dataclasses import dataclass


def _install_lmcache_stubs():
    class Bitmap:
        def __init__(self, size):
            self.bits = [False] * size

        def set(self, index):
            self.bits[index] = True

        def test(self, index):
            return self.bits[index]

    class L2StoreResult(int):
        def __new__(cls, success, bytes_transferred=0):
            return super().__new__(cls, bytes_transferred if success else -1)

    class L2AdapterInterface:
        def __init__(self, max_capacity_bytes=0):
            self.events = {"stored": [], "accessed": [], "deleted": []}

        def _notify_keys_stored(self, keys, sizes):
            self.events["stored"].extend(keys)

        def _notify_keys_accessed(self, keys):
            self.events["accessed"].extend(keys)

        def _notify_keys_deleted(self, keys, sizes):
            self.events["deleted"].extend(keys)

    class Notifier:
        def __init__(self):
            self.read_fd, self.write_fd = os.pipe()

        def fileno(self):
            return self.read_fd

        def notify(self):
            os.write(self.write_fd, b"x")

        def close(self):
            os.close(self.read_fd)
            os.close(self.write_fd)

    modules = {
        "lmcache": {},
        "lmcache.lmcache_native": {"Bitmap": Bitmap},
        "lmcache.v1": {},
        "lmcache.v1.platform": {"create_event_notifier": Notifier},
        "lmcache.v1.distributed": {},
        "lmcache.v1.distributed.internal_api": {"L2StoreResult": L2StoreResult},
        "lmcache.v1.distributed.l2_adapters": {},
        "lmcache.v1.distributed.l2_adapters.base": {"L2AdapterInterface": L2AdapterInterface, "L2TaskId": int},
        "lmcache.v1.distributed.l2_adapters.config": {"L2AdapterConfigBase": object},
    }
    for name, attributes in modules.items():
        module = types.ModuleType(name)
        module.__dict__.update(attributes)
        sys.modules[name] = module


try:
    import lmcache.v1.distributed.l2_adapters.base  # noqa: F401

    HAVE_LMCACHE = True
except ImportError:
    HAVE_LMCACHE = False
    _install_lmcache_stubs()

# The adapter imports torch only to set LMCache's import order. Where torch is absent, stub it for
# this one import so that other test modules still see that it's missing.
_torch_missing = "torch" not in sys.modules and importlib.util.find_spec("torch") is None
if _torch_missing:
    sys.modules["torch"] = types.ModuleType("torch")
from hypercache.integrations.lmcache_l2 import HyperCacheL2Adapter, HyperCacheL2AdapterConfig  # noqa: E402

if _torch_missing:
    del sys.modules["torch"]

CHUNK_SIZE = 4 * 1024 * 1024


@dataclass(frozen=True)
class Key:
    chunk_hash: bytes
    model_name: str = "org/model"
    kv_rank: int = 0
    object_group_id: int = 0
    cache_salt: str = ""


class Obj:
    def __init__(self, data):
        self.byte_array = memoryview(bytearray(data))


def payload(size, seed=1):
    return bytes((i * seed) % 251 for i in range(4096)) * (size // 4096) + bytes(size % 4096)


class FakeDaemon:
    """Speaks enough of the daemon protocol, with the same chunking and commit rules."""

    def __init__(self, directory, reports_sync_status=True):
        self.arena = os.path.join(directory, "arena.raw")
        with open(self.arena, "wb") as f:
            f.truncate(256 * 1024 * 1024)
        self.socket_path = os.path.join(directory, "d.sock")
        self.reports_sync_status = reports_sync_status
        self.objects = {}
        self.requests = []
        self.next_offset = 0
        self.generation = 0
        self.lock = threading.Lock()
        self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.server.bind(self.socket_path)
        self.server.listen(32)
        threading.Thread(target=self._accept, daemon=True).start()

    def stop(self):
        self.server.close()
        os.unlink(self.socket_path)

    def _accept(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        with conn, conn.makefile("rb") as lines:
            for line in lines:
                request = json.loads(line)
                with self.lock:
                    self.requests.append(request)
                    response = self._handle(request)
                conn.sendall((json.dumps(response) + "\n").encode())

    def _handle(self, r):
        action, uri = r.get("action"), r.get("uri")
        if action == "stage_write":
            layout, left = [], r["size_bytes"]
            while left > 0:
                length = min(left, CHUNK_SIZE)
                layout.append({"device": self.arena, "offset": self.next_offset, "length": length, "tier": "nvme"})
                self.next_offset += CHUNK_SIZE
                left -= length
            self.generation += 1
            self.objects[uri] = {
                "size": r["size_bytes"],
                "layout": layout,
                "status": "STAGED",
                "generation": self.generation,
            }
            return {
                "status": "WRITESTAGED",
                "object_size": r["size_bytes"],
                "chunk_size": CHUNK_SIZE,
                "chunk_layout": layout,
                "sync_status": "STAGED",
            }
        if action == "commit_write":
            entry = self.objects.get(uri)
            if entry is None:
                return {"status": "ERROR", "error_message": "not staged"}
            crcs = r.get("chunk_checksums")
            if crcs is not None:
                if len(crcs) != len(entry["layout"]):
                    return {"status": "ERROR", "error_message": "bad checksum count"}
                for chunk, crc in zip(entry["layout"], crcs):
                    chunk["checksum"] = crc
            entry["status"] = "LOCAL_ONLY" if r.get("local_only") else "COMMITTED"
            return {"status": "WRITECOMMITTED", "object_size": entry["size"]}
        if action in ("locate", "ingest"):
            entry = self.objects.get(uri)
            if entry is None:
                return {"status": "MISS"}
            response = {
                "status": "HIT",
                "object_size": entry["size"],
                "chunk_size": CHUNK_SIZE,
                "chunk_layout": entry["layout"],
                "generation": entry["generation"],
            }
            if self.reports_sync_status:
                response["sync_status"] = entry["status"]
            return response
        if action == "evict":
            self.objects.pop(uri, None)
            return {"status": "EVICTED"}
        return {"status": "PONG"}

    def actions(self, name):
        with self.lock:
            return [r for r in self.requests if r.get("action") == name]

    def corrupt(self, uri, chunk_index=0):
        chunk = self.objects[uri]["layout"][chunk_index]
        with open(self.arena, "r+b") as f:
            f.seek(chunk["offset"] + 17)
            byte = f.read(1)
            f.seek(chunk["offset"] + 17)
            f.write(bytes([byte[0] ^ 0xFF]))


class AdapterTestCase(unittest.TestCase):
    def setUp(self):
        # Short path: AF_UNIX paths have a low length limit on macOS
        self.directory = tempfile.mkdtemp(prefix="hcl2", dir="/tmp")
        self.daemon = FakeDaemon(self.directory)
        self.adapter = self.make_adapter()
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.addCleanup(self.adapter.close)

    def make_adapter(self, namespace="ns1"):
        return HyperCacheL2Adapter(
            HyperCacheL2AdapterConfig(namespace=namespace, socket_path=self.daemon.socket_path, timeout_s=2)
        )

    def store(self, adapter, keys, objects):
        task_id = adapter.submit_store_task(keys, objects)
        deadline = time.time() + 10
        while time.time() < deadline:
            select.select([adapter.get_store_event_fd()], [], [], 0.05)
            done = adapter.pop_completed_store_tasks()
            if task_id in done:
                return done[task_id]
        self.fail("store task did not finish")

    def lookup(self, adapter, keys):
        task_id = adapter.submit_lookup_and_lock_task(keys, {})
        return self._wait(adapter.query_lookup_and_lock_result, task_id, len(keys))

    def load(self, adapter, keys, objects):
        task_id = adapter.submit_load_task(keys, objects)
        return self._wait(adapter.query_load_result, task_id, len(keys))

    def _wait(self, query, task_id, count):
        deadline = time.time() + 10
        while time.time() < deadline:
            bitmap = query(task_id)
            if bitmap is not None:
                self.assertIsNone(query(task_id), "results must be returned once")
                return [bitmap.test(i) for i in range(count)]
            time.sleep(0.005)
        self.fail("task did not finish")

    def test_round_trip_across_chunks(self):
        key = Key(b"\x01\x02")
        data = payload(CHUNK_SIZE * 2 + 8192)
        self.assertEqual(self.store(self.adapter, [key], [Obj(data)]), len(data))
        self.assertEqual(self.lookup(self.adapter, [key, Key(b"\x09")]), [True, False])

        target = Obj(bytes(len(data)))
        self.assertEqual(self.load(self.adapter, [key], [target]), [True])
        self.assertEqual(bytes(target.byte_array), data)

        commit = self.daemon.actions("commit_write")[0]
        self.assertTrue(commit["local_only"])
        self.assertEqual(len(commit["chunk_checksums"]), 3)
        self.assertFalse(self.daemon.actions("stage_write")[0]["auto_flush_s3"])

    @unittest.skipIf(HAVE_LMCACHE, "inspects the stub's event log")
    def test_notifies_only_real_changes(self):
        key = Key(b"\x01")
        self.store(self.adapter, [key], [Obj(payload(8192))])
        self.store(self.adapter, [key], [Obj(payload(8192))])
        self.assertEqual(self.adapter.events["stored"], [key])
        self.load(self.adapter, [key, Key(b"\x02")], [Obj(bytes(8192)), Obj(bytes(8192))])
        self.assertEqual(self.adapter.events["accessed"], [key])

    def test_event_fds_are_distinct_and_fire(self):
        fds = {
            self.adapter.get_store_event_fd(),
            self.adapter.get_lookup_and_lock_event_fd(),
            self.adapter.get_load_event_fd(),
        }
        self.assertEqual(len(fds), 3)
        self.adapter.submit_lookup_and_lock_task([Key(b"\x01")], {})
        ready, _, _ = select.select([self.adapter.get_lookup_and_lock_event_fd()], [], [], 5)
        self.assertTrue(ready)

    def test_key_fields_isolate_blocks(self):
        base = Key(b"\xaa", cache_salt="tenant-a")
        self.store(self.adapter, [base], [Obj(payload(8192))])
        others = [
            Key(b"\xab", cache_salt="tenant-a"),
            Key(b"\xaa", cache_salt="tenant-b"),
            Key(b"\xaa", cache_salt=""),
            Key(b"\xaa", model_name="org/other", cache_salt="tenant-a"),
            Key(b"\xaa", kv_rank=1, cache_salt="tenant-a"),
            Key(b"\xaa", object_group_id=1, cache_salt="tenant-a"),
        ]
        self.assertEqual(self.lookup(self.adapter, others), [False] * len(others))
        self.assertEqual(self.lookup(self.adapter, [base]), [True])

    def test_no_salt_cannot_collide_with_any_salt(self):
        self.store(self.adapter, [Key(b"\xaa")], [Obj(payload(8192))])
        self.assertEqual(self.lookup(self.adapter, [Key(b"\xaa", cache_salt="_")]), [False])

    def test_model_name_cannot_add_path_levels(self):
        uri = self.adapter._uri(Key(b"\x01", model_name="a/b/../c", cache_salt="x/y"))
        self.assertEqual(uri.count("/"), len("hypercache://kv_cache/ns/m/r/g/s/h".split("/")) - 1)

    def test_changed_namespace_misses(self):
        key = Key(b"\x01")
        self.store(self.adapter, [key], [Obj(payload(8192))])
        other = self.make_adapter(namespace="ns2")
        self.addCleanup(other.close)
        self.assertEqual(self.lookup(other, [key]), [False])

    def test_namespace_is_required(self):
        with self.assertRaises(ValueError):
            HyperCacheL2AdapterConfig.from_dict({"socket_path": "/tmp/x"})

    def test_corrupt_block_is_a_miss_and_evicted(self):
        key = Key(b"\x01")
        data = payload(CHUNK_SIZE + 4096)
        self.store(self.adapter, [key], [Obj(data)])
        self.daemon.corrupt(self.adapter._uri(key), chunk_index=1)

        self.assertEqual(self.load(self.adapter, [key], [Obj(bytes(len(data)))]), [False])
        self.assertEqual(self.lookup(self.adapter, [key]), [False])
        self.assertEqual(self.adapter.report_status()["checksum_failures"], 1)

    def test_one_bad_key_does_not_fail_the_others(self):
        good, bad = Key(b"\x01"), Key(b"\x02")
        self.store(self.adapter, [good, bad], [Obj(payload(8192)), Obj(payload(8192, seed=3))])
        self.daemon.corrupt(self.adapter._uri(bad))
        targets = [Obj(bytes(8192)), Obj(bytes(8192))]
        self.assertEqual(self.load(self.adapter, [good, bad], targets), [True, False])
        self.assertEqual(bytes(targets[0].byte_array), payload(8192))

    def test_wrong_buffer_size_is_a_miss_and_keeps_the_block(self):
        key = Key(b"\x01")
        self.store(self.adapter, [key], [Obj(payload(8192))])
        self.assertEqual(self.load(self.adapter, [key], [Obj(bytes(4096))]), [False])
        self.assertEqual(self.lookup(self.adapter, [key]), [True])

    def test_block_replaced_during_read_is_not_evicted(self):
        key = Key(b"\x01")
        self.store(self.adapter, [key], [Obj(payload(8192))])
        uri = self.adapter._uri(key)
        self.daemon.corrupt(uri)

        class ReplacingReader:
            def __init__(self, reader, daemon):
                self.reader, self.daemon = reader, daemon

            def read_range_into(self, *args):
                try:
                    return self.reader.read_range_into(*args)
                finally:
                    self.daemon.objects[uri]["generation"] += 1

            def close(self):
                self.reader.close()

        self.adapter._reader = ReplacingReader(self.adapter._reader, self.daemon)
        self.assertEqual(self.load(self.adapter, [key], [Obj(bytes(8192))]), [False])
        self.assertIn(uri, self.daemon.objects)

    def test_uncommitted_and_unchecked_blocks_are_misses(self):
        key = Key(b"\x01")
        uri = self.adapter._uri(key)
        client = self.adapter._client()
        client.stage_write(uri, 8192, auto_flush_s3=False)
        self.assertEqual(self.lookup(self.adapter, [key]), [False])
        self.assertEqual(self.load(self.adapter, [key], [Obj(bytes(8192))]), [False])

        client.commit_write(uri, local_only=True)
        self.assertEqual(self.lookup(self.adapter, [key]), [False], "no checksums")

    def test_interrupted_store_can_be_retried(self):
        key = Key(b"\x01")
        self.adapter._client().stage_write(self.adapter._uri(key), 8192, auto_flush_s3=False)
        self.assertEqual(self.store(self.adapter, [key], [Obj(payload(8192))]), 8192)
        self.assertEqual(self.lookup(self.adapter, [key]), [True])

    def test_older_daemon_is_refused_before_any_write(self):
        self.daemon.reports_sync_status = False
        key = Key(b"\x01")
        self.assertEqual(self.store(self.adapter, [key], [Obj(payload(8192))]), -1)
        self.assertEqual(self.daemon.actions("commit_write"), [])
        self.assertEqual(self.daemon.objects, {})
        self.assertEqual(self.lookup(self.adapter, [key]), [False])

    def test_failed_write_fails_the_task_and_cleans_up(self):
        def fail(*_):
            raise OSError("disk error")

        self.adapter._writer.write_chunks = fail
        self.assertEqual(self.store(self.adapter, [Key(b"\x01")], [Obj(payload(8192))]), -1)
        self.assertEqual(self.daemon.objects, {})
        self.assertEqual(self.daemon.actions("commit_write"), [])

    def test_delete_skips_locked_blocks_until_unlocked(self):
        key = Key(b"\x01")
        self.store(self.adapter, [key], [Obj(payload(8192))])
        self.lookup(self.adapter, [key])
        self.lookup(self.adapter, [key])

        self.adapter.delete([key])
        self.assertEqual(len(self.daemon.objects), 1)
        self.adapter.submit_unlock([key])
        self.adapter.delete([key])
        self.assertEqual(len(self.daemon.objects), 1)
        self.adapter.submit_unlock([key])
        self.adapter.delete([key])
        self.assertEqual(self.daemon.objects, {})

    def test_unlock_of_unknown_key_is_harmless(self):
        self.adapter.submit_unlock([Key(b"\x77")])
        self.assertEqual(self.adapter.report_status()["locked_keys"], 0)

    def test_daemon_down_means_misses_and_failed_stores(self):
        key = Key(b"\x01")
        self.store(self.adapter, [key], [Obj(payload(8192))])
        self.adapter.close()
        self.daemon.stop()

        adapter = self.make_adapter()
        self.adapter = adapter
        self.addCleanup(adapter.close)
        self.assertEqual(self.lookup(adapter, [key]), [False])
        self.assertEqual(self.load(adapter, [key], [Obj(bytes(8192))]), [False])
        self.assertEqual(self.store(adapter, [key], [Obj(payload(8192))]), -1)
        adapter.submit_unlock([key])
        self.assertFalse(adapter.report_status()["is_healthy"])

    def test_concurrent_stores_of_the_same_key_stay_readable(self):
        key = Key(b"\x01")
        objects = [Obj(payload(8192)) for _ in range(16)]
        self.store(self.adapter, [key] * 16, objects)
        target = Obj(bytes(8192))
        self.assertEqual(self.load(self.adapter, [key], [target]), [True])
        self.assertEqual(bytes(target.byte_array), payload(8192))


class KVStoreLocalOnlyTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp(prefix="hckv", dir="/tmp")
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)
        self.daemon = FakeDaemon(self.directory)

    def test_commit_is_local_only_and_staged_blocks_are_misses(self):
        from hypercache.kv_store import HyperCacheKVStore

        store = HyperCacheKVStore(socket_path=self.daemon.socket_path, default_tier="nvme", prefer_gds=False)
        self.addCleanup(store.close)
        block = payload(8192)
        self.assertTrue(store.put_kv_block("committed", block))
        self.assertTrue(self.daemon.actions("commit_write")[0]["local_only"])
        self.assertEqual(bytes(store.get_kv_block("committed")), block)

        store._client.stage_write(store._make_uri("staged"), 8192)
        self.assertFalse(store.exists_kv_block("staged"))
        self.assertIsNone(store.get_kv_block("staged"))


if __name__ == "__main__":
    unittest.main()
