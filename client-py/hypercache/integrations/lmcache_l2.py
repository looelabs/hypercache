"""
LMCache L2 adapter backed by the HyperCache daemon.

Lets an LMCache MP-mode server keep KV-cache chunks on HyperCache's RAM and NVMe tiers.
LMCache loads it as a plugin, so `hypercache` must be importable by the LMCache server:

    {
      "type": "plugin",
      "module_path": "hypercache.integrations.lmcache_l2",
      "class_name": "HyperCacheL2Adapter",
      "adapter_params": {
        "namespace": "qwen2.5-7b-awq-rev1234",
        "socket_path": "/run/hypercache/hypercache.sock",
        "tier": "nvme"
      }
    }

Behaviour worth knowing:

- Blocks are LOCAL_ONLY. They are never flushed to a backend and stay evictable, because
  KV data derives from user prompts and can be recomputed.
- `namespace` is required. LMCache's key has a model name but no revision, dtype or engine
  settings, so changing weights under the same name would otherwise return wrong KV data.
  Set it to something that changes when any of those change.
- Any doubt about a block is a miss. Lookup only accepts committed blocks with a checksum
  for every chunk, and load verifies every chunk before reporting success.
- Capacity belongs to the daemon, which evicts under pressure. LMCache's global eviction is
  off, so `list_l2_keys` and IsolatedLRU quotas are not supported. The daemon can evict a
  block after lookup succeeded; that load then reports a miss for the key.

Requires LMCache 0.5.5 or 0.5.6, and a daemon with local-only commit support. Against an
older daemon, stores fail before any data is written and lookups miss.

Only this module imports LMCache. `hypercache.integrations` must not import it.
"""

from __future__ import annotations

import itertools
import logging
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import quote

import torch  # noqa: F401  # must precede lmcache_native
from lmcache.lmcache_native import Bitmap
from lmcache.v1.distributed.internal_api import L2StoreResult
from lmcache.v1.distributed.l2_adapters.base import L2AdapterInterface, L2TaskId
from lmcache.v1.distributed.l2_adapters.config import L2AdapterConfigBase
from lmcache.v1.platform import create_event_notifier

from hypercache.client import HyperCacheClient, LocateResult
from hypercache.datapath import ChecksumMismatchError, DirectBlockReader, DirectBlockWriter

logger = logging.getLogger(__name__)

_LOCAL_ONLY = "LOCAL_ONLY"


class HyperCacheL2AdapterConfig(L2AdapterConfigBase):
    """Config for HyperCacheL2Adapter."""

    def __init__(
        self,
        namespace: str,
        socket_path: Optional[str] = None,
        tier: str = "nvme",
        num_workers: int = 4,
        timeout_s: float = 30.0,
    ):
        if not namespace:
            raise ValueError("namespace is required: use a value that changes when the model weights or dtype change")
        if num_workers < 1:
            raise ValueError("num_workers must be at least 1")
        self.namespace = namespace
        self.socket_path = socket_path
        self.tier = tier
        self.num_workers = num_workers
        self.timeout_s = timeout_s

    @classmethod
    def from_dict(cls, d: dict) -> "HyperCacheL2AdapterConfig":
        return cls(
            namespace=d.get("namespace", ""),
            socket_path=d.get("socket_path"),
            tier=d.get("tier", "nvme"),
            num_workers=int(d.get("num_workers", 4)),
            timeout_s=float(d.get("timeout_s", 30.0)),
        )

    @classmethod
    def help(cls) -> str:
        return (
            "HyperCacheL2Adapter config fields:\n"
            "- namespace (str, required): model fingerprint; change it when weights or dtype change\n"
            "- socket_path (str): daemon socket (default: the client's default path)\n"
            "- tier (str): 'nvme' or 'ram' (default 'nvme')\n"
            "- num_workers (int): parallel chunk reads and writes per direction (default 4)\n"
            "- timeout_s (float): daemon request timeout in seconds (default 30)\n"
        )


class HyperCacheL2Adapter(L2AdapterInterface):
    """Stores LMCache objects in HyperCache. Called from two LMCache controller threads."""

    def __init__(self, config: Any, **_kwargs: object):
        if isinstance(config, dict):
            config = HyperCacheL2AdapterConfig.from_dict(config)
        super().__init__(0)  # 0 turns off LMCache's global eviction: the daemon owns capacity
        self._config = config
        self._namespace = quote(config.namespace, safe="")

        self._store_efd = create_event_notifier()
        self._lookup_efd = create_event_notifier()
        self._load_efd = create_event_notifier()

        self._lock = threading.Lock()
        self._task_ids = itertools.count()
        self._done_store: Dict[L2TaskId, L2StoreResult] = {}
        self._done_lookup: Dict[L2TaskId, Bitmap] = {}
        self._done_load: Dict[L2TaskId, Bitmap] = {}
        self._locks: Dict[str, int] = {}
        self._storing: set = set()
        self._daemon_supports_local_only = False
        self._closed = False
        self._counters = {"checksum_failures": 0, "load_failures": 0, "store_failures": 0}

        self._clients = threading.local()
        self._all_clients: List[HyperCacheClient] = []
        self._reader = DirectBlockReader(max_workers=config.num_workers, verify_checksums=True)
        self._writer = DirectBlockWriter(max_workers=config.num_workers, direct_on_block_devices=True)

        # Tasks and their per-key work use separate pools, so a task waiting on its keys can't
        # occupy the thread they need. Each direction has its own, so slow commits can't delay reads.
        self._lookup_tasks = ThreadPoolExecutor(2, thread_name_prefix="hc-l2-lookup")
        self._store_tasks = ThreadPoolExecutor(2, thread_name_prefix="hc-l2-store")
        self._load_tasks = ThreadPoolExecutor(2, thread_name_prefix="hc-l2-load")
        self._store_keys = ThreadPoolExecutor(config.num_workers, thread_name_prefix="hc-l2-store-key")
        self._load_keys = ThreadPoolExecutor(config.num_workers, thread_name_prefix="hc-l2-load-key")

    # Event fds

    def get_store_event_fd(self) -> int:
        return self._store_efd.fileno()

    def get_lookup_and_lock_event_fd(self) -> int:
        return self._lookup_efd.fileno()

    def get_load_event_fd(self) -> int:
        return self._load_efd.fileno()

    # Store

    def submit_store_task(self, keys: list, objects: list) -> L2TaskId:
        task_id = next(self._task_ids)
        self._store_tasks.submit(self._run_store, list(keys), list(objects), task_id)
        return task_id

    def pop_completed_store_tasks(self) -> Dict[L2TaskId, L2StoreResult]:
        with self._lock:
            done = self._done_store
            self._done_store = {}
        return done

    def _run_store(self, keys: list, objects: list, task_id: L2TaskId) -> None:
        stored_keys, stored_sizes = [], []
        success = True
        try:
            futures = [self._store_keys.submit(self._store_one, k, o) for k, o in zip(keys, objects)]
            for key, future in zip(keys, futures):
                try:
                    size = future.result()
                except Exception as e:
                    logger.warning("Store failed for %s: %s", self._uri(key), e)
                    self._count("store_failures")
                    success = False
                    continue
                if size > 0:
                    stored_keys.append(key)
                    stored_sizes.append(size)
            if stored_keys:
                self._notify_keys_stored(stored_keys, stored_sizes)
        except Exception:
            logger.exception("Store task %s failed", task_id)
            success = False
        with self._lock:
            self._done_store[task_id] = L2StoreResult(success, sum(stored_sizes))
        self._store_efd.notify()

    def _store_one(self, key: Any, obj: Any) -> int:
        """Returns the bytes stored, or 0 if the object is already stored or being stored."""
        uri = self._uri(key)
        with self._lock:
            if uri in self._storing:
                return 0
            self._storing.add(uri)
        try:
            client = self._client()
            if self._is_valid(client.locate(uri, auto_ingest=False)):
                return 0

            view = memoryview(obj.byte_array).cast("B")
            staged = client.stage_write(uri, len(view), tier=self._config.tier, auto_flush_s3=False)
            if not staged.is_staged:
                raise IOError(f"stage_write failed: {staged.error_message}")
            try:
                self._require_local_only_support(client, uri)
                self._writer.write_chunks(staged.chunk_layout, view)
                checksums = self._chunk_checksums(staged.chunk_layout, view)
                if not client.commit_write(uri, local_only=True, chunk_checksums=checksums):
                    raise IOError("commit_write failed")
            except Exception:
                self._evict_quietly(client, uri)
                raise
            return len(view)
        finally:
            with self._lock:
                self._storing.discard(uri)

    def _require_local_only_support(self, client: HyperCacheClient, uri: str) -> None:
        # An older daemon ignores local_only and would upload the block to its backend, so this
        # has to be settled before the first commit. Only newer daemons report status on locate.
        if self._daemon_supports_local_only:
            return
        if client.locate(uri, auto_ingest=False).sync_status != "STAGED":
            raise IOError("the HyperCache daemon has no local-only commit support; upgrade it")
        self._daemon_supports_local_only = True

    @staticmethod
    def _chunk_checksums(layout: list, view: memoryview) -> List[int]:
        checksums = []
        position = 0
        for chunk in layout:
            checksums.append(zlib.crc32(view[position : position + chunk.length]) & 0xFFFFFFFF)
            position += chunk.length
        return checksums

    # Lookup and lock

    def submit_lookup_and_lock_task(self, keys: list, group_layout_descs: dict) -> L2TaskId:
        task_id = next(self._task_ids)
        self._lookup_tasks.submit(self._run_lookup, list(keys), task_id)
        return task_id

    def query_lookup_and_lock_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._lock:
            return self._done_lookup.pop(task_id, None)

    def submit_unlock(self, keys: list) -> None:
        with self._lock:
            for key in keys:
                uri = self._uri(key)
                count = self._locks.get(uri, 0)
                if count <= 1:
                    self._locks.pop(uri, None)
                else:
                    self._locks[uri] = count - 1

    def _run_lookup(self, keys: list, task_id: L2TaskId) -> None:
        bitmap = Bitmap(len(keys))
        try:
            client = self._client()
            for i, key in enumerate(keys):
                uri = self._uri(key)
                if self._is_valid(client.locate(uri, auto_ingest=False)):
                    with self._lock:
                        self._locks[uri] = self._locks.get(uri, 0) + 1
                    bitmap.set(i)
        except Exception as e:
            # Keys already locked stay reported as hits; the rest are misses
            logger.warning("Lookup failed: %s", e)
        with self._lock:
            self._done_lookup[task_id] = bitmap
        self._lookup_efd.notify()

    @staticmethod
    def _is_valid(location: LocateResult) -> bool:
        return (
            location.is_hit
            and location.sync_status == _LOCAL_ONLY
            and bool(location.chunk_layout)
            and all(chunk.checksum is not None for chunk in location.chunk_layout)
        )

    # Load

    def submit_load_task(self, keys: list, objects: list) -> L2TaskId:
        task_id = next(self._task_ids)
        self._load_tasks.submit(self._run_load, list(keys), list(objects), task_id)
        return task_id

    def query_load_result(self, task_id: L2TaskId) -> Optional[Bitmap]:
        with self._lock:
            return self._done_load.pop(task_id, None)

    def _run_load(self, keys: list, objects: list, task_id: L2TaskId) -> None:
        bitmap = Bitmap(len(keys))
        loaded = []
        try:
            futures = [self._load_keys.submit(self._load_one, k, o) for k, o in zip(keys, objects)]
            for i, (key, future) in enumerate(zip(keys, futures)):
                if future.result():
                    bitmap.set(i)
                    loaded.append(key)
            if loaded:
                self._notify_keys_accessed(loaded)
        except Exception:
            logger.exception("Load task %s failed", task_id)
        with self._lock:
            self._done_load[task_id] = bitmap
        self._load_efd.notify()

    def _load_one(self, key: Any, obj: Any) -> bool:
        uri = self._uri(key)
        client = None
        location = None
        try:
            client = self._client()
            location = client.locate(uri, auto_ingest=False)
            if not self._is_valid(location):
                return False
            view = memoryview(obj.byte_array).cast("B")
            if location.object_size != len(view):
                logger.warning("Size mismatch for %s: stored %d, buffer %d", uri, location.object_size, len(view))
                return False
            return self._reader.read_range_into(location.chunk_layout, 0, len(view), view) == len(view)
        except ChecksumMismatchError as e:
            logger.error("Treating %s as a miss: %s", uri, e)
            self._count("checksum_failures")
            self._evict_if_unchanged(client, uri, location)
            return False
        except Exception as e:
            logger.warning("Load failed for %s: %s", uri, e)
            self._count("load_failures")
            return False

    def _evict_if_unchanged(self, client: HyperCacheClient, uri: str, seen: LocateResult) -> None:
        # A changed generation means the block was replaced during the read, so it isn't corrupt
        try:
            if client.locate(uri, auto_ingest=False).generation == seen.generation:
                client.evict(uri)
        except Exception as e:
            logger.warning("Could not evict %s: %s", uri, e)

    # Delete, status, shutdown

    def delete(self, keys: list) -> None:
        deleted_keys, deleted_sizes = [], []
        client = self._client()
        for key in keys:
            uri = self._uri(key)
            with self._lock:
                if self._locks.get(uri, 0) > 0:
                    continue
            try:
                location = client.locate(uri, auto_ingest=False)
                if location.is_hit and client.evict(uri):
                    deleted_keys.append(key)
                    deleted_sizes.append(location.object_size)
            except Exception as e:
                logger.warning("Delete failed for %s: %s", uri, e)
        if deleted_keys:
            self._notify_keys_deleted(deleted_keys, deleted_sizes)

    def report_status(self) -> dict:
        status: Dict[str, Any] = {"is_healthy": True, "namespace": self._config.namespace}
        try:
            self._client().get_status()
        except Exception as e:
            status.update(is_healthy=False, error=str(e))
        with self._lock:
            status.update(self._counters)
            status["locked_keys"] = len(self._locks)
        return status

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        # Tasks first, because they wait on the per-key pools
        for pool in (self._lookup_tasks, self._store_tasks, self._load_tasks, self._store_keys, self._load_keys):
            pool.shutdown(wait=True)
        self._reader.close()
        self._writer.close()
        with self._lock:
            clients, self._all_clients = self._all_clients, []
        for client in clients:
            client.close()
        self._store_efd.close()
        self._lookup_efd.close()
        self._load_efd.close()

    # Helpers

    def _uri(self, key: Any) -> str:
        # Every field is quoted so none can add a path level. Empty and non-empty salts have
        # different shapes, so no salt can collide with "no salt".
        salt = f"s-{quote(key.cache_salt, safe='')}" if key.cache_salt else "_"
        model = quote(key.model_name, safe="")
        return (
            f"hypercache://kv_cache/{self._namespace}/{model}/{key.kv_rank:x}/"
            f"{key.object_group_id:x}/{salt}/{key.chunk_hash.hex()}.kv"
        )

    def _client(self) -> HyperCacheClient:
        # One per thread: a client serializes its requests, so a slow commit would block lookups
        client = getattr(self._clients, "client", None)
        if client is None:
            client = HyperCacheClient(socket_path=self._config.socket_path, timeout=self._config.timeout_s)
            self._clients.client = client
            with self._lock:
                self._all_clients.append(client)
        return client

    def _count(self, name: str) -> None:
        with self._lock:
            self._counters[name] += 1

    @staticmethod
    def _evict_quietly(client: HyperCacheClient, uri: str) -> None:
        try:
            client.evict(uri)
        except Exception as e:
            logger.warning("Could not clean up %s: %s", uri, e)
