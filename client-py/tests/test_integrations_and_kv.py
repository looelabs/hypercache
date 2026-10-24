"""
Unit tests for HyperCache Ecosystem Integrations (Safetensors/vLLM, PyTorch DCP, and KV-Cache Store).
"""

import json
import os
import socket
import threading
import unittest

from hypercache.integrations import (
    HyperCacheDCPStorageReader,
    HyperCacheDCPStorageWriter,
    HyperCachevLLMWeightLoader,
    load_safetensors_state_dict,
    save_safetensors_state_dict,
)
from hypercache.kv_store import HyperCacheKVStore


class TestIntegrationsAndKVStore(unittest.TestCase):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".tmp_integrations"))
    SOCKET_PATH = os.path.join(BASE_DIR, "test_integrations.sock")
    ARENA_FILE = os.path.join(BASE_DIR, "arena.raw")

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.BASE_DIR, exist_ok=True)
        with open(cls.ARENA_FILE, "wb") as f:
            f.write(b"\x00" * (16 * 1024 * 1024))

        if os.path.exists(cls.SOCKET_PATH):
            os.remove(cls.SOCKET_PATH)

        cls._catalog = {}
        cls._next_offset = 4096
        cls._lock = threading.Lock()
        cls._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        cls._server_sock.bind(cls.SOCKET_PATH)
        cls._server_sock.listen(5)
        cls._running = True

        def handle_conn(conn):
            buf = ""
            while cls._running:
                try:
                    data = conn.recv(4096)
                    if not data:
                        break
                    buf += data.decode("utf-8")
                    while "\n" in buf:
                        line, buf = buf.split("\n", 1)
                        line = line.strip()
                        if not line:
                            continue
                        req = json.loads(line)
                        action = req.get("action")
                        uri = req.get("uri", "")

                        with cls._lock:
                            if action == "stage_write":
                                size = int(req.get("size_bytes", 0))
                                off = cls._next_offset
                                cls._next_offset += ((size + 4095) // 4096) * 4096
                                layout = [{"device": cls.ARENA_FILE, "offset": off, "length": size, "tier": "nvme"}]
                                cls._catalog[uri] = {"size": size, "layout": layout}
                                resp = {
                                    "status": "WRITESTAGED",
                                    "object_size": size,
                                    "chunk_size": 4194304,
                                    "chunk_layout": layout,
                                    "sync_status": "STAGED",
                                }
                            elif action == "commit_write":
                                info = cls._catalog.get(uri, {"size": 0})
                                resp = {
                                    "status": "WRITECOMMITTED",
                                    "object_size": info["size"],
                                    "sync_status": "COMMITTED",
                                }
                            elif action in ("locate", "ingest"):
                                if uri in cls._catalog:
                                    info = cls._catalog[uri]
                                    resp = {
                                        "status": "HIT",
                                        "object_size": info["size"],
                                        "chunk_size": 4194304,
                                        "chunk_layout": info["layout"],
                                    }
                                else:
                                    resp = {"status": "MISS"}
                            else:
                                resp = {"status": "PONG"}

                        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                except Exception:
                    break
            try:
                conn.close()
            except Exception:
                pass

        def accept_loop():
            while cls._running:
                try:
                    conn, _ = cls._server_sock.accept()
                    threading.Thread(target=handle_conn, args=(conn,), daemon=True).start()
                except Exception:
                    break

        cls._thread = threading.Thread(target=accept_loop, daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls):
        cls._running = False
        try:
            cls._server_sock.close()
        except Exception:
            pass
        for p in [cls.SOCKET_PATH, cls.ARENA_FILE]:
            if os.path.exists(p):
                os.remove(p)

    def test_safetensors_save_load_and_vllm_loader(self):
        uri = "hypercache://models/llama3/shard_001.safetensors"
        try:
            import torch
            w0 = torch.arange(16, dtype=torch.float32).reshape(4, 4)
            w1 = torch.ones((2, 8), dtype=torch.float16)
            state_in = {"layers.0.attn.weight": w0, "layers.0.mlp.weight": w1}
            has_torch = True
        except ImportError:
            state_in = {"layers.0.attn.weight": b"ATTN_WEIGHT_BYTES_1234", "layers.0.mlp.weight": b"MLP_WEIGHT_BYTES_5678"}
            has_torch = False

        written = save_safetensors_state_dict(
            state_in,
            uri,
            metadata={"format": "pt"},
            socket_path=self.SOCKET_PATH,
            pinned=True,
        )
        self.assertGreater(written, 0)

        # Full load
        loaded_all = load_safetensors_state_dict(uri, socket_path=self.SOCKET_PATH, prefer_gds=False)
        self.assertEqual(set(loaded_all.keys()), {"layers.0.attn.weight", "layers.0.mlp.weight"})
        if has_torch:
            import torch
            self.assertTrue(torch.equal(loaded_all["layers.0.attn.weight"], state_in["layers.0.attn.weight"]))
            self.assertTrue(torch.equal(loaded_all["layers.0.mlp.weight"], state_in["layers.0.mlp.weight"]))

        # Selective tensor slice load
        loaded_sub = load_safetensors_state_dict(
            uri,
            socket_path=self.SOCKET_PATH,
            prefer_gds=False,
            tensor_names=["layers.0.mlp.weight"],
        )
        self.assertEqual(list(loaded_sub.keys()), ["layers.0.mlp.weight"])

        # vLLM Weight Loader iterator
        vllm_loader = HyperCachevLLMWeightLoader(socket_path=self.SOCKET_PATH, prefer_gds=False)
        pairs = list(vllm_loader.iter_shard_tensors([uri]))
        self.assertEqual(len(pairs), 2)

    def test_dcp_storage_writer_and_reader(self):
        base_uri = "hypercache://checkpoints/fsdp_step_100"
        writer = HyperCacheDCPStorageWriter(
            base_uri=base_uri,
            rank=0,
            world_size=2,
            socket_path=self.SOCKET_PATH,
        )
        shard_data = {"step": 100, "loss": 1.25}
        saved_uri = writer.save_rank_shard(shard_data, extra_metadata={"epoch": 3})
        self.assertTrue(saved_uri.endswith("shard_rank_00000_of_00002.pt"))

        reader = HyperCacheDCPStorageReader(
            base_uri=base_uri,
            rank=0,
            world_size=2,
            socket_path=self.SOCKET_PATH,
            prefer_gds=False,
        )
        meta = reader.read_metadata()
        self.assertEqual(meta["world_size"], 2)
        self.assertEqual(meta["extra_metadata"]["epoch"], 3)

        restored = reader.load_rank_shard(rank=0)
        self.assertEqual(restored["step"], 100)
        self.assertAlmostEqual(restored["loss"], 1.25)

    def test_kv_cache_prefix_store(self):
        kv = HyperCacheKVStore(socket_path=self.SOCKET_PATH, prefer_gds=False)
        tokens = [128000, 9906, 1917, 315, 42]
        block_key = kv.compute_prefix_hash(tokens, model_id="meta-llama/Llama-3.1-70B", layer_idx=0)
        self.assertFalse(kv.exists_kv_block(block_key))

        kv_payload = b"KV_CACHE_FP16_ATTENTION_STATES_" * 64
        ok = kv.put_kv_block(block_key, kv_payload, tier="ram", pinned=True)
        self.assertTrue(ok)
        self.assertTrue(kv.exists_kv_block(block_key))

        fetched = kv.get_kv_block(block_key)
        self.assertEqual(fetched, kv_payload)
        kv.close()


if __name__ == "__main__":
    unittest.main()
