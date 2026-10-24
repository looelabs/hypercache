"""
Unit tests for HyperCache UDS Client.
"""

import json
import os
import socket
import threading
import unittest

from hypercache.client import HyperCacheClient, LocateResult
from hypercache.datapath import ChunkLocation


class TestHyperCacheClient(unittest.TestCase):
    SOCKET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".tmp"))
    SOCKET_PATH = os.path.join(SOCKET_DIR, "test_hypercache_uds.sock")

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.SOCKET_DIR, exist_ok=True)
        if os.path.exists(cls.SOCKET_PATH):
            os.remove(cls.SOCKET_PATH)

        cls._server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        cls._server_sock.bind(cls.SOCKET_PATH)
        cls._server_sock.listen(5)
        cls._running = True

        def handle_client_conn(conn):
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

                        if req.get("action") == "ping":
                            resp = {"status": "PONG"}
                        elif req.get("action") == "get_devices":
                            resp = {
                                "status": "DEVICES",
                                "devices": [
                                    {"path": "/dev/nvme0n1", "is_nvme": True, "is_virtual": False, "is_partition": False, "is_rotational": False, "size_bytes": 1000000000000, "model": "Samsung 990 Pro"},
                                    {"path": "/dev/vdc1", "is_nvme": False, "is_virtual": True, "is_partition": True, "is_rotational": False, "size_bytes": 5497558138880, "model": "VirtIO Virtual Block Device"},
                                ]
                            }
                        elif req.get("action") in ("locate", "ingest"):
                            uri = req.get("uri")
                            if "llama" in uri:
                                resp = {
                                    "status": "HIT",
                                    "object_size": 20971520,
                                    "chunk_size": 4194304,
                                    "chunk_layout": [
                                        {"device": "/tmp/nvme0.raw", "offset": 4096, "length": 4194304},
                                        {"device": "/tmp/nvme1.raw", "offset": 4096, "length": 4194304},
                                    ],
                                }
                            else:
                                resp = {"status": "MISS"}
                        elif req.get("action") == "stage_write":
                            resp = {
                                "status": "WRITESTAGED",
                                "object_size": req.get("size_bytes", 8388608),
                                "chunk_size": 4194304,
                                "chunk_layout": [
                                    {"device": "/tmp/nvme0.raw", "offset": 4096, "length": 4194304},
                                    {"device": "/tmp/nvme1.raw", "offset": 4096, "length": 4194304},
                                ],
                                "sync_status": "STAGED",
                            }
                        elif req.get("action") == "commit_write":
                            sync_status = "SYNCED_BACKEND" if req.get("write_through") else "COMMITTED"
                            resp = {
                                "status": "WRITECOMMITTED",
                                "object_size": 8388608,
                                "sync_status": sync_status,
                            }
                        elif req.get("action") == "get_write_status":
                            resp = {
                                "status": "WRITESTATUS",
                                "sync_status": "COMMITTED",
                                "sync_progress_percent": 100.0,
                            }
                        else:
                            resp = {"status": "ERROR", "error_message": "Unknown action"}

                        conn.sendall((json.dumps(resp) + "\n").encode("utf-8"))
                except Exception:
                    break
            try:
                conn.close()
            except Exception:
                pass

        def server_loop():
            while cls._running:
                try:
                    conn, _ = cls._server_sock.accept()
                    threading.Thread(target=handle_client_conn, args=(conn,), daemon=True).start()
                except Exception:
                    break

        cls._thread = threading.Thread(target=server_loop, daemon=True)
        cls._thread.start()

    @classmethod
    def tearDownClass(cls):
        cls._running = False
        try:
            cls._server_sock.close()
        except Exception:
            pass
        if os.path.exists(cls.SOCKET_PATH):
            os.remove(cls.SOCKET_PATH)

    def test_ping(self):
        client = HyperCacheClient(socket_path=self.SOCKET_PATH)
        self.assertTrue(client.ping())
        client.close()

    def test_locate_hit(self):
        client = HyperCacheClient(socket_path=self.SOCKET_PATH)
        res = client.locate("hypercache://models/llama-3/weights.safetensors")
        self.assertTrue(res.is_hit)
        self.assertEqual(res.object_size, 20971520)
        self.assertEqual(res.chunk_size, 4194304)
        self.assertEqual(len(res.chunk_layout), 2)
        self.assertEqual(res.chunk_layout[0].device, "/tmp/nvme0.raw")
        self.assertEqual(res.chunk_layout[0].offset, 4096)
        client.close()

    def test_locate_miss(self):
        client = HyperCacheClient(socket_path=self.SOCKET_PATH)
        res = client.locate("hypercache://models/nonexistent/weights.pt")
        self.assertFalse(res.is_hit)
        self.assertEqual(res.status, "MISS")
        client.close()

    def test_get_devices(self):
        client = HyperCacheClient(socket_path=self.SOCKET_PATH)
        devices = client.get_devices()
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0].path, "/dev/nvme0n1")
        self.assertTrue(devices[0].is_nvme)
        self.assertFalse(devices[0].is_virtual)
        self.assertEqual(devices[0].model, "Samsung 990 Pro")

        # Test virtual disk / partition
        self.assertEqual(devices[1].path, "/dev/vdc1")
        self.assertFalse(devices[1].is_nvme)
        self.assertTrue(devices[1].is_virtual)
        self.assertTrue(devices[1].is_partition)
        self.assertEqual(devices[1].size_bytes, 5497558138880)
        self.assertEqual(devices[1].model, "VirtIO Virtual Block Device")
        client.close()

    def test_preload(self):
        import hypercache
        # Test client.preload
        client = HyperCacheClient(socket_path=self.SOCKET_PATH)
        res = client.preload("hypercache://models/llama-3/weights.safetensors")
        self.assertTrue(res.is_hit)
        self.assertEqual(res.object_size, 20971520)
        client.close()

        # Test top-level hypercache.preload
        res2 = hypercache.preload("hypercache://models/llama-3/weights.safetensors", socket_path=self.SOCKET_PATH)
        self.assertTrue(res2.is_hit)
        self.assertEqual(res2.object_size, 20971520)

    def test_ram_tier_datapath_reader(self):
        import tempfile
        from hypercache.datapath import DirectBlockReader

        with tempfile.NamedTemporaryFile(suffix=".shm", delete=False) as f:
            test_data = b"HYPERCACHE_TIER1_RAM_CHUNK_" * 1024
            f.write(test_data)
            shm_path = f.name

        try:
            chunk = ChunkLocation(device=shm_path, offset=0, length=len(test_data), tier="ram")
            self.assertEqual(chunk.tier, "ram")
            self.assertEqual(chunk.to_dict()["tier"], "ram")

            reader = DirectBlockReader()
            read_bytes = reader.read_range([chunk], 0, len(test_data))
            self.assertEqual(bytes(read_bytes), test_data)

            # Test slice within RAM chunk
            slice_bytes = reader.read_range([chunk], 100, 50)
            self.assertEqual(bytes(slice_bytes), test_data[100:150])
            reader.close()
        finally:
            if os.path.exists(shm_path):
                os.remove(shm_path)

    def test_stage_write_and_commit(self):
        client = HyperCacheClient(socket_path=self.SOCKET_PATH)
        uri = "hypercache://checkpoints/epoch_1.pt"
        res = client.stage_write(uri, size_bytes=8388608)
        self.assertTrue(res.is_staged)
        self.assertEqual(res.object_size, 8388608)
        self.assertEqual(len(res.chunk_layout), 2)
        self.assertEqual(res.sync_status, "STAGED")

        # Commit write (asynchronous write-back)
        committed = client.commit_write(uri, write_through=False)
        self.assertTrue(committed)

        # Get write status
        status_info = client.get_write_status(uri)
        self.assertEqual(status_info["status"], "COMMITTED")
        self.assertEqual(status_info["progress_percent"], 100.0)
        client.close()

    def test_stage_write_and_commit_write_through(self):
        client = HyperCacheClient(socket_path=self.SOCKET_PATH)
        uri = "hypercache://checkpoints/epoch_sync.pt"
        res = client.stage_write(uri, size_bytes=8388608, write_through=True)
        self.assertTrue(res.is_staged)
        self.assertEqual(res.object_size, 8388608)
        self.assertEqual(len(res.chunk_layout), 2)

        # Commit write (synchronous write-through)
        committed = client.commit_write(uri, write_through=True)
        self.assertTrue(committed)
        client.close()


if __name__ == "__main__":
    unittest.main()
