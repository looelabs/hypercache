"""
Unit tests for NUMA Topology, MAP_SHARED Multi-Tenancy, and Pipelined Streaming Loader.
"""

import json
import mmap
import os
import socket
import tempfile
import threading
import unittest

from hypercache.client import DiscoveredDeviceInfo, HyperCacheClient
from hypercache.datapath import ChunkLocation, DirectBlockReader
from hypercache.pipeline import HyperCachePipelinedLoader, pipeline_load


class TestPipelinedAndNuma(unittest.TestCase):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".tmp_adv"))
    SOCKET_PATH = os.path.join(BASE_DIR, "test_adv.sock")
    DEV0 = os.path.join(BASE_DIR, "test_nvme0.raw")
    DEV1 = os.path.join(BASE_DIR, "test_nvme1.raw")

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.BASE_DIR, exist_ok=True)
        # Create 2 test backing device files (32 MB total = 8 layers of 4 MB)
        cls.layer_size = 4 * 1024 * 1024
        cls.total_size = 32 * 1024 * 1024
        cls.full_payload = b"A" * (cls.total_size // 2) + b"B" * (cls.total_size // 2)

        with open(cls.DEV0, "wb") as f:
            f.seek(4096)
            f.write(cls.full_payload[: cls.total_size // 2])

        with open(cls.DEV1, "wb") as f:
            f.seek(4096)
            f.write(cls.full_payload[cls.total_size // 2 :])

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

                        if req.get("action") == "get_devices":
                            resp = {
                                "status": "DEVICES",
                                "devices": [
                                    {"path": "/dev/nvme0n1", "is_nvme": True, "is_virtual": False, "is_partition": False, "is_rotational": False, "size_bytes": 1000000000000, "model": "Samsung 990 Pro", "numa_node": 0},
                                    {"path": "/dev/nvme1n1", "is_nvme": True, "is_virtual": False, "is_partition": False, "is_rotational": False, "size_bytes": 1000000000000, "model": "Samsung 990 Pro", "numa_node": 1},
                                ]
                            }
                        elif req.get("action") in ("locate", "ingest"):
                            resp = {
                                "status": "HIT",
                                "object_size": cls.total_size,
                                "chunk_size": cls.total_size // 2,
                                "chunk_layout": [
                                    {"device": cls.DEV0, "offset": 4096, "length": cls.total_size // 2},
                                    {"device": cls.DEV1, "offset": 4096, "length": cls.total_size // 2},
                                ],
                            }
                        else:
                            resp = {"status": "PONG"}
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
        for p in [cls.SOCKET_PATH, cls.DEV0, cls.DEV1]:
            if os.path.exists(p):
                os.remove(p)

    def test_numa_device_info_parsing(self):
        client = HyperCacheClient(socket_path=self.SOCKET_PATH)
        devices = client.get_devices()
        self.assertEqual(len(devices), 2)
        self.assertEqual(devices[0].path, "/dev/nvme0n1")
        self.assertEqual(devices[0].numa_node, 0)
        self.assertEqual(devices[1].path, "/dev/nvme1n1")
        self.assertEqual(devices[1].numa_node, 1)
        client.close()

    def test_map_shared_ram_chunk_reading(self):
        # Create a mock .shm chunk file
        with tempfile.NamedTemporaryFile(suffix=".shm", delete=False) as f:
            payload = b"MAP_SHARED_TEST_PAYLOAD_" * 1024
            f.write(payload)
            shm_path = f.name

        try:
            chunk = ChunkLocation(device=shm_path, offset=0, length=len(payload), tier="ram")
            reader = DirectBlockReader()
            data = reader.read_range([chunk], 0, len(payload))
            self.assertEqual(bytes(data), payload)
            reader.close()
        finally:
            if os.path.exists(shm_path):
                os.remove(shm_path)

    def test_pipelined_layer_loader(self):
        uri = "hypercache://models/llama-3/weights.safetensors"
        loader = HyperCachePipelinedLoader(
            uri=uri,
            socket_path=self.SOCKET_PATH,
            layer_size_bytes=self.layer_size,
            prefetch_ahead=2,
        )

        # 8 layers of 4MB each
        self.assertEqual(len(loader.layers), 8)
        self.assertEqual(loader.layers[0].name, "embedding")
        self.assertEqual(loader.layers[1].name, "layer_0")

        # Test loading first layer immediately (<50ms cold start unblock)
        first_layer, first_data = loader.load_initial_layer()
        self.assertEqual(first_layer.layer_idx, 0)
        self.assertEqual(len(first_data), self.layer_size)
        self.assertEqual(bytes(first_data), self.full_payload[: self.layer_size])

        # Test streaming subsequent layers
        streamed_chunks = []
        for desc, layer_bytes in loader.stream_layers():
            streamed_chunks.append(bytes(layer_bytes))

        self.assertEqual(len(streamed_chunks), 8)
        full_streamed = b"".join(streamed_chunks)
        self.assertEqual(full_streamed, self.full_payload)
        loader.close()


if __name__ == "__main__":
    unittest.main()
