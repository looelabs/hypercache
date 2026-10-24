"""
Unit tests for HyperCache fsspec integration.
"""

import json
import os
import socket
import threading
import unittest
import fsspec

from hypercache.fsspec_fs import HyperCacheFileSystem, HyperCacheFile
from hypercache.datapath import ChunkLocation


class TestHyperCacheFSSpec(unittest.TestCase):
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".tmp_fsspec"))
    SOCKET_PATH = os.path.join(BASE_DIR, "test_fsspec.sock")
    DEV0 = os.path.join(BASE_DIR, "test_dev0.raw")
    DEV1 = os.path.join(BASE_DIR, "test_dev1.raw")

    @classmethod
    def setUpClass(cls):
        os.makedirs(cls.BASE_DIR, exist_ok=True)
        # Create test backing device files
        payload0 = b"CHUNK0_HELLO_HYPERCACHE_" * 1024 # 24 KB
        payload1 = b"CHUNK1_WORLD_HYPERCACHE_" * 1024 # 24 KB

        with open(cls.DEV0, "wb") as f:
            f.seek(4096)
            f.write(payload0)

        with open(cls.DEV1, "wb") as f:
            f.seek(4096)
            f.write(payload1)

        cls._full_payload = payload0 + payload1

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

                        if req.get("action") == "locate":
                            resp = {
                                "status": "HIT",
                                "object_size": len(cls._full_payload),
                                "chunk_size": len(payload0),
                                "chunk_layout": [
                                    {"device": cls.DEV0, "offset": 4096, "length": len(payload0)},
                                    {"device": cls.DEV1, "offset": 4096, "length": len(payload1)},
                                ],
                            }
                        elif req.get("action") == "stage_write":
                            size = req.get("size_bytes", 4096)
                            resp = {
                                "status": "WRITESTAGED",
                                "object_size": size,
                                "chunk_size": size,
                                "chunk_layout": [
                                    {"device": cls.DEV0, "offset": 8192, "length": size},
                                ],
                                "sync_status": "STAGED",
                            }
                        elif req.get("action") == "append_write":
                            size = req.get("size_bytes", 4096)
                            resp = {
                                "status": "WRITESTAGED",
                                "object_size": size * 2,
                                "chunk_size": size,
                                "chunk_layout": [
                                    {"device": cls.DEV1, "offset": 8192, "length": size},
                                ],
                                "sync_status": "STAGED",
                            }
                        elif req.get("action") == "commit_write":
                            resp = {
                                "status": "WRITECOMMITTED",
                                "object_size": 4096,
                                "sync_status": "COMMITTED",
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

    def test_fsspec_open_and_read(self):
        fs = HyperCacheFileSystem(socket_path=self.SOCKET_PATH, prefer_gds=False)
        
        with fs.open("hypercache://models/mock-model.bin", "rb") as f:
            content = f.read()
            self.assertEqual(len(content), len(self._full_payload))
            self.assertEqual(content, self._full_payload)

    def test_fsspec_seek_tell(self):
        fs = HyperCacheFileSystem(socket_path=self.SOCKET_PATH, prefer_gds=False)
        with fs.open("hypercache://models/mock-model.bin", "rb") as f:
            self.assertEqual(f.tell(), 0)
            f.seek(10)
            self.assertEqual(f.tell(), 10)
            data = f.read(20)
            self.assertEqual(data, self._full_payload[10:30])
            self.assertEqual(f.tell(), 30)

            # Seek across chunk boundary
            chunk0_len = 24 * 1024
            boundary_offset = chunk0_len - 5
            f.seek(boundary_offset)
            boundary_data = f.read(10)
            self.assertEqual(boundary_data, self._full_payload[boundary_offset : boundary_offset + 10])

    def test_fsspec_global_dispatch(self):
        # Test fsspec.open directly with hypercache:// protocol
        with fsspec.open(
            "hypercache://models/mock-model.bin",
            mode="rb",
            socket_path=self.SOCKET_PATH,
            prefer_gds=False,
        ) as f:
            header = f.read(24)
            self.assertEqual(header, b"CHUNK0_HELLO_HYPERCACHE_")

    def test_fsspec_uri_formatting(self):
        fs = HyperCacheFileSystem(socket_path=self.SOCKET_PATH, prefer_gds=False)
        self.assertEqual(fs._format_uri("models/llama.pt"), "hypercache://models/llama.pt")
        self.assertEqual(fs._format_uri("hypercache://models/llama.pt"), "hypercache://models/llama.pt")
        self.assertEqual(fs._format_uri("s3://bucket/models/llama.pt"), "s3://bucket/models/llama.pt")
        self.assertEqual(fs._format_uri("file:///mnt/nfs/models/llama.pt"), "file:///mnt/nfs/models/llama.pt")
        self.assertEqual(fs._format_uri("nfs:///mnt/nfs/models/llama.pt"), "nfs:///mnt/nfs/models/llama.pt")
        self.assertEqual(fs._format_uri("/mnt/nfs/models/llama.pt"), "file:///mnt/nfs/models/llama.pt")

    def test_fsspec_write(self):
        fs = HyperCacheFileSystem(socket_path=self.SOCKET_PATH, prefer_gds=False)
        test_write_payload = b"NEW_WRITTEN_CHECKPOINT_DATA_STREAM_" * 64
        with fs.open("hypercache://checkpoints/epoch_1.pt", "wb") as f:
            f.write(test_write_payload)

        # Verify bytes were written to DEV0 at offset 8192
        with open(self.DEV0, "rb") as f:
            f.seek(8192)
            read_back = f.read(len(test_write_payload))
            self.assertEqual(read_back, test_write_payload)

    def test_fsspec_write_through(self):
        fs = HyperCacheFileSystem(socket_path=self.SOCKET_PATH, prefer_gds=False)
        test_write_payload = b"WRITE_THROUGH_SYNC_CHECKPOINT_BYTES_" * 64
        with fs.open("hypercache://checkpoints/epoch_wt.pt", "wb", write_through=True) as f:
            f.write(test_write_payload)

        with open(self.DEV0, "rb") as f:
            f.seek(8192)
            read_back = f.read(len(test_write_payload))
            self.assertEqual(read_back, test_write_payload)

    def test_fsspec_streaming_multi_chunk_write(self):
        from hypercache.fsspec_fs import HyperCacheWriteFile
        fs = HyperCacheFileSystem(socket_path=self.SOCKET_PATH, prefer_gds=False)
        chunk0 = b"STREAM_CHUNK_0_" * 128  # 1920 bytes
        chunk1 = b"STREAM_CHUNK_1_" * 128  # 1920 bytes
        wf = HyperCacheWriteFile(
            fs=fs,
            uri="hypercache://checkpoints/streamed.pt",
            client=fs._client,
            chunk_size=len(chunk0),
        )
        wf.write(chunk0)
        # First chunk should already be flushed to DEV0 before close()
        self.assertEqual(len(wf._buffer), 0)
        self.assertEqual(wf._flushed_bytes, len(chunk0))
        wf.write(chunk1)
        wf.close()

        with open(self.DEV0, "rb") as f0, open(self.DEV1, "rb") as f1:
            f0.seek(8192)
            f1.seek(8192)
            self.assertEqual(f0.read(len(chunk0)), chunk0)
            self.assertEqual(f1.read(len(chunk1)), chunk1)


if __name__ == "__main__":
    unittest.main()
