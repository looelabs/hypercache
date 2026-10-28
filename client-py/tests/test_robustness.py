"""
Regression tests for client robustness: thread/fork safety, response framing, O_DIRECT
bounce buffers, tier-based SHM detection, and opt-in CRC32 verification.
"""

import json
import os
import socket
import sys
import tempfile
import threading
import time
import unittest
import warnings
import zlib

from hypercache.client import HyperCacheClient
from hypercache.datapath import (
    ChecksumMismatchError,
    ChunkLocation,
    DirectBlockReader,
    get_data_reader,
)


def _fork():
    # The test process is multi-threaded (daemon threads); the child only touches the
    # objects under test, so the generic fork-safety warning does not apply here.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return os.fork()


class _EchoDaemon:
    """Minimal UDS daemon that answers each request with its own id, optionally
    splitting responses across writes or batching several into one write."""

    def __init__(self, path: str, mode: str = "normal", delay: float = 0.0):
        self.path = path
        self.mode = mode
        self.delay = delay
        if os.path.exists(path):
            os.remove(path)
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.bind(path)
        self.sock.listen(64)
        self.running = True
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while self.running:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn):
        buf = b""
        with conn:
            while True:
                try:
                    data = conn.recv(65536)
                except OSError:
                    return
                if not data:
                    return
                buf += data
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    req = json.loads(line)
                    if self.delay:
                        time.sleep(self.delay)
                    resp = {
                        "status": "HIT",
                        "object_size": int(req.get("uri", "0").rsplit("/", 1)[-1] or 0),
                        "chunk_size": 4096,
                        "chunk_layout": [],
                        "generation": 7,
                    }
                    out = (json.dumps(resp) + "\n").encode()
                    try:
                        if self.mode == "split":
                            # Deliver the response one byte at a time
                            for i in range(len(out)):
                                conn.sendall(out[i : i + 1])
                        elif self.mode == "trailing_blank":
                            conn.sendall(b"\n" + out)
                        else:
                            conn.sendall(out)
                    except OSError:
                        # The client gave up (timeout test) and closed its end
                        return

    def close(self):
        self.running = False
        try:
            self.sock.close()
        except OSError:
            pass
        if os.path.exists(self.path):
            os.remove(self.path)


class TestClientConcurrencyAndFraming(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="hc_rb_", dir="/tmp")
        self.path = os.path.join(self.dir, "d.sock")

    def tearDown(self):
        for name in os.listdir(self.dir):
            try:
                os.remove(os.path.join(self.dir, name))
            except OSError:
                pass
        os.rmdir(self.dir)

    def test_threads_sharing_one_client_get_their_own_responses(self):
        daemon = _EchoDaemon(self.path)
        try:
            client = HyperCacheClient(socket_path=self.path)
            errors = []

            def worker(n):
                for i in range(50):
                    size = n * 1000 + i
                    res = client.locate(f"hypercache://x/{size}", auto_ingest=False)
                    if res.object_size != size:
                        errors.append((size, res.object_size))

            threads = [threading.Thread(target=worker, args=(n,)) for n in range(1, 9)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(errors, [])
            client.close()
        finally:
            daemon.close()

    def test_response_split_across_reads_and_blank_lines(self):
        for mode in ("split", "trailing_blank"):
            daemon = _EchoDaemon(self.path, mode=mode)
            try:
                client = HyperCacheClient(socket_path=self.path)
                for size in (1, 22, 333):
                    res = client.locate(f"hypercache://x/{size}", auto_ingest=False)
                    self.assertEqual(res.object_size, size)
                    self.assertEqual(res.generation, 7)
                client.close()
            finally:
                daemon.close()

    def test_timeout_drops_connection_and_next_request_is_clean(self):
        daemon = _EchoDaemon(self.path, delay=0.5)
        try:
            client = HyperCacheClient(socket_path=self.path, timeout=0.1)
            with self.assertRaises(OSError):
                client.locate("hypercache://x/5", auto_ingest=False)
            # The late answer to the first request must not be returned for the second one
            daemon.delay = 0.0
            res = client.locate("hypercache://x/9", auto_ingest=False)
            self.assertEqual(res.object_size, 9)
            client.close()
        finally:
            daemon.close()

    def test_ingest_calls_use_ingest_timeout(self):
        daemon = _EchoDaemon(self.path, delay=0.3)
        try:
            client = HyperCacheClient(socket_path=self.path, timeout=0.1, ingest_timeout=None)
            # Metadata-only calls honor the short timeout...
            with self.assertRaises(OSError):
                client.locate("hypercache://x/1", auto_ingest=False)
            # ...while ingesting calls wait for the daemon
            self.assertEqual(client.locate("hypercache://x/2", auto_ingest=True).object_size, 2)
            self.assertEqual(client.ingest("hypercache://x/3").object_size, 3)
            client.close()
        finally:
            daemon.close()

    def test_unreachable_daemon_raises_connection_error(self):
        client = HyperCacheClient(socket_path=os.path.join(self.dir, "missing.sock"))
        with self.assertRaises(ConnectionError):
            client.locate("hypercache://x/1", auto_ingest=False)
        self.assertFalse(client.ping())

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork()")
    def test_client_reconnects_after_fork(self):
        daemon = _EchoDaemon(self.path)
        try:
            client = HyperCacheClient(socket_path=self.path)
            self.assertEqual(client.locate("hypercache://x/4", auto_ingest=False).object_size, 4)
            parent_sock = client._sock
            r, w = os.pipe()
            pid = _fork()
            if pid == 0:  # child
                code = 1
                try:
                    os.close(r)
                    res = client.locate("hypercache://x/11", auto_ingest=False)
                    ok = res.object_size == 11 and client._sock is not parent_sock
                    os.write(w, b"1" if ok else b"0")
                    code = 0
                finally:
                    os._exit(code)
            os.close(w)
            _, status = os.waitpid(pid, 0)
            result = os.read(r, 1)
            os.close(r)
            self.assertEqual(status, 0)
            self.assertEqual(result, b"1")
            # The parent's connection is unaffected
            self.assertEqual(client.locate("hypercache://x/12", auto_ingest=False).object_size, 12)
            client.close()
        finally:
            daemon.close()


class TestDatapathReads(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="hc_dp_")
        self.dev = os.path.join(self.dir, "dev0.raw")
        self.payload = bytes((i * 7) % 251 for i in range(3 * 65536 + 777))
        with open(self.dev, "wb") as f:
            f.write(b"\0" * 4096)
            f.write(self.payload)
            f.write(b"\0" * 8192)

    def tearDown(self):
        for name in os.listdir(self.dir):
            os.remove(os.path.join(self.dir, name))
        os.rmdir(self.dir)

    def _layout(self, chunk=65536, crc=True):
        out = []
        pos = 0
        while pos < len(self.payload):
            n = min(chunk, len(self.payload) - pos)
            out.append(
                ChunkLocation(
                    device=self.dev,
                    offset=4096 + pos,
                    length=n,
                    checksum=(zlib.crc32(self.payload[pos : pos + n]) if crc else None),
                )
            )
            pos += n
        return out

    def test_unaligned_ranges_read_correctly(self):
        reader = DirectBlockReader()
        try:
            layout = self._layout()
            for start, length in [(0, len(self.payload)), (1, 4095), (65530, 20), (100, 3 * 65536), (len(self.payload) - 5, 5)]:
                got = bytes(reader.read_range(layout, start, length))
                self.assertEqual(got, self.payload[start : start + length], (start, length))
            stats = reader.get_stats()
            if sys.platform.startswith("linux") and hasattr(os, "O_DIRECT"):
                # With aligned bounce buffers O_DIRECT engages even for unaligned requests,
                # unless the filesystem (e.g. tmpfs) does not support it at all
                if stats["direct_reads"] == 0:
                    self.assertGreater(stats["direct_fallbacks"], 0)
                else:
                    self.assertEqual(stats["direct_fallbacks"], 0)
            else:
                self.assertGreater(stats["buffered_reads"], 0)
        finally:
            reader.close()

    def test_checksum_verification_is_opt_in(self):
        layout = self._layout()
        # Corrupt one byte inside chunk 1
        with open(self.dev, "r+b") as f:
            f.seek(4096 + 65536 + 10)
            f.write(b"\xff" if self.payload[65536 + 10] != 0xFF else b"\x00")

        off = DirectBlockReader(verify_checksums=False)
        try:
            off.read_range(layout, 0, len(self.payload))
        finally:
            off.close()

        on = DirectBlockReader(verify_checksums=True)
        try:
            # Chunk 0 alone is intact
            self.assertEqual(bytes(on.read_range(layout, 0, 65536)), self.payload[:65536])
            with self.assertRaises(ChecksumMismatchError):
                on.read_range(layout, 0, len(self.payload))
            # Partial reads cannot be verified and are served as-is
            on.read_range(layout, 65536 + 5, 100)
        finally:
            on.close()

        os.environ["HYPERCACHE_VERIFY_CHECKSUMS"] = "1"
        try:
            env_reader = DirectBlockReader()
            self.assertTrue(env_reader.verify_checksums)
            env_reader.close()
        finally:
            del os.environ["HYPERCACHE_VERIFY_CHECKSUMS"]

    def test_ram_tier_detected_by_tier_not_path(self):
        shm_file = os.path.join(self.dir, "chunk_0_abc")  # no "/shm" in the name
        with open(shm_file, "wb") as f:
            f.write(self.payload[:5000])
        reader = DirectBlockReader()
        try:
            loc = ChunkLocation(device=shm_file, offset=0, length=5000, tier="ram", checksum=zlib.crc32(self.payload[:5000]))
            self.assertTrue(loc.is_shm)
            self.assertEqual(bytes(reader.read_range([loc], 10, 100)), self.payload[10:110])
            self.assertEqual(reader.get_stats()["shm_reads"], 1)
        finally:
            reader.close()

    def test_chunk_location_round_trip_and_legacy(self):
        legacy = ChunkLocation.from_dict({"device": "/dev/x", "offset": 4096, "length": 10})
        self.assertIsNone(legacy.checksum)
        self.assertNotIn("checksum", legacy.to_dict())
        new = ChunkLocation.from_dict({"device": "/dev/x", "offset": 4096, "length": 10, "tier": "RAM", "checksum": 123})
        self.assertEqual(new.tier, "ram")
        self.assertEqual(new.to_dict()["checksum"], 123)

    @unittest.skipUnless(hasattr(os, "fork"), "requires fork()")
    def test_shared_reader_is_recreated_after_fork(self):
        parent_reader = get_data_reader(prefer_gds=False)
        r, w = os.pipe()
        pid = _fork()
        if pid == 0:
            code = 1
            try:
                os.close(r)
                child_reader = get_data_reader(prefer_gds=False)
                data = bytes(child_reader.read_range(self._layout(), 0, 100))
                ok = child_reader is not parent_reader and data == self.payload[:100]
                os.write(w, b"1" if ok else b"0")
                code = 0
            finally:
                os._exit(code)
        os.close(w)
        _, status = os.waitpid(pid, 0)
        result = os.read(r, 1)
        os.close(r)
        self.assertEqual(status, 0)
        self.assertEqual(result, b"1")
        self.assertIs(get_data_reader(prefer_gds=False), parent_reader)


if __name__ == "__main__":
    unittest.main()
