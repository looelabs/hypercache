#!/usr/bin/env python3
"""
HyperCache end-to-end verification driver.

Runs the real `hypercache` / `hypercache-admin` binaries against file-backed mock block
devices in an isolated temp directory (own socket, lock, manifest and NFS root), and checks
behavior through the CLI, the raw UDS protocol and the Python client datapath.

Scenarios:
  daemon  single-node daemon: ingest, integrity, concurrency, confinement, write paths,
          eviction pressure, restart / crash recovery, purge, bench, zero network exposure
  fleet   controller + two agents: auth, targeting, preload/evict/purge, offline handling
  s3      daemon against a live S3 endpoint (MinIO): parallel GET, multipart write-through,
          write-back, URI encoding, no synthetic fallback (needs --s3-endpoint and boto3)

Usage:
  scripts/e2e/hypercache_e2e.py --bin-dir control-plane/target/release daemon fleet
  scripts/e2e/hypercache_e2e.py --bin-dir ... s3 --s3-endpoint http://127.0.0.1:9000

Exit status is non-zero if any check fails.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import urllib.error
import urllib.request
from typing import Any, Callable, Dict, List, Optional

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "client-py"))

from hypercache.client import HyperCacheClient  # noqa: E402
from hypercache.datapath import DirectBlockReader, DirectBlockWriter  # noqa: E402

MIB = 1024 * 1024


# ----------------------------------------------------------------------------------------------
# Reporting
# ----------------------------------------------------------------------------------------------


class Report:
    def __init__(self) -> None:
        self.results: List[tuple] = []

    def check(self, name: str, fn: Callable[[], Any]) -> bool:
        t0 = time.time()
        try:
            fn()
        except Exception as e:  # noqa: BLE001 - every failure is reported, not raised
            dt = time.time() - t0
            self.results.append((name, False, dt, f"{type(e).__name__}: {e}"))
            print(f"  [FAIL] {name} ({dt:.2f}s)\n         {type(e).__name__}: {e}", flush=True)
            if os.environ.get("HYPERCACHE_E2E_TRACE"):
                traceback.print_exc()
            return False
        dt = time.time() - t0
        self.results.append((name, True, dt, ""))
        print(f"  [PASS] {name} ({dt:.2f}s)", flush=True)
        return True

    def summary(self) -> int:
        failed = [r for r in self.results if not r[1]]
        print("-" * 100)
        print(f"E2E summary: {len(self.results) - len(failed)} passed, {len(failed)} failed")
        for name, _, _, err in failed:
            print(f"  FAILED: {name}: {err}")
        return 1 if failed else 0


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def wait_until(pred: Callable[[], bool], timeout: float, what: str, interval: float = 0.1) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if pred():
                return
        except Exception:  # noqa: BLE001 - keep polling until the deadline
            pass
        time.sleep(interval)
    raise TimeoutError(f"timed out after {timeout:.0f}s waiting for {what}")


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def port_is_listening(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(("127.0.0.1", port)) == 0


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def random_bytes(n: int) -> bytes:
    return os.urandom(n)


# ----------------------------------------------------------------------------------------------
# Process management
# ----------------------------------------------------------------------------------------------


def toml_value(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    return json.dumps(str(v))


def write_toml(path: str, values: Dict[str, Any]) -> None:
    lines = []
    tables = {}
    for k, v in values.items():
        if isinstance(v, dict):
            tables[k] = v
        else:
            lines.append(f"{k} = {toml_value(v)}")
    for name, table in tables.items():
        lines.append(f"\n[{name}]")
        for k, v in table.items():
            lines.append(f"{k} = {toml_value(v)}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def clean_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ)
    # Never let a developer's real cluster token, controller or S3 credentials leak into tests
    for k in ("HYPERCACHE_AUTH_TOKEN", "HYPERCACHE_CONTROLLER_URL", "HYPERCACHE_DEVICES", "AWS_SESSION_TOKEN"):
        env.pop(k, None)
    env.setdefault("RUST_LOG", "info")
    env.update(extra or {})
    return env


class Node:
    """One isolated `hypercache start` daemon with its own devices, socket and manifest."""

    def __init__(self, ctx: "Ctx", name: str, device_mb: int = 64, devices: int = 4, overrides=None):
        self.ctx = ctx
        self.name = name
        self.dir = os.path.join(ctx.workdir, name)
        os.makedirs(os.path.join(self.dir, "state"), exist_ok=True)
        os.makedirs(os.path.join(self.dir, "run"), exist_ok=True)
        self.socket_path = os.path.join(self.dir, "run", "hypercache.sock")
        self.devices = []
        for i in range(devices):
            p = os.path.join(self.dir, f"nvme{i}.raw")
            if not os.path.exists(p):
                with open(p, "wb") as f:
                    f.truncate(device_mb * MIB)
            self.devices.append(p)
        self.config_path = os.path.join(self.dir, "agent.toml")
        self.log_path = os.path.join(self.dir, "daemon.log")
        self.config: Dict[str, Any] = {
            "device_discovery_mode": "manual",
            "block_devices": self.devices,
            "auto_create_mock_devices": False,
            "chunk_size_bytes": MIB,
            "device_start_offset_bytes": 8192,
            "metadata_persistence_enabled": True,
            "metadata_persistence_path": os.path.join(self.dir, "state", "metadata.json"),
            "authoritative_backend": "nfs",
            "nfs_mount_path": ctx.nfs_root,
            "metrics_enabled": False,
            "agent_enabled": False,
            "node_id": name,
        }
        self.config.update(overrides or {})
        self.proc: Optional[subprocess.Popen] = None

    @property
    def env(self) -> Dict[str, str]:
        return clean_env({"HYPERCACHE_SOCKET_PATH": self.socket_path})

    def start(self, timeout: float = 30.0) -> None:
        write_toml(self.config_path, self.config)
        log = open(self.log_path, "ab")
        self.proc = subprocess.Popen(
            [self.ctx.bin("hypercache"), "-c", self.config_path, "start"],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=self.env,
        )
        self.ctx.track(self)
        try:
            wait_until(lambda: self.proc.poll() is None and self.client().ping(), timeout, f"daemon {self.name}")
        except TimeoutError:
            raise RuntimeError(f"daemon {self.name} did not start; log tail:\n{self.log_tail()}")

    def stop(self, sig: int = signal.SIGINT, timeout: float = 20.0) -> int:
        if self.proc is None or self.proc.poll() is not None:
            return self.proc.returncode if self.proc else 0
        self.proc.send_signal(sig)
        try:
            return self.proc.wait(timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            return self.proc.wait(5)

    def client(self, **kw) -> HyperCacheClient:
        return HyperCacheClient(socket_path=self.socket_path, **kw)

    def cli(self, *args: str, timeout: float = 300.0, check: bool = True) -> subprocess.CompletedProcess:
        res = subprocess.run(
            [self.ctx.bin("hypercache"), "-c", self.config_path, *args],
            capture_output=True,
            text=True,
            env=self.env,
            timeout=timeout,
        )
        if check and res.returncode != 0:
            raise AssertionError(
                f"`hypercache {' '.join(args)}` exited {res.returncode}\nstdout:\n{res.stdout[-2000:]}\nstderr:\n{res.stderr[-2000:]}"
            )
        return res

    def log_tail(self, n: int = 40) -> str:
        try:
            with open(self.log_path, "rb") as f:
                return b"\n".join(f.read().splitlines()[-n:]).decode("utf-8", "replace")
        except OSError:
            return "(no log)"

    def read_object(self, uri: str, verify: bool = True) -> bytes:
        res = self.client().locate(uri, auto_ingest=False)
        expect(res.is_hit, f"{uri} not cached on {self.name}: {res.status} {res.error_message}")
        reader = DirectBlockReader(verify_checksums=verify)
        try:
            return bytes(reader.read_range(res.chunk_layout, 0, res.object_size))
        finally:
            reader.close()


class Ctx:
    def __init__(self, bin_dir: str, keep: bool) -> None:
        self.bin_dir = os.path.abspath(bin_dir)
        self.keep = keep
        self.workdir = tempfile.mkdtemp(prefix="hc-e2e-")
        self.nfs_root = os.path.join(self.workdir, "nfs")
        os.makedirs(self.nfs_root)
        self.procs: List[Any] = []

    def bin(self, name: str) -> str:
        p = os.path.join(self.bin_dir, name)
        if not os.access(p, os.X_OK):
            raise FileNotFoundError(f"binary not found: {p} (build first)")
        return p

    def track(self, proc_owner: Any) -> None:
        self.procs.append(proc_owner)

    def nfs_file(self, rel: str, data: bytes) -> str:
        path = os.path.join(self.nfs_root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return "file://" + path

    def cleanup(self) -> None:
        for owner in self.procs:
            proc = getattr(owner, "proc", None)
            if proc is not None and proc.poll() is None:
                proc.kill()
                try:
                    proc.wait(5)
                except subprocess.TimeoutExpired:
                    pass
        if self.keep:
            print(f"Keeping E2E work directory: {self.workdir}")
        else:
            subprocess.run(["rm", "-rf", self.workdir], check=False)


def send_raw(socket_path: str, payload: bytes, timeout: float = 30.0) -> bytes:
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        s.connect(socket_path)
        try:
            s.sendall(payload)
        except (BrokenPipeError, ConnectionResetError):
            pass
        buf = b""
        while b"\n" not in buf:
            chunk = s.recv(65536)
            if not chunk:
                break
            buf += chunk
        return buf


def write_object(client: HyperCacheClient, uri: str, data: bytes, write_through: bool, sha: Optional[str]) -> bool:
    staged = client.stage_write(uri, len(data))
    expect(staged.is_staged, f"stage_write failed: {staged.error_message}")
    writer = DirectBlockWriter()
    try:
        writer.write_chunks(staged.chunk_layout, data)
    finally:
        writer.close()
    return client.commit_write(uri, sha256=sha, write_through=write_through)


# ----------------------------------------------------------------------------------------------
# Scenario: single-node daemon
# ----------------------------------------------------------------------------------------------


def scenario_daemon(ctx: Ctx, rep: Report) -> None:
    print("== Scenario: single-node daemon")
    node = Node(ctx, "solo", device_mb=64, devices=4, overrides={"reclaim_grace_secs": 30})
    rep.check("daemon starts on file-backed devices with an isolated socket", node.start)
    if node.proc is None or node.proc.poll() is not None:
        print(node.log_tail())
        return

    rep.check("version flag reports the package version", lambda: expect(
        "0.9.1" in subprocess.run([ctx.bin("hypercache"), "--version"], capture_output=True, text=True).stdout,
        "hypercache --version does not report 0.9.1"))
    rep.check("status command talks to the daemon", lambda: node.cli("status"))

    def zero_exposure():
        # AGENTS.md rule 2: no TCP listeners unless metrics/agent are enabled
        for port in (9110, 9115):
            if port_is_listening(port):
                # Another process may own the port; only fail if it is our daemon
                lsof = subprocess.run(["lsof", "-nP", "-a", "-p", str(node.proc.pid), "-iTCP", "-sTCP:LISTEN"],
                                      capture_output=True, text=True)
                expect(lsof.returncode != 0 or not lsof.stdout.strip(), f"daemon listens on TCP: {lsof.stdout}")
        lsof = subprocess.run(["lsof", "-nP", "-a", "-p", str(node.proc.pid), "-iTCP", "-sTCP:LISTEN"],
                              capture_output=True, text=True)
        if lsof.returncode == 0:
            expect(not lsof.stdout.strip(), f"daemon has TCP listeners by default:\n{lsof.stdout}")

    rep.check("default daemon opens zero TCP ports", zero_exposure)

    data_a = random_bytes(9 * MIB + 12345)
    uri_a = ctx.nfs_file("models/a.bin", data_a)

    def preload_and_verify():
        node.cli("preload", uri_a)
        res = node.client().locate(uri_a, auto_ingest=False)
        expect(res.is_hit, f"not cached after preload: {res.status}")
        expect(res.object_size == len(data_a), "size mismatch")
        expect(res.generation is not None, "locate did not return a layout generation")
        expect(len(res.chunk_layout) == 10, f"expected 10 x 1 MiB chunks, got {len(res.chunk_layout)}")
        expect(all(c.checksum is not None for c in res.chunk_layout), "chunks are missing CRC32 checksums")
        expect(len({c.device for c in res.chunk_layout}) == 4, "chunks were not striped across all 4 devices")
        expect(min(c.offset for c in res.chunk_layout) >= 8192, "chunk placed below device_start_offset_bytes")
        expect(sha256(node.read_object(uri_a)) == sha256(data_a), "read-back data differs from source")

    rep.check("preload via CLI, then read back through the O_DIRECT datapath with CRC verification", preload_and_verify)

    def ls_json():
        out = json.loads(node.cli("ls", "--json").stdout)
        text = json.dumps(out)
        expect(uri_a in text, f"ls --json does not list {uri_a}")

    rep.check("ls --json lists the cached object", ls_json)

    data_b = random_bytes(24 * MIB + 1)
    uri_b = ctx.nfs_file("models/b.bin", data_b)

    def single_flight():
        results: List[Any] = [None] * 8

        def worker(i: int) -> None:
            results[i] = node.client().locate(uri_b, auto_ingest=True)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        expect(all(r is not None and r.is_hit for r in results), f"some concurrent locates failed: {[getattr(r, 'error_message', None) for r in results]}")
        gens = {r.generation for r in results}
        expect(len(gens) == 1, f"8 concurrent locates produced {len(gens)} separate ingests (generations {gens})")
        layouts = {tuple((c.device, c.offset) for c in r.chunk_layout) for r in results}
        expect(len(layouts) == 1, "concurrent locates returned different layouts")
        expect(sha256(node.read_object(uri_b)) == sha256(data_b), "data mismatch after concurrent ingest")

    rep.check("8 concurrent locate(auto_ingest) calls share one ingest", single_flight)

    def confinement():
        c = node.client()
        for bad in ("file:///etc/passwd", f"file://{ctx.nfs_root}/../../../etc/passwd", "/etc/hosts"):
            res = c.locate(bad, auto_ingest=True)
            expect(not res.is_hit, f"daemon ingested a file outside the NFS root: {bad}")
        expect(node.cli("preload", "file:///etc/passwd", check=False).returncode != 0,
               "CLI preload of a forbidden path exited 0")

    rep.check("file:// URIs outside nfs_mount_path are rejected", confinement)

    def no_synthetic():
        res = node.client().locate("hypercache://models/benchmark/fake_8mb.safetensors", auto_ingest=True)
        expect(not res.is_hit, "a missing benchmark-looking URI was served with synthetic data")

    rep.check("missing objects are never replaced with synthetic data", no_synthetic)

    def stale_layout_survives_grace():
        c = node.client()
        old = c.locate(uri_a, auto_ingest=False)
        node.cli("rm", uri_a)
        expect(not c.locate(uri_a, auto_ingest=False).is_hit, "object still cached after rm")
        # New ingests must not reuse the evicted extents while the grace period runs
        data_c = random_bytes(9 * MIB + 12345)
        uri_c = ctx.nfs_file("models/c.bin", data_c)
        node.cli("preload", uri_c)
        reader = DirectBlockReader(verify_checksums=True)
        try:
            stale = bytes(reader.read_range(old.chunk_layout, 0, old.object_size))
        finally:
            reader.close()
        expect(stale == data_a, "a reader holding an evicted layout read another object's bytes")
        expect(sha256(node.read_object(uri_c)) == sha256(data_c), "new object corrupt")

    rep.check("evicted extents are quarantined while old layouts may still be read", stale_layout_survives_grace)

    def write_through():
        payload = random_bytes(3 * MIB + 77)
        uri = f"file://{ctx.nfs_root}/ckpt/wt.pt"
        expect(write_object(node.client(), uri, payload, True, sha256(payload)), "write-through commit failed")
        with open(os.path.join(ctx.nfs_root, "ckpt", "wt.pt"), "rb") as f:
            expect(f.read() == payload, "authoritative file differs after write-through")
        expect(node.client().get_write_status(uri)["status"] == "SYNCED_BACKEND", "status is not SYNCED_BACKEND")

    rep.check("stage -> direct write -> commit(write_through, sha256) lands on the backend", write_through)

    def write_back():
        payload = random_bytes(2 * MIB + 5)
        uri = f"file://{ctx.nfs_root}/ckpt/wb.pt"
        expect(write_object(node.client(), uri, payload, False, sha256(payload)), "write-back commit failed")
        wait_until(lambda: node.client().get_write_status(uri)["status"] == "SYNCED_BACKEND", 30, "background sync")
        with open(os.path.join(ctx.nfs_root, "ckpt", "wb.pt"), "rb") as f:
            expect(f.read() == payload, "authoritative file differs after write-back")

    rep.check("asynchronous write-back reaches SYNCED_BACKEND", write_back)

    def sha_mismatch():
        payload = random_bytes(MIB)
        uri = f"file://{ctx.nfs_root}/ckpt/bad.pt"
        expect(not write_object(node.client(), uri, payload, False, "0" * 64), "commit with a wrong sha256 succeeded")
        expect(node.client().get_write_status(uri)["status"] == "STAGED", "rejected commit changed the sync status")
        expect(not os.path.exists(os.path.join(ctx.nfs_root, "ckpt", "bad.pt")), "rejected object was flushed")

    rep.check("commit with a mismatching sha256 is rejected and never flushed", sha_mismatch)

    def failed_flush_stays_dirty():
        payload = random_bytes(MIB)
        uri = "file:///hypercache-e2e-forbidden/x.pt"
        expect(not write_object(node.client(), uri, payload, True, None), "write-through to a forbidden path succeeded")
        expect(node.client().get_write_status(uri)["status"] == "SYNC_FAILED", "failed flush not marked SYNC_FAILED")

    rep.check("failed write-through is reported and marked SYNC_FAILED", failed_flush_stays_dirty)

    def oversized_request():
        resp = send_raw(node.socket_path, b"x" * (17 * MIB))
        expect(b"limit" in resp, f"oversized request not rejected cleanly: {resp[:200]!r}")
        expect(node.client().ping(), "daemon unhealthy after oversized request")
        resp = send_raw(node.socket_path, b'{"action":"ping"}\n{"action":"ping"}\n')
        expect(b"PONG" in resp, "pipelined requests failed")

    rep.check("oversized IPC requests are rejected without hurting the daemon", oversized_request)

    def pressure_eviction():
        # 256 MiB of devices: ingesting 6 x 60 MiB must evict older clean objects, not fail
        last = None
        for i in range(6):
            data = random_bytes(60 * MIB)
            uri = ctx.nfs_file(f"pressure/p{i}.bin", data)
            node.cli("preload", uri)
            last = (uri, data)
        expect(sha256(node.read_object(last[0])) == sha256(last[1]), "newest object corrupt after pressure eviction")
        # Pending write-backs and their data must survive the pressure
        with open(os.path.join(ctx.nfs_root, "ckpt", "wb.pt"), "rb") as f:
            expect(len(f.read()) > 0, "write-back file missing")

    rep.check("allocation under capacity pressure evicts clean objects instead of failing", pressure_eviction)

    survivors: Dict[str, bytes] = {}

    def graceful_restart():
        data = random_bytes(5 * MIB + 3)
        uri = ctx.nfs_file("models/restart.bin", data)
        node.cli("preload", uri)
        gen = node.client().locate(uri, auto_ingest=False).generation
        survivors[uri] = data
        node.stop(signal.SIGINT)
        node.start()
        res = node.client().locate(uri, auto_ingest=False)
        expect(res.is_hit, "object lost across a graceful restart")
        expect(res.generation == gen, f"generation changed across restart ({gen} -> {res.generation})")
        expect(sha256(node.read_object(uri)) == sha256(data), "data corrupt after warm boot")

    rep.check("graceful restart restores the catalog from the manifest (warm boot)", graceful_restart)

    def crash_restart():
        data = random_bytes(4 * MIB + 9)
        uri = ctx.nfs_file("models/crash.bin", data)
        node.cli("preload", uri)
        time.sleep(1.0)  # manifest writes are debounced by milliseconds
        node.stop(signal.SIGKILL)
        node.start()
        res = node.client().locate(uri, auto_ingest=False)
        expect(res.is_hit, "object lost after SIGKILL")
        expect(sha256(node.read_object(uri, verify=True)) == sha256(data), "data corrupt after crash recovery")
        for u, d in survivors.items():
            expect(sha256(node.read_object(u)) == sha256(d), f"{u} corrupt after crash recovery")

    rep.check("SIGKILL crash: committed objects survive and verify", crash_restart)

    def purge_resets_to_configured_offset():
        node.cli("purge")
        out = json.loads(node.cli("ls", "--json").stdout)
        expect("models/" not in json.dumps(out.get("objects", out)), f"objects remain after purge: {out}")
        data = random_bytes(4 * MIB)
        uri = ctx.nfs_file("models/after_purge.bin", data)
        node.cli("preload", uri)
        res = node.client().locate(uri, auto_ingest=False)
        expect(min(c.offset for c in res.chunk_layout) == 8192,
               f"after purge allocation restarted at {min(c.offset for c in res.chunk_layout)}, expected 8192")
        expect(sha256(node.read_object(uri)) == sha256(data), "data corrupt after purge")

    rep.check("purge clears the catalog and restarts allocation at device_start_offset_bytes", purge_resets_to_configured_offset)

    rep.check("bench runs end-to-end with explicit synthetic opt-in", lambda: node.cli(
        "bench", "--size-mb", "16", "--iterations", "1", timeout=300))

    def check_command():
        res = node.cli("check", "--json", check=False)
        json.loads(res.stdout[res.stdout.index("{"):])

    rep.check("check --json produces a parseable report", check_command)

    rc = node.stop(signal.SIGINT)
    rep.check("daemon shuts down cleanly on SIGINT", lambda: expect(rc in (0, -2, 130), f"exit code {rc}"))


# ----------------------------------------------------------------------------------------------
# Scenario: controller + agents
# ----------------------------------------------------------------------------------------------


class Controller:
    def __init__(self, ctx: Ctx, token: str, bypass: bool) -> None:
        self.ctx = ctx
        self.port = free_port()
        self.url = f"http://127.0.0.1:{self.port}"
        self.token = token
        self.dir = os.path.join(ctx.workdir, "controller")
        os.makedirs(self.dir, exist_ok=True)
        self.config_path = os.path.join(self.dir, "controller.toml")
        write_toml(self.config_path, {
            "controller_bind_address": "127.0.0.1",
            "controller_port": self.port,
            "cluster_auth_token": token,
            "allow_local_admin_bypass": bypass,
            "authoritative_backend": "nfs",
            "nfs_mount_path": ctx.nfs_root,
        })
        self.log_path = os.path.join(self.dir, "controller.log")
        self.proc: Optional[subprocess.Popen] = None

    def start(self) -> None:
        self.proc = subprocess.Popen(
            [self.ctx.bin("hypercache-admin"), "-c", self.config_path, "controller"],
            stdout=open(self.log_path, "ab"), stderr=subprocess.STDOUT, env=clean_env(),
        )
        self.ctx.track(self)
        wait_until(lambda: port_is_listening(self.port), 20, "controller port")

    def admin(self, *args: str, token: Optional[str] = "default", check: bool = True, timeout: float = 300):
        tok = self.token if token == "default" else token
        cmd = [self.ctx.bin("hypercache-admin"), "--controller", self.url]
        if tok:
            cmd += ["--token", tok]
        res = subprocess.run(cmd + list(args), capture_output=True, text=True, env=clean_env(), timeout=timeout)
        if check and res.returncode != 0:
            raise AssertionError(f"`hypercache-admin {' '.join(args)}` exited {res.returncode}\n{res.stdout[-2000:]}\n{res.stderr[-2000:]}")
        return res

    def nodes(self) -> Dict[str, Any]:
        return json.loads(self.admin("nodes", "--json").stdout)

    def node_status(self) -> Dict[str, str]:
        return {n["registration"]["node_id"]: n["status"] for n in self.nodes()["nodes"]}


def http(method: str, url: str, token: Optional[str] = None, body: Optional[dict] = None, timeout: float = 30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def scenario_fleet(ctx: Ctx, rep: Report) -> None:
    print("== Scenario: controller + agents")
    token = "e2e-" + os.urandom(12).hex()
    ctrl = Controller(ctx, token, bypass=False)
    if not rep.check("controller starts", ctrl.start):
        return

    agents = []
    for i in (1, 2):
        port = free_port()
        node = Node(ctx, f"e2e-node-{i}", device_mb=64, devices=2, overrides={
            "controller_url": ctrl.url,
            "cluster_auth_token": token,
            "agent_enabled": True,
            "agent_bind_address": "127.0.0.1",
            "agent_http_port": port,
            "node_labels": {"gpu": "h100", "idx": str(i)},
        })
        node.agent_port = port
        agents.append(node)
        rep.check(f"agent {node.name} starts", node.start)
    n1, n2 = agents

    rep.check("both agents register and report ONLINE", lambda: wait_until(
        lambda: ctrl.node_status() == {"e2e-node-1": "ONLINE", "e2e-node-2": "ONLINE"}, 30, "two online nodes"))

    def auth_required():
        res = ctrl.admin("nodes", token=None, check=False)
        expect(res.returncode != 0, "controller served nodes without a token while bypass is disabled")
        expect("401" in res.stderr + res.stdout, "missing token did not produce a 401 error")
        res = ctrl.admin("nodes", token="wrong-token", check=False)
        expect(res.returncode != 0, "controller accepted a wrong token")
        status, _ = http("GET", f"http://127.0.0.1:{n1.agent_port}/api/v1/agent/status")
        expect(status == 401, f"agent API without token returned {status}")
        status, _ = http("GET", f"http://127.0.0.1:{n1.agent_port}/api/v1/agent/status", token=token)
        expect(status == 200, f"agent API with token returned {status}")
        status, _ = http("POST", f"http://127.0.0.1:{n1.agent_port}/api/v1/agent/purge", token="nope", body={})
        expect(status == 401, f"agent purge with a wrong token returned {status}")

    rep.check("controller and agent APIs enforce the bearer token", auth_required)

    def rogue_registration():
        reg = {"node_id": "e2e-node-1", "address": "10.255.255.1", "http_port": 1, "labels": {}, "drives": [],
               "total_capacity_bytes": 0, "used_capacity_bytes": 0, "cached_objects_count": 0, "cached_uris": []}
        status, _ = http("POST", f"{ctrl.url}/api/v1/register", body=reg)
        expect(status == 401, f"unauthenticated registration returned {status}")

    rep.check("registration without a token is rejected", rogue_registration)

    data = random_bytes(6 * MIB + 1)
    uri = ctx.nfs_file("fleet/model.bin", data)

    def fleet_preload_all():
        ctrl.admin("preload", uri, "--nodes", "all")
        for n in agents:
            expect(sha256(n.read_object(uri)) == sha256(data), f"{n.name} data mismatch after fleet preload")

    rep.check("fleet preload --nodes all caches the object on every agent", fleet_preload_all)

    def selector_typo_rejected():
        res = ctrl.admin("purge", "--selector", "gpu", check=False)
        expect(res.returncode != 0, "malformed selector was accepted")
        for n in agents:
            expect(n.client().locate(uri, auto_ingest=False).is_hit, f"malformed selector purged {n.name}")

    rep.check("malformed --selector is rejected and purges nothing", selector_typo_rejected)

    def targeted_ops():
        data2 = random_bytes(2 * MIB)
        uri2 = ctx.nfs_file("fleet/only2.bin", data2)
        ctrl.admin("preload", uri2, "--nodes", "e2e-node-2")
        expect(not n1.client().locate(uri2, auto_ingest=False).is_hit, "--nodes e2e-node-2 also hit node 1")
        expect(n2.client().locate(uri2, auto_ingest=False).is_hit, "--nodes e2e-node-2 missed node 2")
        uri3 = ctx.nfs_file("fleet/sel1.bin", random_bytes(MIB))
        ctrl.admin("preload", uri3, "--selector", "idx=1")
        expect(n1.client().locate(uri3, auto_ingest=False).is_hit, "selector idx=1 missed node 1")
        expect(not n2.client().locate(uri3, auto_ingest=False).is_hit, "selector idx=1 hit node 2")
        uri4 = ctx.nfs_file("fleet/range.bin", random_bytes(MIB))
        ctrl.admin("preload", uri4, "--nodes", "e2e-node-1..e2e-node-2")
        expect(all(n.client().locate(uri4, auto_ingest=False).is_hit for n in agents), "numeric node range missed a node")
        ctrl.admin("evict", uri, "--nodes", "all")
        expect(not any(n.client().locate(uri, auto_ingest=False).is_hit for n in agents), "fleet evict left a copy")
        ctrl.admin("purge", "--nodes", "e2e-node-1")
        expect(not n1.client().locate(uri4, auto_ingest=False).is_hit, "targeted purge did not clear node 1")
        expect(n2.client().locate(uri4, auto_ingest=False).is_hit, "targeted purge of node 1 cleared node 2")
        res = ctrl.admin("preload", "file:///etc/passwd", "--nodes", "all", check=False)
        expect(res.returncode != 0, "fleet preload of a forbidden path exited 0")

    rep.check("node lists, label selectors, numeric ranges, evict and targeted purge", targeted_ops)

    def unsafe_agent_refuses_to_bind():
        port = free_port()
        rogue = Node(ctx, "rogue", device_mb=16, devices=1, overrides={
            "agent_enabled": True, "agent_bind_address": "0.0.0.0", "agent_http_port": port,
        })
        rogue.start()
        time.sleep(1.0)
        expect(not port_is_listening(port), "agent API bound 0.0.0.0 without a cluster token")
        expect(rogue.client().ping(), "daemon should keep serving the local socket")
        rogue.stop()

    rep.check("agent API refuses a non-loopback bind without a token", unsafe_agent_refuses_to_bind)

    def offline_node_skipped():
        n2.stop(signal.SIGKILL)
        wait_until(lambda: ctrl.node_status().get("e2e-node-2") == "OFFLINE", 40, "node 2 OFFLINE", interval=1.0)
        t0 = time.time()
        uri5 = ctx.nfs_file("fleet/after_offline.bin", random_bytes(MIB))
        res = ctrl.admin("preload", uri5, "--nodes", "all")
        expect(time.time() - t0 < 15, "fleet preload waited on an offline node")
        expect("e2e-node-2" not in res.stdout, f"offline node was targeted:\n{res.stdout}")
        expect(n1.client().locate(uri5, auto_ingest=False).is_hit, "online node missed the preload")

    rep.check("offline nodes are detected and excluded from dispatch", offline_node_skipped)

    def bypass_on_loopback():
        ctrl2 = Controller(ctx, token, bypass=True)
        ctrl2.dir = os.path.join(ctx.workdir, "controller2")
        ctrl2.start()
        res = ctrl2.admin("nodes", "--json", token=None)
        json.loads(res.stdout)
        ctrl2.proc.terminate()

    rep.check("zero-config loopback admin works when allow_local_admin_bypass = true", bypass_on_loopback)

    n1.stop()
    ctrl.proc.terminate()


# ----------------------------------------------------------------------------------------------
# Scenario: live S3 (MinIO)
# ----------------------------------------------------------------------------------------------


def scenario_s3(ctx: Ctx, rep: Report, endpoint: str, access: str, secret: str, bucket: str) -> None:
    print(f"== Scenario: live S3 at {endpoint}")
    try:
        import boto3
        from botocore.config import Config as BotoConfig
    except ImportError:
        rep.check("boto3 is installed (needed for the S3 scenario)", lambda: (_ for _ in ()).throw(ImportError("pip install boto3")))
        return

    s3 = boto3.client("s3", endpoint_url=endpoint, aws_access_key_id=access, aws_secret_access_key=secret,
                      region_name="us-east-1", config=BotoConfig(s3={"addressing_style": "path"}))

    def ensure_bucket():
        try:
            s3.head_bucket(Bucket=bucket)
        except Exception:  # noqa: BLE001
            s3.create_bucket(Bucket=bucket)

    if not rep.check("bucket exists", ensure_bucket):
        return

    node = Node(ctx, "s3node", device_mb=128, devices=2, overrides={
        "authoritative_backend": "s3",
        "s3_endpoint": endpoint,
        "s3_region": "us-east-1",
        "s3_bucket": bucket,
        "s3_access_key": access,
        "s3_secret_key": secret,
        "s3_parallel_streams": 8,
        "s3_multipart_threshold_bytes": 16 * MIB,
        "chunk_size_bytes": 4 * MIB,
        "allow_synthetic_s3_fallback": False,
    })
    if not rep.check("daemon starts with the S3 backend", node.start):
        return

    run = os.urandom(4).hex()
    big = random_bytes(70 * MIB + 333)
    key = f"e2e-{run}/models/big.bin"

    def parallel_ingest():
        s3.put_object(Bucket=bucket, Key=key, Body=big)
        uri = f"s3://{bucket}/{key}"
        node.cli("preload", uri)
        expect(sha256(node.read_object(uri)) == sha256(big), "S3-ingested object differs from the source")

    rep.check("70 MiB object ingests via parallel ranged GETs and verifies", parallel_ingest)

    def special_key():
        k = f"e2e-{run}/odd dir/name with space+plus=eq.bin"
        body = random_bytes(5 * MIB)
        s3.put_object(Bucket=bucket, Key=k, Body=body)
        uri = f"s3://{bucket}/{k}"
        node.cli("preload", uri)
        expect(sha256(node.read_object(uri)) == sha256(body), "object with an encoded key differs")

    rep.check("keys needing URI encoding are fetched (SigV4 canonical path)", special_key)

    def missing_key():
        res = node.client().locate(f"s3://{bucket}/e2e-{run}/models/benchmark/missing_8mb.bin", auto_ingest=True)
        expect(not res.is_hit, "missing S3 key was served (synthetic fallback leaked)")

    rep.check("missing S3 objects fail instead of returning synthetic data", missing_key)

    def multipart_write_through():
        payload = random_bytes(40 * MIB + 17)
        k = f"e2e-{run}/ckpt/epoch1.pt"
        uri = f"s3://{bucket}/{k}"
        expect(write_object(node.client(), uri, payload, True, sha256(payload)), "write-through commit failed")
        obj = s3.get_object(Bucket=bucket, Key=k)
        expect(obj["Body"].read() == payload, "S3 object differs after write-through")
        etag = s3.head_object(Bucket=bucket, Key=k)["ETag"].strip('"')
        expect("-" in etag, f"object above the multipart threshold was not uploaded with multipart (ETag {etag})")

    rep.check("write-through above the threshold uses multipart upload", multipart_write_through)

    def write_back_s3():
        payload = random_bytes(3 * MIB + 1)
        k = f"e2e-{run}/ckpt/small.pt"
        uri = f"s3://{bucket}/{k}"
        expect(write_object(node.client(), uri, payload, False, sha256(payload)), "write-back commit failed")
        wait_until(lambda: node.client().get_write_status(uri)["status"] == "SYNCED_BACKEND", 60, "S3 write-back")
        expect(s3.get_object(Bucket=bucket, Key=k)["Body"].read() == payload, "S3 object differs after write-back")

    rep.check("write-back to S3 reaches SYNCED_BACKEND", write_back_s3)

    node.stop()


# ----------------------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scenarios", nargs="*", default=["daemon", "fleet"], choices=["daemon", "fleet", "s3"])
    ap.add_argument("--bin-dir", default=os.path.join(REPO_ROOT, "control-plane", "target", "release"))
    ap.add_argument("--keep", action="store_true", help="keep the temp work directory for debugging")
    ap.add_argument("--s3-endpoint", default=os.environ.get("HYPERCACHE_TEST_S3_ENDPOINT"))
    ap.add_argument("--s3-access-key", default=os.environ.get("HYPERCACHE_TEST_S3_ACCESS_KEY", "minioadmin"))
    ap.add_argument("--s3-secret-key", default=os.environ.get("HYPERCACHE_TEST_S3_SECRET_KEY", "minioadmin"))
    ap.add_argument("--s3-bucket", default=os.environ.get("HYPERCACHE_TEST_S3_BUCKET", "hypercache-e2e"))
    args = ap.parse_args()

    ctx = Ctx(args.bin_dir, args.keep)
    rep = Report()
    print(f"E2E work directory: {ctx.workdir}")
    try:
        for name in args.scenarios:
            if name == "daemon":
                scenario_daemon(ctx, rep)
            elif name == "fleet":
                scenario_fleet(ctx, rep)
            elif name == "s3":
                if not args.s3_endpoint:
                    rep.check("s3 scenario requires --s3-endpoint", lambda: (_ for _ in ()).throw(ValueError("no endpoint")))
                else:
                    scenario_s3(ctx, rep, args.s3_endpoint, args.s3_access_key, args.s3_secret_key, args.s3_bucket)
    finally:
        failed = any(not r[1] for r in rep.results)
        if failed:
            for owner in ctx.procs:
                if isinstance(owner, Node):
                    print(f"--- log tail: {owner.name} ---\n{owner.log_tail(25)}")
        ctx.cleanup()
    return rep.summary()


if __name__ == "__main__":
    sys.exit(main())
