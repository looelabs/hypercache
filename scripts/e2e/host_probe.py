"""Host-side client probe for scripts/e2e/container_host_access.sh.

Runs outside the daemon's container (a second container sharing the daemon's volumes at the
same paths, as a host process or a training pod would) and uses the cache like a training job.

Environment:
  PROBE_EXPECT   ok        every operation must succeed (default)
                 denied    the socket must refuse this user with an error that names client_gid
                 unshared  RAM-tier writes must fail loudly (this process has its own /dev/shm)
  PROBE_RAM      1         also exercise the RAM tier
  PROBE_SOURCE   path of the source model under nfs_mount_path (default /mnt/nfs/model.bin);
                 the model-read checks are skipped when it does not exist
"""

import hashlib
import os
import sys

import fsspec
import hypercache  # noqa: F401  registers the hypercache:// protocol
from hypercache.client import HyperCacheClient
from hypercache.datapath import DirectBlockReader

EXPECT = os.environ.get("PROBE_EXPECT", "ok")
SOURCE = os.environ.get("PROBE_SOURCE", "/mnt/nfs/model.bin")
results = []


def check(name, fn):
    try:
        results.append((True, name, fn() or ""))
    except Exception as e:
        results.append((False, name, f"{type(e).__name__}: {e}"))


def expect_error(name, fn, needle):
    def run():
        try:
            fn()
        except Exception as e:
            if needle not in str(e):
                raise AssertionError(f"error does not mention {needle!r}: {e}") from e
            return f"refused as expected: {type(e).__name__}"
        raise AssertionError("operation succeeded but should have failed")

    check(name, run)


client = HyperCacheClient()
fs = fsspec.filesystem("hypercache")


def ping():
    if not client.ping():
        raise AssertionError("daemon did not answer ping")


def read_model():
    expected = hashlib.sha256(open(SOURCE, "rb").read()).hexdigest()
    with fs.open("file://" + SOURCE, "rb") as f:
        data = bytes(f.read())
    if hashlib.sha256(data).hexdigest() != expected:
        raise AssertionError(f"sha256 mismatch on {len(data)} bytes")
    loc = client.locate("file://" + SOURCE, auto_ingest=False)
    return f"{len(data)} bytes, {loc.chunk_layout[0].tier} tier, {loc.chunk_layout[0].device}"


def direct_io_read():
    loc = client.locate("file://" + SOURCE, auto_ingest=False)
    nvme = [c for c in loc.chunk_layout if c.tier != "ram"]
    if not nvme:
        return "model is RAM-resident; O_DIRECT not applicable"
    reader = DirectBlockReader()
    try:
        reader.read_range(nvme[:1], 0, nvme[0].length)
        stats = reader.get_stats()
    finally:
        reader.close()
    if stats.get("direct_fallbacks") or not stats.get("direct_reads"):
        raise AssertionError(f"O_DIRECT did not engage through the shared volume: {stats}")
    return str(stats)


def checkpoint(tier):
    payload = os.urandom(9 * 1024 * 1024 + 123)
    uri = f"hypercache://checkpoints/probe-{tier}-{os.getuid()}-{os.getpid()}.pt"
    with fs.open(uri, "wb", tier=tier) as f:
        f.write(payload)
    with fs.open(uri, "rb") as f:
        back = bytes(f.read())
    if back != payload:
        raise AssertionError("checkpoint read-back mismatch")
    loc = client.locate(uri, auto_ingest=False)
    tiers = {c.tier for c in loc.chunk_layout}
    if tiers != {tier}:
        raise AssertionError(f"expected {tier} tier, got {tiers}")
    return f"{len(payload)} bytes on {loc.chunk_layout[0].device}"


print(f"probe uid={os.getuid()} groups={os.getgroups()} expect={EXPECT}")
if EXPECT == "denied":
    expect_error("socket refuses a user outside client_gid", lambda: client.locate("hypercache://x", False), "client_gid")
elif EXPECT == "unshared":
    check("connect to daemon socket", ping)
    expect_error("RAM checkpoint write fails loudly without shared /dev/shm", lambda: checkpoint("ram"), "/dev/shm")
else:
    check("connect to daemon socket", ping)
    if os.path.exists(SOURCE):
        check("read cached model, sha256 verified", read_model)
        check("O_DIRECT read through the shared volume", direct_io_read)
    check("NVMe arena checkpoint write + read back", lambda: checkpoint("nvme"))
    if os.environ.get("PROBE_RAM") == "1":
        check("RAM tier checkpoint write + read back", lambda: checkpoint("ram"))

for ok, name, detail in results:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}  {detail}")
sys.exit(0 if results and all(ok for ok, _, _ in results) else 1)
