#!/usr/bin/env bash
# Runs scripts/e2e/lmcache_l2_e2e.py against a real daemon on a loop device. Used by CI, and
# runnable on any Linux host with sudo (for losetup only) and no GPU.
#
#   scripts/e2e/lmcache_l2_ci.sh [--python python3] [--daemon control-plane/target/release/hypercache]
#
# Expects LMCache, torch and the connector wheel installed for that Python. Never touches a real
# drive: the device is a sparse file in a temporary directory, removed on exit.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
PY="python3"
DAEMON="$REPO_ROOT/control-plane/target/release/hypercache"
while [ $# -gt 0 ]; do
    case "$1" in
        --python) PY="$2"; shift 2 ;;
        --daemon) DAEMON="$2"; shift 2 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

WORK="$(mktemp -d)"
LOOP=""
DAEMON_PID=""
cleanup() {
    local status=$?
    if [ -n "$DAEMON_PID" ]; then
        kill "$DAEMON_PID" 2>/dev/null || true
        wait "$DAEMON_PID" 2>/dev/null || true
    fi
    if [ "$status" -ne 0 ] && [ -f "$WORK/daemon.log" ]; then
        echo "--- daemon log (last 40 lines)"
        tail -40 "$WORK/daemon.log"
    fi
    if [ -n "$LOOP" ]; then
        sudo losetup -d "$LOOP" || true
    fi
    rm -rf "$WORK"
    exit "$status"
}
trap cleanup EXIT

# 4 KiB sectors like the NVMe drives this targets, so O_DIRECT alignment is exercised
truncate -s 4G "$WORK/device.img"
LOOP="$(sudo losetup --find --show --sector-size 4096 "$WORK/device.img")"
# The daemon and the corruption check both open the device; neither needs root otherwise
sudo chown "$(id -u)" "$LOOP"
echo "device: $LOOP"

SOCK="$WORK/hypercache.sock"
cat > "$WORK/daemon.toml" <<EOF
device_discovery_mode = "manual"
block_devices = ["$LOOP"]
include_partitions = false
exclude_root_device = true
storage_backing_mode = "raw_block"
chunk_size_bytes = 4194304
device_start_offset_bytes = 4096
metadata_persistence_enabled = false
# Local-only blocks are never flushed, so this backend must never be contacted
authoritative_backend = "s3"
s3_endpoint = "http://127.0.0.1:9"
s3_region = "us-east-1"
s3_bucket = "must-not-be-used"
allow_synthetic_s3_fallback = false
agent_enabled = false
metrics_enabled = false
ram_cache_enabled = false
EOF

HYPERCACHE_SOCKET_PATH="$SOCK" "$DAEMON" -c "$WORK/daemon.toml" start > "$WORK/daemon.log" 2>&1 &
DAEMON_PID=$!
for _ in $(seq 1 60); do
    [ -S "$SOCK" ] && break
    kill -0 "$DAEMON_PID" 2>/dev/null || { echo "daemon exited during startup"; exit 1; }
    sleep 0.5
done
[ -S "$SOCK" ] || { echo "daemon socket never appeared"; exit 1; }

LMCACHE_TRACK_USAGE=false LMCACHE_DISABLE_BANNER=1 PYTHONPATH="$REPO_ROOT/client-py${PYTHONPATH:+:$PYTHONPATH}" \
    "$PY" "$REPO_ROOT/scripts/e2e/lmcache_l2_e2e.py" --socket "$SOCK"

# A panic would not necessarily fail a check, so look for one explicitly
if grep -qi 'panicked' "$WORK/daemon.log"; then
    echo "the daemon panicked during the run"
    exit 1
fi
