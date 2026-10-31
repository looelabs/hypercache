#!/usr/bin/env bash
# ==============================================================================
# Bare-metal Linux scenario: release binaries run directly on the host, no containers
# ==============================================================================
# Needs Linux, root, losetup, mkfs.ext4, mount and setpriv. The daemon runs as root (as it
# must for raw devices); clients run as unprivileged users via setpriv.
#
# Verifies:
#   raw_block   a loop device stands in for an NVMe drive. With client_gid the daemon shares
#               the device node (0660, client group); a non-root client in that group reads
#               models with O_DIRECT and writes + reads back checkpoints on the raw device;
#               a user outside the group is refused.
#   capacity    filesystem arenas on a small ext4 filesystem: an arena larger than the free
#               space makes the daemon refuse to start (and `check` report FAIL) instead of
#               failing writes later; a fitting arena is fully reserved on disk; opting out with
#               filesystem_arena_preallocate = false starts with a sparse arena.
#   shutdown    SIGTERM stops the daemon cleanly and removes the socket.
#
# Usage: sudo scripts/e2e/bare_metal_access.sh BIN_DIR [PYTHON]
# ==============================================================================
set -u

BIN_DIR=${1:?usage: bare_metal_access.sh BIN_DIR [PYTHON]}
PY=${2:-python3}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
HC="$BIN_DIR/hypercache"
CLIENT_GID=2000
WORK=$(mktemp -d /tmp/hc-bare.XXXXXX)
FAILS=0
LOOP=""
DAEMON_PID=""

pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; FAILS=$((FAILS + 1)); }

if [ "$(id -u)" -ne 0 ] || [ "$(uname -s)" != "Linux" ]; then
    echo "bare_metal_access.sh needs root on Linux"
    exit 2
fi
for tool in losetup mkfs.ext4 mount setpriv; do
    command -v "$tool" >/dev/null || { echo "missing tool: $tool"; exit 2; }
done

cleanup() {
    [ -n "$DAEMON_PID" ] && kill -9 "$DAEMON_PID" 2>/dev/null
    mountpoint -q "$WORK/smallfs" 2>/dev/null && umount "$WORK/smallfs"
    [ -n "$LOOP" ] && losetup -d "$LOOP" 2>/dev/null
    rm -rf "$WORK"
}
trap cleanup EXIT

# Clients traverse the work dir to reach the socket and the model
chmod 755 "$WORK"
mkdir -p "$WORK/nfs" "$WORK/run"
head -c 20971527 /dev/urandom > "$WORK/nfs/model.bin"
chmod 644 "$WORK/nfs/model.bin"
export HYPERCACHE_SOCKET_PATH="$WORK/run/hypercache.sock"

start_daemon() { # config log
    "$HC" -c "$1" start >"$2" 2>&1 &
    DAEMON_PID=$!
    for _ in $(seq 1 60); do
        [ -S "$HYPERCACHE_SOCKET_PATH" ] && return 0
        kill -0 "$DAEMON_PID" 2>/dev/null || return 1
        sleep 0.25
    done
    return 1
}

stop_daemon() {
    [ -z "$DAEMON_PID" ] && return 0
    kill -TERM "$DAEMON_PID" 2>/dev/null
    for _ in $(seq 1 40); do
        kill -0 "$DAEMON_PID" 2>/dev/null || { DAEMON_PID=""; return 0; }
        sleep 0.25
    done
    kill -9 "$DAEMON_PID" 2>/dev/null
    DAEMON_PID=""
    return 1
}

probe() { # name uid groups expect
    local name=$1 uid=$2 groups=$3 expect=$4
    echo "-- $name"
    local grp_args=(--clear-groups)
    [ -n "$groups" ] && grp_args=(--groups "$groups")
    if setpriv --reuid "$uid" --regid "$uid" "${grp_args[@]}" env HOME=/tmp \
        PYTHONPATH="$REPO/client-py" HYPERCACHE_SOCKET_PATH="$HYPERCACHE_SOCKET_PATH" \
        PROBE_EXPECT="$expect" PROBE_SOURCE="$WORK/nfs/model.bin" \
        "$PY" "$REPO/scripts/e2e/host_probe.py"; then
        pass "$name"
    else
        fail "$name"
    fi
}

echo "== bare metal: raw block device shared with client_gid"
truncate -s 512M "$WORK/raw.img"
LOOP=$(losetup -f --show "$WORK/raw.img") || { fail "attach loop device"; exit 1; }
cat > "$WORK/raw.toml" <<EOF
storage_backing_mode = "raw_block"
device_discovery_mode = "manual"
block_devices = ["$LOOP"]
metadata_persistence_path = "$WORK/meta/metadata.json"
authoritative_backend = "nfs"
nfs_mount_path = "$WORK/nfs"
client_gid = $CLIENT_GID
EOF
if start_daemon "$WORK/raw.toml" "$WORK/raw.log"; then
    pass "daemon starts on $LOOP"
    mode=$(stat -c '%a %g' "$LOOP")
    if [ "$mode" = "660 $CLIENT_GID" ]; then
        pass "device node shared with client_gid ($LOOP $mode)"
    else
        fail "device node shared with client_gid (got $mode)"
    fi
    probe "non-root client in client_gid: O_DIRECT model read + raw-device checkpoints" 1000 "$CLIENT_GID" ok
    probe "user outside client_gid is refused" 1001 "" denied
    if setpriv --reuid 1001 --regid 1001 --clear-groups dd if="$LOOP" of=/dev/null bs=4096 count=1 2>/dev/null; then
        fail "user outside client_gid cannot open the raw device"
    else
        pass "user outside client_gid cannot open the raw device"
    fi
    t0=$(date +%s)
    if stop_daemon && [ ! -e "$HYPERCACHE_SOCKET_PATH" ]; then
        pass "SIGTERM stops the daemon and removes the socket ($(($(date +%s) - t0))s)"
    else
        fail "SIGTERM stops the daemon and removes the socket"
    fi
else
    fail "daemon starts on $LOOP"
    tail -20 "$WORK/raw.log"
fi

echo "== bare metal: filesystem arena capacity on a 128 MiB ext4 filesystem"
rm -rf "$WORK/meta"
truncate -s 128M "$WORK/small.img"
mkfs.ext4 -q -F "$WORK/small.img"
mkdir -p "$WORK/smallfs"
mount -o loop "$WORK/small.img" "$WORK/smallfs" || { fail "mount small ext4 filesystem"; exit 1; }
arena_cfg() { # file size preallocate
    cat > "$1" <<EOF
storage_backing_mode = "filesystem"
filesystem_cache_paths = ["$WORK/smallfs/cache"]
filesystem_arena_size_bytes = $2
filesystem_arena_preallocate = $3
metadata_persistence_path = "$WORK/meta/metadata.json"
authoritative_backend = "nfs"
nfs_mount_path = "$WORK/nfs"
EOF
}

arena_cfg "$WORK/big.toml" 536870912 true
if "$HC" -c "$WORK/big.toml" check >"$WORK/check.log" 2>&1; then
    fail "check reports an arena larger than the filesystem as FAIL"
else
    grep -q "Arena" "$WORK/check.log" && pass "check reports an arena larger than the filesystem as FAIL" ||
        fail "check reports an arena larger than the filesystem as FAIL"
fi
[ ! -e "$WORK/smallfs/cache/hypercache_arena.dat" ] && pass "check creates nothing" || fail "check creates nothing"

timeout 20 "$HC" -c "$WORK/big.toml" start >"$WORK/big.log" 2>&1
rc=$?
if [ "$rc" -ne 0 ] && [ "$rc" -ne 124 ] && grep -q "are free" "$WORK/big.log"; then
    pass "daemon refuses to start with an arena larger than the free space (exit $rc)"
else
    fail "daemon refuses to start with an arena larger than the free space (exit $rc)"
    tail -5 "$WORK/big.log"
fi
[ ! -e "$WORK/smallfs/cache/hypercache_arena.dat" ] && pass "no partial arena left behind" ||
    fail "no partial arena left behind"

arena_cfg "$WORK/fit.toml" 67108864 true
if start_daemon "$WORK/fit.toml" "$WORK/fit.log"; then
    reserved=$(($(stat -c '%b' "$WORK/smallfs/cache/hypercache_arena.dat") * 512))
    if [ "$reserved" -ge 67108864 ]; then
        pass "fitting arena is fully reserved on disk ($reserved bytes)"
    else
        fail "fitting arena is fully reserved on disk (only $reserved bytes)"
    fi
    probe "client reads and writes through the preallocated arena" 0 "" ok
    stop_daemon || fail "SIGTERM stops the arena daemon"
else
    fail "daemon starts with a fitting arena"
    tail -20 "$WORK/fit.log"
fi
rm -f "$WORK/smallfs/cache/hypercache_arena.dat" "$WORK/meta/metadata.json"

arena_cfg "$WORK/sparse.toml" 536870912 false
if start_daemon "$WORK/sparse.toml" "$WORK/sparse.log"; then
    pass "filesystem_arena_preallocate = false starts with a sparse arena"
    stop_daemon
else
    fail "filesystem_arena_preallocate = false starts with a sparse arena"
    tail -20 "$WORK/sparse.log"
fi

echo
if [ "$FAILS" -eq 0 ]; then
    echo "bare_metal_access: all checks passed"
else
    echo "bare_metal_access: $FAILS check(s) failed"
fi
exit "$FAILS"
