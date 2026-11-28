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
#   ownership   auto discovery (limited to one loop device) refuses a drive that holds data and
#               leaves it untouched; prepare-drive needs confirmation, then erases and marks it;
#               the marked drive is used, and still used after a restart with cached data on it;
#               a drive HyperCache used before markers (listed in the manifest) is kept, not
#               erased; zero_devices_with_data erases only drives named by /dev/disk/by-id/.
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
LOOP2=""
BYID=""
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
    [ -n "$LOOP2" ] && losetup -d "$LOOP2" 2>/dev/null
    [ -n "$BYID" ] && rm -f "$BYID"
    rm -rf "$WORK"
}
trap cleanup EXIT

# Clients traverse the work dir to reach the socket and the model. The probe and the client
# package are copied there too: the checkout may sit in a home directory other users cannot
# enter (GitHub runners: /home/runner is 0750).
chmod 755 "$WORK"
mkdir -p "$WORK/nfs" "$WORK/run" "$WORK/client"
cp -r "$REPO/client-py/hypercache" "$WORK/client/"
cp "$REPO/scripts/e2e/host_probe.py" "$WORK/client/"
chmod -R a+rX "$WORK/client"
head -c 20971527 /dev/urandom > "$WORK/nfs/model.bin"
chmod 644 "$WORK/nfs/model.bin"
export HYPERCACHE_SOCKET_PATH="$WORK/run/hypercache.sock"

# Clients run as uid 1000 / 1001: the interpreter and its packages must be usable by them
if ! setpriv --reuid 1000 --regid 1000 --clear-groups env HOME=/tmp PYTHONPATH="$WORK/client" \
    "$PY" -c "import fsspec, numpy, hypercache" 2>"$WORK/py.err"; then
    echo "  FAIL  $PY with fsspec/numpy is not usable by uid 1000:"
    sed 's/^/        /' "$WORK/py.err"
    exit 1
fi

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
        PYTHONPATH="$WORK/client" HYPERCACHE_SOCKET_PATH="$HYPERCACHE_SOCKET_PATH" \
        PROBE_EXPECT="$expect" PROBE_SOURCE="$WORK/nfs/model.bin" \
        "$PY" "$WORK/client/host_probe.py"; then
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

echo "== bare metal: auto discovery uses only blank, marked or allowed drives"
rm -rf "$WORK/meta"
truncate -s 256M "$WORK/auto.img"
LOOP2=$(losetup -f --show "$WORK/auto.img") || { fail "attach second loop device"; exit 1; }
L2=${LOOP2#/dev/}
auto_cfg() { # file [extra toml line]
    cat > "$1" <<EOF
storage_backing_mode = "raw_block"
device_discovery_mode = "auto"
device_prefixes = ["loop"]
device_filter = "$L2"
auto_create_mock_devices = false
ram_cache_enabled = false
metadata_persistence_path = "$WORK/meta/metadata.json"
authoritative_backend = "nfs"
nfs_mount_path = "$WORK/nfs"
${2:-}
EOF
}
uses_loop2() { grep -q "Discovered 1 Active Block Device" "$1" && grep -q "$LOOP2" "$1"; }
marked() { head -c 20 "$LOOP2" | grep -q "HYPERCACHE-DEVICE-V1"; }
past_marker() { dd if="$LOOP2" bs=4096 skip=1 count=16383 status=none | sha256sum; } # 4 KiB..64 MiB

# Someone else's data in the middle of an otherwise empty drive
dd if=/dev/urandom of="$LOOP2" bs=1M seek=100 count=4 conv=notrunc status=none
before=$(sha256sum < "$LOOP2")
auto_cfg "$WORK/auto.toml"
"$HC" -c "$WORK/auto.toml" inspect >"$WORK/inspect.log" 2>&1
if grep -q "not blank and is not a HyperCache drive" "$WORK/inspect.log" && grep -q "prepare-drive" "$WORK/inspect.log"; then
    pass "a drive holding data is refused, with how to erase it"
else
    fail "a drive holding data is refused, with how to erase it"
    tail -5 "$WORK/inspect.log"
fi
if start_daemon "$WORK/auto.toml" "$WORK/auto-refused.log"; then
    if uses_loop2 "$WORK/auto-refused.log"; then
        fail "daemon does not use the drive holding data"
    else
        pass "daemon does not use the drive holding data"
    fi
    stop_daemon
else
    fail "daemon starts without the drive holding data"
    tail -5 "$WORK/auto-refused.log"
fi
[ "$(sha256sum < "$LOOP2")" = "$before" ] && pass "the drive holding data is left untouched" ||
    fail "the drive holding data is left untouched"

echo no | "$HC" prepare-drive "$LOOP2" >"$WORK/prep-no.log" 2>&1
[ "$(sha256sum < "$LOOP2")" = "$before" ] && pass "prepare-drive changes nothing without confirmation" ||
    fail "prepare-drive changes nothing without confirmation"
if echo "$L2" | "$HC" prepare-drive "$LOOP2" >"$WORK/prep.log" 2>&1 && marked &&
    [ "$(dd if="$LOOP2" bs=1M skip=100 count=4 status=none | tr -d '\0' | wc -c)" -eq 0 ]; then
    pass "prepare-drive erases the drive and marks it after confirmation"
else
    fail "prepare-drive erases the drive and marks it after confirmation"
    cat "$WORK/prep.log"
fi

if start_daemon "$WORK/auto.toml" "$WORK/auto-marked.log" && uses_loop2 "$WORK/auto-marked.log"; then
    pass "auto discovery uses the marked drive"
    probe "client reads and writes through the auto-discovered drive" 0 "" ok
    stop_daemon
    if start_daemon "$WORK/auto.toml" "$WORK/auto-restart.log" && uses_loop2 "$WORK/auto-restart.log"; then
        pass "the marked drive is still used after a restart with cached data on it"
    else
        fail "the marked drive is still used after a restart with cached data on it"
        tail -5 "$WORK/auto-restart.log"
    fi
    stop_daemon
else
    fail "auto discovery uses the marked drive"
    tail -10 "$WORK/auto-marked.log"
    stop_daemon
fi

# Upgrade: a drive HyperCache used before markers existed (manual mode never marks)
rm -rf "$WORK/meta"
blkdiscard "$LOOP2" 2>/dev/null || dd if=/dev/zero of="$LOOP2" bs=1M count=256 status=none
cat > "$WORK/manual2.toml" <<EOF
storage_backing_mode = "raw_block"
device_discovery_mode = "manual"
block_devices = ["$LOOP2"]
ram_cache_enabled = false
metadata_persistence_path = "$WORK/meta/metadata.json"
authoritative_backend = "nfs"
nfs_mount_path = "$WORK/nfs"
EOF
if start_daemon "$WORK/manual2.toml" "$WORK/manual2.log"; then
    probe "a drive used in manual mode gets cached data" 0 "" ok
    stop_daemon
fi
if marked; then fail "manual mode leaves no marker"; else pass "manual mode leaves no marker"; fi
cached=$(past_marker)
if start_daemon "$WORK/auto.toml" "$WORK/auto-upgrade.log" && uses_loop2 "$WORK/auto-upgrade.log" && marked &&
    [ "$(past_marker)" = "$cached" ]; then
    pass "a drive the manifest lists is kept and marked, not erased"
    probe "cached data on the kept drive still reads correctly" 0 "" ok
    stop_daemon
else
    fail "a drive the manifest lists is kept and marked, not erased"
    tail -10 "$WORK/auto-upgrade.log"
    stop_daemon
fi

# zero_devices_with_data: only /dev/disk/by-id/ names count
rm -rf "$WORK/meta"
dd if=/dev/urandom of="$LOOP2" bs=1M count=4 conv=notrunc status=none
mkdir -p /dev/disk/by-id
BYID="/dev/disk/by-id/hypercache-e2e-$L2-$$"
ln -sf "$LOOP2" "$BYID"
auto_cfg "$WORK/zero-bad.toml" "zero_devices_with_data = [\"$LOOP2\"]"
if start_daemon "$WORK/zero-bad.toml" "$WORK/zero-bad.log"; then
    if ! uses_loop2 "$WORK/zero-bad.log" && grep -q "by-id" "$WORK/zero-bad.log" && ! marked; then
        pass "zero_devices_with_data ignores kernel names like $LOOP2"
    else
        fail "zero_devices_with_data ignores kernel names like $LOOP2"
    fi
    stop_daemon
else
    fail "daemon starts with a kernel name in zero_devices_with_data"
fi
auto_cfg "$WORK/zero.toml" "zero_devices_with_data = [\"$BYID\"]"
if start_daemon "$WORK/zero.toml" "$WORK/zero.log" && uses_loop2 "$WORK/zero.log" && marked &&
    grep -q "Erased $LOOP2" "$WORK/zero.log"; then
    pass "zero_devices_with_data erases and claims a drive named by /dev/disk/by-id/"
    stop_daemon
    if start_daemon "$WORK/zero.toml" "$WORK/zero2.log" && uses_loop2 "$WORK/zero2.log" &&
        ! grep -q "Erased" "$WORK/zero2.log"; then
        pass "a drive already claimed is not erased again"
    else
        fail "a drive already claimed is not erased again"
    fi
    stop_daemon
else
    fail "zero_devices_with_data erases and claims a drive named by /dev/disk/by-id/"
    tail -10 "$WORK/zero.log"
    stop_daemon
fi

echo
if [ "$FAILS" -eq 0 ]; then
    echo "bare_metal_access: all checks passed"
else
    echo "bare_metal_access: $FAILS check(s) failed"
fi
exit "$FAILS"
