#!/usr/bin/env bash
# ==============================================================================
# Container <-> host cache access scenario
# ==============================================================================
# Runs the daemon from the release image (as root, like the Kubernetes DaemonSet) and uses
# the cache from a second container that shares the daemon's volumes at the same paths. The
# second container plays the host / a training pod and runs as an unprivileged user.
#
# Verifies:
#   - socket, arena and RAM-tier files are shared with client_gid (modes and ownership)
#   - a non-root client in client_gid can read models (sha256, O_DIRECT through the volume)
#     and write + read back checkpoints on the NVMe arena and the RAM tier
#   - a user outside client_gid is refused with an error that names client_gid
#   - a client without the shared /dev/shm fails loudly instead of losing a RAM checkpoint
#   - `docker stop` (SIGTERM) shuts the daemon down cleanly and removes the socket
#   - a restarted daemon warm-boots the manifest and keeps serving the host
#
# Usage: scripts/e2e/container_host_access.sh [IMAGE]   (default hypercache:verify)
# ==============================================================================
set -u

IMAGE=${1:-hypercache:verify}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
ID="hcx$$"
DAEMON="$ID-daemon"
CLIENT_GID=2000
WORK=$(mktemp -d)
FAILS=0

pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; FAILS=$((FAILS + 1)); }

cleanup() {
    docker rm -f "$DAEMON" >/dev/null 2>&1
    for v in sock cache meta nfs shm; do docker volume rm -f "$ID-$v" >/dev/null 2>&1; done
    rm -rf "$WORK"
}
trap cleanup EXIT

for v in sock cache meta nfs shm; do docker volume create "$ID-$v" >/dev/null; done
SHARED="-v $ID-sock:/run/hypercache -v $ID-cache:/var/cache/hypercache -v $ID-nfs:/mnt/nfs"
SHM="-v $ID-shm:/dev/shm/hypercache"

cat > "$WORK/agent.toml" <<EOF
storage_backing_mode = "filesystem"
filesystem_cache_paths = ["/var/cache/hypercache"]
filesystem_arena_size_bytes = 268435456
metadata_persistence_enabled = true
metadata_persistence_path = "/var/lib/hypercache/metadata.json"
authoritative_backend = "nfs"
nfs_mount_path = "/mnt/nfs"
ram_cache_enabled = true
ram_cache_shm_path = "/dev/shm/hypercache"
ram_cache_capacity_bytes = 134217728
client_gid = $CLIENT_GID
EOF
cp "$REPO/scripts/e2e/host_probe.py" "$WORK/host_probe.py"

# Source model on the shared "NFS" volume (readable by everyone, like a dataset share)
docker run --rm -v "$ID-nfs:/mnt/nfs" --entrypoint sh "$IMAGE" -c \
    'head -c 20971527 /dev/urandom > /mnt/nfs/model.bin && chmod 644 /mnt/nfs/model.bin'

start_daemon() {
    docker rm -f "$DAEMON" >/dev/null 2>&1
    docker run -d --name "$DAEMON" $SHARED $SHM -v "$ID-meta:/var/lib/hypercache" \
        -v "$WORK/agent.toml:/etc/hypercache/agent.toml:ro" \
        "$IMAGE" -c /etc/hypercache/agent.toml start >/dev/null
    for _ in $(seq 1 60); do
        docker exec "$DAEMON" test -S /run/hypercache/hypercache.sock 2>/dev/null && return 0
        sleep 0.5
    done
    docker logs "$DAEMON" 2>&1 | tail -20
    return 1
}

# probe NAME USER EXTRA_DOCKER_ARGS... ; env PROBE_EXPECT / PROBE_RAM pass through
probe() {
    local name=$1 user=$2
    shift 2
    echo "-- $name"
    if docker run --rm --user "$user" $SHARED "$@" \
        -e PROBE_EXPECT="${PROBE_EXPECT:-ok}" -e PROBE_RAM="${PROBE_RAM:-0}" \
        -v "$WORK/host_probe.py:/probe.py:ro" --entrypoint python3 "$IMAGE" /probe.py; then
        pass "$name"
    else
        fail "$name"
    fi
}

echo "== container <-> host cache access ($IMAGE)"
if ! start_daemon; then
    fail "daemon container starts"
    exit 1
fi
pass "daemon container starts"

# Ingest once as root so the arena file exists before the mode checks
docker exec "$DAEMON" hypercache preload file:///mnt/nfs/model.bin --tier nvme >/dev/null 2>&1

modes=$(docker exec "$DAEMON" stat -c '%n %a %g' /run/hypercache/hypercache.sock \
    /var/cache/hypercache/hypercache_arena.dat /dev/shm/hypercache)
echo "$modes" | sed 's/^/    /'
if echo "$modes" | grep -q "hypercache.sock 660 $CLIENT_GID" &&
    echo "$modes" | grep -q "hypercache_arena.dat 660 $CLIENT_GID" &&
    echo "$modes" | grep -q "/dev/shm/hypercache 2770 $CLIENT_GID"; then
    pass "socket, arena and RAM directory are shared with client_gid"
else
    fail "socket, arena and RAM directory are shared with client_gid"
fi

PROBE_RAM=1 probe "non-root client in client_gid: read, O_DIRECT, NVMe + RAM checkpoints" \
    "1000:1000" --group-add "$CLIENT_GID" $SHM
PROBE_RAM=1 probe "root client: read, O_DIRECT, NVMe + RAM checkpoints" "0:0" $SHM
PROBE_EXPECT=denied probe "user outside client_gid is refused with a clear error" "1001:1001"
PROBE_EXPECT=unshared probe "client without the shared /dev/shm fails loudly on RAM writes" \
    "1000:1000" --group-add "$CLIENT_GID"

t0=$(date +%s)
docker stop -t 30 "$DAEMON" >/dev/null
elapsed=$(($(date +%s) - t0))
if [ "$elapsed" -lt 8 ] && docker logs "$DAEMON" 2>&1 | grep -q "stopped cleanly"; then
    pass "docker stop (SIGTERM) shuts down cleanly in ${elapsed}s"
else
    fail "docker stop (SIGTERM) shuts down cleanly (took ${elapsed}s)"
fi
if docker run --rm -v "$ID-sock:/run/hypercache" --entrypoint sh "$IMAGE" -c 'test ! -e /run/hypercache/hypercache.sock'; then
    pass "socket removed on shutdown"
else
    fail "socket removed on shutdown"
fi

if start_daemon && docker logs "$DAEMON" 2>&1 | grep -q "Warm-Boot Recovery: Restored"; then
    pass "restarted daemon warm-boots the manifest"
else
    fail "restarted daemon warm-boots the manifest"
fi
probe "non-root client after restart" "1000:1000" --group-add "$CLIENT_GID" $SHM

echo
if [ "$FAILS" -eq 0 ]; then
    echo "container_host_access: all checks passed"
else
    echo "container_host_access: $FAILS check(s) failed"
    docker logs "$DAEMON" 2>&1 | tail -30
fi
exit "$FAILS"
