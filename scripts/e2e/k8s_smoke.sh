#!/usr/bin/env bash
# ==============================================================================
# Kubernetes smoke test of deploy/kubernetes/hypercache.yaml on a throwaway minikube cluster
# ==============================================================================
# Deploys the bundled manifest (with the locally built image and a 1 GiB arena, since the
# 100 GiB default is now reserved up front and does not fit a minikube node), then checks:
#   - controller and agent pods pass their readiness probes
#   - the agent registers with the controller (token auth, no loopback bypass)
#   - a non-root training pod (supplementalGroups = client_gid) using the documented
#     hostPath mounts can write and read back checkpoints through the node's cache
#   - deleting the agent pod terminates quickly (SIGTERM handled) and the replacement
#     warm-boots and serves the training pod again
#   - raw_block mode as documented (privileged agent, host /dev): the loop device on the node
#     is shared with client_gid and a privileged non-root training pod writes checkpoints to it,
#     while an unprivileged pod cannot open it (the documented limitation)
#   - an arena larger than the node's free disk keeps the agent from starting, with the numbers
#     in its log
#
# Usage: scripts/e2e/k8s_smoke.sh [IMAGE]    (default hypercache:verify, must exist locally)
# Env:   KEEP_CLUSTER=1  keep the minikube profile for debugging
# ==============================================================================
set -u

IMAGE=${1:-hypercache:verify}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PROFILE="hypercache-e2e"
NS="hypercache-system"
MANIFEST_IMAGE="ghcr.io/looelabs/hypercache:latest"
FAILS=0
K="kubectl --context $PROFILE"

pass() { printf '  PASS  %s\n' "$1"; }
fail() { printf '  FAIL  %s\n' "$1"; FAILS=$((FAILS + 1)); }

cleanup() {
    [ -n "${WORK:-}" ] && rm -rf "$WORK"
    if [ "${KEEP_CLUSTER:-0}" != "1" ]; then
        minikube delete -p "$PROFILE" >/dev/null 2>&1
    fi
}
trap cleanup EXIT

echo "== kubernetes smoke test ($IMAGE on minikube profile $PROFILE)"
minikube start -p "$PROFILE" --driver=docker --cpus=2 --memory=3072 --wait=all >/dev/null 2>&1 || {
    fail "minikube cluster starts"
    exit 1
}
pass "minikube cluster starts"

# The manifest pins ghcr.io/looelabs/hypercache:latest with IfNotPresent; preload the local build under that name
docker tag "$IMAGE" "$MANIFEST_IMAGE"
minikube -p "$PROFILE" image load "$MANIFEST_IMAGE" >/dev/null 2>&1 || fail "load image into minikube"

WORK=$(mktemp -d)
python3 - "$REPO/deploy/kubernetes/hypercache.yaml" "$WORK/hypercache.yaml" <<'PY'
import sys
src, dst = sys.argv[1], sys.argv[2]
text = open(src).read()
old = "filesystem_arena_size_bytes = 107374182400"
assert old in text, "manifest arena size line changed; update k8s_smoke.sh"
open(dst, "w").write(text.replace(old, "filesystem_arena_size_bytes = 1073741824"))
PY
if $K apply -f "$WORK/hypercache.yaml" >/dev/null; then
    pass "manifest applies cleanly"
else
    fail "manifest applies cleanly"
    exit 1
fi

if $K -n "$NS" rollout status deploy/hypercache-controller --timeout=180s >/dev/null &&
    $K -n "$NS" rollout status ds/hypercache-agent --timeout=180s >/dev/null; then
    pass "controller and agent pass readiness probes"
else
    fail "controller and agent pass readiness probes"
    $K -n "$NS" get pods -o wide
    $K -n "$NS" logs ds/hypercache-agent --tail=30
fi

registered=0
for _ in $(seq 1 30); do
    if $K -n "$NS" exec deploy/hypercache-controller -- hypercache-admin nodes 2>/dev/null | grep -q ONLINE; then
        registered=1
        break
    fi
    sleep 2
done
[ "$registered" -eq 1 ] && pass "agent registers with the controller (token auth)" ||
    fail "agent registers with the controller (token auth)"

# Training pod, exactly as documented at the end of the manifest
$K -n "$NS" create configmap host-probe --from-file=probe.py="$REPO/scripts/e2e/host_probe.py" \
    --dry-run=client -o yaml | $K apply -f - >/dev/null
run_trainer() { # name [privileged]
    local name=$1 privileged=${2:-false}
    $K -n "$NS" delete pod "$name" --ignore-not-found >/dev/null
    $K -n "$NS" apply -f - >/dev/null <<EOF
apiVersion: v1
kind: Pod
metadata:
  name: $name
spec:
  restartPolicy: Never
  securityContext:
    runAsUser: 1000
    runAsGroup: 1000
    supplementalGroups: [2000]
  containers:
    - name: trainer
      image: $MANIFEST_IMAGE
      imagePullPolicy: IfNotPresent
      command: ["python3", "/probe/probe.py"]
      securityContext:
        privileged: $privileged
      volumeMounts:
        - {name: hypercache-socket, mountPath: /run/hypercache}
        - {name: hypercache-cache, mountPath: /var/cache/hypercache}
        - {name: probe, mountPath: /probe}
  volumes:
    - name: hypercache-socket
      hostPath: {path: /run/hypercache, type: Directory}
    - name: hypercache-cache
      hostPath: {path: /var/cache/hypercache, type: Directory}
    - name: probe
      configMap: {name: host-probe}
EOF
    for _ in $(seq 1 90); do
        phase=$($K -n "$NS" get pod "$name" -o jsonpath='{.status.phase}' 2>/dev/null)
        case "$phase" in Succeeded | Failed) break ;; esac
        sleep 2
    done
    $K -n "$NS" logs "$name" >"$WORK/$name.log" 2>&1
    sed 's/^/    /' "$WORK/$name.log"
    [ "$phase" = "Succeeded" ]
}

if run_trainer trainer-1; then
    pass "non-root training pod writes and reads checkpoints through the node cache"
else
    fail "non-root training pod writes and reads checkpoints through the node cache"
fi

old_pod=$($K -n "$NS" get pod -l app=hypercache-agent -o jsonpath='{.items[0].metadata.name}')
t0=$(date +%s)
$K -n "$NS" delete pod "$old_pod" --wait=true >/dev/null
elapsed=$(($(date +%s) - t0))
[ "$elapsed" -lt 15 ] && pass "agent pod terminates promptly on SIGTERM (${elapsed}s)" ||
    fail "agent pod terminates promptly on SIGTERM (took ${elapsed}s, grace period is 30s)"

if $K -n "$NS" rollout status ds/hypercache-agent --timeout=120s >/dev/null &&
    $K -n "$NS" logs ds/hypercache-agent 2>/dev/null | grep -q "Warm-Boot Recovery: Restored"; then
    pass "replacement agent warm-boots the node's manifest"
else
    fail "replacement agent warm-boots the node's manifest"
fi
if run_trainer trainer-2; then
    pass "training pod works against the replacement agent"
else
    fail "training pod works against the replacement agent"
fi

# Agent config edits: fetch both ConfigMap files, rewrite the agent's, apply, restart
$K -n "$NS" get cm hypercache-config -o jsonpath='{.data.hypercache-controller\.toml}' >"$WORK/controller.toml"
$K -n "$NS" get cm hypercache-config -o jsonpath='{.data.hypercache-agent\.toml}' >"$WORK/agent.orig.toml"
apply_agent_config() { # python-expression-on-text
    python3 - "$WORK/agent.orig.toml" "$WORK/agent.toml" "$1" <<'PY'
import re, sys
text = open(sys.argv[1]).read()
for pattern, repl in eval(sys.argv[3]):
    text, n = re.subn(pattern, repl, text, flags=re.M)
    assert n, f"pattern not found in agent config: {pattern}"
open(sys.argv[2], "w").write(text)
PY
    $K -n "$NS" create cm hypercache-config --from-file=hypercache-agent.toml="$WORK/agent.toml" \
        --from-file=hypercache-controller.toml="$WORK/controller.toml" --dry-run=client -o yaml | $K apply -f - >/dev/null
}

echo "-- raw_block mode"
LOOP=$(minikube -p "$PROFILE" ssh -- "sudo truncate -s 512M /var/tmp/hc-raw.img && sudo losetup -f --show /var/tmp/hc-raw.img" | tr -d '\r')
if [ -z "$LOOP" ]; then
    fail "attach a loop device on the node"
else
    apply_agent_config "[(r'^storage_backing_mode = .*$', 'storage_backing_mode = \"raw_block\"\\ndevice_discovery_mode = \"manual\"\\nblock_devices = [\"$LOOP\"]')]"
    # The documented raw_block changes: privileged agent with the host's /dev
    $K -n "$NS" patch ds hypercache-agent --type=json -p '[
      {"op": "add", "path": "/spec/template/spec/containers/0/securityContext", "value": {"privileged": true}},
      {"op": "add", "path": "/spec/template/spec/volumes/-", "value": {"name": "host-dev", "hostPath": {"path": "/dev", "type": "Directory"}}},
      {"op": "add", "path": "/spec/template/spec/containers/0/volumeMounts/-", "value": {"name": "host-dev", "mountPath": "/dev"}}
    ]' >/dev/null
    if $K -n "$NS" rollout status ds/hypercache-agent --timeout=180s >/dev/null; then
        mode=$($K -n "$NS" exec ds/hypercache-agent -- stat -c '%a %g' "$LOOP" 2>/dev/null)
        [ "$mode" = "660 2000" ] && pass "raw device on the node shared with client_gid ($LOOP $mode)" ||
            fail "raw device on the node shared with client_gid (got '$mode')"
        if run_trainer trainer-raw true; then
            pass "privileged non-root training pod writes checkpoints to the raw device"
        else
            fail "privileged non-root training pod writes checkpoints to the raw device"
        fi
        # Documented limitation: without privileged (or a device plugin) the container runtime's
        # device policy blocks the raw device even though the file mode would allow it
        if ! run_trainer trainer-raw-unprivileged false &&
            grep -qE "Operation not permitted|No such file or directory|Permission denied" "$WORK/trainer-raw-unprivileged.log"; then
            pass "unprivileged training pod cannot open the raw device (documented limitation)"
        else
            fail "unprivileged training pod cannot open the raw device (documented limitation)"
        fi
    else
        fail "agent comes up in raw_block mode"
        $K -n "$NS" logs ds/hypercache-agent --tail=20
    fi
fi

echo "-- arena larger than the node's free disk"
apply_agent_config "[(r'^filesystem_cache_paths = .*$', 'filesystem_cache_paths = [\"/var/cache/hypercache/oversized\"]'), (r'^filesystem_arena_size_bytes = .*$', 'filesystem_arena_size_bytes = 1125899906842624')]"
$K -n "$NS" rollout restart ds/hypercache-agent >/dev/null
refused=0
for _ in $(seq 1 60); do
    pod=$($K -n "$NS" get pod -l app=hypercache-agent --sort-by=.metadata.creationTimestamp -o jsonpath='{.items[-1:].metadata.name}')
    if { $K -n "$NS" logs "$pod" 2>/dev/null; $K -n "$NS" logs "$pod" --previous 2>/dev/null; } | grep -q "are free"; then
        refused=1
        break
    fi
    sleep 3
done
ready=$($K -n "$NS" get pod "$pod" -o jsonpath='{.status.containerStatuses[0].ready}' 2>/dev/null)
if [ "$refused" -eq 1 ] && [ "$ready" != "true" ]; then
    pass "agent refuses to start with an oversized arena and logs the free space"
else
    fail "agent refuses to start with an oversized arena (log match: $refused, ready: $ready)"
fi
minikube -p "$PROFILE" ssh -- "sudo losetup -d $LOOP" >/dev/null 2>&1

echo
if [ "$FAILS" -eq 0 ]; then
    echo "k8s_smoke: all checks passed"
else
    echo "k8s_smoke: $FAILS check(s) failed"
fi
exit "$FAILS"
