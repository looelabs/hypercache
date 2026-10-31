#!/usr/bin/env bash
# ==============================================================================
# Kubernetes smoke test of deploy/kubernetes/hypercache.yaml on a throwaway minikube cluster
# ==============================================================================
# Deploys the bundled manifest unmodified (except the image, which is the locally built one),
# then checks:
#   - controller and agent pods pass their readiness probes
#   - the agent registers with the controller (token auth, no loopback bypass)
#   - a non-root training pod (supplementalGroups = client_gid) using the documented
#     hostPath mounts can write and read back checkpoints through the node's cache
#   - deleting the agent pod terminates quickly (SIGTERM handled) and the replacement
#     warm-boots and serves the training pod again
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

if $K apply -f "$REPO/deploy/kubernetes/hypercache.yaml" >/dev/null; then
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
run_trainer() {
    local name=$1
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
    $K -n "$NS" logs "$name" 2>&1 | sed 's/^/    /'
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

echo
if [ "$FAILS" -eq 0 ]; then
    echo "k8s_smoke: all checks passed"
else
    echo "k8s_smoke: $FAILS check(s) failed"
fi
exit "$FAILS"
