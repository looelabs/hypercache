#!/usr/bin/env bash
# ==============================================================================
# HyperCache release verification harness
# ==============================================================================
# Catches the classes of breakage that would ship a broken HyperCache:
#
#   static  version consistency, zero-emoji policy, cargo fmt, clippy -D warnings,
#           Cargo.lock in sync, Python byte-compile, shell/YAML/TOML syntax
#   rust    cargo test (unit + integration), release build, binary smoke (--version/--help)
#   python  pytest suite, wheel build (when `build` is installed)
#   e2e     real daemon + controller + agents on file-backed devices
#           (scripts/e2e/hypercache_e2e.py daemon fleet)
#   s3      live S3 against a throwaway versitygw container: SigV4, pagination,
#           multipart, parallel ranged GET, write-back (Rust + E2E). Needs Docker.
#   linux   re-runs static,rust,python,e2e,s3 inside a clean rust:1-bookworm container,
#           so Linux-only paths (O_DIRECT, sysfs, fdatasync) are exercised. Needs Docker.
#   image   builds the release Dockerfile and smoke-tests the image (binaries, configs,
#           Python client import, daemon start on a mock device), then uses the containerized
#           daemon from a second container as an unprivileged host client
#           (scripts/e2e/container_host_access.sh). Needs Docker.
#   k8s     deploys deploy/kubernetes/hypercache.yaml on a throwaway minikube cluster and runs a
#           non-root training pod against the node cache (scripts/e2e/k8s_smoke.sh). Needs
#           Docker, minikube and kubectl; skipped when they are missing.
#
# Usage:
#   scripts/verify_release.sh                 # all stages (Docker stages skipped if unavailable)
#   scripts/verify_release.sh --quick         # static,rust,python
#   scripts/verify_release.sh --stages e2e,s3
#   scripts/verify_release.sh --no-docker     # everything that does not need Docker
#
# Environment:
#   PYTHON                  Python interpreter (default: .venv/bin/python, else python3)
#   HYPERCACHE_TEST_S3_*    Use an existing S3 endpoint for the s3 stage instead of Docker
#   S3_IMAGE                S3 server image (default: versity/versitygw:v1.10.0)
#   RUST_IMAGE              Linux build image (default: rust:1-bookworm)
#   KEEP_E2E=1              Keep E2E temp directories for debugging
#
# Exit status is non-zero if any stage fails. A summary table is printed at the end.
# ==============================================================================
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

ALL_STAGES="static,rust,python,e2e,s3,linux,image,k8s"
STAGES="$ALL_STAGES"
NO_DOCKER=0
IN_CONTAINER=0
S3_IMAGE="${S3_IMAGE:-versity/versitygw:v1.10.0}"
RUST_IMAGE="${RUST_IMAGE:-rust:1-bookworm}"
S3_ACCESS="hcverify"
S3_SECRET="hcverify-secret-$$"
S3_BUCKET="hypercache-e2e"

while [ $# -gt 0 ]; do
    case "$1" in
        --stages) STAGES="$2"; shift 2 ;;
        --quick) STAGES="static,rust,python"; shift ;;
        --no-docker) NO_DOCKER=1; shift ;;
        --in-container) IN_CONTAINER=1; NO_DOCKER=1; shift ;;
        -h|--help) sed -n '2,34p' "$0"; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; exit 2 ;;
    esac
done

if [ -n "${PYTHON:-}" ]; then
    PY="$PYTHON"
elif [ -x "$REPO_ROOT/.venv/bin/python" ]; then
    PY="$REPO_ROOT/.venv/bin/python"
else
    PY="python3"
fi

CARGO_MANIFEST="control-plane/Cargo.toml"
RELEASE_DIR="$REPO_ROOT/control-plane/target/release"
LOG_DIR="$(mktemp -d "${TMPDIR:-/tmp}/hc-verify-logs.XXXXXX")"

RESULTS=()
FAILED=0
CLEANUP_CMDS=()

cleanup() {
    for cmd in "${CLEANUP_CMDS[@]+"${CLEANUP_CMDS[@]}"}"; do
        eval "$cmd" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT

has_stage() { [[ ",$STAGES," == *",$1,"* ]]; }

docker_ok() {
    [ "$NO_DOCKER" -eq 0 ] && command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1
}

# step <name> <command...>: runs a check, logs output, records PASS/FAIL
step() {
    local name="$1"; shift
    local log="$LOG_DIR/$(echo "$name" | tr -c 'a-zA-Z0-9' '_').log"
    local t0=$SECONDS
    printf '  %-62s ' "$name"
    if "$@" >"$log" 2>&1; then
        printf 'PASS (%ss)\n' "$((SECONDS - t0))"
        RESULTS+=("PASS  $name")
    else
        printf 'FAIL (%ss)\n' "$((SECONDS - t0))"
        RESULTS+=("FAIL  $name  (log: $log)")
        FAILED=1
        echo "    ---- last 30 lines of $log ----"
        tail -n 30 "$log" | sed 's/^/    /'
    fi
}

skip() {
    printf '  %-62s SKIP (%s)\n' "$1" "$2"
    RESULTS+=("SKIP  $1  ($2)")
}

section() { echo; echo "== $1"; }

# ------------------------------------------------------------------------------
# static
# ------------------------------------------------------------------------------

check_versions() {
    local cargo_v py_v init_v rel_v
    cargo_v=$(grep -m1 '^version' control-plane/Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
    py_v=$(grep -m1 '^version' client-py/pyproject.toml | sed 's/.*"\(.*\)".*/\1/')
    init_v=$(grep -m1 '__version__' client-py/hypercache/__init__.py | sed 's/.*"\(.*\)".*/\1/')
    rel_v=$(grep -m1 'default: "v' .github/workflows/release.yml | sed 's/.*"v\(.*\)".*/\1/')
    echo "Cargo.toml=$cargo_v pyproject=$py_v __init__=$init_v release.yml=$rel_v"
    [ -n "$cargo_v" ] && [ "$cargo_v" = "$py_v" ] && [ "$cargo_v" = "$init_v" ] && [ "$cargo_v" = "$rel_v" ]
}

check_no_hardcoded_versions() {
    # Binaries must take their version from Cargo.toml
    ! grep -rn 'version = "[0-9]' control-plane/src --include='*.rs' | grep -v CARGO_PKG_VERSION
}

check_emoji_policy() {
    "$PY" - <<'EOF'
import re, subprocess, sys
files = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard"],
                       capture_output=True, text=True, check=True).stdout.split("\n")
# Built from code points so this file itself stays ASCII-only
RANGES = [(0x1F000, 0x1FAFF), (0x2600, 0x27BF), (0x2B00, 0x2BFF)]
SINGLES = [0xFE0F, 0x200D]
pat = re.compile("[" + "".join(f"{chr(a)}-{chr(b)}" for a, b in RANGES) + "".join(map(chr, SINGLES)) + "]")
bad = []
for f in filter(None, files):
    try:
        with open(f, encoding="utf-8") as fh:
            for i, line in enumerate(fh, 1):
                if pat.search(line):
                    bad.append(f"{f}:{i}: {line.strip()[:100]}")
    except (UnicodeDecodeError, IsADirectoryError, FileNotFoundError):
        continue
print("\n".join(bad) or "no emojis found")
sys.exit(1 if bad else 0)
EOF
}

check_python_compile() {
    "$PY" -m compileall -q client-py/hypercache scripts
}

check_shell_syntax() {
    local f rc=0
    for f in build.sh install.sh scripts/*.sh; do
        [ -f "$f" ] || continue
        bash -n "$f" || rc=1
    done
    return $rc
}

check_yaml() {
    "$PY" - <<'EOF'
import glob, sys
try:
    import yaml
except ImportError:
    print("PyYAML not installed; skipping YAML syntax check")
    sys.exit(0)
files = glob.glob(".github/workflows/*.yml") + glob.glob("deploy/**/*.yaml", recursive=True) + ["docker-compose.yml"]
for f in files:
    try:
        with open(f) as fh:
            docs = list(yaml.safe_load_all(fh))
        print(f"{f}: {len(docs)} document(s)")
    except FileNotFoundError:
        pass
EOF
}

check_toml() {
    "$PY" - <<'EOF'
import glob, sys
try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib
    except ImportError:
        print("tomllib unavailable (Python < 3.11); skipping")
        sys.exit(0)
for f in glob.glob("configs/*.toml") + ["control-plane/Cargo.toml", "client-py/pyproject.toml", "control-plane/rustfmt.toml"]:
    with open(f, "rb") as fh:
        tomllib.load(fh)
    print(f"{f}: ok")
EOF
}

check_config_keys_load() {
    # Every shipped config must load in the real binary (catches renamed/removed keys)
    local f
    for f in configs/hypercache-agent.toml configs/hypercache-controller.toml; do
        "$RELEASE_DIR/hypercache" -c "$f" check --json >/dev/null 2>"$LOG_DIR/check_stderr.txt" || {
            if grep -qiE "parse|unknown field|invalid type|TOML" "$LOG_DIR/check_stderr.txt"; then
                cat "$LOG_DIR/check_stderr.txt"; return 1
            fi
        }
    done
}

stage_static() {
    section "static"
    step "version numbers agree (Cargo, pyproject, __init__, release.yml)" check_versions
    step "no hardcoded version strings in Rust sources" check_no_hardcoded_versions
    step "zero-emoji policy (AGENTS.md rule 1)" check_emoji_policy
    step "cargo fmt --check" cargo fmt --manifest-path "$CARGO_MANIFEST" -- --check
    step "cargo clippy --all-targets -D warnings (Cargo.lock locked)" \
        cargo clippy --locked --manifest-path "$CARGO_MANIFEST" --all-targets -- -D warnings
    step "python byte-compile (client + scripts)" check_python_compile
    step "shell script syntax (bash -n)" check_shell_syntax
    step "YAML syntax (workflows, K8s manifests, compose)" check_yaml
    step "TOML syntax (configs, Cargo, pyproject)" check_toml
}

# ------------------------------------------------------------------------------
# rust
# ------------------------------------------------------------------------------

check_binaries() {
    local want
    want=$(grep -m1 '^version' control-plane/Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
    "$RELEASE_DIR/hypercache" --version | grep -q "$want" || { echo "hypercache --version mismatch"; return 1; }
    "$RELEASE_DIR/hypercache-admin" --version | grep -q "$want" || { echo "hypercache-admin --version mismatch"; return 1; }
    local cmd
    for cmd in start bench inspect check ls rm refresh preload purge status; do
        "$RELEASE_DIR/hypercache" "$cmd" --help >/dev/null || { echo "hypercache $cmd --help failed"; return 1; }
    done
    for cmd in controller nodes preload preload-batch evict purge status; do
        "$RELEASE_DIR/hypercache-admin" "$cmd" --help >/dev/null || { echo "hypercache-admin $cmd --help failed"; return 1; }
    done
}

check_missing_config_is_error() {
    # Starting without -c must fail fast with a clear message, never fall back silently
    ! "$RELEASE_DIR/hypercache" start </dev/null
}

stage_rust() {
    section "rust"
    step "cargo test (unit + integration, locked)" cargo test --locked --manifest-path "$CARGO_MANIFEST"
    step "cargo build --release" cargo build --locked --release --manifest-path "$CARGO_MANIFEST"
    step "release binaries: --version and every subcommand --help" check_binaries
    step "daemon refuses to start without a config file" check_missing_config_is_error
    step "shipped configs load in the release binary" check_config_keys_load
}

# ------------------------------------------------------------------------------
# python
# ------------------------------------------------------------------------------

check_wheel() {
    if ! "$PY" -c "import build" 2>/dev/null; then
        echo "python 'build' package not installed; skipping wheel build"
        return 0
    fi
    local out
    out=$(mktemp -d)
    CLEANUP_CMDS+=("rm -rf '$out'")
    "$PY" -m build --wheel client-py --outdir "$out" && ls "$out"/*.whl
}

stage_python() {
    section "python"
    # Test the working tree, not a previously installed copy of the package
    step "pytest client-py/tests" env PYTHONPATH="$REPO_ROOT/client-py${PYTHONPATH:+:$PYTHONPATH}" \
        "$PY" -m pytest -q client-py/tests -W error::pytest.PytestUnhandledThreadExceptionWarning
    step "client-py wheel builds" check_wheel
}

# ------------------------------------------------------------------------------
# e2e
# ------------------------------------------------------------------------------

stage_e2e() {
    section "e2e (real binaries on file-backed devices)"
    if [ ! -x "$RELEASE_DIR/hypercache" ]; then
        step "cargo build --release (for e2e)" cargo build --locked --release --manifest-path "$CARGO_MANIFEST"
    fi
    local keep=()
    [ "${KEEP_E2E:-0}" = "1" ] && keep=(--keep)
    step "e2e: single-node daemon scenario" \
        "$PY" scripts/e2e/hypercache_e2e.py --bin-dir "$RELEASE_DIR" "${keep[@]+"${keep[@]}"}" daemon
    step "e2e: controller + agents fleet scenario" \
        "$PY" scripts/e2e/hypercache_e2e.py --bin-dir "$RELEASE_DIR" "${keep[@]+"${keep[@]}"}" fleet
}

# ------------------------------------------------------------------------------
# s3
# ------------------------------------------------------------------------------

start_s3_container() {
    local name="hc-verify-s3-$$" port
    port=$("$PY" -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])')
    docker run -d --rm --name "$name" -p "127.0.0.1:$port:7070" \
        -e ROOT_ACCESS_KEY="$S3_ACCESS" -e ROOT_SECRET_KEY="$S3_SECRET" \
        "$S3_IMAGE" posix /tmp >/dev/null || return 1
    CLEANUP_CMDS+=("docker rm -f '$name'")
    export HYPERCACHE_TEST_S3_ENDPOINT="http://127.0.0.1:$port"
    export HYPERCACHE_TEST_S3_ACCESS_KEY="$S3_ACCESS"
    export HYPERCACHE_TEST_S3_SECRET_KEY="$S3_SECRET"
    export HYPERCACHE_TEST_S3_BUCKET="$S3_BUCKET"
    local i
    for i in $(seq 1 60); do
        curl -s -o /dev/null "$HYPERCACHE_TEST_S3_ENDPOINT/" && return 0
        sleep 0.5
    done
    echo "S3 container did not become ready"; docker logs "$name" 2>&1 | tail -20
    return 1
}

create_bucket() {
    "$PY" - <<'EOF'
import os, boto3
from botocore.config import Config
s3 = boto3.client("s3", endpoint_url=os.environ["HYPERCACHE_TEST_S3_ENDPOINT"],
                  aws_access_key_id=os.environ["HYPERCACHE_TEST_S3_ACCESS_KEY"],
                  aws_secret_access_key=os.environ["HYPERCACHE_TEST_S3_SECRET_KEY"],
                  region_name="us-east-1", config=Config(s3={"addressing_style": "path"}))
bucket = os.environ["HYPERCACHE_TEST_S3_BUCKET"]
try:
    s3.head_bucket(Bucket=bucket)
except Exception:
    s3.create_bucket(Bucket=bucket)
print("bucket ready:", bucket)
EOF
}

stage_s3() {
    section "s3 (live S3-compatible endpoint)"
    if [ -z "${HYPERCACHE_TEST_S3_ENDPOINT:-}" ]; then
        if ! docker_ok; then
            skip "live S3 tests" "no Docker and HYPERCACHE_TEST_S3_ENDPOINT unset"
            return
        fi
        step "start throwaway S3 server ($S3_IMAGE)" start_s3_container
        [ -z "${HYPERCACHE_TEST_S3_ENDPOINT:-}" ] && return
    fi
    if ! "$PY" -c "import boto3" 2>/dev/null; then
        step "boto3 installed (pip install boto3)" false
        return
    fi
    step "create test bucket" create_bucket
    step "rust live S3: SigV4, >1000-key pagination, multipart, ranged GET" \
        cargo test --locked --manifest-path "$CARGO_MANIFEST" --test s3_live -- --ignored --test-threads=2
    if [ ! -x "$RELEASE_DIR/hypercache" ]; then
        step "cargo build --release (for s3 e2e)" cargo build --locked --release --manifest-path "$CARGO_MANIFEST"
    fi
    step "e2e: daemon against live S3" "$PY" scripts/e2e/hypercache_e2e.py --bin-dir "$RELEASE_DIR" s3 \
        --s3-endpoint "$HYPERCACHE_TEST_S3_ENDPOINT" --s3-access-key "$HYPERCACHE_TEST_S3_ACCESS_KEY" \
        --s3-secret-key "$HYPERCACHE_TEST_S3_SECRET_KEY" --s3-bucket "$HYPERCACHE_TEST_S3_BUCKET"
}

# ------------------------------------------------------------------------------
# linux (container)
# ------------------------------------------------------------------------------

run_linux_container() {
    local net="hc-verify-net-$$" s3name="hc-verify-s3l-$$"
    docker network create "$net" >/dev/null || return 1
    CLEANUP_CMDS+=("docker network rm '$net'")
    docker run -d --rm --name "$s3name" --network "$net" \
        -e ROOT_ACCESS_KEY="$S3_ACCESS" -e ROOT_SECRET_KEY="$S3_SECRET" "$S3_IMAGE" posix /tmp >/dev/null || return 1
    CLEANUP_CMDS=("docker rm -f '$s3name'" "${CLEANUP_CMDS[@]}")

    # Copy the working tree (without build outputs) so the container never writes to the host repo.
    # Named volumes cache the cargo registry and target dir between runs.
    docker run --rm --network "$net" \
        -v "$REPO_ROOT:/src:ro" \
        -v hc-verify-cargo-registry:/usr/local/cargo/registry \
        -v hc-verify-target:/work/control-plane/target \
        -e HYPERCACHE_TEST_S3_ENDPOINT="http://$s3name:7070" \
        -e HYPERCACHE_TEST_S3_ACCESS_KEY="$S3_ACCESS" \
        -e HYPERCACHE_TEST_S3_SECRET_KEY="$S3_SECRET" \
        -e HYPERCACHE_TEST_S3_BUCKET="$S3_BUCKET" \
        -e CARGO_TERM_COLOR=never \
        "$RUST_IMAGE" bash -euo pipefail -c '
            export DEBIAN_FRONTEND=noninteractive
            apt-get update -qq >/dev/null
            apt-get install -y -qq python3 python3-venv python3-yaml lsof curl git rsync >/dev/null
            rustup component add rustfmt clippy >/dev/null 2>&1
            mkdir -p /work
            rsync -a --exclude control-plane/target --exclude .venv --exclude "__pycache__" \
                --exclude "*.raw" --exclude client-py/tests/.tmp /src/ /work/
            cd /work
            git config --global --add safe.directory /work
            python3 -m venv /venv
            /venv/bin/pip install -q --upgrade pip
            /venv/bin/pip install -q "numpy>=1.22" "fsspec>=2023.1.0" pytest boto3 safetensors pyyaml build
            PYTHON=/venv/bin/python scripts/verify_release.sh --in-container --stages static,rust,python,e2e,s3
        '
}

stage_linux() {
    section "linux (clean $RUST_IMAGE container)"
    if [ "$IN_CONTAINER" -eq 1 ]; then
        return
    fi
    if ! docker_ok; then
        skip "linux container run" "Docker unavailable or --no-docker"
        return
    fi
    step "full verification inside $RUST_IMAGE (O_DIRECT, sysfs, fdatasync paths)" run_linux_container
}

# ------------------------------------------------------------------------------
# image (release Dockerfile)
# ------------------------------------------------------------------------------

IMAGE_TAG="hypercache:verify-$$"

build_image() {
    docker build -t "$IMAGE_TAG" "$REPO_ROOT" || return 1
    CLEANUP_CMDS+=("docker rmi -f '$IMAGE_TAG'")
}

smoke_image() {
    local want
    want=$(grep -m1 '^version' control-plane/Cargo.toml | sed 's/.*"\(.*\)".*/\1/')
    docker run --rm "$IMAGE_TAG" --version | grep -q "$want" || { echo "image hypercache --version mismatch"; return 1; }
    docker run --rm --entrypoint hypercache-admin "$IMAGE_TAG" --version | grep -q "$want" || return 1
    docker run --rm --entrypoint python3 "$IMAGE_TAG" -c \
        "import hypercache, fsspec; assert hypercache.__version__ == '$want', hypercache.__version__; fsspec.filesystem('hypercache')" || return 1
    # The default agent config must load, and the daemon must come up on a mock device
    docker run --rm --entrypoint bash "$IMAGE_TAG" -euc '
        truncate -s 64M /tmp/dev0.raw
        cat > /tmp/agent.toml <<EOT
device_discovery_mode = "manual"
block_devices = ["/tmp/dev0.raw"]
metadata_persistence_path = "/tmp/meta/metadata.json"
EOT
        hypercache -c /tmp/agent.toml start >/tmp/daemon.log 2>&1 &
        pid=$!
        for i in $(seq 1 50); do
            if hypercache status >/dev/null 2>&1; then
                hypercache ls --json >/dev/null
                kill -INT "$pid"; wait "$pid" || true
                exit 0
            fi
            sleep 0.2
        done
        cat /tmp/daemon.log; exit 1
    '
}

stage_image() {
    section "image (release Dockerfile)"
    if [ "$IN_CONTAINER" -eq 1 ]; then
        return
    fi
    if ! docker_ok; then
        skip "docker image build" "Docker unavailable or --no-docker"
        return
    fi
    local before=$FAILED
    FAILED=0
    step "docker build (release Dockerfile, --locked)" build_image
    if [ "$FAILED" -eq 0 ]; then
        step "image smoke: versions, configs, python client, daemon start" smoke_image
        step "container <-> host access: client_gid, O_DIRECT, RAM tier, SIGTERM, warm boot" \
            "$REPO_ROOT/scripts/e2e/container_host_access.sh" "$IMAGE_TAG"
    else
        skip "image smoke" "image build failed"
    fi
    [ "$before" -ne 0 ] && FAILED=1
}

stage_k8s() {
    section "k8s (minikube)"
    if [ "$IN_CONTAINER" -eq 1 ]; then
        return
    fi
    if ! docker_ok || ! command -v minikube >/dev/null 2>&1 || ! command -v kubectl >/dev/null 2>&1; then
        skip "kubernetes smoke test" "needs Docker, minikube and kubectl"
        return
    fi
    if ! docker image inspect "$IMAGE_TAG" >/dev/null 2>&1; then
        step "docker build (release Dockerfile, --locked)" build_image || return
    fi
    step "kubernetes manifest on minikube: probes, registration, non-root pod, SIGTERM, warm boot" \
        "$REPO_ROOT/scripts/e2e/k8s_smoke.sh" "$IMAGE_TAG"
}

# ------------------------------------------------------------------------------

echo "HyperCache release verification"
echo "  repo:    $REPO_ROOT"
echo "  stages:  $STAGES"
echo "  python:  $PY ($("$PY" --version 2>&1))"
echo "  host:    $(uname -sm)"
echo "  logs:    $LOG_DIR"

has_stage static && stage_static
has_stage rust && stage_rust
has_stage python && stage_python
has_stage e2e && stage_e2e
has_stage s3 && stage_s3
has_stage linux && stage_linux
has_stage image && stage_image
has_stage k8s && stage_k8s

echo
echo "=============================================================================="
echo "Verification summary"
echo "=============================================================================="
printf '%s\n' "${RESULTS[@]+"${RESULTS[@]}"}"
echo "------------------------------------------------------------------------------"
if [ "$FAILED" -ne 0 ]; then
    echo "RESULT: FAILED"
    exit 1
fi
echo "RESULT: PASSED"
