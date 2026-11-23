#!/usr/bin/env bash
# Builds the LMCache connector's PyPI files: manylinux_2_28 x86_64 wheels for CPython 3.10 to 3.13
# and an sdist. Used by CI (to catch a broken build before a release) and by release.yml (to
# publish). Needs Docker. Builds only; uploading is release.yml's job.
#
#   scripts/build_connector_wheels.sh [out-dir]     # default: dist-connector
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-dist-connector}"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

# Pinned so a release builds the same way the last CI run did
IMAGE="quay.io/pypa/manylinux_2_28_x86_64@sha256:55a36963bc0504cdf004011001c1072306402f0b8b83be5a5eba7068f93eec53"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
# A clean copy, so a local target/ directory never ends up in the build
tar -C "$REPO_ROOT" -c --exclude=target lmcache-connector | tar -x -C "$WORK"

docker run --rm -u "$(id -u):$(id -g)" -e HOME=/tmp/home \
    -v "$WORK:/w" -v "$OUT:/out" -w /w/lmcache-connector "$IMAGE" bash -euc '
mkdir -p "$HOME"
curl -sSf https://sh.rustup.rs | sh -s -- -y -q --profile minimal >/dev/null
. "$HOME/.cargo/env"
/opt/python/cp312-cp312/bin/python -m venv /tmp/maturin
/tmp/maturin/bin/pip install -q "maturin>=1.8,<2"
interpreters=()
for v in cp310-cp310 cp311-cp311 cp312-cp312 cp313-cp313; do
    interpreters+=(-i "/opt/python/$v/bin/python")
done
/tmp/maturin/bin/maturin build --release --locked --compatibility manylinux_2_28 --out /out "${interpreters[@]}"
/tmp/maturin/bin/maturin sdist --out /out
'

count=$(ls "$OUT"/*.whl | wc -l)
[ "$count" -eq 4 ] || { echo "expected 4 wheels, got $count"; exit 1; }
ls -l "$OUT"
