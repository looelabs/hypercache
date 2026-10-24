#!/usr/bin/env bash
# ==============================================================================
# HyperCache Fast Native Binary Builder
# ==============================================================================
set -euo pipefail

# Ensure Cargo/Rust is on PATH
if [ -f "$HOME/.cargo/env" ]; then
    source "$HOME/.cargo/env"
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo "Error: cargo not found. Installing Rust toolchain..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    source "$HOME/.cargo/env"
fi

echo "Building native release binary for this machine..."
cargo build --release --manifest-path control-plane/Cargo.toml

cp control-plane/target/release/hypercache ./hypercache
cp control-plane/target/release/hypercache-admin ./hypercache-admin
chmod +x ./hypercache ./hypercache-admin

echo ""
echo "================================================================="
echo "  HyperCache binaries built successfully:"
echo "    - Worker Node Agent:     ./hypercache"
echo "    - Fleet Admin Suite:     ./hypercache-admin"
echo "================================================================="
echo "Run Agent:      ./hypercache start"
echo "Run Controller: ./hypercache-admin controller --port 9120"
