#!/usr/bin/env bash
# ==============================================================================
# HyperCache Complete Environment Bootstrap & Installation Script
# Supports: Ubuntu, Debian, RHEL, CentOS, Rocky Linux, Fedora, Arch, macOS
# ==============================================================================

set -euo pipefail

# ANSI Color Codes
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BOLD='\033[1m'
NC='\033[0m' # No Color

echo -e "${BOLD}${BLUE}=================================================================${NC}"
echo -e "${BOLD}${BLUE}        HYPERCACHE: AUTOMATED INSTALLATION & SETUP               ${NC}"
echo -e "${BOLD}${BLUE}=================================================================${NC}"

# Detect Sudo
SUDO=""
if [ "$EUID" -ne 0 ]; then
    if command -v sudo >/dev/null 2>&1; then
        SUDO="sudo"
    else
        echo -e "${YELLOW}[!] Not running as root and sudo not found. Attempting user-space installation...${NC}"
    fi
fi

# ------------------------------------------------------------------------------
# 1. Install System Dependencies & C Compilers (if missing)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}[1/6] Checking system packages & build tools...${NC}"

NEED_SYSTEM_PKGS=false
if ! command -v curl >/dev/null 2>&1 || \
   ! command -v git >/dev/null 2>&1 || \
   ! command -v cc >/dev/null 2>&1 && ! command -v gcc >/dev/null 2>&1 || \
   ! command -v pkg-config >/dev/null 2>&1 || \
   ! command -v python3 >/dev/null 2>&1; then
    NEED_SYSTEM_PKGS=true
fi

# On Debian/Ubuntu, ensure python3-venv, libssl-dev, and pkg-config are installed
if command -v apt-get >/dev/null 2>&1; then
    if ! dpkg -s python3-venv >/dev/null 2>&1 || ! dpkg -s libssl-dev >/dev/null 2>&1 || ! dpkg -s pkg-config >/dev/null 2>&1; then
        NEED_SYSTEM_PKGS=true
    fi
fi

if [ "$NEED_SYSTEM_PKGS" = true ]; then
    echo -e "  -> Installing missing base build packages..."
    if command -v apt-get >/dev/null 2>&1; then
        echo -e "  -> Detected Debian/Ubuntu system."
        $SUDO apt-get update -y
        $SUDO apt-get install -y build-essential curl git pkg-config libssl-dev python3 python3-pip python3-venv python3-dev
    elif command -v dnf >/dev/null 2>&1; then
        echo -e "  -> Detected RHEL/Fedora/CentOS/Rocky system."
        $SUDO dnf install -y gcc gcc-c++ make curl git pkgconfig openssl-devel python3 python3-pip python3-devel
    elif command -v yum >/dev/null 2>&1; then
        echo -e "  -> Detected legacy YUM system."
        $SUDO yum install -y gcc gcc-c++ make curl git pkgconfig openssl-devel python3 python3-pip python3-devel
    elif command -v pacman >/dev/null 2>&1; then
        echo -e "  -> Detected Arch Linux."
        $SUDO pacman -Sy --noconfirm base-devel curl git pkgconf openssl python python-pip
    elif command -v brew >/dev/null 2>&1; then
        echo -e "  -> Detected macOS with Homebrew."
        brew install pkg-config openssl python3
    else
        echo -e "${YELLOW}  -> Unknown package manager. Please ensure gcc, pkg-config, openssl, and python3 are installed.${NC}"
    fi
else
    echo -e "  -> All base build packages are available."
fi

# ------------------------------------------------------------------------------
# 2. Install Rust Toolchain (if not present)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}[2/5] Checking Rust toolchain...${NC}"

if [ -f "$HOME/.cargo/env" ]; then
    # shellcheck source=/dev/null
    source "$HOME/.cargo/env"
fi

if ! command -v cargo >/dev/null 2>&1; then
    echo -e "  -> Rust not found in current PATH. Installing via rustup..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
    export PATH="$HOME/.cargo/bin:$PATH"
    if [ -f "$HOME/.cargo/env" ]; then
        # shellcheck source=/dev/null
        source "$HOME/.cargo/env"
    fi
fi

# Ensure ~/.bashrc and ~/.profile have cargo env
if [ -f "$HOME/.bashrc" ] && ! grep -q ".cargo/env" "$HOME/.bashrc" 2>/dev/null; then
    echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> "$HOME/.bashrc"
    echo '[ -f "$HOME/.cargo/env" ] && source "$HOME/.cargo/env"' >> "$HOME/.bashrc"
fi

# Symlink cargo and rustc into /usr/local/bin so it is immediately available in all shells
if [ -d "$HOME/.cargo/bin" ]; then
    $SUDO ln -sf "$HOME/.cargo/bin/cargo" /usr/local/bin/cargo 2>/dev/null || true
    $SUDO ln -sf "$HOME/.cargo/bin/rustc" /usr/local/bin/rustc 2>/dev/null || true
    $SUDO ln -sf "$HOME/.cargo/bin/rustup" /usr/local/bin/rustup 2>/dev/null || true
fi

echo -e "  -> Rust toolchain ready: $(cargo --version 2>/dev/null || echo "$HOME/.cargo/bin/cargo")"

# ------------------------------------------------------------------------------
# 3. Setup Python Virtual Environment (with self-healing and sanity check)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}[3/5] Setting up Python virtual environment (.venv)...${NC}"

PYTHON_BIN="python3"
if ! command -v python3 >/dev/null 2>&1; then
    if command -v python >/dev/null 2>&1; then
        PYTHON_BIN="python"
    else
        echo -e "${RED}[ERROR] Python 3 is required but not found.${NC}"
        exit 1
    fi
fi

# Validate existing .venv directory or recreate if corrupt/copied from another host
REBUILD_VENV=false
if [ -d ".venv" ]; then
    if [ ! -f ".venv/bin/python" ] || ! .venv/bin/python --version >/dev/null 2>&1; then
        echo -e "  -> Existing .venv is invalid or from a different architecture. Rebuilding..."
        rm -rf .venv
        REBUILD_VENV=true
    fi
else
    REBUILD_VENV=true
fi

if [ "$REBUILD_VENV" = true ]; then
    echo -e "  -> Creating virtualenv at .venv using $PYTHON_BIN..."
    if ! $PYTHON_BIN -m venv .venv 2>/dev/null; then
        echo -e "  -> Standard venv failed. Attempting with system python3-venv..."
        if command -v apt-get >/dev/null 2>&1; then
            $SUDO apt-get update -y && $SUDO apt-get install -y python3-venv python3-pip
        fi
        $PYTHON_BIN -m venv .venv
    fi
fi

VENV_PY=".venv/bin/python"

# Ensure pip is functional inside the virtual environment
if ! $VENV_PY -m pip --version >/dev/null 2>&1; then
    echo -e "  -> Bootstrapping pip inside .venv..."
    $VENV_PY -m ensurepip --upgrade 2>/dev/null || curl -sS https://bootstrap.pypa.io/get-pip.py | $VENV_PY
fi

echo -e "  -> Installing/updating HyperCache client package and dependencies..."
$VENV_PY -m pip install -q --upgrade pip setuptools wheel numpy
$VENV_PY -m pip install -q -e "./client-py[dev]"

# Detect NVIDIA GPU and auto-install GPUDirect Storage (KvikIO) acceleration
if command -v nvidia-smi >/dev/null 2>&1; then
    echo -e "  -> ${BOLD}${GREEN}NVIDIA GPU detected (${NC}$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1 2>/dev/null || echo "NVIDIA CUDA GPU")${BOLD}${GREEN})!${NC}"
    echo -e "  -> Installing NVIDIA GPUDirect Storage (kvikio-cu12) & CuPy for zero-copy DMA to GPU VRAM..."
    $VENV_PY -m pip install -q --extra-index-url https://pypi.nvidia.com kvikio-cu12 cupy-cuda12x numpy || {
        echo -e "${YELLOW}  -> Optional kvikio installation finished. DirectBlockReader fallback is ready.${NC}"
    }

    # Attempt to load nvidia-fs kernel driver for GPUDirect Storage if available
    if command -v modprobe >/dev/null 2>&1; then
        $SUDO modprobe nvidia-fs 2>/dev/null || true
    fi
fi

echo -e "  -> Python virtual environment (.venv) and dependencies ready."

# ------------------------------------------------------------------------------
# 4. Compile Rust Control Plane Daemon
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}[4/5] Building HyperCache Control Plane Daemon (Rust Release)...${NC}"
cargo build --release --manifest-path control-plane/Cargo.toml

echo -e "  -> Control plane binary compiled successfully."

# Copy hypercache binary to project root
cp control-plane/target/release/hypercache ./hypercache
chmod +x ./hypercache
echo -e "  -> ${GREEN}Created standalone binary:${NC} ./hypercache"

# Optionally install to /usr/local/bin
if [ -w "/usr/local/bin" ] || [ -n "$SUDO" ]; then
    $SUDO cp ./hypercache /usr/local/bin/hypercache 2>/dev/null || true
    echo -e "  -> Installed globally to ${GREEN}/usr/local/bin/hypercache${NC}"
fi

# ------------------------------------------------------------------------------
# 5. Storage Hardware & SSD Discovery Check
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}[5/5] Inspecting host storage hardware & SSDs...${NC}"

REAL_DRIVES_FOUND=false
if [ -d "/sys/class/block" ]; then
    echo -e "  -> Scanning for attached NVMe, VirtIO (/dev/vd*), and SSD devices..."
    for dev_path in /sys/class/block/*; do
        dev_name=$(basename "$dev_path")
        # Check matching prefixes: nvme, vd, xvd, sd
        if [[ "$dev_name" =~ ^(nvme|vd|xvd|sd) ]]; then
            # Exclude loop, ram
            if [ -f "$dev_path/size" ]; then
                sectors=$(cat "$dev_path/size" 2>/dev/null || echo 0)
                if [ "$sectors" -gt 0 ]; then
                    size_bytes=$((sectors * 512))
                    if [ "$size_bytes" -gt 1073741824 ]; then # > 1 GB
                        size_gb=$(awk "BEGIN {printf \"%.2f\", $size_bytes / 1073741824}")
                        echo -e "  -> ${GREEN}Discovered block device:${NC} /dev/$dev_name (${size_gb} GiB)"
                        REAL_DRIVES_FOUND=true
                    fi
                fi
            fi
        fi
    done
fi

if [ "$REAL_DRIVES_FOUND" = true ]; then
    echo -e "  -> ${BOLD}${GREEN}Real hardware/virtual SSDs detected on this host!${NC}"
    echo -e "     When you start the HyperCache daemon, it will automatically target"
    echo -e "     your unmounted real SSDs (e.g. /dev/vdc, /dev/vdd, /dev/nvme*) for zero-copy striping."
else
    echo -e "  -> No physical/virtual block drives found (dev/laptop environment). Provisioning fallback mock block devices..."
    chmod +x scripts/setup_mock_devices.sh
    ./scripts/setup_mock_devices.sh
fi

echo -e "\n${BOLD}${GREEN}=================================================================${NC}"
echo -e "${BOLD}${GREEN}         HYPERCACHE SINGLE BINARY READY!                         ${NC}"
echo -e "${BOLD}${GREEN}=================================================================${NC}"
echo -e ""
echo -e "You can now run HyperCache entirely through the single ${BOLD}./hypercache${NC} executable:"
echo -e ""
echo -e "  ${BOLD}1. Start Cache Daemon & Prometheus Exporter:${NC}"
echo -e "     ${GREEN}./hypercache start${NC}"
echo -e "     ${YELLOW}• Prometheus Metrics: http://localhost:9110/metrics (Dedicated port to avoid collisions)${NC}"
echo -e ""
echo -e "  ${BOLD}2. Run End-to-End Verification Benchmark:${NC}"
echo -e "     ${GREEN}./hypercache bench --size-mb 5000${NC}"
echo -e ""
echo -e "  ${BOLD}3. Inspect Storage Pool & Active Disks:${NC}"
echo -e "     ${GREEN}./hypercache inspect${NC}"
echo -e ""
echo -e "  ${BOLD}4. Check Live Daemon Status:${NC}"
echo -e "     ${GREEN}./hypercache status${NC}"
echo -e ""
