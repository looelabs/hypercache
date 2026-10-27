#!/usr/bin/env bash
# ==============================================================================
# HyperCache Complete Environment Bootstrap & Installation Script
# Supports: Ubuntu, Debian, RHEL, CentOS, Rocky Linux, Fedora, Arch, macOS
# Usage:
#   curl -fsSL https://hypercache.sh/install.sh | bash
#   OR inside a cloned repo:
#   ./install.sh
# ==============================================================================

set -euo pipefail

REPO_URL="${HYPERCACHE_REPO_URL:-https://github.com/looelabs/hypercache.git}"
INSTALL_DIR="${HYPERCACHE_INSTALL_DIR:-hypercache}"

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
if [ "${EUID:-$(id -u)}" -ne 0 ]; then
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
        echo -e "${YELLOW}  -> Unknown package manager. Please ensure gcc, git, pkg-config, openssl, and python3 are installed.${NC}"
    fi
else
    echo -e "  -> All base build packages are available."
fi

# ------------------------------------------------------------------------------
# 2. Ensure HyperCache Repository is Present (Supports curl | bash)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}[2/6] Resolving HyperCache source repository...${NC}"

if [ -f "control-plane/Cargo.toml" ] && [ -d "client-py" ]; then
    echo -e "  -> Running inside existing HyperCache repository: $(pwd)"
else
    if [ -d "$INSTALL_DIR/.git" ] && [ -f "$INSTALL_DIR/control-plane/Cargo.toml" ]; then
        echo -e "  -> Found existing './$INSTALL_DIR' directory. Pulling latest updates..."
        cd "$INSTALL_DIR"
        git pull --ff-only || true
    else
        echo -e "  -> Cloning HyperCache from $REPO_URL into ./$INSTALL_DIR..."
        git clone "$REPO_URL" "$INSTALL_DIR"
        cd "$INSTALL_DIR"
    fi
    echo -e "  -> Working directory: $(pwd)"
fi

# ------------------------------------------------------------------------------
# 3. Install Rust Toolchain (if not present)
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}[3/6] Checking Rust toolchain...${NC}"

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
# 4. Setup Python Virtual Environment & Install hypercache Python Client
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}[4/6] Setting up Python virtual environment (.venv) & client-py...${NC}"

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

echo -e "  -> Installing/updating HyperCache Python client package (client-py)..."
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

echo -e "  -> Python virtual environment (.venv) and hypercache client ready."

# ------------------------------------------------------------------------------
# 5. Compile Rust Control Plane & Admin Binaries
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}[5/6] Building HyperCache Release Binaries (Rust)...${NC}"
cargo build --release --manifest-path control-plane/Cargo.toml

cp control-plane/target/release/hypercache ./hypercache
cp control-plane/target/release/hypercache-admin ./hypercache-admin
chmod +x ./hypercache ./hypercache-admin
echo -e "  -> ${GREEN}Compiled release binaries:${NC} ./hypercache and ./hypercache-admin"

# Optionally install to /usr/local/bin
if [ -w "/usr/local/bin" ] || [ -n "$SUDO" ]; then
    $SUDO cp ./hypercache /usr/local/bin/hypercache 2>/dev/null || true
    $SUDO cp ./hypercache-admin /usr/local/bin/hypercache-admin 2>/dev/null || true
    echo -e "  -> Installed globally to ${GREEN}/usr/local/bin/hypercache${NC} and ${GREEN}/usr/local/bin/hypercache-admin${NC}"
fi

# ------------------------------------------------------------------------------
# 6. Storage Hardware & NVMe Discovery Check
# ------------------------------------------------------------------------------
echo -e "\n${BOLD}${GREEN}[6/6] Inspecting host storage hardware & NVMe drives...${NC}"

REAL_DRIVES_FOUND=false
if [ -d "/sys/class/block" ]; then
    echo -e "  -> Scanning for attached NVMe (/dev/nvme*), VirtIO (/dev/vd*), and SSD (/dev/sd*) devices..."
    for dev_path in /sys/class/block/*; do
        dev_name=$(basename "$dev_path")
        if [[ "$dev_name" =~ ^(nvme|vd|xvd|sd) ]]; then
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
    echo -e "  -> ${BOLD}${GREEN}Local block devices detected on this host.${NC}"
    echo -e "     HyperCache will automatically discover unmounted raw NVMe/SSD drives while excluding the OS root disk."
else
    echo -e "  -> ${YELLOW}No unmounted raw block drives detected yet.${NC}"
    echo -e "     Ensure raw NVMe drives are attached (without a filesystem) or set storage_backing_mode = \"filesystem\" in configs/hypercache-agent.toml."
fi

echo -e "\n${BOLD}${GREEN}=================================================================${NC}"
echo -e "${BOLD}${GREEN}         HYPERCACHE INSTALLATION COMPLETE                        ${NC}"
echo -e "${BOLD}${GREEN}=================================================================${NC}"
echo -e ""
echo -e "Working Directory: ${BOLD}$(pwd)${NC}"
echo -e ""
echo -e "  ${BOLD}1. Run Pre-Flight Host Diagnostics:${NC}"
echo -e "     ${GREEN}./hypercache -c configs/hypercache-agent.toml check${NC}"
echo -e ""
echo -e "  ${BOLD}2. Start Worker Node Daemon:${NC}"
echo -e "     ${GREEN}./hypercache -c configs/hypercache-agent.toml start${NC}"
echo -e ""
echo -e "  ${BOLD}3. Inspect Discovered NVMe Drives & NUMA Topology:${NC}"
echo -e "     ${GREEN}./hypercache inspect${NC}"
echo -e ""
echo -e "  ${BOLD}4. Run NVMe Hardware DMA Benchmark:${NC}"
echo -e "     ${GREEN}./hypercache bench --tier nvme --size-mb 5000${NC}"
echo -e ""
echo -e "  ${BOLD}5. Install Python Client into Another Conda / Virtualenv (Optional):${NC}"
echo -e "     ${GREEN}pip install -e $(pwd)/client-py${NC}"
echo -e ""
