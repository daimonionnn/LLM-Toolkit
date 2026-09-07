#!/bin/bash
#
# Install ROCm 7.2 on the host for baremetal llama.cpp inference on Vega 8.
#
# Neither Ubuntu 25.10 (questing) nor 26.04 (resolute) is officially supported
# by AMD — this script pins to Ubuntu 24.04 (noble) packages, which are
# ABI-compatible and confirmed working on questing / kernel 6.17 and on
# resolute / kernel 7.0. On 26.04 the noble deps still resolve because
# libelf1t64 provides libelf1 and libncurses-dev provides libtinfo-dev.
#
# The KFD module (/dev/kfd) is already functional — proven by Docker ROCm
# working. We only need the ROCm userspace stack here.
#
# After this script completes, run:
#   ./build/build-llamacpp-rocm7-baremetal.sh
#   (applies gfx900 tensile backport and builds llama.cpp for Vega 8)
#
# Usage:
#   sudo bash setup/install-rocm7-host.sh
#   # or:
#   bash setup/install-rocm7-host.sh          # will sudo as needed
#
# Idempotent — safe to run more than once.

set -euo pipefail

ROCM_VERSION="7.2"
ROCM_REPO_BASE="https://repo.radeon.com/rocm/apt/${ROCM_VERSION}"
AMDGPU_REPO_BASE="https://repo.radeon.com/amdgpu/6.3.4/ubuntu"
UBUNTU_CODENAME="noble"        # Use 24.04 packages on Ubuntu 25.10 / 26.04
ROCM_KEYRING_URL="https://repo.radeon.com/rocm/rocm.gpg.key"
ROCM_KEYRING_PATH="/etc/apt/keyrings/rocm.gpg"

# Minimal package set for building and running llama.cpp HIP/ROCm
# -dev packages are needed at build time; the non-dev ones at runtime.
ROCM_PACKAGES=(
    "rocm-hip-runtime"
    "hip-runtime-amd"
    "hipcc"
    "hip-dev"
    "hsa-rocr-dev"
    "rocblas"
    "rocblas-dev"
    "hipblas"
    "hipblas-dev"
    "rocsolver"
    "rocsolver-dev"
    "rocminfo"
    "rocm-smi-lib"
)

# ─────────────────────────────────────────────────────────────────────────────

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
    SUDO="sudo"
fi

# Under `sudo bash setup/install-rocm7-host.sh`, $USER is root — adding root to
# render/video would leave the actual human without /dev/kfd access. Prefer
# SUDO_USER, which sudo sets to the invoking account.
TARGET_USER="${SUDO_USER:-$USER}"

echo "═══════════════════════════════════════════════════════════"
echo "  Install ROCm ${ROCM_VERSION} (host / baremetal) — Vega 8 / $(. /etc/os-release && echo "$VERSION_ID")"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Ubuntu codename forced to: ${UBUNTU_CODENAME} (24.04 packages — ABI compatible)"
echo "  ROCm repo: ${ROCM_REPO_BASE}"
echo ""

check_modular_rocm_conflict() {
    # AMD's modular ROCm packages (amdrocm-core*, e.g. for RDNA4 cards) manage
    # /opt/rocm via update-alternatives. Installing classic ROCm 7.2 packages
    # alongside them will fight over /opt/rocm and can break the existing
    # (e.g. R9700) setup. Refuse to continue if they are present.
    local dpkg_list
    dpkg_list=$(dpkg -l 2>/dev/null || true)
    if grep -q "^ii  amdrocm-core" <<<"$dpkg_list"; then
        echo "✗  AMD modular ROCm packages (amdrocm-core*) are installed:"
        grep "^ii  amdrocm-core" <<<"$dpkg_list" | awk '{print "     " $2 "  " $3}'
        echo ""
        echo "   Installing classic ROCm ${ROCM_VERSION} packages on top of these would"
        echo "   conflict over /opt/rocm (alternatives-managed) and could break the"
        echo "   GPU setup they were installed for."
        echo ""
        echo "   For ROCm inference on the Vega 8, use the self-contained Docker image:"
        echo "     docker build -t llama-rocm7-vega -f build/Dockerfile.rocm7-vega build/"
        echo "     ./run/run-docker-rocm7.sh /path/to/model.gguf"
        exit 1
    fi
}

check_kfd() {
    echo "─── Checking /dev/kfd ────────────────────────────────────────"
    if [ -c /dev/kfd ]; then
        echo "  ✓  /dev/kfd present — AMD GPU kernel module (amdgpu) is active"
    else
        echo "  ✗  /dev/kfd not found — amdgpu module may not be loaded"
        echo "     Check: lsmod | grep amdgpu"
        echo "     The GPU kernel driver is required before installing ROCm userspace."
        echo "     On this system, running a Docker ROCm container requires /dev/kfd —"
        echo "     if Docker ROCm works, /dev/kfd should be present."
        exit 1
    fi

    # Check group membership of the invoking human, not of root
    local user_groups
    user_groups=$(id -nG "$TARGET_USER")
    if [[ " $user_groups " == *" render "* && " $user_groups " == *" video "* ]]; then
        echo "  ✓  $TARGET_USER is in 'render' and 'video' groups"
    else
        echo "  ⚠  $TARGET_USER not in 'render' and/or 'video' groups."
        echo "     Adding $TARGET_USER..."
        $SUDO usermod -aG render,video "$TARGET_USER"
        echo "     You will need to log out and back in (or run 'newgrp render') for this to take effect."
    fi
    echo ""
}

install_rocm_key() {
    echo "─── Installing ROCm GPG signing key ──────────────────────────"
    $SUDO mkdir -p /etc/apt/keyrings
    if [ ! -f "$ROCM_KEYRING_PATH" ]; then
        echo "  Downloading ROCm GPG key..."
        wget -qO - "$ROCM_KEYRING_URL" | $SUDO gpg --dearmor -o "$ROCM_KEYRING_PATH"
        echo "  ✓  Key installed at $ROCM_KEYRING_PATH"
    else
        echo "  ✓  Key already present at $ROCM_KEYRING_PATH"
    fi
    echo ""
}

add_rocm_repo() {
    echo "─── Adding ROCm ${ROCM_VERSION} apt repository ─────────────────────────"

    # ROCm repo
    local rocm_list="/etc/apt/sources.list.d/rocm.list"
    local rocm_line="deb [arch=amd64 signed-by=${ROCM_KEYRING_PATH}] ${ROCM_REPO_BASE}/ ${UBUNTU_CODENAME} main"
    if [ -f "$rocm_list" ] && grep -qF "$ROCM_REPO_BASE" "$rocm_list" 2>/dev/null; then
        echo "  ✓  ROCm repo already configured ($rocm_list)"
    else
        echo "  Adding: $rocm_line"
        echo "$rocm_line" | $SUDO tee "$rocm_list" > /dev/null
        echo "  ✓  Created $rocm_list"
    fi

    # Package priority — prefer ROCm packages from AMD repo over Ubuntu universe
    local pref_file="/etc/apt/preferences.d/rocm"
    if [ ! -f "$pref_file" ]; then
        $SUDO tee "$pref_file" > /dev/null <<EOF
Package: *
Pin: release o=repo.radeon.com
Pin-Priority: 600
EOF
        echo "  ✓  Created $pref_file (priority 600 for AMD repo)"
    fi

    echo ""
}

apt_update_install() {
    echo "─── Updating package lists ───────────────────────────────────"
    $SUDO apt-get update -qq
    echo "  ✓  Package lists updated"
    echo ""

    echo "─── Installing ROCm packages ────────────────────────────────"
    echo "  Packages: ${ROCM_PACKAGES[*]}"
    echo ""
    $SUDO apt-get install -y --no-install-recommends "${ROCM_PACKAGES[@]}"
    echo ""
    echo "  ✓  ROCm packages installed"
    echo ""
}

fix_libxml2_soname() {
    # ROCm 7's LLVM/clang links against libxml2.so.2. Ubuntu 25.10+ ships
    # libxml2.so.16 only (soname bumped in libxml2 2.15), so hipcc fails to
    # start with "libxml2.so.2: cannot open shared object file". The symlink
    # goes in the ROCm lib dir rather than /lib/x86_64-linux-gnu so it cannot
    # affect any system package.
    echo "─── libxml2 soname compatibility ─────────────────────────────"
    local rocm_lib="$1/lib"

    if [ -e /lib/x86_64-linux-gnu/libxml2.so.2 ]; then
        echo "  ✓  libxml2.so.2 present system-wide — no shim needed"
        echo ""
        return
    fi

    local newest
    newest=$(ls -1 /lib/x86_64-linux-gnu/libxml2.so.* 2>/dev/null \
             | grep -E 'libxml2\.so\.[0-9]+$' | sort -V | tail -1 || true)
    if [ -z "$newest" ]; then
        echo "  ⚠  No libxml2.so.* found — install libxml2 if hipcc fails to start"
        echo ""
        return
    fi

    if [ -e "$rocm_lib/libxml2.so.2" ]; then
        echo "  ✓  Shim already present: $rocm_lib/libxml2.so.2"
    else
        $SUDO ln -sf "$newest" "$rocm_lib/libxml2.so.2"
        echo "  +  $rocm_lib/libxml2.so.2 -> $newest"
    fi

    # The ROCm LLVM binaries live in llvm/bin and resolve via $ORIGIN/../lib.
    if [ -d "$1/llvm/lib" ] && [ ! -e "$1/llvm/lib/libxml2.so.2" ]; then
        $SUDO ln -sf "$newest" "$1/llvm/lib/libxml2.so.2"
        echo "  +  $1/llvm/lib/libxml2.so.2 -> $newest"
    fi
    echo ""
}

verify_install() {
    echo "─── Verifying installation ───────────────────────────────────"

    local rocm_path=""
    for candidate in /opt/rocm /opt/rocm-${ROCM_VERSION}.0 /opt/rocm-${ROCM_VERSION}; do
        if [ -d "$candidate/bin" ]; then
            rocm_path="$candidate"
            break
        fi
    done

    if [ -z "$rocm_path" ]; then
        echo "  ✗  ROCm not found at /opt/rocm — installation may have failed"
        echo "     Check: ls /opt/rocm*"
        exit 1
    fi

    echo "  ✓  ROCm installed at: $rocm_path"
    echo ""

    fix_libxml2_soname "$rocm_path"

    if command -v hipcc &>/dev/null || [ -x "$rocm_path/bin/hipcc" ]; then
        local hipcc_bin
        hipcc_bin=$(command -v hipcc 2>/dev/null || echo "$rocm_path/bin/hipcc")
        echo "  ✓  hipcc: $(LD_LIBRARY_PATH="$rocm_path/lib:$rocm_path/llvm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" "$hipcc_bin" --version 2>&1 | head -1)"
    else
        echo "  ⚠  hipcc not found on PATH — add $rocm_path/bin to PATH"
    fi

    if "$rocm_path/bin/rocminfo" &>/dev/null; then
        echo ""
        echo "  GPU agents detected by rocminfo:"
        "$rocm_path/bin/rocminfo" 2>/dev/null \
            | awk '/^Agent [0-9]+/{agent=$0} /Name:/{print "    " agent " — " $0}' \
            | grep -v "^$" | head -10 || true
    else
        echo "  ⚠  rocminfo failed — /dev/kfd permissions may need a re-login"
    fi

    echo ""
}

print_next_steps() {
    echo "═══════════════════════════════════════════════════════════"
    echo "  ROCm ${ROCM_VERSION} installed successfully!"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "  If you were just added to the 'video'/'render' groups,"
    echo "  log out and back in (or: newgrp render) before proceeding."
    echo ""
    echo "  Next steps:"
    echo ""
    echo "  1. Add ROCm to PATH (add to ~/.bashrc for persistence):"
    echo "       export PATH=/opt/rocm/bin:\$PATH"
    echo "       export LD_LIBRARY_PATH=/opt/rocm/lib:\$LD_LIBRARY_PATH"
    echo ""
    echo "  2. libxml2 soname shim (.so.2 → .so.16) — applied automatically above,"
    echo "     scoped to /opt/rocm/lib so no system package is affected."
    echo ""
    echo "  3. Vega 8 GPU index — verify with:"
    echo "       rocminfo | grep -E 'Agent|Name.*gfx'"
    echo "     The index depends on which other GPUs are installed; the run"
    echo "     script auto-detects it (override with VEGA8_ROCM_DEVICE=N)."
    echo ""
    echo "  4. Build llama.cpp for Vega 8 (applies gfx900 tensile backport):"
    echo "       bash build/build-llamacpp-rocm7-baremetal.sh"
    echo ""
    echo "  5. Run (auto-detects the Vega 8 device index):"
    echo "       bash run/run-rocm7-baremetal.sh /path/to/model.gguf"
    echo ""
}

check_modular_rocm_conflict
check_kfd
install_rocm_key
add_rocm_repo
apt_update_install
verify_install
print_next_steps
