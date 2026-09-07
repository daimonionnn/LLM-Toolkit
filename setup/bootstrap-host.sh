#!/bin/bash
#
# Post-reinstall host bootstrap for the Vega 8 (gfx90c) LLM rig.
#
# A fresh Ubuntu install leaves the amdgpu kernel module working but wipes
# everything this project needs on top of it:
#
#   * GRUB is back to "quiet splash"     → GTT drops to ~30 GB, and loading a
#                                          ~20 GB model on ROCm HARD-FREEZES the
#                                          PC (happened 2026-06-13 and again on
#                                          the 26.04 reinstall).
#   * the user is not in render/video    → /dev/kfd is unreadable, rocminfo fails
#   * /opt/rocm is empty                 → no HIP runtime, no hipcc, no rocBLAS
#   * no compiler/cmake/vulkan-tools     → nothing can be built or diagnosed
#
# This script restores all of that in one pass. It is idempotent — safe to
# re-run — and it changes nothing that is already correct.
#
# Usage:
#   sudo bash setup/bootstrap-host.sh                 # everything except Docker
#   sudo bash setup/bootstrap-host.sh --with-docker   # also install Docker
#   sudo bash setup/bootstrap-host.sh --skip-rocm     # host prep only, no ROCm
#
# A REBOOT is required afterwards for the GRUB params and group membership.

set -euo pipefail

WITH_DOCKER=0
SKIP_ROCM=0
for arg in "$@"; do
    case "$arg" in
        --with-docker) WITH_DOCKER=1 ;;
        --skip-rocm)   SKIP_ROCM=1 ;;
        -h|--help)     sed -n '2,25p' "$0"; exit 0 ;;
        *) echo "Unknown option: $arg (try --help)"; exit 1 ;;
    esac
done

if [ "$(id -u)" -ne 0 ]; then
    echo "✗  Must run as root:  sudo bash $0 $*"
    exit 1
fi

# The invoking human, not root — group membership must land on them.
TARGET_USER="${SUDO_USER:-}"
if [ -z "$TARGET_USER" ] || [ "$TARGET_USER" = "root" ]; then
    echo "✗  Could not determine the target user (SUDO_USER unset)."
    echo "   Run via 'sudo bash $0', not from a root login shell."
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# GTT params only. Deliberately NOT set here:
#   amdgpu.cwsr_enable=0 — documented in docs/benchmarks.md as breaking the
#                          Qwen 35B load (crash during model loading).
#   amd_iommu=on         — not part of the 2026-06-13 verified-working config.
GRUB_PARAMS=(
    "amdgpu.gttsize=65536"
    "ttm.pages_limit=16777216"
)

APT_PACKAGES=(
    build-essential cmake ninja-build ccache pkg-config git wget curl
    libcurl4-openssl-dev            # llama.cpp server model downloads
    vulkan-tools libvulkan-dev      # vulkaninfo + the Vulkan llama.cpp backend
    glslc glslang-tools             # shader compilation — llama.cpp's Vulkan
                                    # backend calls glslc specifically; the
                                    # glslangValidator from glslang-tools is
                                    # not a substitute
    spirv-tools spirv-headers       # ggml-vulkan does find_package(SPIRV-Headers
                                    # CONFIG REQUIRED); without the headers
                                    # package CMake aborts at configure time
    libnuma-dev                     # ROCm runtime
)

hr() { echo "─────────────────────────────────────────────────────────────"; }

# ─── 1. GPU sanity ───────────────────────────────────────────────────────────
check_gpu() {
    echo "─── Checking the AMD GPU kernel driver ───────────────────────"
    # NOTE: never pipe into `grep -q` under `set -o pipefail` — grep exits at the
    # first match, the writer gets SIGPIPE (141), and the pipeline is reported as
    # failed even though the match succeeded. Capture first, match second.
    local lsmod_out
    lsmod_out=$(lsmod)
    if ! grep -q '^amdgpu' <<<"$lsmod_out"; then
        echo "  ✗  amdgpu module not loaded — this is a kernel-level problem."
        echo "     Nothing below will help until 'lsmod | grep amdgpu' shows it."
        exit 1
    fi
    echo "  ✓  amdgpu module loaded"

    if [ ! -c /dev/kfd ]; then
        echo "  ✗  /dev/kfd missing — the compute (KFD) interface is not up."
        exit 1
    fi
    echo "  ✓  /dev/kfd present"

    local gfx
    gfx=$(awk '$1 == "gfx_target_version" && $2 != 0 { print $2; exit }' \
          /sys/class/kfd/kfd/topology/nodes/*/properties 2>/dev/null || true)
    case "$gfx" in
        90012) echo "  ✓  Compute GPU: gfx90c (Vega 8 / Cezanne APU) — expected" ;;
        "")    echo "  ⚠  No compute-capable GPU node found in the KFD topology" ;;
        *)     echo "  ⚠  Unexpected gfx_target_version: $gfx (expected 90012 for Vega 8)" ;;
    esac
    echo ""
}

# ─── 2. render / video groups ────────────────────────────────────────────────
setup_groups() {
    echo "─── User groups (render, video) ──────────────────────────────"
    local current added=0
    current=$(id -nG "$TARGET_USER")
    for g in render video; do
        if [[ " $current " == *" $g "* ]]; then
            echo "  ✓  $TARGET_USER already in '$g'"
        else
            usermod -aG "$g" "$TARGET_USER"
            echo "  +  added $TARGET_USER to '$g'"
            added=1
        fi
    done
    [ "$added" = 1 ] && echo "     (takes effect after logout/reboot)"
    echo ""
}

# ─── 3. GRUB kernel parameters ───────────────────────────────────────────────
setup_grub() {
    echo "─── GRUB kernel parameters (64 GB GTT) ───────────────────────"
    local grub_file=/etc/default/grub

    local gtt_bytes current_gtt
    gtt_bytes=$(cat /sys/class/drm/card*/device/mem_info_gtt_total 2>/dev/null | head -1)
    if [ -n "$gtt_bytes" ]; then
        current_gtt=$(( gtt_bytes / 1024 / 1024 ))
        echo "  Current GTT: ${current_gtt} MiB (target: 65536 MiB after reboot)"
    else
        echo "  Current GTT: unknown"
    fi

    local line value changed=0
    line=$(grep '^GRUB_CMDLINE_LINUX_DEFAULT=' "$grub_file" || true)
    if [ -z "$line" ]; then
        echo "  ✗  GRUB_CMDLINE_LINUX_DEFAULT not found in $grub_file"
        exit 1
    fi
    value=$(echo "$line" | sed -e 's/^GRUB_CMDLINE_LINUX_DEFAULT="//' -e 's/"$//')

    for p in "${GRUB_PARAMS[@]}"; do
        local key="${p%%=*}"
        if [[ " $value " == *" ${key}="* ]]; then
            echo "  ✓  $key already set"
        else
            value="$value $p"
            echo "  +  adding $p"
            changed=1
        fi
    done

    if [ "$changed" = 0 ]; then
        echo "  ✓  GRUB already configured — no change"
        echo ""
        return
    fi

    value=$(echo "$value" | tr -s ' ' | sed -e 's/^ //' -e 's/ $//')
    cp -a "$grub_file" "${grub_file}.bak.$(date +%Y%m%d-%H%M%S)"
    echo "  ✓  backed up $grub_file"

    sed -i "s|^GRUB_CMDLINE_LINUX_DEFAULT=.*|GRUB_CMDLINE_LINUX_DEFAULT=\"${value}\"|" "$grub_file"
    echo "  New: GRUB_CMDLINE_LINUX_DEFAULT=\"${value}\""

    update-grub
    echo "  ✓  update-grub done — 64 GB GTT active after reboot"
    echo ""
}

# ─── 4. Build + Vulkan userspace ─────────────────────────────────────────────
install_apt_packages() {
    echo "─── Build tools and Vulkan userspace ─────────────────────────"
    apt-get update -qq
    apt-get install -y --no-install-recommends "${APT_PACKAGES[@]}"
    echo "  ✓  installed: ${APT_PACKAGES[*]}"
    echo ""
}

install_docker() {
    [ "$WITH_DOCKER" = 1 ] || return 0
    echo "─── Docker (for run/run-docker-rocm7.sh) ─────────────────────"
    apt-get install -y --no-install-recommends docker.io docker-buildx
    systemctl enable --now docker
    local docker_groups
    docker_groups=$(id -nG "$TARGET_USER")
    if [[ " $docker_groups " != *" docker "* ]]; then
        usermod -aG docker "$TARGET_USER"
        echo "  +  added $TARGET_USER to 'docker' (takes effect after reboot)"
    fi
    echo "  ✓  Docker installed and enabled"
    echo ""
}

# ─── 5. ROCm userspace ───────────────────────────────────────────────────────
install_rocm() {
    [ "$SKIP_ROCM" = 0 ] || { echo "─── ROCm install skipped (--skip-rocm) ───"; echo ""; return 0; }
    echo "─── ROCm 7.2 userspace ───────────────────────────────────────"
    if [ -x /opt/rocm/bin/hipcc ]; then
        echo "  ✓  ROCm already present at /opt/rocm — skipping install"
        echo ""
        return 0
    fi
    bash "$SCRIPT_DIR/install-rocm7-host.sh"
}

# ─── 6. Summary ──────────────────────────────────────────────────────────────
summary() {
    hr
    echo "  Host bootstrap complete"
    hr
    echo ""
    echo "  REBOOT NOW — required for:"
    echo "    * the 64 GB GTT kernel params to take effect"
    echo "    * '$TARGET_USER' to actually be in render/video"
    echo ""
    echo "      sudo reboot"
    echo ""
    echo "  After the reboot, verify:"
    echo "      cat /proc/cmdline | grep -o 'amdgpu.gttsize=[0-9]*'"
    echo "      awk '{print \$1/1024/1024\" MiB GTT\"}' /sys/class/drm/card*/device/mem_info_gtt_total"
    echo "      groups | grep -o 'render\\|video'"
    echo "      /opt/rocm/bin/rocminfo | grep -E 'Name:.*gfx'"
    echo ""
    echo "  Then build llama.cpp:"
    echo "      bash build/build-llamacpp-rocm7-baremetal.sh"
    echo ""
}

hr
echo "  Vega 8 LLM rig — post-reinstall host bootstrap"
hr
echo "  Target user : $TARGET_USER"
echo "  Distro      : $(. /etc/os-release && echo "$PRETTY_NAME")"
echo "  Kernel      : $(uname -r)"
echo "  Docker      : $([ "$WITH_DOCKER" = 1 ] && echo yes || echo "no (--with-docker to include)")"
echo ""

check_gpu
setup_groups
setup_grub
install_apt_packages
install_docker
install_rocm
summary
