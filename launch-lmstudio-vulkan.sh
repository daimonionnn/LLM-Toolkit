#!/bin/bash
#
# Launch LM Studio with AMD Vega 8 (Ryzen 5700G APU) GPU acceleration
#
# ╔══════════════════════════════════════════════════════════════════════╗
# ║  IMPORTANT: ROCm backend WILL NOT WORK with this APU!             ║
# ║                                                                    ║
# ║  LM Studio's ROCm backend (v2.12 & v2.13) is compiled ONLY for:  ║
# ║    gfx1030, gfx1100, gfx1101, gfx1102, gfx1151, gfx1200, gfx1201║
# ║                                                                    ║
# ║  The Vega 8 iGPU = gfx90c (GCN 5), which is NOT in the binary.   ║
# ║  HSA_OVERRIDE_GFX_VERSION makes ROCm *detect* the GPU, but the   ║
# ║  compute kernels don't exist → "invalid device function" crash.   ║
# ║                                                                    ║
# ║  ERROR from logs:                                                  ║
# ║    ggml_cuda_compute_forward: MUL_MAT failed                      ║
# ║    ROCm error: invalid device function                            ║
# ║    llama.cpp abort:98: ROCm error                                 ║
# ║                                                                    ║
# ║  WORKING SOLUTION: Use the Vulkan backend (RADV RENOIR).         ║
# ║  It fully supports gfx90c and is already proven working.          ║
# ║                                                                    ║
# ║  For ROCm, build llama.cpp from source with -DAMDGPU_TARGETS=    ║
# ║  gfx900 — see build-llamacpp-rocm-vega.sh                        ║
# ╚══════════════════════════════════════════════════════════════════════╝
#
# The Vega 8 iGPU (gfx90c) is not officially supported by ROCm,
# but works via:
#   - Vulkan backend (RADV driver) for LM Studio inference  ← USE THIS
#   - Custom-built llama.cpp with HIP for gfx900            ← Advanced option
#   - HSA_OVERRIDE_GFX_VERSION=9.0.0 for tools only (rocminfo, rocm-smi)
#
# Hardware: AMD Ryzen 7 5700G - Vega 8 (gfx90c / Renoir)
# PCI:      0b:00.0 -> /dev/dri/renderD129, /dev/dri/card2
# VRAM:     16 GB UMA (BIOS) + ~23 GB GTT (shared system RAM)
#
# TIP: Increase VRAM in BIOS (UMA Frame Buffer Size) to 16GB for best results.
#      Look for: Settings > Advanced > AMD CBS > NBIO > GFX > UMA Frame Buffer Size
#

set -euo pipefail

LMSTUDIO_PATH="$HOME/ai/lmstudio/squashfs-root/lm-studio"
LMSTUDIO_LOG="/tmp/lmstudio.log"
BACKEND_MODE="${1:-vulkan}"   # vulkan (default) | rocm-custom | diagnose
DRY_RUN=false

for arg in "$@"; do
    case "$arg" in
        --dry-run) DRY_RUN=true ;;
        --vulkan)  BACKEND_MODE="vulkan" ;;
        --rocm)    BACKEND_MODE="rocm-custom" ;;
        --diagnose) BACKEND_MODE="diagnose" ;;
        --help|-h)
            echo "Usage: $0 [--vulkan|--rocm|--diagnose] [--dry-run] [--help]"
            echo ""
            echo "  --vulkan    Use Vulkan backend (default, RECOMMENDED)"
            echo "  --rocm      Use custom-built ROCm backend (requires build-llamacpp-rocm-vega.sh)"
            echo "  --diagnose  Run diagnostics and show GPU/memory info without launching"
            echo "  --dry-run   Show configuration without launching LM Studio"
            echo "  --help      Show this help"
            exit 0
            ;;
    esac
done

# ─── ROCm workaround for unsupported gfx90c (Vega 8 APU) ───
# NOTE: This only helps rocminfo/rocm-smi. The LM Studio ROCm backend
# does NOT contain gfx900 kernels — it will crash with "invalid device function".
# This override is kept for diagnostic tools only.
export HSA_OVERRIDE_GFX_VERSION=9.0.0

# ─── Vulkan device selection (PRIMARY BACKEND) ───
# GPU0 = AMD Radeon Graphics (RADV RENOIR) — the Vega 8 iGPU
# GPU1 = NVIDIA GeForce RTX 5090
# GPU2 = llvmpipe (software)
# Force llama.cpp (inside LM Studio) to use the AMD iGPU for Vulkan compute
export GGML_VK_DEVICE=0

# Only expose the AMD RADV Vulkan driver (hide NVIDIA and llvmpipe)
# LM Studio hides integrated GPUs when a discrete GPU is also visible,
# so we must restrict to AMD-only for the Vega 8 to appear in the UI.
# To use the RTX 5090, use the CUDA runtime in LM Studio instead.
export VK_ICD_FILENAMES="/usr/share/vulkan/icd.d/radeon_icd.json"

# ─── AMD GPU memory tuning (critical for APU shared memory) ───
# Allow the GPU to allocate up to 100% of available memory (default is ~75%)
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100
# Allow larger GTT (Graphics Translation Table) allocation for APU shared memory
export GPU_MAX_HEAP_SIZE=100
# ROCm: disable SDMA — known to cause hangs/crashes on APU iGPUs
export HSA_ENABLE_SDMA=0
export HCC_SERIALIZE_KERNEL=3
export HCC_SERIALIZE_COPY=3
# Tell llama.cpp this is a UMA (Unified Memory Architecture) APU
# This changes memory allocation strategy to use system RAM directly
export GGML_HIP_UMA=1

# ─── AMD GPU performance ───
# Use high-performance power profile when running inference
export GPU_FORCE_64BIT_PTR=1
export ROC_ENABLE_PRE_VEGA=1

# ─── Render device hint ───
# Point to the AMD APU render node explicitly
export DRI_PRIME=pci-0000_0b_00.0

# ─── Verify prerequisites ───
check_prereqs() {
    local ok=true

    # Check render group membership
    if ! id -nG | grep -qw render; then
        echo "⚠  WARNING: User '$(whoami)' is not in the 'render' group in this session."
        echo "   Run: sudo usermod -aG render,video $(whoami)"
        echo "   Then log out and back in (or reboot)."
        ok=false
    fi

    # Check /dev/kfd access
    if [ ! -r /dev/kfd ] || [ ! -w /dev/kfd ]; then
        echo "⚠  WARNING: Cannot read/write /dev/kfd — ROCm won't work."
        echo "   Ensure you are in the 'render' group and have logged in fresh."
        ok=false
    fi

    # Check Vulkan ICD
    if [ ! -f /usr/share/vulkan/icd.d/radeon_icd.json ]; then
        echo "⚠  WARNING: AMD Vulkan ICD not found."
        echo "   Install: sudo apt install mesa-vulkan-drivers"
        ok=false
    fi

    # Check LM Studio binary
    if [ ! -x "$LMSTUDIO_PATH" ]; then
        echo "✗  ERROR: LM Studio not found at: $LMSTUDIO_PATH"
        exit 1
    fi

    if $ok; then
        echo "✓  All prerequisites OK"
    fi
}

get_lmstudio_pids() {
    pgrep -f "^$LMSTUDIO_PATH" || true
}

stop_lmstudio_if_running() {
    local pids
    pids="$(get_lmstudio_pids)"

    if [ -z "$pids" ]; then
        return 0
    fi

    echo "  Found existing LM Studio process(es): $pids"
    echo "  Stopping stale/background instance..."
    kill $pids 2>/dev/null || true
    sleep 2

    pids="$(get_lmstudio_pids)"
    if [ -n "$pids" ]; then
        echo "  Force-killing remaining process(es): $pids"
        kill -9 $pids 2>/dev/null || true
        sleep 1
    fi
}

show_config() {
    echo "═══════════════════════════════════════════════════════════"
    echo "  LM Studio — AMD Vega 8 (gfx90c) GPU Launch"
    echo "═══════════════════════════════════════════════════════════"
    echo ""
    echo "  GPU:                   AMD Radeon Graphics (Vega 8 / Renoir)"
    echo "  Architecture:          gfx90c (GCN 5) — NOT in LM Studio ROCm binary"
    echo "  Backend mode:          $BACKEND_MODE"
    echo "  Vulkan device:         GPU0 (RADV RENOIR)"
    echo "  Vulkan ICD:            radeon_icd.json (AMD only)"
    echo "  Render node:           /dev/dri/renderD129"
    echo ""
    echo "  Key environment:"
    echo "    HSA_OVERRIDE_GFX_VERSION = $HSA_OVERRIDE_GFX_VERSION (for diag tools only)"
    echo "    GGML_VK_DEVICE           = $GGML_VK_DEVICE"
    echo "    VK_ICD_FILENAMES         = $VK_ICD_FILENAMES"
    echo "    GPU_MAX_ALLOC_PERCENT    = $GPU_MAX_ALLOC_PERCENT"
    echo "    GGML_HIP_UMA             = $GGML_HIP_UMA"
    echo ""

    # Show VRAM/GTT if rocm-smi is available
    if command -v rocm-smi &>/dev/null && [ -r /dev/kfd ]; then
        echo "  Memory:"
        local vram_total vram_used gtt_total gtt_used
        vram_total=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Total Memory" | awk '{print $NF}')
        vram_used=$(rocm-smi --showmeminfo vram 2>/dev/null | grep "Used Memory" | awk '{print $NF}')
        gtt_total=$(rocm-smi --showmeminfo gtt 2>/dev/null | grep "Total Memory" | awk '{print $NF}')
        gtt_used=$(rocm-smi --showmeminfo gtt 2>/dev/null | grep "Used Memory" | awk '{print $NF}')
        if [ -n "$vram_total" ]; then
            echo "    VRAM:  $(( vram_total / 1048576 )) MB total, $(( vram_used / 1048576 )) MB used"
        fi
        if [ -n "$gtt_total" ]; then
            echo "    GTT:   $(( gtt_total / 1048576 )) MB total, $(( gtt_used / 1048576 )) MB used"
        fi
        echo ""
    fi

    check_prereqs
    echo ""
}

# ─── Main ───
show_config

if [ "$BACKEND_MODE" = "diagnose" ]; then
    echo "  ─── Diagnostics ───"
    echo ""
    # Show ROCm backend targets
    echo "  LM Studio ROCm backend compiled targets:"
    for manifest in $HOME/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-amd-rocm-*/backend-manifest.json; do
        if [ -f "$manifest" ]; then
            ver=$(python3 -c "import json; print(json.load(open('$manifest'))['version'])" 2>/dev/null)
            targets=$(python3 -c "import json; print(', '.join(json.load(open('$manifest'))['gpu']['targets']))" 2>/dev/null)
            echo "    v$ver: $targets"
        fi
    done
    echo "    Your GPU: gfx90c → override gfx900 (NOT in any of the above!)"
    echo ""

    # Recent ROCm errors
    latest_log=$(ls -t $HOME/.lmstudio/server-logs/2026-*/*.log 2>/dev/null | head -1)
    if [ -n "$latest_log" ]; then
        echo "  Recent ROCm errors from logs:"
        grep -i "rocm error\|invalid device\|abort" "$latest_log" 2>/dev/null | tail -5 | sed 's/^/    /'
        echo ""
    fi

    echo "  ─── Recommendation ───"
    echo ""
    echo "  ✗ ROCm backend: WILL CRASH (no gfx900 kernels in binary)"
    echo "  ✓ Vulkan backend: WORKS (RADV fully supports gfx90c)"
    echo "  ✓ Custom ROCm build: POSSIBLE (see build-llamacpp-rocm-vega.sh)"
    echo ""
    exit 0
fi

if $DRY_RUN; then
    echo "  (dry-run mode — not launching LM Studio)"
    echo "  Would stop stale LM Studio processes (if any), then launch: $LMSTUDIO_PATH"
    exit 0
fi

echo "  Launching LM Studio..."
echo "═══════════════════════════════════════════════════════════"

# Self-contained restart logic: stop stale instance and launch fresh
stop_lmstudio_if_running
nohup "$LMSTUDIO_PATH" >"$LMSTUDIO_LOG" 2>&1 &
LMSTUDIO_PID=$!
sleep 2

if ! ps -p "$LMSTUDIO_PID" >/dev/null 2>&1; then
    echo "  ✗ LM Studio failed to start. Recent log output:"
    tail -n 80 "$LMSTUDIO_LOG" 2>/dev/null || true
    exit 1
fi

if [ -n "${LMSTUDIO_PID:-}" ]; then
    echo "  PID: $LMSTUDIO_PID"
    echo "  Log: $LMSTUDIO_LOG"
fi
echo ""
echo "  ╔══════════════════════════════════════════════════════╗"
echo "  ║  In LM Studio:                                      ║"
echo "  ║                                                      ║"
echo "  ║  1. Settings → My GPUs                              ║"
echo "  ║  2. Select 'Vulkan' as GPU backend (NOT ROCm!)      ║"
echo "  ║  3. Ensure 'AMD Radeon Graphics' is selected        ║"
echo "  ║  4. Load a GGUF model and set GPU offload layers    ║"
echo "  ║                                                      ║"
echo "  ║  WARNING: Do NOT select ROCm — it will crash!       ║"
echo "  ║  The ROCm binary has no gfx900/gfx90c kernels.     ║"
echo "  ╚══════════════════════════════════════════════════════╝"
echo ""
echo "  With 16GB UMA + GTT, you can run:"
echo "    • 3-4B Q4 models: full GPU offload (~2-3GB)"
echo "    • 7-8B Q4 models: full GPU offload (~4-5GB)"
echo "    • 13B Q4 models:  partial GPU offload (~7-8GB)"
echo ""
echo "  Vulkan performance is comparable to ROCm for inference."
echo ""
