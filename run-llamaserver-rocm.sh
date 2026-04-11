#!/bin/bash
#
# Run llama-server with ROCm/HIP on Vega 8 APU (gfx90c)
#
# This wraps the custom-built llama-server (from build-llamacpp-rocm-vega.sh)
# with the correct environment variables for the Vega 8 APU.
#
# Usage:
#   ./run-llamaserver-rocm.sh /path/to/model.gguf [options]
#   ./run-llamaserver-rocm.sh ~/models/qwen2.5-3b-q4_k_m.gguf -ngl 99
#   ./run-llamaserver-rocm.sh ~/models/llama-7b-q4.gguf -ngl 33 -c 4096
#

set -euo pipefail

LLAMA_SERVER="$(dirname "$0")/llama.cpp-rocm-vega/bin/llama-server"

if [ ! -x "$LLAMA_SERVER" ]; then
    echo "✗  llama-server not found at: $LLAMA_SERVER"
    echo "   Run build-llamacpp-rocm-vega.sh first to build it."
    exit 1
fi

if [ $# -eq 0 ] || [[ "$1" == --help ]] || [[ "$1" == -h ]]; then
    echo "Usage: $0 <model.gguf> [llama-server options]"
    echo ""
    echo "Examples:"
    echo "  $0 ~/models/qwen2.5-3b-q4_k_m.gguf -ngl 99"
    echo "  $0 ~/models/llama-7b-q4.gguf -ngl 33 -c 4096"
    echo ""
    echo "Common options:"
    echo "  -ngl N     Number of layers to offload to GPU (99 = all)"
    echo "  -c N       Context size (default: 2048)"
    echo "  --host IP  Listen address (default: 127.0.0.1)"
    echo "  --port N   Listen port (default: 8080)"
    echo ""
    echo "The server exposes an OpenAI-compatible API at http://host:port/v1"
    echo "You can connect LM Studio or any OpenAI client to it."
    exit 0
fi

MODEL="$1"
shift

if [ ! -f "$MODEL" ]; then
    echo "✗  Model file not found: $MODEL"
    exit 1
fi

# ─── Library path for custom-built llama.cpp ───
LLAMA_LIB_DIR="$(dirname "$LLAMA_SERVER")/../lib"
export LD_LIBRARY_PATH="$LLAMA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ─── ROCm environment for Vega 8 APU ───
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_ENABLE_SDMA=0
export HCC_SERIALIZE_KERNEL=3
export HCC_SERIALIZE_COPY=3
export HSA_XNACK=${HSA_XNACK:-0}   # 0=xnack-, 1=xnack+; must match build target
export GGML_HIP_UMA=1
export GPU_MAX_ALLOC_PERCENT=100
export GPU_SINGLE_ALLOC_PERCENT=100
export GPU_MAX_HEAP_SIZE=100
export GPU_FORCE_64BIT_PTR=1
export HIP_VISIBLE_DEVICES=${HIP_VISIBLE_DEVICES:-0}
export DRI_PRIME=pci-0000_0b_00.0

# ─── Pre-flight: warn if GRUB params are missing ───
# These kernel params fix ROCm memory faults / segfaults on AMD APUs:
#   amdgpu.cwsr_enable=1  — disable compute wave save/restore (crash trigger)
#   amd_iommu=on         — disable IOMMU ("page not present" faults)
#   ttm.pages_limit=25165824 — raise TTM page limit for large allocations
# Ref: https://medium.com/@agentz/how-to-fix-rocm-pytorch-memory-faults-on-amd-gpus-segmentation-fault-page-not-present-544b9f62f627
CMDLINE=$(cat /proc/cmdline 2>/dev/null || true)
if ! echo "$CMDLINE" | grep -q 'cwsr_enable=0'; then
    echo "⚠  Kernel param 'amdgpu.cwsr_enable=1' not detected."
    echo "   CWSR (compute wave save/restore) can crash Vega APUs under ROCm."
    echo "   Add to GRUB and reboot:  sudo sed -i 's/GRUB_CMDLINE_LINUX_DEFAULT=\"\(.*\)\"/GRUB_CMDLINE_LINUX_DEFAULT=\"\1 amdgpu.cwsr_enable=1 amd_iommu=on ttm.pages_limit=12582912\"/' /etc/default/grub && sudo update-grub"
    echo ""
fi

echo "═══════════════════════════════════════════════════════════"
echo "  llama-server (ROCm/HIP) — Vega 8 APU"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Model:  $MODEL"
echo "  Args:   $*"
echo "  GPU:    gfx90c → gfx900 (override)"
echo "  UMA:    enabled (GGML_HIP_UMA=1)"
echo "  CWSR:   $(cat /sys/module/amdgpu/parameters/cwsr_enable 2>/dev/null || echo '?') (need 0)"
echo "  IOMMU:  $(grep -q 'amd_iommu=on' /proc/cmdline 2>/dev/null && echo 'off' || echo 'on (may crash)')"
echo ""
echo "  API endpoint: http://127.0.0.1:8080/v1"
echo "  (Use Ctrl+C to stop)"
echo ""

exec "$LLAMA_SERVER" -m "$MODEL" "$@"
