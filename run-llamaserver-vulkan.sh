#!/bin/bash
#
# Run llama-server with Vulkan backend
#
# This wraps the Vulkan-built llama-server with correct library paths.
# By default, it selects the fastest discrete GPU available.
#
# Usage:
#   ./run-llamaserver-vulkan.sh /path/to/model.gguf [options]
#   ./run-llamaserver-vulkan.sh ~/models/llama-7b-q4.gguf -ngl 99
#   ./run-llamaserver-vulkan.sh ~/models/llama-7b-q4.gguf -ngl 99 -dev Vulkan0
#
# Devices on this system (see: llama-server --list-devices):
#   Vulkan0 = AMD Radeon Graphics (RADV RENOIR / Vega 8 iGPU)  ~24 GB shared
#   Vulkan1 = NVIDIA GeForce RTX 5090                          ~32 GB dedicated
#
# Performance (Llama 2 7B Q4_K_S, -ngl 99 -c 512):
#   Vulkan0 (Vega 8):   ~49 t/s prompt,  ~14 t/s generation
#   Vulkan1 (RTX 5090): ~2117 t/s prompt, ~273 t/s generation
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
LLAMA_SERVER="$SCRIPT_DIR/llama.cpp-vulkan/bin/llama-server"
LLAMA_LIB_DIR="$SCRIPT_DIR/llama.cpp-vulkan/lib"

if [ ! -x "$LLAMA_SERVER" ]; then
    echo "✗  llama-server (Vulkan) not found at: $LLAMA_SERVER"
    echo "   Build it first — see docs/BUILD.md"
    exit 1
fi

if [ $# -eq 0 ] || [[ "$1" == --help ]] || [[ "$1" == -h ]]; then
    echo "Usage: $0 <model.gguf> [llama-server options]"
    echo ""
    echo "Examples:"
    echo "  $0 ~/models/llama-7b-q4.gguf -ngl 99                # auto-select GPU"
    echo "  $0 ~/models/llama-7b-q4.gguf -ngl 99 -dev Vulkan1   # force RTX 5090"
    echo "  $0 ~/models/llama-7b-q4.gguf -ngl 99 -dev Vulkan0   # force Vega 8"
    echo ""
    echo "Available devices:"
    LD_LIBRARY_PATH="$LLAMA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$LLAMA_SERVER" --list-devices 2>&1 || true
    echo ""
    echo "Common options:"
    echo "  -ngl N     Number of layers to offload to GPU (99 = all)"
    echo "  -c N       Context size (default: 2048)"
    echo "  -dev NAME  Vulkan device (Vulkan0, Vulkan1, ...)"
    echo "  --host IP  Listen address (default: 127.0.0.1)"
    echo "  --port N   Listen port (default: 8080)"
    echo ""
    echo "The server exposes an OpenAI-compatible API at http://host:port/v1"
    exit 0
fi

MODEL="$1"
shift

if [ ! -f "$MODEL" ]; then
    echo "✗  Model file not found: $MODEL"
    exit 1
fi

# ─── Library path for Vulkan llama.cpp build ───
export LD_LIBRARY_PATH="$LLAMA_LIB_DIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ─── Launch ───
echo "═══════════════════════════════════════════════════════════"
echo "  Backend:  Vulkan"
echo "  Model:    $MODEL"
echo "  Args:     $*"
echo "═══════════════════════════════════════════════════════════"
echo ""

exec "$LLAMA_SERVER" -m "$MODEL" "$@"
