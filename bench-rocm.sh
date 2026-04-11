#!/usr/bin/env bash

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 /path/to/model.gguf [additional llama-bench args]"
    echo "Example: $0 ~/models/llama-2-7b-chat.Q4_K_S.gguf"
    exit 1
fi

MODEL="$1"
shift

BENCH_BIN="$(dirname "$0")/llama.cpp-rocm-vega/bin/llama-bench"

if [ ! -x "$BENCH_BIN" ]; then
    echo "✗ Error: $BENCH_BIN not found or not executable. Did you build the ROCm backend yet?"
    exit 1
fi

# Apply the same safe ROCm logic we've discovered for this APU setup
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_ENABLE_SDMA=0
export HCC_SERIALIZE_KERNEL=3
export HCC_SERIALIZE_COPY=3
export GGML_HIP_UMA=1

echo "=========================================================="
echo " Starting ROCm/HIP llama-bench for Vega 8 (gfx90c/900)"
echo " Environment overrides applied (HSA, SDMA, HCC_SERIALIZE)"
echo " Model: $MODEL"
echo "=========================================================="

"$BENCH_BIN" -m "$MODEL" -p 128,512,1024,2048 -n 128 -ngl 99 "$@"
