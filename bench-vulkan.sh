#!/usr/bin/env bash

set -euo pipefail

if [ $# -eq 0 ]; then
    echo "Usage: $0 /path/to/model.gguf [additional llama-bench args]"
    echo "Example: $0 ~/models/llama-2-7b-chat.Q4_K_S.gguf"
    exit 1
fi

MODEL="$1"
shift

BENCH_BIN="$(dirname "$0")/llama.cpp-vulkan/bin/llama-bench"

if [ ! -x "$BENCH_BIN" ]; then
    echo "✗ Error: $BENCH_BIN not found or not executable."
    exit 1
fi

echo "=========================================================="
echo " Starting Vulkan llama-bench"
echo " Model: $MODEL"
echo "=========================================================="

"$BENCH_BIN" -m "$MODEL" -p 128,512,1024,2048 -n 128 -ngl 99 "$@"
