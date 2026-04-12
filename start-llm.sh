#!/bin/bash
# Quick-start: Llama 2 7B Chat — Vulkan on Vega8 by default
#
# Kills anything on port 8081, then launches llama-server.
# API: http://127.0.0.1:8081/v1
#
# Defaults to Vulkan backend on the RTX 5090 (Vulkan1).
# ROCm/HIP compute is broken on kernel 6.17 (hard-crashes the PC).
# Vulkan works perfectly via Mesa RADV / NVIDIA proprietary drivers.
#
# Usage:
#   ./start-llm.sh                # Vulkan on iGPU
#   ./start-llm.sh --vega         # Vulkan on Vega 8 iGPU (~49 t/s prompt, ~14 t/s gen)
#   ./start-llm.sh --cpu          # CPU-only via ROCm build (~55 t/s prompt, ~12 t/s gen)
#   ./start-llm.sh --rocm         # ROCm GPU offload (WARNING: crashes on kernel 6.17)
#
# Backends:
#   Vulkan (default) — uses Mesa RADV or NVIDIA proprietary driver
#   ROCm (legacy)    — HIP 5.7 compute, broken on kernel 6.17

set -euo pipefail

export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_ENABLE_SDMA=0
export HCC_SERIALIZE_KERNEL=3
export HCC_SERIALIZE_COPY=3

MODEL="/home/matt/.lmstudio/models/TheBloke/Llama-2-7B-Chat-GGUF/llama-2-7b-chat.Q4_K_S.gguf"
#MODEL="/home/matt/.lmstudio/models/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"
PORT=8081

# ─── Defaults: Vulkan0 ───
BACKEND="vulkan"
VULKAN_DEV="Vulkan0"     # Vega8
NGL=99
CTX=4096
EXTRA_ARGS=()

# ─── Minimum available RAM (MB) required to proceed ───
# LLM model + runtime overhead
MIN_AVAIL_MB=32000

case "${1:-}" in
    --vega)
        VULKAN_DEV="Vulkan0"   # AMD Radeon Vega 8 (RADV RENOIR)
        NGL=99
        CTX=512
        echo "  Mode: Vulkan on Vega 8 iGPU (~49/14 t/s)"
        shift
        ;;
    --cpu)
        BACKEND="rocm"
        NGL=0
        CTX=512
        export HIP_VISIBLE_DEVICES=-1   # hide GPU, pure CPU mode
        echo "  Mode: CPU-only (~55/12 t/s)"
        shift
        ;;
    --rocm)
        BACKEND="rocm"
        NGL=1
        CTX=512
        echo "⚠  ROCm GPU mode — may hard-crash the PC on kernel 6.17!"
        echo "   (HIP compute ring timeouts + MODE2 GPU reset)"
        echo ""
        shift
        ;;
    --help|-h)
        echo "Usage: $0 [--vega|--cpu|--rocm|--help]"
        echo ""
        echo "  (default)   Vulkan on RTX 5090 — ~2117/273 t/s"
        echo "  --vega      Vulkan on Vega 8 iGPU — ~49/14 t/s"
        echo "  --cpu       CPU-only (ROCm build) — ~55/12 t/s"
        echo "  --rocm      ROCm GPU offload (CRASHES on kernel 6.17)"
        echo ""
        exit 0
        ;;
    *)
        echo "  Mode: Vulkan on RTX 5090 (~2117/273 t/s)"
        ;;
esac

# ─── Safeguard: stop memory-hungry services to free shared RAM ───
# On UMA APUs, every GB counts — background services can push us into OOM territory.

# Stop OpenClaw gateway if running (~0.5-1.8 GB)
if systemctl --user is-active openclaw-gateway.service &>/dev/null; then
    echo "⚠  OpenClaw gateway is running (uses 0.5-1.8 GB RAM)."
    echo "   Stopping temporarily to free memory for the model..."
    systemctl --user stop openclaw-gateway.service 2>/dev/null || true
    sleep 1
    echo "   Stopped. Restart later with: systemctl --user start openclaw-gateway"
    echo ""
fi

# Stop Elasticsearch/RAGFlow containers if running (~4-5 GB)
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "ragflow-es"; then
    echo "⚠  Elasticsearch (ragflow) is running (~4.7 GB RAM)."
    echo "   Stopping Docker containers to free memory..."
    docker stop ragflow-es-01 2>/dev/null || true
    sleep 2
    echo "   Stopped. Restart later with: docker start ragflow-es-01"
    echo ""
fi

# Unload LM Studio models to free shared RAM
if command -v lms &>/dev/null; then
    LOADED=$(lms ps 2>/dev/null | grep -cE "^  " || true)
    if [ "$LOADED" -gt 0 ] 2>/dev/null; then
        echo "⚠  LM Studio has models loaded in memory."
        echo "   On a UMA APU they share RAM with the GPU — running both will OOM."
        echo ""
        echo "   Unloading all LM Studio models..."
        lms unload --all 2>/dev/null || true
        sleep 2
        echo "   Done."
        echo ""
    fi
fi

# ─── Safeguard: check available memory before launching ───
AVAIL_MB=$(awk '/MemAvailable/ {printf "%d", $2/1024}' /proc/meminfo)
echo "  Available RAM: ${AVAIL_MB} MB  (need ${MIN_AVAIL_MB} MB)"

if [ "$AVAIL_MB" -lt "$MIN_AVAIL_MB" ]; then
    echo ""
    echo "✗  Not enough free memory to safely load the model."
    echo "   Available: ${AVAIL_MB} MB — need at least ${MIN_AVAIL_MB} MB."
    echo ""
    echo "   This Vega 8 iGPU shares system RAM (UMA). Loading the model"
    echo "   with insufficient memory will hard-lock the entire system."
    echo ""
    echo "   Try:"
    echo "     • Close browsers / heavy apps"
    echo "     • lms unload --all   (if LM Studio has models loaded)"
    echo "     • Reboot if memory is fragmented"
    echo ""
    echo "   To bypass this check (at your own risk):"
    echo "     SKIP_MEM_CHECK=1 ./start-llm.sh"
    if [[ "${SKIP_MEM_CHECK:-}" == "1" ]]; then
        echo ""
        echo "   SKIP_MEM_CHECK=1 set — proceeding anyway..."
    else
        exit 1
    fi
fi
echo ""

# Kill existing process on port 8080
PID=$(lsof -ti :$PORT 2>/dev/null) || true
if [ -n "$PID" ]; then
    echo "Killing process $PID on port $PORT..."
    kill -9 "$PID" 2>/dev/null || true
    sleep 1
fi

exec "$(dirname "$0")/run-llamaserver-${BACKEND}.sh" \
    "$MODEL" \
    -ngl "$NGL" -c "$CTX" --port "$PORT" \
    -b 64 -ub 64 \
    --no-warmup \
    ${BACKEND:+$([ "$BACKEND" = "vulkan" ] && echo "-dev $VULKAN_DEV" || echo "-fa off")} \
    "${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"}"
