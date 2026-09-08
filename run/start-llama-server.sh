#!/bin/bash
# start-llama-server.sh — Launch llama-server on Vega 8 iGPU
#
# Defaults to Vulkan / Mesa RADV (best decode, works on any host ROCm state).
# Model defaults to Qwen3.5-35B-A3B-Q4_K_M — override with MODEL= env var.
#
# Usage:
#   ./run/start-llama-server.sh                # Vulkan / Mesa RADV (default)
#   ./run/start-llama-server.sh --cpu          # CPU only (no GPU offload)
#   ./run/start-llama-server.sh --rocm-docker  # ROCm 7.2 via Docker (recommended ROCm path)
#   ./run/start-llama-server.sh --rocm         # ROCm 7.2 baremetal (needs ROCm 7.2 host install)
#
# Environment variables:
#   MODEL=/path/to/model.gguf        — override default model path
#   CTX=8192                         — context size (default: 8192)
#   PORT=8080                        — server port  (default: 8080)
#
# Backends at a glance (Vega 8 iGPU, Qwen3.5-35B-A3B-Q4_K_M, May 2026):
#   Vulkan (Mesa RADV)  — prefill ~50 t/s @ 4K ctx, decode ~20 t/s  ← best decode
#   CPU only            — prefill ~233 t/s @ 4K ctx, decode ~13 t/s ← best prefill
#   ROCm 7.2            — prefill ~68 t/s @ 4K ctx, decode ~14 t/s  ← best GPU prefill
#
# Note: ROCm 7.2 baremetal requires gfx900 support on the host. AMD's modular
# ROCm packages (amdrocm-core 7.13+/gfx120x) break it — the launcher detects
# this and tells you to use --rocm-docker instead. See README "ROCm on Vega 8".
#
# See docs/benchmarks.md for full comparison.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# ─── Config ──────────────────────────────────────────────────────────────────

# TODO: PORT= only takes effect for the Vulkan/CPU/baremetal modes.
#       run-docker-rocm7.sh hardcodes the -p 8080:8080 mapping, so with
#       --rocm-docker a non-default PORT makes llama-server listen on $PORT
#       *inside* the container while the host still forwards only 8080 —
#       the endpoint silently breaks. Plumb PORT through the Docker launchers.
MODEL="${MODEL:-$HOME/.lmstudio/models/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf}"
PORT="${PORT:-8080}"
CTX="${CTX:-8192}"
# Prefill micro-batch. Measured 2026-09-08 on the 35B at a 3330-token prompt
# (llama-bench, cold prefill): ROCm 84 -> 144 t/s and Vulkan 139 -> 198 t/s going
# from the upstream default of 512 to 4096. The win comes from filling the MoE
# expert tiles; it costs ~2 GB of extra GTT at -c 8192.
#
# The cap is NOT optional. At 32K context with -ub 4096 the Vulkan backend
# exceeds the GPU's compute-ring watchdog and the driver resets the ring:
#   amdgpu: ring comp_1.0.1 timeout ... device wedged, but recovered through reset
# and llama-bench dies with vk::DeviceLostError. Measured 2026-09-08:
#   ctx  8192 x ub 4096 =  33.6M  OK
#   ctx 16384 x ub 4096 =  67.1M  OK
#   ctx 32768 x ub 2048 =  67.1M  OK   (120.1 t/s, the best that works at 32K)
#   ctx 32768 x ub 4096 = 134.2M  DEVICE LOST
# The failures track the ctx x ubatch product, so cap it at 2^26. That is a
# three-point fit, not a law — treat it as a conservative bound and lower UBATCH
# further if you see a ring reset in dmesg.
BATCH="${BATCH:-4096}"
if [ -z "${UBATCH:-}" ]; then
    UBATCH=$(( 67108864 / CTX ))
    [ "$UBATCH" -gt 4096 ] && UBATCH=4096
    [ "$UBATCH" -lt 512 ]  && UBATCH=512
fi
[ "$BATCH" -lt "$UBATCH" ] && BATCH="$UBATCH"

# ─── Helpers ─────────────────────────────────────────────────────────────────

free_port() {
    local pid
    pid=$(lsof -ti :"$PORT" 2>/dev/null) || true
    if [ -n "$pid" ]; then
        echo "Killing existing process $pid on :$PORT"
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi
}

# The Vega 8 shows up as "RADV RENOIR" in llama.cpp's Vulkan device list.
# Its index can shift when discrete GPUs are added/removed, so detect it.
detect_vega_vulkan_dev() {
    local dev
    dev=$(LD_LIBRARY_PATH="$SCRIPT_DIR/../llm/vulkan/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$SCRIPT_DIR/../llm/vulkan/bin/llama-server" --list-devices 2>/dev/null \
        | awk -F: '/RENOIR/ { gsub(/^[ \t]+/, "", $1); print $1; exit }') || true
    echo "${dev:-Vulkan0}"
}

banner() {
    echo "  Backend: $1"
    echo "  API:     http://127.0.0.1:$PORT/v1"
    echo ""
}

# ─── Parse mode ──────────────────────────────────────────────────────────────

MODE="${1:-}"

case "$MODE" in
    ""|--vulkan)
        [ -n "$MODE" ] && shift || true
        VULKAN_DEV="$(detect_vega_vulkan_dev)"
        banner "Vulkan (Mesa RADV / Vega 8, $VULKAN_DEV)"
        free_port
        exec "$SCRIPT_DIR/run-llamaserver-vulkan.sh" \
            "$MODEL" \
            -ngl 99 -c "$CTX" --port "$PORT" --no-warmup \
            -dev "$VULKAN_DEV" -fa 1 \
            -b "$BATCH" -ub "$UBATCH" \
            "$@"
        ;;
    --cpu)
        shift
        banner "CPU only (no GPU offload)"
        free_port
        # -dev none, NOT -ngl 0: upstream changed the -ngl default to auto, and
        # with a GPU backend present `-ngl 0` still offloads (measured 2026-09-08:
        # 91% GPU busy, 6.5 GB in GTT). Only -dev none is actually CPU-only.
        exec "$SCRIPT_DIR/run-llamaserver-vulkan.sh" \
            "$MODEL" \
            -dev none -c "$CTX" --port "$PORT" --no-warmup \
            -fa 1 \
            "$@"
        ;;
    --rocm-docker)
        shift
        banner "ROCm 7.2 Docker"
        free_port
        exec "$SCRIPT_DIR/run-docker-rocm7.sh" \
            "$MODEL" \
            -ngl 99 -c "$CTX" --port "$PORT" --no-warmup \
            -fa 0 \
            "$@"
        ;;
    --rocm|--rocm7|--baremetal)
        shift
        banner "ROCm 7.2 baremetal (Vega 8, gfx900)"
        free_port
        exec "$SCRIPT_DIR/run-rocm7-baremetal.sh" \
            "$MODEL" \
            -ngl 99 -c "$CTX" --port "$PORT" --no-warmup \
            -fa 0 \
            "$@"
        ;;
    --help|-h)
        echo "Usage: $0 [--vulkan|--cpu|--rocm-docker|--rocm|--help]"
        echo ""
        echo "  (default)       Vulkan / Mesa RADV  — best decode throughput"
        echo "  --cpu           CPU only            — best prefill at large context"
        echo "  --rocm-docker   ROCm 7.2 in Docker  — best GPU prefill, self-contained"
        echo "  --rocm          ROCm 7.2 baremetal  — needs ROCm 7.2 + gfx900 backport on host"
        echo ""
        echo "Env vars: MODEL=  CTX=  PORT="
        echo ""
        exit 0
        ;;
    *)
        echo "Unknown option: $MODE"
        echo "Run '$0 --help' for usage."
        exit 1
        ;;
esac
