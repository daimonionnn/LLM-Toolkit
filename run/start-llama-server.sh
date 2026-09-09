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
# The cap is NOT optional. Too large a micro-batch makes one Vulkan compute
# dispatch outrun the driver's watchdog; the ring is reset, the server dies with
# vk::DeviceLostError, and on one occasion the machine hard-locked:
#   amdgpu: ring comp_1.1.0 timeout ... device wedged, but recovered through reset
#
# The limit is NOT the ctx x ubatch product alone — it scales with the model's
# attention head dimension, i.e. with the work in one dispatch. Measured:
#   model  head  n_kv   ub     n_kv*ub   n_kv*ub*head  result
#   Qwen    256  32768  2048    67.1M       17.2e9     OK (120.1 t/s)
#   Qwen    256  32768  4096   134.2M       34.4e9     DEVICE LOST
#   gemma   512  32768  2048    67.1M       34.4e9     DEVICE LOST
#   gemma   512  16384  2048    33.6M       17.2e9     OK
# So the failures line up on n_kv*ub*head, not on n_kv*ub: gemma dies at exactly
# the product this cap used to call safe.
#
# n_kv, NOT the allocated context. Verified 2026-09-09 by holding -c at 131072
# and varying only the prompt (gemma, Vulkan, -ub 4096): 61 tokens = 1.3 s ok,
# 6021 = 35.9 s ok, ~16000 = DEVICE LOST. The same allocation both works and
# wedges the ring. Allocating a big context costs memory, not dispatch time.
# This is why LM Studio runs gemma at 128k with evalBatchSize 4096 on Vulkan
# without trouble: a chat turn never fills the cache. Paste a 16k-token document
# in and it hangs identically.
#
# A launcher cannot know n_kv in advance, so it derives from CTX -- the worst
# case, a prompt that fills the context. Right default for a server that may be
# handed anything, pessimistic if you know your prompts stay short: then set
# UBATCH= explicitly and a much larger micro-batch is safe at any allocation. Lowering iGPU clocks, disabling its
# Curve Optimizer and cutting CPU boost did not change it — it is the workload,
# not the voltage.
#
# A shell launcher cannot read the head dimension out of the GGUF, so the default
# assumes the larger case (512) and caps ctx*ub at 2^25. On a 256-head model that
# leaves roughly a factor of two on the table; raise it deliberately with UBATCH=
# if you know your model's head dimension, and lower it if dmesg shows a ring
# reset. Guessing high here costs a frozen desktop, guessing low costs prefill.
#
# The 2^25 cap is Vulkan's, and only Vulkan's. ROCm ran the exact product that
# kills Vulkan — 32768 x 4096 x 256 = 34.4e9 on the 35B — at 99.31 t/s prefill,
# so the limit is a property of Vulkan's dispatch shape, not of the hardware.
# ROCm therefore gets its own cap, at 2^26: twice Vulkan's, and the largest value
# any ROCm run has actually demonstrated (34.4e9, reached from both directions —
# 35B at 32768x4096x256, gemma at 32768x2048x512).
#
# Not a flat -ub 4096, tempting as that is. That would put gemma at 32k on
# 32768 x 4096 x 512 = 68.7e9, double anything measured on either backend, and
# the point of this whole block is that an unvalidated guess here costs a frozen
# desktop. Probing ROCm's real ceiling is open work — see the README TODO.
BATCH="${BATCH:-4096}"
UBATCH_SET=1
if [ -z "${UBATCH:-}" ]; then
    UBATCH_SET=0
    UBATCH=$(( 33554432 / CTX ))
    [ "$UBATCH" -gt 4096 ] && UBATCH=4096
    [ "$UBATCH" -lt 512 ]  && UBATCH=512
fi
[ "$BATCH" -lt "$UBATCH" ] && BATCH="$UBATCH"

# ROCm's own batch flags: the derived cap above unless you asked for one.
ROCM_UBATCH="$UBATCH"
ROCM_BATCH="$BATCH"
if [ "$UBATCH_SET" -eq 0 ]; then
    ROCM_UBATCH=$(( 67108864 / CTX ))
    [ "$ROCM_UBATCH" -gt 4096 ] && ROCM_UBATCH=4096
    [ "$ROCM_UBATCH" -lt 512 ]  && ROCM_UBATCH=512
    [ "$ROCM_BATCH" -lt "$ROCM_UBATCH" ] && ROCM_BATCH="$ROCM_UBATCH"
fi

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
        # -fa 0, not -fa 1: on the CPU backend flash attention collapses with
        # context. Measured 2026-09-08 (llama-bench, decode t/s at depth):
        #   35B    4K/16K/32K   -fa 0: 16.13 / 13.46 / 11.08
        #                       -fa 1: 15.29 /  8.14 /  4.31
        #   gemma  4K/16K/32K   -fa 0: 13.78 / 11.41 /  9.71
        #                       -fa 1: 12.55 /  7.94 /  5.29
        # -fa 1 is marginally better only at short context on the 35B, and worse
        # everywhere else — by a factor of two past 16K.
        exec "$SCRIPT_DIR/run-llamaserver-vulkan.sh" \
            "$MODEL" \
            -dev none -c "$CTX" --port "$PORT" --no-warmup \
            -fa 0 \
            "$@"
        ;;
    --rocm-docker)
        shift
        banner "ROCm 7.2 Docker"
        free_port
        # -fa 0 here but -fa 1 for baremetal below, deliberately: the Dockerfile
        # builds llama.cpp straight from the pinned ref and does NOT apply
        # patches/0001, so this image still has the slow gfx900 FA tile kernel
        # that costs 57% of prefill. Rebuild the image with the patch and this
        # branch should follow the baremetal one.
        exec "$SCRIPT_DIR/run-docker-rocm7.sh" \
            "$MODEL" \
            -ngl 99 -c "$CTX" --port "$PORT" --no-warmup \
            -fa 0 \
            -b "$ROCM_BATCH" -ub "$ROCM_UBATCH" \
            "$@"
        ;;
    --rocm|--rocm7|--baremetal)
        shift
        banner "ROCm 7.2 baremetal (Vega 8, gfx900)"
        free_port
        # -fa 1: this build carries patches/0001 (v_mad_mix_f32), which makes FA
        # win prefill and decode at every context. Decode at 32K on the 35B:
        # 5.9 t/s at -fa 0 against 15.9 at -fa 1. See run-rocm7-baremetal.sh.
        exec "$SCRIPT_DIR/run-rocm7-baremetal.sh" \
            "$MODEL" \
            -ngl 99 -c "$CTX" --port "$PORT" --no-warmup \
            -fa 1 \
            -b "$ROCM_BATCH" -ub "$ROCM_UBATCH" \
            "$@"
        ;;
    --help|-h)
        echo "Usage: $0 [--vulkan|--cpu|--rocm-docker|--rocm|--help]"
        echo ""
        echo "  (default)       Vulkan / Mesa RADV  — fastest almost everywhere; start here"
        echo "  --cpu           CPU only            — fallback; ~half the GPU prefill rate"
        echo "  --rocm          ROCm 7.2 baremetal  — wins 32k prefill on dense models,"
        echo "                                        and is the only path that runs gemma"
        echo "                                        at 32k. Needs the gfx900 backport"
        echo "                                        plus patches/0001 on the host"
        echo "  --rocm-docker   ROCm 7.2 in Docker  — self-contained, but the image lacks"
        echo "                                        patches/0001, so it stays on -fa 0"
        echo ""
        echo "Env vars: MODEL=  CTX=  PORT=  BATCH=  UBATCH="
        echo ""
        echo "UBATCH is derived from CTX per backend: a large enough attention dispatch"
        echo "hangs the Vulkan compute ring, and ROCm tolerates twice the product Vulkan"
        echo "does. Setting UBATCH= overrides both. See docs/benchmarks.md."
        echo ""
        exit 0
        ;;
    *)
        echo "Unknown option: $MODE"
        echo "Run '$0 --help' for usage."
        exit 1
        ;;
esac
