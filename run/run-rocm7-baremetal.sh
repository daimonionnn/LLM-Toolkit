#!/bin/bash
#
# Run llama-server (ROCm 7 — baremetal / host install) on Vega 8 iGPU.
#
# Requires ROCm 7.x installed in /opt/rocm and llama.cpp built for gfx900.
# See:
#   setup/install-rocm7-host.sh          — install ROCm 7 on the host
#   build/build-llamacpp-rocm7-baremetal.sh — build llama.cpp (gfx900)
#
# Usage:
#   ./run/run-rocm7-baremetal.sh /path/to/model.gguf [llama-server options]
#   ./run/run-rocm7-baremetal.sh /path/to/model.gguf -ngl 99 -c 8192 -fa 1
#
# Multi-GPU system note:
#   Auto-detects Vega 8 by its rocminfo agent index (gfx90x family).
#   If auto-detect fails, set: VEGA8_ROCM_DEVICE=0 (or the correct index).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_BIN="$REPO_DIR/llm/rocm7-vega/bin/llama-server"
ROCM_LIB_DIR="$REPO_DIR/llm/rocm7-vega/lib"
ROCM_PATH="${ROCM_PATH:-/opt/rocm}"

# ─── Validate ────────────────────────────────────────────────────────────────

if [ ! -x "$LLAMA_BIN" ]; then
    echo "✗  llama-server not found at: $LLAMA_BIN"
    echo "   Build first: bash build/build-llamacpp-rocm7-baremetal.sh"
    exit 1
fi

if [ -z "${1:-}" ]; then
    echo "Usage: $0 /path/to/model.gguf [llama-server options]"
    echo ""
    echo "  -ngl 99       Offload all layers to GPU"
    echo "  -c 8192       Context size"
    echo "  -fa 1         Flash attention (ON — this build carries patches/0001)"
    echo "  --no-warmup   Skip warmup inference"
    echo ""
    echo "Example:"
    echo "  $0 ~/.lmstudio/models/.../model.gguf -ngl 99 -c 8192 -fa 1"
    exit 0
fi

MODEL="$1"
shift

if [ ! -f "$MODEL" ]; then
    echo "✗  Model file not found: $MODEL"
    exit 1
fi

# ─── Detect Vega 8 ROCm agent index ─────────────────────────────────────────
# rocminfo prints each agent's short "Name: gfxXXX" line BEFORE its
# "Device Type: GPU" line, so we remember the last seen name and assign
# 0-based GPU indices in enumeration order (same order ROCR_VISIBLE_DEVICES
# uses). Vega APUs are gfx900/gfx902/gfx909/gfx90c.

detect_vega8_rocm_index() {
    if [ -n "${VEGA8_ROCM_DEVICE:-}" ]; then
        echo "$VEGA8_ROCM_DEVICE"
        return
    fi

    local rocminfo_bin="$ROCM_PATH/bin/rocminfo"
    [ -x "$rocminfo_bin" ] || rocminfo_bin="$(command -v rocminfo || true)"
    if [ -z "$rocminfo_bin" ]; then
        echo "0"   # fallback
        return
    fi

    "$rocminfo_bin" 2>/dev/null | awk '
        $1 == "Name:" && $2 ~ /^gfx/  { name = $2 }
        /Device Type:[[:space:]]+GPU/ {
            # print gpu+0, not gpu: when the Vega 8 is the first GPU its index
            # is 0 and `gpu` was never assigned, so bare `print gpu` emits an
            # EMPTY string. That becomes ROCR_VISIBLE_DEVICES="", which hides
            # every GPU and silently falls back to CPU ("no usable GPU found").
            # Masked until September 2026, when the dGPUs left and the Vega
            # became index 0 for the first time.
            if (name ~ /^gfx90[029c]$/) { print gpu+0; found = 1; exit }
            gpu++
        }
        END { if (!found) print 0 }
    '
}

VEGA8_IDX=$(detect_vega8_rocm_index)
echo "  Vega 8 ROCm device index: $VEGA8_IDX"

# ─── Preflight: host ROCm must still support the gfx900 path ────────────────
# This launcher needs two things from the host ROCm install:
#   1. gfx900 rocBLAS tensile kernels (backported from ROCm 6.3.4 by
#      build/build-llamacpp-rocm7-baremetal.sh)
#   2. a ROCr runtime that accepts HSA_OVERRIDE_GFX_VERSION=9.0.0
# AMD's newer modular packages (e.g. amdrocm-core7.14-gfx120x) provide
# neither — installing them replaces /opt/rocm and breaks this path.
# In that case use the self-contained Docker image instead:
#   ./run/run-docker-rocm7.sh /path/to/model.gguf
# Set SKIP_ROCM_CHECKS=1 to bypass these checks.

if [ -z "${SKIP_ROCM_CHECKS:-}" ]; then
    GFX900_KERNELS=$(find -L "$ROCM_PATH/lib/rocblas/library" -name '*gfx900*' 2>/dev/null | wc -l)
    if [ "$GFX900_KERNELS" -eq 0 ]; then
        echo "✗  No gfx900 rocBLAS kernels in $ROCM_PATH/lib/rocblas/library"
        echo "   The host ROCm install cannot run llama.cpp on the Vega 8 (first"
        echo "   GEMM will fail). This happens when the gfx900 tensile backport is"
        echo "   missing or the host ROCm was replaced (e.g. by amdrocm-core gfx120x"
        echo "   packages for RDNA4 cards)."
        echo ""
        echo "   Options:"
        echo "     • Use Docker (self-contained ROCm 7.2 + backport, still works):"
        echo "         ./run/run-docker-rocm7.sh $MODEL"
        echo "     • Or reinstall ROCm 7.2 + backport:  setup/install-rocm7-host.sh"
        echo "       then build/build-llamacpp-rocm7-baremetal.sh"
        exit 1
    fi
    if ! HSA_OVERRIDE_GFX_VERSION=9.0.0 ROCR_VISIBLE_DEVICES="$VEGA8_IDX" \
            "$ROCM_PATH/bin/rocminfo" >/dev/null 2>&1; then
        echo "✗  HSA_OVERRIDE_GFX_VERSION=9.0.0 crashes this ROCr runtime"
        echo "   (newer modular ROCm builds reject the gfx version override)."
        echo "   Use Docker instead:  ./run/run-docker-rocm7.sh $MODEL"
        exit 1
    fi
fi

# ─── Environment ─────────────────────────────────────────────────────────────
#
# ROCR_VISIBLE_DEVICES=<idx> — expose only the Vega 8 to the HSA runtime.
# HIP_VISIBLE_DEVICES=0 — HIP indexes into the ROCR-filtered list, where the
#   Vega 8 is the only (first) device. Do NOT set this to the rocminfo index.
# HSA_OVERRIDE_GFX_VERSION=9.0.0 — Vega 8 APU (gfx90c) overridden to gfx900
#   so that gfx900 code objects and tensile kernels are loaded.
# HSA_ENABLE_SDMA=0 — disable System DMA; required for stability on Vega 8 APU
#   (SDMA engine not present / unreliable on integrated Vega).
# HSA_XNACK=0 — XNACK=1 hard-freezes the entire PC on Vega 8.
# GPU_MAX_ALLOC_PERCENT=100 — allow full GTT allocation (64 GB on this system).
#
# Note: GGML_HIP_UMA was removed from llama.cpp; plain hipMalloc into GTT is
# what the May 2026 benchmarks used and needs no extra env var.

export ROCR_VISIBLE_DEVICES="$VEGA8_IDX"
export HIP_VISIBLE_DEVICES=0
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_ENABLE_SDMA=0
export HSA_XNACK=0
export GPU_MAX_ALLOC_PERCENT=100

# Prepend the baremetal lib dir (contains RPATH-relative libs from build),
# then the ROCm system libs.
export LD_LIBRARY_PATH="${ROCM_LIB_DIR}:${ROCM_PATH}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

# ─── Launch ───────────────────────────────────────────────────────────────────

echo "  Model:  $(basename "$MODEL")"
echo "  Binary: $LLAMA_BIN"
echo "  ROCm:   $ROCM_PATH"
echo "  Env:    ROCR_VISIBLE_DEVICES=$VEGA8_IDX  HSA_OVERRIDE_GFX_VERSION=9.0.0"
echo ""

exec "$LLAMA_BIN" \
    -m "$MODEL" \
    --host 0.0.0.0 \
    --port 8080 \
    -fa 1 \
    -b 4096 -ub 4096 \
    "$@"
# All three flags are defaults, not constraints: "$@" comes last, so anything
# you pass on the command line overrides them.
#
# -fa 1 (flash attention ON): correct ONLY because build/apply-patches.sh put
# patches/0001 into this build. That patch makes the FA KQ accumulate use
# v_mad_mix_f32, which gfx900 has and llama.cpp does not emit for it. With it,
# -fa 1 wins both prefill and decode at every context (llama-bench, 2026-09-08):
#   35B    prefill 4K/16K/32K   -fa 0: 136.5 / 119.3 /  96.6
#                               -fa 1: 139.3 / 123.7 / 107.9
#   35B    decode  4K/16K/32K   -fa 0:  15.2 /   9.4 /   5.9
#                               -fa 1:  18.8 /  17.5 /  15.9
#   gemma  decode  4K/16K/32K   -fa 0:  12.0 /   9.3 /   7.1
#                               -fa 1:  15.0 /  13.6 /  12.2
# On a STOCK ROCm build the opposite holds and -fa 0 is required: the generic FA
# tile kernel falls back to ~5 VALU ops per 2 MACs and more than halves prefill
# (gemma, 3330-token prompt: -fa 0 = 112.8 t/s, -fa 1 = 48.9). That is why
# run-docker-rocm7.sh still passes -fa 0 — the Dockerfile does not apply the patch.
#
# Never leave -fa unset. It means -fa auto, which probes the backend, finds that
# the tile kernel compiles for gfx900, and enables FA regardless of which build
# you are running (verified: -fa auto = 48.9 t/s on the stock build, identical
# to -fa 1). The probe cannot tell a patched build from an unpatched one.
#
# -ub 4096 (full-batch prefill): the single largest ROCm win found so far.
# Measured 2026-09-08 with llama-bench on the 35B (cold prefill, -r 2):
#   3330-token prompt: ub 512 = 84.2 t/s, 1024 = 110.0, 2048 = 130.2, 4096 = 143.5
#   937-token prompt:  ub 512 = 84.5,     1024 = 116.0, 2048 = 115.9, 4096 = 115.0
# The optimum is roughly "ubatch >= prompt length": a 256-expert MoE with 8 active
# experts spreads a 512-token ubatch over ~16 tokens per expert, leaving MMQ's
# 64-column tiles three-quarters empty. Costs ~2 GB of extra GTT at -c 8192
# (22.9 vs 20.8 GB on the 35B). No decode cost.
