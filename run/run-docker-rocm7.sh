#!/bin/bash
#
# Run llama-server (ROCm 7 / Vega 8 build) inside Docker.
# Mirrors run-docker-rocm.sh but targets the ROCm 7 experimental image.
#
# With multiple AMD GPUs (Vega 8 iGPU + Radeon AI PRO R9700 dGPUs) this script
# auto-detects the Vega 8 render node by PCI device ID and passes ONLY that
# device into the container.  ROCR_VISIBLE_DEVICES=0 inside the container is
# therefore always the Vega 8, never an R9700.
#
# Usage:
#   ./run-docker-rocm7.sh /path/to/model.gguf [llama-server options]
#
# Build the image first (one-time, ~20–40 min):
#   docker build -t llama-rocm7-vega -f Dockerfile.rocm7-vega .

set -euo pipefail

IMAGE_NAME="llama-rocm7-vega"
MODEL="${1:-}"

if [ -z "$MODEL" ]; then
    echo "Usage: ./run-docker-rocm7.sh /path/to/model.gguf [llama-server options]"
    exit 1
fi

MODEL_DIR="$(dirname "$MODEL")"
MODEL_NAME="$(basename "$MODEL")"
shift

# ── Detect Vega 8 render node by PCI device ID ───────────────────────────────────────
# Detection lives in lib/vega8.sh so every script in this repo agrees. The render
# node number moves when discrete GPUs change (renderD128 as of September 2026),
# hence PCI-ID matching. Override with VEGA8_RENDER_NODE=/dev/dri/renderDXXX.
. "$(cd "$(dirname "$0")" && pwd)/../lib/vega8.sh"
VEGA8_RENDER_NODE="$(vega8_render_node || true)"

if [ -z "$VEGA8_RENDER_NODE" ]; then
    echo "⚠  Could not auto-detect Vega 8 render node (PCI ID $VEGA8_PCI_ID)."
    echo "   Falling back to passing all /dev/dri devices — ROCm may pick the wrong GPU."
    echo "   Override: VEGA8_RENDER_NODE=/dev/dri/renderDXXX ./run-docker-rocm7.sh ..."
    EXTRA_DEVICES="--device=/dev/dri"
else
    echo "  Vega 8 render node: $VEGA8_RENDER_NODE"
    EXTRA_DEVICES="--device=$VEGA8_RENDER_NODE"
fi

# ── Auto-build image if missing ───────────────────────────────────────────────
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Docker image '$IMAGE_NAME' not found. Building (this will take ~20-40 min)..."
    # Pass the shared pin so the image compiles the same llama.cpp commit as the
    # baremetal and Vulkan builds — see build/llama.cpp-ref.
    . "$(dirname "$0")/../build/llama-cpp-ref.sh"
    docker build -t "$IMAGE_NAME" \
        --build-arg "LLAMA_CPP_REF=$LLAMA_CPP_REF" \
        -f "$(dirname "$0")/../build/Dockerfile.rocm7-vega" "$(dirname "$0")/.."
fi

# ── Stop any existing container using this image or port 8080 ─────────────────
EXISTING=$(docker ps -q --filter "ancestor=$IMAGE_NAME")
if [ -n "$EXISTING" ]; then
    echo "  Stopping existing $IMAGE_NAME container(s): $EXISTING"
    docker stop $EXISTING
fi
PORT_CONFLICT=$(docker ps -q --filter "publish=8080")
if [ -n "$PORT_CONFLICT" ]; then
    echo "  Stopping container(s) holding port 8080: $PORT_CONFLICT"
    docker stop $PORT_CONFLICT
fi
echo "  Dir:    $MODEL_DIR → /models"
echo ""

# shellcheck disable=SC2086
# Use -it when stdin is a TTY (interactive), drop -t when piped/backgrounded
[[ -t 0 ]] && TTY_FLAG="-it" || TTY_FLAG="-i"
# TODO: honor PORT= (host mapping is hardcoded to -p 8080:8080; a --port flag
#       passed by the caller only changes the port *inside* the container).
docker run --rm $TTY_FLAG \
  --device=/dev/kfd \
  $EXTRA_DEVICES \
  --group-add=video \
  --group-add=render \
  --ipc=host \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  -e ROCR_VISIBLE_DEVICES=0 \
  -e HIP_VISIBLE_DEVICES=0 \
  -v "$MODEL_DIR:/models:ro" \
  -p 8080:8080 \
  "$IMAGE_NAME" \
  --host 0.0.0.0 \
  -m "/models/$MODEL_NAME" \
  -fa 1 \
  -ngl 99 \
  -b 4096 -ub 4096 \
  "$@"
# -fa 1 (flash attention ON): correct only because this image now applies
# patches/0001 at build time (v_mad_mix_f32, which gfx900 has and llama.cpp does
# not emit for it). Verified in the image on 2026-09-09, gemma decode at depth
# 16384: -fa 0 = 10.14 t/s, -fa 1 = 13.99 t/s, +38 %. On a stock ROCm build the
# opposite holds and FA more than halves prefill, so if you rebuild this image
# without patches/, put this back to -fa 0. Never leave -fa unset: it means
# -fa auto, whose probe succeeds on gfx900 either way and cannot tell a patched
# build from a stock one.
#
# -ub 4096 (full-batch prefill): the measured optimum. Worth +23-55 % prefill on
# a dense model and +3-9 % on an MoE against -ub 2048; -ub 8192 is a loss on
# every cell tested. See docs/benchmarks.md. Overridable — pass your own -b/-ub
# after the model.
