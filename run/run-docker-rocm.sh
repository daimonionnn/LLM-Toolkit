#!/bin/bash
#
# Run llama-server (ROCm 6) inside Docker, targeting the Vega 8 iGPU.
#
# With multiple AMD GPUs in the system (e.g. Vega 8 iGPU + Radeon AI PRO R9700
# dGPUs) this script auto-detects the Vega 8 render node by PCI device ID and
# passes ONLY that device into the container, so ROCR_VISIBLE_DEVICES=0 inside
# the container is always the Vega 8.
#
# Usage:
#   ./run-docker-rocm.sh /path/to/model.gguf [llama-server options]

set -euo pipefail

IMAGE_NAME="llama-server-rocm-vega"
MODEL="${1:-}"

if [ -z "$MODEL" ]; then
    echo "Usage: ./run-docker-rocm.sh /path/to/model.gguf [llama-server options]"
    exit 1
fi

MODEL_DIR=$(dirname "$MODEL")
MODEL_NAME=$(basename "$MODEL")
shift

# ── Detect Vega 8 render node by PCI device ID ────────────────────────────────
# Detection lives in lib/vega8.sh so every script in this repo agrees. Override
# with VEGA8_RENDER_NODE=/dev/dri/renderDXXX.
# TODO: honor PORT= (host mapping below is hardcoded to -p 8080:8080).
. "$(cd "$(dirname "$0")" && pwd)/../lib/vega8.sh"
VEGA8_RENDER_NODE="$(vega8_render_node || true)"

if [ -z "$VEGA8_RENDER_NODE" ]; then
    echo "⚠  Could not auto-detect Vega 8 render node (PCI ID $VEGA8_PCI_ID)."
    echo "   Falling back to passing all /dev/dri devices — ROCm may pick the wrong GPU."
    echo "   Override: VEGA8_RENDER_NODE=/dev/dri/renderDXXX ./run-docker-rocm.sh ..."
    EXTRA_DEVICES="--device=/dev/dri"
else
    echo "  Vega 8 render node: $VEGA8_RENDER_NODE"
    EXTRA_DEVICES="--device=$VEGA8_RENDER_NODE"
fi

# ── Build image if missing ───────────────────────────────────────────────────
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Docker image '$IMAGE_NAME' not found. Building (this will take ~10 min)..."
    # Pass the shared pin — see build/llama.cpp-ref.
    . "$(dirname "$0")/../build/llama-cpp-ref.sh"
    docker build -t "$IMAGE_NAME" \
        --build-arg "LLAMA_CPP_REF=$LLAMA_CPP_REF" \
        -f "$(dirname "$0")/../build/Dockerfile.rocm64" "$(dirname "$0")/.."
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

echo "════════════════════════════════════════════════════════"
echo "  llama-server  ·  ROCm 6  ·  Vega 8 iGPU"
echo "════════════════════════════════════════════════════════"
echo "  Model:  $MODEL_NAME"
echo "  Dir:    $MODEL_DIR -> /models"
echo ""

# shellcheck disable=SC2086
# Use -it when stdin is a TTY (interactive), drop -t when piped/backgrounded
[[ -t 0 ]] && TTY_FLAG="-it" || TTY_FLAG="-i"
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
  -fa 0 \
  -ngl 99 \
  "$@"