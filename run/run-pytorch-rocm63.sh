#!/usr/bin/env bash
# PyTorch on the Vega 8 iGPU, via the ROCm 6.3 wheel.
#
#   ./run/run-pytorch-rocm63.sh                 # run the smoke test
#   ./run/run-pytorch-rocm63.sh python3         # interactive python
#   ./run/run-pytorch-rocm63.sh bash            # shell
#
# Build the image first:
#   docker build -t pytorch-rocm63-vega -f build/Dockerfile.pytorch-rocm63-vega .
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
. "$SCRIPT_DIR/../lib/vega8.sh"

IMAGE="${PYTORCH_IMAGE:-pytorch-rocm63-vega}"
RENDER_NODE="$(vega8_render_node || true)"
if [ -z "$RENDER_NODE" ]; then
    echo "✗  Vega 8 render node not found (PCI $VEGA8_PCI_ID)." >&2
    echo "   Override with VEGA8_RENDER_NODE=/dev/dri/renderDXXX" >&2
    exit 1
fi

# Numeric gids, not names. This image is Debian slim and has no `render` group,
# so --group-add=render makes Docker refuse to start the container -- and it
# fails with "Unable to find group render", after which the container is dead
# and `docker logs` has nothing, which looks exactly like PyTorch hanging.
GID_RENDER="$(getent group render | cut -d: -f3)"
GID_VIDEO="$(getent group video  | cut -d: -f3)"

echo "  Image:  $IMAGE"
echo "  Device: $RENDER_NODE  (render gid $GID_RENDER, video gid $GID_VIDEO)"
echo ""

ARGS=()
[ $# -gt 0 ] && ARGS=(--entrypoint "$1") && shift

exec docker run --rm -it \
    --device=/dev/kfd \
    --device="$RENDER_NODE" \
    --group-add "$GID_RENDER" \
    --group-add "$GID_VIDEO" \
    --ipc=host \
    --security-opt seccomp=unconfined \
    --ulimit memlock=-1 \
    "${ARGS[@]}" \
    "$IMAGE" "$@"
