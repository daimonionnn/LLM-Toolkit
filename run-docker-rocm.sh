#!/bin/bash

# Configuration
IMAGE_NAME="llama-server-rocm-vega"
MODEL="$1"

if [ -z "$MODEL" ]; then
    echo "Usage: ./run-docker-rocm.sh /path/to/model.gguf [llama-server options]"
    exit 1
fi

MODEL_DIR=$(dirname "$MODEL")
MODEL_NAME=$(basename "$MODEL")

shift

# Build the Docker image if it doesn't exist
if ! docker image inspect "$IMAGE_NAME" &>/dev/null; then
    echo "Docker image '$IMAGE_NAME' not found. Building it... this will take a few minutes."
    docker build -t "$IMAGE_NAME" -f Dockerfile.rocm64 .
fi

echo "Starting llama-server inside ROCm docker..."
echo "Model: $MODEL_NAME"
echo "Mapping: $MODEL_DIR -> /models"

docker run --rm -it \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add=video \
  --group-add=render \
  --ipc=host \
  --security-opt seccomp=unconfined \
  --ulimit memlock=-1 \
  -v "$MODEL_DIR:/models:ro" \
  -p 8080:8080 \
  "$IMAGE_NAME" \
  --host 0.0.0.0 \
  -m "/models/$MODEL_NAME" \
  -fa 1 \
  -ngl 99 \
# --no-kv-offload
  "$@"