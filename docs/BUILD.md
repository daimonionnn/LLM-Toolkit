# Build Guide

Building llama.cpp from source with ROCm/HIP support for the AMD Vega 8 APU (gfx90c).

## Why Build From Source?

LM Studio's bundled ROCm backend only includes kernels for RDNA2+ GPUs (gfx1030 and newer). The Vega 8 iGPU uses the GCN 5 architecture (gfx90c), which isn't supported. Building llama.cpp ourselves lets us target `gfx900` — the closest official ROCm target to gfx90c.

> **⚠️ Current Status (April 2026):** The custom ROCm build compiles successfully and individual GPU kernel tests pass (SCALE, MUL_MAT across all quantization types). However, **actual model inference crashes** due to a fundamental version mismatch in Ubuntu 25.10's ROCm packages (clang-21 + HIP 5.7.1). This affects all models and all GPU layer counts, including `-ngl 0`. **Use Vulkan instead** — see [ARCHITECTURE.md](ARCHITECTURE.md#rocm-runtime-crash-analysis) for the full analysis.

## Prerequisites

### Required Packages

```bash
# HIP compiler and ROCm tools
sudo apt install -y hipcc

# hipBLAS (GPU-accelerated BLAS for ROCm)
sudo apt install -y libhipblas-dev

# Build tools
sudo apt install -y cmake git build-essential python3
```

On Ubuntu 25.10, `hipcc` pulls in `clang-21`, `llvm-21`, `libamdhip64-dev`, and `rocm-device-libs-21`.

### CMake Symlinks (Ubuntu Multiarch Fix)

Ubuntu puts HIP/ROCm CMake configs under `/usr/lib/x86_64-linux-gnu/cmake/` instead of the standard `/usr/lib/cmake/`. CMake can't find them without symlinks:

```bash
for dir in hip hip-lang hipblas rocblas rocsolver AMDDeviceLibs amd_comgr; do
    src="/usr/lib/x86_64-linux-gnu/cmake/$dir"
    dst="/usr/lib/cmake/$dir"
    if [ -d "$src" ] && [ ! -e "$dst" ]; then
        sudo ln -sf "$src" "$dst"
        echo "Linked: $dir"
    fi
done
```

The build script handles the `CMAKE_PREFIX_PATH` automatically, but the symlinks ensure CMake's `find_package()` works consistently.

### Verify Setup

```bash
hipcc --version        # Should show HIP version and clang
cmake --version        # Need 3.21+
rocminfo 2>/dev/null   # Should list your GPU (with HSA_OVERRIDE_GFX_VERSION=9.0.0)
```

## Building

```bash
cd LLMToolkit
chmod +x build-llamacpp-rocm-vega.sh
./build-llamacpp-rocm-vega.sh
```

### What the Build Script Does

1. **Clones/updates** llama.cpp from `ggml-org/llama.cpp` master branch
2. **Resets source** (`git checkout -- .`) to remove any previous patches
3. **Applies 6 patches** for HIP 5.7 compatibility (see [HIP57-PATCHES.md](HIP57-PATCHES.md))
4. **Configures CMake** with:
   - `GGML_HIP=ON` — Enable HIP/ROCm backend
   - `AMDGPU_TARGETS=gfx900:xnack+` — Target Vega architecture with xnack page-fault support (required for UMA)
   - `CMAKE_HIP_FLAGS="-mcode-object-version=5"` — Force COv5 (clang-21 defaults to COv6 which HIP 5.7 can't parse)
   - `GGML_HIP_UMA=ON` — Unified Memory Architecture (APU)
   - `LLAMA_BUILD_SERVER=ON` — Build the HTTP server
   - `CMAKE_HIP_COMPILER=/usr/bin/clang++-21`
5. **Builds** with all available CPU cores
6. **Installs** to `llama.cpp-rocm-vega/`

### Build Output

```
llama.cpp-rocm-vega/
├── bin/
│   ├── llama-server          # OpenAI-compatible HTTP API server
│   ├── llama-cli             # Interactive chat CLI
│   ├── llama-bench           # Benchmarking tool
│   ├── llama-quantize        # Model quantization
│   └── ...                   # ~30+ tools
└── lib/
    ├── libggml-hip.so        # HIP/ROCm GPU backend (gfx900 kernels)
    ├── libggml-base.so
    ├── libggml-cpu.so
    ├── libggml.so
    ├── libllama.so
    └── libmtmd.so
```

### Build Time

On a Ryzen 7 5700G (8 cores / 16 threads):
- First build: ~15-30 minutes (compiling ~150 HIP kernel files)
- Rebuild after source update: Varies (CMake incremental build)

### Rebuilding

The script automatically:
- `git fetch` + `git pull` to get latest llama.cpp
- `git checkout -- .` to reset any previous patches
- Re-applies all patches fresh
- Does a clean build (`rm -rf build`)

Just re-run:

```bash
./build-llamacpp-rocm-vega.sh
```

## Running

### Standalone llama-server (Recommended)

```bash
./run-llamaserver-rocm.sh ~/models/your-model.gguf -ngl 99
```

This wrapper sets all the required environment variables and launches the server. The API is available at `http://127.0.0.1:8080/v1`.

### Manual Run

```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_ENABLE_SDMA=0
export GGML_HIP_UMA=1
export GPU_MAX_ALLOC_PERCENT=100

./llama.cpp-rocm-vega/bin/llama-server \
    -m ~/models/your-model.gguf \
    -ngl 99 \
    --host 0.0.0.0 --port 8080
```

### Interactive CLI Chat

```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_ENABLE_SDMA=0
export GGML_HIP_UMA=1

./llama.cpp-rocm-vega/bin/llama-cli \
    -m ~/models/your-model.gguf \
    -ngl 99 \
    -c 4096 \
    --chat-template chatml
```

### Key CLI Options

| Option | Description |
|--------|-------------|
| `-m PATH` | Path to GGUF model file |
| `-ngl N` | Number of layers to offload to GPU (`99` = all) |
| `-c N` | Context window size (tokens) |
| `--host IP` | Listen address (default: `127.0.0.1`) |
| `--port N` | Listen port (default: `8080`) |
| `-t N` | Number of CPU threads |
| `--chat-template NAME` | Chat template (chatml, llama2, etc.) |

## Connecting to LM Studio

You can run llama-server alongside LM Studio and connect to it as a remote endpoint:

1. Start the server: `./run-llamaserver-rocm.sh model.gguf -ngl 99`
2. In LM Studio: **Developer → Connect to external endpoint**
3. Enter: `http://127.0.0.1:8080/v1`

## Replacing LM Studio's ROCm Backend (Experimental)

> **Warning:** This may break LM Studio. Back up first.

```bash
BACKEND="$HOME/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-amd-rocm-avx2-2.13.0"

# Backup
cp -a "$BACKEND" "$BACKEND.bak"

# Replace libs
cp llama.cpp-rocm-vega/lib/libggml-hip.so "$BACKEND/"
cp llama.cpp-rocm-vega/lib/libggml-base.so "$BACKEND/"
cp llama.cpp-rocm-vega/lib/libggml-cpu.so "$BACKEND/"
cp llama.cpp-rocm-vega/lib/libllama.so "$BACKEND/"
```

This is risky due to potential ABI mismatches between our build and LM Studio's engine. The standalone server approach is much safer.
