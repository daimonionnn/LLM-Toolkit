#!/bin/bash
#
# Build llama.cpp with ROCm/HIP support for AMD Vega 8 (gfx90c / gfx900)
#
# LM Studio's bundled ROCm backend only supports RDNA2+ (gfx1030+).
# This script builds llama.cpp from source targeting gfx900, which is the
# closest supported architecture to gfx90c (Vega 8 APU / Renoir).
#
# The resulting binary can replace LM Studio's ROCm backend or be used
# standalone as a llama-server.
#
# Prerequisites:
#   - ROCm 6.x installed (rocm-dev, hip-dev)
#   - CMake 3.21+
#   - git
#

set -euo pipefail

# ─── Configuration ───
LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_BRANCH="master"
BUILD_DIR="$(dirname "$0")/llama.cpp-build"
INSTALL_DIR="$(dirname "$0")/llama.cpp-rocm-vega"
# Target gfx900 (Vega 10) — closest to gfx90c (Vega 8 APU)
# Use plain "gfx900" without xnack qualifier, matching AMD's own rocBLAS
# convention (see ROCm/rocBLAS CMakeLists.txt). This produces a generic
# code object compatible with both xnack+ and xnack- at runtime.
# Ref: https://github.com/ROCm/rocBLAS/blob/9391ecc/CMakeLists.txt#L82
AMDGPU_TARGET="gfx900"
JOBS=$(nproc)

echo "═══════════════════════════════════════════════════════════"
echo "  Build llama.cpp with ROCm/HIP for Vega 8 (gfx90c)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Target GPU arch:  $AMDGPU_TARGET"
echo "  Build dir:        $BUILD_DIR"
echo "  Install dir:      $INSTALL_DIR"
echo "  Parallel jobs:    $JOBS"
echo ""

# ─── Check prerequisites ───
check_prereqs() {
    local ok=true

    if ! command -v cmake &>/dev/null; then
        echo "✗  cmake not found. Install: sudo apt install cmake"
        ok=false
    fi

    if ! command -v hipcc &>/dev/null; then
        echo "✗  hipcc not found. Install ROCm HIP SDK:"
        echo "   sudo apt install rocm-dev hip-dev rocm-hip-sdk"
        ok=false
    fi

    if ! command -v git &>/dev/null; then
        echo "✗  git not found. Install: sudo apt install git"
        ok=false
    fi

    if ! $ok; then
        echo ""
        echo "Prerequisites missing. Install them and retry."
        exit 1
    fi
    echo "✓  All build prerequisites found"
    echo ""
}

check_prereqs

# ─── Clone / update source ───
if [ -d "$BUILD_DIR/llama.cpp" ]; then
    echo "  Updating existing llama.cpp source..."
    cd "$BUILD_DIR/llama.cpp"
    git checkout -- .
    git fetch origin
    git checkout "$LLAMA_CPP_BRANCH"
    git pull origin "$LLAMA_CPP_BRANCH"
else
    echo "  Cloning llama.cpp..."
    mkdir -p "$BUILD_DIR"
    cd "$BUILD_DIR"
    git clone --depth 1 -b "$LLAMA_CPP_BRANCH" "$LLAMA_CPP_REPO" llama.cpp
    cd llama.cpp
fi
echo ""

# ─── Build with HIP/ROCm for gfx900 ───
echo "  Configuring CMake with GGML_HIP=ON, target=$AMDGPU_TARGET ..."
echo ""

rm -rf build

# On Debian/Ubuntu, HIP CMake configs live under the multiarch path
# rather than the standard /usr/lib/cmake/ that CMake expects
HIP_CMAKE_PREFIX="/usr/lib/x86_64-linux-gnu/cmake"
if [ -f "$HIP_CMAKE_PREFIX/hip-lang/hip-lang-config.cmake" ]; then
    echo "  Found HIP CMake configs at: $HIP_CMAKE_PREFIX"
    EXTRA_CMAKE_ARGS="-DCMAKE_PREFIX_PATH=$HIP_CMAKE_PREFIX"
else
    EXTRA_CMAKE_ARGS=""
fi

# CMake needs to find clang++-21 as the HIP compiler
# On Ubuntu with llvm-21, it's not auto-detected
export HIPCXX="/usr/bin/clang++-21"
export HIP_CLANG_PATH="/usr/lib/llvm-21/bin"
echo "  Using HIP compiler: $HIPCXX"

# Patch: relax HIP version check (Ubuntu packages HIP 5.7, llama.cpp wants 6.1+)
# The core HIP/hipBLAS API is compatible enough for our use case
HIP_CMAKELISTS="ggml/src/ggml-hip/CMakeLists.txt"
if grep -q 'VERSION_LESS 6.1' "$HIP_CMAKELISTS" 2>/dev/null; then
    echo "  Patching HIP version check (5.7 → relaxed)..."
    sed -i 's/if (${hip_VERSION} VERSION_LESS 6.1)/if (${hip_VERSION} VERSION_LESS 5.5)/' "$HIP_CMAKELISTS"
fi

# Patch: HIP 5.7 requires 3 args for hipStreamWaitEvent (stream, event, flags)
# HIP 6.x added a 2-arg overload. Fix calls missing the flags argument.
CUDA_CU="ggml/src/ggml-cuda/ggml-cuda.cu"
if grep -q 'cudaStreamWaitEvent.*fork_event)' "$CUDA_CU" 2>/dev/null; then
    echo "  Patching hipStreamWaitEvent 2-arg calls → 3-arg..."
    sed -i 's/cudaStreamWaitEvent(stream, concurrent_event->fork_event)/cudaStreamWaitEvent(stream, concurrent_event->fork_event, 0)/g' "$CUDA_CU"
    sed -i 's/cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events\[i - 1\])/cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1], 0)/g' "$CUDA_CU"
fi

# Patch: HIP 5.7 hipblasStrsmBatched uses float*const* not const float**
# Need to cast away const on inner pointers for older hipBLAS API
SOLVE_TRI="ggml/src/ggml-cuda/solve_tri.cu"
if [ -f "$SOLVE_TRI" ]; then
    echo "  Patching solve_tri.cu for hipBLAS 5.7 const correctness..."
    # Replace the specific cublasStrsmBatched call with proper casts for HIP 5.7
    python3 -c "
import re
with open('$SOLVE_TRI', 'r') as f:
    content = f.read()
# Replace the strsmBatched call - cast A to float*const* and X to float**
old = 'CUBLAS_CHECK(cublasStrsmBatched(ctx.cublas_handle(id), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,\n                                    CUBLAS_DIAG_NON_UNIT, k, n, &alpha, A_ptrs_dev, n, X_ptrs_dev, k, total_batches));'
new = 'CUBLAS_CHECK(cublasStrsmBatched(ctx.cublas_handle(id), CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_N,\n                                    CUBLAS_DIAG_NON_UNIT, k, n, &alpha, (float *const *)A_ptrs_dev, n, X_ptrs_dev, k, total_batches));'
content = content.replace(old, new)
with open('$SOLVE_TRI', 'w') as f:
    f.write(content)
print('  Done')
"
fi

# HIP 5.7 bfloat16 header functions are not marked inline, causing
# multiple definition errors at link time. Work around with linker flag.
#
# clang-21 defaults to Code Object V6 (ELFABIVERSION=4), but the
# HIP 5.7 runtime (libamdhip64) only supports up to COv5.
# Force COv5 so COMGR can parse the embedded GPU code objects.
cmake -B build \
    -DGGML_HIP=ON \
    -DAMDGPU_TARGETS="$AMDGPU_TARGET" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DCMAKE_HIP_COMPILER="$HIPCXX" \
    -DCMAKE_HIP_FLAGS="-mcode-object-version=5" \
    -DCMAKE_INSTALL_RPATH="\$ORIGIN" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DGGML_HIP_UMA=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,muldefs" \
    $EXTRA_CMAKE_ARGS

echo ""
echo "  Building with $JOBS parallel jobs..."
echo "  (This may take 10-30 minutes depending on your CPU)"
echo ""

cmake --build build -j "$JOBS"

echo ""
echo "  Installing to $INSTALL_DIR ..."
cmake --install build --prefix "$INSTALL_DIR"

echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Build complete!"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Binaries installed to: $INSTALL_DIR/bin/"
echo ""
echo "  ─── Option A: Use llama-server standalone ───"
echo ""
echo "    export HSA_OVERRIDE_GFX_VERSION=9.0.0"
echo "    export HSA_ENABLE_SDMA=0"
echo "    export GGML_HIP_UMA=1"
echo "    $INSTALL_DIR/bin/llama-server \\"
echo "      -m /path/to/model.gguf \\"
echo "      -ngl 99 \\"
echo "      --host 0.0.0.0 --port 8080"
echo ""
echo "  ─── Option B: Replace LM Studio ROCm backend ───"
echo ""
echo "    # Dependencies needed to fix lib metadata:"
echo "    # python3 -m pip install --user --break-system-packages patchelf"
echo ""
echo "    # Back up existing backend first:"
echo "    BACKEND_DIR=\"\$HOME/.lmstudio/extensions/backends\""
echo "    ROCM_DIR=\"\$BACKEND_DIR/llama.cpp-linux-x86_64-amd-rocm-avx2-2.13.0\""
echo "    cp -a \"\$ROCM_DIR\" \"\$ROCM_DIR.bak\""
echo ""
echo "    # Replace the critical libraries:"
echo "    cp -a $INSTALL_DIR/lib/libggml*.so* \"\$ROCM_DIR/\""
echo "    cp -a $INSTALL_DIR/lib/libllama*.so* \"\$ROCM_DIR/\""
echo ""
echo "    # Fix RUNPATH for dependencies so LM Studio's engine can find them:"
echo "    for f in \"\$ROCM_DIR\"/*.so; do"
echo "        if [ -f \"\$f\" ] && [ ! -L \"\$f\" ]; then"
echo "            ~/.local/bin/patchelf --set-rpath '\$ORIGIN' \"\$f\""
echo "        fi"
echo "    done"
echo ""
echo "    # NOTE: This may or may not work depending on ABI compatibility"
echo "    # between the custom build and LM Studio's engine. The standalone"
echo "    # llama-server (Option A) is the safer choice."
echo ""
echo "    # IMPORTANT: If LM Studio crashes with 'Exit code: null' when loading a model,"
echo "    # it wasn't launched with the required ROCm environment variables."
echo "    # Close LM Studio and launch it from a terminal with:"
echo "    #   export HSA_OVERRIDE_GFX_VERSION=9.0.0"
echo "    #   export HSA_ENABLE_SDMA=0"
echo "    #   export HSA_XNACK=0"
echo "    #   export GGML_HIP_UMA=0"
echo "    #   /path/to/lmstudio"
echo ""
echo "  ─── Option C: Use with LM Studio API ───"
echo ""
echo "    # Run llama-server, then point LM Studio to it as a remote model."
echo "    # In LM Studio: Developer → Connect to external endpoint"
echo ""
