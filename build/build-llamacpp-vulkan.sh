#!/bin/bash
#
# Build llama.cpp with the Vulkan backend for the Vega 8 iGPU (Mesa RADV).
#
# Vulkan is this rig's default serving backend — it needs no ROCm at all, just
# Mesa RADV and the Vulkan loader, and on Vega 8 it gives the best decode
# throughput of any backend (see docs/benchmarks.md).
#
# It is also what `bench/run-all-benchmarks.sh` uses for BOTH its Vulkan rows
# and its CPU rows (the CPU rows run this same binary with `-ngl 0`), so the
# benchmark harness cannot run at all without this build.
#
# Shares the llama.cpp checkout in llm/build/ with the ROCm build script, so
# both backends are compiled from the identical commit — a prerequisite for
# comparing their numbers.
#
# Prerequisites (installed by setup/bootstrap-host.sh):
#   cmake, build-essential, libvulkan-dev, glslang-tools, mesa-vulkan-drivers
#
# Usage:
#   bash build/build-llamacpp-vulkan.sh

set -euo pipefail

LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp.git"
# Shared pin — see build/llama.cpp-ref. Keeping the Vulkan and ROCm builds on the
# same commit is what makes the backend comparison in docs/benchmarks.md valid.
# shellcheck source=build/llama-cpp-ref.sh
. "$(dirname "$0")/llama-cpp-ref.sh"
BUILD_DIR="$(realpath -m "$(dirname "$0")/../llm/build")"
INSTALL_DIR="$(realpath -m "$(dirname "$0")/../llm/vulkan")"
JOBS=$(nproc)

echo "═══════════════════════════════════════════════════════════"
echo "  Build llama.cpp  ·  Vulkan (Mesa RADV)  ·  Vega 8"
echo "═══════════════════════════════════════════════════════════"
echo "  Build dir   : $BUILD_DIR"
echo "  Install dir : $INSTALL_DIR"
echo "  Parallel    : $JOBS"
echo ""

check_prereqs() {
    local ok=true
    for tool in cmake git glslc; do
        if ! command -v "$tool" &>/dev/null; then
            echo "✗  $tool not found"
            ok=false
        fi
    done
    if ! [ -f /usr/include/vulkan/vulkan.h ] && ! [ -f /usr/include/vulkan/vulkan_core.h ]; then
        echo "✗  Vulkan headers not found — install libvulkan-dev"
        ok=false
    fi
    # ggml-vulkan does find_package(SPIRV-Headers CONFIG REQUIRED); catch it here
    # rather than 200 lines into a CMake configure log.
    # `ls A B` fails if EITHER path is missing, so test the two locations
    # separately — Debian/Ubuntu ships the config under /usr/share/cmake.
    if ! compgen -G "/usr/lib/*/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake" >/dev/null \
       && [ ! -f /usr/share/cmake/SPIRV-Headers/SPIRV-HeadersConfig.cmake ]; then
        echo "✗  SPIRV-Headers CMake config not found — install spirv-headers"
        ok=false
    fi
    $ok || { echo ""; echo "Install: sudo apt install -y cmake git glslc glslang-tools libvulkan-dev spirv-headers spirv-tools"; exit 1; }

    # A build is useless if the runtime cannot see the iGPU.
    if command -v vulkaninfo &>/dev/null; then
        local dev
        dev=$(vulkaninfo --summary 2>/dev/null | grep -m1 'deviceName' | cut -d= -f2- | xargs || true)
        echo "✓  Vulkan device: ${dev:-none detected}"
    fi
    echo "✓  Prerequisites found"
    echo ""
}

fetch_source() {
    echo "─── Fetching llama.cpp source ────────────────────────────────"
    if [ -d "$BUILD_DIR/llama.cpp/.git" ]; then
        echo "  Reusing existing checkout (shared with the ROCm build)"
        cd "$BUILD_DIR/llama.cpp"
        if [ "$(git rev-parse HEAD)" != "$LLAMA_CPP_REF" ]; then
            echo "  Checkout is at $(git rev-parse --short HEAD), pin is ${LLAMA_CPP_REF:0:7} — fetching"
            git fetch --depth 1 origin "$LLAMA_CPP_REF"
            git checkout --detach FETCH_HEAD
        fi
    else
        mkdir -p "$BUILD_DIR"
        cd "$BUILD_DIR"
        mkdir -p llama.cpp && cd llama.cpp
        git init -q
        git remote add origin "$LLAMA_CPP_REPO"
        git fetch --depth 1 origin "$LLAMA_CPP_REF"
        git checkout --detach FETCH_HEAD
    fi
    echo "  Commit: $(git log --oneline -1)"
    echo ""
}

build() {
    echo "─── Configuring ─────────────────────────────────────────────"
    # Flags mirror build-llamacpp-rocm7-baremetal.sh so the two backends differ
    # only in the GPU backend itself.
    #   GGML_BACKEND_DL       — backends load as .so at runtime
    #   GGML_CPU_ALL_VARIANTS — needed for the harness's CPU rows off this binary
    #   CMAKE_INSTALL_RPATH   — upstream installs libllama-*-impl.so into lib/,
    #                           while executables land in bin/, so $ORIGIN alone
    #                           leaves the installed binaries unable to start.
    cmake -B build-vulkan \
        -DGGML_VULKAN=ON \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
        -DGGML_BACKEND_DL=ON \
        -DGGML_CPU_ALL_VARIANTS=ON \
        -DLLAMA_BUILD_SERVER=ON \
        -DLLAMA_BUILD_EXAMPLES=ON \
        -DCMAKE_INSTALL_RPATH='$ORIGIN;$ORIGIN/../lib' \
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

    echo ""
    echo "─── Building ────────────────────────────────────────────────"
    cmake --build build-vulkan -j "$JOBS"

    echo ""
    echo "─── Installing ──────────────────────────────────────────────"
    cmake --install build-vulkan --prefix "$INSTALL_DIR"
    echo ""
}

verify() {
    echo "─── Verifying ───────────────────────────────────────────────"
    local bin="$INSTALL_DIR/bin/llama-server"
    [ -x "$bin" ] || { echo "✗  $bin missing"; exit 1; }
    echo "  ✓  $bin"

    # Runs without LD_LIBRARY_PATH only if the RPATH above is correct.
    # Capture the output first: piping into `grep -q` under `set -o pipefail`
    # makes grep exit at the first match, llama-bench die of SIGPIPE (141), and
    # the pipeline report failure even though the device WAS found.
    local devices
    devices=$("$INSTALL_DIR/bin/llama-bench" --list-devices 2>&1 || true)
    if grep -q 'Vulkan' <<<"$devices"; then
        echo "  ✓  Vulkan device visible to the binary:"
        grep -A3 'Available devices' <<<"$devices" || true
    else
        echo "  ⚠  No Vulkan device reported — check 'vulkaninfo --summary'"
    fi
    echo ""
    echo "═══════════════════════════════════════════════════════════"
    echo "  Done. Run with:"
    echo "    ./run/start-llama-server.sh            # Vulkan is the default"
    echo "    bash bench/run-all-benchmarks.sh       # Vulkan + CPU rows"
    echo "═══════════════════════════════════════════════════════════"
}

check_prereqs
fetch_source
build
verify
