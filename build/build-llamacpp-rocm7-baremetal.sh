#!/bin/bash
#
# Build llama.cpp with ROCm 7.x for AMD Vega 8 iGPU — BAREMETAL (no Docker)
#
# ROCm 7.x officially dropped gfx900-family support. This script reinstates it
# for Vega 8 by backporting gfx900 tensile libraries from
# ROCm 6.3.4, then builds llama.cpp against a locally installed ROCm 7.x.
#
# Technique credit:
#   garymathews/frigate:440056a-rocm-7.2.0
#   https://github.com/garymathews/frigate/releases/tag/440056a-rocm-7.2.0
#
# Prerequisites (baremetal):
#   - ROCm 7.x installed system-wide (see https://rocm.docs.amd.com/en/latest/deploy/linux/index.html)
#     Tested path: /opt/rocm  (symlink created by ROCm installer)
#   - CMake 3.21+
#   - git, wget, dpkg-deb
#   - User in 'video' and 'render' groups (for /dev/kfd + /dev/dri access)
#
# Usage:
#   ./build/build-llamacpp-rocm7-baremetal.sh [--skip-backport]
#
#   --skip-backport  Skip the tensile library backport step
#                    (use if you already ran it once and /opt/rocm still has the files)
#
# After build, prefer the launcher (auto-detects the Vega 8 agent index and
# sets all required HSA env vars):
#   ./run/run-rocm7-baremetal.sh /path/to/model.gguf -ngl 99 -c 8192 -fa 0
#
# IMPORTANT: this path requires ROCm 7.0–7.2 on the host. AMD's newer modular
# packages (amdrocm-core 7.13+/gfx120x) ship a ROCr that rejects
# HSA_OVERRIDE_GFX_VERSION and a rocBLAS without gfx9 kernels — the gfx900
# backport cannot work there. Use build/Dockerfile.rocm7-vega instead.
#

set -euo pipefail

# ─── Config ──────────────────────────────────────────────────────────────────
# TODO: pin LLAMA_CPP_BRANCH to a tested tag/commit instead of tracking master
#       (non-reproducible builds; upstream CMake flag renames have already
#       bitten this repo — GGML_HIP_UMA, AMDGPU_TARGETS → GPU_TARGETS).
LLAMA_CPP_REPO="https://github.com/ggml-org/llama.cpp.git"
LLAMA_CPP_BRANCH="master"
BUILD_DIR="$(realpath -m "$(dirname "$0")/../llm/build")"
INSTALL_DIR="$(realpath -m "$(dirname "$0")/../llm/rocm7-vega")"
AMDGPU_TARGET="gfx900"
JOBS=$(nproc)
SKIP_BACKPORT=false

# ROCm 6.3.4 repo for fetching tensile libs only
ROCM634_REPO="https://repo.radeon.com/rocm/apt/6.3.4"
ROCM634_DISTRO="jammy"   # Ubuntu 22.04 — adjust to 'noble' for 24.04 if needed

for arg in "$@"; do
    case "$arg" in
        --skip-backport) SKIP_BACKPORT=true ;;
    esac
done

echo "═══════════════════════════════════════════════════════════"
echo "  Build llama.cpp  ·  ROCm 7.x  ·  Vega 8"
echo "═══════════════════════════════════════════════════════════"
echo "  Target GPU arch : $AMDGPU_TARGET"
echo "  Build dir       : $BUILD_DIR"
echo "  Install dir     : $INSTALL_DIR"
echo "  Skip backport   : $SKIP_BACKPORT"
echo "  Parallel jobs   : $JOBS"
echo ""
echo "  *** NOTE — gfx900 is not officially supported in ROCm 7; applying backport ***"
echo ""

# ─── Check prerequisites ──────────────────────────────────────────────────────
check_prereqs() {
    local ok=true

    if ! command -v cmake &>/dev/null; then
        echo "✗  cmake not found. Install: sudo apt install cmake"
        ok=false
    fi

    # Detect ROCm installation
    ROCM_PATH=""
    for candidate in /opt/rocm /opt/rocm-7.2.0 /opt/rocm-7.1.0 /opt/rocm-7.0.0; do
        if [ -d "$candidate/bin" ]; then
            ROCM_PATH="$candidate"
            break
        fi
    done

    if [ -z "$ROCM_PATH" ]; then
        echo "✗  ROCm install not found. Checked /opt/rocm, /opt/rocm-7.x.x"
        echo "   Install ROCm 7.x: https://rocm.docs.amd.com/en/latest/deploy/linux/index.html"
        ok=false
    else
        ROCM_VER="$(cat "$ROCM_PATH/.info/version" 2>/dev/null || "$ROCM_PATH/bin/rocminfo" 2>/dev/null | grep -m1 'ROCk module' | awk '{print $4}' || echo 'unknown')"
        echo "✓  ROCm found at $ROCM_PATH  (version: $ROCM_VER)"
        # Warn if not ROCm 7.x
        if [[ "$ROCM_VER" != 7.* ]] && [[ "$ROCM_VER" != "unknown" ]]; then
            echo "⚠  Detected ROCm $ROCM_VER — this script targets ROCm 7.x."
            echo "   If you want ROCm 6.x builds, use build-llamacpp-rocm-vega.sh instead."
            read -rp "   Continue anyway? [y/N] " ans
            [[ "$ans" =~ ^[Yy]$ ]] || exit 1
        fi
    fi

    if ! command -v hipcc &>/dev/null && ! [ -x "${ROCM_PATH:-}/bin/hipcc" ]; then
        echo "✗  hipcc not found"
        ok=false
    fi

    # The gfx900 path needs a ROCr runtime that accepts HSA_OVERRIDE_GFX_VERSION.
    # AMD's modular packages (amdrocm-core 7.13+/gfx120x) reject it — the build
    # would succeed but inference can never run. Fail early instead.
    if [ -n "${ROCM_PATH:-}" ] && [ -x "$ROCM_PATH/bin/rocminfo" ]; then
        if ! HSA_OVERRIDE_GFX_VERSION=9.0.0 "$ROCM_PATH/bin/rocminfo" &>/dev/null; then
            echo "✗  This ROCm runtime rejects HSA_OVERRIDE_GFX_VERSION=9.0.0,"
            echo "   which the gfx900-on-gfx90c path requires. This happens with"
            echo "   AMD's modular ROCm packages (amdrocm-core 7.13+/gfx120x)."
            echo "   Use the Docker image instead (self-contained ROCm 7.2):"
            echo "     docker build -t llama-rocm7-vega -f build/Dockerfile.rocm7-vega build/"
            echo "     ./run/run-docker-rocm7.sh /path/to/model.gguf"
            ok=false
        fi
    fi

    if ! command -v git &>/dev/null; then
        echo "✗  git not found. Install: sudo apt install git"
        ok=false
    fi

    if ! command -v wget &>/dev/null; then
        echo "✗  wget not found. Install: sudo apt install wget"
        ok=false
    fi

    if ! command -v dpkg-deb &>/dev/null; then
        echo "✗  dpkg-deb not found. Install: sudo apt install dpkg-dev"
        ok=false
    fi

    if ! $ok; then
        echo ""
        echo "Prerequisites missing — install them and retry."
        exit 1
    fi

    echo "✓  All build prerequisites found"
    echo ""
}

check_prereqs

# ─── Backport gfx900 tensile libraries from ROCm 6.3.4 ───────────────────────
#
#  ROCm 7.x ships rocBLAS without gfx900 tensile kernels.
#  We grab the prebuilt .co files from the ROCm 6.3.4 rocBLAS package and
#  copy them into the ROCm 7 library directory.  rocBLAS scans that directory
#  at runtime and loads the right code objects for the detected GPU.
#
backport_tensile_libs() {
    echo "─── Step 1/4: Backporting gfx900 tensile libs from ROCm 6.3.4 ───"
    echo ""

    # Find the rocBLAS library directory in the active ROCm install
    # (use -L to follow symlinks; /opt/rocm is often a symlink itself)
    if [ -d "${ROCM_PATH}/lib/rocblas/library" ]; then
        ROCBLAS_LIB_DIR="${ROCM_PATH}/lib/rocblas/library"
    else
        ROCBLAS_LIB_DIR="$(find -L "$ROCM_PATH" -maxdepth 6 -type d -path '*/rocblas/library' 2>/dev/null | head -1 || true)"
    fi
    if [ -z "$ROCBLAS_LIB_DIR" ]; then
        echo "✗  Could not locate rocBLAS library directory under $ROCM_PATH"
        echo "   Install: sudo apt install rocblas"
        exit 1
    fi
    echo "  rocBLAS library path: $ROCBLAS_LIB_DIR"

    # Check if gfx900 kernels already present
    GFX900_COUNT=$(find "$ROCBLAS_LIB_DIR" -name '*gfx900*' 2>/dev/null | wc -l)
    if [ "$GFX900_COUNT" -gt 0 ]; then
        echo "  ✓ gfx900 tensile files already present ($GFX900_COUNT files) — skipping download"
        echo ""
        return
    fi

    WORK_TMPDIR="$(mktemp -d /tmp/rocblas634.XXXXXX)"
    trap "rm -rf $WORK_TMPDIR" EXIT

    echo "  Downloading rocBLAS 6.3.4 package from AMD repo..."
    cd "$WORK_TMPDIR"

    # Fetch package list and find the rocblas .deb URL
    PKGLIST_URL="${ROCM634_REPO}/dists/${ROCM634_DISTRO}/main/binary-amd64/Packages"
    wget -q "$PKGLIST_URL" -O Packages

    ROCBLAS_PKG_PATH="$(grep -A5 '^Package: rocblas$' Packages | grep '^Filename:' | head -1 | awk '{print $2}' || true)"
    if [ -z "$ROCBLAS_PKG_PATH" ]; then
        echo "✗  Could not find rocblas package in ${ROCM634_REPO}"
        echo "   Try '--skip-backport' if you have gfx900 tensile files already."
        exit 1
    fi

    DEB_URL="${ROCM634_REPO}/${ROCBLAS_PKG_PATH}"
    DEB_FILE="$(basename "$ROCBLAS_PKG_PATH")"
    echo "  Downloading: $DEB_FILE"
    wget -q --show-progress "$DEB_URL" -O "$DEB_FILE"

    echo "  Extracting package contents..."
    mkdir extracted
    dpkg-deb -x "$DEB_FILE" extracted

    GFX900_FILES="$(find extracted -name '*gfx900*' 2>/dev/null)"
    if [ -z "$GFX900_FILES" ]; then
        echo "✗  No gfx900 files found inside the rocBLAS 6.3.4 package"
        echo "   This is unexpected — please report this issue."
        exit 1
    fi

    echo "  Copying gfx900 tensile files into $ROCBLAS_LIB_DIR ..."
    echo "  (requires sudo — you may be prompted for your password)"
    echo "$GFX900_FILES" | while read -r f; do
        echo "    $(basename "$f")"
        sudo cp "$f" "$ROCBLAS_LIB_DIR/"
    done

    echo ""
    echo "  ✓ Tensile backport complete"
    echo ""
    return 0
}

if [ "$SKIP_BACKPORT" = false ]; then
    backport_tensile_libs
else
    echo "─── Step 1/4: Skipping tensile backport (--skip-backport) ───"
    echo ""
fi

# ─── Clone / update llama.cpp source ─────────────────────────────────────────
echo "─── Step 2/4: Fetching llama.cpp source ───"
echo ""
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

# ─── Configure and build ──────────────────────────────────────────────────────
echo "─── Step 3/4: Configuring CMake ───"
echo ""
cd "$BUILD_DIR/llama.cpp"
rm -rf build-rocm7

# Ensure ROCm tools are on PATH
export PATH="$ROCM_PATH/bin:$PATH"
export ROCM_PATH

# ROCm 7 HIP >= 6.1, so no version-check patch needed.
# ROCm 7 has native FP8 types, so no FP8 typedef stub patch needed.
# Code object version: ROCm 7 LLVM defaults to COv6 which its runtime supports.

# Note: GGML_HIP_UMA and GGML_FLASH_ATTN are no longer llama.cpp CMake
# options (UMA was removed upstream; FA is a runtime flag, -fa). GPU_TARGETS
# replaces the deprecated AMDGPU_TARGETS spelling.
cmake -B build-rocm7 \
    -DGGML_HIP=ON \
    -DGPU_TARGETS="$AMDGPU_TARGET" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
    -DGGML_HIP_GRAPHS=OFF \
    -DGGML_BACKEND_DL=ON \
    -DGGML_CPU_ALL_VARIANTS=ON \
    -DLLAMA_BUILD_SERVER=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DCMAKE_INSTALL_RPATH="\$ORIGIN;\$ORIGIN/../lib" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON

echo ""
echo "─── Step 4/4: Building ($JOBS parallel jobs) ───"
echo "  (This may take 10–30 minutes depending on your CPU)"
echo ""
cmake --build build-rocm7 -j "$JOBS"

echo ""
echo "  Installing to $INSTALL_DIR ..."
cmake --install build-rocm7 --prefix "$INSTALL_DIR"

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════"
echo "  Build complete!  (ROCm 7 / Vega 8)"
echo "═══════════════════════════════════════════════════════════"
echo ""
echo "  Binaries: $INSTALL_DIR/bin/"
echo ""
echo "  Run via the launcher (auto-detects the Vega 8 index, sets HSA env):"
echo ""
echo "    ./run/run-rocm7-baremetal.sh /path/to/model.gguf -ngl 99 -c 8192 -fa 0"
echo ""
echo "  Or manually:"
echo ""
echo "    export ROCR_VISIBLE_DEVICES=<idx>  # Vega 8 GPU index from rocminfo"
echo "    export HIP_VISIBLE_DEVICES=0       # index within the ROCR mask"
echo "    export HSA_OVERRIDE_GFX_VERSION=9.0.0"
echo "    export HSA_ENABLE_SDMA=0"
echo "    export HSA_XNACK=0                 # =1 freezes the whole PC on Vega 8"
echo "    export GPU_MAX_ALLOC_PERCENT=100"
echo ""
echo "    $INSTALL_DIR/bin/llama-server \\"
echo "      -m /path/to/model.gguf \\"
echo "      -ngl 99 \\"
echo "      --host 0.0.0.0 --port 8080"
echo ""
echo "  Compare performance vs ROCm 6 build:"
echo "    ./bench/bench-rocm.sh  (uses llm/rocm-vega/)"
echo "    # swap INSTALL_DIR to llm/rocm7-vega/ to bench the 7.x build"
echo ""
