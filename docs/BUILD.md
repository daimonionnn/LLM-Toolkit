# Build Guide

Building llama.cpp from source with ROCm/HIP support for the AMD Vega 8 APU (gfx90c).

## Why Build From Source?

LM Studio's bundled ROCm backend only includes kernels for RDNA2+ GPUs (gfx1030 and newer). The Vega 8 iGPU uses the GCN 5 architecture (gfx90c), which isn't supported. Building llama.cpp ourselves lets us target `gfx900` — the closest official ROCm target to gfx90c.

> **Status (June 2026):**
> - **Ubuntu's own HIP packages** — on Ubuntu 25.10 the distro shipped HIP 5.7.1/Clang-21, a ~2 major version mismatch that segfaulted during GPU inference (`run/run-llamaserver-rocm.sh`, kept only for reference). Ubuntu 26.04 ships `libamdhip64-dev` 7.1 instead, which has not been tested here — this project installs AMD's own ROCm 7.2 packages and uses those.
> - **ROCm 6.2.4 Docker** (`./run/run-docker-rocm.sh`) — stable, full GPU offload confirmed.
> - **ROCm 7.2 Docker** (`./run/run-docker-rocm7.sh`) — re-verified 2026-06-13; 35B full offload stable, gfx900 tensile backport applied. **Recommended ROCm path.**
> - **ROCm 7.2 Baremetal** — **working again, re-verified 2026-09-07** on Ubuntu 26.04 / kernel 7.0 with classic ROCm 7.2.0 (built with `GGML_HIP_GRAPHS=OFF`, `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`). It was broken May–September 2026 only because the modular `amdrocm-core` 7.13+/gfx120x packages had replaced classic ROCm for the R9700s; with the dGPUs gone and classic ROCm reinstalled, the path works. Install via `setup/bootstrap-host.sh` (or `setup/install-rocm7-host.sh` alone), build via `build/build-llamacpp-rocm7-baremetal.sh`, run via `run/run-rocm7-baremetal.sh`.
>
> For native GPU inference without ROCm, **use Vulkan** (the `run/start-llama-server.sh` default) — see [ARCHITECTURE.md](ARCHITECTURE.md#vulkan-vs-rocm-on-this-system).
>
> **Device note:** The Vega 8 is currently the only GPU — `/dev/dri/renderD128`, ROCm agent index **0**. Both Docker scripts still auto-detect the render node by PCI ID (`0x1638`) and pass only that device into the container, and `run/run-rocm7-baremetal.sh` still auto-detects the agent index, so nothing needs changing if a dGPU is added back (with the two R9700s installed these were `renderD130` and index 2).

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

On Ubuntu 25.10 the distro's `hipcc` pulled in `clang-21`, `llvm-21`, `libamdhip64-dev`, and `rocm-device-libs-21`. On this host the AMD repo is pinned at priority 600, so `hipcc` resolves to AMD's `1.1.1.70200-43~24.04` instead.

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
chmod +x build/build-llamacpp-rocm-vega.sh
./build/build-llamacpp-rocm-vega.sh
```

### Pinned commit and local patches

All four build paths — the baremetal script, the Vulkan script and both Dockerfiles —
compile the **same llama.cpp commit**, taken from [`build/llama.cpp-ref`](../build/llama.cpp-ref).
They each used to clone `master` independently, which on 2026-09-08 produced a baremetal
install at `465e49b` and a Docker image at `67672dc` and made the two non-comparable.

The scripts read the pin through `build/llama-cpp-ref.sh`; the Dockerfiles take it as
`ARG LLAMA_CPP_REF`, passed by `run/run-docker-rocm*.sh`. It must be a full 40-character
SHA — `git fetch --depth 1 origin <short-sha>` fails with "couldn't find remote ref", and
a script that ignores that silently compiles whatever the checkout already had.

After checkout, `build/apply-patches.sh` applies everything in
[`patches/`](../patches/README.md) in filename order. Currently one patch, which makes the
flash-attention KQ accumulate use `v_mad_mix_f32` on gfx900 and is worth +157 % ROCm
decode at 32K context.
**A patch that no longer applies is a hard error**, not a warning: it means the pin moved
and the patch needs re-validating, and building without it would quietly undo a measured
improvement.

To move the pin: edit `build/llama.cpp-ref`, rebuild every path, re-check that each patch
still applies, and re-run the benchmarks that justify them.

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
6. **Installs** to `llm/rocm-vega/`

### Build Output

```
llm/rocm-vega/
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
./build/build-llamacpp-rocm-vega.sh
```

## ROCm 7.2 Build (Default ROCm Path)

ROCm 7.x dropped official gfx900 support, but llama.cpp can still be built and run by backporting `gfx900` tensile GEMM kernels from ROCm 6.3.4. Confirmed working on Vega 8 as of 2026-05-14 (Qwen3.5-35B-A3B-Q4_K_M, 41/41 layers, sustained inference stable).

**Key fix:** `TensileLibrary_lazy_gfx900.dat` must be present — ROCm 7 looks up this lazy index file first at runtime. Without it, inference crashes with `rocBLAS error: Cannot read TensileLibrary.dat: Illegal seek for GPU arch: gfx900`. The Dockerfile multi-stage build installs rocBLAS into a `rocm/dev-ubuntu-22.04:6.3.4` stage and copies the file across.

### Docker (containerized alternative)

```bash
# Build image (one-time, ~20-40 min — downloads ROCm 6.3.4 rocblas inside)
docker build -t llama-rocm7-vega -f build/Dockerfile.rocm7-vega build/

# Run (auto-detects the Vega 8 render node by PCI ID, ignores the R9700s)
./run/run-docker-rocm7.sh /path/to/model.gguf -ngl 99 -c 2048
```

Key differences from ROCm 6 Docker:
- Based on `rocm/dev-ubuntu-22.04:7.2` (Ubuntu 22.04 + ROCm 7.2)
- No FP8 stub patch needed — ROCm 7 HIP has native FP8 types
- No HIP version-check patch needed — ROCm 7 HIP ≥ 6.1
- Code object version: ROCm 7 LLVM defaults to COv6 which its runtime supports
- Tensile backport: gfx900 `.co` files copied from ROCm 6.3.4 rocBLAS package at build time

### Baremetal (requires classic ROCm 7.0–7.2 on the host)

> ✅ Working on the current host (re-verified 2026-09-07). This path needs **classic**
> ROCm 7.0–7.2. AMD's modular `amdrocm-core` 7.13+/gfx120x packages — installed here
> May–September 2026 for the R9700s — reject `HSA_OVERRIDE_GFX_VERSION` and ship no
> gfx9 rocBLAS kernels; the build and run scripts detect that and abort with a pointer
> to Docker.

#### Baremetal Prerequisites (Ubuntu 25.10 / 26.04)

Neither release is officially supported by AMD, so the installer pins to the
noble (24.04) packages. On 26.04 those still resolve because `libelf1t64` provides
`libelf1` and `libncurses-dev` provides `libtinfo-dev`.

```bash
sudo bash setup/bootstrap-host.sh     # full host prep, calls the installer below
# or just the ROCm part:
sudo bash setup/install-rocm7-host.sh
```

This adds the ROCm 7.2 apt repo pinned to `noble` and installs ~29 packages including
`hip-dev` and `hsa-rocr-dev`.

**libxml2 soname shim — now automatic.** ROCm LLVM's linker (`lld`) was built against
`libxml2.so.2`, but Ubuntu 25.10+ ships only `libxml2.so.16` (soname bumped in libxml2
2.15), and without a shim CMake's HIP compiler test fails. `install-rocm7-host.sh`
creates the symlink itself, inside `/opt/rocm/lib` (and `/opt/rocm/llvm/lib`) rather
than `/lib/x86_64-linux-gnu`, so no system package is affected.

#### Build and Run

```bash
# Build (add --skip-backport on subsequent runs once tensile files are installed)
export PATH=/opt/rocm/bin:$PATH
bash build/build-llamacpp-rocm7-baremetal.sh
bash build/build-llamacpp-rocm7-baremetal.sh --skip-backport  # subsequent runs

# Verify GPU indices (June 2026: R9700s = index 0+1, Vega 8 = index 2)
rocminfo | grep -B2 -A5 'gfx90'

# Run via the wrapper script (auto-detects Vega 8 index)
bash run/run-rocm7-baremetal.sh /path/to/model.gguf -ngl 99 -c 8192

# Or manually:
export ROCR_VISIBLE_DEVICES=2   # Vega 8 GPU index from rocminfo (depends on installed dGPUs)
export HIP_VISIBLE_DEVICES=0    # Relative index within ROCR_VISIBLE_DEVICES mask
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_ENABLE_SDMA=0
export HSA_XNACK=0              # =1 freezes the whole PC on Vega 8
export GPU_MAX_ALLOC_PERCENT=100
export LD_LIBRARY_PATH="$PWD/llm/rocm7-vega/lib:/opt/rocm/lib"
./llm/rocm7-vega/bin/llama-server -m /path/to/model.gguf -ngl 99
```

> **Confirmed:** Binary detects `AMD Radeon Graphics, gfx900:xnack-` with 65536 MiB VRAM (full 64 GB GTT).

---

## Running

### Standalone llama-server (Default ROCm 7.2 Baremetal)

```bash
./run/start-llama-server.sh
# Server: http://127.0.0.1:8080/v1
```

The default launcher delegates to `run/run-rocm7-baremetal.sh`, selects the Vega 8 HSA agent, and uses `-fa 0` by default for best ROCm prefill on gfx900.

### Standalone llama-server via Docker (ROCm 6.2.4 legacy)

The host ROCm stack (HIP 5.7.1) crashes on GPU inference. Use the Docker launcher instead:

```bash
./run/run-docker-rocm.sh ~/models/your-model.gguf -ngl 99 -c 2048 --no-warmup
# Auto-builds the Docker image on first run (~10 min)
# Server: http://127.0.0.1:8080/v1
```

See [TROUBLESHOOTING.md](TROUBLESHOOTING.md#docker-rocm-624-workaround-working-legacy-solution) for full details.

### Standalone llama-server (Legacy host ROCm, CPU-only)

```bash
./run/run-llamaserver-rocm.sh ~/models/your-model.gguf -ngl 0
```

This wrapper sets all the required environment variables and launches the server. The API is available at `http://127.0.0.1:8080/v1`.

> **Note:** `-ngl 99` will crash on the Ubuntu-packaged HIP 5.7.1 host stack. For native ROCm GPU offload, use ROCm 7.2 baremetal instead.

### Manual Run

```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_ENABLE_SDMA=0
export HSA_XNACK=0        # 1 hard-freezes Vega 8 PC
export GGML_HIP_UMA=0     # UMA=1 segfaults when XNACK=0
export GPU_MAX_ALLOC_PERCENT=100

./llm/rocm-vega/bin/llama-server \
    -m ~/models/your-model.gguf \
    -ngl 0 \
    --host 0.0.0.0 --port 8080
```

> Use `-ngl 0` for CPU-only. Any GPU offload (`-ngl 1+`) will segfault on the host (HIP 5.7.1 mismatch). Use `./run/run-docker-rocm.sh` for GPU offload.

### Interactive CLI Chat

```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_ENABLE_SDMA=0
export HSA_XNACK=0
export GGML_HIP_UMA=0

./llm/rocm-vega/bin/llama-cli \
    -m ~/models/your-model.gguf \
    -ngl 0 \
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

1. Start the server: `./run/start-llama-server.sh`
2. In LM Studio: **Developer → Connect to external endpoint**
3. Enter: `http://127.0.0.1:8080/v1`

## Replacing LM Studio's ROCm Backend (Experimental)

> **Warning:** This may break LM Studio. Back up first.

You will need `patchelf` to fix library paths so the backend can find its dependencies. If you don't have it installed:
```bash
python3 -m pip install --user --break-system-packages patchelf
```

```bash
BACKEND="$HOME/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-amd-rocm-avx2-2.13.0"

# Backup
cp -a "$BACKEND" "$BACKEND.bak"

# Replace libs (including versioned symlinks)
cp -a llm/rocm-vega/lib/libggml*.so* "$BACKEND/"
cp -a llm/rocm-vega/lib/libllama*.so* "$BACKEND/"

# Fix RUNPATH for dependencies so LM Studio's engine can find them
for f in "$BACKEND"/*.so; do
    if [ -f "$f" ] && [ ! -L "$f" ]; then
        ~/.local/bin/patchelf --set-rpath '$ORIGIN' "$f"
    fi
done
```

This is risky due to potential ABI mismatches between our build and LM Studio's engine. The standalone server approach is much safer.

### Troubleshooting: "Exit code: null"
If you load a model in LM Studio and immediately get a silent crash (`Exit code: null`), the ROCm driver sequence is segfaulting because LM Studio did not load with the necessary hardware override variables. 

To fix this, you must **close LM Studio completely** and launch it from a terminal where the environment variables are exported:

```bash
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_ENABLE_SDMA=0
export HSA_XNACK=0
export GGML_HIP_UMA=0

# Now launch LM Studio from this terminal!
# (e.g. run `lmstudio` or `/path/to/LM_Studio.AppImage`)
```

If it continues to crash even with these variables, rollback the backend to the `.bak` folder and rely on the **Docker standalone server** (Option A) to handle the GPU inference safely.
