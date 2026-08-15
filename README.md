# amd-vega-rocm-vulkan-llm-toolkit for Linux 

Toolkit for ROCm and Vulkan LLM inference on Vega APUs/GPUs (tested on AMD Ryzen 5700G APU) + tools for multi-GPU LLM management (Vega + AMD/NVIDIA dGPUs) — llama.cpp (`llama-server`) and LM Studio.

## Hardware

| Component   | Detail                                                                    |
| ----------- | ------------------------------------------------------------------------- |
| CPU/APU     | AMD Ryzen 7 5700G (8C/16T, Zen 3) — slightly undervolted: Curve Optimizer all-core offset −10 |
| iGPU        | Radeon Vega 8 — gfx90c (GCN 5, 8 CUs, 512 MB dedicated + UMA shared RAM)  |
| dGPU 1+2    | 2× AMD Radeon AI PRO R9700 (RDNA4 / gfx1201, 32 GB VRAM each) — June 2026 benchmark layout; one sold July 2026, lineup in flux (device indexes shift, but the scripts auto-detect the Vega 8) |
| RAM         | 64 GB DDR4 — 2× 32 GB Kingston Fury 3600 MT/s, overclocked to 4200 MT/s (shared with the Vega 8 iGPU via UMA) |
| Motherboard | ASRock Fatal1ty B450 Gaming-ITX/ac                                        |
| OS          | Ubuntu 25.10 "Questing", kernel 6.17                                      |
| Host ROCm   | AMD modular packages (`amdrocm-core` 7.13/7.14, gfx120x — for the R9700s) |

> **Why the RAM overclock matters:** on an APU the iGPU has no dedicated VRAM — all weights and KV cache live in UMA/GTT system RAM, so decode throughput is directly bound by DDR4 bandwidth (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)). All benchmark numbers in this repo were measured with this exact memory tune (4200 MT/s); stock 3200–3600 MT/s will decode proportionally slower.

> **GPU targeting note:** Scripts in this toolkit explicitly target the **Vega 8 iGPU**, auto-detected by PCI ID `0x1638` (`/dev/dri/renderD130` as of June 2026 — the node number moves when dGPUs change). ROCm agent order: GPU 0+1 = gfx1201 (R9700s), GPU 2 = gfx90c (Vega 8) — baremetal scripts auto-detect this index (override with `VEGA8_ROCM_DEVICE=N`). Docker scripts pass only the Vega render node into the container so `ROCR_VISIBLE_DEVICES=0` applies there. Vulkan scripts auto-detect the `RADV RENOIR` device (currently `Vulkan0`). The R9700s are not used by these scripts unless you explicitly change device selection.

## Performance

### Vega 8 iGPU — Qwen3.5-35B-A3B Q4_K_M (`-ngl 99 -c 8192`, May 2026)

| Backend                        | Prefill (t/s) | Generation (t/s) | Notes                                                                      |
| ------------------------------ | ------------- | ---------------- | -------------------------------------------------------------------------- |
| **CPU FA ON** (`-ngl 0 -fa 1`) | **57–233**    | 13–16            | **Best prefill overall.** AVX2 SDPA scales ~4× at large context with FA ON |
| CPU FA OFF (`-ngl 0 -fa 0`)    | 56–226        | 12–14            | Similar to FA ON; use `-fa 1` for best CPU results                         |
| Vulkan native (FA OFF default) | 45–50         | 19–20            | **Best generation throughput** — stable across all context sizes           |
| ROCm 6.2.4 — **FA OFF**        | 40–64         | 12–14            | `-fa 0` recommended — FA ON hurts prefill ~33–83% on Vega 8                |
| ROCm 6.2.4 — FA ON             | 35–49         | 11–13            | Default in old config; suboptimal, use `-fa 0`                             |
| ROCm 7.2 — **FA OFF**          | 39–84*        | 12–15            | **Best GPU prefill at large context.** `-fa 0` recommended; *84 t/s @4K with `-ub 2048` (June 2026 tuning) |
| ROCm 7.2 — FA ON               | 36–53         | 12–15            | Available for comparison; weaker prefill than FA OFF at ≥ 1K tokens        |
| LM Studio (Vulkan)             | 49–158*       | 18–19            | *Prefill inflated by LM Studio batching — not directly comparable          |

> Full benchmark data in [docs/benchmarks.md](docs/benchmarks.md).

## Quick Start

```bash
# Vulkan / Mesa RADV on Vega 8 (default, best decode)
./run/start-llama-server.sh

# CPU only (best prefill at large context)
./run/start-llama-server.sh --cpu

# ROCm 7.2 via Docker (recommended ROCm path — best GPU prefill)
./run/start-llama-server.sh --rocm-docker
# or directly:
./run/run-docker-rocm7.sh /path/to/model.gguf -ngl 99 -c 8192 --no-warmup

# ROCm 7.2 baremetal — only if the host still has ROCm 7.2 + gfx900 backport
# (broken since the host moved to modular ROCm 7.13+/gfx120x — see below)
./run/start-llama-server.sh --rocm

# API endpoint: http://127.0.0.1:8080/v1
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}]}'
```

## Scripts

| Script                                                                   | Purpose                                                                                                           |
| ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------- |
| [`run/start-llama-server.sh`](run/start-llama-server.sh)                                     | **Main launcher.** Vulkan by default (auto-detects the Vega 8 Vulkan device). `--cpu`/`--rocm-docker`/`--rocm` modes. |
| [`run/run-llamaserver-vulkan.sh`](run/run-llamaserver-vulkan.sh)         | Direct Vulkan llama-server wrapper with full device selection (`-dev VulkanN`).                                   |
| [`run/run-docker-rocm.sh`](run/run-docker-rocm.sh)                       | Docker ROCm 6.2.4 launcher. Auto-builds `build/Dockerfile.rocm64` image on first run.                            |
| [`run/run-llamaserver-rocm.sh`](run/run-llamaserver-rocm.sh)             | Legacy native ROCm wrapper — broken on host (HIP 5.7.1/Clang-21 mismatch); kept for reference.                    |
| [`build/build-llamacpp-rocm-vega.sh`](build/build-llamacpp-rocm-vega.sh) | Build llama.cpp with ROCm/HIP targeting gfx900 (used inside Docker, or for host experiments).                     |
| [`build/Dockerfile.rocm7-vega`](build/Dockerfile.rocm7-vega)             | ROCm 7.2 image with gfx900 tensile backport from ROCm 6.3.4.                                                      |
| [`run/run-docker-rocm7.sh`](run/run-docker-rocm7.sh)                     | Docker launcher for the ROCm 7.2 image. Same device isolation as `run/run-docker-rocm.sh`.                        |
| [`build/build-llamacpp-rocm7-baremetal.sh`](build/build-llamacpp-rocm7-baremetal.sh) | Baremetal ROCm 7 build — downloads tensile backport, no Docker required (needs ROCm 7 on host).        |
| [`run/launch-lmstudio-vulkan.sh`](run/launch-lmstudio-vulkan.sh)         | Launch LM Studio with Vulkan env for Vega 8. Has `--diagnose` mode.                                               |
| [`bench/test-server-perf.py`](bench/test-server-perf.py)                 | Benchmark llama-server (port 8080) — prefill and decode t/s across 3 context sizes.                              |
| [`bench/test-lmstudio-perf.py`](bench/test-lmstudio-perf.py)             | Benchmark LM Studio (port 1234) — streaming time-to-first-token and decode t/s.                                  |
| [`bench/run-all-benchmarks.sh`](bench/run-all-benchmarks.sh)             | **Multi-backend benchmark runner** — iterates all enabled backends×models, collects CSV results, prints summary.  |
| [`setup/install-rocm7-host.sh`](setup/install-rocm7-host.sh)             | Install ROCm 7.2 on Ubuntu 25.10 host (uses noble/24.04 packages, ABI-compatible). Run once before baremetal build. |
| [`run/run-rocm7-baremetal.sh`](run/run-rocm7-baremetal.sh)               | Launch llama-server with ROCm 7.2 baremetal — sets all HSA env vars, auto-detects Vega 8 device index.             |

## ROCm on Vega 8

**Status (June 2026):**

| Path                                | Status | Notes                                                                |
| ----------------------------------- | ------ | -------------------------------------------------------------------- |
| **Docker ROCm 7.2** (`run/run-docker-rocm7.sh`) | ✅ working — **needs 64 GB GTT** | 35B-A3B re-verified 2026-06-13 (20.4 prefill / 15.9 decode t/s). **Requires the GTT GRUB params** (below) — without them the 35B overflows GTT and hard-freezes the PC |
| **Docker ROCm 6.2.4** (`run/run-docker-rocm.sh`) | ⚠️ unverified on current host | Self-contained `rocm/dev-ubuntu-24.04:6.2.4` image; was working before the May 2026 host changes |
| **Baremetal ROCm 7.2** (`run/run-rocm7-baremetal.sh`) | ❌ broken on current host | Host ROCm was replaced by modular `amdrocm-core` 7.13/7.14 (gfx120x, for the R9700s) |
| Baremetal HIP 5.7.1 (Ubuntu repo)   | ❌ broken | HIP 5.7.1 + Clang-21 mismatch — segfaults at slot init               |

**Why baremetal broke:** the gfx900-on-gfx90c technique needs (a) `HSA_OVERRIDE_GFX_VERSION=9.0.0`, which the new modular ROCr **rejects** (crashes with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`), and (b) gfx900 rocBLAS tensile kernels, which the gfx120x-only packages **don't ship** (the old backported files were wiped from `/opt/rocm/lib/rocblas/library/`). Note the new runtime *does* enumerate the Vega natively as gfx90c, and hipcc 7.13 still compiles gfx90c code — but rocBLAS has no gfx9 kernels at all, and llama.cpp's prefill GEMMs require rocBLAS, so a native gfx90c rebuild can't work either (it would need rocBLAS/Tensile built from source for gfx90c). `run/run-rocm7-baremetal.sh` detects all of this and fails early with instructions.

**Docker is the only ROCm path that initialises at all** — both images bundle their own complete ROCm userspace (where the gfx version override still works) and only share the kernel driver with the host. Small models (7B) load and run.

> ⚠️ **Large models on ROCm REQUIRE the 64 GB GTT GRUB params.** On 2026-06-13, loading **Qwen3.5-35B-A3B-Q4_K_M** (20 GB) via ROCm Docker **hard-froze the entire PC within ~3 seconds** — because a fresh Ubuntu reinstall had left GRUB without `amdgpu.gttsize=65536 ttm.pages_limit=16777216`, so the Vega 8 had only ~30 GB GTT and the allocation overflowed it. **With those params restored (64 GB GTT), the 35B loads to ~21 GB and runs fine** (re-verified 2026-06-13: 20.4 prefill / 15.9 decode t/s). Confirm with `cat /proc/cmdline | grep gttsize` and that the Vega 8 reports `65536M of GTT memory ready`. Without the params, do **not** load >~10 GB models on ROCm — use Vulkan (`./run/start-llama-server.sh`, the default) for large models instead. See [Model Capacity](#model-capacity) for the GRUB setup.

**Baremetal ROCm 7.2 worked before the host ROCm swap** (confirmed 2026-05-14) and still applies to hosts with classic ROCm 7.0–7.2 packages: install via `setup/install-rocm7-host.sh`, build via `build/build-llamacpp-rocm7-baremetal.sh`, run via `run/run-rocm7-baremetal.sh`. Two Ubuntu 25.10 workarounds required: use AMD's noble/24.04 packages (ABI-compatible), and create `sudo ln -sf /lib/x86_64-linux-gnu/libxml2.so.16 /lib/x86_64-linux-gnu/libxml2.so.2` for ROCm LLVM. The install script now refuses to run if modular `amdrocm-core` packages are present (they'd conflict over `/opt/rocm` and could break the R9700 setup).

```bash
# Start (auto-builds image on first run, ~10 min)
./run/run-docker-rocm.sh /path/to/model.gguf -ngl 99 -c 2048 --no-warmup
# Server: http://127.0.0.1:8080

# Stop
docker stop $(docker ps -q --filter ancestor=llama-server-rocm-vega)
```

Key env vars baked into `build/Dockerfile.rocm64`:

| Variable                   | Value   | Reason                                                       |
| -------------------------- | ------- | ------------------------------------------------------------ |
| `HSA_XNACK`                | `0`     | `1` hard-freezes the entire PC on Vega 8                     |
| `GGML_HIP_UMA`             | `0`     | UMA mode requires XNACK page-fault handling (disabled above) |
| `HSA_OVERRIDE_GFX_VERSION` | `9.0.0` | Treat gfx90c as gfx900                                       |
| `GPU_MAX_ALLOC_PERCENT`    | `100`   | Allow full GTT allocation                                    |

### Docker & `llama.cpp` Runtime Optimizations

The `run/run-docker-rocm.sh` script applies several crucial flags to maximize inference speed for the ROCm container on your APU:

#### Docker Flags
* `--ipc=host`: Essential for ROCm containers. Bypasses standard shared memory limits, allowing the GPU/CPU to exchange data structures continuously without bottlenecks.
* `--security-opt seccomp=unconfined`: Disables Docker's default syscall filtering. When passing raw character devices (`/dev/kfd`, `/dev/dri`), seccomp adds overhead; removing it grants native bare-metal performance.
* `--ulimit memlock=-1`: Allows unlimited locked memory pages. ROCm relies on memory pinning to stream data between system RAM and the GPU cores without CPU pagetable management. Docker's default limit severely bottlenecks ROCm or causes crashes.

#### `llama.cpp` Flags
* `-fa 0` (Flash Attention OFF): benchmarked faster for ROCm on Vega 8 — FA ON costs 33–83 % prefill (see [docs/benchmarks.md](docs/benchmarks.md)). The launch scripts default to `-fa 0` for ROCm and `-fa 1` for CPU, where FA ON wins.
* `-ngl 99`: Offloads all layers to the GPU.
* `-b 2048 -ub 2048` (full-batch prefill): **~+22 % prefill at 4K context** on the Vega 8 vs the default `-ub 512`, no decode cost — now the default in `run/run-docker-rocm7.sh`. Smaller `-ub` *hurts* (under-fills the 8-CU GEMMs). See [docs/benchmarks.md](docs/benchmarks.md#rocm-72--vega-8-tuning-sweep-2026-06).
* `-ctk q8_0` (optional): quantize the K cache — small decode gain at long context (+3.5 % @4K) and halves K-cache memory. K-only, since `-ctv` needs flash attention (which loses on Vega).
* `-t N` (Recommended to add at runtime): Set to your physical CPU core count (e.g., `-t 4` or `-t 8`). Prevents CPU thrashing and saves thermal/power budget for the Vega iGPU.
* `-nkvo` (`--no-kv-offload`): **Do not use unless necessary!** Forces the Key-Value (KV) cache to stay in standard CPU RAM instead of VRAM. Only use this if your model is so large that adding a context window crashes the GPU with Out-of-Memory (OOM) errors.

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#docker-rocm-624-workaround-working-legacy-solution) for full details and the FP8 stub patch needed for gfx900.

### ROCm 7.2 on Vega 8 — the gfx900 backport technique

**Status: working in Docker** (re-verified 2026-06-13; baremetal variant requires classic ROCm 7.2 on the host — see status table above). ROCm 7.x officially dropped `gfx900` support, but the technique used by
[garymathews/frigate:440056a-rocm-7.2.0](https://github.com/garymathews/frigate/releases/tag/440056a-rocm-7.2.0)
(originally for Frigate NVR / MIGraphX object detection) can be adapted for llama.cpp:

- ROCm 7 LLVM still compiles `gfx900` device code via `hipcc`.
- `rocBLAS` 7.x ships without `gfx900` tensile GEMM kernels — so large-matrix multiply falls back to a slow reference path or fails entirely.
- **Fix:** copy the prebuilt `gfx900` `.co` kernel files **and `TensileLibrary_lazy_gfx900.dat`** from the ROCm **6.3.4** `rocblas` package into ROCm 7's library directory. rocBLAS probes that directory at runtime and picks them up automatically. The lazy `.dat` index file is essential — without it ROCm 7 crashes on the first GEMM with `Illegal seek for GPU arch: gfx900`.

#### Option A — Docker (recommended)

```bash
# Build the image (one-time, ~20–40 min — downloads ROCm 6.3.4 rocblas inside)
docker build -t llama-rocm7-vega -f build/Dockerfile.rocm7-vega build/

# Run (auto-selects Vega 8 render node)
./run/run-docker-rocm7.sh /path/to/model.gguf -ngl 99 -c 2048
```

#### Option B — Baremetal (requires classic ROCm 7.2 on the host)

> ⚠ Broken on the current host since the modular `amdrocm-core` 7.13+/gfx120x
> packages replaced ROCm 7.2 (May 2026). The scripts below now detect this and
> abort with a pointer to Option A. Kept for hosts running classic ROCm 7.0–7.2.

```bash
# One-time host setup (Ubuntu 25.10 — uses noble/24.04 AMD packages)
sudo bash setup/install-rocm7-host.sh
# Ubuntu 25.10 extra: create libxml2 compat symlink for ROCm LLVM
sudo ln -sf /lib/x86_64-linux-gnu/libxml2.so.16 /lib/x86_64-linux-gnu/libxml2.so.2

# Build (downloads gfx900 tensile backport, then compiles llama.cpp)
export PATH=/opt/rocm/bin:$PATH
bash build/build-llamacpp-rocm7-baremetal.sh
# Subsequent runs (tensile already installed):
bash build/build-llamacpp-rocm7-baremetal.sh --skip-backport

# Run (auto-detects Vega 8 device index)
bash run/run-rocm7-baremetal.sh /path/to/model.gguf -ngl 99 -c 8192
```

> **Device index note:** The Vega 8's ROCm GPU index depends on which dGPUs are installed (currently **2**, after the two R9700s). The run script auto-detects it; override with `VEGA8_ROCM_DEVICE=N` if needed. When setting manually, remember `HIP_VISIBLE_DEVICES` indexes into the `ROCR_VISIBLE_DEVICES`-filtered list, so use `ROCR_VISIBLE_DEVICES=<idx> HIP_VISIBLE_DEVICES=0`.

## LM Studio (Vulkan)

[`run/launch-lmstudio-vulkan.sh`](run/launch-lmstudio-vulkan.sh) launches LM Studio with the correct Vulkan environment for Vega 8.

```bash
./run/launch-lmstudio-vulkan.sh              # Launch with Vulkan backend
./run/launch-lmstudio-vulkan.sh --diagnose   # Check GPU/memory, show backend targets
./run/launch-lmstudio-vulkan.sh --dry-run    # Print config without launching
```

> LM Studio's bundled ROCm backend only targets RDNA2+ (gfx1030+). Always select **Vulkan** in *Settings → My GPUs*.

## Model Capacity

### Vega 8 — 64 GB UMA (GTT)

| Model Size | Quantization | VRAM Usage | Notes                              |
| ---------- | ------------ | ---------- | ---------------------------------- |
| 3-4B       | Q4_K_M       | ~2-3 GB    | Full offload                       |
| 7-8B       | Q4_K_M       | ~4-5 GB    | Full offload                       |
| 13B        | Q4_K_M       | ~7-8 GB    | Full offload                       |
| 35B (MoE)  | Q4_K_M       | ~20 GB     | Full offload — tested ✓            |
| 70B        | Q4_K_M       | ~35-40 GB  | Should fit in 64 GB GTT — untested |

> 64 GB GTT requires GRUB params: `amdgpu.gttsize=65536 ttm.pages_limit=16777216` — see [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).


## Documentation

| Doc                                                | Contents                                                            |
| -------------------------------------------------- | ------------------------------------------------------------------- |
| [docs/benchmarks.md](docs/benchmarks.md)           | Full benchmark results — ROCm Docker, Vulkan native, LM Studio, CPU |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common errors, Docker ROCm workaround, diagnostic commands          |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)       | GPU architecture, Vulkan vs ROCm analysis, UMA memory model         |
| [docs/BUILD.md](docs/BUILD.md)                     | Build prerequisites, ROCm build from source, HIP patches            |
| [docs/HIP57-PATCHES.md](docs/HIP57-PATCHES.md)     | Technical details of HIP 5.7 compatibility patches                  |

## Project Structure

```
amd-vega-rocm-vulkan-llm-toolkit/
├── README.md
├── LICENSE                             ← MIT
├── run/
│   ├── start-llama-server.sh          ← Main launcher (ROCm 7.2 baremetal default)
│   ├── run-docker-rocm.sh             ← Docker ROCm 6.2.4 launcher (working, auto-selects Vega 8)
│   ├── run-docker-rocm7.sh            ← Docker ROCm 7.2 launcher
│   ├── run-rocm7-baremetal.sh         ← Baremetal ROCm 7.2 launcher (sets all HSA env vars)
│   ├── run-llamaserver-vulkan.sh      ← Vulkan llama-server wrapper
│   ├── run-llamaserver-rocm.sh        ← Native ROCm wrapper (broken on host HIP 5.7.1, kept for reference)
│   └── launch-lmstudio-vulkan.sh      ← LM Studio launcher (Vulkan)
│
├── setup/                             ← Host setup scripts
│   └── install-rocm7-host.sh          ← Install ROCm 7.2 on Ubuntu 25.10 (noble packages)
│
├── build/                             ← Dockerfiles & build scripts
│   ├── Dockerfile.rocm64              ← ROCm 6.2.4 image (working)
│   ├── Dockerfile.rocm7-vega          ← ROCm 7.2 image + gfx900 tensile backport
│   ├── build-llamacpp-rocm-vega.sh    ← ROCm 6 build script (runs inside Docker)
│   └── build-llamacpp-rocm7-baremetal.sh ← ROCm 7 baremetal build + tensile backport (working)
│
├── bench/                             ← Benchmarks & performance tests
│   ├── bench-rocm.sh                  ← llama-bench (ROCm build)
│   ├── bench-vulkan.sh                ← llama-bench (Vulkan build)
│   ├── run-all-benchmarks.sh          ← Multi-backend runner (ROCm Docker, Vulkan, CPU; multi-model)
│   ├── test-server-perf.py            ← llama-server benchmark (port 8080)
│   └── test-lmstudio-perf.py          ← LM Studio benchmark (port 1234, streaming)
│
├── docs/
│   ├── benchmarks.md                  ← Benchmark results (all backends)
│   ├── BUILD.md                       ← Build prerequisites and instructions
│   ├── HIP57-PATCHES.md               ← HIP 5.7 compatibility patches
│   ├── TROUBLESHOOTING.md             ← Common errors and debug tips
│   └── ARCHITECTURE.md               ← GPU architecture, Vulkan vs ROCm analysis
│
├── llm/                               ← llama.cpp build outputs
│   ├── vulkan/                        ← Vulkan build (production)
│   ├── rocm-vega/                     ← ROCm 6 build
│   ├── rocm7-vega/                    ← ROCm 7 build (default baremetal path, created by build script)
│   ├── rocm64/                        ← ROCm 6.4 build
│   └── build/                         ← llama.cpp source workspace
```

## TODO

- [x] Build llama.cpp with ROCm/HIP for gfx900
- [x] Fix xnack (plain gfx900 = xnack-agnostic)
- [x] Fix COv6 incompatibility (force `-mcode-object-version=5`)
- [x] Isolate host crash to HIP 5.7.1 / Clang-21 version mismatch
- [x] Fix GRUB params for 64 GB GTT (`amdgpu.gttsize=65536 ttm.pages_limit=16777216`)
- [x] **Docker ROCm 6.2.4 — working, full GPU offload confirmed**
- [x] **Build llama.cpp with Vulkan backend**
- [x] **Test Vulkan on Vega 8 (stable)**
- [x] **Create Vulkan launcher scripts**
- [x] Benchmark all backends (ROCm Docker, Vulkan native, CPU, LM Studio)
- [x] Document all findings
- [x] **Benchmark flash attention ON vs OFF for all backends** — FA OFF wins for both ROCm 6 and ROCm 7 on Vega 8; FA ON wins for CPU (AVX2 SDPA); see [benchmarks.md](docs/benchmarks.md)
- [x] **Re-run ROCm 6 + CPU benchmarks with consistent settings** — done 2026-05-14, `-c 8192 --no-warmup`, both FA ON and FA OFF
- [x] **Install official AMD ROCm 7.2 on host** — done 2026-05-14 via `setup/install-rocm7-host.sh`; Ubuntu 25.10 uses noble/24.04 packages (ABI-compatible); two workarounds needed (libxml2.so.2 symlink, hip-dev package)
- [x] **Housekeeping: reorganised into build/ run/ bench/ folders**
- [x] **Test ROCm 7.2 Docker build on Vega 8** (`build/Dockerfile.rocm7-vega` + `run/run-docker-rocm7.sh`) — confirmed working 2026-05-14, 35B full offload
- [x] **Baremetal ROCm 7.2 build working** — confirmed 2026-05-14; binary sees Vega 8 as `gfx900:xnack-` with 65536 MiB; `ROCR_VISIBLE_DEVICES=1` (Vega 8 is GPU index 1 with RX 9700 as index 0)
- [x] **Compare ROCm 6.x vs ROCm 7.x inference speed on Vega 8** — both benefit from `-fa 0`; ROCm 7 FA OFF (70 t/s) edges out ROCm 6 FA OFF (64 t/s) at 1K/4K context; CPU FA ON wins overall (233 t/s at 4K)
- [x] **Adopt improvements from mixa3607/ML-gfx906** (same GCN5/Vega arch, gfx906): disable `GGML_HIP_GRAPHS` everywhere (stability fix), add `GGML_BACKEND_DL=ON` + `GGML_CPU_ALL_VARIANTS=ON` to ROCm 7 builds, apply to ROCm 6 Docker too, add `numactl` to Docker images, `hipconfig`-based HIP compiler auto-detection in build scripts
- [x] Benchmark ROCm 7 builds after `GGML_HIP_GRAPHS=OFF` + `GGML_BACKEND_DL=ON` — rebuild succeeded 2026-05-14; **re-benchmarking needed** to compare before/after performance
- [x] **Hardware change (June 2026):** RTX 5090 removed, second Radeon AI PRO R9700 added; host ROCm replaced by modular `amdrocm-core` 7.13/7.14 (gfx120x) — Vega 8 is now ROCm GPU index 2 / `renderD130`
- [x] **Toolkit fixes (2026-06-13):** repaired broken Vega-8 ROCm index auto-detect (always returned 0 → would select an R9700), fixed `HIP_VISIBLE_DEVICES` misuse, removed dangerous `HSA_XNACK=1` from the benchmark runner, added preflight guards for the modular-ROCm host, switched default launcher backend to Vulkan, removed dead CMake flags (`GGML_HIP_UMA`, `GGML_FLASH_ATTN`); Vulkan + Docker ROCm 7.2 paths re-verified on hardware
- [ ] **ROCm 7.2 / Vega 8 tuning sweep (in progress, June 2026):** baseline 35B → `-ub`/`-b` batch sizes → `-ctk q8_0` K-cache quant → `rocm-smi --setperflevel high` → maybe `-DGGML_CUDA_FORCE_MMQ=ON`. Harness: `bench/tune-rocm7-vega.sh`. Ceiling analysis (no hardware dp4a, DDR4 bandwidth-bound) in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) and [docs/benchmarks.md](docs/benchmarks.md)
- [ ] Document `numactl --membind=0 llama-server` usage for NUMA-sensitive workloads
- [ ] Extract the copy-pasted Vega 8 detection (PCI-ID render node + rocminfo agent index) into one shared, sourced helper — currently duplicated across `run/` and `bench/` scripts
- [ ] Make the server port configurable end-to-end — the Docker launchers hardcode the `-p 8080:8080` mapping, so `PORT=` in `run/start-llama-server.sh` only works for the Vulkan/CPU/baremetal modes
- [ ] Pin llama.cpp to a tested commit in the Dockerfiles and build scripts (they track `master`, so builds are not reproducible; upstream flag renames have already broken builds once)
- [ ] **Speculative decoding on Vega 8** — decode is DDR4-bandwidth-bound (~25–30 t/s ceiling for the 35B-A3B at 4200 MT/s); draft-token batching is the only lever past that ceiling since drafted tokens are verified in one batched pass over the weights. Test `llama-server -md <draft.gguf> --draft-max 16 --draft-min 1` with a small same-family draft (e.g. Qwen3.5-0.5B/1.7B Q4). Expected +30–80 % decode if acceptance rate is good; works today on the iGPU alone, and if a dGPU accelerator is installed later, pin the draft to it with `--device-draft`
- [ ] **MoE hybrid CPU+iGPU experiment** — CPU prefill (233 t/s) beats GPU prefill (84 t/s); try `--override-tensor "ffn_.*_exps.*=CPU"` to run the expert FFNs on the CPU while attention/shared weights stay on the iGPU. On UMA there is no transfer penalty — only the compute engine changes — so this is a cheap test with real upside for prefill
- [ ] **Quant-format decode sweep** — decode is bandwidth-bound, so smaller quants can win despite costlier dequant: benchmark Q4_K_M vs IQ4_XS vs Q4_0 of the same model on Vulkan and ROCm
- [ ] **RAM timing tune + FCLK check** — decode scales ~linearly with DDR4 bandwidth: tighten secondary/tertiary timings at 4200 MT/s and verify FCLK runs 1:1 (2100 MHz — Cezanne usually manages it; 2:1 costs latency). Re-run `bench/run-all-benchmarks.sh` after
- [ ] **Raise PPT / PBO power limit** — stock 65 W PPT throttles sustained iGPU clocks under combined CPU+iGPU load; the all-core −10 undervolt is already applied, a higher PPT is the next lever
- [ ] **Transparent Huge Pages experiment** — `transparent_hugepage=always` reduces TLB pressure on large GTT allocations; cheap A/B benchmark
- [ ] **Benchmark methodology** — the 50-token decode window is noisy: raise to 256+, add 2–3 repeats with stddev, log power via `rocm-smi` for perf/W, and add standard `llama-bench` pp512/tg128 rows for cross-project comparability
- [ ] **Track upstream llama.cpp** — after pinning (above), bump the pin periodically and re-run the benchmark suite as a regression/gain check (the Vulkan backend improves fast); same for host Mesa/RADV updates
- [ ] **CI smoke checks** — GitHub Action running `shellcheck` + `bash -n` over `run/ bench/ build/` and `py_compile` over the python benches
- [ ] **Future accelerator (AMD or NVIDIA dGPU)** — the R9700s are on their way out (one sold July 2026, the second — which had hardware issues — may follow). If a dGPU returns: benchmark it on this same harness and use it as the draft-model device for speculative decoding on the Vega 8 (`--device-draft`)
- [ ] **Restore baremetal ROCm on Vega 8 under modular ROCm:** needs rocBLAS/Tensile built from source for gfx90c (no override, native arch) — large effort, Docker path covers the use case meanwhile
- [ ] **Future / community:** Vega 56/64 (gfx900) and Radeon VII/MI50/MI60 (gfx906) discrete GPU support — PyTorch, ComfyUI, vLLM. See [docs/ARCHITECTURE.md — Future: Vega 56/64](docs/ARCHITECTURE.md) and [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906). Forks and PRs welcome.

## License

MIT — see [LICENSE](LICENSE).

This repo contains only scripts, Dockerfiles and documentation. It builds
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) from upstream source and uses AMD's
ROCm packages and Docker images, each under their own licenses — nothing from those projects
is redistributed here.
