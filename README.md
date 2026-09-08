# amd-vega-rocm-vulkan-llm-toolkit for Linux 

Toolkit for ROCm and Vulkan LLM inference on Vega APUs/GPUs (tested on AMD Ryzen 5700G APU) + tools for multi-GPU LLM management (Vega + AMD/NVIDIA dGPUs) — llama.cpp (`llama-server`) and LM Studio.

## Hardware

| Component   | Detail                                                                    |
| ----------- | ------------------------------------------------------------------------- |
| CPU/APU     | AMD Ryzen 7 5700G (8C/16T, Zen 3) — undervolted: Curve Optimizer all-core offset **−15**; IOMMU disabled; CPU throttle limit raised to 99 °C (stock Tjmax is 95 °C) |
| iGPU        | Radeon Vega 8 — gfx90c (GCN 5, 8 CUs, **16 GB BIOS carve-out** + up to 64 GB UMA/GTT) — **the only GPU in the box as of September 2026** |
| dGPU        | none — both R9700s now live in a different machine (September 2026); `lspci` shows the Cezanne iGPU only. Benchmark rows dated May/June 2026 were recorded while they were still installed here |
| RAM         | 64 GB DDR4 — 2× 32 GB Kingston Fury 3600 MT/s, overclocked to 4200 MT/s (shared with the Vega 8 iGPU via UMA) |
| Motherboard | ASRock Fatal1ty B450 Gaming-ITX/ac                                        |
| OS          | Ubuntu 26.04.1 LTS "Resolute Raccoon", kernel 7.0                         |
| Host ROCm   | classic ROCm 7.2.0 from repo.radeon.com (noble/24.04 packages)             |

> **Why the RAM overclock matters:** on an APU the iGPU has no dedicated VRAM — all weights and KV cache live in UMA/GTT system RAM, so decode throughput is directly bound by DDR4 bandwidth (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)). All benchmark numbers in this repo were measured with this exact memory tune (4200 MT/s); stock 3200–3600 MT/s will decode proportionally slower.

> **GPU targeting note:** Scripts in this toolkit explicitly target the **Vega 8 iGPU**, auto-detected by PCI ID `0x1638`. With the dGPUs gone it is the only GPU: `/dev/dri/renderD128`, `card0`, ROCm agent index **0** (override with `VEGA8_ROCM_DEVICE=N`). These numbers are not stable — they shift whenever a dGPU is added or removed, and the 26.04 reinstall alone moved the Vega from `card1` to `card0`, which is why every script detects by PCI ID rather than hardcoding a node. Docker scripts pass only the Vega render node into the container so `ROCR_VISIBLE_DEVICES=0` applies there. Vulkan scripts auto-detect the `RADV RENOIR` device (currently `Vulkan0`).

## Performance

### Vega 8 iGPU — September 2026 baseline (`-ngl 99 -c 8192 -ub 4096`)

Prefill / decode t/s at ~128 / ~1024 / ~4096 tokens. Measured 2026-09-08 on Ubuntu
26.04, 16 GB UMA carve-out, after the cooling fix. Each backend at its best flash-
attention setting.

**Qwen3.5-35B-A3B Q4_K_M** (MoE)

| Backend | Prefill | Decode |
| ------- | ------- | ------ |
| **Vulkan `-fa 1`** | **63 / 165 / 190** | **21 / 21 / 21** |
| ROCm 7.2 `-fa 0` | 44 / 122 / 141 | 19 / 18 / 15 |
| CPU (`-dev none`) | 84 / 91 / 86 | 18 / 18 / 15 |

**gemma-4-E4B-it Q4_K_M** (dense)

| Backend | Prefill | Decode |
| ------- | ------- | ------ |
| **ROCm 7.2 `-fa 0`** | 70 / 111 / **192** | 16 / 14 / 10 |
| **Vulkan `-fa 1`** | **127 / 173** / 170 | **18 / 18 / 17** |
| CPU (`-dev none`) | 96 / 94 / 88 | 15 / 14 / 12 |

**Pick the backend by workload, not by reputation:**

| Workload | Use |
| -------- | --- |
| Decode, any model, any context | **Vulkan `-fa 1`** — wins everywhere by 13–41 % |
| Short prompts (~128 tok) | **Vulkan** — wins by 43–81 % |
| Long prompts, MoE model | **Vulkan `-fa 1`** — wins by 35 % |
| Long prompts, dense model | **ROCm `-fa 0`** — wins by 13–17 % |

Vulkan remains the right default (`run/start-llama-server.sh` with no flags), but it is
no longer a clean sweep: on a dense model with long prompts ROCm is measurably faster.

> **`-fa 1` on ROCm halves prefill** (35B 4K: 53 vs 141). Never use it there. On Vulkan
> `-fa 1` is best for both metrics.

> **`-ub 4096` is not universally right.** It is worth +58 % to +85 % on 4K prompts, but
> it *costs* 8–14 % on ~128-token prompts, where the larger buffers do not pay for
> themselves. Rule of thumb: **ubatch ≈ prompt length**. Override with `UBATCH=1024` (or
> `512`) for short-prompt chat; it also costs ~2 GB of GTT.

> Full benchmark data in [docs/benchmarks.md](docs/benchmarks.md).

## Quick Start

### After a fresh Ubuntu install — do this first

A reinstall keeps the `amdgpu` kernel module working but wipes everything this
project needs on top of it: the GRUB GTT params, `render`/`video` membership,
`/opt/rocm`, and the toolchain. This has bitten the rig twice (2026-06-13 and
2026-09-07); the second time the missing GTT params alone would have hard-frozen
the PC on the first large model. One command restores all of it:

```bash
sudo bash setup/bootstrap-host.sh --with-docker   # omit the flag to skip Docker
sudo reboot                                       # required: GRUB params + groups
```

Then verify:

```bash
grep -o 'amdgpu.gttsize=[0-9]*' /proc/cmdline          # expect 65536
awk '{print $1/1024/1024" MiB GTT"}' /sys/class/drm/card*/device/mem_info_gtt_total
/opt/rocm/bin/rocminfo | grep -m1 'Name:.*gfx'          # expect gfx90c
```

### Serving

```bash
# Vulkan / Mesa RADV on Vega 8 (default, best decode)
./run/start-llama-server.sh

# CPU only (best prefill at large context)
./run/start-llama-server.sh --cpu

# ROCm 7.2 via Docker (recommended ROCm path — best GPU prefill)
./run/start-llama-server.sh --rocm-docker
# or directly:
./run/run-docker-rocm7.sh /path/to/model.gguf -ngl 99 -c 8192 --no-warmup

# ROCm 7.2 baremetal (working again as of 2026-09-07 — classic ROCm 7.2 host)
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
| [`setup/bootstrap-host.sh`](setup/bootstrap-host.sh)                     | **Post-reinstall host bootstrap.** GRUB GTT params, `render`/`video` groups, build tools, Vulkan userspace, optional Docker, then ROCm. Idempotent. Run this first on a fresh OS. |
| [`setup/install-rocm7-host.sh`](setup/install-rocm7-host.sh)             | Install ROCm 7.2 on an Ubuntu 25.10/26.04 host (uses noble/24.04 packages, ABI-compatible). Called by the bootstrap; can be run on its own. |
| [`bench/log-thermals.sh`](bench/log-thermals.sh)                         | Log CPU/iGPU temps, SCLK, package power and VRAM/GTT to CSV while a benchmark runs, then flag whether the run was thermally valid. Wraps any command: `bench/log-thermals.sh -- <cmd>`. |
| [`run/run-rocm7-baremetal.sh`](run/run-rocm7-baremetal.sh)               | Launch llama-server with ROCm 7.2 baremetal — sets all HSA env vars, auto-detects Vega 8 device index.             |

## ROCm on Vega 8

**Status (September 2026 — Ubuntu 26.04, kernel 7.0, classic ROCm 7.2.0):**

| Path                                | Status | Notes                                                                |
| ----------------------------------- | ------ | -------------------------------------------------------------------- |
| **Baremetal ROCm 7.2** (`run/run-rocm7-baremetal.sh`) | ✅ working — re-verified 2026-09-07 | Works again now that the modular `amdrocm-core` packages are gone with the dGPUs. Binary reports `gfx900:xnack-`, 65536 MiB. gemma-4-E4B `-fa 0`: **70.0 / 109.1 / 106.1 prefill, 15.9 / 14.5 / 11.9 decode** at the 16 GB carve-out — 29 % ahead of May 2026 at 1K |
| **Docker ROCm 7.2** (`run/run-docker-rocm7.sh`) | ✅ working — re-verified 2026-09-08 | Image rebuilt and benchmarked on both models. Prefill matches baremetal to within noise (35B 4K: 88.2 vs 89.0; gemma 4K: 106.0 vs 106.1). **The 35B loaded without the 2026-06-13 freeze** — that was missing GRUB params, not Docker. **Still requires the GTT GRUB params** (below) |
| **Docker ROCm 6.2.4** (`run/run-docker-rocm.sh`) | ROCm 6 comparison path | Self-contained `rocm/dev-ubuntu-24.04:6.2.4` image; last measured May 2026 (40–64 prefill / 12–14 decode on the 35B). Kept for ROCm-6-vs-7 comparison rather than for use |
| Baremetal HIP 5.7.1 (Ubuntu repo)   | ❌ broken | HIP 5.7.1 + Clang-21 mismatch — segfaults at slot init               |

> **The tables were re-baselined at `-ub 4096` on 2026-09-08** and the harness now sets
> it for GPU backends (`BENCH_UBATCH=512` reproduces the older numbers). The change was
> worth +58 % to +85 % on 4K prompts and flipped the dense-model prefill ranking.

**Why baremetal broke in June 2026, and why it works again:** the gfx900-on-gfx90c technique needs (a) `HSA_OVERRIDE_GFX_VERSION=9.0.0` and (b) gfx900 rocBLAS tensile kernels. AMD's modular packages (`amdrocm-core` 7.13/7.14), installed for the R9700s, **rejected** the override (`HSA_STATUS_ERROR_OUT_OF_RESOURCES`) and shipped no gfx9 kernels at all — and since llama.cpp's prefill GEMMs go through rocBLAS, even a native gfx90c rebuild could not have worked there. With the dGPUs moved out and classic ROCm 7.2.0 installed from repo.radeon.com, both preconditions hold again: the runtime accepts the override and the ROCm 6.3.4 tensile backport applies cleanly. `run/run-rocm7-baremetal.sh` still preflight-checks all of this and fails early with instructions if a modular-ROCm host reappears.

> ⚠️ **Large models on ROCm REQUIRE the GTT GRUB params.** On 2026-06-13, loading **Qwen3.5-35B-A3B-Q4_K_M** (20 GB) via ROCm **hard-froze the entire PC within ~3 seconds** — a fresh Ubuntu install had left GRUB without `amdgpu.gttsize=65536 ttm.pages_limit=16777216`, so the Vega 8 had only ~30 GB of GTT and the allocation overflowed it. With the params present the 35B loads to ~21 GB and runs on both paths — re-verified 2026-09-07 (baremetal) and 2026-09-08 (Docker), no freeze either time. `setup/bootstrap-host.sh` sets them; confirm with `grep -o 'amdgpu.gttsize=[0-9]*' /proc/cmdline`. **This warning stays because the failure mode is a hard lockup, not an error message** — if the params are missing, do not load >~10 GB models on ROCm; use Vulkan instead. See [Model Capacity](#model-capacity).

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

> ✅ Working, re-verified 2026-09-07 on Ubuntu 26.04 / kernel 7.0 / ROCm 7.2.0.
> Requires **classic** ROCm 7.0–7.2; the scripts abort early if AMD's modular
> `amdrocm-core` packages are present, since those reject the gfx version
> override the Vega 8 depends on.

```bash
# One-time host setup (Ubuntu 25.10/26.04 — uses noble/24.04 AMD packages).
# The libxml2 soname shim (.so.2 -> .so.16) is applied automatically, scoped
# to /opt/rocm/lib so no system package is touched.
sudo bash setup/install-rocm7-host.sh

# Build (downloads gfx900 tensile backport, then compiles llama.cpp)
export PATH=/opt/rocm/bin:$PATH
bash build/build-llamacpp-rocm7-baremetal.sh
# Subsequent runs (tensile already installed):
bash build/build-llamacpp-rocm7-baremetal.sh --skip-backport

# Run (auto-detects Vega 8 device index)
bash run/run-rocm7-baremetal.sh /path/to/model.gguf -ngl 99 -c 8192
```

> **Device index note:** The Vega 8's ROCm GPU index depends on which dGPUs are installed — **0** now that it is the only GPU (it was 2 with the two R9700s). The run script auto-detects it; override with `VEGA8_ROCM_DEVICE=N` if needed. When setting manually, remember `HIP_VISIBLE_DEVICES` indexes into the `ROCR_VISIBLE_DEVICES`-filtered list, so use `ROCR_VISIBLE_DEVICES=<idx> HIP_VISIBLE_DEVICES=0`.

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
| [docs/ROCM-PERF-AUDIT.md](docs/ROCM-PERF-AUDIT.md) | Why ROCm trails Vulkan on this silicon, ranked fixes, experiment plan |
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
│   ├── bootstrap-host.sh              ← Post-reinstall bootstrap: GRUB GTT, groups, toolchain, Vulkan, Docker, ROCm
│   └── install-rocm7-host.sh          ← Install ROCm 7.2 on Ubuntu 25.10/26.04 (noble packages)
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
│   ├── log-thermals.sh                ← CPU/iGPU temps, SCLK, package power, VRAM/GTT → CSV; flags throttled runs
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
- [x] **Hardware + OS change (September 2026):** both R9700s moved to another machine — the Vega 8 is the only GPU again (`card0` / `renderD128` / ROCm index 0). OS reinstalled as Ubuntu 26.04.1 / kernel 7.0; classic ROCm 7.2.0 restored from repo.radeon.com, so **baremetal ROCm works again** — re-verified 2026-09-07 against the May 2026 gemma-4-E4B numbers, which both Vulkan and ROCm reproduce (see [benchmarks.md](docs/benchmarks.md))
- [x] **Post-reinstall recovery is now one command** — `setup/bootstrap-host.sh` restores GRUB GTT params, `render`/`video` groups, toolchain, Vulkan userspace, optional Docker and ROCm. Written after the reinstall wiped the host config for the second time (2026-06-13, 2026-09-07); the missing GTT params alone hard-freeze the PC on a large model
- [x] **Script fixes (2026-09-07):** `install-rocm7-host.sh` added *root* rather than the invoking user to `render`/`video` under `sudo` (`$USER` vs `$SUDO_USER`), and `dpkg -l | grep -q` aborted it via SIGPIPE under `pipefail` before anything installed; libxml2 soname shim is now automatic and scoped to `/opt/rocm/lib`. `build-llamacpp-rocm7-baremetal.sh` had `CMAKE_INSTALL_RPATH=$ORIGIN`, but upstream moved `libllama-*-impl.so` to `lib/`, so installed binaries would not start — now `$ORIGIN;$ORIGIN/../lib`
- [x] **Re-verify gemma-4-E4B on 26.04 (2026-09-07)** — Vulkan and ROCm both reproduce the May 2026 numbers; Vulkan `-fa 1` remains the best path. Added `build/build-llamacpp-vulkan.sh` (the harness needed `llm/vulkan/` for both its Vulkan *and* CPU rows, and no script in the repo built it)
- [x] **Harness bugs found while re-verifying (2026-09-07)** — (a) `_detect_vega8_rocm_index` printed an *empty* string when the Vega is GPU 0, because awk's `print gpu` on an unassigned variable emits nothing; that set `ROCR_VISIBLE_DEVICES=""` and silently ran ROCm benchmarks on the CPU. Masked until the dGPUs left. (b) `start_cpu` used `-ngl 0`, which no longer keeps the model off the GPU now that upstream defaults `-ngl` to `auto` — the "CPU" rows were GPU runs. Now `-dev none`
- [x] **CPU prefill gap resolved (2026-09-08)** — the pre-September CPU rows were measured with `-ngl 0`, which no longer forces CPU-only execution (91 % GPU busy, 6.5 GB in GTT), so they are GPU runs mislabelled as CPU. Real CPU-only prefill is 99 / 98 / 93 t/s on gemma, confirmed by `llama-bench -dev none` and the server harness independently (within 3 %). The May figure of 840 t/s is 1.6–3× above the arithmetic ceiling of eight Zen 3 cores and cannot be a real measurement. Harness fixed to `-dev none`; historical rows struck through
- [x] **Cooling fixed enough to stop throttling (2026-09-07)** — raising the fan curve took the peak from 105.4 °C to 89.5 °C, the average from 90.5 °C to 79.7 °C, and samples over Tjmax from 27 to **0**; iGPU SCLK now holds 2400 → 2351 MHz instead of dropping to 2208. Worth ~6–8 % prefill on every backend, so all published numbers were re-measured after the fix
- [x] **BIOS retune (2026-09-07)** — UMA carve-out 2 → 16 GB, Curve Optimizer −10 → −15, IOMMU off, iGPU boost +200 MHz, throttle limit 90 → 99 °C. Net on gemma: ROCm +22 % prefill / +12 % decode, Vulkan +5 % decode, CPU ~unchanged; peak temp fell to 86.6 °C. The iGPU boost changed the DPM table but not the observed 2400 MHz clock, and IOMMU off changed nothing measurable
- [x] **`-fa auto` trap fixed in the ROCm launcher (2026-09-08)** — `run/run-rocm7-baremetal.sh` passed no `-fa`, so it resolved to `auto`, which probes the backend, finds the generic FA tile kernel compiles for gfx900, and enables flash attention. Measured on gemma at a 3330-token prompt: `-fa 0` = 112.8 t/s, `-fa 1` = 48.9, **`-fa auto` = 48.9**. Anyone using the launcher without passing `-fa 0` was silently getting 43 % of achievable prefill. Launcher now defaults to `-fa 0` (still overridable)
- [x] **Qwen3.5-35B re-benchmarked (2026-09-07)** — 20 GB model loads on ROCm without the documented hard freeze. Vulkan `-fa 1` is the best path (21 t/s decode to 4K, 159 t/s prefill at 1K); both GPU backends beat May 2026 by 12–24 %
- [ ] **Explain ROCm's gain from the BIOS retune** — per-phase sampling shows **ROCm never uses the BIOS carve-out** (VRAM ~300 MB, whole model in GTT) on both gemma and the 35B, so the carve-out cannot be the cause. An earlier commit claimed it was, from a whole-run VRAM peak that actually belonged to the Vulkan phase; corrected in [benchmarks.md](docs/benchmarks.md). The real cause is unidentified — four other settings changed at once
- [ ] **Repaste the CPU** — peak is 86.6 °C on a 65 W APU. Not throttling, but the throttle limit is now set to 99 °C, above the 95 °C stock Tjmax, so the usual safety margin is gone
- [ ] **Re-run the remaining backends on 26.04** — Docker ROCm 6.2.4/7.2 images have not been rebuilt or re-verified since the reinstall, and the 35B-A3B model is not downloaded
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
- [ ] **Future accelerator (AMD or NVIDIA dGPU)** — both R9700s now live in another machine (September 2026), so this rig is iGPU-only. If a dGPU returns: benchmark it on this same harness and use it as the draft-model device for speculative decoding on the Vega 8 (`--device-draft`)
- [ ] **Restore baremetal ROCm on Vega 8 under modular ROCm:** needs rocBLAS/Tensile built from source for gfx90c (no override, native arch) — large effort, Docker path covers the use case meanwhile
- [ ] **Future / community:** Vega 56/64 (gfx900) and Radeon VII/MI50/MI60 (gfx906) discrete GPU support — PyTorch, ComfyUI, vLLM. See [docs/ARCHITECTURE.md — Future: Vega 56/64](docs/ARCHITECTURE.md) and [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906). Forks and PRs welcome.

## License

MIT — see [LICENSE](LICENSE).

This repo contains only scripts, Dockerfiles and documentation. It builds
[llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT) from upstream source and uses AMD's
ROCm packages and Docker images, each under their own licenses — nothing from those projects
is redistributed here.
