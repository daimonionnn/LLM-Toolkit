# LLM Inference Toolkit for Linux — Vulkan/ROCm/CUDA management for AMD APU + dGPU (ROCm for AMD APUs or older AMD GPUs)

Personal toolkit for running local LLMs on a dual-GPU system using llama.cpp (Llama Server) and LM Studio.

## Hardware

| Component | Detail |
|-----------|--------|
| CPU/APU | AMD Ryzen 7 5700G (8C/16T, Zen 3) |
| iGPU | Radeon Vega 8 — gfx90c (GCN 5, 8 CUs, UMA) |
| **dGPU** | **NVIDIA GeForce RTX 5090 (32 GB VRAM)** |
| RAM | 64 GB DDR4 (shared with Vega 8 iGPU via UMA) |
| OS | Ubuntu 25.10 "Questing", kernel 6.17 |

## Performance

### RTX 5090 — Llama 2 7B Chat Q4_K_S (`-ngl 99 -c 512`)

| Backend | Device | Prompt (t/s) | Generation (t/s) |
|---------|--------|-------------|-------------------|
| **Vulkan** | **RTX 5090** | **2,117** | **273** |

### Vega 8 iGPU — Qwen3.5-35B-A3B Q4_K_M (`-ngl 99 -c 2048`)

| Backend | Prefill (t/s) | Generation (t/s) | Notes |
|---------|--------------|------------------|-------|
| CPU only (AVX2) | 82–88 | 17–18 | Fastest prefill — SIMD beats GPU dispatch overhead on UMA |
| Vulkan native | 45–50 | 19–20 | Best generation throughput, stable across all context sizes |
| ROCm 6.2.4 (Docker) | 37–51 | 11–14 | Working — requires Docker workaround (host HIP 5.7.1 incompatible) |
| LM Studio (Vulkan) | 49–158* | 18–19 | *Prefill inflated by LM Studio batching — not directly comparable |

> Full benchmark data in [docs/benchmarks.md](docs/benchmarks.md).

## Quick Start

```bash
# Vulkan on RTX 5090 (default, fastest)
./start-llm.sh

# Vega 8 iGPU via Vulkan
./start-llm.sh --vega

# ROCm GPU via Docker (Vega 8, full GPU offload)
./run-docker-rocm.sh /path/to/model.gguf -ngl 99 -c 2048 --no-warmup

# API endpoint: http://127.0.0.1:8080/v1
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}]}'
```

## Scripts

| Script | Purpose |
|--------|---------|
| [`start-llm.sh`](start-llm.sh) | **Main launcher.** Vulkan on RTX 5090 by default. Memory safeguards, `--vega`/`--cpu`/`--rocm` modes. |
| [`run-llamaserver-vulkan.sh`](run-llamaserver-vulkan.sh) | Direct Vulkan llama-server wrapper with full device selection (`-dev Vulkan0`/`Vulkan1`). |
| [`run-docker-rocm.sh`](run-docker-rocm.sh) | **Docker ROCm launcher.** Auto-builds `Dockerfile.rocm64` image on first run, passes GPU devices into container. |
| [`run-llamaserver-rocm.sh`](run-llamaserver-rocm.sh) | Native ROCm wrapper — broken on host (HIP 5.7.1/Clang-21 mismatch); use `run-docker-rocm.sh` instead. |
| [`build-llamacpp-rocm-vega.sh`](build-llamacpp-rocm-vega.sh) | Build llama.cpp with ROCm/HIP targeting gfx900 (used inside Docker, or for host experiments). |
| [`launch-lmstudio-vulkan.sh`](launch-lmstudio-vulkan.sh) | Launch LM Studio with Vulkan env for Vega 8. Has `--diagnose` mode. |
| [`test-server-perf.py`](test-server-perf.py) | Benchmark llama-server (port 8080) — prefill and decode t/s across 3 context sizes. |
| [`test-lmstudio-perf.py`](test-lmstudio-perf.py) | Benchmark LM Studio (port 1234) — streaming time-to-first-token and decode t/s. |

## ROCm on Vega 8

**Host ROCm (HIP 5.7.1) is broken** — Ubuntu 25.10 ships HIP 5.7.1 paired with Clang-21, a ~2 major version mismatch. Individual GPU kernel tests pass, but inference segfaults at slot initialization regardless of model, flags, or layer count.

**Docker ROCm 6.2.4 works.** Running llama.cpp in a `rocm/dev-ubuntu-24.04:6.2.4` container provides a coherent stack. Full 41/41 layer GPU offload confirmed on Qwen3.5-35B-A3B-Q4_K_M (20 GB model into 64 GB GTT).

```bash
# Start (auto-builds image on first run, ~10 min)
./run-docker-rocm.sh /path/to/model.gguf -ngl 99 -c 2048 --no-warmup
# Server: http://127.0.0.1:8080

# Stop
docker stop $(docker ps -q --filter ancestor=llama-server-rocm-vega)
```

Key env vars baked into `Dockerfile.rocm64`:

| Variable | Value | Reason |
|----------|-------|--------|
| `HSA_XNACK` | `0` | `1` hard-freezes the entire PC on Vega 8 |
| `GGML_HIP_UMA` | `0` | UMA mode requires XNACK page-fault handling (disabled above) |
| `HSA_OVERRIDE_GFX_VERSION` | `9.0.0` | Treat gfx90c as gfx900 |
| `GPU_MAX_ALLOC_PERCENT` | `100` | Allow full GTT allocation |

### Docker & `llama.cpp` Runtime Optimizations

The `run-docker-rocm.sh` script applies several crucial flags to maximize inference speed for the ROCm container on your APU:

#### Docker Flags
* `--ipc=host`: Essential for ROCm containers. Bypasses standard shared memory limits, allowing the GPU/CPU to exchange data structures continuously without bottlenecks.
* `--security-opt seccomp=unconfined`: Disables Docker's default syscall filtering. When passing raw character devices (`/dev/kfd`, `/dev/dri`), seccomp adds overhead; removing it grants native bare-metal performance.
* `--ulimit memlock=-1`: Allows unlimited locked memory pages. ROCm relies on memory pinning to stream data between system RAM and the GPU cores without CPU pagetable management. Docker's default limit severely bottlenecks ROCm or causes crashes.

#### `llama.cpp` Flags
* `-fa 1` (Flash Attention): Significantly reduces memory bandwidth overhead and allows larger context sizes. Essential for APUs where memory bandwidth is the primary bottleneck.
* `-ngl 99`: Offloads all layers to the GPU.
* `-t N` (Recommended to add at runtime): Set to your physical CPU core count (e.g., `-t 4` or `-t 8`). Prevents CPU thrashing and saves thermal/power budget for the Vega iGPU.
* `-nkvo` (`--no-kv-offload`): **Do not use unless necessary!** Forces the Key-Value (KV) cache to stay in standard CPU RAM instead of VRAM. Only use this if your model is so large that adding a context window crashes the GPU with Out-of-Memory (OOM) errors.

See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md#docker-rocm-workaround-working-solution) for full details and the FP8 stub patch needed for gfx900.

## LM Studio (Vulkan)

[`launch-lmstudio-vulkan.sh`](launch-lmstudio-vulkan.sh) launches LM Studio with the correct Vulkan environment for Vega 8.

```bash
./launch-lmstudio-vulkan.sh              # Launch with Vulkan backend
./launch-lmstudio-vulkan.sh --diagnose   # Check GPU/memory, show backend targets
./launch-lmstudio-vulkan.sh --dry-run    # Print config without launching
```

> LM Studio's bundled ROCm backend only targets RDNA2+ (gfx1030+). Always select **Vulkan** in *Settings → My GPUs*.

## Model Capacity

### Vega 8 — 64 GB UMA (GTT)

| Model Size | Quantization | VRAM Usage | Notes |
|------------|-------------|------------|-------|
| 3-4B | Q4_K_M | ~2-3 GB | Full offload |
| 7-8B | Q4_K_M | ~4-5 GB | Full offload |
| 13B | Q4_K_M | ~7-8 GB | Full offload |
| 35B (MoE) | Q4_K_M | ~20 GB | Full offload — tested ✓ |
| 70B | Q4_K_M | ~35-40 GB | Should fit in 64 GB GTT — untested |

> 64 GB GTT requires GRUB params: `amdgpu.gttsize=65536 ttm.pages_limit=16777216` — see [TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

### RTX 5090 — 32 GB VRAM

| Model Size | Quantization | VRAM Usage | Notes |
|------------|-------------|------------|-------|
| 7-8B | Q4_K_M | ~4-5 GB | Full offload |
| 13B | Q4_K_M | ~7-8 GB | Full offload |
| 34B | Q4_K_M | ~18-20 GB | Full offload |
| 70B | Q4_K_M | ~35 GB | Mostly (may need `-ngl ~60`) |

## Documentation

| Doc | Contents |
|-----|---------|
| [docs/benchmarks.md](docs/benchmarks.md) | Full benchmark results — ROCm Docker, Vulkan native, LM Studio, CPU |
| [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) | Common errors, Docker ROCm workaround, diagnostic commands |
| [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) | GPU architecture, Vulkan vs ROCm analysis, UMA memory model |
| [docs/BUILD.md](docs/BUILD.md) | Build prerequisites, ROCm build from source, HIP patches |
| [docs/HIP57-PATCHES.md](docs/HIP57-PATCHES.md) | Technical details of HIP 5.7 compatibility patches |

## Project Structure

```
LLM-Toolkit/
├── README.md
├── start-llm.sh                   ← Main launcher (Vulkan/RTX 5090 default)
├── run-llamaserver-vulkan.sh      ← Vulkan llama-server wrapper
├── run-docker-rocm.sh             ← Docker ROCm launcher (working solution for Vega 8)
├── run-llamaserver-rocm.sh        ← Native ROCm wrapper (broken on host, kept for reference)
├── Dockerfile.rocm64              ← Builds llama.cpp with ROCm 6.2.4 for gfx900
├── build-llamacpp-rocm-vega.sh    ← ROCm build script (used inside Docker)
├── launch-lmstudio-vulkan.sh      ← LM Studio launcher (Vulkan)
├── test-server-perf.py            ← llama-server benchmark (port 8080)
├── test-lmstudio-perf.py          ← LM Studio benchmark (port 1234, streaming)
├── docs/
│   ├── benchmarks.md              ← Benchmark results (all backends)
│   ├── BUILD.md                   ← Build prerequisites and instructions
│   ├── HIP57-PATCHES.md           ← HIP 5.7 compatibility patches
│   ├── TROUBLESHOOTING.md         ← Common errors and debug tips
│   └── ARCHITECTURE.md            ← GPU architecture, Vulkan vs ROCm analysis
├── llama.cpp-vulkan/              ← Vulkan build (production)
├── llama.cpp-rocm-vega/           ← ROCm build (gfx900; CPU-only usable on host)
└── llama.cpp-build/               ← Build workspace (llama.cpp source)
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
- [ ] Benchmark with flash attention enabled
- [ ] Install official AMD ROCm on host to eliminate Docker dependency
