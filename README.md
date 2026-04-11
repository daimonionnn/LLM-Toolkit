# LLM Inference Toolkit for Linux — Vulkan/RocM/CUDA management for AMD APU machine with dedicated GPU

Beta version - created for personal use and as playground
Running local LLMs on a dual-GPU system using Llama Server and LM Studio (llama.cpp)

## Hardware

| Component | Detail |
|-----------|--------|
| CPU/APU | AMD Ryzen 7 5700G (8C/16T, Zen 3) |
| iGPU | Radeon Vega 8 — gfx90c (GCN 5, 8 CUs, UMA) |
| **dGPU** | **NVIDIA GeForce RTX 5090 (32 GB VRAM)** |
| RAM | 48 GB DDR4 (shared with Vega 8 iGPU) |
| OS | Ubuntu 25.10 "Questing", kernel 6.17 |

## Performance example

Llama 2 7B Chat Q4_K_S (3.59 GiB), `-ngl 99 -c 512`:

| Backend | Device | Prompt (t/s) | Generation (t/s) |
|---------|--------|-------------|-------------------|
| **Vulkan** | **RTX 5090** | **2,117** | **273** |
| Vulkan | Vega 8 iGPU | 49 | 14 |
| CPU-only | Ryzen 5700G | 55 | 12 |

## Quick Start

```bash
# Default: Vulkan on RTX 5090 
./start-llm.sh

# API endpoint: http://127.0.0.1:8081/v1
# Works with any OpenAI-compatible client
```

### Other modes

```bash
./start-llm.sh              # RTX 5090 (default, fastest)
./start-llm.sh --vega       # Vega 8 iGPU via Vulkan
./start-llm.sh --cpu        # CPU-only (ROCm build, no GPU)
./start-llm.sh --help       # Show all options
```

## Scripts

| Script | Purpose |
|--------|---------|
| [`start-llm.sh`](start-llm.sh) | **Main launcher.** Vulkan on RTX 5090 by default. Memory safeguards, `--vega`/`--cpu`/`--rocm` modes. |
| [`run-llamaserver-vulkan.sh`](run-llamaserver-vulkan.sh) | Direct Vulkan llama-server wrapper with full device selection (`-dev Vulkan0`/`Vulkan1`). |
| [`run-llamaserver-rocm.sh`](run-llamaserver-rocm.sh) | Legacy ROCm/HIP wrapper with Vega 8 env vars. Only useful for CPU-only mode. |
| [`build-llamacpp-rocm-vega.sh`](build-llamacpp-rocm-vega.sh) | Build llama.cpp with ROCm/HIP targeting gfx900. Applies HIP 5.7 compatibility patches. |
| [`launch-lmstudio-vulkan.sh`](launch-lmstudio-vulkan.sh) | Launch LM Studio with Vulkan backend. Has `--diagnose` mode. |

## Quick Start

```bash
# Vulkan on RTX 5090 (default, fastest)
./start-llm.sh

# Direct Vulkan launcher with more control
./run-llamaserver-vulkan.sh ~/models/your-model.gguf -ngl 99 -dev Vulkan1 -c 4096

# API endpoint
curl http://127.0.0.1:8081/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"test","messages":[{"role":"user","content":"Hello!"}]}'
```

> ⚠️ If your desktop freezes during launch, avoid full offload (`-ngl 99`) on Vega 8.
> See [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md) for recovery/tuning guidance.

## LM Studio (Vulkan)

[`launch-lmstudio-vulkan.sh`](launch-lmstudio-vulkan.sh) launches LM Studio with the correct Vulkan environment for AMD Vega 8 (gfx90c). It sets up GPU device selection, Vulkan ICD, memory tuning, and guards against known pitfalls before launching.

### Usage

```bash
./launch-lmstudio-vulkan.sh              # Launch with Vulkan backend (default)
./launch-lmstudio-vulkan.sh --diagnose   # Show GPU/memory info and check backends without launching
./launch-lmstudio-vulkan.sh --dry-run    # Print configuration without launching
./launch-lmstudio-vulkan.sh --help       # Show all options
```

### What `--diagnose` checks

- Which GPU architectures LM Studio's ROCm backends were compiled for (and why gfx90c isn't in them)
- Recent ROCm errors from LM Studio logs
- VRAM and GTT memory availability
- Render group membership and `/dev/kfd` access
- Vulkan ICD presence (`radeon_icd.json`)

### Environment it sets

| Variable | Value | Purpose |
|----------|-------|---------|
| `GGML_VK_DEVICE` | `0` | Select AMD Radeon Graphics (RADV RENOIR) for Vulkan |
| `VK_ICD_FILENAMES` | `radeon_icd.json` | Restrict to AMD Vulkan driver only (hides NVIDIA/llvmpipe so Vega 8 appears in LM Studio UI) |
| `GPU_MAX_ALLOC_PERCENT` | `100` | Allow full UMA memory allocation |
| `HSA_ENABLE_SDMA` | `0` | Disable SDMA (prevents crashes on APU iGPUs) |
| `GGML_HIP_UMA` | `1` | Unified Memory Architecture mode |
| `DRI_PRIME` | `pci-0000_0b_00.0` | Pin to AMD render node (`/dev/dri/renderD129`) |

> **Note:** The ROCm backend in LM Studio only targets RDNA2+ (gfx1030+) and will crash on Vega 8.
> Always select **Vulkan** in *Settings → My GPUs* after launch.

### First-time in-app setup

1. Open **Settings → My GPUs**
2. Select **Vulkan** as the GPU backend (not ROCm)
3. Ensure **AMD Radeon Graphics** is selected
4. Load a GGUF model and set GPU offload layers

### Model capacity (Vega 8, 16 GB UMA)

| Model Size | Quantization | VRAM Usage | GPU Offload |
|------------|-------------|------------|-------------|
| 3-4B | Q4_K_M | ~2-3 GB | Full (`-ngl 99`) |
| 7-8B | Q4_K_M | ~4-5 GB | Full (`-ngl 99`) |
| 13B | Q4_K_M | ~7-8 GB | Partial |

## Model Sizing Guide (RTX 5090 — 32 GB VRAM)

| Model Size | Quantization | VRAM Usage | GPU Offload |
|------------|-------------|------------|-------------|
| 1-3B | Q4_K_M | ~1-2 GB | Full (`-ngl 99`) |
| 7-8B | Q4_K_M | ~4-5 GB | Full (`-ngl 99`) |
| 13B | Q4_K_M | ~7-8 GB | Full (`-ngl 99`) |
| 34B | Q4_K_M | ~18-20 GB | Full (`-ngl 99`) |
| 70B | Q4_K_M | ~35 GB | Mostly (may need -ngl ~60) |

## Documentation

Detailed docs are in the [`docs/`](docs/) folder:

- [**Build Guide**](docs/BUILD.md) — Prerequisites, building from source, what the build does
- [**HIP 5.7 Patches**](docs/HIP57-PATCHES.md) — Technical details of patches needed for Ubuntu's HIP 5.7
- [**Troubleshooting**](docs/TROUBLESHOOTING.md) — Common errors, diagnostics, debug tips
- [**Architecture Notes**](docs/ARCHITECTURE.md) — Why gfx90c needs gfx900, GCN vs RDNA, memory model

## Project Structure

```
LLMToolkit/
├── README.md                      ← You are here
├── start-llm.sh                   ← Main launcher (Vulkan/RTX 5090 default)
├── run-llamaserver-vulkan.sh      ← Vulkan llama-server wrapper
├── run-llamaserver-rocm.sh        ← Legacy ROCm wrapper (CPU-only fallback)
├── build-llamacpp-rocm-vega.sh    ← Build llama.cpp with ROCm for gfx900
├── launch-lmstudio-vulkan.sh      ← LM Studio launcher (Vulkan)
├── Dockerfile.rocm64              ← Docker image for ROCm 6.4.4 testing
├── docs/                          ← Detailed documentation
│   ├── BUILD.md                   ← Build prerequisites and instructions
│   ├── HIP57-PATCHES.md          ← HIP 5.7 compatibility patches
│   ├── TROUBLESHOOTING.md        ← Common errors and debug tips
│   └── ARCHITECTURE.md           ← GPU architecture, Vulkan vs ROCm analysis
├── llama.cpp-vulkan/              ← Vulkan build (production)
│   ├── bin/llama-server
│   └── lib/
├── llama.cpp-rocm-vega/           ← ROCm build (legacy, CPU-only)
│   ├── bin/llama-server
│   └── lib/
├── llama.cpp-rocm64/              ← Docker ROCm 6.4.4 build (crashes)
└── llama.cpp-build/               ← Build workspace (source)
    └── llama.cpp/
```

## ROCm Status (Legacy)

> **ROCm/HIP compute on Vega 8 is broken with kernel 6.17.** Both Ubuntu's ROCm 5.7 packages and Docker ROCm 6.4.4 crash at the kernel driver level (`no-retry page fault` → MODE2 GPU reset). This is an amdgpu driver bug, not fixable from userspace. The ROCm build is preserved for CPU-only fallback. See [Architecture Notes](docs/ARCHITECTURE.md) for the full analysis.

## TODO

- [x] Build llama.cpp with ROCm/HIP for gfx900
- [x] Fix xnack (plain gfx900 = xnack-agnostic)
- [x] Fix COv6 incompatibility (force -mcode-object-version=5)
- [x] Isolate crash to HIP 5.7 runtime + kernel 6.17 driver bug
- [x] Test ROCm 6.4.4 via Docker (also crashes — kernel-level)
- [x] **Build llama.cpp with Vulkan backend**
- [x] **Discover RTX 5090**
- [x] **Test Vulkan on Vega 8 (stable)**
- [x] **Create Vulkan launcher scripts**
- [x] Update documentation with full performance comparison
- [ ] Test larger models
- [ ] Benchmark with flash attention enabled
