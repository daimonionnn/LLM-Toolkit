# Architecture Notes

Technical background on GPU inference for this system and why Vulkan is the production backend.

## System Hardware

| Component | Details |
|-----------|---------|
| CPU | AMD Ryzen 7 5700G (8C/16T, Zen 3) |
| iGPU | AMD Radeon Vega 8 (gfx90c, 8 CUs, UMA — 16 GB VRAM / ~23 GB GTT from 64 GB system RAM) |
| dGPU | **NVIDIA GeForce RTX 5090** (32 GB dedicated VRAM, PCI 0000:01:00.0) |
| RAM | 64 GB DDR4 (shared with Vega 8 iGPU) |
| OS | Ubuntu 25.10 (Questing), kernel 6.17.0-20-generic |

### Vulkan devices

```
Vulkan0: AMD Radeon Graphics (RADV RENOIR)     — Vega 8 iGPU, ~24 GB shared
Vulkan1: NVIDIA GeForce RTX 5090               — 32 GB dedicated VRAM
Vulkan2: llvmpipe (LLVM 21.1.2, 256 bits)      — CPU software rasterizer
```

## Performance Summary

Llama 2 7B Chat Q4_K_S (3.59 GiB), `-ngl 99 -c 512`:

| Backend | Device | Prompt (t/s) | Generation (t/s) | Status |
|---------|--------|-------------|-------------------|--------|
| **Vulkan** | **RTX 5090 (Vulkan1)** | **2,117** | **273** | **Production** |
| Vulkan | Vega 8 (Vulkan0) | 49 | 14 | Works |
| ROCm (CPU-only) | Ryzen 5700G | 55 | 12 | Works (HIP_VISIBLE_DEVICES=-1) |
| ROCm (GPU, host) | Vega 8 (HIP 5.7.1) | — | — | **CRASHES** (HIP 5.7.1/Clang-21 mismatch) |
| ROCm 6.4.4 (Docker) | Vega 8 (gfx90c) | — | — | **CRASHES** (kernel amdgpu bug) |
| **ROCm 6.2.4 (Docker)** | **Vega 8 (gfx900)** | **37–51** | **11–14** | **✅ Working** |

Qwen3.5-35B-A3B Q4_K_M (`-ngl 99 -c 2048`):

| Backend | Prefill (t/s) | Generation (t/s) | Status |
|---------|--------------|------------------|--------|
| CPU only (AVX2) | 82–88 | 17–18 | ✅ Working |
| Vulkan native | 45–50 | 19–20 | ✅ Working |
| **ROCm 6.2.4 (Docker)** | **37–51** | **11–14** | **✅ Working** |

> Full details in [benchmarks.md](benchmarks.md).

## GPU Architecture Generations

AMD's GPU architectures relevant to ROCm:

| Generation | Codename | GFX ID | Examples | ROCm Status |
|-----------|----------|--------|----------|-------------|
| GCN 5 | Vega | gfx900, gfx906, **gfx90c** | Vega 56/64, **Vega 8 APU** | Legacy — limited support |
| CDNA 1 | Arcturus | gfx908 | MI100 | Supported (datacenter) |
| CDNA 2 | Aldebaran | gfx90a | MI200 series | Supported (datacenter) |
| RDNA 2 | Navi 2x | gfx1030, gfx1031 | RX 6600-6950 XT | Supported |
| RDNA 3 | Navi 3x | gfx1100, gfx1101, gfx1102 | RX 7600-7900 XTX | Supported |
| RDNA 3.5 | — | gfx1151 | Strix APU (Ryzen AI) | Supported |
| RDNA 4 | Navi 4x | gfx1200, gfx1201 | RX 9060-9070 XT | Supported |

## Why gfx90c → gfx900?

The Vega 8 iGPU in the Ryzen 5700G reports as **gfx90c**. This is a cut-down variant of the Vega architecture:

- **gfx900** = Vega 10 (discrete Vega 56/64)
- **gfx90c** = Vega APU variant (Renoir, Cezanne)

The "c" suffix indicates an APU variant with:
- Fewer Compute Units (8 CUs vs 64 on Vega 64)
- Unified Memory Architecture (shares system RAM)
- Slightly different memory controller

The ISA (instruction set architecture) is identical between gfx900 and gfx90c. Code compiled for gfx900 runs on gfx90c. This is why `HSA_OVERRIDE_GFX_VERSION=9.0.0` works — it tells the ROCm runtime "treat this as gfx900" and the kernels execute correctly.

## Why LM Studio's ROCm Backend Doesn't Work

LM Studio ships pre-built ROCm backends. Checking their `backend-manifest.json`:

```json
{
  "gpu": {
    "targets": ["gfx1030", "gfx1100", "gfx1101", "gfx1102", "gfx1151", "gfx1200", "gfx1201"]
  }
}
```

These are all RDNA2+ targets. No GCN 5 (gfx900/gfx90c). When the backend tries to launch a compute kernel, it searches for a code object matching the GPU architecture and finds nothing → `hipErrorInvalidDeviceFunction`.

Even with `HSA_OVERRIDE_GFX_VERSION=9.0.0`, the runtime sees "gfx900" but the binary only has code objects for gfx1030+. The actual ISA is completely different between GCN and RDNA — GCN kernels can't run RDNA instructions and vice versa.

## UMA Memory Model

APUs like the Ryzen 5700G use **Unified Memory Architecture** — the GPU shares system RAM with the CPU. There are two memory pools:

### VRAM (Video RAM / Carve-out)
- Configured in BIOS as "UMA Frame Buffer Size"
- Set to **16 GB** on this system
- This is a portion of system RAM reserved for GPU use
- Appears as "VRAM" in `rocm-smi`
- Fastest access for GPU (direct, no translation needed)

### GTT (Graphics Translation Table)
- Dynamically managed by the kernel
- ~23 GB on this system (depends on available RAM)
- Backed by system RAM with GPU-accessible page table mappings
- Slightly slower than VRAM due to translation overhead
- Appears as "GTT" in `rocm-smi`

### Implications for LLM Inference

- `GGML_HIP_UMA=1` tells llama.cpp this is a UMA system — it can use both VRAM and GTT
- `GPU_MAX_ALLOC_PERCENT=100` prevents the runtime from capping allocation at 75%
- Total usable GPU memory: VRAM + GTT ≈ 16 + 23 = **~39 GB** (theoretical)
- Practical limit depends on system RAM pressure from other processes

## Vulkan vs ROCm on This System

### Why Vulkan wins

ROCm/HIP compute is **fundamentally broken** on kernel 6.17 for the Vega 8 iGPU. This was confirmed with both:
- **Ubuntu ROCm packages** (HIP 5.7.1) — segfaults in `libamdhip64.so`
- **Docker ROCm 6.4.4** (official AMD image) — kernel-level compute ring timeouts

Both paths trigger the same kernel-level failure: `no-retry page fault` → `IB test failed on comp_1.1.0 (-110)` → MODE2 GPU reset → display wedge (since the iGPU drives the monitor). This is an **amdgpu kernel driver bug**, not a userspace issue.

Vulkan bypasses the entire ROCm/HIP/HSA stack and uses:
- **RADV** (Mesa) for the Vega 8 — proven, shipped with distro
- **NVIDIA proprietary driver** (580.126.09) for the RTX 5090

### Backend comparison

| Aspect | Vulkan (RADV/NVIDIA) | ROCm host (HIP 5.7.1) | ROCm 6.4.4 (Docker) | ROCm 6.2.4 (Docker) |
|--------|---------------------|----------------------|----------------------|---------------------|
| Driver | Mesa RADV / NVIDIA proprietary | ROCm/HSA + HIP | ROCm 6.4.4 + HIP | ROCm 6.2.4 + HIP |
| gfx90c support | Native (RADV) | Via gfx900 override | Native | Via gfx900 override |
| RTX 5090 support | **Yes** (Vulkan1) | No (AMD only) | No | No |
| Setup complexity | Zero (works OOTB) | Build from source + patches | Docker (crashes) | `./run-docker-rocm.sh` |
| Stability | **Rock solid** | Segfaults in libamdhip64 | Kernel crashes (MODE2 reset) | **✅ Stable** |
| RTX 5090 perf | **2,117 / 273 t/s** | N/A | N/A | N/A |
| Vega 8 perf (35B) | 45–50 / 19–20 t/s | N/A (crashes) | N/A (crashes) | **37–51 / 11–14 t/s** |
| Crash risk | None | Hard system locks | Hard system locks | None |

### Docker ROCm test results

**ROCm 6.4.4 (`rocm/dev-ubuntu-24.04:6.4.4`) — CRASHES:**
- `rocminfo` inside Docker detected **gfx90c natively**
- llama.cpp built targeting native `gfx90c`
- GPU inference immediately triggered: `no-retry page fault` storm → `IB test failed on comp_1.1.0 (-110)` → MODE2 GPU reset → display wedged
- Two separate test runs both hard-crashed the PC
- This appeared to prove a kernel amdgpu driver bug

**ROCm 6.2.4 (`rocm/dev-ubuntu-24.04:6.2.4`) — WORKS ✅:**
- Targets `gfx900` with `HSA_OVERRIDE_GFX_VERSION=9.0.0` (not native gfx90c)
- Required: `HSA_XNACK=0` (XNACK=1 hard-freezes the PC), `GGML_HIP_UMA=0` (UMA requires XNACK)
- Required: FP8 stub patch in `vendors/hip.h` (gfx900 has no FP8 instructions)
- Full 41/41 layer offload of Qwen3.5-35B-A3B-Q4_K_M (20 GB) into 64 GB GTT
- Output: `ggml_cuda_init: found 1 ROCm devices (Total VRAM: 65536 MiB)`
- Confirmed stable across multiple requests

**Why 6.2.4 works but 6.4.4 crashes:** The 6.4.4 image targeted gfx90c natively, triggering a kernel-level compute ring issue. The 6.2.4 image uses `HSA_OVERRIDE_GFX_VERSION=9.0.0` to present as gfx900 and avoids that code path. The FP8 patch resolves the remaining compile error for gfx900. See `Dockerfile.rocm64` for the full solution.

**Primary recommendation: Vulkan on RTX 5090** for maximum throughput (RTX 5090 is ~20x faster). **Secondary: Docker ROCm 6.2.4** for ROCm-specific testing on Vega 8.

## SDMA and APU Quirks

**SDMA (System DMA)** is the hardware DMA engine for GPU memory transfers. On APU iGPUs, SDMA can cause:
- System hangs during large transfers
- Corrupted data
- Kernel oops/panics

Setting `HSA_ENABLE_SDMA=0` disables the hardware DMA engine and falls back to shader-based copies. This is slower for large transfers but completely reliable.

## xnack (Page Fault Handling)

**xnack** (eXtended NACK) enables GPU page fault handling — the ability for the GPU to handle missing page table entries by requesting pages from the CPU. On UMA APUs, xnack is **always enabled** because the GPU accesses system RAM through the CPU's page tables.

### Code object xnack tagging

GPU code objects (the compiled kernels embedded in `.so` files) are tagged with their xnack compatibility:

| Build Target | Code Object Tag | Runtime Compatibility |
|---|---|---|
| `gfx900` (plain) | **xnack-agnostic** | Works with both `HSA_XNACK=0` and `HSA_XNACK=1` |
| `gfx900:xnack+` | xnack=on only | Requires `HSA_XNACK=1` |
| `gfx900:xnack-` | xnack=off only | Requires `HSA_XNACK=0` |

### rocBLAS convention: plain target (xnack-agnostic)

AMD's own [rocBLAS CMakeLists.txt](https://github.com/ROCm/rocBLAS/blob/develop/CMakeLists.txt) uses plain `gfx900` (no xnack qualifier). Verified via LLVM IR inspection: plain `gfx900` produces code objects with **no** `+xnack` or `-xnack` feature flags — these are xnack-agnostic and work regardless of the `HSA_XNACK` setting.

This disproves the earlier assumption that plain `gfx900` produces `xnack=off(unsupported)` code objects. The build script now uses:
```bash
AMDGPU_TARGETS="gfx900"    # plain, xnack-agnostic (matches rocBLAS convention)
```

### Earlier xnack-related symptoms (no longer relevant)

With the old `gfx900:xnack+` build:
- `HSA_XNACK=1`: GPU page fault storm (`svm_range_restore_pages`, `amdgpu_irq_handle_ih_soft hogged CPU for >10000us`)
- `HSA_XNACK=0`: "invalid device function" (runtime reports `gfx900:xnack-`, no matching code object)

The plain `gfx900` build eliminates both of these issues. The remaining crash is in the HIP runtime itself (see below).

## Code Object Versions (COv5 vs COv6)

AMD GPU code objects have a version number encoded in the ELF `EI_ABIVERSION` field:

| COv | EI_ABIVERSION | Introduced | Notes |
|-----|---------------|------------|-------|
| COv4 | 2 | ROCm 4.x | Legacy |
| COv5 | 3 | ROCm 5.x | Supported by HIP 5.7 |
| COv6 | 4 | ROCm 6.x | **Not supported by HIP 5.7** |

`clang-21` (Ubuntu 25.10) defaults to COv6. The HIP 5.7.1 runtime's COMGR library can only parse up to COv5. When COv6 code objects are embedded in the shared library, COMGR silently fails to load them and GPU operations crash.

### The fix

Force COv5 at build time:
```cmake
-DCMAKE_HIP_FLAGS="-mcode-object-version=5"
```

## ROCm Runtime Crash Analysis

### The problem

After fixing xnack (plain `gfx900` build) and COv5, any GPU offload (`-ngl 1` or more) **segfaults immediately** during slot initialization. The crash is 100% reproducible:

```
dmesg: llama-server[PID]: segfault at 0 ip ...3320e1... error 4
        in libamdhip64.so.5.7.31921[3320e1,...]
```

The crash is always at the **same offset** (`0x3320e1`) inside `libamdhip64.so.5.7.31921` — a NULL pointer dereference during HIP device setup. This is a **bug in the HIP 5.7 runtime** when paired with the kernel 6.14 amdgpu driver and HSA 6.1.2 runtime.

| Configuration | Result |
|---|---|
| `-ngl 1` with `HSA_XNACK=0`, `-fa off` | Segfault at slot init (libamdhip64.so offset 0x3320e1) |
| `-ngl 1` with `HSA_XNACK=0`, `-fa auto` | Segfault at slot init (same offset) |
| `-ngl 1` with `HSA_XNACK=1` | Page fault storm (GPU 99% busy, no progress) — old build |
| `-ngl 0` (ROCm backend active, no offload) | Segfault on first inference |
| `HIP_VISIBLE_DEVICES=-1` (ROCm disabled) | **Works perfectly** — 55 t/s prompt, 12 t/s generation |

CPU-only mode (hiding the GPU entirely with `HIP_VISIBLE_DEVICES=-1`) works perfectly, confirming the bug is in HIP's device/kernel initialization path.

### Root cause: Ubuntu 25.10 ROCm version mismatch

The Ubuntu 25.10 ROCm packages have severe internal version mismatches:

| Component | Ubuntu Version | Role |
|---|---|---|
| clang / LLVM | **21.1.2** | Compiler (very new) |
| hipcc / comgr / device-libs | **7.0.1** | Compiler support (experimental) |
| HSA runtime (libhsa-runtime64) | **6.1.2** | Low-level GPU runtime |
| HIP runtime (libamdhip64) | **5.7.1** | High-level GPU API (**~2 versions behind**) |

In AMD's official ROCm releases, all these components are version-locked (e.g., all 6.2.x or all 7.0.x). Ubuntu repackaged them independently, creating a Frankenstein stack where:

- The compiler (clang-21) generates code using conventions from ROCm 7.x
- The device libraries (7.0.1) provide intrinsics matching the compiler
- The runtime (5.7.1) expects code from the ROCm 5.x era

Simple kernel dispatches happen to work (the ISA is compatible), but the runtime's internal scheduling, graph management, and memory tracking code paths diverge enough to crash during sustained inference workloads.

### Tested models

Both models crash identically, confirming it's not model-specific:
- **Gemma 4 E2B** (Q4_K_M, 3.18 GiB, 4.65B params) — hard crash at `-ngl 35`
- **Llama 2 7B Chat** (Q4_K_S) — hard crash, then segfault at `-ngl 0`

### Resolution

1. ~~**Increase `amdgpu.gttsize`**~~ — Applied (`amdgpu.gttsize=65536 ttm.pages_limit=16777216`), unlocks 64 GB GTT but doesn't fix the HIP crash.
2. ~~**ROCm 6.4.4 Docker**~~ — Crashed at kernel level (MODE2 GPU reset). Tests confirmed gfx90c native targeting triggers an amdgpu driver bug.
3. **Vulkan (RADV/NVIDIA) — ✅ Production** — Zero setup, rock-solid stability, best throughput on RTX 5090.
4. **ROCm 6.2.4 Docker — ✅ Working** — `run-docker-rocm.sh` / `Dockerfile.rocm64`. Coherent ROCm stack avoids the host HIP 5.7.1 mismatch. Uses gfx900 override + FP8 stub patch. Full GPU offload confirmed. 

### Launcher scripts

| Script | Backend | Usage |
|--------|---------|-------|
| `start-llm.sh` | Vulkan (RTX 5090 default) | `./start-llm.sh` |
| `start-llm.sh --vega` | Vulkan (Vega 8) | iGPU via RADV |
| `start-llm.sh --cpu` | CPU-only | No GPU, ~18 t/s gen |
| `start-llm.sh --rocm` | ROCm host (GPU) | **WARNING: crashes** |
| `run-llamaserver-vulkan.sh` | Vulkan | Direct launcher with full options |
| `run-docker-rocm.sh` | ROCm 6.2.4 (Docker) | **Working ROCm GPU offload** |
| `run-llamaserver-rocm.sh` | ROCm host | Legacy, CPU-only usable |

## ROCm Software Stack on Ubuntu 25.10

Ubuntu doesn't ship full ROCm packages. The available components:

```
hipcc 7.0.1+dfsg     → HIP compiler (wraps clang-21)
libamdhip64-dev       → HIP runtime library
libhipblas-dev        → hipBLAS (wraps rocBLAS)
librocblas-dev        → rocBLAS (BLAS for ROCm)
librocsolver-dev      → rocSOLVER (LAPACK for ROCm)
rocm-device-libs-21   → GPU intrinsics and device library bitcode
rocminfo              → GPU info tool
rocm-smi              → GPU monitoring tool
```

These are the Ubuntu-repackaged versions, not AMD's official ROCm releases. The HIP version (5.7) lags behind AMD's current ROCm (6.x), which is why patches are needed.

## Further Reading

- [llama.cpp HIP/ROCm documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip)
- [ROCm GPU support matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
- [AMD GPU ISA documentation](https://gpuopen.com/documentation/amd-isa-documentation/)
- [Mesa RADV driver](https://docs.mesa3d.org/drivers/radv.html)
