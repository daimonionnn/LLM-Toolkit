# Architecture Notes

Technical background on GPU inference for this system.

## System Hardware

| Component | Details |
|-----------|---|
| CPU | AMD Ryzen 7 5700G (8C/16T, Zen 3) |
| iGPU | AMD Radeon Vega 8 (gfx90c, 8 CUs, UMA — 2 GB BIOS carve-out / up to 64 GB GTT after GRUB tuning) — `card0` / `/dev/dri/renderD128`, PCI ID `0x1638` |
| dGPU | none — both R9700s moved to another machine (September 2026) |
| RAM | 64 GB DDR4 (shared with Vega 8 iGPU) |
| OS | Ubuntu 26.04.1 LTS (Resolute Raccoon), kernel 7.0 |
| ROCm | 7.2.0, classic packages from repo.radeon.com (noble/24.04) |

(September 2026 configuration. Device numbering here is history-dependent and worth
distrusting: the Vega 8 has been `renderD129`, then `renderD130` with two R9700s
installed, and is `renderD128` now that it is alone — and the 26.04 reinstall moved it
from `card1` to `card0` with no hardware change at all. Every script in this repo
therefore resolves it by PCI ID `0x1638`, never by node number.)

### Vulkan devices

```
Vulkan0: AMD Radeon Graphics (RADV RENOIR)      — Vega 8 iGPU, ~32 GB shared
Vulkan1: AMD Radeon AI PRO R9700 (RADV GFX1201) — 32 GB dedicated VRAM
Vulkan2: AMD Radeon AI PRO R9700 (RADV GFX1201) — 32 GB dedicated VRAM
```

> Device indices may differ depending on PCIe enumeration order. Use `vulkaninfo --summary` to verify.

## Performance Summary

This repository now treats **ROCm 7.2 baremetal on the Vega 8 iGPU** as the default llama-server path. Vulkan remains the best decode/interactive path on Vega 8, and CPU FA ON has the strongest large-context prefill.

Qwen3.5-35B-A3B Q4_K_M (`-ngl 99 -c 8192`, unless noted):

| Backend | Prefill (t/s) | Generation (t/s) | Notes |
|---------|--------------|------------------|-------|
| **CPU FA ON** (`-ngl 0`) | **57–233** | 13–16 | **Best prefill overall** — AVX2 SDPA scales ~4× at large context |
| CPU FA OFF (`-ngl 0`) | 56–226 | 12–14 | Very similar; use `-fa 1` |
| Vulkan native (`-dev Vulkan0`) | 45–50 | **19–20** | **Best generation** — stable across all contexts |
| **ROCm 6.2.4 FA OFF** | **40–64** | **12–14** | `-fa 0` recommended — FA ON hurts 33–83% at large context |
| ROCm 6.2.4 FA ON | 35–49 | 11–13 | Suboptimal on Vega 8 |
| **ROCm 7.2 Baremetal FA OFF** | **39–69** | **11–15** | Default launcher path; near-identical to Docker, no container overhead |
| ROCm 7.2 Baremetal FA ON | 35–52 | 12–15 | FA ON hurts prefill at ≥1K tokens; use `-fa 0` |
| **ROCm 7.2 Docker FA OFF** | **39–70** | **12–15** | Same gfx900 tensile backport, containerized |
| ROCm 7.2 Docker FA ON | 36–53 | 12–15 | Same FA penalty as baremetal |
> Full details in [benchmarks.md](benchmarks.md).

## GPU Architecture Generations

AMD's GPU architectures relevant to ROCm:

| Generation | Codename | GFX ID | Examples | ROCm Status |
|-----------|----------|--------|----------|-------------|
| GCN 5 | Vega | gfx900, gfx906, **gfx90c** | Vega 56/64, **Vega 8 APU** | Legacy — unofficially supported via gfx900 override + tensile backport |
| CDNA 1 | Arcturus | gfx908 | MI100 | Supported (datacenter) |
| CDNA 2 | Aldebaran | gfx90a | MI200 series | Supported (datacenter) |
| RDNA 2 | Navi 2x | gfx1030, gfx1031 | RX 6600-6950 XT | Supported |
| RDNA 3 | Navi 3x | gfx1100, gfx1101, gfx1102 | RX 7600-7900 XTX | Supported |
| RDNA 3.5 | — | gfx1151 | Strix APU (Ryzen AI) | Supported |
| RDNA 4 | Navi 4x | gfx1200, gfx1201 | **Radeon AI PRO R9700 (2× in this system)**, RX 9060-9070 XT | Supported |

## Why gfx90c → gfx900?

The Vega 8 iGPU in the Ryzen 5700G reports as **gfx90c**. This is a cut-down variant of the Vega architecture:

- **gfx900** = Vega 10 (discrete Vega 56/64)
- **gfx90c** = Vega APU variant (Renoir, Cezanne)

The "c" suffix indicates an APU variant with:
- Fewer Compute Units (8 CUs vs 64 on Vega 64)
- Unified Memory Architecture (shares system RAM)
- Slightly different memory controller

The ISA (instruction set architecture) is identical between gfx900 and gfx90c. Code compiled for gfx900 runs on gfx90c. This is why `HSA_OVERRIDE_GFX_VERSION=9.0.0` works — it tells the ROCm runtime "treat this as gfx900" and the kernels execute correctly.

## Performance ceiling and tuning levers (gfx900 / Vega 8)

Two hardware facts bound what any amount of build/flag tuning can achieve on this iGPU:

1. **No hardware `dp4a`.** The byte-wise integer dot-product instruction that llama.cpp's quantized matmul (MMQ) kernels depend on first appears on **Vega 20 / gfx906** (`ggml/src/ggml-cuda/common.cuh` — "VEGA20 … minimum for dp4a"). gfx900/gfx90c lacks it, so MMQ runs through a **software-emulated** dp4a (≈3 instructions per op). Consequently the default prefill path leans on rocBLAS/Tensile (dequantize → FP GEMM) using the gfx900 kernels backported from ROCm 6.3.4 — those kernels are AMD's last gfx900 Tensile tuning and are effectively fixed.
2. **Shared DDR4 bandwidth (~40–50 GB/s).** Decode reads the active weights once per token, so token rate is bandwidth-bound, not compute-bound. For the 35B-A3B MoE (~3B active params at Q4 ≈ 1.5–1.7 GB/token) the theoretical ceiling is ~25–30 t/s; measured ROCm decode is 12–15 t/s and Vulkan/RADV reaches 19–20 t/s on the *same* silicon — so ROCm's decode kernels, not the memory wall, are the limiter, and **Vulkan remains the better decode backend**.

What this means for tuning the ROCm 7 build:

| Lever | Type | Expected effect on Vega 8 |
| --- | --- | --- |
| `-ub` / `-b` ubatch/batch size | runtime | Main prefill knob — 8 CUs may prefer a different ubatch than the default 512 |
| `-ctk q8_0` (K-cache quant) | runtime | Cuts KV read bandwidth; helps decode most at large context. `-ctv` needs flash attention, which loses on Vega (use `-fa 0`), so K-only |
| `rocm-smi --setperflevel high` | runtime | Pins GPU clocks; trades shared CPU/APU power budget — can help or hurt |
| `GGML_CUDA_FORCE_MMQ=ON` | build | Ambiguous: emulated dp4a likely *loses* on prefill, but MMQ never materializes dequantized weights so it may *win* on decode bandwidth — must be measured |
| `GGML_CUDA_F16` | build | **Gone** — no longer a CMake option; FP16 paths are auto-selected by arch (gfx900 has fast packed FP16) |
| `GGML_HIP_ROCWMMA_FATTN`, `GGML_HIP_MMQ_MFMA` | build | **N/A** — require CDNA MFMA units; GCN5 has none |
| HIP graphs | build | Kept OFF for stability; low cost on a single small device |

See [benchmarks.md — ROCm 7 tuning sweep](benchmarks.md#rocm-72-vega-8-tuning-sweep-2026-06) for measured results.

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
- Dynamically managed by the kernel.
- Default is often 8GB or 16GB, but can be raised to **64 GB** with `amdgpu.gttsize=65536 ttm.pages_limit=16777216` (only needed for models > 16GB).
- Backed by system RAM with GPU-accessible page table mappings.
- **Performance consideration:** Expanding this to 64GB induces translation overhead. Benchmarks show a ~15-20% drop in generation speed (t/s) when running the 64GB GTT over falling back to the 16GB limit, likely due to page fault/translation efficiency on the memory controller.
- Appears as "GTT" in `rocm-smi`; llama.cpp reports the Vega 8 as `gfx900:xnack-` with 65536 MiB visible (if tuned) or 16384 MiB visible (by default).

### Implications for LLM Inference

- `GGML_HIP_UMA=1` tells llama.cpp this is a UMA system — it can use both VRAM and GTT.
- `GPU_MAX_ALLOC_PERCENT=100` prevents the runtime from capping allocation at 75%.
- ROCm-visible memory varies based on GRUB tuning: **~16 GB default** vs **~64 GB GTT** when tuned. Use the 64GB tune *only* for huge models.
- Practical limit depends on system RAM pressure from other processes

#### Investigation: `GGML_HIP_UMA=0` (Dedicated VRAM mode)

**Hypothesis:** The Vega 8 APU has a 16 GB BIOS-reserved framebuffer carveout. Setting `GGML_HIP_UMA=0` causes llama.cpp to use `hipMalloc` (the discrete-GPU VRAM path) instead of `hipMallocManaged` (the unified memory path). For small models like Gemma 4 E4B (~3.5 GB GPU buffer), this might improve memory bandwidth and throughput by using the dedicated VRAM chunk.

**Result: No difference via UMA parameter local override, but system-wide GTT limit has a massive impact.**

| Mode | FA | Prefill ~128 | Prefill ~1024 | Decode ~128 | Decode ~1024 |
| ---- | -- | ------------ | ------------- | ----------- | ------------ |
| `GGML_HIP_UMA=1` (64GB GTT override enabled) | OFF | 69.7 | 83.1 | 14.0 | 12.6 |
| `GGML_HIP_UMA=0` (64GB GTT override enabled) | OFF | 68.9 | 84.0 | 13.9 | 12.6 |
| `GGML_HIP_UMA=1` (16GB default GTT size limit)| OFF | 76.8 | 89.6 | 15.7 | 14.3 |

**Root Cause & Hardware Reality:**
Confirmed by inspection of the `llama-server` startup logs and `rocminfo`. Even when forced to use `GGML_HIP_UMA=0`, the ROCm HSA runtime reports `VRAM: 65536 MiB` (combining the memory into a single global pool). 

Why does ROCm do this, and why wouldn't "VRAM" give a speedup?
- **No Physical VRAM:** On a Vega 8 APU, the 16 GB "VRAM" is just a BIOS-reserved chunk of standard system DDR4 RAM. It is not dedicated high-speed GDDR6 like on a discrete GPU.
- **Identical Bandwidth:** Because both the 16 GB BIOS carveout and the 64 GB shared UMA/GTT pool live on the exact same physical memory sticks and route through the exact same CPU memory controller, they share the exact same maximum bandwidth (~45-50 GB/s on dual-channel DDR4).
- **HSA Architecture:** ROCm's Heterogeneous System Architecture (HSA) runtime automatically merges these pools on APUs to maximize memory capacity. The 16 GB carveout is merely a memory map reservation trick; it has no separate or faster bandwidth path.
- **Conclusion:** Forcing allocations into the "16 GB carveout" is virtually impossible via ROCm on an APU because the topology merges them — and it wouldn't improve speed even if you could. It would only artificially break the ability to run large models like Qwen 35B (which require ~20 GB). `GGML_HIP_UMA=1` must remain the default.

## Vulkan vs ROCm on This System

### Current recommendation

Use **ROCm 7.2 baremetal** through `run/start-llama-server.sh` for the default OpenAI-compatible server. It provides the best ROCm path on Vega 8, full 35B offload, and performance matching the ROCm 7 Docker image.

Use **Vulkan on Vega 8** (`run/start-llama-server.sh --vulkan`) when decode speed and simple native setup matter more than ROCm validation. Vulkan still wins generation throughput on Qwen and Gemma and avoids the HIP/HSA stack entirely.

The only broken ROCm path is the **Ubuntu-packaged host HIP 5.7.1 stack**. The working ROCm paths use coherent AMD ROCm releases:
- **ROCm 7.2 baremetal** from AMD packages installed under `/opt/rocm-7.2.0`, plus gfx900 tensile backport.
- **ROCm 7.2 Docker** from `rocm/dev-ubuntu-22.04:7.2`, plus the same gfx900 tensile backport.
- **ROCm 6.2.4 Docker** from `rocm/dev-ubuntu-24.04:6.2.4`, with gfx900 override and FP8 stubs.

The known-bad paths remain useful for historical context:
- **Ubuntu ROCm packages** (HIP 5.7.1 + Clang-21) — segfaults in `libamdhip64.so` during inference.
- **Docker ROCm 6.4.4** targeting native `gfx90c` — kernel-level compute ring timeouts and MODE2 reset.

### Backend comparison

| Aspect | ROCm 7.2 Baremetal | Vulkan (RADV) | ROCm 7.2 Docker | ROCm 6.2.4 Docker | Host HIP 5.7.1 | ROCm 6.4.4 Docker |
|--------|----------------------|---------------|-----------------|-------------------|----------------|-------------------|
| Status | Broken on current host¹ | **Default** | ⚠️ small models only² | Working legacy | Broken | Broken |
| Driver/runtime | classic ROCm 7.2 host install | Mesa RADV | ROCm 7.2 + HIP | ROCm 6.2.4 + HIP | Ubuntu HIP 5.7.1 | ROCm 6.4.4 + HIP |
| gfx90c support | gfx900 override + tensile backport | Native RADV | gfx900 override + tensile backport | gfx900 override + FP8 stub | gfx900 override | Native gfx90c |
| Setup complexity | `setup/` + `build/` once, then `run/start-llama-server.sh` | Native Vulkan build | `./run/run-docker-rocm7.sh` | `./run/run-docker-rocm.sh` | Build + patches | Docker, but crashes |
| Stability | **✅ Stable** | **✅ Stable** | **✅ Stable** | **✅ Stable** | Segfaults | Kernel crashes |
| Vega 8 perf (35B) | **39–69 / 11–15 t/s** (FA OFF) | **45–50 / 19–20 t/s** | **39–70 / 12–15 t/s** (FA OFF) | **40–64 / 12–14 t/s** (FA OFF) | N/A | N/A |
| Best use | ROCm server on classic ROCm 7.2 hosts | Best decode/interactive (default) | Recommended ROCm path | ROCm 6 comparison | Historical only | Historical only |
| Crash risk | None observed | None observed | None observed | None observed | Segfaults / hangs | MODE2 reset |
| Multi-GPU isolation | HSA agent auto-detect (`ROCR_VISIBLE_DEVICES=2` here) | `-dev Vulkan0` (auto-detected) | PCI ID render-node isolation | PCI ID render-node isolation | N/A | N/A |

¹ Worked until May 2026, when the host's classic ROCm 7.2 was replaced by modular `amdrocm-core` 7.13/7.14 (gfx120x) packages for the R9700s — that ROCr rejects `HSA_OVERRIDE_GFX_VERSION` and ships no gfx9 rocBLAS kernels. Still valid for hosts running classic ROCm 7.0–7.2. See README "ROCm on Vega 8".

² Loads and runs small models (7B verified 2026-06-13), but loading **Qwen3.5-35B-A3B-Q4_K_M hard-froze the entire machine within ~3 s** (2026-06-13, instant lockup, no kernel log, forced power-cycle). The pre-May-2026 "35B full offload stable" result no longer holds after the host changes. Use Vulkan for large models until root-caused. The earlier "✅ Stable / no crash observed" entries in this table refer to the pre-change host and are kept for history only.

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

**Why 6.2.4 works but 6.4.4 crashes:** The 6.4.4 image targeted gfx90c natively, triggering a kernel-level compute ring issue. The 6.2.4 image uses `HSA_OVERRIDE_GFX_VERSION=9.0.0` to present as gfx900 and avoids that code path. The FP8 patch resolves the remaining compile error for gfx900. See `build/Dockerfile.rocm64` for the full solution.

**ROCm 7.2 (`/opt/rocm-7.2.0` baremetal or `rocm/dev-ubuntu-22.04:7.2` Docker) — WORKS ✅:**
- Targets `gfx900` with `HSA_OVERRIDE_GFX_VERSION=9.0.0`
- Required: gfx900 tensile backport from ROCm 6.3.4 rocBLAS — includes `TensileLibrary_lazy_gfx900.dat` (the index file ROCm 7 looks up first; missing = `Illegal seek for GPU arch: gfx900` crash on first GEMM)
- Docker backport is installed via multi-stage build: `rocm/dev-ubuntu-22.04:6.3.4` stage with `apt-get install rocblas`, then `*gfx900*` files copied to ROCm 7 layer
- Baremetal backport is installed by `build/build-llamacpp-rocm7-baremetal.sh` into `/opt/rocm-7.2.0/lib/rocblas/library/`
- `gfx900:xnack-` with Wave Size 64 — correct Vega 8 (GCN5/Wave64) execution
- Confirmed stable 2026-05-14: Qwen3.5-35B-A3B-Q4_K_M and Gemma 4 E4B, full offload, sustained inference, no crash

**Primary repository recommendation (June 2026):** Vulkan via `run/start-llama-server.sh` (default — best decode, unaffected by host ROCm changes). **Use ROCm 7.2 Docker** via `--rocm-docker` for the best GPU prefill. Baremetal ROCm (`--rocm`) only applies to hosts with classic ROCm 7.0–7.2 — the current host's modular `amdrocm-core` 7.13+/gfx120x install broke it (see README "ROCm on Vega 8").

### Multi-GPU isolation (Vega 8 + 2× Radeon AI PRO R9700)

With three AMD GPUs on the system, ROCm enumerates all of them as HSA agents (GPU 0+1 = R9700s, GPU 2 = Vega 8). Without explicit selection, it picks an R9700 (32 GB VRAM) instead of the Vega 8 (64 GB UMA).

`run/start-llama-server.sh` delegates to wrappers that handle this automatically:
1. Baremetal ROCm 7 parses `rocminfo` agent order to find the gfx90x APU and sets `ROCR_VISIBLE_DEVICES` to its index — 0 now that the Vega 8 is the only GPU, 2 when the two R9700s were installed (`HIP_VISIBLE_DEVICES=0` inside that mask). Override: `VEGA8_ROCM_DEVICE=N`.
2. Docker ROCm scans `/sys/class/drm/renderD*/device/device` for PCI ID `0x1638` (Vega 8).
3. Docker passes **only** the Vega 8 render node (`/dev/dri/renderD128`) into the container.
4. Vulkan auto-detects the `RADV RENOIR` device from `llama-server --list-devices`.

If Docker auto-detect fails: `VEGA8_RENDER_NODE=/dev/dri/renderD128 ./run/run-docker-rocm7.sh model.gguf`

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

### Earlier xnack-related symptoms (historical)

With the old `gfx900:xnack+` build:
- `HSA_XNACK=1`: GPU page fault storm (`svm_range_restore_pages`, `amdgpu_irq_handle_ih_soft hogged CPU for >10000us`)
- `HSA_XNACK=0`: "invalid device function" (runtime reports `gfx900:xnack-`, no matching code object)

The plain `gfx900` build eliminates both of these issues for the supported Docker/ROCm 7 paths. The remaining crash described below applies to the Ubuntu-packaged HIP 5.7.1 host runtime, not to ROCm 7.2 baremetal.

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

## Legacy Host HIP 5.7.1 Crash Analysis

### The problem

After fixing xnack (plain `gfx900` build) and COv5, the Ubuntu-packaged host HIP 5.7.1 stack still segfaults during slot initialization. The crash is 100% reproducible:

```
dmesg: llama-server[PID]: segfault at 0 ip ...3320e1... error 4
        in libamdhip64.so.5.7.31921[3320e1,...]
```

The crash is always at the **same offset** (`0x3320e1`) inside `libamdhip64.so.5.7.31921` — a NULL pointer dereference during HIP device setup. This is a **bug in the HIP 5.7 runtime** when paired with the Ubuntu 25.10 kernel 6.17 amdgpu driver and mismatched HSA/compiler packages.

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
3. **ROCm 7.2 baremetal — ✅ Default** — `run/start-llama-server.sh` / `run/run-rocm7-baremetal.sh`. Official AMD ROCm 7.2 packages with gfx900 tensile backport. Full GPU offload confirmed.
4. **ROCm 7.2 Docker — ✅ Working** — `run/run-docker-rocm7.sh` / `build/Dockerfile.rocm7-vega`. Same runtime approach as baremetal, containerized.
5. **Vulkan (RADV) — ✅ Working** — Zero ROCm dependency, best decode speed on Vega 8.
6. **ROCm 6.2.4 Docker — ✅ Working legacy** — `run/run-docker-rocm.sh` / `build/Dockerfile.rocm64`. Coherent ROCm stack avoids the host HIP 5.7.1 mismatch. Uses gfx900 override + FP8 stub patch. Full GPU offload confirmed.

### Launcher scripts

| Script | Backend | Usage |
|--------|---------|-------|
| `run/start-llama-server.sh` | **Vulkan (Vega 8, RADV) — default** | Best decode (~20 t/s gen); no host-ROCm dependency |
| `run/start-llama-server.sh --cpu` | CPU-only | Best prefill at large context |
| `run/start-llama-server.sh --rocm-docker` | ROCm 7.2 Docker | Best GPU prefill (~84 t/s @4K with `-ub 2048`); **needs 64 GB GTT** |
| `run/start-llama-server.sh --rocm` | ROCm 7.2 baremetal | Classic-ROCm 7.0–7.2 hosts only — working here again since September 2026; aborts early on modular ROCm |
| `run/run-llamaserver-vulkan.sh` | Vulkan | Direct launcher with full device options |
| `run/run-docker-rocm.sh` | ROCm 6.2.4 (Docker) | **Working ROCm GPU offload — auto-selects Vega 8** |
| `run/run-docker-rocm7.sh` | ROCm 7.2 (Docker) | **Confirmed working 2026-05-14 — 35B full offload, sustained inference stable** |
| `run/run-rocm7-baremetal.sh` | ROCm 7.2 baremetal | Direct wrapper — sets all HSA env vars, auto-detects Vega 8 |
| `run/run-llamaserver-rocm.sh` | ROCm host HIP 5.7.1 | Legacy reference only; GPU path broken |

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

## Related Projects and References

### mixa3607/ML-gfx906 — ML builds for AMD GFX906 (Radeon VII / MI50 / MI60)

**[https://github.com/mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906)**

A well-maintained project providing Docker images and build scripts for llama.cpp, ComfyUI, vLLM, and PyTorch on **gfx906** GPUs (Radeon VII / MI50 / MI60). gfx906 is the same GCN 5 / Vega generation as gfx90c (Vega 8 APU) and gfx900 (Vega 10), so its build findings transfer directly to this project.

Prebuild Docker images are published to Docker Hub:
- `docker.io/mixa3607/llama.cpp-gfx906:<ver>-rocm-6.3.3`
- `docker.io/mixa3607/llama.cpp-gfx906:<ver>-rocm-7.2.1`
- ROCm patched base images: `docker.io/mixa3607/rocm-gfx906:<ver>-complete` (ROCm 6.3.3 – 7.2.1)

**What has been adopted from this project:**

| Improvement | Applied to | Details |
|---|---|---|
| `GGML_HIP_GRAPHS=OFF` | All build scripts + Dockerfiles | HIP graph execution is broken/unstable on GCN5 Vega. Explicitly disabled following their "llamacpp: disable HIP_GRAPHS" commit. Confirmed build success 2026-05-14 (baremetal ROCm 7 rebuild). |
| `GGML_BACKEND_DL=ON` | ROCm 7 builds (baremetal + Docker) + ROCm 6 Docker | Dynamic backend loading: HIP and CPU backends are shared libs loaded at runtime. More robust than static linking; enables graceful CPU fallback on OOM. Confirmed: 14 CPU variant `.so` files installed in baremetal rebuild 2026-05-14. |
| `GGML_CPU_ALL_VARIANTS=ON` | ROCm 7 builds (baremetal + Docker) + ROCm 6 Docker | Compiles multiple CPU SIMD variants into a single install; best variant selected at runtime. Confirmed: 14 variants built (x64, sse42, sandybridge, ivybridge, piledriver, haswell, skylakex, cannonlake, cascadelake, icelake, cooperlake, zen4, alderlake, sapphirerapids). Ryzen 5700G (Zen 3 / AVX2) selects `haswell` at runtime. |

**What was reviewed but not adopted:**

| Item | Reason |
|---|---|
| `GGML_HIP_RCCL=ON` | Multi-GPU collective comms — Vega 8 is single-GPU only |
| ROCm patched base images (`rocm-gfx906`) | Their patches re-enable gfx906 in ROCm 6.4+ which officially dropped it. gfx90c/gfx900 remains **natively supported** in the ROCm 6.3.x / 7.x versions used here — no re-patching needed |
| ComfyUI / vLLM / PyTorch | gfx906 has 16 GB HBM2. Vega 8 iGPU shares system RAM with limited bandwidth — insufficient for ComfyUI diffusion or vLLM's memory requirements |
| AVX-512 CPU build flags | Host CPU (Ryzen 7 5700G, Zen 3) does not have AVX-512 — would fail to build |
| `numactl` in Docker | Useful on NUMA servers; minor relevance for desktop APU (see TODO below) |

**Remaining TODOs (from this project's analysis):**

- [x] `GGML_BACKEND_DL=ON` + `GGML_CPU_ALL_VARIANTS=ON` for ROCm 6.2.4 Docker (`Dockerfile.rocm64`) — applied
- [x] Replace hardcoded `HIPCXX=/usr/bin/clang++-21` in `build-llamacpp-rocm-vega.sh` with `hipconfig`-based auto-detection — applied; falls back to LLVM scan if `hipconfig` unavailable
- [x] Add `numactl` to Docker images — applied to both `Dockerfile.rocm64` and `Dockerfile.rocm7-vega`
- [ ] Document `numactl --membind=0 llama-server` as a low-latency option for NUMA-sensitive workloads — low priority on desktop APU
- [x] AVX-512 build flags — **N/A**: Ryzen 7 5700G (Zen 3) has no AVX-512; `GGML_CPU_ALL_VARIANTS=ON` auto-selects the best available SIMD (AVX2 on this CPU) at runtime without needing explicit flags

### Future: Vega 56 / Vega 64 support (gfx900 / gfx906 discrete)

> **Note:** This project is developed and tested on a **Vega 8 APU** (gfx90c, shared RAM). Discrete Vega cards are the same ISA generation and would benefit from the same ROCm build approach — but with significantly better hardware characteristics.

| Card | GFX ID | VRAM | HBM2 bandwidth | CUs |
|------|--------|------|----------------|-----|
| Radeon RX Vega 56 | gfx900 | 8 GB HBM2 | ~410 GB/s | 56 |
| Radeon RX Vega 64 | gfx900 | 8 GB HBM2 | ~484 GB/s | 64 |
| Radeon VII | **gfx906** | **16 GB HBM2** | ~1 TB/s | 60 |
| MI50 / MI60 | gfx906 | 16–32 GB HBM2 | ~1 TB/s | 60 |
| **Vega 8 APU (this system)** | gfx90c | UMA (shared DDR4) | ~50 GB/s | 8 |

Discrete Vega 56/64 use the **same gfx900 target** as this project already builds for, so the existing Dockerfiles and build scripts would work with no changes. The HBM2 bandwidth (8-20×) and dedicated VRAM make practical use cases that are out of reach on the APU:

- **PyTorch** — 8 GB HBM2 is enough for fine-tuning small models (7B at INT4), inference, and many computer vision tasks. [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906) already provides working PyTorch images for gfx906.
- **ComfyUI** — Stable Diffusion inference (SD1.5, SDXL with `--lowvram`) is feasible on 8 GB HBM2. FLUX.1-dev needs 16 GB (Radeon VII / MI50). [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906) has ComfyUI Docker images too.
- **vLLM** — Requires PyTorch; feasible for 7B models on 8 GB, larger quantized models on 16 GB.

**For contributors with Vega 56/64 hardware:**

The gfx906 project is the primary reference. Key differences vs this project:
- No `HSA_OVERRIDE_GFX_VERSION` needed on true gfx900/gfx906 hardware (set natively by ROCm)
- No UMA quirks — discrete VRAM, `GGML_HIP_UMA=0`, no GTT tuning needed
- ROCm 6.4+ dropped gfx906 support; either use ROCm ≤ 6.3.x, or use the [gfx906 patched base images](https://github.com/mixa3607/ML-gfx906/tree/master/rocm) for ROCm 6.4+/7.x
- PyTorch/ComfyUI Dockerfiles would need to be adapted from [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906) to use this repo's build conventions

**Pull requests and forks are welcome** — hardware not available for testing in this repo. See [mixa3607/ML-gfx906](https://github.com/mixa3607/ML-gfx906) for reference implementations.

## Further Reading

- [llama.cpp HIP/ROCm documentation](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md#hip)
- [ROCm GPU support matrix](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html)
- [AMD GPU ISA documentation](https://gpuopen.com/documentation/amd-isa-documentation/)
- [Mesa RADV driver](https://docs.mesa3d.org/drivers/radv.html)
