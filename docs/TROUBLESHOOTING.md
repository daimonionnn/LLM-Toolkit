# Troubleshooting

## Quick Diagnostics

Run the built-in diagnostic mode:

```bash
./launch-lmstudio-vulkan.sh --diagnose
```

This shows:
- Which GPU architectures LM Studio's ROCm backends were compiled for
- Recent ROCm errors from LM Studio logs
- Memory information (VRAM/GTT)
- A recommendation for your hardware

---

## Common Errors

### "ROCm error: invalid device function"

```
ggml_cuda_compute_forward: MUL_MAT failed
ROCm error: invalid device function
llama.cpp abort:98: ROCm error
```

**Cause:** The ROCm binary doesn't contain kernels for your GPU architecture. LM Studio's ROCm backend has kernels for gfx1030+ only. Your Vega 8 is gfx90c.

**Fix:** Use Vulkan (`./launch-lmstudio-vulkan.sh`) or build llama.cpp with gfx900 (`./build-llamacpp-rocm-vega.sh`).

### "hipcc not found"

**Fix:**
```bash
sudo apt install -y hipcc
```

This installs the HIP compiler toolchain (clang-21, llvm-21, libamdhip64-dev, rocm-device-libs-21).

### "Could not find hip-lang-config.cmake"

```
CMake Error: Could NOT find hip (missing: hip_DIR)
```

**Cause:** Ubuntu puts CMake configs in `/usr/lib/x86_64-linux-gnu/cmake/` instead of `/usr/lib/cmake/`.

**Fix:** Create symlinks:
```bash
for dir in hip hip-lang hipblas rocblas rocsolver AMDDeviceLibs amd_comgr; do
    src="/usr/lib/x86_64-linux-gnu/cmake/$dir"
    dst="/usr/lib/cmake/$dir"
    if [ -d "$src" ] && [ ! -e "$dst" ]; then
        sudo ln -sf "$src" "$dst"
    fi
done
```

### "HIP version must be at least 6.1"

**Cause:** Ubuntu 25.10 ships HIP 5.7. llama.cpp wants 6.1+.

**Fix:** The build script patches this automatically. If running manually:
```bash
sed -i 's/VERSION_LESS 6.1/VERSION_LESS 5.5/' ggml/src/ggml-hip/CMakeLists.txt
```

### "multiple definition of `__float2bfloat16`" (linker errors)

Hundreds of lines like:
```
/usr/bin/ld: multiple definition of `__float2bfloat16(float)'; acc.cu first defined here
```

**Cause:** HIP 5.7 headers don't mark bfloat16 helper functions as `inline`.

**Fix:** The build script adds `-z muldefs` to linker flags. If building manually:
```bash
cmake -B build ... -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,muldefs"
```

### "unrecognized option '--allow-multiple-definitions'"

**Cause:** The system uses `mold` as the default linker, which doesn't support the GNU ld `--allow-multiple-definitions` flag.

**Fix:** Use `-z muldefs` instead (portable across ld, gold, mold, lld):
```cmake
-DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,muldefs"
```

### "SCALE failed" / "shared object initialization failed" (xnack mismatch)

```
ggml_cuda_compute_forward: SCALE failed
ROCm error: shared object initialization failed
```

**Cause:** The GPU code objects were compiled without xnack support (`xnack=off`), but the Vega 8 iGPU uses UMA (shared system RAM) which **requires** xnack page-fault handling. The HSA loader refuses to load code objects with `xnack=unsupported` on a GPU that has xnack enabled.

You can verify with:
```bash
readelf -n libggml-hip.so | grep -o 'xnack[^ ]*'
# Bad:  xnack=off(unsupported)  or  no xnack mention
# Good: xnack=on
```

**Fix:** Build with the xnack+ feature flag and set the runtime variable:
```bash
# In CMake / build script:
AMDGPU_TARGETS="gfx900:xnack+"    # not just "gfx900"

# At runtime:
export HSA_XNACK=1
```

The build script (`build-llamacpp-rocm-vega.sh`) applies this automatically.

### "COMGR fails to parse code objects" / silent GPU failures (COv6 mismatch)

GPU operations silently fail or return garbage. The build completes but kernels don't load at runtime. COMGR error messages may appear in debug output.

**Cause:** `clang-21` (from Ubuntu 25.10) defaults to **Code Object v6** (ELF `EI_ABIVERSION=4`), but the HIP 5.7.1 runtime (`libamdhip64`) only supports up to **Code Object v5** (`EI_ABIVERSION=3`). The runtime silently fails to parse COv6 code objects.

Verify with:
```bash
readelf -h libggml-hip.so | grep ABI
# Bad:  OS/ABI: AMDGPU_HSA - AMDGPU OS, ABI Version: 4   (COv6)
# Good: OS/ABI: AMDGPU_HSA - AMDGPU OS, ABI Version: 3   (COv5)
```

**Fix:** Force Code Object v5 in the build:
```cmake
-DCMAKE_HIP_FLAGS="-mcode-object-version=5"
```

The build script applies this automatically.

### System hard-lock / Ubuntu becomes unresponsive (OOM)

The system completely freezes and only a hard reset helps. No mouse, no keyboard, no SSH.

**Cause:** On UMA APUs (like Vega 8), the GPU shares system RAM. With `-ngl 99` and `GPU_MAX_ALLOC_PERCENT=100`, the GPU can allocate nearly **all** system memory — the Linux OOM killer can't intervene fast enough because the allocation happens in kernel/GPU space, not in a killable userspace process.

A 9B Q4_K_M model needs ~5.5 GB for weights alone, plus KV cache (which grows with context size). With `-ngl 99 -c 4096`, total GPU memory usage can reach 8-10 GB on top of whatever the desktop and other apps are using.

**Fix:**
1. **Reduce GPU layers** — start with `-ngl 35` and increase gradually:
   ```bash
   ./start-qwen.sh              # safe defaults: -ngl 35 -c 2048
   ```
2. **Limit context size** — use `-c 2048` instead of `-c 4096`
3. **Use a smaller model** — 3B models (~2 GB) are much safer on Vega 8
4. **Close browsers and compositors** before running large models
5. **Monitor memory** in a separate terminal while loading:
   ```bash
   watch -n1 'free -h; echo "---"; HSA_OVERRIDE_GFX_VERSION=9.0.0 rocm-smi --showmeminfo vram 2>/dev/null'
   ```

If you want to try full offload at your own risk:
```bash
./start-qwen.sh --aggressive    # -ngl 99 -c 4096 (may hard-lock!)
```

### Inference segfault / hard crash with ROCm despite kernel tests passing

Individual GPU kernel tests (SCALE, MUL_MAT) pass, but actual model inference crashes:
- `-ngl 35`: Hard PC crash (GPU hang freezes the APU display adapter)
- `-ngl 1`: Segfault (exit 139) during model warmup
- `-ngl 0` with ROCm active: First prompt works, second prompt segfaults
- `HIP_VISIBLE_DEVICES=-1` (ROCm fully disabled): **Works perfectly**

**Cause:** This is a fundamental incompatibility between clang-21's generated code and the HIP 5.7.1 runtime's scheduler/graph execution engine. Individual kernel dispatches work fine, but the sustained, complex dispatch patterns during model inference trigger a bug in the HIP runtime. The crash occurs even with zero GPU layers (`-ngl 0`) as long as the ROCm backend is initialized.

This was confirmed through systematic isolation testing:

| Configuration | Result |
|---|---|
| `-ngl 35` (full GPU offload) | Hard PC crash (GPU hang) |
| `-ngl 1` `--no-mmap` | Segfault (exit 139) |
| `-ngl 0` (CPU only, ROCm active) | First prompt OK, second segfaults |
| `HIP_VISIBLE_DEVICES=-1` (ROCm disabled) | ✅ Works perfectly |
| `test-backend-ops -o SCALE` | ✅ 4/4 pass |
| `test-backend-ops -o MUL_MAT` (all quants) | ✅ All pass |

**Root cause:** The Ubuntu 25.10 ROCm stack has severe version mismatches:

| Component | Version | Expected |
|---|---|---|
| clang/LLVM | 21.1.2 | Matched set |
| hipcc / comgr / device-libs | 7.0.1 (experimental) | Matched set |
| HSA runtime | 6.1.2 | Matched set |
| HIP runtime (libamdhip64) | **5.7.1** | Should match above |

The HIP 5.7.1 runtime is ~2 major versions behind the compiler/device libs. This mismatch causes the scheduler to crash during sustained inference workloads.

**Action Plan / Fixes:** The current plan to resolve this is:
1. ~~**Increase `amdgpu.gttsize` in GRUB:**~~ (Failed - still segfaults).
2. **Kernel params from [AgentZ article](https://medium.com/@agentz/how-to-fix-rocm-pytorch-memory-faults-on-amd-gpus-segmentation-fault-page-not-present-544b9f62f627):** Three GRUB parameters that fix ROCm memory faults on AMD GPUs:
   ```
   amdgpu.cwsr_enable=1    # Disable compute wave save/restore (crash trigger on APUs)
   amd_iommu=on           # Disable IOMMU ("page not present" faults)
   ttm.pages_limit=12582912  # Raise TTM page limit (~48 GB vs default ~23 GB)
   ```
   Applied via: `GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amdgpu.gttsize=8192 amdgpu.cwsr_enable=1 amd_iommu=on ttm.pages_limit=12582912"` — requires `sudo update-grub` and reboot.
3. **Install Official AMD Drivers:** The Ubuntu 25.10 toolchain mismatch (Clang 21 + HIP 5.7.1) is to blame. Purge Ubuntu ROCm packages and install the official matched AMD ROCm stack.
3. **Fallback to Vulkan:** If official ROCm is unstable or slow, use the Vulkan backend which has native support for gfx90c/Vega 8 via Mesa RADV.

See the [Architecture Notes](ARCHITECTURE.md#rocm-runtime-crash-analysis) for the full technical analysis.

### "No GPU detected" / "Failed to open /dev/kfd"

**Fix:** Add your user to the `render` and `video` groups:
```bash
sudo usermod -aG render,video $(whoami)
```
Then **log out and log back in** (or reboot). Verify:
```bash
id -nG | grep render
ls -la /dev/kfd
```

### "hipblasStrsmBatched: candidate function not viable"

```
error: no matching function for call to 'hipblasStrsmBatched'
candidate function not viable: Nth argument would lose const qualifier
```

**Cause:** HIP 5.7's hipBLAS uses `float* const*` where modern CUDA/HIP uses `const float**`.

**Fix:** The build script patches this automatically (see [HIP57-PATCHES.md](HIP57-PATCHES.md) Patch 3).

### "hipStreamWaitEvent: no matching function"

```
error: no matching function for call to 'hipStreamWaitEvent'
candidate expects 3 arguments, 2 provided
```

**Cause:** HIP 5.7 only has the 3-arg version. HIP 6.x added a 2-arg overload.

**Fix:** The build script patches this automatically (see [HIP57-PATCHES.md](HIP57-PATCHES.md) Patch 2).

---

## Diagnostic Commands

### Check GPU detection

```bash
# ROCm (needs HSA_OVERRIDE_GFX_VERSION for Vega)
HSA_OVERRIDE_GFX_VERSION=9.0.0 rocminfo 2>/dev/null | grep -E "Name:|Marketing"

# ROCm memory
HSA_OVERRIDE_GFX_VERSION=9.0.0 rocm-smi --showmeminfo vram
HSA_OVERRIDE_GFX_VERSION=9.0.0 rocm-smi --showmeminfo gtt

# Vulkan
vulkaninfo --summary 2>/dev/null | grep -E "GPU|driver|apiVersion"
```

### Check HIP version

```bash
hipcc --version
# Look for: HIP version: 5.7.31921
```

### Check render node

```bash
ls -la /dev/dri/render*
# renderD128 = usually dGPU (NVIDIA)
# renderD129 = usually iGPU (AMD Vega 8)
```

### Check LM Studio ROCm backend targets

```bash
cat ~/.lmstudio/extensions/backends/llama.cpp-linux-x86_64-amd-rocm-*/backend-manifest.json | python3 -m json.tool | grep -A5 targets
```

### Check LM Studio logs for errors

```bash
# Most recent log
ls -t ~/.lmstudio/server-logs/2026-*/*.log | head -1 | xargs grep -i "rocm\|error\|abort\|invalid"
```

### Verify the custom build has gfx900 kernels

```bash
# Check the compiled shared library for gfx900 code objects
readelf -p .note llama.cpp-rocm-vega/lib/libggml-hip.so 2>/dev/null | grep gfx
# or
strings llama.cpp-rocm-vega/lib/libggml-hip.so | grep gfx900
```

### Verify xnack and Code Object version

```bash
# Check xnack status in all code objects
readelf -n llama.cpp-rocm-vega/lib/libggml-hip.so | grep -o 'xnack[^ ]*' | sort | uniq -c
# Should show: xnack=on  (NOT xnack=off or xnack=unsupported)

# Check Code Object version (need COv5 for HIP 5.7)
readelf -h llama.cpp-rocm-vega/lib/libggml-hip.so | grep 'ABI Version'
# Should show: ABI Version: 3  (COv5, NOT 4 which is COv6)

# Count all ELF code objects and verify they're all correct
for f in llama.cpp-rocm-vega/lib/libggml-*.so; do
    total=$(readelf -n "$f" 2>/dev/null | grep -c 'gfx900' || echo 0)
    xnack_on=$(readelf -n "$f" 2>/dev/null | grep -c 'xnack=on' || echo 0)
    echo "$f: $total code objects, $xnack_on with xnack=on"
done
```

### Run GPU kernel tests safely

```bash
# Test individual ops with a timeout to prevent hangs
export HSA_OVERRIDE_GFX_VERSION=9.0.0 HSA_ENABLE_SDMA=0 HSA_XNACK=1 GGML_HIP_UMA=1

# SCALE test (should pass quickly)
timeout 10 ./llama.cpp-rocm-vega/bin/test-backend-ops -o SCALE -b ROCm0

# MUL_MAT test (tests all quantization types)
timeout 60 ./llama.cpp-rocm-vega/bin/test-backend-ops -o MUL_MAT -b ROCm0
```

---

## Environment Variables Reference

| Variable | Value | Purpose |
|----------|-------|---------|
| `HSA_OVERRIDE_GFX_VERSION` | `9.0.0` | Make ROCm see gfx90c as gfx900 |
| `HSA_ENABLE_SDMA` | `0` | Disable SDMA (crashes on APU iGPUs) |
| `HSA_XNACK` | `1` | Enable xnack (required for UMA APU code objects) |
| `GGML_HIP_UMA` | `1` | Enable Unified Memory Architecture mode |
| `GPU_MAX_ALLOC_PERCENT` | `100` | Allow full GPU memory allocation |
| `GPU_SINGLE_ALLOC_PERCENT` | `100` | Allow single large allocations |
| `GPU_MAX_HEAP_SIZE` | `100` | Allow full heap usage |
| `GPU_FORCE_64BIT_PTR` | `1` | Force 64-bit pointers (needed > 4GB) |
| `HIP_VISIBLE_DEVICES` | `0` | Use only the first HIP device |
| `GGML_VK_DEVICE` | `0` | Vulkan: use first GPU (AMD iGPU) |
| `VK_ICD_FILENAMES` | `radeon_icd.json` | Vulkan: only load AMD RADV driver |
| `DRI_PRIME` | `pci-0000_0b_00.0` | Hint to use AMD iGPU render node |

### Kernel Boot Parameters (GRUB)


| Parameter | Value | Purpose |

|-----------|-------|---------|

| `amdgpu.gttsize` | `65536` | Increase GTT (Graphics Translation Table) size to 64GB. This is the max amount of system RAM the AMD driver is allowed to map as VRAM. |

| `ttm.pages_limit` | `16777216` | Raise TTM (Translation Table Maps) page limit to 64GB (16,777,216 pages * 4KB). This is the max amount of RAM the Linux kernel is allowed to give to the graphics subsystem. **Note: Both `amdgpu.gttsize` and `ttm.pages_limit` must be set together to exceed the default 8GB limit.** |

| `amdgpu.cwsr_enable` | `0` | Disable compute wave save/restore (fixes segfaults during inference) |

| `amd_iommu` | `on` | Enable/force IOMMU (fixes "page not present" memory faults) |



These are set in `/etc/default/grub` → `GRUB_CMDLINE_LINUX_DEFAULT`, then `sudo update-grub` + reboot.
Ref: [AgentZ — How to Fix ROCm Memory Faults on AMD GPUs](https://medium.com/@agentz/how-to-fix-rocm-pytorch-memory-faults-on-amd-gpus-segmentation-fault-page-not-present-544b9f62f627)

---

## Performance Tips

1. **Increase UMA VRAM in BIOS** to 16 GB:
   - Settings → Advanced → AMD CBS → NBIO → GFX → UMA Frame Buffer Size

2. **Use Q4_K_M quantization** for best quality/size tradeoff

3. **Try `-ngl 99`** to offload all layers to GPU, then reduce if out-of-memory

4. **Reduce context size** (`-c 2048` instead of `-c 4096`) to save VRAM

5. **Close other GPU-using apps** (browsers, compositors) to free VRAM
