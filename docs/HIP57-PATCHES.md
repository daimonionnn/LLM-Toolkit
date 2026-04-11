# HIP 5.7 Compatibility Patches

Ubuntu 25.10 packages HIP as version **5.7.31921** (via `hipcc 7.0.1+dfsg`, using `clang-21`). Modern llama.cpp expects HIP 6.1+. Six patches are needed to bridge the gap — four source/linker patches and two build configuration fixes.

All patches are applied automatically by `build-llamacpp-rocm-vega.sh`.

## Patch 1: HIP Version Check

**File:** `ggml/src/ggml-hip/CMakeLists.txt`

**Problem:** llama.cpp's CMake rejects HIP versions below 6.1:
```cmake
if (${hip_VERSION} VERSION_LESS 6.1)
    message(FATAL_ERROR "HIP version must be at least 6.1")
endif()
```

**Fix:** Relax the minimum to 5.5:
```bash
sed -i 's/VERSION_LESS 6.1/VERSION_LESS 5.5/' ggml/src/ggml-hip/CMakeLists.txt
```

**Risk:** Low. The core HIP runtime API used by llama.cpp hasn't changed significantly between 5.7 and 6.1. The version bump was mainly for new features (hipGraph, etc.) that llama.cpp doesn't use for basic inference.

---

## Patch 2: hipStreamWaitEvent Missing Flags Argument

**File:** `ggml/src/ggml-cuda/ggml-cuda.cu`

**Problem:** HIP 6.x added a 2-argument overload of `hipStreamWaitEvent(stream, event)`. HIP 5.7 only has the 3-argument form `hipStreamWaitEvent(stream, event, flags)`. The llama.cpp code uses the 2-arg version:

```cpp
cudaStreamWaitEvent(stream, concurrent_event->fork_event);        // Missing flags!
cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1]);  // Missing flags!
```

**Fix:** Add the third argument `0` (no flags):
```bash
sed -i 's/cudaStreamWaitEvent(stream, concurrent_event->fork_event)/cudaStreamWaitEvent(stream, concurrent_event->fork_event, 0)/g' ggml-cuda.cu
sed -i 's/cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events\[i - 1\])/cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[i - 1], 0)/g' ggml-cuda.cu
```

**Risk:** None. The flags argument `0` means "default behavior", which is identical to the 2-arg overload.

---

## Patch 3: hipblasStrsmBatched Const Correctness

**File:** `ggml/src/ggml-cuda/solve_tri.cu`

**Problem:** The `hipblasStrsmBatched` function signature changed between HIP 5.7 and 6.x:

```cpp
// HIP 5.7 signature (parameter 9 = A, parameter 11 = B):
hipblasStrsmBatched(..., float* const AP[], ..., float* BP[], ...)

// HIP 6.x / CUDA signature:
cublasStrsmBatched(..., const float** A, ..., float** B, ...)
```

The llama.cpp code declares:
```cpp
const float ** A_ptrs_dev;    // const float** — incompatible with float* const*
float ** X_ptrs_dev;          // float** — OK
```

HIP 5.7 wants `float* const*` for A (pointer to const-pointer-to-mutable-float), but the code has `const float**` (pointer to pointer-to-const-float). These are different const qualifications.

**Fix:** Cast `A_ptrs_dev` to `float *const *` at the call site (using a Python script for reliability):

```python
# In build script — replaces the exact cublasStrsmBatched call
old = '... A_ptrs_dev, n, X_ptrs_dev, ...'
new = '... (float *const *)A_ptrs_dev, n, X_ptrs_dev, ...'
```

**Risk:** Low. The cast changes which level of indirection is `const`, but we control the allocation and the data isn't modified through these pointers during the call.

---

## Patch 4: bfloat16 Multiple Definition (Linker)

**Not a source patch — CMake linker flag.**

**Problem:** HIP 5.7's `hip_bfloat16.h` header defines helper functions like `__float2bfloat16()`, `__bfloat1622float2()`, etc. **without** the `inline` keyword. Since these headers are included in every `.cu` file, the functions get emitted into every object file, causing hundreds of "multiple definition" linker errors.

```
/usr/bin/ld: multiple definition of `__float2bfloat16(float)'; acc.cu:(.text+0x0): first defined here
/usr/bin/ld: multiple definition of `__bfloat1622float2(__hip_bfloat162)'; ...
... (hundreds more)
```

This was fixed in HIP 6.x by marking these functions `inline` or moving them.

**Fix:** Pass `-z muldefs` to the linker, which allows multiple definitions (first one wins):

```cmake
-DCMAKE_SHARED_LINKER_FLAGS="-Wl,-z,muldefs"
```

Note: `-Wl,--allow-multiple-definitions` (GNU ld syntax) doesn't work if the system uses `mold` as the default linker. The `-z muldefs` flag is more portable.

**Risk:** Low. All the duplicate definitions are identical (same header, same code). The linker just picks the first one, which is correct.

---

## Patch 5: xnack+ Target for UMA APUs

**Not a source patch — CMake target configuration.**

**Problem:** Building with `AMDGPU_TARGETS=gfx900` (no feature suffix) produces code objects tagged `xnack=off(unsupported)`. The Vega 8 APU uses UMA (shared system RAM), which **requires** xnack page-fault handling. The HSA loader refuses to load code objects lacking xnack support → `"shared object initialization failed"`.

Verify the problem:
```bash
readelf -n libggml-hip.so | grep -o 'xnack[^ ]*' | sort | uniq -c
# Bad output:   125 xnack=off(unsupported)
# Good output:  125 xnack=on
```

**Fix:** Add the `:xnack+` feature flag to the GPU target:

```bash
AMDGPU_TARGETS="gfx900:xnack+"
```

And set at runtime:
```bash
export HSA_XNACK=1
```

**Risk:** None. The xnack+ flag enables page fault handling code paths in the generated kernels. This is required for UMA systems and has no downside.

---

## Patch 6: Force Code Object v5 (COv5)

**Not a source patch — CMake compiler flag.**

**Problem:** `clang-21` (from Ubuntu 25.10's LLVM 21) defaults to **Code Object v6** (`EI_ABIVERSION=4`). The HIP 5.7.1 runtime's COMGR library can only parse up to **Code Object v5** (`EI_ABIVERSION=3`). COv6 code objects are silently rejected at runtime, causing GPU operations to fail or produce garbage output.

Verify the problem:
```bash
readelf -h libggml-hip.so | grep 'ABI Version'
# Bad output:   ABI Version: 4  (COv6)
# Good output:  ABI Version: 3  (COv5)
```

**Fix:** Force COv5 via HIP compiler flags:

```cmake
-DCMAKE_HIP_FLAGS="-mcode-object-version=5"
```

**Risk:** None. COv5 is functionally identical to COv6 for the kernels llama.cpp uses. COv6 added metadata features that aren't relevant here.

---

## Version Compatibility Matrix

| Component | Ubuntu 25.10 Version | llama.cpp Expects | Notes |
|-----------|---------------------|-------------------|-------|
| HIP Runtime (libamdhip64) | **5.7.31921** | ≥ 6.1 | Patched (version check relaxed) |
| hipcc | 7.0.1+dfsg | — | Wraps clang-21 |
| Clang/LLVM | 21.1.2 | — | Generates COv6 by default (must force COv5) |
| comgr / device-libs | 7.0.1 | — | Experimental; mismatched with HIP runtime |
| HSA runtime | 6.1.2 | — | Mismatched with HIP runtime |
| hipBLAS | 2.2.0 (approx) | ≥ 6.1 API | Const correctness differs from 6.x |
| CMake | 3.31 | ≥ 3.21 | OK |

> **Warning:** Despite all 6 patches, the Ubuntu 25.10 ROCm stack has a **fundamental runtime bug** — individual GPU kernel tests pass, but sustained inference crashes due to the severe mismatch between the compiler (clang 21 / ROCm 7.x era) and the HIP runtime (5.7 / ROCm 5.x era). See [ARCHITECTURE.md — ROCm Runtime Crash Analysis](ARCHITECTURE.md#rocm-runtime-crash-analysis) for details. **Use Vulkan instead.**

## When Patches May Break

These patches target the HIP 5.7 → 6.1 gap. They may need updating when:

- **llama.cpp adds new HIP API calls** that don't exist in 5.7
- **Ubuntu upgrades HIP** to 6.x+ (patches become unnecessary)  
- **New `.cu` files** use `hipStreamWaitEvent` without 3 args
- **hipBLAS calls change** signatures in new llama.cpp code

When the build breaks after a `git pull`, check the error message and compare with the patches above — it's usually the same pattern.
