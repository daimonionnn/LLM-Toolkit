# Troubleshooting

## Quick Diagnostics

Run the built-in diagnostic mode:

```bash
./run/launch-lmstudio-vulkan.sh --diagnose
```

This shows:
- Which GPU architectures LM Studio's ROCm backends were compiled for
- Recent ROCm errors from LM Studio logs
- Memory information (VRAM/GTT)
- A recommendation for your hardware

---

## Multi-GPU: Targeting the Vega 8 when dGPUs are also present

> **Current rig (September 2026): single GPU.** The Vega 8 is alone at
> `/dev/dri/renderD128` / `card0`, ROCm index 0, so nothing here is needed today.
> Kept because the numbering *will* move again — this section is what to read when
> a dGPU is added back.

With several AMD GPUs present, ROCm picks the wrong one without explicit selection. Render node numbers shift whenever dGPUs are added or removed — and, as the 26.04 reinstall showed, they can move even without a hardware change. That is why all scripts detect by PCI ID instead of hardcoding a node.

### Docker scripts (automatic)

`run/run-docker-rocm.sh` and `run/run-docker-rocm7.sh` auto-detect the Vega 8 by PCI ID:

```bash
# Auto-detect confirmed: looks for PCI ID 0x1638 (Vega 8)
./run/run-docker-rocm.sh /path/to/model.gguf
# → "  Vega 8 render node: /dev/dri/renderD128"
```

If auto-detect fails (different APU revision / PCI ID):
```bash
VEGA8_RENDER_NODE=/dev/dri/renderD128 ./run/run-docker-rocm.sh /path/to/model.gguf
```

To identify your render nodes:
```bash
for node in /sys/class/drm/renderD*/device; do
    render=$(basename $(dirname "$node"))
    dev=$(cat "$node/device" 2>/dev/null)
    vendor=$(cat "$node/vendor" 2>/dev/null)
    echo "$render  vendor=$vendor  device=$dev"
done
# On the current single-GPU rig:
# renderD128  vendor=0x1002  device=0x1638   ← Vega 8 iGPU
#
# With the two R9700s installed (June 2026) it looked like this instead:
# renderD128  vendor=0x1002  device=0x7551   ← Radeon AI PRO R9700
# renderD129  vendor=0x1002  device=0x7551   ← Radeon AI PRO R9700
# renderD130  vendor=0x1002  device=0x1638   ← Vega 8 iGPU
```

### Baremetal (manual)

`run/run-rocm7-baremetal.sh` auto-detects the Vega 8 agent index (override with `VEGA8_ROCM_DEVICE=N`). To do it manually, verify which HSA agent is the Vega 8:
```bash
rocminfo | grep -B2 -A8 'gfx90'
```

Then set before running any ROCm binary:
```bash
# June 2026: GPU 0+1 = gfx1201 (R9700s), GPU 2 = gfx90c (Vega 8)
export ROCR_VISIBLE_DEVICES=2   # index of Vega 8 — verify with rocminfo (may differ on your system)
export HIP_VISIBLE_DEVICES=0    # relative index within ROCR_VISIBLE_DEVICES mask
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HSA_ENABLE_SDMA=0
export HSA_XNACK=0
```

Or simply use the wrapper script which auto-detects the correct index:
```bash
bash run/run-rocm7-baremetal.sh /path/to/model.gguf -ngl 99
```

---

## Common Errors

### `vk::DeviceLostError` / `amdgpu: ring comp_1.0.1 timeout` — micro-batch too large for the context

```
terminate called after throwing an instance of 'vk::DeviceLostError'
  what():  vk::Queue::submit: ErrorDeviceLost
```

and in `dmesg`:

```
amdgpu: ring comp_1.0.1 timeout, signaled seq=..., emitted seq=...
amdgpu: Starting comp_1.0.1 ring reset
amdgpu: [drm] device wedged, but recovered through reset
```

**Cause:** a single compute dispatch runs longer than the driver's watchdog. It is *not*
out of memory — it happens with tens of GB free. The work in one attention dispatch scales
with `ctx × ubatch × head_dim`, and that product, not `ctx × ubatch` alone, is what
predicts the hang. Measured on Vulkan:

| Model | Head dim | ctx | ub | ctx × ub | × head dim | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-35B-A3B | 256 | 32768 | 2048 | 67.1 M | 17.2e9 | OK (120.1 t/s) |
| Qwen3.5-35B-A3B | 256 | 32768 | 4096 | 134.2 M | 34.4e9 | **DEVICE LOST** |
| gemma-4-E4B-it | 512 | 32768 | 2048 | 67.1 M | 34.4e9 | **DEVICE LOST** |
| gemma-4-E4B-it | 512 | 16384 | 2048 | 33.6 M | 17.2e9 | OK |

The two failures share a product of 34.4e9 and the two passes share 17.2e9, while
`ctx × ubatch` alone puts a pass and a failure in the same 67.1 M bucket. An earlier
version of this page blamed `ctx × ubatch > 2²⁶` from the Qwen row only; that rule calls
the gemma 32K/2048 case safe, and it crashes.

**Fix:** lower the micro-batch. `run/start-llama-server.sh` derives it automatically as
`min(4096, 2²⁵/CTX)` — 2²⁵ rather than 2²⁶ because the cap has to hold for the largest
head dim in use here (512). If you pass `-ub` yourself, keep `ctx × ub × head_dim` under
about 20e9. The GPU recovers on its own via a ring reset, but the server process dies.

This is a Vulkan limit, not a hardware one: ROCm ran `32768 × 4096` on the 35B fine
(99.31 t/s prefill). The launcher caps the two backends separately — `2²⁵/CTX` for Vulkan,
`2²⁶/CTX` for ROCm. See [benchmarks.md](benchmarks.md#the--ub-ceiling-that-hangs-the-gpu).

### `-ngl 0` does not run on the CPU

`-ngl 0` used to mean "no GPU offload". Upstream changed the `-ngl` default to `auto`, and
with a GPU backend present the model is still offloaded — measured 91 % GPU busy and
6.5 GB in GTT with `-ngl 0`, versus 157 MB with `-dev none`.

**Use `-dev none`.** Any CPU benchmark or comparison made with `-ngl 0` on current
llama.cpp is measuring the GPU. This affected both this repo's benchmark harness and
`start-llama-server.sh --cpu`; both are fixed.

### `-fa auto` silently enables flash attention on ROCm, halving prefill

`-fa` defaults to `auto`, which probes the backend. On gfx900 the probe succeeds — the
generic FA tile kernel compiles — so FA is enabled, and without the local FA patch that is
the worst setting for prefill. Measured on gemma at a 3330-token prompt:
`-fa 0` = 112.8 t/s, `-fa 1` = 48.9, **`-fa auto` = 48.9**.

Pass `-fa` explicitly — always, on every ROCm launch. The right value depends on the
build, and the `auto` probe cannot tell the difference: `-fa 1` on a build carrying
`patches/0001`, `-fa 0` on a stock one. `run/run-rocm7-baremetal.sh` (patched) defaults to
`-fa 1`; `run/run-docker-rocm7.sh` (unpatched image) to `-fa 0`.



### "no ROCm-capable device is detected" / rocminfo `HSA_STATUS_ERROR_OUT_OF_RESOURCES` with the gfx900 override

```
ggml_cuda_init: failed to initialize ROCm: no ROCm-capable device is detected
# and, with HSA_OVERRIDE_GFX_VERSION=9.0.0 set:
Call returned HSA_STATUS_ERROR_OUT_OF_RESOURCES
```

**Cause:** AMD's modular ROCm packages (`amdrocm-core` 7.13+, arch-specific builds like gfx120x) replaced the classic ROCm install. That ROCr runtime rejects `HSA_OVERRIDE_GFX_VERSION=9.0.0`, and its rocBLAS ships no gfx9 tensile kernels (the backported gfx900 files in `/opt/rocm/lib/rocblas/library/` are gone), so the baremetal gfx900 path can't work — even though the runtime *enumerates* the Vega natively as gfx90c.

**Fix:** Use the self-contained Docker image, which bundles its own ROCm 7.2 userspace where the override works:
```bash
./run/run-docker-rocm7.sh /path/to/model.gguf -ngl 99 -c 8192 -fa 0
```
`run/run-rocm7-baremetal.sh` detects this situation and aborts with the same advice (bypass with `SKIP_ROCM_CHECKS=1`).

### "ROCm error: invalid device function"

```
ggml_cuda_compute_forward: MUL_MAT failed
ROCm error: invalid device function
llama.cpp abort:98: ROCm error
```

**Cause:** The ROCm binary doesn't contain kernels for your GPU architecture. LM Studio's ROCm backend has kernels for gfx1030+ only. Your Vega 8 is gfx90c.

**Fix:** Use the main launcher (`./run/start-llama-server.sh`), Vulkan (`./run/launch-lmstudio-vulkan.sh`), or build llama.cpp with gfx900 (`./build/build-llamacpp-rocm7-baremetal.sh`).

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

**Note:** This was an earlier approach that was superseded. The Docker ROCm 6.2.4 build uses plain `gfx900` (xnack-agnostic) with `HSA_XNACK=0`, which avoids this entirely. The `gfx900:xnack+` / `HSA_XNACK=1` combination hard-freezes the Vega 8 PC.

~~**Fix:** Build with the xnack+ feature flag and set the runtime variable:~~
```bash
# DO NOT USE — HSA_XNACK=1 hard-freezes Vega 8 PC
# AMDGPU_TARGETS="gfx900:xnack+"
# export HSA_XNACK=1
```

**Current approach (Docker ROCm 6.2.4):** Use plain `gfx900` (xnack-agnostic) with `HSA_XNACK=0`.

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
  ./run/run-rocm7-baremetal.sh /path/to/model.gguf -ngl 35 -c 2048
   ```
2. **Limit context size** — use `-c 2048` instead of `-c 4096`
3. **Use a smaller model** — 3B models (~2 GB) are much safer on Vega 8
4. **Close browsers and compositors** before running large models
5. **Monitor memory** in a separate terminal while loading:
   ```bash
   watch -n1 'free -h; echo "---"; HSA_OVERRIDE_GFX_VERSION=9.0.0 rocm-smi --showmeminfo vram 2>/dev/null'
   ```

If you want to try full offload explicitly:
```bash
./run/start-llama-server.sh                # -ngl 99 on Vulkan (default backend)
./run/start-llama-server.sh --rocm-docker  # -ngl 99 on ROCm 7.2 Docker (needs 64 GB GTT)
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

**Action Plan / Fixes:**
1. ~~**Increase `amdgpu.gttsize` in GRUB:**~~ (Failed - still segfaults).
2. ~~**Kernel params (AgentZ article):**~~ Applied and verified — GRUB params fix memory faults but the HIP 5.7.1 / Clang-21 mismatch still causes slot-initialization segfaults on the host.
3. ✅ **ROCm 7.2 baremetal:** Default path — use `./run/start-llama-server.sh` or `./run/run-rocm7-baremetal.sh`.
4. ✅ **Docker with ROCm 7.2 or 6.2.4:** Fully working — use `./run/run-docker-rocm7.sh` or `./run/run-docker-rocm.sh`.
5. **Fallback to Vulkan:** Use the Vulkan backend which has native support for gfx90c/Vega 8 via Mesa RADV (`./run/start-llama-server.sh --vulkan`).

See the [Architecture Notes](ARCHITECTURE.md#legacy-host-hip-571-crash-analysis) for the full technical analysis.

### Docker ROCm 6.2.4 Workaround (Working Legacy Solution)

The host Ubuntu ROCm stack (HIP 5.7.1 + Clang-21) has an unresolvable version mismatch that causes segfaults at slot initialization. Running llama.cpp in a Docker container with a coherent **ROCm 6.2.4** base image fixes this completely.

**Quick start:**
```bash
./run/run-docker-rocm.sh /path/to/model.gguf -ngl 99 -c 2048 --no-warmup
# Server available at http://127.0.0.1:8080
```

**How it works:** `run/run-docker-rocm.sh` auto-builds the image from `build/Dockerfile.rocm64` on first run, then starts the container with GPU device passthrough (`/dev/kfd`, `/dev/dri`).

**Key findings and fixes applied in `build/Dockerfile.rocm64`:**

| Problem | Fix |
|---------|-----|
| `__hip_fp8_e4m3` undefined (gfx900 has no FP8) | Shell loop replaces each `typedef __hip_fp8_* __nv_fp8_*;` with a stub struct in `vendors/hip.h` |
| `HSA_XNACK=1` freezes entire PC | `HSA_XNACK=0` — disables xnack to prevent system hang |
| `GGML_HIP_UMA=1` + `XNACK=0` = segfault | Historical — `GGML_HIP_UMA` has since been removed from llama.cpp; plain hipMalloc into GTT is the default and works. The lesson stands: anything relying on page faults needs XNACK, and XNACK=1 freezes the Vega 8 |
| Server only reachable inside container | `--host 0.0.0.0` passed to `llama-server` so Docker port mapping works |
| Old ROCm 6.4.4 image: `.so.6` vs `.so.5` ABI mismatch | Use `rocm/dev-ubuntu-24.04:6.2.4` (stable, gfx900-compatible) |

**Confirmed working output:**
```
ggml_cuda_init: found 1 ROCm devices (Total VRAM: 65536 MiB)
  Device 0: AMD Radeon Graphics, gfx900:xnack- (0x900)
load_tensors: offloaded 33/33 layers to GPU
main: server is listening on http://0.0.0.0:8080
```

**Stop the container:**
```bash
docker stop $(docker ps -q --filter ancestor=llama-server-rocm-vega)
```

**Rebuild image after Dockerfile changes:**
```bash
docker rmi llama-server-rocm-vega
./run/run-docker-rocm.sh /path/to/model.gguf ...
```

**Check what's using GPU memory:**
```bash
rocm-smi --showmeminfo vram
fuser /dev/kfd /dev/dri/render* | tr ' ' '\n' | sort -u | \
  xargs -I{} sh -c 'echo -n "PID {}: "; cat /proc/{}/cmdline | tr "\0" " "; echo'
```
The desktop compositor (GNOME Shell, Xwayland), VS Code, and Firefox each hold a small amount of VRAM (~1 GiB total). The model itself is the dominant consumer.

---

### ROCm 7.2 Docker: `rocBLAS error: Cannot read TensileLibrary.dat: Illegal seek for GPU arch: gfx900`

Inference completes model loading but crashes on the first matrix multiply:

```
rocBLAS error: Cannot read /opt/rocm-7.2.0/lib/rocblas/library/TensileLibrary.dat: Illegal seek for GPU arch : gfx900
```

**Cause:** ROCm 7.x ships `TensileLibrary.dat` without any `gfx900` entries. When rocBLAS looks up kernels for gfx900, it falls through to the lazy loader and looks for `TensileLibrary_lazy_gfx900.dat` — which also doesn't exist in ROCm 7.

**Fix:** The `build/Dockerfile.rocm7-vega` uses a multi-stage build to backport gfx900 tensile files from ROCm 6.3.4:

```dockerfile
FROM rocm/dev-ubuntu-22.04:6.3.4 AS rocm6-libs
RUN apt-get update -qq && apt-get install -y rocblas   # not included in base image!

FROM rocm/dev-ubuntu-22.04:7.2
COPY --from=rocm6-libs /opt/rocm/lib/rocblas/library/ /tmp/rocm6-rocblas/
RUN # copies *gfx900* files + TensileLibrary_lazy_gfx900.dat to /opt/rocm/lib/rocblas/library/
```

The key file is `TensileLibrary_lazy_gfx900.dat` — it's the kernel index rocBLAS checks first. The full set of `*gfx900*.co`, `*gfx900*.dat`, and `*gfx900*.hsaco` files must also be present alongside it.

**Note:** The `rocm/dev-ubuntu-22.04:6.3.4` base image does **not** have rocblas installed — `apt-get install rocblas` is required inside that stage or the COPY will produce an empty directory.

Rebuild to apply:
```bash
docker rmi llama-rocm7-vega
docker build -t llama-rocm7-vega -f build/Dockerfile.rocm7-vega build/
```

---

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
# Node numbers depend on PCIe enumeration and change when dGPUs are
# added/removed — identify by PCI device ID instead (Vega 8 = 0x1638):
for n in /sys/class/drm/renderD*/device; do
    echo "$(basename "$(dirname "$n")")  $(cat "$n/device")"
done
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
readelf -p .note llm/rocm-vega/lib/libggml-hip.so 2>/dev/null | grep gfx
# or
strings llm/rocm-vega/lib/libggml-hip.so | grep gfx900
```

### Verify xnack and Code Object version

```bash
# Check xnack status in all code objects
readelf -n llm/rocm-vega/lib/libggml-hip.so | grep -o 'xnack[^ ]*' | sort | uniq -c
# Should show: xnack=on  (NOT xnack=off or xnack=unsupported)

# Check Code Object version (need COv5 for HIP 5.7)
readelf -h llm/rocm-vega/lib/libggml-hip.so | grep 'ABI Version'
# Should show: ABI Version: 3  (COv5, NOT 4 which is COv6)

# Count all ELF code objects and verify they're all correct
for f in llm/rocm-vega/lib/libggml-*.so; do
    total=$(readelf -n "$f" 2>/dev/null | grep -c 'gfx900' || echo 0)
    xnack_on=$(readelf -n "$f" 2>/dev/null | grep -c 'xnack=on' || echo 0)
    echo "$f: $total code objects, $xnack_on with xnack=on"
done
```

### Run GPU kernel tests safely

```bash
# Test individual ops with a timeout to prevent hangs
# XNACK=0: XNACK=1 hard-freezes the Vega 8 PC
export HSA_OVERRIDE_GFX_VERSION=9.0.0 HSA_ENABLE_SDMA=0 HSA_XNACK=0

# SCALE test (should pass quickly)
timeout 10 ./llm/rocm-vega/bin/test-backend-ops -o SCALE -b ROCm0

# MUL_MAT test (tests all quantization types)
timeout 60 ./llm/rocm-vega/bin/test-backend-ops -o MUL_MAT -b ROCm0
```

---

## Environment Variables Reference

| Variable | Value | Purpose |
|----------|-------|---------|
| `HSA_OVERRIDE_GFX_VERSION` | `9.0.0` | Make ROCm see gfx90c as gfx900 |
| `HSA_ENABLE_SDMA` | `0` | Disable SDMA (crashes on APU iGPUs) |
| `HSA_XNACK` | `0` | Disable xnack — prevents PC hard-freeze on Vega 8 iGPU (`1` freezes the entire system) |
| `GGML_HIP_UMA` | — | Removed from llama.cpp (no longer a build option or env var); listed here because older docs reference it |
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

| ~~`amdgpu.cwsr_enable`~~ | ~~`0`~~ | **Do not set.** Was believed to fix inference segfaults, but with CWSR disabled the Qwen 35B model fails to load and crashes during loading (see [benchmarks.md](benchmarks.md)). Leave CWSR at the driver default. |

| `amd_iommu` | `on` | Optional. Was applied while chasing "page not present" memory faults; **not** part of the verified-working config and not set by `setup/bootstrap-host.sh`. Add only if you actually hit those faults. |



**The verified-working set is just the first two.** That is exactly what
`setup/bootstrap-host.sh` writes:

```
GRUB_CMDLINE_LINUX_DEFAULT="quiet splash amdgpu.gttsize=65536 ttm.pages_limit=16777216"
```

These are set in `/etc/default/grub` → `GRUB_CMDLINE_LINUX_DEFAULT`, then `sudo update-grub` + reboot.
Ref: [AgentZ — How to Fix ROCm Memory Faults on AMD GPUs](https://medium.com/@agentz/how-to-fix-rocm-pytorch-memory-faults-on-amd-gpus-segmentation-fault-page-not-present-544b9f62f627)

---

## Performance Tips

1. **Unlock 64 GB GTT** via GRUB (required for large models on Vega 8 UMA):
   Add to `/etc/default/grub` `GRUB_CMDLINE_LINUX_DEFAULT`:
   ```
   amdgpu.gttsize=65536 ttm.pages_limit=16777216
   ```
   Then `sudo update-grub && reboot`.
   Verify: `cat /sys/kernel/debug/dri/*/vma_mm` or `rocm-smi --showmeminfo gtt`

2. **Set UMA VRAM carve-out in BIOS** to 16 GB (optional — GTT covers most use cases):
   Settings → Advanced → AMD CBS → NBIO → GFX → UMA Frame Buffer Size

3. **Use Q4_K_M quantization** for best quality/size tradeoff

4. **Try `-ngl 99`** to offload all layers to GPU, then reduce if out-of-memory

5. **Reduce context size** (`-c 2048` instead of `-c 4096`) to save VRAM

6. **Close other GPU-using apps** (browsers, compositors) to free VRAM
