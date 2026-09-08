# ROCm performance audit — Vega 8 (gfx90c as gfx900), 2026-09-08

Why ROCm trails Vulkan on identical silicon, and what to do about it — ranked.

Method: five parallel code readers mapped build, runtime, kernel dispatch, memory
path and project history (llama.cpp 465e49b, ROCm 7.2.0, kernel 7.0). Five lenses
proposed 35 candidates. The adversarial-verification stage of the workflow did not
run (spend limit), so every candidate below was re-checked by hand against three
questions: *does the code path exist and apply on gfx900?*, *was it already tried or
contradicted by a measurement?*, *is the gain physically plausible?* Items marked
**(verified)** were confirmed directly on this box; **(agent)** means the claim rests
on an agent's cited file:line or binary metadata that was not independently
re-extracted.

Starting point (Qwen3.5-35B-A3B Q4_K_M, `-c 8192`, prefill / decode t/s at ~128/~1K/~4K):

| | Prefill | Decode |
|---|---|---|
| Vulkan `-fa 1` | 73.35 / 159.06 / 153.67 | 21.73 / 21.56 / 20.95 |
| ROCm `-fa 0` | 47.73 / 94.48 / 88.96 | 18.70 / 17.79 / 14.66 |
| gap | −35 % / −41 % / −42 % | −14 % / −17 % / −30 % |

---

## 1. Central diagnosis

The gap is not one thing. It is three stacked prefill causes and two separate decode
causes, each with a code-level mechanism.

### Prefill (−40 %)

**a. MoE expert GEMMs run emulated dp4a.** `mmq.cu:378-383` has an explicit rule:
on `cc == GGML_CUDA_CC_VEGA`, MMQ is used *only* for MoE (`return n_experts > 0`);
dense GEMMs go to rocBLAS. gfx900 has no `v_dot4`, so `ggml_cuda_dp4a`
(`common.cuh:717-730`) is inline asm: 4× `v_mul_i32_i24` + 2× `v_add3_u32` = **6 VALU
instructions per 4 MACs**. Vulkan's `mul_mm_id` shaders on the same GPU use packed
fp16 FMA at 2 MACs/instruction. The 35B's expert matmuls are roughly half its prefill
FLOPs. **(verified)**

**b. MMQ has no GCN tile config.** `mmq.cuh:229-269` routes everything that is not
CDNA/RDNA3/RDNA4 to the *RDNA2* table (wave32, I=128, occupancy 2). On gfx900 that
occupancy caps kernels at 128 VGPRs; the installed `mul_mat_q<Q4_K,64>` and
`<Q6_K,64>` sit at 128 VGPRs with 15-16 VGPRs spilled to scratch inside an
~8500-instruction loop. **(agent — binary metadata)**

**c. The harness runs `-ub 512`; the launchers run `-ub 2048`.** With 256 experts,
8 active, a 512-token ubatch gives ~16 tokens per expert, so MMQ's 64-column tiles
are ~75 % empty. `run/run-rocm7-baremetal.sh:167` and the Docker launcher both pass
`-b 2048 -ub 2048` (June 2026 measured +22 % at 4K); `bench/run-all-benchmarks.sh`
never does. **Every ROCm number in the tables is a configuration nobody runs.**
**(verified)**

**d. Dense GEMMs: dequant → F16 → rocBLAS `HH` kernels** with fp16 accumulation, from
the 6.3.4 backport's *thin* Alik_Bljk library (122 solutions vs 880 in the SS/f32
set). Which Tensile solution rocBLAS actually dispatches for llama.cpp's shapes has
**never been observed** on this rig. **(verified: no trace in repo)**

### Decode — two components

**e. Context-independent (−14 % at 128 ctx).** `GGML_HIP_GRAPHS=OFF` was imported
from a gfx906 project for stability and **never A/B'd here**. Every one of the 35B's
~1500-2000 ops per token (40 layers, 30 gated-DeltaNet layers with many small ops)
is a separate `hipLaunchKernel`; Vulkan batches ~100 nodes per command buffer and
lets the GPU's CP dispatch. Effective bandwidth: ROCm ~37 GB/s, Vulkan ~43 GB/s.
**(verified: build flag; launch count is agent arithmetic)**

**f. Context-dependent falloff (ROCm −22 % from 128→4K; Vulkan −4 %).** This is
**not bandwidth**: KV adds ~80 MiB to a ~2 GB/token stream (~4 %). With `-fa 0`
(ROCm's best), decode attention is `mmvf` with 128-thread blocks and one block per
KV row → ~65 000 two-wave blocks per attention layer per token at 4K, and every K
head is re-streamed by its 8 Q heads (GQA ratio not folded; Vulkan folds it).
`-fa 1` cannot rescue it: on gfx900 the only FA kernel is the generic tile kernel,
using the CDNA-shared config (occupancy 2 → 128-VGPR cap → spills) and a 5-instruction
fp16 MAC fallback because `V_DOT2_F32_F16_AVAILABLE` is not defined for gfx900
(`common.cuh:760-778`). That is also exactly why FA ON collapses prefill.
**(verified: fattn.cu / common.cuh dispatch; spill counts agent)**

**g. Memory placement — root cause found.** ROCm ignores the BIOS carve-out because
of a *kernel* rule, not llama.cpp: `amdgpu_ttm_init()` sets `apu_prefer_gtt` when
`AMD_IS_APU && real_vram_size (16 GiB) < gtt_size (64 GiB from amdgpu.gttsize)`, and
`amdgpu_amdkfd_gpuvm_alloc_memory_of_gpu()` then rewrites every VRAM allocation
(= every `hipMalloc`) to GTT. Proof on this box: `/sys/class/kfd/kfd/topology/nodes/1`
reports `local_mem_size 0` and a single FB_PUBLIC bank of exactly
`68719476736 B = ttm.pages_limit << 12`. Consequence: ROCm weights get snooped GTT
PTEs, Vulkan's get non-snooped carve-out VRAM. Magnitude on this GPU: **unmeasured**.
**(verified in sysfs)**

---

## 2. The numbers themselves are less clean than the tables imply

These affect every backend equally, so the *comparisons* stand; the *absolute*
figures do not mean what the column headers say.

1. **The ~128 row absorbs cold start.** `--no-warmup` plus a fresh server per
   backend means the first request pays code-object load and rocBLAS's lazy Tensile
   library load. That is most of why ROCm shows 47.73 at 128 but 94.48 at 1K.
2. **The 1K/4K rows are incremental prefill at a warm KV, not cold prefill.** The
   harness builds all three prompts by truncating the *same* repeated word list
   (`bench/run-all-benchmarks.sh:403`), sends them to one slot, and never sets
   `cache_prompt` (server default `true`, `common.h:627`). `prompt_per_second` is
   computed over `n_prompt_processed`, which is zeroed after a prefix hit
   (`server-context.cpp:3394`) — so it is a genuine throughput of the *new* tokens,
   at a nonzero KV depth. **(verified)**
3. **n = 1**, no repetitions, no spread. (Cross-checked 2026-09-08: `llama-bench
   -dev none -r 2` and the server harness agree within 3 % on CPU prefill, so the
   harness is accurate — it is just unreplicated.)
4. **`-ub 512`** (item c above).

Fix: `llama-bench -p 512,1024,4096 -n 64 -d 0,1024,4096 -r 3` gives cold prefill and
decode-at-depth with stddev, no server, no cache ambiguity. Or send
`"cache_prompt": false` and a warmup request in the harness.

---

## 3. Ranked recommendations

Score = (expected gain × confidence) / effort. "Test" names the exact command.
`$ROCMENV` = `ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 HSA_OVERRIDE_GFX_VERSION=9.0.0 HSA_ENABLE_SDMA=0 HSA_XNACK=0 GPU_MAX_ALLOC_PERCENT=100`.
`$M35` = the Qwen3.5-35B GGUF, `$MG` = gemma-4-E4B.

### Tier 1 — no rebuild, high confidence

| # | Change | Expected | Why it should work |
|---|---|---|---|
| 1 | **`-ub 4096`** — DONE 2026-09-08, adopted in all three launchers | **Measured +70 % ROCm prefill at 4K** (84.2 → 143.5 t/s) and **+42 % on Vulkan** (139.1 → 197.9). Estimate above was +18-22 %; the real figure is far larger because the June estimate came from the server harness (warm-KV incremental prefill), which flattens the effect. Optimum is `ubatch ≥ prompt length`. Costs ~2 GB GTT. |
| 2 | **Pin `-fa 0` in `run/run-rocm7-baremetal.sh`** | **DONE 2026-09-08** — was costing 57 % of prefill | Confirmed by measurement, not inference: gemma, 3330-token prompt, `-fa 0` = 112.78 t/s, `-fa 1` = 48.91, **`-fa auto` = 48.90**. The launcher passed no `-fa`, so every launch through it ran with FA on. Fixed; still overridable. |
| 3 | **Fix the harness** (warmup request, `cache_prompt:false` or `llama-bench -d`, `-r 3`) | trust, not speed | Section 2. Cheapest change with the largest effect on decision quality. |
| 4 | `-ctk q8_0` (with `-fa 0`; `-ctv` needs FA) | decode 4K +3-4 % | June: +3.5 % at 4K. Never combined with `-ub 2048`, never on baremetal. |
| 5 | `power_dpm_force_performance_level=high`, then `pp_power_profile_mode=5` (COMPUTE) | +2-4 % both | June: high = +3 %. COMPUTE profile never tried; per-token idle gaps let the SMU drop SCLK. Needs root, not persistent. |

### Tier 2 — one reboot or one env var, medium confidence

| # | Change | Expected | Caveat |
|---|---|---|---|
| 6 | **`amdgpu.gttsize=16384`** (below the 16 GiB carve-out) | gemma decode +8-13 %, prefill +5-10 % | Flips `apu_prefer_gtt` off → `hipMalloc` lands in the non-snooped carve-out (mechanism g). **The 35B will not fit in that boot** (single 20.9 GB `hipMalloc` > 16 GiB pool). Evidence for the size of the gain is the old GGML_HIP_UMA table, which other rows contradict — measure, do not assume. Verify the flip: the KFD bank must read 16 GiB, `local_mem_size` non-zero. |
| 7 | `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` | 35B prefill −10..+15 %, gemma larger either way | Moves dense GEMMs from the 122-solution fp16 `HH` library to the 880-solution `SS` set. Unknown sign; also removes fp16 accumulation. Env only. |
| 8 | **Diagnostic: which Tensile kernels run** — `AMD_LOG_LEVEL=4` or `ROCBLAS_LAYER` on one prefill, histogram `ShaderName` | decides #7 and whether `ROCBLAS_TENSILE_GEMM_OVERRIDE_PATH` is worth it | Nobody has ever looked. A fallback/GSU/32×32 solution on the dominant shape would be a +10-40 % lever on the dense share. |
| 9 | Page-contiguity hygiene before load: `sync; drop_caches; compact_memory` (+ optionally `--load-mode dio`) | decode 0..+10 %, variance ↓ | GTT is `ttm_cached` → not pooled → order stepped down under fragmentation → 4 KiB PTEs instead of 2 MiB fragments. Gain is 0 if TTM already gets contiguous runs; `/proc/buddyinfo` before/after tells. |
| 10 | `-bs` (GPU-side sampling) | decode +1-4 % | Removes the 1 MB logits round trip for the 248k vocab. Experimental flag. |

### Tier 3 — rebuild A/B (separate install prefix, one flag each)

| # | Change | Expected | Risk |
|---|---|---|---|
| 11 | **`GGML_HIP_GRAPHS=ON`** | decode +5-15 % at every context if launch overhead is the gap (mechanism e); 0 if kernel time dominates | Stability on gfx900/ROCm 7.2 untracked; `GGML_CUDA_DISABLE_GRAPHS=1` is the runtime escape hatch. The OFF decision was never measured here. |
| 12 | `GGML_HIP_EXPORT_METRICS=ON` (diagnostic build) | per-kernel VGPR/spill/occupancy on gfx900 | Settles #13/#14 before patching. |
| 13 | `GGML_CUDA_FORCE_CUBLAS=ON`, **measured only at `-ub ≥ 2048`** | 35B prefill −30..+20 % | Routes experts off emulated-dp4a MMQ onto per-expert dequant + Tensile fp16 (2 MACs/instr, hand-scheduled asm) at the cost of ~120 stream syncs per ubatch. Certain loss at `-ub 512`. |

### Tier 4 — source patches (upstream-divergent, but small)

| # | Change | Expected | Notes |
|---|---|---|---|
| 14 | **GCN MMQ config**: route `GGML_CUDA_CC_IS_GCN` to the `pascal_dp4a` table (I=64) in `mmq.cuh:243` + device `#elif defined(GCN)` | 35B prefill +0-8 % | Two lines. Halves the tile so the 512-row expert matrices and 128-VGPR cap fit without spilling. Could regress if LDS-bound; `test-backend-ops perf -o MUL_MAT_ID` settles it in minutes. |
| 15 | **FA tile GCN config** (occupancy 1 or smaller tile for D ≥ 128) + **`v_mad_mix_f32` MAC** for gfx900 in `ggml_cuda_mad(float&, half2, half2)` with `-fgpu-flush-denormals-to-zero` | FA-ON prefill 4K: 41.5 → ~85-95 (parity with FA OFF); **decode 4K 14.7 → ~17-18** via the GQA-fused tile decode and unlocking `-ctv q8_0` | This is the one that targets the context falloff (mechanism f). Medium effort; illegal-instruction risk if the mad-mix probe is wrong — check `llvm-mc` first. |
| 16 | mmvf: fold the GQA ratio into `ncols_dst` (Vulkan's p021 trick) | decode 4K +15-20 % with `-fa 0` | Alternative to #15 for the same falloff. Larger patch, strided-dst correctness cases. First run `llama-bench -fa 0,1 -d 128,4096` — if FA0 == FA1 at 4096 the mechanism is refuted. |

### Tier 5 — algorithmic, workload-dependent

| # | Change | Expected | Constraint |
|---|---|---|---|
| 17 | **Draft-free n-gram speculation** (`--spec-type ngram-mod --spec-ngram-mod-n-max 3 --spec-ngram-mod-n-min 1`) | decode +10-40 % on code/edits/JSON, ~0 on prose | Verify batch **must stay ≤ 4 tokens** on the 35B: `get_mmvq_mmid_max_batch_gcn` = 4 for Q4_K/Q6_K, above that experts drop to MMQ. The defaults (n_max 64) would turn every verify into a prefill. 35B also pays a ~60 MiB recurrent-state checkpoint per drafted step. |
| 18 | **MTP-head speculation** (`--spec-type draft-mtp --spec-draft-n-max 3`) on the MTP GGUF | decode +25-50 % at α ≥ 0.6 | Needs the `-MTP` GGUF (unsloth Qwen3.6-35B-A3B-MTP is partly downloaded — but that is Qwen3.6, so no comparison to the tables). Same ≤ 4 cap. |
| 19 | Context-checkpoint density for chat (`-ctxcp 32 -cms 512`) | TTFT on branch/edit from ~40 s to ~5 s | No throughput change; the 30 gated-DeltaNet layers make prefix reuse checkpoint-only. |

---

## 4. Closed — do not spend runs on these

| Item | Why |
|---|---|
| `HSA_ENABLE_SDMA` / `HIP_FORCE_DEV_KERNARG` / `GPU_FORCE_64BIT_PTR` | The only recurring copy is ~1 MB logits/token; blit vs SDMA is < 0.2 % of wall time. SDMA=1 is documented unstable anyway. |
| Native `--offload-arch=gfx90c`, LTO, early-inline flags, `-ffast-math` alone | LLVM emits an identical feature set for gfx900 and gfx90c; native target would break the rocBLAS backport lookup and the `HSA_OVERRIDE`. |
| `GGML_CUDA_FORCE_MMQ` on the 35B | Experts are already MMQ on Vega by the explicit rule; the flag only moves *dense* GEMMs to emulated dp4a. June "wash" is explained. On gemma it would likely regress. |
| MoE experts on CPU (`-ncmoe`) | ~80 D2H/H2D boundaries per token; expected decode −10..−20 %. The README TODO's premise ("CPU prefill 233 beats GPU") was itself a mis-measurement (`-ngl 0` offloads). |
| BIOS carve-out for ROCm, +200 MHz iGPU boost, IOMMU | Measured 2026-09-07: no effect (carve-out is bypassed by `apu_prefer_gtt`; boost changed the DPM table but not the observed 2400 MHz). |
| `numactl`, THP, governor, `amdgpu.vm_fragment_size` | Single NUMA node; THP does not apply to TTM pages; no measurable lever identified. |
| "64 GB GTT costs 15-20 % decode" (ARCHITECTURE.md) | The claim's sole support is one old gemma table contradicted by adjacent rows. It now has a *real mechanism* (#6: snooped GTT vs carve-out) but no measurement. Reword as a hypothesis until #6 is run. |

---

## 5. Experiment plan — one variable per step

Each step uses `llama-bench` for clean numbers; the harness only for the final
re-baseline.

```bash
ENV='ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 HSA_OVERRIDE_GFX_VERSION=9.0.0 HSA_ENABLE_SDMA=0 HSA_XNACK=0 GPU_MAX_ALLOC_PERCENT=100'
B=llm/rocm7-vega/bin/llama-bench
```

1. **Baseline with proper methodology** — establishes what everything below is
   measured against:
   `env $ENV $B -m $M35 -ngl 99 -fa 0 -p 512,1024,4096 -n 64 -d 0,4096 -r 3`
2. **`-ub`** (Tier 1 #1): same + `-b 4096 -ub 2048` then `-ub 4096`. Decides the
   new default. Expect +18-22 % at 4K.
3. **Decode-at-depth split** (decides #15 vs #16):
   `env $ENV $B -m $M35 -ngl 99 -fa 0,1 -p 0 -n 64 -d 128,1024,4096 -r 3`
   If FA0 ≈ FA1 at depth 4096, the mmvf GQA mechanism is refuted and the falloff is
   in the shared softmax/dispatch cost.
4. **Clocks** (#5): repeat step 2's best with `perf_level=high`, then COMPUTE profile.
5. **KV quant** (#4): best-so-far + `-ctk q8_0`.
6. **Tensile trace** (#8): one `-p 1024 -n 0` run under `AMD_LOG_LEVEL=4`, histogram
   kernel names. Then decide `GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32` (#7).
7. **`GGML_HIP_GRAPHS=ON` build** (#11) into `llm/rocm7-vega-graphs`, step 3's
   decode command. This is the only cheap shot at the context-*independent* gap.
8. **`gttsize=16384` boot** (#6) — gemma only: `env $ENV $B -m $MG -ngl 99 -fa 0
   -p 1024 -n 64 -d 0,4096 -r 3` before and after the reboot. Confirm the KFD bank
   flipped before trusting the number.
9. **GCN MMQ config patch** (#14): `test-backend-ops perf -o MUL_MAT_ID -b ROCm0`
   before/after, then step 2's command.
10. **FA tile GCN config + mad-mix** (#15): `test-backend-ops perf -o FLASH_ATTN_EXT
    -b ROCm0 -p 'hsk=256'`, then step 3. This is the largest single decode lever
    identified and the only one that makes `-fa 1` usable on ROCm.
11. **Speculation** (#17): server with `--spec-type ngram-mod` on a real code prompt,
    300 generated tokens, compare `predicted_per_second`; confirm greedy output is
    token-identical with and without.
12. Re-run `bench/run-all-benchmarks.sh` (now with warmup, `cache_prompt:false`,
    `-ub 2048`, `-r`) to re-baseline the tables.

Steps 1-6 need no rebuild and fit in an evening. Steps 7-10 are one rebuild each into
a separate prefix. If steps 2, 4, 5 land as expected, ROCm prefill at 4K moves from
~89 to ~110-115 t/s — still ~25 % behind Vulkan, which is the emulated-dp4a floor
(mechanism a) that only #13/#14 can move. The decode gap is the more tractable one:
#11 + #15 target both of its components with identified mechanisms.

---

## 6. What this audit could not establish

- The *magnitude* of snooped-GTT vs carve-out on this GPU (#6 is the measurement).
- Whether launch overhead or kernel time dominates the context-independent decode gap
  (#11's A/B, or a `rocprofv3 --kernel-trace` on one token — no profiler run exists
  in the repo).
- Which Tensile solutions actually execute (#8).
- Whether the RDNA2 MMQ spills cost anything measurable (#12 then #14).
- What caused ROCm's +22 % on gemma at the BIOS retune. It was not the carve-out.
  Of the remaining four changes, Curve Optimizer −15 (more sustained boost under the
  shared PPT) is the only one with a plausible mechanism; an A/B at −10 would settle it.
