# LLM Inference Benchmarks — Vega 8 iGPU

Compact benchmark log for llama.cpp on AMD Ryzen 7 5700G / Radeon Vega 8. Latest run is first; older results are kept where they explain behaviour changes.

## Re-verification after the 26.04 reinstall — 2026-09-07

Same model and same harness as the 2026-05-15 gemma rows, so these are directly
comparable. Ubuntu 26.04.1 / kernel 7.0 / classic ROCm 7.2.0, Vega 8 as the only GPU
(`card0`, ROCm index 0), BIOS carve-out 2 GB, **64 GB GTT** (the May rows were recorded
at 32 GB GTT). Thermals logged throughout with `bench/log-thermals.sh`.

The numbers below are from the **post-cooling-fix run** — fan speed was raised after an
identical earlier sweep showed the chip throttling (see Thermals). That earlier sweep
measured 6–8 % lower prefill across every GPU backend, so treat any figure recorded
while this rig is throttling as a floor, not a result.

### gemma-4-E4B-it-Q4_K_M (`-c 8192`), prefill / decode t/s at ~128 / ~1024 / ~4096

| Backend | FA | Prefill | Decode | vs. 2026-05-15 |
| ------- | -- | ------- | ------ | -------------- |
| **Vulkan GPU** | ON | **124.23 / 170.73 / 171.29** | **17.55 / 17.16 / 16.36** | reproduces, slightly ahead (May: 121.97 / 166.81 / 160.02) |
| Vulkan GPU | OFF | 95.60 / 149.48 / 155.30 | 17.19 / 16.39 / 14.29 | — |
| **ROCm 7.2 baremetal** | OFF | **56.46 / 89.10 / 87.16** | **14.28 / 12.91 / 10.63** | reproduces at 1K/4K (May: 70.91 / 84.88 / 84.95); ~128 stays low |
| ROCm 7.2 baremetal | ON | 55.23 / 47.36 / 27.11 | 14.32 / 13.31 / 11.44 | FA ON collapses prefill — as documented |
| CPU (`-dev none`) | ON | 95.38 / 94.66 / 88.37 | 15.05 / 14.24 / 12.05 | decode reproduces; **prefill does not** (May: 249.57 / 754.82 / 840.50) |
| CPU (`-dev none`) | OFF | 94.88 / 93.23 / 84.69 | 14.95 / 14.39 / 13.44 | — |

**Vulkan and ROCm both reproduce**, which is the main result: the restored stack performs
as it did before the reinstall. Vulkan remains the best path — use `-fa 1` for Vulkan and
`-fa 0` for ROCm.

### The CPU rows were measuring the GPU

`start_cpu()` used `-ngl 0`, which no longer keeps the model off the GPU — upstream
changed the `-ngl` default to `auto`. Measured directly on this build:

| Flag | GPU busy | GTT used |
| ---- | -------- | -------- |
| `-ngl 0` | 91 % | 6567 MB |
| `-dev none` | idle | 157 MB |

So every `-ngl 0` "CPU" row on current llama.cpp is a GPU run with a CPU label — which is
why the first attempt landed near the Vulkan figure instead of anywhere near the
historical CPU numbers. Fixed to `-dev none`; the CPU rows above are genuine (GTT 157 MB,
iGPU idle while the CPU is under full load).

### Unresolved: CPU prefill is ~10× below the May figure

With real CPU-only execution and no throttling, prefill measures **85–95 t/s**, stable
across 8 and 16 threads, cold and warm start, harness and direct `llama-bench`. May
recorded 840 t/s at 4096. Decode reproduces fine (12.05 vs 12.17), so this is specific to
prefill, and raising cooling did not close it.

Not explained. Two observations, neither conclusive:

- The May shape (249 → 755 → 840, rising 3.4× with context) is not how prefill behaves;
  throughput normally flattens or falls as context grows, which is what the current
  numbers do (95 → 95 → 88).
- The model is **7.52 B** parameters total. Sustaining 840 tok/s prefill implies roughly
  12 TFLOP/s; a 5700G's 8 Zen 3 cores on AVX2 are an order of magnitude below that. The
  measured ~90 t/s corresponds to ~1.3 TFLOP/s, which is the right order for this chip.

That points at the May figure being an artefact rather than a regression, but the binary
and config that produced it were wiped by the reinstall, so it cannot be checked. Treat
the historical CPU prefill rows as suspect until someone reproduces them.

### Thermals — before and after raising fan speed

Two identical 6-backend sweeps on the same rig, the second after the fan curve was raised:

| | Before | After |
| --- | ------ | ----- |
| Idle | ~50 °C | **40–41 °C** |
| Peak under load | **105.4 °C** | **89.5 °C** |
| Average under load | 90.5 °C | **79.7 °C** |
| Samples ≥ 95 °C (Tjmax) | 27 | **0** |
| Samples ≥ 100 °C | 22 | **0** |
| iGPU SCLK under load | 2400 → 2208 MHz | **2400 → 2351 MHz** |
| CPU clocks | 4000 → 3450 MHz | no downward drift |

The chip no longer exceeds Tjmax and no longer throttles. This also recovered real
throughput — prefill rose ~6 % on Vulkan, ~8 % on ROCm and ~7 % on CPU purely from
running cooler, which is why the pre-fix numbers are not published above.

> The CPU phase is not a strict A/B between the two sweeps: the earlier one still had the
> `-ngl 0` bug, so its "CPU" rows ran on the GPU. The Vulkan and ROCm phases are identical
> workloads in both, and carry the comparison.

> Peak is still 89.5 °C, which is warm for a 65 W APU. A repaste is worth doing; the fan
> change fixed the throttling but the headroom to Tjmax is only ~5 °C.

---

## Current Status — 2026-05-15

**Latest run:** `bench/run-all-benchmarks.sh`, completed `12 / 12` backend × model combinations at `2026-05-15 00:26`.

> **Environment changes since this run (September 2026):** two rounds of changes.
> *June 2026* — the RTX 5090 was removed and a second R9700 added (Vega 8 became
> `renderD130` / ROCm index 2), and classic ROCm 7.2 was replaced by modular
> `amdrocm-core` 7.13/7.14, which broke the baremetal rows on that host.
> *September 2026* — both R9700s moved to another machine and the OS was reinstalled
> as Ubuntu 26.04 / kernel 7.0 with classic ROCm 7.2.0 restored. The Vega 8 is now the
> **only** GPU: `card0` / `/dev/dri/renderD128` / ROCm index **0**, and the baremetal
> path works again (re-verified 2026-09-07).
>
> The rows below therefore record the May 2026 layout and are **not** directly
> reproducible today: different OS, different kernel, different device indices, and a
> benchmark runner since changed to auto-detect the Vega index and use the
> documented-safe HSA env (`HSA_XNACK=0`, `HSA_ENABLE_SDMA=0` — the run below used
> `XNACK=1`/`SDMA=1`). Treat them as historical baselines, not as current numbers.

### Current benchmark parameters

| Area                         | Current setting                                                                                                      |
| ---------------------------- | -------------------------------------------------------------------------------------------------------------------- |
| BIOS UMA / iGPU VRAM         | **2 GB**                                                                                                             |
| OS / kernel                  | Ubuntu 25.10, kernel 6.17 *(run-time; host is now Ubuntu 26.04 / kernel 7.0)*                                        |
| CPU                          | AMD Ryzen 7 5700G, 8C/16T, Zen 3, AVX2/FMA — slight undervolt: Curve Optimizer all-core offset −10                   |
| RAM                          | 64 GB DDR4 — 2× 32 GB Kingston Fury 3600 MT/s overclocked to **4200 MT/s** (UMA-shared with the iGPU; decode is memory-bandwidth-bound, so slower RAM ⇒ proportionally slower decode) |
| Motherboard                  | ASRock Fatal1ty B450 Gaming-ITX/ac                                                                                   |
| Target GPU                   | Radeon Vega 8 iGPU, gfx90c/gfx900-compatible (`/dev/dri/renderD129` at run time; `renderD130` June 2026; **`renderD128` now**) |
| Other GPUs                   | RX 9700 AI Pro present at run time but not used by benchmark scripts (June 2026: 2× R9700, RTX 5090 removed; **none present since September 2026**) |
| Context                      | `-c 8192`                                                                                                            |
| Prompt sizes                 | `128`, `1024`, `4096` requested; effective prompts about `140/141`, `937`, `3330` tokens                             |
| Decode length                | `50` generated tokens                                                                                                |
| Warmup                       | `--no-warmup`                                                                                                        |
| GPU offload                  | `-ngl 99` for ROCm/Vulkan, `-ngl 0` for CPU                                                                          |
| Vulkan device                | `-dev Vulkan0` to pin RADV RENOIR / Vega 8                                                                           |
| ROCm 7 baremetal binary      | `llm/rocm7-vega/bin/llama-server`                                                                                    |
| ROCm 7 build flags           | `GGML_HIP_GRAPHS=OFF`, `GGML_BACKEND_DL=ON`, `GGML_CPU_ALL_VARIANTS=ON`, `GPU_TARGETS=gfx900`                        |
| ROCm 7 env, device           | `ROCR_VISIBLE_DEVICES=1` (Vega 8 index at run time; now 2), `HIP_VISIBLE_DEVICES=0`, `HSA_OVERRIDE_GFX_VERSION=9.0.0` |
| ROCm 7 env, memory/runtime   | run used `HSA_ENABLE_SDMA=1`, `HSA_XNACK=1`; **now `=0`/`=0`** (XNACK=1 can freeze the PC), `GPU_MAX_ALLOC_PERCENT=100` |
| Docker ROCm env              | Docker sees only Vega 8 render node, so `ROCR_VISIBLE_DEVICES=0`, `HIP_VISIBLE_DEVICES=0`                            |
| Result directory             | `/tmp/bench-results-20260515-001400/`                                                                                |

**Kernel/ROCm stability note:** do **not** force `amdgpu.cwsr_enable=0` for this setup. With that parameter set, the large Qwen model did not load and crashed during loading. Leave CWSR at the driver default unless re-testing explicitly.

**Large-model memory note:** Qwen 35B Q4 needs the large UMA/GTT path despite the 2 GB BIOS frame buffer. The 64 GB GTT workaround remains the known path for full offload of ~20 GB models: `amdgpu.gttsize=65536 ttm.pages_limit=16777216`. For small models that fit in the BIOS carveout, previous testing showed smaller GTT/BIOS allocation can improve throughput.

---

## ROCm 7.2 / Vega 8 tuning sweep (2026-06)

Goal: find any runtime/build tweak that improves the **working ROCm 7.2 Docker** path
on the Vega 8. Background and rationale in
[ARCHITECTURE.md — Performance ceiling and tuning levers](ARCHITECTURE.md#performance-ceiling-and-tuning-levers-gfx900--vega-8):
gfx900 has **no hardware dp4a** (MMQ runs emulated) and decode is **DDR4-bandwidth-bound**,
so expectations are modest — prefill batch tuning is the best bet; Vulkan still wins decode.

**Fixed conditions:** model `Qwen3.5-35B-A3B-Q4_K_M` (20 GB, full offload), ROCm 7.2 Docker
image `llama-rocm7-vega`, `-ngl 99 -fa 0 --no-warmup`, prompts ~140 / 937 / 3330 tokens,
50 decode tokens. Each config = fresh container. Harness: `bench/tune-rocm7-vega.sh`.

**Configs tested:**
1. Baseline — `-c 8192` (current `run-docker-rocm7.sh` defaults)
2. ubatch sweep — `-b 2048 -ub {256, 1024, 2048}` (default ub is 512)
3. K-cache quant — baseline + `-ctk q8_0`
4. GPU clocks — baseline with host `rocm-smi --setperflevel high` (Vega 8 = card3)
5. (conditional) build `-DGGML_CUDA_FORCE_MMQ=ON` and A/B vs baseline

> Results table populated by the sweep run — see below.

<!-- TUNING_RESULTS -->
> **First attempt (aborted):** the initial sweep hard-froze the whole PC within ~3 s
> of loading the 35 B — because a fresh Ubuntu reinstall had left GRUB without the
> 64 GB-GTT params (`amdgpu.gttsize=65536 ttm.pages_limit=16777216`), so the Vega 8
> had only ~30 GB GTT and the 20 GB allocation overflowed it. After restoring the
> params + reboot (Vega 8 → 64 GB GTT), the 35 B loads to ~21 GB and the sweep ran
> clean. **These params are mandatory for large models on ROCm.**

**Results (2026-06-13, 35B-A3B-Q4_K_M, ROCm 7.2 Docker, `-ngl 99 -fa 0 -c 8192`):**

Prefill (t/s):

| Config | ~140 tok | ~937 tok | ~3330 tok |
| --- | --- | --- | --- |
| baseline (`-ub 512`) | 41.0 | 69.8 | 68.9 |
| `-ub 256` | 41.0 | 54.4 | 54.3 |
| `-ub 1024` | 41.1 | 81.1 | 78.6 |
| **`-ub 2048`** | 41.0 | 80.6 | **84.0** |
| `-ctk q8_0` | 41.2 | 69.1 | 67.9 |

Decode (t/s):

| Config | ~140 tok | ~937 tok | ~3330 tok |
| --- | --- | --- | --- |
| baseline | 16.0 | 15.2 | 12.6 |
| `-ub 2048` | 16.0 | 15.2 | 12.6 |
| **`-ctk q8_0`** | 15.7 | 15.3 | **13.0** |

**Findings:**
- **`-ub 2048` (full-batch prefill) is a clean win: +22 % prefill at 4 K ctx
  (84 vs 69 t/s), +15 % at 1 K, no decode penalty.** Now the default in
  `run/run-docker-rocm7.sh`. Smaller `-ub 256` *hurts* (under-fills the 8-CU GEMMs).
- **`-ctk q8_0`** gives a small decode bump at large context (13.0 vs 12.6 t/s,
  +3.5 %) and halves K-cache memory — worth adding for long-context decode.
  (`-ctv` needs flash attention, which loses on Vega, so K-only with `-fa 0`.)
- Decode is otherwise **flat across all configs** — confirming it's DDR4-bandwidth-
  bound, not batch-bound, exactly as the ceiling analysis predicted. The decode win
  remains on Vulkan (~19–20 t/s).
**GPU clocks — `power_dpm_force_performance_level=high` (35B, `-ub 2048`):**

| metric | auto | high | Δ |
| --- | --- | --- | --- |
| prefill @937 | 80.6 | 82.8 | +2.7% |
| prefill @3330 | 84.0 | 86.9 | +3.4% |
| decode @937 | 15.2 | 15.7 | +3.3% |
| decode @3330 | 12.6 | 12.7 | +0.8% |

Pinning clocks (GPU hit 2400 MHz under load) is a **real but small win (~+3% prefill)**.
Costs: needs root, not persistent across reboots, and draws more power continuously
on a shared-TDP APU. Optional, not a default.

**`-DGGML_CUDA_FORCE_MMQ=ON` (rebuilt in-image, A/B at high clocks + `-ub 2048`):**

| metric | default (cuBLAS dispatch) | FORCE_MMQ | Δ |
| --- | --- | --- | --- |
| prefill @3330 | 86.9 | 86.9 | ~0% |
| prefill @937 | 82.8 | 83.1 | +0.4% |
| decode @3330 | 12.7 | 13.0 | +1.8% (noise) |

**A wash** — neither the prefill regression expected from emulated dp4a nor any real
gain. Not adopted; the default MMQ/cuBLAS auto-dispatch is fine on gfx900.

**Net conclusion:** the one keeper is **`-ub 2048`** (~+22% prefill at 4K, now the
default in `run/run-docker-rocm7.sh`). `-ctk q8_0` is a minor opt-in for long-context
decode. Clock-pinning and FORCE_MMQ are not worth the cost/complexity. Decode stays
bandwidth-bound (~13 t/s on ROCm vs ~19–20 on Vulkan), as the ceiling analysis predicted.

---

## How to run

```bash
./bench/run-all-benchmarks.sh 2>&1 | tee /tmp/bench-$(date +%Y%m%d-%H%M).log
```

The runner starts each backend sequentially, waits for `/health`, runs `bench/test-server-perf.py`, writes per-backend CSVs, then prints model-grouped summary tables.

### Enabled backends in the latest run

| Backend label                 | Server path                      | Key flags                              |
| ----------------------------- | -------------------------------- | -------------------------------------- |
| `ROCm-7.2-Baremetal-FA-OFF`   | ROCm 7 baremetal                 | `-ngl 99 -fa 0 -c 8192`                |
| `ROCm-7.2-Baremetal-FA-ON`    | ROCm 7 baremetal                 | `-ngl 99 -fa 1 -c 8192`                |
| `Vulkan-GPU-FA-OFF`           | Native Vulkan                    | `-ngl 99 -dev Vulkan0 -fa 0 -c 8192`   |
| `Vulkan-GPU-FA-ON`            | Native Vulkan                    | `-ngl 99 -dev Vulkan0 -fa 1 -c 8192`   |
| `CPU-FA-ON`                   | Native Vulkan binary, CPU mode   | `-ngl 0 -fa 1 -c 8192`                 |
| `CPU-FA-OFF`                  | Native Vulkan binary, CPU mode   | `-ngl 0 -fa 0 -c 8192`                 |

Docker ROCm 6/7 entries remain in the script but were disabled for the 2026-05-15 run.

---

## Latest Results — 2026-05-15

### Qwen3.5-35B-A3B-Q4_K_M

Model size is ~20 GB. Full GPU offload requires the large GTT/UMA path.

#### Prefill — tokens/s, higher is better

| Backend                | FA    | ~128 tok | ~1024 tok | ~4096 tok | vs previous comparable run                                      |
| ---------------------- | ----- | -------: | --------: | --------: | ---------------------------------------------------------------- |
| CPU                    | OFF   |    59.33 |    186.34 |    208.17 | Prefill down vs 2026-05-14 CPU; decode improved at 1K            |
| CPU                    | ON    |    58.18 |    196.63 |    210.79 | Still best prefill overall; smaller gap than before              |
| ROCm 7.2 baremetal     | OFF ✅ |    42.47 |     72.61 |     71.65 | Faster than previous baremetal FA-OFF at all contexts            |
| ROCm 7.2 baremetal     | ON ⚠  |    40.48 |     55.79 |     37.55 | FA still hurts large-context prefill                             |
| Vulkan GPU             | OFF   |    64.73 |    138.08 |    136.44 | Huge prefill jump vs older Vulkan baseline                       |
| Vulkan GPU             | ON ✅  |    65.00 |    138.57 |    137.11 | Best GPU prefill, FA neutral/slightly positive                   |

#### Decode — tokens/s, higher is better

| Backend                | FA    | ~128 tok | ~1024 tok | ~4096 tok | vs previous comparable run                                      |
| ---------------------- | ----- | -------: | --------: | --------: | ---------------------------------------------------------------- |
| CPU                    | OFF   |    17.01 |     16.77 |     15.94 | Improved vs previous CPU OFF at all contexts                     |
| CPU                    | ON    |    17.04 |     16.65 |     13.59 | Improved short/1K, 4K still lower than OFF                       |
| ROCm 7.2 baremetal     | OFF ✅ |    16.67 |     15.85 |     13.03 | Improved vs previous baremetal FA-OFF                            |
| ROCm 7.2 baremetal     | ON    |    16.53 |     15.50 |     13.17 | Decode similar to FA-OFF                                         |
| Vulkan GPU             | OFF   |    18.88 |     18.49 |     16.35 | Strong but FA-ON is better                                       |
| Vulkan GPU             | ON ✅  |    19.06 |     18.95 |     18.47 | Best decode overall                                              |

**Qwen takeaway:** Vulkan is now the best GPU path overall for Qwen, especially decode. ROCm 7.2 baremetal FA-OFF is improved and stable, but still behind Vulkan. CPU remains strongest for bulk prefill, while Vulkan gives the best interactive generation.

### gemma-4-E4B-it-Q4_K_M

Smaller model; all GPU backends fully offload.

#### Prefill — tokens/s, higher is better

| Backend                | FA    | ~128 tok | ~1024 tok | ~4096 tok | vs previous comparable run                                      |
| ---------------------- | ----- | -------: | --------: | --------: | ---------------------------------------------------------------- |
| CPU                    | OFF ⚠️  |   235.17 |    693.91 |    772.09 | ⚠️ Suspect — see below                                            |
| CPU                    | ON ⚠️  |   249.57 |    754.82 |    840.50 | ⚠️ **Did not reproduce 2026-09-07** (measured 95 / 95 / 88 with verified CPU-only execution, no throttling). Both CPU prefill rows here are suspect; decode reproduced fine |
| ROCm 7.2 baremetal     | OFF ✅ |    70.91 |     84.88 |     84.95 | Close to previous; still recommended ROCm mode                   |
| ROCm 7.2 baremetal     | ON ⚠  |    65.77 |     47.70 |     27.67 | FA still severely hurts ROCm prefill                             |
| Vulkan GPU             | OFF   |    91.11 |    143.58 |    147.37 | Strong GPU path                                                  |
| Vulkan GPU             | ON ✅  |   121.97 |    166.81 |    160.02 | Best GPU prefill; FA helps Vulkan                                |

#### Decode — tokens/s, higher is better

| Backend                | FA    | ~128 tok | ~1024 tok | ~4096 tok | vs previous comparable run                                      |
| ---------------------- | ----- | -------: | --------: | --------: | ---------------------------------------------------------------- |
| CPU                    | OFF   |    14.83 |     14.41 |     13.36 | Improved vs previous CPU OFF                                     |
| CPU                    | ON    |    15.06 |     14.24 |     12.17 | Short-context better; 4K lower than OFF                          |
| ROCm 7.2 baremetal     | OFF ✅ |    14.50 |     13.15 |     10.89 | Slightly lower than previous best baremetal decode               |
| ROCm 7.2 baremetal     | ON    |    14.69 |     13.66 |     11.68 | Decode a little higher, but prefill penalty is too large         |
| Vulkan GPU             | OFF   |    16.64 |     15.68 |     13.62 | Good baseline                                                    |
| Vulkan GPU             | ON ✅  |    16.97 |     16.56 |     15.84 | Best decode overall                                              |

**Gemma takeaway:** CPU FA-ON dominates prefill. Vulkan FA-ON is the best GPU/interactive setting. ROCm FA-OFF remains the only sensible ROCm setting.

---

## Best Settings

| Use case                | Qwen 35B recommendation                    | Gemma 4B recommendation                    | Why                                               |
| ----------------------- | ------------------------------------------ | ------------------------------------------ | ------------------------------------------------- |
| Best interactive chat   | Vulkan GPU `-ngl 99 -dev Vulkan0 -fa 1`    | Vulkan GPU `-ngl 99 -dev Vulkan0 -fa 1`    | Best decode and strong prefill                    |
| Best GPU prefill        | Vulkan GPU `-fa 1`                         | Vulkan GPU `-fa 1`                         | Latest Vulkan prefill beats ROCm by a large margin |
| Best CPU prefill        | CPU `-ngl 0 -fa 1`                         | CPU `-ngl 0 -fa 1`                         | AVX2 CPU path scales very well with large prompts |
| Best ROCm               | ROCm 7.2 baremetal `-ngl 99 -fa 0`         | ROCm 7.2 baremetal `-ngl 99 -fa 0`         | FA-ON hurts ROCm gfx900 prefill                   |
| Large models > BIOS VRAM | Use large GTT/UMA path                    | Usually not needed                         | Qwen needs memory beyond 2 GB BIOS carveout       |

---

## Merged Historical Results

These tables preserve important previous runs without repeating every per-backend section.

### Qwen3.5-35B-A3B-Q4_K_M — selected history

| Date       | Memory / setup                               | Backend              | FA      | Prefill ~128 / ~1024 / ~4096 | Decode ~128 / ~1024 / ~4096 | Note                                      |
| ---------- | -------------------------------------------- | -------------------- | ------- | ----------------------------- | ---------------------------- | ----------------------------------------- |
| 2026-05-15 | BIOS 2 GB;32 GB GTT, current ROCm tweaks     | Vulkan GPU           | ON      | 65.00 / 138.57 / 137.11       | 19.06 / 18.95 / 18.47        | Latest best GPU path                      |
| 2026-05-15 | BIOS 2 GB;32 GB GTT, current ROCm tweaks     |  ROCm 7.2 baremetal   | OFF     | 42.47 / 72.61 / 71.65         | 16.67 / 15.85 / 13.03        | Latest recommended ROCm                   |
| 2026-05-15 | BIOS 2 GB;32 GB GTT, current ROCm tweaks     |  CPU                  | ON      | 58.18 / 196.63 / 210.79       | 17.04 / 16.65 / 13.59        | Latest best CPU prefill                   |
| 2026-05-14 | 64 GB GTT, `HSA_ENABLE_SDMA=1`, `HSA_XNACK=1`| ROCm 7.2 baremetal   | OFF     | 40.55 / 68.18 / 67.32         | 14.50 / 14.30 / 11.88        | Previous baremetal baseline               |
| 2026-05-14 | BIOS 2 GB; 64 GB GTT                         | ROCm 7.2 Docker      | OFF     | 38.63 / 70.41 / 68.87         | 15.49 / 15.06 / 12.43        | Docker near baremetal                     |
| 2026-05-14 | BIOS 2 GB; 64 GB GTT                         | ROCm 6.2.4 Docker    | OFF     | 40.15 / 64.43 / 63.97         | 14.40 / 13.82 / 11.64        | ROCm 6 stable via Docker                  |
| 2026-05-14 | BIOS 2 GB; 64 GB GTT                         | CPU                  | ON      | 57.07 / 215.04 / 233.12       | 15.67 / 15.39 / 12.95        | Earlier stronger CPU prefill              |
| 2026-04-12 | BIOS 2 GB; 64 GB GTT                         | Vulkan GPU           | default | 44.58 / 50.62 / 50.00         | 19.90 / 20.46 / 19.76        | Older Vulkan prefill much lower, decode excellent |
| 2026-04-12 | BIOS 2 GB; 64 GB GTT                         | LM Studio Vulkan     | UI      | 48.84 / 137.85 / 157.60       | 19.58 / 19.05 / 18.05        | LM Studio prefill not directly comparable |

### gemma-4-E4B-it-Q4_K_M — selected history

| Date       | Memory / setup                         | Backend              | FA  | Prefill ~128 / ~1024 / ~4096 | Decode ~128 / ~1024 / ~4096 | Note                                  |
| ---------- | -------------------------------------- | -------------------- | --- | ----------------------------- | ---------------------------- | ------------------------------------- |
| 2026-05-15 | BIOS 2GB;32GB GTT,current ROCm tweaks  | Vulkan GPU           | ON  | 121.97 / 166.81 / 160.02      | 16.97 / 16.56 / 15.84        | Latest best GPU path                  |
| 2026-05-15 | BIOS 2GB;32GB GTT,current ROCm tweaks  | ROCm 7.2 baremetal   | OFF | 70.91 / 84.88 / 84.95         | 14.50 / 13.15 / 10.89        | Latest recommended ROCm               |
| 2026-05-15 | BIOS 2GB;32GB GTT,current ROCm tweaks  | CPU                  | ON  | 249.57 / 754.82 / 840.50 ⚠️     | 15.06 / 14.24 / 12.17        | ⚠️ **Prefill suspect** — did not reproduce 2026-09-07 (measured 95 / 95 / 88 with verified CPU-only execution and no thermal throttling); decode did reproduce. See the 2026-09-07 section |
| 2026-05-14 | 16 GB BIOS carveout, no 64 GB GTT      | Vulkan GPU           | ON  | 115.76 / 158.83 / 156.18      | 14.73 / 14.55 / 14.03        | Earlier best GPU; latest is faster decode |
| 2026-05-14 | 16 GB BIOS carveout, no 64 GB GTT      | ROCm 7.2 baremetal   | OFF | 67.96 / 81.96 / 81.77         | 13.32 / 12.19 / 10.00        | Prior baremetal baseline              |
| 2026-05-14 | 16 GB BIOS carveout, no 64 GB GTT      | CPU                  | ON  | 257.48 / 810.80 / 923.35      | 12.33 / 11.90 / 10.10        | Highest recorded Gemma prefill        |
| 2026-05-14 | BIOS 2GB; 64 GB GTT historical         | ROCm 7.2 Docker      | OFF | 67.56 / 80.73 / 80.74         | 13.13 / 11.94 / 9.77         | Docker near baremetal                 |
| 2026-05-14 | BIOS 2GB; 64 GB GTT historical         | ROCm 6.2.4 Docker    | OFF | 66.97 / 80.33 / 79.98         | 10.58 / 9.74 / 8.32          | ROCm 6 slower decode                  |

---

## Important Findings

- **Vulkan is the current best GPU backend.** On 2026-05-15 it wins decode for both models and now also wins GPU prefill.
- **ROCm on Vega 8 should use `-fa 0`.** Flash attention consistently damages ROCm gfx900/gfx90c prefill, especially at 1K–4K context. Decode changes are small and do not justify FA-ON.
- **Vulkan should use `-fa 1`.** FA-ON is neutral-to-positive for Qwen and clearly positive for Gemma.
- **CPU should usually use `-fa 1` for prefill.** CPU decode can vary by context, but FA-ON remains the best bulk prompt-processing option.
- **ROCm 7.2 baremetal now works and is stable with the current launch env.** Latest results are slightly better than the previous baremetal run for Qwen and close for Gemma.
- **Docker ROCm remains useful as a reproducible fallback.** Historical ROCm 7 Docker numbers were near baremetal; ROCm 6 Docker is stable but slower, especially decode.
- **`amdgpu.cwsr_enable=0` is a bad setting for this large-model setup.** It caused Qwen 35B loading failure/crash and should stay out of the default benchmark profile.
- **Large GTT is model-size dependent.** Use 64 GB GTT for Qwen 35B full offload. For smaller models that fit in BIOS carveout, previous results showed less GTT overhead and better throughput.

---

## Archived detail notes

### ROCm 7.2 / gfx900 backport

ROCm 7.2 on Vega 8 depends on gfx900-compatible libraries and the lazy rocBLAS index. The critical runtime file is `TensileLibrary_lazy_gfx900.dat`; without it, rocBLAS can fail on first GEMM with `Illegal seek for GPU arch: gfx900`. The project build copies the required `*gfx900*` files and lazy index into the ROCm 7 layer/install.
