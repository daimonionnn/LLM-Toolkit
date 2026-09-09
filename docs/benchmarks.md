# LLM Inference Benchmarks — Vega 8 iGPU

Compact benchmark log for llama.cpp on AMD Ryzen 7 5700G / Radeon Vega 8. Latest run is first; older results are kept where they explain behaviour changes.

## Flash attention fixed on gfx900 — the long-context collapse is gone — 2026-09-08

One instruction. `patches/0001-ggml-cuda-mad-gfx900-mad-mix.patch` makes the FA KQ
accumulate use `v_mad_mix_f32`, which gfx900 has and llama.cpp did not emit for it.

### The bug

`V_DOT2_F32_F16_AVAILABLE` (`common.cuh:760`) is defined only for RDNA2+, gfx906 and
CDNA. On GCN5 `ggml_cuda_mad(float&, half2, half2)` therefore falls back to
`__half22float2(v*u); acc += tmp.x + tmp.y` — `v_pk_mul_f16` + 2× `v_cvt_f32_f16` +
2× `v_add_f32`, **5 VALU ops per 2 MACs**, with the product formed in fp16.

Those intermediates also cost registers. Measured with
`-Rpass-analysis=kernel-resource-usage`: the FA tile kernels sat at the 128-VGPR cap,
**50 of 60 config rows spilling**, up to 2262 VGPRs and 2.9 KB/lane of scratch.

gfx900 has the VOP3P mad-mix family — f16 × f16 + f32 in one instruction, product in
fp32. Two ops with `op_sel` cover a half2: **1 VALU op per MAC**.

### The effect

| Metric | Before | After |
| ------ | -----: | ----: |
| VALU ops per MAC | 2.5 | **1** |
| Spills, 256×256 FA rows | 10 598 | **6** |
| VGPR use | 128 (capped) | 76–130 |
| Best occupancy reached | 2 | **3** |
| Error vs fp64 on 256 random pairs | 6.1e-3 | **1.4e-6** |

Decode t/s at KV depth, `llama-bench -n 32 -d -ub 512`:

| Model | Depth | `-fa 0` (was best) | `-fa 1` before | `-fa 1` **after** | vs. best before |
| ----- | ----- | -----------------: | -------------: | ----------------: | --------------: |
| Qwen 35B | 4 096 | 15.12 | 14.95 | **18.68** | +24 % |
| Qwen 35B | 32 768 | 6.16 | *timed out* | **15.86** | **+157 %** |
| gemma | 4 096 | 13.56 | 13.08 | **15.44** | +14 % |
| gemma | 32 768 | 7.61 | 5.97 | **12.49** | **+64 %** |

Prefill improves too, though `-fa 0` still wins there.

### What this changes

**`-fa 1` is now the right setting for ROCm decode.** Every table and launcher note in
this repo said the opposite, correctly, until this patch.

**The long-context collapse is gone.** ROCm decode fell 66 % from 1K to 32K; it now falls
about as much as Vulkan's 21 %. The gap to Vulkan at 32K drops from 178 % to **8 %**.

### The first attempt fixed the symptom

Before finding the instruction, the spill data pointed at the FA tile *occupancy*: the
config table is shared with CDNA, which has AGPRs to spill into and GCN5 does not, so
`occupancy = 2` caps every kernel at 128 VGPRs. Giving GCN `occupancy = 1` lifted the cap
to 256 and took 35B decode at 32K from 6.16 to 14.85 — a real improvement with a correctly
identified mechanism.

But the kernels were short of registers only because the fp16 fallback materialised
intermediates that need not exist. Removing them dropped VGPR use below the cap by itself.
Head to head at 32K on the 35B:

| | t/s |
| --- | ---: |
| occupancy only | 14.85 |
| **mad-mix only** | **15.86 ± 0.02** |
| both | 15.41 |

Occupancy 1 *on top of* mad-mix is a net loss — it pulls kernels that now reach occupancy
3 back down to 1. The occupancy patch was dropped. Had the instruction been found first,
it would never have been written; the audit listed it first and this work did it second,
because spill data from a diagnostic build was already in hand.

### Validation

- `test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0`: **2959/2959 passed**, on the shipped
  configuration and on both superseded variants.
- Numerics checked against an fp64 reference on the GPU before the patch was written.
- `-fa 0` numbers are unchanged, confirming nothing outside the FA path moved.
- Untested: FA shapes other than 256×256 and 512×512, and non-FA callers of
  `ggml_cuda_mad(float&, half2, half2)`.

---

## Cheap tuning items settled — 2026-09-08

### `-ctk q8_0`: small at 4K, large at 32K

Quantizing the K cache to q8_0 (V stays f16 — quantizing V requires flash attention,
which ROCm cannot use here). 35B, ROCm `-fa 0`, decode t/s by depth:

| Depth | `-ctk f16` | `-ctk q8_0` | Gain |
| ----- | ---------: | ----------: | ---: |
| 1 024 | 18.10 | 18.58 | +2.7 % |
| 4 096 | 15.09 | 16.11 | +6.8 % |
| 16 384 | 9.32 | 10.89 | +16.8 % |
| 32 768 | 6.16 | **7.61** | **+23.5 %** |

The June 2026 sweep measured this as "+3.5 %, small" and did not adopt it. That was
correct *at 4K* — the effect simply was not measured where it matters. **Adopt it for
long-context ROCm work**; at short context the gain is noise and costs K-cache precision.

It also decomposes the long-context decode collapse. Halving the K cache would nearly
double throughput if KV bandwidth were the whole bottleneck; it gives +23.5 %, so KV
bandwidth is roughly a quarter of the problem at 32K. The rest is the `mmvf` dispatch
structure — one block per KV row, GQA ratio not folded, so each K head is re-streamed by
all 8 of its Q heads — plus the V cache, which cannot be quantized without FA. Neither is
reachable with a flag; both need the flash-attention kernel fixed.

### Clock pinning and the COMPUTE power profile: no effect

35B, `-ub 4096`, `-p 3330 -n 32`:

| Setting | Prefill | Decode |
| ------- | ------: | -----: |
| `auto` (default) | 129.63 | 19.11 |
| `power_dpm_force_performance_level=high` | 130.96 | 19.01 |
| `pp_power_profile_mode=5` (COMPUTE) | 130.73 | 19.19 |

All within ±1 %, i.e. noise. June measured `high` as worth +3 % prefill and this run does
not reproduce that — **because the cooling fix removed the reason**. In June the GPU was
throttling, so pinning the top DPM state helped; now SCLK holds 2400 MHz under load on its
own and there is nothing left to pin. Not adopted; it needs root and does not survive a
reboot for no measurable gain.

---

## Long-context prefill, and a `-ub` limit that hangs the GPU — 2026-09-08

`llama-bench -p <n> -n 0 -b 4096`, 35B, ROCm `-fa 0` / Vulkan `-fa 1`.

| Prompt | ROCm ub 512 | ROCm ub 4096 | Vulkan ub 512 | Vulkan ub 2048 | Vulkan ub 4096 |
| ------ | ----------: | -----------: | ------------: | -------------: | -------------: |
| 3 330 | 84.23 | **143.54** | 139.05 | 185.01 | **197.88** |
| 16 384 | 78.01 | **122.97** | 118.49 | — | **157.78** |
| 32 768 | 67.38 | **99.31** | 97.90 | **120.09** | ☠ **device lost** |

Two things fall out of this.

### The `-ub` win holds at long context, and the prefill gap narrows

The gain from a larger micro-batch shrinks with context but stays large: ROCm +70 % at
3.3K, +58 % at 16K, +47 % at 32K. And unlike decode, the ROCm-vs-Vulkan *prefill* gap
**narrows** as context grows — 27 % at 3.3K, 22 % at 16K — because attention takes a
growing share of prefill work and attention is F16, so the emulated-dp4a penalty (which
only hits the quantized FFN and expert GEMMs) is diluted. Vulkan still wins at every
length; at 32K it wins 120.09 to 99.31 using the largest micro-batch that does not crash.

### `-ub 4096` at 32K context hangs the Vulkan queue

`llama-bench -p 32768 -ub 4096 -fa 1` on Vulkan produces:

```
terminate called after throwing an instance of 'vk::DeviceLostError'
  what():  vk::Queue::submit: ErrorDeviceLost
```

and in `dmesg`:

```
amdgpu: ring comp_1.0.1 timeout, signaled seq=51833, emitted seq=51835
amdgpu: Starting comp_1.0.1 ring reset
amdgpu: Ring comp_1.0.1 reset succeeded
amdgpu: [drm] device wedged, but recovered through reset
```

This is a **compute-ring watchdog timeout, not an out-of-memory** — a single dispatch runs
too long and the driver resets the ring. The GPU recovered on its own and both runtimes
still enumerate it, but a server would die and the desktop would freeze for a few seconds.

The failures track the `ctx × ubatch` product:

| ctx × ubatch | product | result |
| ------------ | ------: | ------ |
| 8192 × 4096 | 33.6 M | OK |
| 16384 × 4096 | 67.1 M | OK |
| 32768 × 2048 | 67.1 M | OK |
| 32768 × 4096 | 134.2 M | **device lost** |

`run/start-llama-server.sh` now derives `UBATCH` as `min(4096, 2^26 / CTX)` unless it is
set explicitly, which yields 4096 up to 16K context and 2048 at 32K. **This is a
three-point fit, not a law** — it is a conservative bound, and if a ring reset shows up in
`dmesg` the answer is to lower `UBATCH` further.

> This corrects a mistake introduced earlier the same day: `UBATCH=4096` was made the
> launcher default after measuring only at `-c 8192`, and generalised to all context
> sizes without testing one. Anyone running `CTX=32768` with that default would have hung
> the GPU.

---

## Long context: the ROCm gap widens sharply — 2026-09-08

`llama-bench -n 32 -d <depth> -ub 512`, decode t/s at KV depth. `-ub 512` because at
`-fa 0` the KQ intermediate scales as `n_kv × n_ubatch`; at 32k with `-ub 4096` it would
need 8 GB on its own.

### Qwen3.5-35B-A3B — decode t/s by depth

| Depth | ROCm `-fa 0` | Vulkan `-fa 1` | Vulkan ahead |
| ----- | -----------: | -------------: | -----------: |
| 1 024 | 18.22 | 21.69 | +19 % |
| 4 096 | 15.18 | 21.18 | +40 % |
| 16 384 | 9.33 | 19.22 | **+106 %** |
| 32 768 | **6.16** | **17.10** | **+178 %** |
| *loss 1k → 32k* | *−66 %* | *−21 %* | |

### gemma-4-E4B — decode t/s by depth

| Depth | ROCm `-fa 0` | Vulkan `-fa 1` | Vulkan ahead |
| ----- | -----------: | -------------: | -----------: |
| 1 024 | 14.87 | 18.24 | +23 % |
| 4 096 | 13.56 | 17.62 | +30 % |
| 16 384 | 10.17 | 15.57 | +53 % |
| 32 768 | **7.61** | **13.49** | **+77 %** |
| *loss 1k → 32k* | *−49 %* | *−26 %* | |

**Mechanism.** ROCm cannot use flash attention on gfx900 — the only FA kernel available
is the generic tile kernel, which collapses prefill (see the FA section below), so `-fa 0`
is forced. With `-fa 0`, decode attention runs through `mmvf`, which launches one block
per KV row and does not fold the GQA ratio, so each K head is re-streamed by all 8 of its
Q heads and the cost grows linearly with `n_kv`. Vulkan's flash-attention path has neither
term. This is the same mechanism [ROCM-PERF-AUDIT.md](ROCM-PERF-AUDIT.md) identified for
the 4K falloff; at 32k it dominates.

**At 32k, ROCm decode is not usable** — 6.2 t/s on the 35B, 7.6 on gemma.

### This bounds the dense-model ROCm win

ROCm's prefill advantage on gemma is paid once per prompt; the decode deficit is paid per
token. For a 3330-token prompt, ROCm saves 2.71 s on prefill (15.97 s vs 18.67 s). How
many generated tokens before Vulkan has taken that back:

| Decode depth | Vulkan saves | Crossover |
| ------------ | -----------: | --------: |
| 1 024 | 12.4 ms/token | **218 tokens** |
| 4 096 | 17.0 ms/token | **159 tokens** |
| 16 384 | 34.1 ms/token | **79 tokens** |
| 32 768 | 57.3 ms/token | **47 tokens** |

So ROCm on a dense model is the right choice only for **short answers to long prompts** —
classification, extraction, "answer in one sentence" over a big document. For anything
that generates more than ~150 tokens, or at any context beyond a few thousand, Vulkan wins
overall despite the slower prefill.

---

## Current baseline — 2026-09-08, `-ub 4096`

All backends, both models, re-measured after `-ub` was raised from the upstream default
of 512. Harness: `bench/run-all-benchmarks.sh`, `-c 8192`, prompts ~141 / ~937 / ~3330
tokens. GPU backends at `-b 4096 -ub 4096`; the CPU rows stay at the default because
they are not tile-bound and `start-llama-server.sh --cpu` does not set the flag either.

### Qwen3.5-35B-A3B-Q4_K_M — prefill / decode t/s

| Backend | FA | Prefill | Decode |
| ------- | -- | ------- | ------ |
| **Vulkan** | **ON** | **63.25 / 164.54 / 190.20** | **21.15 / 21.15 / 20.76** |
| Vulkan | OFF | 63.22 / 163.83 / 187.21 | 21.29 / 21.02 / 18.43 |
| ROCm 7.2 Docker | OFF | 48.09 / 122.91 / 141.99 | 20.19 / 19.11 / 15.52 |
| ROCm 7.2 baremetal | OFF | 44.22 / 121.77 / 141.00 | 18.50 / 17.85 / 14.70 |
| ROCm 7.2 Docker | ON | 41.89 / 76.10 / 53.34 | 19.92 / 18.53 / 15.32 |
| ROCm 7.2 baremetal | ON | 45.61 / 76.01 / 53.47 | 18.73 / 17.48 / 14.58 |
| CPU (`-dev none`) | ON | 84.11 / 90.77 / 86.38 | 18.39 / 17.94 / 15.03 |
| CPU (`-dev none`) | OFF | 83.23 / 90.39 / 85.88 | 18.32 / 18.12 / 17.11 |

**Vulkan `-fa 1` is the best backend for this model at every prompt size and for decode.**

### gemma-4-E4B-it-Q4_K_M — prefill / decode t/s

| Backend | FA | Prefill | Decode |
| ------- | -- | ------- | ------ |
| **ROCm 7.2 baremetal** | OFF | 69.94 / 110.62 / **192.29** | 15.91 / 14.43 / 10.23 |
| ROCm 7.2 Docker | OFF | 73.13 / 111.46 / **192.25** | 16.81 / 15.16 / 10.47 |
| **Vulkan** | **ON** | **126.57 / 173.41** / 170.20 | **18.24 / 17.96 / 16.83** |
| Vulkan | OFF | 101.02 / 152.28 / 141.85 | 18.16 / 17.07 / 13.28 |
| ROCm 7.2 baremetal | ON | 65.28 / 48.17 / 30.13 | 16.02 / 14.79 / 12.15 |
| ROCm 7.2 Docker | ON | 66.98 / 47.39 / 30.15 | 16.66 / 15.34 / 12.67 |
| CPU (`-dev none`) | ON | 96.05 / 94.31 / 87.83 | 15.08 / 14.24 / 11.96 |
| CPU (`-dev none`) | OFF | 95.25 / 93.73 / 85.19 | 14.97 / 14.32 / 13.40 |

**On this model ROCm overtakes Vulkan on long-prompt prefill** — 192.3 vs 170.2 at ~3330
tokens, a 13 % lead. Confirmed independently with `llama-bench` (cold prefill, `-r 2`):
ROCm 208.57 ± 0.02 vs Vulkan 178.34 ± 0.01, a 17 % lead. Vulkan is already saturated at
`-ub 512` on gemma (178.34 → 175.82 at 4096) and gains nothing from the larger batch,
while ROCm gains +85 %.

### Which backend to use

| Workload | Winner | Margin |
| -------- | ------ | ------ |
| Long-prompt prefill, **dense** model (gemma) | **ROCm `-fa 0`** | +13 % (harness) / +17 % (llama-bench) |
| Long-prompt prefill, **MoE** model (35B) | **Vulkan `-fa 1`** | +35 % |
| Short-prompt prefill (~128 tok), either model | **Vulkan** | +43 % to +81 % |
| Decode, either model, any context | **Vulkan `-fa 1`** | +13 % to +41 % |

The blanket claim that Vulkan is the better backend on this GPU no longer holds. It is
still the right default — it wins decode everywhere, wins short prompts everywhere, and
wins the MoE model outright — but for long prompts on a dense model ROCm is now faster.

> **Thermal note:** the gemma sweep peaked at 94.1 °C against the board-set 99 °C limit,
> so those numbers may be slightly depressed; the 35B sweep peaked at 82.8 °C and was
> clean. The `llama-bench` confirmation above was run from a 49 °C cold start and agrees,
> so the gemma conclusion does not rest on the warm run.

---

## Prefill micro-batch (`-ub`) — the largest tuning win found — 2026-09-08

`llama-bench`, cold prefill, Qwen3.5-35B-A3B-Q4_K_M, `-ngl 99 -b 4096 -r 2`.
ROCm at `-fa 0`, Vulkan at `-fa 1` (each backend's best flash-attention setting).

| `-ub` | ROCm pp937 | ROCm pp3330 | Vulkan pp937 | Vulkan pp3330 |
| ----- | ---------: | ----------: | -----------: | ------------: |
| 512 (upstream default) | 84.46 | 84.23 | 145.16 | 139.05 |
| 1024 | 115.95 | 110.04 | — | — |
| 2048 | 115.92 | 130.22 | 174.09 | 185.01 |
| **4096** | 114.98 | **143.54** | 173.92 | **197.88** |
| **gain at 4K** | | **+70 %** | | **+42 %** |

**The optimum is roughly `ubatch ≥ prompt length.`** A 937-token prompt is saturated at
`-ub 1024`; a 3330-token prompt keeps improving to 4096. The mechanism is MoE tile fill:
with 256 experts and 8 active per token, a 512-token ubatch puts ~16 tokens on each
expert, while MMQ's tiles are 64 columns wide — so three quarters of every tile that is
fetched from GTT and unpacked into LDS is thrown away. Larger ubatches fill them.

**Cost:** ~2 GB of additional GTT at `-c 8192` (35B: 22.9 GB at `-ub 4096` vs 20.8 GB at
`-ub 512`). No decode cost. On a 42 GB-free machine this is comfortable; lower `UBATCH`
if memory is tight, since `-ub 1024` already captures most of the gain for ~1K prompts.

### What this changes about the ROCm-vs-Vulkan gap

| `-ub` | ROCm | Vulkan | ROCm behind by |
| ----- | ---: | -----: | -------------: |
| 512 | 84.23 | 139.05 | **39 %** |
| 2048 | 130.22 | 185.01 | 30 % |
| 4096 | 143.54 | 197.88 | **27 %** |

Both backends gain, so the gap narrows but does not close. [ROCM-PERF-AUDIT.md](ROCM-PERF-AUDIT.md)
put the prefill gap at ~40 %; that figure was measured at `-ub 512`. At each backend's
best setting it is **27 %**, and the audit's explanation for the remainder — emulated
dp4a on gfx900, 6 VALU instructions per 4 MACs against Vulkan's packed FP16 FMA — still
stands as the floor.

### Launchers were leaving this on the table

| Launcher | Was | Now |
| -------- | --- | --- |
| `run/run-rocm7-baremetal.sh` | `-b 2048 -ub 2048` | `-b 4096 -ub 4096` |
| `run/run-docker-rocm7.sh` | `-b 2048 -ub 2048` | `-b 4096 -ub 4096` |
| `run/start-llama-server.sh` (Vulkan — **the default path**) | nothing, i.e. `-ub 512` | `-b 4096 -ub 4096`, tunable via `BATCH=`/`UBATCH=` |

The Vulkan default path set no batch flags at all, so the backend this repo recommends
was running 42 % below its own capability at 4K context.

### `start-llama-server.sh --cpu` was not CPU-only either

Same `-ngl 0` bug as the benchmark harness: the CPU mode offloaded to the GPU. Now
`-dev none`, verified — GTT 55 MiB and 0 % GPU busy during a request, against ~6.5 GB
and 91 % before.

> **The published backend tables above and below still use `-ub 512`**, because that is
> what the harness passes. They remain internally consistent (every backend measured the
> same way) but they understate every GPU row, and at `-ub 512` they overstate the
> ROCm-vs-Vulkan gap. Re-baselining the tables at `-ub 4096` is open work.

---

## ROCm 7.2 Docker re-verified — 2026-09-08

Image rebuilt from `build/Dockerfile.rocm7-vega` and benchmarked on both models with
the same harness as the baremetal rows. **The 20 GB model loaded without the
2026-06-13 hard freeze** (GTT used 23284 MiB), settling the last open question about
this path.

| Model | FA | Prefill | Decode |
| ----- | -- | ------- | ------ |
| gemma-4-E4B | OFF | 69.98 / 109.49 / 106.04 | 16.06 / 14.62 / 11.97 |
| gemma-4-E4B | ON | 67.55 / 53.83 / 29.41 | 16.86 / 15.56 / 13.07 |
| Qwen3.5-35B | OFF | 46.62 / 94.25 / 88.17 | 19.84 / 18.83 / 15.30 |
| Qwen3.5-35B | ON | 46.33 / 67.82 / 41.39 | 19.92 / 18.56 / 15.29 |

### Docker vs baremetal

| Model | Metric | Baremetal | Docker | |
| ----- | ------ | --------- | ------ | -- |
| gemma | prefill `-fa 0` | 69.99 / 109.10 / 106.13 | 69.98 / 109.49 / 106.04 | identical |
| gemma | decode `-fa 0` | 15.93 / 14.48 / 11.89 | 16.06 / 14.62 / 11.97 | +1 % |
| 35B | prefill `-fa 0` | 47.73 / 94.48 / 88.96 | 46.62 / 94.25 / 88.17 | identical |
| 35B | decode `-fa 0` | 18.70 / 17.79 / 14.66 | **19.84 / 18.83 / 15.30** | **+4-6 %** |

Prefill matches to within measurement noise on both models. The 35B decode difference
looked real (+4-6 %) and was chased down; **it is a measurement artefact of the server
harness, not a backend difference.** Four hypotheses, each tested:

| Hypothesis | Test | Result |
| ---------- | ---- | ------ |
| The image's three extra env vars (`GPU_SINGLE_ALLOC_PERCENT`, `GPU_MAX_HEAP_SIZE`, `GPU_FORCE_64BIT_PTR`) | baremetal harness run with them exported | 18.76 / 17.99 / 14.72 vs 18.70 / 17.79 / 14.66 — **no effect** |
| Different llama.cpp commit (Docker 67672dc vs baremetal 465e49b) | rebuilt baremetal at 67672dc, same harness | 18.93 / 18.00 / 14.72 — **+0.2, does not close a 1.1 gap** |
| Container's `--ulimit memlock=-1` vs the host's 8 MB | same binary, `llama-bench` with and without unlimited memlock | 20.39 / 19.27 / 15.91 vs 20.30 / 19.27 / 15.90 — **no effect** |
| A real backend difference | `llama-bench -n 64 -d 128,1024,4096` on baremetal at Docker's commit | **20.30 / 19.27 / 15.90 — higher than the Docker harness figure of 19.84 / 18.83 / 15.30** |

The last row settles it: measured directly at matched KV depths, baremetal is *faster*
than the number the harness reports for Docker. Whatever produces the apparent Docker
advantage lives in the llama-server measurement path (HTTP, slot handling, sampling,
the `--no-warmup` cold start), not in the HIP backend. Docker and baremetal compile the
same sources against the same ROCm and perform the same.

Two lasting consequences:

- **All four build paths are now pinned** to one commit via `build/llama.cpp-ref`, read
  by the baremetal script, the Vulkan script and both Dockerfiles (`--build-arg
  LLAMA_CPP_REF`, passed by `run/run-docker-*.sh`). They had each been cloning `master`
  independently, which is how a 465e49b baremetal and a 67672dc image came to be
  compared in the first place.
- **`llama-bench -d` is the trustworthy instrument for decode**, and the server harness
  should not be used to compare deployments at the few-percent level. See
  [ROCM-PERF-AUDIT.md](ROCM-PERF-AUDIT.md) §2.

FA ON collapses prefill on Docker exactly as it does on baremetal (35B 4K: 41.4 vs
88.2), so that is a property of the gfx900 kernel, not of the packaging.

---

## Qwen3.5-35B-A3B-Q4_K_M — 2026-09-07

First 35B run since the reinstall, at the 16 GB carve-out / 64 GB GTT configuration.
Same harness and model as the May 2026 35B rows. The 20 GB model loaded on ROCm without
the hard freeze documented for 2026-06-13 — GTT headroom is adequate (42 GB free after
the carve-out).

### Prefill / decode t/s at ~128 / ~1024 / ~4096 (`-c 8192`)

| Backend | FA | Prefill | Decode | vs. May 2026 |
| ------- | -- | ------- | ------ | ------------ |
| **Vulkan GPU** | ON | **73.35 / 159.06 / 153.67** | **21.73 / 21.56 / 20.95** | +13 % prefill, +13 % decode (May: 65.00 / 138.57 / 137.11, 19.06 / 18.95 / 18.47) |
| Vulkan GPU | OFF | 66.89 / 157.57 / 155.91 | 21.30 / 21.21 / 18.60 | — |
| **ROCm 7.2 baremetal** | OFF | **47.73 / 94.48 / 88.96** | **18.70 / 17.79 / 14.66** | +24 % prefill, +12 % decode (May: 42.47 / 72.61 / 71.65, 16.67 / 15.85 / 13.03) |
| ROCm 7.2 baremetal | ON | 43.60 / 67.60 / 41.53 | 18.74 / 17.55 / 14.65 | FA ON hurts *prefill* on every model. For **decode** this reversed on 2026-09-08 with `patches/0001` — see the FA section at the top |
| CPU (`-dev none`) | ON | 88.31 / 94.33 / 87.46 | 18.39 / 17.95 / 14.36 | decode +8 %; prefill not comparable (May used `-ngl 0`, which offloads) |
| CPU (`-dev none`) | OFF | 86.07 / 91.48 / 86.54 | 18.25 / 18.14 / 17.05 | — |

**Vulkan FA ON is the clear best**: 21 t/s decode holding to 4K context, and ~159 t/s
prefill at 1K. Both GPU backends beat May across the board.

### Memory residency — where the model actually sits

| Phase | VRAM (16 GB carve-out) | GTT |
| ----- | ---------------------- | --- |
| Vulkan | **16354 MB** | 5013 MB |
| ROCm | **311 MB** | 20787 MB |
| CPU | 304 MB | 67 MB |

Vulkan fills the carve-out and spills the remainder to GTT. **ROCm ignores the carve-out
entirely** and maps the whole model through GTT — the same pattern the gemma run shows,
and the reason the carve-out helps Vulkan but not ROCm.

Thermals: peak 81.8 °C, average 74.6 °C over 502 s — the coolest sweep recorded on this
rig, and comfortably clear of any throttling.

---

## BIOS retune — 2026-09-07 (current configuration)

Five BIOS changes made together: UMA framebuffer 2 GB → **16 GB**, IOMMU disabled,
iGPU max auto boost +200 MHz, CPU Curve Optimizer −10 → **−15** all-core, CPU throttle
limit 90 → 99 °C. Same model and harness as every row below, so the comparison against
the 2 GB-carve-out run is clean — but with five variables moved at once, only the
aggregate effect is attributable.

### gemma-4-E4B-it-Q4_K_M (`-c 8192`), prefill / decode t/s at ~128 / ~1024 / ~4096

| Backend | FA | Prefill | Decode |
| ------- | -- | ------- | ------ |
| **Vulkan GPU** | ON | **127.11 / 171.57 / 172.05** | **18.39 / 17.99 / 17.15** |
| Vulkan GPU | OFF | 101.36 / 153.07 / 156.85 | 18.10 / 17.14 / 14.91 |
| **ROCm 7.2 baremetal** | OFF | **69.99 / 109.10 / 106.13** | **15.93 / 14.48 / 11.89** |
| ROCm 7.2 baremetal | ON | 65.36 / 53.72 / 29.47 | 16.05 / 14.83 / 12.56 |
| CPU (`-dev none`) | ON | 96.26 / 96.59 / 90.13 | 15.18 / 14.36 / 12.26 |
| CPU (`-dev none`) | OFF | 95.68 / 94.96 / 86.26 | 14.96 / 14.46 / 13.46 |

### What the retune changed, versus the 2 GB carve-out

| Backend | Prefill @4K | Decode @4K |
| ------- | ----------- | ---------- |
| ROCm FA OFF | 87.16 → **106.13** (+22 %) | 10.63 → **11.89** (+12 %) |
| Vulkan FA ON | 171.29 → 172.05 (+0.4 %) | 16.36 → **17.15** (+4.8 %) |
| CPU FA ON | 88.37 → 90.13 (+2 %) | 12.05 → 12.26 (+1 %) |

Attribution below was revised after the 35B run — read the correction before relying on it.

**Correction (2026-09-07, after the 35B run):** an earlier version of this section
claimed the carve-out was what moved ROCm, citing the run's VRAM peak rising from
1754 MB to 3645 MB. That was wrong — the figure was a whole-run peak taken during the
*Vulkan* phase and attributed to ROCm without checking the per-phase split.

Sampling VRAM and GTT per backend shows **ROCm does not use the BIOS carve-out at all**:

| Model | Phase | VRAM (carve-out) | GTT |
| ----- | ----- | ---------------- | --- |
| gemma-4-E4B | Vulkan | 3638 MB | 2872 MB |
| gemma-4-E4B | **ROCm** | **281 MB** | **3660 MB** |
| Qwen3.5-35B | Vulkan | 16354 MB | 5013 MB |
| Qwen3.5-35B | **ROCm** | **311 MB** | **20787 MB** |

ROCm puts the whole model in GTT regardless of how much carve-out is available. The
carve-out benefits **Vulkan**, and roughly in proportion to how much of the model it can
hold — which is why Vulkan gained only ~5 % decode on the 5 GB gemma (already mostly
resident at 2 GB) but 12–15 % on the 20 GB Qwen (16 GB moved out of GTT).

**So what caused ROCm's +22 % here is not established.** It is not the carve-out. The
retune changed four other things at once, and this run cannot separate them.

**The +200 MHz iGPU boost did nothing measurable.** `pp_dpm_sclk` reports a 2200 MHz top
state instead of 2000, but the clock actually observed under load is still 2400 MHz —
the same as before the change. The DPM table and the real boost ceiling are not the same
thing on this APU.

**IOMMU off broke nothing.** `/dev/kfd` and `rocminfo` still work; ROCm on this APU does
not depend on it. No measurable speed change either.

**Curve Optimizer −15 outweighed the raised throttle limit**: peak 86.6 °C and average
77.8 °C, versus 89.5 / 79.7 at −10. The run never approached the new 99 °C limit, so
that setting did not come into play — but it does remove the safety margin that the
95 °C stock Tjmax provided, and a repaste is still outstanding.

---

## Re-verification after the 26.04 reinstall — 2026-09-07

The run that established the restored stack works, at the **2 GB** carve-out the rig had
before the BIOS retune above. Same model and harness as the 2026-05-15 gemma rows, so
these are what reproduce (or fail to reproduce) the May numbers. Ubuntu 26.04.1 /
kernel 7.0 / classic ROCm 7.2.0, Vega 8 as the only GPU, 64 GB GTT (May was 32 GB GTT).

Recorded after fan speed was raised — an identical earlier sweep throttled and measured
6–8 % lower prefill on every backend.

| Backend | FA | Prefill | Decode | vs. 2026-05-15 |
| ------- | -- | ------- | ------ | -------------- |
| Vulkan GPU | ON | 124.23 / 170.73 / 171.29 | 17.55 / 17.16 / 16.36 | reproduces (May: 121.97 / 166.81 / 160.02) |
| Vulkan GPU | OFF | 95.60 / 149.48 / 155.30 | 17.19 / 16.39 / 14.29 | — |
| ROCm 7.2 baremetal | OFF | 56.46 / 89.10 / 87.16 | 14.28 / 12.91 / 10.63 | reproduces at 1K/4K (May: 70.91 / 84.88 / 84.95) |
| ROCm 7.2 baremetal | ON | 55.23 / 47.36 / 27.11 | 14.32 / 13.31 / 11.44 | FA ON collapses prefill — as documented |
| CPU (`-dev none`) | ON | 95.38 / 94.66 / 88.37 | 15.05 / 14.24 / 12.05 | first genuine CPU-only measurement; the May prefill row was `-ngl 0` and is not comparable |
| CPU (`-dev none`) | OFF | 94.88 / 93.23 / 84.69 | 14.95 / 14.39 / 13.44 | — |

**Vulkan and ROCm both reproduce**, which is the main result: the restored stack performs
as it did before the reinstall. Use `-fa 1` for Vulkan and `-fa 0` for ROCm.

### `-fa auto` resolves to ON on ROCm, and that costs 57 % of prefill

`-fa` defaults to `auto`, which is resolved by probing the backend for
`FLASH_ATTN_EXT` support. On gfx900 `ggml_cuda_get_best_fattn_kernel` falls through
to the generic tile kernel (`fattn.cu:652-666`), so the probe succeeds and FA is
enabled — the worst setting on this GPU. Measured 2026-09-08, gemma-4-E4B, 3330-token
prompt, `llama-bench -ngl 99`:

| `-fa` | Prefill t/s |
| ----- | ----------- |
| `0` | **112.78** |
| `1` | 48.91 |
| `auto` | 48.90 |

`run/run-rocm7-baremetal.sh` passed no `-fa` at all, so every launch through it ran
with flash attention on. Fixed: the launcher now defaults to `-fa 0` (still
overridable by passing your own `-fa` after the model). The benchmark harness was
never affected — it always passed an explicit `-fa`.

### Closed: the pre-September CPU prefill figures were not CPU measurements

**Cause identified.** `-ngl 0` does not force CPU-only execution on current
llama.cpp — upstream changed the `-ngl` default to `auto`, and with a GPU backend
present the model is still offloaded. Measured directly:

| Flag | GPU busy | GTT used |
| ---- | -------- | -------- |
| `-ngl 0` | 91 % | 6567 MB |
| `-dev none` | idle | 157 MB |

Every CPU row in this file dated before 2026-09-07 was produced with `-ngl 0`, so
none of them is a CPU measurement. `start_cpu()` in the harness now uses `-dev none`.

**Current figures, confirmed two independent ways** (gemma-4-E4B, prefill t/s at the
harness prompt sizes):

| Method | ~141 | ~937 | ~3330 |
| ------ | ---- | ---- | ----- |
| `llama-bench -dev none -t 8 -r 2` | 99.31 ± 0.07 | 98.06 ± 0.27 | 93.07 ± 0.65 |
| harness (llama-server, `-dev none`) | 96.26 | 96.59 | 90.13 |
| *May 2026 (`-ngl 0`)* | *249.57* | *754.82* | *840.50* |

The two current methods share no code path beyond the model file — no server, no
HTTP, no prompt cache in the first — and agree within 3 %. The result is stable
across 8 and 16 threads, cold and warm start, and both models.

**Why the May figure cannot be a real CPU measurement.** gemma-4-E4B is 7.52 B
parameters (≈ 4 B "effective"). Prefill costs about `2 × N × T` operations, so
840 tok/s needs 6.7 TOP/s at 4 B active, or 12.6 TOP/s at 7.52 B. A 5700G's eight
Zen 3 cores issue at most two 256-bit `vpmaddubsw` per cycle per core = 128 int8
ops/cycle, i.e. **≈ 4.1 TOP/s peak at 4 GHz**. The claimed number is 1.6–3× *above*
the theoretical ceiling of the whole CPU, while the measured 93 t/s sits at 34 % of
it — a normal efficiency for real quantized GEMM.

> An earlier revision of this file said the gap was "an order of magnitude" beyond
> the chip's ability. That was an overstatement: it is 1.6–3× above peak. The
> conclusion is unchanged — above 100 % of peak is impossible — but the factor was
> wrong and is corrected here.

A second, independent sanity check: at 840 t/s the CPU would be outrunning the iGPU
(ROCm 106, Vulkan 172 on the same model) by 5× on a compute-bound task, using the
same memory bus.

**35B-A3B**: the May CPU row (58 / 197 / 211) is not provably impossible on
arithmetic alone, but it was produced the same way and did not reproduce either
(measured 88 / 94 / 87 with `-dev none`). It is treated as superseded for the same
reason.

**Both models' pre-September CPU prefill rows are therefore erroneous, not a
regression.** They are left in the historical tables, struck through, so the record
of what was believed stays intact.

### Thermals — before and after raising fan speed

Two identical 6-backend sweeps, the second after the fan curve was raised:

| | Before | After |
| --- | ------ | ----- |
| Idle | ~50 °C | **40–41 °C** |
| Peak under load | **105.4 °C** | **89.5 °C** |
| Average under load | 90.5 °C | **79.7 °C** |
| Samples ≥ 95 °C (stock Tjmax) | 27 | **0** |
| Samples ≥ 100 °C | 22 | **0** |
| iGPU SCLK under load | 2400 → 2208 MHz | **2400 → 2351 MHz** |
| CPU clocks | 4000 → 3450 MHz | no downward drift |

Throttling stopped, and that alone recovered ~6 % prefill on Vulkan, ~8 % on ROCm and
~7 % on CPU — which is why numbers measured while this rig throttles are a floor, not a
result.

> The CPU phase is not a strict A/B between those two sweeps: the earlier one still had
> the `-ngl 0` bug, so its "CPU" rows ran on the GPU. The Vulkan and ROCm phases are
> identical workloads in both and carry the comparison.

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
| ROCm 7.2 baremetal     | ON    |    40.48 |     55.79 |     37.55 | FA hurts large-context prefill (confirmed; see FA note)          |
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
| CPU                    | OFF   | ~~235.17~~ |  ~~693.91~~ |  ~~772.09~~ | **Not a CPU measurement** — `-ngl 0` offloaded to the GPU. Decode figures stand |
| CPU                    | ON    | ~~249.57~~ |  ~~754.82~~ |  ~~840.50~~ | **Not a CPU measurement** (above the CPU's arithmetic ceiling). Real CPU-only: 99 / 98 / 93. See "Closed: the pre-September CPU prefill figures" |
| ROCm 7.2 baremetal     | OFF ✅ |    70.91 |     84.88 |     84.95 | Close to previous; still recommended ROCm mode                   |
| ROCm 7.2 baremetal     | ON    |    65.77 |     47.70 |     27.67 | FA severely hurts ROCm prefill (confirmed; see FA note)          |
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
| 2026-05-15 | BIOS 2GB;32GB GTT,current ROCm tweaks  | CPU                  | ON  | ~~249.57 / 754.82 / 840.50~~     | 15.06 / 14.24 / 12.17        | **Prefill is not a CPU measurement** (`-ngl 0` offloaded; figure also exceeds the CPU's arithmetic ceiling). Real CPU-only 2026-09-08: 99 / 98 / 93. Decode reproduced |
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
