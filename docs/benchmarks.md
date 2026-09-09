# LLM Inference Benchmarks — Vega 8 iGPU

llama.cpp on an AMD Ryzen 7 5700G / Radeon Vega 8 (gfx90c presented as gfx900), Ubuntu
26.04 / kernel 7.0, 64 GB DDR4-4200, 16 GB BIOS carve-out, 64 GB GTT.

The file is organised by *question*, not by date:

| Section | Answers |
| --- | --- |
| [The matrix](#the-matrix--2026-09-0809) | What does each backend actually do, on both models, from 4K to 32K? |
| [What moves the numbers](#what-moves-the-numbers) | Which knobs and patches are worth anything, and by how much |
| [Measurement traps](#measurement-traps) | Five ways this project measured the wrong thing, and how each was caught |
| [Where the model sits](#where-the-model-sits-in-memory) | VRAM vs GTT residency per backend |
| [History](#history) | Every earlier run, one table per model |

---

## The matrix — 2026-09-08/09

`llama-bench -ngl 99 -b 2048 -ub 2048 -r 1`, prefill via `-p <ctx>`, decode via
`-n 32 -d <ctx>`. ROCm build carries [`patches/0001`](../patches/README.md)
(`v_mad_mix_f32`). Machine cooled below 55 °C between runs. Raw data:
[`bench/results/2026-09-08-matrix.tsv`](../bench/results/2026-09-08-matrix.tsv).

`-ub 2048` uniformly, including where a backend could go higher, so every cell in the
matrix is comparable. The larger micro-batch is worth more on ROCm — see
[micro-batch](#micro-batch--ub--the-largest-runtime-knob).

### Qwen3.5-35B-A3B Q4_K_M — MoE, 34.7 B total / ~3 B active, head dim 256

| Backend | Prefill 4K | Prefill 16K | Prefill 32K | TG 4K | TG 16K | TG 32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vulkan `-fa 1`** | **188.5** | 151.5 | **120.1** | **21.2** | **19.2** | **17.1** |
| Vulkan `-fa 0` | 187.4 | **156.6** | 116.4 | 19.0 | 13.5 | 9.7 |
| **ROCm `-fa 1`** | 139.3 | 123.7 | 107.9 | 18.8 | 17.5 | 15.9 |
| ROCm `-fa 0` | 136.5 | 119.3 | 96.6 | 15.2 | 9.4 | 5.9 |
| CPU `-fa 0` | 83.9 | 71.7 | 59.7 | 16.1 | 13.5 | 11.1 |
| CPU `-fa 1` | 85.1 | 71.1 | 58.3 | 15.3 | 8.1 | 4.3 |

### gemma-4-E4B-it Q4_K_M — dense, 7.5 B, head dim 512

| Backend | Prefill 4K | Prefill 16K | Prefill 32K | TG 4K | TG 16K | TG 32K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Vulkan `-fa 1`** | **179.4** | **155.1** | — | **17.3** | **15.3** | — |
| Vulkan `-fa 0` | 166.5 | 150.3 | — | 15.0 | 12.0 | — |
| **ROCm `-fa 1`** | 160.4 | 145.9 | **131.9** | 15.0 | 13.6 | **12.2** |
| ROCm `-fa 0` | 153.6 | 138.7 | 121.2 | 12.0 | 9.3 | 7.1 |
| CPU `-fa 0` | 84.9 | 77.3 | 69.1 | 13.8 | 11.4 | 9.7 |
| CPU `-fa 1` | 89.4 | 78.0 | 67.1 | 12.6 | 7.9 | 5.3 |

> **The four em dashes are a crash, not missing work.** gemma on Vulkan at 32K hangs the
> GPU compute ring — `ring comp_1.1.1` / `comp_1.0.1 timeout`, `device wedged`, and on the
> third attempt a full machine lock. It reproduces with the iGPU downclocked to 2200 MHz
> and its Curve Optimizer disabled, so it is the workload, not the silicon's margins. Log:
> [`bench/results/gemma-vulkan-32k-retry.log`](../bench/results/gemma-vulkan-32k-retry.log).
> Cause and the derived `-ub` cap: [the `-ub` ceiling](#the--ub-ceiling-that-hangs-the-gpu).

### What the matrix says

| Question | Answer |
| --- | --- |
| Default backend? | **Vulkan `-fa 1`** — fastest at almost every cell, and the only path needing no local patch |
| 32K prefill on the dense model? | **ROCm `-fa 1`** (131.9). Vulkan cannot run that cell at all |
| `-fa` on ROCm? | **`-fa 1`, always** — with `patches/0001` it wins prefill *and* decode at every context. Without the patch, `-fa 0` |
| `-fa` on Vulkan? | **`-fa 1`.** One exception: 16K prefill on the 35B, where `-fa 0` is 3 % faster |
| `-fa` on CPU? | **`-fa 0`.** `-fa 1` costs 61 % of decode at 32K on the 35B, 46 % on gemma |
| Is ROCm worth it? | Long-context decode is now 8–20 % behind Vulkan, down from 178 %. Vulkan still wins overall |

Three patterns worth naming:

**The prefill gap closes with context, the decode gap does not.** On the 35B, Vulkan leads
ROCm by 35 % in prefill at 4K but only 11 % at 32K — attention takes a growing share of
prefill work, attention is F16, and the emulated-dp4a penalty only hits the quantized FFN
and expert GEMMs. Decode holds at 8 % (`-fa 1`) because both backends now run a real
flash-attention kernel.

**Falloff from 4K to 32K separates a working FA path from a missing one.** Decode with FA:
Vulkan −19 %, ROCm −15 %. Without it: ROCm `-fa 0` −61 %, CPU `-fa 1` −73 %.

**The CPU is not embarrassing on the MoE model.** 11.1 t/s decode at 32K is 35 % behind
Vulkan and *ahead* of un-patched ROCm (5.9). Prefill is a different story — roughly half
the GPU rate at every length.

---

## What moves the numbers

| Lever | Type | Best measured effect | Adopted |
| --- | --- | --- | --- |
| [`v_mad_mix_f32` FA patch](#flash-attention-on-gfx900--patches0001) | patch | ROCm decode 32K **+157 %** | **Yes** — `patches/0001` |
| [`-ub` 512 → 4096](#micro-batch--ub--the-largest-runtime-knob) | runtime | ROCm prefill **+70 %**, Vulkan +42 % | **Yes**, capped by context |
| [`-ctk q8_0`](#-ctk-q8_0--scales-with-context) | runtime | ROCm decode 32K **+23.5 %** | **Yes**, for long context |
| [Cooling](#cooling) | hardware | +6–8 % prefill, all backends | **Yes** — fan curve raised |
| BIOS carve-out 2 → 16 GB | BIOS | Vulkan decode +12–15 % on the 20 GB model | **Yes** — helps Vulkan only |
| [FA occupancy = 1 on GCN](#the-first-fa-attempt-fixed-the-symptom) | patch | 32K decode 6.16 → 14.85 | **No** — superseded, and a net loss on top of `patches/0001` |
| [Clock pinning / COMPUTE profile](#things-that-do-nothing-now) | runtime | ±1 % (noise) | No |
| [`-DGGML_CUDA_FORCE_MMQ=ON`](#things-that-do-nothing-now) | build | ±0.4 % | No |
| [iGPU boost +200 MHz](#things-that-do-nothing-now) | BIOS | none — observed clock unchanged | Irrelevant |
| [IOMMU off](#things-that-do-nothing-now) | BIOS | none | Irrelevant |

### Flash attention on gfx900 — `patches/0001`

One instruction. `patches/0001-ggml-cuda-mad-gfx900-mad-mix.patch` makes the FA KQ
accumulate use `v_mad_mix_f32`, which gfx900 has and llama.cpp did not emit for it.

**The bug.** `V_DOT2_F32_F16_AVAILABLE` (`common.cuh:760`) is defined only for RDNA2+,
gfx906 and CDNA. On GCN5 `ggml_cuda_mad(float&, half2, half2)` therefore falls back to
`__half22float2(v*u); acc += tmp.x + tmp.y` — `v_pk_mul_f16` + 2× `v_cvt_f32_f16` +
2× `v_add_f32`, **5 VALU ops per 2 MACs**, product formed in fp16. Those intermediates
also cost registers: with `-Rpass-analysis=kernel-resource-usage` the FA tile kernels sat
at the 128-VGPR cap, **50 of 60 config rows spilling**, up to 2262 VGPRs and 2.9 KB/lane
of scratch. gfx900 has the VOP3P mad-mix family — f16 × f16 + f32 in one instruction,
product in fp32; two ops with `op_sel` cover a half2, so **1 VALU op per MAC**.

| Metric | Before | After |
| --- | ---: | ---: |
| VALU ops per MAC | 2.5 | **1** |
| Spills, 256×256 FA rows | 10 598 | **6** |
| VGPR use | 128 (capped) | 76–130 |
| Best occupancy reached | 2 | **3** |
| Error vs fp64 on 256 random pairs | 6.1e-3 | **1.4e-6** |

Decode t/s at KV depth, `llama-bench -n 32 -d -ub 512`:

| Model | Depth | `-fa 0` (was best) | `-fa 1` before | `-fa 1` **after** | vs. best before |
| --- | --- | ---: | ---: | ---: | ---: |
| Qwen 35B | 4 096 | 15.12 | 14.95 | **18.68** | +24 % |
| Qwen 35B | 32 768 | 6.16 | *timed out* | **15.86** | **+157 %** |
| gemma | 4 096 | 13.56 | 13.08 | **15.44** | +14 % |
| gemma | 32 768 | 7.61 | 5.97 | **12.49** | **+64 %** |

**`-fa 1` is now the right setting for ROCm.** Every table and launcher note in this repo
said the opposite, correctly, until this patch. The long-context collapse is gone: ROCm
decode fell 66 % from 1K to 32K and now falls about as much as Vulkan's 21 %.

**Validation.** `test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0`: **2959/2959 passed**,
on the shipped configuration and on both superseded variants. Numerics checked against an
fp64 GPU reference before the patch was written. `-fa 0` numbers are unchanged, confirming
nothing outside the FA path moved. Untested: FA shapes other than 256×256 and 512×512, and
non-FA callers of `ggml_cuda_mad(float&, half2, half2)`.

#### The first FA attempt fixed the symptom

Before finding the instruction, the spill data pointed at FA tile *occupancy*: the config
table is shared with CDNA, which has AGPRs to spill into and GCN5 does not, so
`occupancy = 2` caps every kernel at 128 VGPRs. Giving GCN `occupancy = 1` lifted the cap
to 256 and took 35B decode at 32K from 6.16 to 14.85 — a real improvement with a correctly
identified mechanism. But the kernels were short of registers only because the fp16
fallback materialised intermediates that need not exist. Head to head at 32K on the 35B:

| | t/s |
| --- | ---: |
| occupancy only | 14.85 |
| **mad-mix only** | **15.86 ± 0.02** |
| both | 15.41 |

Occupancy 1 *on top of* mad-mix is a net loss — it pulls kernels that now reach occupancy 3
back down to 1. The occupancy patch was dropped.

### Micro-batch (`-ub`) — the largest runtime knob

`llama-bench`, cold prefill, 35B, `-ngl 99 -b 4096 -r 2`. ROCm at `-fa 0`, Vulkan at
`-fa 1` (each backend's best FA setting at the time of the run).

| `-ub` | ROCm pp937 | ROCm pp3330 | ROCm pp16384 | ROCm pp32768 | Vulkan pp937 | Vulkan pp3330 | Vulkan pp16384 | Vulkan pp32768 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 512 (upstream default) | 84.46 | 84.23 | 78.01 | 67.38 | 145.16 | 139.05 | 118.49 | 97.90 |
| 1024 | 115.95 | 110.04 | — | — | — | — | — | — |
| 2048 | 115.92 | 130.22 | — | — | 174.09 | 185.01 | — | **120.09** |
| **4096** | 114.98 | **143.54** | **122.97** | **99.31** | 173.92 | **197.88** | **157.78** | ☠ device lost |
| **gain over 512** | +37 % | **+70 %** | **+58 %** | **+47 %** | +20 % | **+42 %** | **+33 %** | — |

**The optimum is roughly `ubatch ≥ prompt length`,** and the gain shrinks with context but
stays large. Mechanism on the MoE model: with 256 experts and 8 active per token, a
512-token ubatch puts ~16 tokens on each expert while MMQ's tiles are 64 columns wide — so
three quarters of every tile fetched from GTT and unpacked into LDS is thrown away.

Prefill was **still climbing at 4096** on ROCm at long prompts, so this knob has not been
shown to saturate; probing `-ub 8192` is open work in the [README](../README.md) TODO.

**Cost:** ~2 GB additional GTT at `-c 8192` (35B: 22.9 GB at `-ub 4096` vs 20.8 at 512).
No decode cost. `-ub 1024` already captures most of the gain for ~1K prompts if memory is
tight.

This also re-dates a claim from [ROCM-PERF-AUDIT.md](ROCM-PERF-AUDIT.md): it put the ROCm
prefill gap at ~40 %, measured at `-ub 512`. At each backend's best micro-batch it is
27 %, and at 32K with `patches/0001` it is 11 %.

Every launcher was leaving this on the table — `run/start-llama-server.sh`, the Vulkan
default path this repo recommends, set no batch flags at all and so ran 42 % below its own
capability at 4K.

### Micro-batch, settled: 4096 is the optimum and the gain depends on the model

The sweep above stops at 4096 and was measured before `patches/0001`. This one
re-measures at each context with the patched build and adds 8192, prefill only
(`llama-bench -p <ctx> -n 0 -r 1`, raw data:
[`bench/results/2026-09-09-ub-sweep.tsv`](../bench/results/2026-09-09-ub-sweep.tsv)).
26 cells, 25 completed, **zero ring resets**.

#### `-ub 4096` against the matrix's `-ub 2048`

| Model | Backend | FA | 4K | 16K | 32K |
| --- | --- | --- | ---: | ---: | ---: |
| **gemma** | ROCm | `1` | **249.0** (+55 %) | **202.1** (+39 %) | **173.8** (+32 %) |
| gemma | ROCm | `0` | 205.3 (+34 %) | 177.3 (+28 %) | 149.5 (+23 %) |
| 35B | ROCm | `1` | 151.9 (+9 %) | 130.1 (+5 %) | 112.6 (+4 %) |
| 35B | ROCm | `0` | 144.2 (+6 %) | 123.2 (+3 %) | 99.3 (+3 %) |
| 35B | Vulkan | `1` | 198.4 (+5 %) | 157.9 (+4 %) | crash |
| 35B | Vulkan | `0` | 191.9 (+2 %) | **144.1 (−8 %)** | crash |
| gemma | Vulkan | `1` | 177.1 (−1 %) | crash | crash |
| gemma | Vulkan | `0` | **152.0 (−9 %)** | crash | crash |

**The gain is a property of the model, not the context.** The dense model gains 23–55 %
at every context and both FA settings; the MoE gains 3–9 %. That is the tile-fill argument
reaching its limit: with 8 of 256 experts active, each expert sees `ub × 8/256` tokens, so
`-ub 2048` already puts 64 columns on each one — exactly MMQ's tile width. Nothing above it
can help. A dense model has no such cutoff: every token passes through every weight, so
doubling the micro-batch keeps doubling the columns per weight fetch. The older sweep shows
the same thing from the other side — on the 35B, `-ub` 512 → 2048 was worth +55 % and
2048 → 4096 only +10 %.

**Vulkan regresses with `-ub 4096` at `-fa 0`.** Two independent cells, different models
and contexts, both about −8 %, while every `-fa 1` cell is flat or positive. Mechanically
consistent: without flash attention the KQ intermediate is `n_kv × n_ubatch`, so a larger
micro-batch doubles it; with `-fa 1` it is never materialised. Two cells at `-r 1` is thin
evidence for a −8 % effect — it needs a repeat at `-r 3` before it is more than a caution.

#### `-ub 8192`: uniformly worse, question closed

| Model | FA | 16K | 32K |
| --- | --- | ---: | ---: |
| 35B | `1` | 129.6 (−0.4 %) | 109.7 (−2.6 %) |
| 35B | `0` | 117.2 (−4.9 %) | **failed to run** |
| gemma | `1` | 184.8 (−8.6 %) | 162.9 (−6.3 %) |
| gemma | `0` | 146.7 (−17.3 %) | 127.0 (−15.1 %) |

Percentages against `-ub 4096`. Every cell is a loss, and the losses are largest exactly
where the KQ intermediate is largest (`-fa 0`, long context). **`-ub 4096` is the optimum;
do not raise it.**

The one cell that did not run is an ordinary out-of-memory, not a watchdog hang — `dmesg`
was clean and the backtrace goes `ggml_cuda_pool_leg::alloc` → `ggml_cuda_error` →
`ggml_abort`. At `-fa 0` the KQ intermediate is materialised as
`n_kv × n_ubatch × n_head × 4 B`, and the 35B has 16 heads:

```
32768 × 8192 × 16 × 4 = 17,179,869,184 B = 16 GiB
```

one contiguous allocation, on top of ~20 GB of weights and the KV cache. At `-ub 4096` the
same buffer is 8 GiB and the cell runs. This is arithmetic, not a hardware limit: it is the
`-fa 0` KQ term, which is why `-fa 1` — where KQ is never materialised — completed both
32K cells at `-ub 8192`. This closes an open question that expected headroom above 4096 because
prefill was still climbing there on the 35B — it was climbing at `-fa 0` pre-patch, and the
climb does not continue.

#### ROCm has no ring-hang ceiling anywhere near Vulkan's

The riskiest cell — gemma at 32K with `-ub 8192`, `n_kv × ub × head` = **137e9** — ran
clean. That is 4× the largest product ROCm had previously demonstrated and **8× the value
that reliably kills Vulkan**. Across all 26 cells `dmesg` recorded not one ring timeout.

So the two backends are not merely differently tuned here, they are differently bounded:
the watchdog limit is a property of Vulkan's dispatch shape and does not describe the
hardware. `run/start-llama-server.sh` accordingly gives ROCm a flat `-ub 4096` — the
measured optimum, and now well inside demonstrated-safe territory — while Vulkan keeps its
context-derived cap.

---

### The `-ub` ceiling that hangs the GPU

`llama-bench -p 32768 -ub 4096 -fa 1` on Vulkan produces
`vk::DeviceLostError … ErrorDeviceLost`, and in `dmesg` a `ring comp_1.0.1 timeout`
followed by a successful ring reset. **A compute-ring watchdog timeout, not an OOM** — it
happens with tens of GB free.

The work in one attention dispatch scales with `n_kv × ubatch × head_dim` — where `n_kv`
is how many tokens are **actually in the KV cache when the dispatch runs**, not the context
the model was loaded with. That product is what predicts the hang:

| Model | Head dim | ctx | ub | ctx × ub | × head dim | Result |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Qwen3.5-35B-A3B | 256 | 32768 | 2048 | 67.1 M | 17.2e9 | OK (120.1 t/s) |
| Qwen3.5-35B-A3B | 256 | 32768 | 4096 | 134.2 M | 34.4e9 | **DEVICE LOST** |
| gemma-4-E4B-it | 512 | 32768 | 2048 | 67.1 M | 34.4e9 | **DEVICE LOST** |
| gemma-4-E4B-it | 512 | 16384 | 2048 | 33.6 M | 17.2e9 | OK |

The two failures share 34.4e9 and the two passes share 17.2e9, while `ctx × ubatch` alone
puts a pass and a failure in the same 67.1 M bucket.

#### Allocated context is not the variable — `n_kv` is

The rows above all fill the context they allocate, so they cannot separate the two. This
run does, by holding the allocation fixed at 128k and varying only the prompt
(gemma, Vulkan, `-c 131072 -b 4096 -ub 4096 -fa 1`, raw data:
[`bench/results/2026-09-09-nkv-not-ctx.tsv`](../bench/results/2026-09-09-nkv-not-ctx.tsv)):

| Allocated `-c` | Actual prompt (`n_kv`) | `n_kv × ub × head` | Result |
| ---: | ---: | ---: | --- |
| 131 072 | 61 | 0.13e9 | OK, 1.3 s |
| 131 072 | 6 021 | 12.6e9 | OK, 35.9 s |
| 131 072 | ~16 000 | **33.6e9** | **DEVICE LOST** — `ring comp_1.2.0 timeout` |

The same 128k context both works and wedges the GPU. **Allocating a large context is just
memory; it costs no dispatch time until you fill it.** The 34.4e9 threshold holds.

This is why LM Studio runs gemma at a 128k context with `evalBatchSize 4096` on Vulkan
without trouble — a chat turn puts a few hundred tokens in the cache, nowhere near the
limit. Paste a 16k-token document into that same configuration and it hangs the ring
exactly as above; the setting is not safer there, it is just never exercised.

A launcher cannot know `n_kv` ahead of time, so `start-llama-server.sh` derives its cap
from `CTX` — the worst case, a prompt that fills the context. That is the right default for
a server that may be handed anything, but it is pessimistic for interactive use: if you
know your prompts stay short, a much larger `-ub` is safe at any allocation, and `UBATCH=`
overrides the derivation.

`run/start-llama-server.sh` derives `UBATCH` as `min(4096, 2²⁵/CTX)` unless it is set
explicitly — 2²⁵ rather than 2²⁶ so the bound holds for head dim 512. **It is a four-point
fit, not a law**; if a ring reset shows up in `dmesg`, lower `UBATCH` further.

> Two corrections are folded into that paragraph. `UBATCH=4096` was first made the default
> after measuring only at `-c 8192` and generalised to all contexts untested — anyone
> running `CTX=32768` would have hung the GPU. The replacement cap, `2²⁶/CTX`, was fitted
> to the 35B alone; it calls gemma at 32K/`-ub 2048` safe, and that configuration crashes.

**This is a Vulkan limit, not a hardware one.** ROCm ran `32768 × 4096` on the 35B fine at
99.31 t/s prefill — the exact product that kills Vulkan. The launcher therefore caps the two
backends separately:

| CTX | Vulkan `-ub` (2²⁵/CTX) | ROCm `-ub` (2²⁶/CTX) |
| ---: | ---: | ---: |
| 4 096 | 4096 | 4096 |
| 8 192 | 4096 | 4096 |
| 16 384 | 2048 | 4096 |
| 32 768 | 1024 | 2048 |

ROCm's bound is the largest product any ROCm run has demonstrated (34.4e9), reached from
both directions — the 35B at `32768 × 4096 × 256` and gemma at `32768 × 2048 × 512`. It is
deliberately not a flat `-ub 4096`: that would put gemma at 32K on `32768 × 4096 × 512` =
68.7e9, double anything measured on either backend. ROCm's actual ceiling is unmeasured and
is probably higher; finding it is open work.

### `-ctk q8_0` — scales with context

K cache to q8_0, V stays f16. 35B, ROCm `-fa 0`, decode t/s by depth:

| Depth | `-ctk f16` | `-ctk q8_0` | Gain |
| --- | ---: | ---: | ---: |
| 1 024 | 18.10 | 18.58 | +2.7 % |
| 4 096 | 15.09 | 16.11 | +6.8 % |
| 16 384 | 9.32 | 10.89 | +16.8 % |
| 32 768 | 6.16 | **7.61** | **+23.5 %** |

The June 2026 sweep measured this as "+3.5 %, small" and did not adopt it. That was correct
*at 4K* — the effect simply was not measured where it matters.

It also decomposes the (then unfixed) long-context decode collapse. Halving the K cache
would nearly double throughput if KV bandwidth were the whole bottleneck; it gives
+23.5 %, so KV bandwidth was roughly a quarter of the problem at 32K. The rest was the
`mmvf` dispatch structure — one block per KV row, GQA ratio not folded, each K head
re-streamed by all 8 of its Q heads — plus the V cache, which cannot be quantized without
FA. Neither is reachable with a flag; both needed the flash-attention kernel fixed, which
`patches/0001` did.

### Cooling

Two identical 6-backend sweeps, the second after the fan curve was raised:

| | Before | After |
| --- | ---: | ---: |
| Idle | ~50 °C | **40–41 °C** |
| Peak under load | **105.4 °C** | **89.5 °C** |
| Average under load | 90.5 °C | **79.7 °C** |
| Samples ≥ 95 °C (stock Tjmax) | 27 | **0** |
| Samples ≥ 100 °C | 22 | **0** |
| iGPU SCLK under load | 2400 → 2208 MHz | **2400 → 2351 MHz** |
| CPU clocks | 4000 → 3450 MHz | no downward drift |

Throttling stopped, and that alone recovered ~6 % prefill on Vulkan, ~8 % on ROCm and ~7 %
on CPU. **Numbers measured while this rig throttles are a floor, not a result.** Curve
Optimizer −15 later brought the peak to 86.6 °C; a repaste is still outstanding, and the
board throttle limit now sits at 99 °C, above the 95 °C stock Tjmax, so the usual safety
margin is gone.

> The CPU phase is not a strict A/B between those two sweeps: the earlier one still had the
> `-ngl 0` bug, so its "CPU" rows ran on the GPU. The Vulkan and ROCm phases are identical
> workloads in both and carry the comparison.

### Things that do nothing now

| Change | Measured |
| --- | --- |
| `power_dpm_force_performance_level=high` | 35B prefill 129.63 → 130.96, decode 19.11 → 19.01 — noise |
| `pp_power_profile_mode=5` (COMPUTE) | prefill 130.73, decode 19.19 — noise |
| `-DGGML_CUDA_FORCE_MMQ=ON` | prefill ±0.4 %, decode +1.8 % (noise) |
| iGPU max boost +200 MHz (BIOS) | `pp_dpm_sclk` top state 2000 → 2200 MHz, **observed clock under load still 2400 MHz** |
| IOMMU disabled (BIOS) | `/dev/kfd` and `rocminfo` unaffected; no speed change |

Clock pinning was worth +3 % in June and is worth nothing now — **because the cooling fix
removed the reason.** In June the GPU was throttling, so pinning the top DPM state helped;
SCLK now holds 2400 MHz on its own and there is nothing left to pin. It needs root and does
not survive a reboot, so it is not adopted.

`FORCE_MMQ` is a wash — neither the prefill regression expected from emulated dp4a nor any
gain. The default MMQ/cuBLAS auto-dispatch is fine on gfx900.

### Unattributed: ROCm's +22 % from the BIOS retune

Five BIOS changes were made together on 2026-09-07 (carve-out 2 → 16 GB, IOMMU off, iGPU
boost +200 MHz, Curve Optimizer −10 → −15, throttle limit 90 → 99 °C). Net on gemma at 4K:

| Backend | Prefill | Decode |
| --- | --- | --- |
| ROCm `-fa 0` | 87.16 → **106.13** (+22 %) | 10.63 → **11.89** (+12 %) |
| Vulkan `-fa 1` | 171.29 → 172.05 (+0.4 %) | 16.36 → **17.15** (+4.8 %) |
| CPU `-fa 1` | 88.37 → 90.13 (+2 %) | 12.05 → 12.26 (+1 %) |

Vulkan's gain is explained: the carve-out. **ROCm's is not** — ROCm never touches the
carve-out (see below), the iGPU boost changed no observed clock, and IOMMU changed nothing
measurable. Two settings remain (Curve Optimizer, throttle limit) and this run cannot
separate them. Recorded as unexplained rather than attributed.

> **Correction.** An earlier revision of this file claimed the carve-out was what moved
> ROCm, citing the run's VRAM peak rising from 1754 MB to 3645 MB. That figure was a
> whole-run peak taken during the *Vulkan* phase and attributed to ROCm without checking
> the per-phase split.

---

## Measurement traps

Five bugs that produced confident wrong numbers in this repo. Each is fixed; each is here
because the failure mode is not obvious.

| Trap | What it did | Caught by |
| --- | --- | --- |
| `-ngl 0` is not CPU-only | Every "CPU" row before 2026-09-07 ran on the GPU | A gemma CPU prefill figure above the CPU's arithmetic ceiling |
| `-fa auto` resolves to ON on ROCm | 57 % of prefill lost on every launch through `run-rocm7-baremetal.sh` | Explicit `-fa 0` / `-fa 1` / `-fa auto` A/B |
| `awk '{print gpu}'` → empty | ROCm benchmarks silently ran on CPU once the dGPUs left | The Vega becoming GPU 0 |
| Server harness vs `llama-bench` | An apparent 4–6 % Docker decode advantage that does not exist | `llama-bench` on baremetal beating the Docker harness figure |
| Unpinned llama.cpp checkouts | A 465e49b baremetal compared against a 67672dc image | Chasing the Docker gap |

### `-ngl 0` does not force CPU-only

Upstream changed the `-ngl` default to `auto`; with a GPU backend present the model is
still offloaded.

| Flag | GPU busy | GTT used |
| --- | --- | --- |
| `-ngl 0` | 91 % | 6567 MB |
| `-dev none` | idle | 157 MB |

Every CPU row dated before 2026-09-07 was produced with `-ngl 0`, so none is a CPU
measurement. Both the harness (`start_cpu()`) and `start-llama-server.sh --cpu` now use
`-dev none`.

**Confirmed two independent ways** (gemma, prefill t/s at the harness prompt sizes):

| Method | ~141 | ~937 | ~3330 |
| --- | ---: | ---: | ---: |
| `llama-bench -dev none -t 8 -r 2` | 99.31 ± 0.07 | 98.06 ± 0.27 | 93.07 ± 0.65 |
| harness (llama-server, `-dev none`) | 96.26 | 96.59 | 90.13 |
| *May 2026 (`-ngl 0`)* | *249.57* | *754.82* | *840.50* |

The two current methods share no code path beyond the model file and agree within 3 %,
across 8 and 16 threads, cold and warm start, on both models.

**Why the May figure cannot be a real CPU measurement.** gemma-4-E4B is 7.52 B parameters
(≈ 4 B effective). Prefill costs about `2 × N × T` operations, so 840 tok/s needs
6.7 TOP/s at 4 B active, or 12.6 TOP/s at 7.52 B. A 5700G's eight Zen 3 cores issue at most
two 256-bit `vpmaddubsw` per cycle per core = 128 int8 ops/cycle, i.e. **≈ 4.1 TOP/s peak
at 4 GHz**. The claimed number is 1.6–3× *above* the theoretical ceiling of the whole CPU,
while the measured 93 t/s sits at 34 % of it — a normal efficiency for real quantized GEMM.
A second check: at 840 t/s the CPU would be outrunning the iGPU (ROCm 106, Vulkan 172 on
the same model) by 5× on a compute-bound task, over the same memory bus.

> An earlier revision said the gap was "an order of magnitude" beyond the chip's ability.
> That was an overstatement: it is 1.6–3× above peak. The conclusion is unchanged.

The 35B May CPU row (58 / 197 / 211) is not provably impossible on arithmetic alone, but it
was produced the same way and did not reproduce (88 / 94 / 87 with `-dev none`). Both
models' pre-September CPU prefill rows are erroneous, not a regression. They are left in
the [history tables](#history), struck through, so the record of what was believed stays
intact.

### `-fa auto` resolves to ON on ROCm, and that costs 57 % of prefill

`-fa` defaults to `auto`, resolved by probing the backend for `FLASH_ATTN_EXT`. On gfx900
`ggml_cuda_get_best_fattn_kernel` falls through to the generic tile kernel
(`fattn.cu:652-666`), so the probe succeeds and FA is enabled. Measured on gemma at a
3330-token prompt, before `patches/0001`:

| `-fa` | Prefill t/s |
| --- | ---: |
| `0` | **112.78** |
| `1` | 48.91 |
| `auto` | 48.90 |

`run/run-rocm7-baremetal.sh` passed no `-fa` at all. It now passes one explicitly — `-fa 1`,
because that build carries `patches/0001`; `run/run-docker-rocm7.sh` still passes `-fa 0`,
because the Dockerfile does not apply the patch. Both remain overridable. The benchmark
harness was never affected — it always passed an explicit `-fa`.

> With `patches/0001` the *decode* verdict reverses (`-fa 1` wins everywhere) but the
> prefill penalty for FA on ROCm is gone rather than reversed: `-fa 1` is now marginally
> ahead on prefill too. The trap remains worth knowing because `auto` still resolves
> without regard to which build you are running.

### The server harness is not an instrument for few-percent comparisons

A rebuilt ROCm 7.2 Docker image appeared to beat baremetal by 4–6 % on 35B decode. Four
hypotheses, each tested:

| Hypothesis | Test | Result |
| --- | --- | --- |
| The image's extra env vars (`GPU_SINGLE_ALLOC_PERCENT`, `GPU_MAX_HEAP_SIZE`, `GPU_FORCE_64BIT_PTR`) | baremetal harness run with them exported | 18.76 / 17.99 / 14.72 vs 18.70 / 17.79 / 14.66 — **no effect** |
| Different llama.cpp commit (Docker 67672dc vs baremetal 465e49b) | rebuilt baremetal at 67672dc | 18.93 / 18.00 / 14.72 — **+0.2, does not close a 1.1 gap** |
| Container `--ulimit memlock=-1` vs the host's 8 MB | same binary, `llama-bench` with and without | 20.39 / 19.27 / 15.91 vs 20.30 / 19.27 / 15.90 — **no effect** |
| A real backend difference | `llama-bench -n 64 -d 128,1024,4096` on baremetal at Docker's commit | **20.30 / 19.27 / 15.90 — higher than the Docker harness figure of 19.84 / 18.83 / 15.30** |

The last row settles it: measured directly at matched KV depths, baremetal is *faster* than
the number the harness reports for Docker. Whatever produces the apparent advantage lives
in the llama-server measurement path (HTTP, slot handling, sampling, the `--no-warmup` cold
start), not in the HIP backend. **`llama-bench -d` is the trustworthy instrument for
decode** — which is why [the matrix](#the-matrix--2026-09-0809) uses it and the
[history tables](#history) are labelled by instrument.

Two lasting consequences: prefill matched to within noise on both models, confirming Docker
and baremetal compile the same sources against the same ROCm and perform the same; and
**all four build paths are now pinned** to one commit via `build/llama.cpp-ref`, read by
the baremetal script, the Vulkan script and both Dockerfiles. They had each been cloning
`master` independently, which is how a 465e49b baremetal and a 67672dc image came to be
compared in the first place.

FA ON collapsed prefill on Docker exactly as on baremetal (35B 4K: 41.4 vs 88.2), so that
was a property of the gfx900 kernel, not of the packaging.

---

## Where the model sits in memory

Sampled per backend phase, 16 GB carve-out:

| Model | Backend | VRAM (carve-out) | GTT |
| --- | --- | ---: | ---: |
| gemma-4-E4B | Vulkan | 3638 MB | 2872 MB |
| gemma-4-E4B | **ROCm** | **281 MB** | **3660 MB** |
| Qwen3.5-35B | Vulkan | **16354 MB** | 5013 MB |
| Qwen3.5-35B | **ROCm** | **311 MB** | **20787 MB** |
| Qwen3.5-35B | CPU | 304 MB | 67 MB |

**ROCm ignores the BIOS carve-out entirely** and maps the whole model through GTT,
regardless of how much carve-out exists. The mechanism is the `apu_prefer_gtt` rule in
amdgpu: when `AMD_IS_APU && real_vram_size < gtt_size`, every allocation is rewritten to
GTT. See [ARCHITECTURE.md](ARCHITECTURE.md).

The carve-out therefore benefits **Vulkan**, roughly in proportion to how much of the model
it can hold — which is why Vulkan gained ~5 % decode on the 5 GB gemma (already mostly
resident at a 2 GB carve-out) but 12–15 % on the 20 GB Qwen (16 GB moved out of GTT).

**Large models need the GTT kernel params.** `amdgpu.gttsize=65536 ttm.pages_limit=16777216`
is mandatory for the 20 GB model; without them a fresh install leaves ~30 GB GTT and the
allocation overflows, hard-freezing the machine within ~3 s of load. Do **not** set
`amdgpu.cwsr_enable=0` — with it the 35B crashes during load.

---

## History

Two instruments, never mixed:

- **harness** — `bench/run-all-benchmarks.sh` → llama-server + `bench/test-server-perf.py`,
  `-c 8192`, prompts ~141 / ~937 / ~3330 tokens, 50 decode tokens, `--no-warmup`.
  All rows below.
- **llama-bench** — direct, `-p`/`-n -d`. [The matrix](#the-matrix--2026-09-0809) only.

The two are not comparable at the few-percent level; see
[the trap above](#the-server-harness-is-not-an-instrument-for-few-percent-comparisons).

### Qwen3.5-35B-A3B-Q4_K_M — harness history

| Date | Configuration | Backend | FA | Pf 141 | Pf 937 | Pf 3330 | TG 141 | TG 937 | TG 3330 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-09-08 | 16 GB carve, `-ub 4096` | **Vulkan** | **ON** | 63.25 | 164.54 | **190.20** | **21.15** | **21.15** | **20.76** |
| 2026-09-08 | | Vulkan | OFF | 63.22 | 163.83 | 187.21 | 21.29 | 21.02 | 18.43 |
| 2026-09-08 | | ROCm 7.2 Docker | OFF | 48.09 | 122.91 | 141.99 | 20.19 | 19.11 | 15.52 |
| 2026-09-08 | | ROCm 7.2 baremetal | OFF | 44.22 | 121.77 | 141.00 | 18.50 | 17.85 | 14.70 |
| 2026-09-08 | | ROCm 7.2 Docker | ON | 41.89 | 76.10 | 53.34 | 19.92 | 18.53 | 15.32 |
| 2026-09-08 | | ROCm 7.2 baremetal | ON | 45.61 | 76.01 | 53.47 | 18.73 | 17.48 | 14.58 |
| 2026-09-08 | CPU rows at default `-ub` | CPU (`-dev none`) | ON | 84.11 | 90.77 | 86.38 | 18.39 | 17.94 | 15.03 |
| 2026-09-08 | | CPU (`-dev none`) | OFF | 83.23 | 90.39 | 85.88 | 18.32 | 18.12 | 17.11 |
| 2026-09-07 | 16 GB carve, `-ub 512` | Vulkan | ON | 73.35 | 159.06 | 153.67 | 21.73 | 21.56 | 20.95 |
| 2026-09-07 | | Vulkan | OFF | 66.89 | 157.57 | 155.91 | 21.30 | 21.21 | 18.60 |
| 2026-09-07 | | ROCm 7.2 baremetal | OFF | 47.73 | 94.48 | 88.96 | 18.70 | 17.79 | 14.66 |
| 2026-09-07 | | ROCm 7.2 baremetal | ON | 43.60 | 67.60 | 41.53 | 18.74 | 17.55 | 14.65 |
| 2026-09-07 | | CPU (`-dev none`) | ON | 88.31 | 94.33 | 87.46 | 18.39 | 17.95 | 14.36 |
| 2026-09-07 | | CPU (`-dev none`) | OFF | 86.07 | 91.48 | 86.54 | 18.25 | 18.14 | 17.05 |
| 2026-05-15 | 2 GB carve, 32 GB GTT, dGPUs present | Vulkan | ON | 65.00 | 138.57 | 137.11 | 19.06 | 18.95 | 18.47 |
| 2026-05-15 | | Vulkan | OFF | 64.73 | 138.08 | 136.44 | 18.88 | 18.49 | 16.35 |
| 2026-05-15 | | ROCm 7.2 baremetal | OFF | 42.47 | 72.61 | 71.65 | 16.67 | 15.85 | 13.03 |
| 2026-05-15 | | ROCm 7.2 baremetal | ON | 40.48 | 55.79 | 37.55 | 16.53 | 15.50 | 13.17 |
| 2026-05-15 | `-ngl 0` — **not CPU-only** | CPU | ON | ~~58.18~~ | ~~196.63~~ | ~~210.79~~ | 17.04 | 16.65 | 13.59 |
| 2026-05-15 | `-ngl 0` — **not CPU-only** | CPU | OFF | ~~59.33~~ | ~~186.34~~ | ~~208.17~~ | 17.01 | 16.77 | 15.94 |
| 2026-05-14 | 2 GB carve, 64 GB GTT | ROCm 7.2 baremetal | OFF | 40.55 | 68.18 | 67.32 | 14.50 | 14.30 | 11.88 |
| 2026-05-14 | | ROCm 7.2 Docker | OFF | 38.63 | 70.41 | 68.87 | 15.49 | 15.06 | 12.43 |
| 2026-05-14 | | ROCm 6.2.4 Docker | OFF | 40.15 | 64.43 | 63.97 | 14.40 | 13.82 | 11.64 |
| 2026-05-14 | `-ngl 0` — **not CPU-only** | CPU | ON | ~~57.07~~ | ~~215.04~~ | ~~233.12~~ | 15.67 | 15.39 | 12.95 |
| 2026-04-12 | 2 GB carve, 64 GB GTT | Vulkan | default | 44.58 | 50.62 | 50.00 | 19.90 | 20.46 | 19.76 |
| 2026-04-12 | LM Studio, not directly comparable | LM Studio Vulkan | UI | 48.84 | 137.85 | 157.60 | 19.58 | 19.05 | 18.05 |

### gemma-4-E4B-it-Q4_K_M — harness history

| Date | Configuration | Backend | FA | Pf 141 | Pf 937 | Pf 3330 | TG 141 | TG 937 | TG 3330 |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2026-09-08 | 16 GB carve, `-ub 4096` | ROCm 7.2 Docker | OFF | 73.13 | 111.46 | **192.25** | 16.81 | 15.16 | 10.47 |
| 2026-09-08 | | **ROCm 7.2 baremetal** | OFF | 69.94 | 110.62 | **192.29** | 15.91 | 14.43 | 10.23 |
| 2026-09-08 | | **Vulkan** | **ON** | **126.57** | **173.41** | 170.20 | **18.24** | **17.96** | **16.83** |
| 2026-09-08 | | Vulkan | OFF | 101.02 | 152.28 | 141.85 | 18.16 | 17.07 | 13.28 |
| 2026-09-08 | | ROCm 7.2 Docker | ON | 66.98 | 47.39 | 30.15 | 16.66 | 15.34 | 12.67 |
| 2026-09-08 | | ROCm 7.2 baremetal | ON | 65.28 | 48.17 | 30.13 | 16.02 | 14.79 | 12.15 |
| 2026-09-08 | CPU rows at default `-ub` | CPU (`-dev none`) | ON | 96.05 | 94.31 | 87.83 | 15.08 | 14.24 | 11.96 |
| 2026-09-08 | | CPU (`-dev none`) | OFF | 95.25 | 93.73 | 85.19 | 14.97 | 14.32 | 13.40 |
| 2026-09-08 | 16 GB carve, `-ub 512`, Docker re-verify | ROCm 7.2 Docker | OFF | 69.98 | 109.49 | 106.04 | 16.06 | 14.62 | 11.97 |
| 2026-09-08 | | ROCm 7.2 Docker | ON | 67.55 | 53.83 | 29.41 | 16.86 | 15.56 | 13.07 |
| 2026-09-07 | 16 GB carve (BIOS retune), `-ub 512` | Vulkan | ON | 127.11 | 171.57 | 172.05 | 18.39 | 17.99 | 17.15 |
| 2026-09-07 | | Vulkan | OFF | 101.36 | 153.07 | 156.85 | 18.10 | 17.14 | 14.91 |
| 2026-09-07 | | ROCm 7.2 baremetal | OFF | 69.99 | 109.10 | 106.13 | 15.93 | 14.48 | 11.89 |
| 2026-09-07 | | ROCm 7.2 baremetal | ON | 65.36 | 53.72 | 29.47 | 16.05 | 14.83 | 12.56 |
| 2026-09-07 | | CPU (`-dev none`) | ON | 96.26 | 96.59 | 90.13 | 15.18 | 14.36 | 12.26 |
| 2026-09-07 | | CPU (`-dev none`) | OFF | 95.68 | 94.96 | 86.26 | 14.96 | 14.46 | 13.46 |
| 2026-09-07 | 2 GB carve — post-reinstall verification | Vulkan | ON | 124.23 | 170.73 | 171.29 | 17.55 | 17.16 | 16.36 |
| 2026-09-07 | | Vulkan | OFF | 95.60 | 149.48 | 155.30 | 17.19 | 16.39 | 14.29 |
| 2026-09-07 | | ROCm 7.2 baremetal | OFF | 56.46 | 89.10 | 87.16 | 14.28 | 12.91 | 10.63 |
| 2026-09-07 | | ROCm 7.2 baremetal | ON | 55.23 | 47.36 | 27.11 | 14.32 | 13.31 | 11.44 |
| 2026-09-07 | first genuine CPU-only rows | CPU (`-dev none`) | ON | 95.38 | 94.66 | 88.37 | 15.05 | 14.24 | 12.05 |
| 2026-09-07 | | CPU (`-dev none`) | OFF | 94.88 | 93.23 | 84.69 | 14.95 | 14.39 | 13.44 |
| 2026-05-15 | 2 GB carve, 32 GB GTT, dGPUs present | Vulkan | ON | 121.97 | 166.81 | 160.02 | 16.97 | 16.56 | 15.84 |
| 2026-05-15 | | Vulkan | OFF | 91.11 | 143.58 | 147.37 | 16.64 | 15.68 | 13.62 |
| 2026-05-15 | | ROCm 7.2 baremetal | OFF | 70.91 | 84.88 | 84.95 | 14.50 | 13.15 | 10.89 |
| 2026-05-15 | | ROCm 7.2 baremetal | ON | 65.77 | 47.70 | 27.67 | 14.69 | 13.66 | 11.68 |
| 2026-05-15 | `-ngl 0` — **not CPU-only** | CPU | ON | ~~249.57~~ | ~~754.82~~ | ~~840.50~~ | 15.06 | 14.24 | 12.17 |
| 2026-05-15 | `-ngl 0` — **not CPU-only** | CPU | OFF | ~~235.17~~ | ~~693.91~~ | ~~772.09~~ | 14.83 | 14.41 | 13.36 |
| 2026-05-14 | 16 GB carve, no 64 GB GTT | Vulkan | ON | 115.76 | 158.83 | 156.18 | 14.73 | 14.55 | 14.03 |
| 2026-05-14 | | ROCm 7.2 baremetal | OFF | 67.96 | 81.96 | 81.77 | 13.32 | 12.19 | 10.00 |
| 2026-05-14 | `-ngl 0` — **not CPU-only** | CPU | ON | ~~257.48~~ | ~~810.80~~ | ~~923.35~~ | 12.33 | 11.90 | 10.10 |
| 2026-05-14 | 2 GB carve, 64 GB GTT | ROCm 7.2 Docker | OFF | 67.56 | 80.73 | 80.74 | 13.13 | 11.94 | 9.77 |
| 2026-05-14 | | ROCm 6.2.4 Docker | OFF | 66.97 | 80.33 | 79.98 | 10.58 | 9.74 | 8.32 |

### June 2026 ROCm tuning sweep

35B on the ROCm 7.2 Docker image, `-ngl 99 -fa 0 -c 8192`, harness
`bench/tune-rocm7-vega.sh`. Superseded by the `-ub` and `-ctk` sections above, which
measured the same knobs at long context and reached different conclusions about their size.

| Config | Pf 140 | Pf 937 | Pf 3330 | TG 140 | TG 937 | TG 3330 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline (`-ub 512`) | 41.0 | 69.8 | 68.9 | 16.0 | 15.2 | 12.6 |
| `-ub 256` | 41.0 | 54.4 | 54.3 | — | — | — |
| `-ub 1024` | 41.1 | 81.1 | 78.6 | — | — | — |
| **`-ub 2048`** | 41.0 | 80.6 | **84.0** | 16.0 | 15.2 | 12.6 |
| `-ctk q8_0` | 41.2 | 69.1 | 67.9 | 15.7 | 15.3 | **13.0** |
| `-ub 2048` + clocks `high` | — | 82.8 | 86.9 | — | 15.7 | 12.7 |
| `-ub 2048` + `FORCE_MMQ` | — | 83.1 | 86.9 | — | — | 13.0 |

`-ub 256` *hurts* — it under-fills the 8-CU GEMMs. Decode was flat across every config,
which at the time read as "DDR4-bandwidth-bound, as predicted"; `patches/0001` later showed
decode was attention-bound instead, and lifted it 157 % at 32K without touching bandwidth.

### Environment changes between runs

| When | Change |
| --- | --- |
| April–May 2026 | RTX 5090 + R9700 present; Vega 8 at `renderD129`, ROCm index 1 |
| June 2026 | 5090 removed, second R9700 added → Vega 8 at `renderD130`, ROCm index 2. Classic ROCm 7.2 replaced by modular `amdrocm-core` 7.13/7.14, which broke the baremetal rows |
| September 2026 | Both R9700s moved to another machine; OS reinstalled as Ubuntu 26.04 / kernel 7.0, classic ROCm 7.2.0 restored. Vega 8 is the **only** GPU: `card0` / `renderD128` / ROCm index **0** |

May rows also ran with `HSA_XNACK=1` / `HSA_ENABLE_SDMA=1`; both are now `0` (XNACK=1 can
freeze the machine), and the runner auto-detects the Vega index rather than hardcoding it.
Treat pre-September rows as historical baselines, not as current numbers.

---

## How to run

```bash
./bench/run-all-benchmarks.sh 2>&1 | tee /tmp/bench-$(date +%Y%m%d-%H%M).log
```

Starts each backend sequentially, waits for `/health`, runs `bench/test-server-perf.py`,
writes per-backend CSVs, then prints model-grouped summary tables. Overridable with
`BENCH_MODELS`, `BENCH_BACKENDS`, `BENCH_BATCH`, `BENCH_UBATCH`.

For anything compared at the few-percent level, use `llama-bench` directly instead:

```bash
llm/rocm7-vega/bin/llama-bench -m <model> -ngl 99 -fa 1 -b 2048 -ub 2048 \
    -p 4096,16384,32768 -n 0
llm/rocm7-vega/bin/llama-bench -m <model> -ngl 99 -fa 1 -n 32 -d 4096,16384,32768 -p 0
```

Thermals during a run: `bench/log-thermals.sh` (CSV, `TJMAX` configurable).

---

## Archived detail notes

### ROCm 7.2 / gfx900 backport

ROCm 7.2 on Vega 8 depends on gfx900-compatible libraries and the lazy rocBLAS index. The
critical runtime file is `TensileLibrary_lazy_gfx900.dat`; without it, rocBLAS can fail on
first GEMM with `Illegal seek for GPU arch: gfx900`. The project build copies the required
`*gfx900*` files and lazy index into the ROCm 7 layer/install.
