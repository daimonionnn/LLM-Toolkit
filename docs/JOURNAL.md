# How this project was built

A narrative log, April 2026 to today. The other documents say what the state
*is*; this one says how it got there, and — more usefully — which things this
project believed and measured and wrote down that later turned out to be false.

That last part is why the file exists. Almost every hard-won result here began
as a correction to something already committed with confidence. If you take one
thing from this document, take the [pattern section](#what-kept-going-wrong) at
the end rather than the chronology.

---

## April 2026 — getting anything to run at all

The Vega 8 in a Ryzen 5700G reports as **gfx90c**, which no AMD software stack
supports. The ISA is identical to **gfx900** (Vega 10, the discrete Vega 56/64),
so `HSA_OVERRIDE_GFX_VERSION=9.0.0` makes the runtime treat it as gfx900 and the
kernels execute correctly. That single fact is the foundation everything else
sits on.

It is not sufficient on its own. rocBLAS ships *prebuilt per-architecture*
kernels, and modern packages contain no gfx9 at all, so the override succeeds
and then the first GEMM fails with `Illegal seek for GPU arch: gfx900`. The fix
that became this repo's central technique: copy the gfx900 kernel files and
`TensileLibrary_lazy_gfx900.dat` out of the **ROCm 6.3.4** package into the
newer install.

Two hardware landmines were found immediately and both are still in the docs
because both are hard lockups rather than error messages:

- **`HSA_XNACK=1` freezes the entire machine** on this APU. Set it to 0.
- **Large models need the GTT kernel params.** Without
  `amdgpu.gttsize=65536 ttm.pages_limit=16777216` the iGPU gets ~30 GB of GTT,
  a 20 GB model overflows it, and the PC hard-freezes within about three
  seconds of loading.

Docker with ROCm 6.2.4 was the first path that worked end to end.

## May 2026 — ROCm 7.2 baremetal, and the first benchmark tables

The backport was made to work against classic ROCm 7.2 packages on the host, and
the first full benchmark matrix was recorded: every backend, both models, three
prompt sizes.

Those tables were wrong in a way nobody noticed for four months. See
[September](#september-2026--the-reinstall-and-the-reckoning).

## June 2026 — two discrete GPUs arrive and break everything

Two Radeon AI PRO R9700s were installed, which meant AMD's **modular**
`amdrocm-core` 7.13/7.14 packages replaced the classic ones. Baremetal ROCm on
the Vega died on two counts at once: the modular runtime **rejected the
override** with `HSA_STATUS_ERROR_OUT_OF_RESOURCES`, and the packages shipped no
gfx9 kernels for the backport to work with either.

The dGPUs also renumbered the Vega — `renderD129` → `renderD130`, ROCm index 1 →
2 — which exposed a detection bug that had been latent since April and would
stay latent for another three months. See the pattern section.

The one durable gain from this period was tuning: `-ub 2048` was found to be
worth about +22 % prefill over the upstream default of 512.

## July–August 2026 — dormant

Roadmap items, hardware documentation, a licence. No measurements.

## September 2026 — the reinstall and the reckoning

The R9700s moved to another machine, leaving the Vega 8 as the only GPU, and the
OS was reinstalled as Ubuntu 26.04 / kernel 7.0. Restoring the stack was
routine. What followed was not.

### Three separate mechanisms had been silently benchmarking the CPU

Found within about a day of each other, all three producing *plausible numbers*
rather than errors:

1. **`-ngl 0` does not mean "no GPU offload".** Upstream changed the `-ngl`
   default to `auto`, and with a GPU backend present the model is still
   offloaded — measured 91 % GPU busy and 6.5 GB in GTT with `-ngl 0`, versus
   157 MB with `-dev none`. **Every CPU row recorded before 2026-09-07 was a GPU
   measurement.** The May figure of 840 t/s CPU prefill on gemma was not merely
   optimistic: it is 1.6–3× above the arithmetic ceiling of the whole CPU.
2. **`awk '{print gpu}'`** in the benchmark harness emitted an *empty string*
   when the Vega was GPU 0, which became `ROCR_VISIBLE_DEVICES=""`, which hides
   every GPU and falls back to CPU while reporting success. Invisible until the
   dGPUs left and the Vega became index 0 for the first time.
3. **`GGML_BACKEND_DL=ON` fails silently.** If `libggml-hip.so` cannot resolve
   its dependencies at run time, llama-bench prints `backend: CPU` and a
   perfectly reasonable number. This one was found in September 2026 on the
   ROCm 7.14 SDK experiment, months after the other two.

### `-fa auto` cost 57 % of prefill on every ROCm launch

`-fa` defaults to `auto`, which probes the backend. On gfx900 the probe finds
the generic flash-attention tile kernel compiles, so FA is enabled — at the time
the worst possible setting. Measured on gemma at a 3330-token prompt: `-fa 0` =
112.8 t/s, `-fa 1` = 48.9, **`-fa auto` = 48.9**. The launcher passed no `-fa`
at all, so anyone using it was getting 43 % of achievable prefill.

### The BIOS carve-out did not do what the commit message said

A BIOS retune moved five settings at once and ROCm gained 22 % prefill. It was
attributed to the 16 GB UMA carve-out, citing a VRAM peak that rose from 1754 MB
to 3645 MB. Per-phase sampling later showed **ROCm never uses the carve-out at
all** — 281 MB of VRAM against 3660 MB of GTT — and the figure quoted had been
recorded during the *Vulkan* phase. The kernel reason was traced afterwards:
`apu_prefer_gtt` rewrites every allocation to GTT when
`AMD_IS_APU && real_vram_size < gtt_size`. The real cause of ROCm's +22 % is
still unexplained; four other settings changed simultaneously.

### Flash attention on gfx900, fixed with one instruction

`V_DOT2_F32_F16_AVAILABLE` is defined only for RDNA2+, gfx906 and CDNA, so on
GCN5 the FA KQ accumulate falls back to `v_pk_mul_f16` + 2× `v_cvt_f32_f16` +
2× `v_add_f32` — **5 VALU ops per 2 MACs**, with the product formed in fp16.
Those intermediates also pushed 50 of 60 FA config rows into register spilling.

gfx900 has the VOP3P mad-mix family: f16 × f16 + f32 in one instruction. Two
`v_mad_mix_f32` with `op_sel` cover a half2, at **1 op per MAC**.

| | Before | After |
| --- | ---: | ---: |
| VALU ops per MAC | 2.5 | 1 |
| Spills, 256×256 FA rows | 10 598 | 6 |
| 35B decode at 32K | 6.16 t/s | **15.86 t/s** |
| Error vs fp64 | 6.1e-3 | 1.4e-6 |

`test-backend-ops -o FLASH_ATTN_EXT`: 2959/2959. This became `patches/0001`.

An earlier attempt had raised FA tile *occupancy* instead, taking 32K decode
from 6.16 to 14.85 — a real improvement with a correctly identified mechanism,
which turned out to be treating a symptom. Occupancy 1 *on top of* mad-mix is a
net loss. It was dropped.

### The micro-batch, and a rule fitted to one model

`-ub 4096` is worth +23–55 % prefill on a dense model and +3–9 % on an MoE — the
difference being that with 8 of 256 experts active, `-ub 2048` already fills
MMQ's 64-column tiles and nothing above it helps. `-ub 8192` loses everywhere.

Along the way, Vulkan was found to hang the GPU compute ring on large attention
dispatches. The first cap derived from it, `ctx × ubatch > 2²⁶`, was fitted to
the 35B alone and called gemma at 32K/`-ub 2048` safe — a configuration that
crashes. The corrected rule includes head dimension.

Then the variable itself turned out to be wrong. Holding the allocation at 128K
and varying only the prompt: 61 tokens fine, 6021 fine, ~16 000 wedges the ring.
**It tracks `n_kv` — tokens actually in the cache — not the allocated context.**
Every prior measurement had filled the context it allocated, so the two moved
together and the data could not tell them apart. This is also why LM Studio runs
gemma at a 128K context without trouble: a chat turn never fills it.

### An outside contribution, and PyTorch

[Issue #1](https://github.com/daimonionnn/amd-vega-rocm-vulkan-llm-toolkit/issues/1)
from **@Josephur** confirmed the rocBLAS backport on discrete Vega10 (V340,
MI25-class) with numerically verified SGEMM, and surveyed the rest of the ROCm
stack — the part this repo had no hardware to do. It corrected an assumption
carried implicitly since April: that the whole ML stack hits the same
prebuilt-kernel wall as rocBLAS. It does not; MIOpen and rocFFT JIT-compile.

Chasing it produced a package survey showing the dividing line is the **ROCm
version, not the library** — 6.3.4 is the last release compiling consumer gfx9,
and 7.x dropped it everywhere at once — and then the practical payoff:

**PyTorch runs on this iGPU.** `torch 2.7.0+rocm6.3`, all four checks passing
against CPU references, 1.40 TFLOP/s fp32. No backport and no source build: the
wheel bundles its own gfx900 rocBLAS, rocRAND and MIOpen, so the override is the
whole trick. rocRAND, which had been named as the blocker, was never one on this
path.

### A dead end worth recording

AMD publishes a complete ROCm 7.14 built for gfx900 as pip wheels, which looked
like it could retire the backport entirely. llama.cpp builds against it cleanly.
The runtime segfaults during GPU enumeration on this APU — with and without the
override — so it is the same wall as June 2026, failing less gracefully. The
backport stays. Written up in
[bench/results/2026-09-11-rocm714-sdk-gfx900.md](../bench/results/2026-09-11-rocm714-sdk-gfx900.md).

---

## What kept going wrong

Four patterns account for nearly every correction above. They are worth more
than the chronology.

### 1. Silent fallback to the CPU, by three different routes

`-ngl 0` offloading anyway, an awk bug emptying `ROCR_VISIBLE_DEVICES`, and a
dlopened backend failing to resolve its libraries. Three independent mechanisms,
all producing believable numbers, one of them undetected for four months.

**The lesson:** never accept a performance number without confirming *which
device produced it*. Check the backend column, check GPU busy, check GTT
residency. A benchmark that cannot fail loudly will fail quietly.

### 2. `cmd | grep -q` under `set -o pipefail`

grep exits on the first match, the writer takes SIGPIPE, and the whole script
dies. This recurred **four separate times**, including twice in code written to
fix earlier instances of it. There is now a CI check for the idiom.

### 3. Generalising from a single measurement

`UBATCH=4096` was made the default after testing only at `-c 8192`. The `2²⁶`
cap was fitted to one model and declared safe for a configuration that crashes.
The `-ub 8192` probe was justified by a "still climbing" curve measured
pre-patch under a different FA setting.

**The lesson:** a rule fitted to one point is a guess with a number attached.
State which measurements a rule rests on, and it becomes obvious when it is
being over-extended.

### 4. Attribution without isolation

Five BIOS settings changed at once and the gain was attributed to the most
interesting one — using a figure that belonged to a different backend's phase.

**The lesson:** if several things moved, the honest write-up is "unexplained".
That entry is still open in the TODO, and it should be.

### And one that worked

Every result that survived was one where the *mechanism* was identified, not
just the correlation: `apu_prefer_gtt` in the kernel source explaining GTT
residency, `V_DOT2_F32_F16_AVAILABLE` explaining the FA collapse, MMQ's
64-column tiles explaining why dense and MoE models respond differently to the
micro-batch. Where only a number was available, the number later moved.
