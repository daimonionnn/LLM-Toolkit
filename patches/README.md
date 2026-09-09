# Local llama.cpp patches

Applied by the build scripts after checking out the pinned commit from
[`build/llama.cpp-ref`](../build/llama.cpp-ref), in filename order.

Each patch must be justified by a measurement recorded in
[`docs/benchmarks.md`](../docs/benchmarks.md), and must be re-validated whenever
the pin moves — `git apply --check` fails loudly rather than silently skipping.

## 0001-ggml-cuda-mad-gfx900-mad-mix.patch

Uses `v_mad_mix_f32` for the flash-attention KQ accumulate on gfx900 / gfx90c.

**Problem.** `V_DOT2_F32_F16_AVAILABLE` is defined only for RDNA2+, gfx906 and
CDNA, so `ggml_cuda_mad(float&, half2, half2)` on GCN5 falls back to
`__half22float2(v*u); acc += tmp.x + tmp.y` — `v_pk_mul_f16` + 2× `v_cvt_f32_f16`
+ 2× `v_add_f32`, i.e. **5 VALU ops per 2 MACs**, forming the product in fp16.
The intermediates also cost registers: the FA tile kernels sat at the 128-VGPR
cap and spilled, 50 of 60 config rows, up to 2262 VGPRs with 2.9 KB/lane of
scratch.

**Fix.** gfx900 has the VOP3P mad-mix family (LLVM feature `mad-mix-insts`):
f16 × f16 + f32 in one instruction, product computed in fp32. Two ops with
`op_sel` cover the half2 — **1 VALU op per MAC**.

Written as inline asm because the compiler only selects mad-mix from a mul+add
expression when *both* `-fgpu-flush-denormals-to-zero` and `-ffp-contract=off`
are set, and those are global flags that would change every other kernel.

**Effect.** Spills across the 256×256 FA rows drop **10 598 → 6**, VGPR use from
128 to 76–130, and some kernels reach occupancy **3** — better than the config
table asks for. Decode t/s at KV depth, `llama-bench -n 32 -d -ub 512`:

| Model | Depth | best before (`-fa 0`) | after (`-fa 1`) | Gain |
| ----- | ----- | --------------------: | --------------: | ---: |
| Qwen3.5-35B | 4 096 | 15.12 | 18.68 | +24 % |
| Qwen3.5-35B | 32 768 | 6.16 | **15.86** | **+157 %** |
| gemma-4-E4B | 4 096 | 13.56 | 15.44 | +14 % |
| gemma-4-E4B | 32 768 | 7.61 | **12.49** | **+64 %** |

It removes the anomalous long-context decode collapse: ROCm fell 66 % from 1K to
32K before, ~21 % after, matching Vulkan. The gap to Vulkan at 32K goes from
178 % to **8 %**.

It is also *more accurate*: on 256 random half2 pairs against an fp64 reference,
the old path errs by 6.1e-3 and mad-mix by 1.4e-6, because the product is formed
in fp32 rather than fp16.

**Validation.** `test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0`: 2959/2959
passed. Numerics also checked directly against an fp64 reference before the patch
was written.

**Not tested:** FA shapes other than the 256×256 (Qwen) and 512×512 (gemma) rows
these two models use, and non-FA callers of `ggml_cuda_mad(float&, half2, half2)`.

### A superseded approach, kept as a note

The first attempt gave GCN its own FA tile *occupancy* (2 → 1), lifting the VGPR
cap from 128 to 256. It worked — 35B decode at 32K went 6.16 → 14.85 — and the
mechanism was correctly identified: the config table is shared with CDNA, which
has AGPRs to spill into and GCN5 does not.

But it treated the symptom. The kernels were short of registers because the fp16
fallback materialised intermediates that need not exist; removing them dropped
VGPR use far below the cap on its own. Measured head to head at 32K on the 35B:

| | t/s |
| --- | ---: |
| occupancy only | 14.85 |
| **mad-mix only** | **15.86 ± 0.02** |
| both | 15.41 |

Occupancy 1 *on top of* mad-mix is a net loss — it drags kernels that now reach
occupancy 3 back down to 1, costing latency hiding for no benefit. The occupancy
patch was therefore dropped.

**Upstream:** this is a general GCN5 fix, not a local workaround. It belongs
upstream; until then it lives here.
