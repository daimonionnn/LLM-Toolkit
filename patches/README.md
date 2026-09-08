# Local llama.cpp patches

Applied by the build scripts after checking out the pinned commit from
[`build/llama.cpp-ref`](../build/llama.cpp-ref), in filename order.

Each patch must be justified by a measurement recorded in
[`docs/benchmarks.md`](../docs/benchmarks.md), and must be re-validated whenever
the pin moves — `git apply --check` fails loudly rather than silently skipping.

## 0001-fattn-tile-gcn-occupancy.patch

Gives GCN5 (gfx900 / gfx90c) its own flash-attention tile occupancy.

**Problem.** `ggml_cuda_fattn_tile_get_config` routes every non-RDNA AMD part to
a table shared with CDNA. CDNA has 256-512 AGPRs a kernel can spill into
cheaply; GCN5 has none — a SIMD holds 256 VGPRs and nothing else. The table's
`occupancy = 2` therefore caps every kernel at 128 VGPRs and the remainder goes
to scratch memory. Measured on gfx900 with `-Rpass-analysis=kernel-resource-usage`:
**50 of 60 table rows spill**, up to 2262 VGPRs with 2.9 KB/lane of scratch.

**Fix.** Take the shared table and rewrite only the occupancy field, so the
tuning stays in one place and CDNA is untouched. Two dispatch sites gain a GCN
branch (host `GGML_CUDA_CC_IS_GCN`, device `#elif defined(GCN)`).

**Effect.** Decode t/s at KV depth, `llama-bench -n 32 -d`, ROCm 7.2 on Vega 8:

| Model | Depth | best before (`-fa 0`) | after (`-fa 1`) | Gain |
| ----- | ----- | --------------------: | --------------: | ---: |
| Qwen3.5-35B | 4 096 | 15.12 | 18.54 | +23 % |
| Qwen3.5-35B | 32 768 | 6.16 | **14.87** | **+141 %** |
| gemma-4-E4B | 4 096 | 13.56 | 15.44 | +14 % |
| gemma-4-E4B | 32 768 | 7.61 | **12.31** | **+62 %** |

It removes the anomalous long-context decode collapse: ROCm fell 66 % from 1K to
32K before, 22 % after — Vulkan falls 21 %, so the curves now match. The gap to
Vulkan at 32K goes from 178 % to 13 %.

**Validation.** `test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0`: 2959/2959
passed. `-fa 0` numbers are byte-identical before and after (13.56 / 7.60 on
gemma), confirming nothing outside the FA path moved.

**Not tested:** rows other than the 256×256 (Qwen) and 512×512 (gemma) shapes
these two models use. The mechanism — no AGPRs on GCN5 — applies to all of them,
but only these were benchmarked.

**Upstream:** this is a general GCN5 fix, not a local workaround. It belongs
upstream; until then it lives here.
