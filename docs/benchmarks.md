# LLM Inference Benchmarks

## Benchmark: Qwen3.5-35B-A3B — ROCm (Docker)

**Date:** 2026-04-12
**Tool:** `test-server-perf.py` — three context sizes: 128, 1024, 4096 tokens

### Hardware

| Component | Details |
|-----------|---------|
| GPU | AMD Radeon Graphics (Vega 8 iGPU, gfx900:xnack-) |
| VRAM | 65536 MiB (64 GB GTT — system RAM mapped via UMA) |
| Backend | ROCm 6.2.4 (Docker: `rocm/dev-ubuntu-24.04:6.2.4`) |

### Model

| Field | Value |
|-------|-------|
| Model | `lmstudio-community/Qwen3.5-35B-A3B-GGUF` |
| File | `Qwen3.5-35B-A3B-Q4_K_M.gguf` |
| Quantization | Q4_K_M |
| Size | ~20 GB |
| Layers offloaded | 41/41 (full GPU offload) |
| GPU model buffer | 19905 MiB (~19.5 GB) |
| CPU mapped buffer | 273 MiB |

### Server Configuration

```bash
./run-docker-rocm.sh .../Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -c 2048 --no-warmup
```

| Setting | Value |
|---------|-------|
| `HSA_XNACK` | `0` (prevents PC freeze on Vega 8) |
| `GGML_HIP_UMA` | `0` (required with XNACK=0) |
| `HSA_OVERRIDE_GFX_VERSION` | `9.0.0` |
| `GPU_MAX_ALLOC_PERCENT` | `100` |
| Context (`-c`) | `2048` |
| GPU layers (`-ngl`) | `99` |
| Server host | `0.0.0.0:8080` |

### Results

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
|------------------------------|-----------|---------------------|------------------------|
| 140 tokens                   | 128       | 38.40               | 14.23                  |
| 937 tokens                   | 1024      | 51.33               | 13.49                  |
| 3330 tokens                  | 4096      | 36.98               | 11.52                  |

### Observations

- **Generation throughput** is consistent at ~11–14 t/s across all context sizes — usable for interactive chat
- **Prefill peaks at ~1024 tokens** (51.33 t/s) — likely optimal batch size for the Vega 8 compute units
- **Prefill drops at large context** (3330 tokens → 37 t/s) — memory bandwidth becomes the bottleneck as the KV cache fills
- Full 41-layer offload achieved with 20 GB model into 64 GB GTT — only possible after GRUB `amdgpu.gttsize=65536` fix
- Host ROCm stack (Ubuntu HIP 5.7.1 + Clang-21) segfaults at slot init; Docker ROCm 6.2.4 resolves this completely

---

## Benchmark: Qwen3.5-35B-A3B — Vulkan (Native, Vega 8)

**Date:** 2026-04-12
**Tool:** `test-server-perf.py` — three context sizes: 128, 1024, 4096 tokens

### Hardware

| Component | Details |
|-----------|---------|
| GPU | AMD Radeon Graphics (Vega 8 iGPU, gfx90c) |
| VRAM | 65536 MiB (64 GB GTT — system RAM mapped via UMA) |
| Backend | Vulkan (Mesa RADV, native — `run-llamaserver-vulkan.sh`) |

### Model

| Field | Value |
|-------|-------|
| Model | `lmstudio-community/Qwen3.5-35B-A3B-GGUF` |
| File | `Qwen3.5-35B-A3B-Q4_K_M.gguf` |
| Quantization | Q4_K_M |
| Size | ~20 GB |

### Server Configuration

```bash
./run-llamaserver-vulkan.sh .../Qwen3.5-35B-A3B-Q4_K_M.gguf -ngl 99 -c 2048
```

### Results

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
|------------------------------|-----------|---------------------|------------------------|
| 140 tokens                   | 128       | 44.58               | 19.90                  |
| 937 tokens                   | 1024      | 50.62               | 20.46                  |
| 3330 tokens                  | 4096      | 50.00               | 19.76                  |

### Observations

- **Generation throughput** is stable at ~19–20 t/s across all context sizes — notably more consistent than ROCm Docker
- **Prefill is flat** across all context sizes (~45–51 t/s) — Vulkan/RADV handles large KV cache much better than ROCm on this hardware
- No crash workarounds needed — Vulkan runs natively without Docker or env var hacks

---

## Benchmark: Qwen3.5-35B-A3B — CPU Only (Native)

**Date:** 2026-04-12
**Tool:** `test-server-perf.py` — three context sizes: 128, 1024, 4096 tokens

### Hardware

| Component | Details |
|-----------|---------|
| CPU | AMD APU (host, all cores) |
| Backend | llama.cpp CPU (no GPU offload, `-ngl 0`) |

### Model

| Field | Value |
|-------|-------|
| Model | `lmstudio-community/Qwen3.5-35B-A3B-GGUF` |
| File | `Qwen3.5-35B-A3B-Q4_K_M.gguf` |
| Quantization | Q4_K_M |
| Size | ~20 GB |

### Results

| Context Size (actual tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
|------------------------------|-----------|---------------------|------------------------|
| 140 tokens                   | 128       | 81.77               | 18.43                  |
| 937 tokens                   | 1024      | 88.32               | 18.47                  |
| 3330 tokens                  | 4096      | 84.19               | 17.45                  |

### Observations

- **Prefill is fastest of all three backends** (~82–88 t/s) — CPU benefits from AVX2/FMA SIMD for matrix multiply and doesn't pay GPU dispatch overhead
- **Generation is on par with ROCm Docker** (~17–18 t/s) — memory bandwidth bound; the UMA APU shares RAM between CPU and GPU so neither has a real bandwidth advantage
- **Flat across all context sizes** — consistent performance regardless of KV cache size

---

## Benchmark: Qwen3.5-35B-A3B — LM Studio (Vulkan)

**Date:** 2026-04-12
**Tool:** `test-lmstudio-perf.py` — three context sizes: 128, 1024, 4096 tokens

### Hardware

| Component | Details |
|-----------|----------|
| GPU | AMD Radeon Graphics (Vega 8 iGPU, gfx90c) |
| VRAM | 65536 MiB (64 GB GTT — system RAM mapped via UMA) |
| Backend | Vulkan (via LM Studio, local server at port 1234) |

### Model

| Field | Value |
|-------|-------|
| Model | `lmstudio-community/Qwen3.5-35B-A3B-GGUF` |
| File | `Qwen3.5-35B-A3B-Q4_K_M.gguf` |
| Quantization | Q4_K_M |
| Size | ~20 GB |

### Server Configuration

LM Studio local server — Vulkan backend, default settings, model loaded via UI.

### Results

> Note: prompt token counts are estimated from word count (LM Studio's `/v1/completions` stream does not always return `usage.prompt_tokens`). Actual token counts may differ slightly.

| Context Size (est. tokens) | Requested | Prefill (Prompt) t/s | Generation (Decode) t/s |
|----------------------------|-----------|---------------------|------------------------|
| ~128 tokens                | 128       | 48.84               | 19.58                  |
| ~1024 tokens               | 1024      | 137.85              | 19.05                  |
| ~4096 tokens               | 4096      | 157.60              | 18.05                  |

### Observations

- **Prefill scales dramatically with context size** (49 → 138 → 158 t/s) — this likely reflects LM Studio's batched prefill processing becoming more efficient with larger prompts
- **Generation is stable** at ~18–20 t/s — consistent with Vulkan native, as expected (same backend)
- **Prefill figures are not directly comparable to other backends** — the estimate-based token counts and LM Studio's internal scheduling make wall-clock prefill appear faster than the raw llama-server measurements
- Decode throughput matches the native Vulkan results (~19–20 t/s), confirming the same underlying GPU path

---

## Comparison: ROCm (Docker) vs Vulkan (Native) vs Vulkan (LM Studio) vs CPU

Model: `Qwen3.5-35B-A3B-Q4_K_M` — AMD Vega 8 iGPU — 2026-04-12

> \* LM Studio prefill uses estimated token counts and includes scheduling overhead — not directly comparable to llama-server prefill.

| Context | ROCm Prefill | Vulkan Prefill | LM Studio Prefill\* | CPU Prefill    | ROCm Decode | Vulkan Decode  | LM Studio Decode | CPU Decode |
|---------|-------------|----------------|---------------------|----------------|-------------|----------------|------------------|------------|
| ~128    | 38.40 t/s   | 44.58 t/s      | 48.84 t/s           | **81.77 t/s**  | 14.23 t/s   | **19.90 t/s**  | 19.58 t/s        | 18.43 t/s  |
| ~1024   | 51.33 t/s   | 50.62 t/s      | 137.85 t/s          | **88.32 t/s**  | 13.49 t/s   | **20.46 t/s**  | 19.05 t/s        | 18.47 t/s  |
| ~4096   | 36.98 t/s   | 50.00 t/s      | 157.60 t/s          | **84.19 t/s**  | 11.52 t/s   | **19.76 t/s**  | 18.05 t/s        | 17.45 t/s  |

**Summary:**
- **Best prefill (raw):** CPU (82–88 t/s) — SIMD wins for prompt processing on this UMA APU
- **Best prefill (LM Studio):** 49–158 t/s — inflated by batched scheduling; not comparable to single-request llama-server measurements
- **Best generation:** Vulkan native & LM Studio both ~19–20 t/s — same Vulkan backend, consistent
- **ROCm (Docker):** Weakest on both metrics due to runtime overhead and `XNACK=0` / `UMA=0` constraints
- On a UMA APU the CPU and GPU share the same memory bus, which explains why CPU prefill can outperform GPU backends
