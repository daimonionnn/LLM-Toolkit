#!/bin/bash
# =============================================================================
# run-all-benchmarks.sh — Comprehensive multi-backend benchmark runner
#
# Starts each selected backend, waits for it to be ready, runs the perf test,
# then stops the server and moves to the next backend.
#
# To run only specific backends: comment out entries in the ENABLED_BACKENDS
# array near the bottom of the config section.
#
# Usage:
#   ./bench/run-all-benchmarks.sh
#   ./bench/run-all-benchmarks.sh 2>&1 | tee /tmp/bench-$(date +%Y%m%d-%H%M).log
# =============================================================================

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# =============================================================================
# ── CONFIG ────────────────────────────────────────────────────────────────────
# =============================================================================

# Models to benchmark — comment out any line to skip that model
MODELS=(
    "$HOME/.lmstudio/models/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf"
    "$HOME/.lmstudio/models/lmstudio-community/gemma-4-E4B-it-GGUF/gemma-4-E4B-it-Q4_K_M.gguf"
)

# Override the model list for a single run without editing this file:
#   BENCH_MODELS="/path/a.gguf /path/b.gguf" bash bench/run-all-benchmarks.sh
if [ -n "${BENCH_MODELS:-}" ]; then
    read -r -a MODELS <<< "$BENCH_MODELS"
fi

# llama-server port (all backends share the same port sequentially)
SERVER_PORT=8080

# Context window size passed to the server
CONTEXT_SIZE=8192

# Maximum seconds to wait for the server to become ready
SERVER_WAIT_TIMEOUT=120

# Prompt sizes (word counts) to test — these map to ~128, ~1024, ~4096 tokens
PROMPT_SIZES=(128 1024 4096)

# Number of tokens to generate per decode measurement
# TODO: this setting is currently ignored — the inline python in run_bench()
#       hardcodes its own GEN_TOKENS=50. Pass it through as an argv, the way
#       bench/tune-rocm7-vega.sh already does.
GEN_TOKENS=50

# Where to write per-backend result CSVs  (label, prompt_words, ctx_tokens, prefill, decode)
RESULTS_DIR="/tmp/bench-results-$(date +%Y%m%d-%H%M%S)"

# Bare-metal ROCm llama-server binaries (leave empty to auto-detect default path)
# Default paths (auto-detected when empty):
#   ROCm 7: llm/rocm7-vega/bin/llama-server   (built by build/build-llamacpp-rocm7-baremetal.sh)
#   ROCm 6: llm/rocm-vega/bin/llama-server    (built by build/build-llamacpp-rocm-vega.sh)
ROCM7_LLAMA_BIN=""   # override: /opt/rocm7/bin/llama-server
ROCM6_LLAMA_BIN=""   # override: /opt/rocm6/bin/llama-server

# =============================================================================
# ── ENABLED BACKENDS ─────────────────────────────────────────────────────────
# Comment out any line here to skip that backend.
# Format: "LABEL:FA_FLAG:START_FUNC"
# =============================================================================
# TODO: add R9700 rows (Vulkan1/2 and native gfx1201 ROCm) so the iGPU and the
#       dGPUs can be compared on the same harness and models.
ENABLED_BACKENDS=(
    # ── ROCm 7.2 via Docker (recommended ROCm path — self-contained image) ───
    "ROCm-7.2-Docker-FA-OFF:-fa 0:start_rocm7_docker"
    "ROCm-7.2-Docker-FA-ON:-fa 1:start_rocm7_docker"

    # ── ROCm 6.2.4 via Docker ────────────────────────────────────────────────
    #"ROCm-6.2.4-Docker-FA-OFF:-fa 0:start_rocm6_docker"
    #"ROCm-6.2.4-Docker-FA-ON:-fa 1:start_rocm6_docker"

    # ── ROCm 7.2 bare-metal (needs classic ROCm 7.0-7.2 + gfx900 backport on
    #    the host; working again since September 2026) ──────────────────────
    "ROCm-7.2-Baremetal-FA-OFF:-fa 0:start_rocm7_baremetal"
    "ROCm-7.2-Baremetal-FA-ON:-fa 1:start_rocm7_baremetal"

    # ── ROCm 6.2.4 bare-metal (uncomment when host ROCm stack is working) ────
    # "ROCm-6.2.4-Baremetal-FA-OFF:-fa 0:start_rocm6_baremetal"
    # "ROCm-6.2.4-Baremetal-FA-ON:-fa 1:start_rocm6_baremetal"

    # ── Vulkan (native, GPU offload) ──────────────────────────────────────────
    "Vulkan-GPU-FA-OFF:-fa 0:start_vulkan_gpu"    
    "Vulkan-GPU-FA-ON:-fa 1:start_vulkan_gpu"

    # ── CPU only (no GPU offload) ─────────────────────────────────────────────
    "CPU-FA-ON:-fa 1:start_cpu"
    "CPU-FA-OFF:-fa 0:start_cpu"
)

# Override the backend list for a single run (one entry per line):
#   BENCH_BACKENDS=$'CPU-FA-ON:-fa 1:start_cpu' bash bench/run-all-benchmarks.sh
if [ -n "${BENCH_BACKENDS:-}" ]; then
    mapfile -t ENABLED_BACKENDS <<< "$BENCH_BACKENDS"
fi

# =============================================================================
# ── BACKEND START FUNCTIONS ───────────────────────────────────────────────────
# Each function starts the server in the background and sets SERVER_PID.
# All functions receive: FA_FLAG (e.g. "-fa 0") as $1
# =============================================================================

# ── Docker-based ROCm backends ───────────────────────────────────────────────

start_rocm7_docker() {
    local fa_flag="$1"
    echo "  [start] ROCm 7.2 — Docker (image: llama-rocm7-vega) | $fa_flag | -ngl 99 | -c $CONTEXT_SIZE"
    # Stop any existing container on this image or port
    docker stop $(docker ps -q --filter "ancestor=llama-rocm7-vega") 2>/dev/null || true
    docker stop $(docker ps -q --filter "publish=$SERVER_PORT") 2>/dev/null || true
    sleep 1

    local render_node
    render_node=$(_detect_vega8_render_node)

    docker run --rm \
        --device=/dev/kfd \
        --device="$render_node" \
        --group-add=video --group-add=render \
        --ipc=host \
        --security-opt seccomp=unconfined \
        --ulimit memlock=-1 \
        -e ROCR_VISIBLE_DEVICES=0 \
        -e HIP_VISIBLE_DEVICES=0 \
        -v "$(dirname "$CURRENT_MODEL"):/models:ro" \
        -p "$SERVER_PORT:8080" \
        llama-rocm7-vega \
        --host 0.0.0.0 \
        -m "/models/$(basename "$CURRENT_MODEL")" \
        $fa_flag \
        -ngl 99 \
        -c "$CONTEXT_SIZE" \
        --no-warmup \
        >/tmp/bench-server.log 2>&1 &
    SERVER_PID=$!
}

start_rocm6_docker() {
    local fa_flag="$1"
    echo "  [start] ROCm 6.2.4 — Docker (image: llama-server-rocm-vega) | $fa_flag | -ngl 99 | -c $CONTEXT_SIZE"
    docker stop $(docker ps -q --filter "ancestor=llama-server-rocm-vega") 2>/dev/null || true
    docker stop $(docker ps -q --filter "publish=$SERVER_PORT") 2>/dev/null || true
    sleep 1

    local render_node
    render_node=$(_detect_vega8_render_node)

    docker run --rm \
        --device=/dev/kfd \
        --device="$render_node" \
        --group-add=video --group-add=render \
        --ipc=host \
        --security-opt seccomp=unconfined \
        --ulimit memlock=-1 \
        -e ROCR_VISIBLE_DEVICES=0 \
        -e HIP_VISIBLE_DEVICES=0 \
        -v "$(dirname "$CURRENT_MODEL"):/models:ro" \
        -p "$SERVER_PORT:8080" \
        llama-server-rocm-vega \
        --host 0.0.0.0 \
        -m "/models/$(basename "$CURRENT_MODEL")" \
        $fa_flag \
        -ngl 99 \
        -c "$CONTEXT_SIZE" \
        --no-warmup \
        >/tmp/bench-server.log 2>&1 &
    SERVER_PID=$!
}

# ── Bare-metal ROCm backends ──────────────────────────────────────────────────
# Requires ROCm installed on the host and llama.cpp built for gfx900.
# Setup: bash setup/install-rocm7-host.sh && bash build/build-llamacpp-rocm7-baremetal.sh
# The HSA_OVERRIDE_GFX_VERSION=9.0.0 override tells the ROCm runtime that Vega 8 APU
# (gfx90c) should use gfx900 kernels (which we backport from ROCm 6.3.4).

start_rocm7_baremetal() {
    local fa_flag="$1"
    local bin="${ROCM7_LLAMA_BIN:-$REPO_DIR/llm/rocm7-vega/bin/llama-server}"
    if [[ ! -x "$bin" ]]; then
        echo "  [skip] ROCm 7.2 bare-metal: binary not found at $bin"
        echo "         Run: bash setup/install-rocm7-host.sh && bash build/build-llamacpp-rocm7-baremetal.sh"
        SERVER_PID=""
        return 1
    fi
    echo "  [start] ROCm 7.2 — bare-metal ($bin) | $fa_flag | -ngl 99 | -c $CONTEXT_SIZE"
    pkill -f "llama-server.*port $SERVER_PORT" 2>/dev/null || true
    sleep 1
    # HSA_XNACK must stay 0 — XNACK=1 hard-freezes the whole PC on Vega 8.
    # HSA_ENABLE_SDMA=0 — SDMA is unreliable on integrated Vega.
    ROCR_VISIBLE_DEVICES="$(_detect_vega8_rocm_index)" \
    HIP_VISIBLE_DEVICES=0 \
    HSA_OVERRIDE_GFX_VERSION=9.0.0 \
    HSA_ENABLE_SDMA=0 \
    HSA_XNACK=0 \
    GPU_MAX_ALLOC_PERCENT=100 \
    LD_LIBRARY_PATH="$(dirname "$(dirname "$bin")")/lib:/opt/rocm/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$bin" \
        -m "$CURRENT_MODEL" \
        $fa_flag \
        -ngl 99 \
        -c "$CONTEXT_SIZE" \
        --host 0.0.0.0 \
        --port "$SERVER_PORT" \
        --no-warmup \
        >/tmp/bench-server.log 2>&1 &
    SERVER_PID=$!
}

start_rocm6_baremetal() {
    local fa_flag="$1"
    if [[ -z "$ROCM6_LLAMA_BIN" || ! -x "$ROCM6_LLAMA_BIN" ]]; then
        echo "  [skip] ROCm 6.2.4 bare-metal: ROCM6_LLAMA_BIN not set or not executable"
        echo "         Set ROCM6_LLAMA_BIN at the top of this script."
        SERVER_PID=""
        return 1
    fi
    echo "  [start] ROCm 6.2.4 — bare-metal ($ROCM6_LLAMA_BIN) | $fa_flag | -ngl 99 | -c $CONTEXT_SIZE"
    pkill -f "llama-server.*port $SERVER_PORT" 2>/dev/null || true
    sleep 1
    ROCR_VISIBLE_DEVICES="$(_detect_vega8_rocm_index)" HIP_VISIBLE_DEVICES=0 \
        "$ROCM6_LLAMA_BIN" \
        -m "$CURRENT_MODEL" \
        $fa_flag \
        -ngl 99 \
        -c "$CONTEXT_SIZE" \
        --host 0.0.0.0 \
        --port "$SERVER_PORT" \
        --no-warmup \
        >/tmp/bench-server.log 2>&1 &
    SERVER_PID=$!
}

# ── Native (non-Docker) backends ─────────────────────────────────────────────

start_vulkan_gpu() {
    local fa_flag="$1"
    echo "  [start] Vulkan GPU — native | $fa_flag | -ngl 99 | -c $CONTEXT_SIZE"
    pkill -f "llama-server.*port $SERVER_PORT" 2>/dev/null || true
    sleep 1

    # TODO: -dev Vulkan0 is hardcoded; auto-detect the RADV RENOIR index like
    #       run/start-llama-server.sh does — the device order shifts whenever
    #       discrete GPUs are added/removed.

    LD_LIBRARY_PATH="$REPO_DIR/llm/vulkan/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$REPO_DIR/llm/vulkan/bin/llama-server" \
        -m "$CURRENT_MODEL" \
        $fa_flag \
        -ngl 99 \
        -dev Vulkan0 \
        -c "$CONTEXT_SIZE" \
        --host 0.0.0.0 \
        --port "$SERVER_PORT" \
        --no-warmup \
        >/tmp/bench-server.log 2>&1 &
    SERVER_PID=$!
}

# CPU-only means -dev none, NOT -ngl 0.
#
# Upstream changed the -ngl default to 'auto', and on this build `-ngl 0` no
# longer keeps the model off the GPU: measured on 2026-09-07 with gemma-4-E4B,
# `-ngl 0` still gave 91 % GPU busy and 6567 MB of GTT, while `-dev none` gave
# 157 MB and left the GPU idle. Benchmarks taken with `-ngl 0` on current
# llama.cpp are therefore GPU runs mislabelled as CPU runs — which is exactly
# what made the 2026-09-07 "CPU" rows land near the Vulkan numbers instead of
# the much higher historical CPU prefill.
start_cpu() {
    local fa_flag="$1"
    echo "  [start] CPU only — native Vulkan binary, -dev none | $fa_flag | -c $CONTEXT_SIZE"
    pkill -f "llama-server.*port $SERVER_PORT" 2>/dev/null || true
    sleep 1

    LD_LIBRARY_PATH="$REPO_DIR/llm/vulkan/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        "$REPO_DIR/llm/vulkan/bin/llama-server" \
        -m "$CURRENT_MODEL" \
        $fa_flag \
        -dev none \
        -c "$CONTEXT_SIZE" \
        --host 0.0.0.0 \
        --port "$SERVER_PORT" \
        --no-warmup \
        >/tmp/bench-server.log 2>&1 &
    SERVER_PID=$!
}

# =============================================================================
# ── HELPERS ───────────────────────────────────────────────────────────────────
# =============================================================================

_detect_vega8_render_node() {
    local pci_id="0x1638"
    for node in /sys/class/drm/renderD*/device; do
        local dev_id
        dev_id="$(cat "$node/device" 2>/dev/null || true)"
        if [[ "$dev_id" == "$pci_id" ]]; then
            echo "/dev/dri/$(basename "$(dirname "$node")")"
            return 0
        fi
    done
    # Fallback: known Vega 8 node on this machine (September 2026, single GPU).
    # This value has changed three times; the PCI-ID scan above is what to rely on.
    echo "/dev/dri/renderD128"
}

# 0-based GPU index of the Vega 8 in rocminfo agent order (= ROCR index).
# rocminfo prints each agent's "Name: gfxXXX" before its "Device Type: GPU".
_detect_vega8_rocm_index() {
    if [[ -n "${VEGA8_ROCM_DEVICE:-}" ]]; then
        echo "$VEGA8_ROCM_DEVICE"
        return
    fi
    local rocminfo_bin="/opt/rocm/bin/rocminfo"
    [[ -x "$rocminfo_bin" ]] || rocminfo_bin="$(command -v rocminfo || true)"
    if [[ -z "$rocminfo_bin" ]]; then
        echo "0"
        return
    fi
    "$rocminfo_bin" 2>/dev/null | awk '
        $1 == "Name:" && $2 ~ /^gfx/  { name = $2 }
        /Device Type:[[:space:]]+GPU/ {
            # print gpu+0, not gpu: when the Vega 8 is the first GPU its index
            # is 0 and `gpu` was never assigned, so bare `print gpu` emits an
            # EMPTY string. That becomes ROCR_VISIBLE_DEVICES="", which hides
            # every GPU and silently falls back to CPU ("no usable GPU found").
            # Masked until September 2026, when the dGPUs left and the Vega
            # became index 0 for the first time.
            if (name ~ /^gfx90[029c]$/) { print gpu+0; found = 1; exit }
            gpu++
        }
        END { if (!found) print 0 }
    '
}

wait_for_server() {
    local label="$1"
    if [[ -z "$SERVER_PID" ]]; then
        echo "  [wait] No server PID — skipping"
        return 1
    fi
    local deadline=$(( $(date +%s) + SERVER_WAIT_TIMEOUT ))
    printf "  [wait] "
    while (( $(date +%s) < deadline )); do
        if curl -sf "http://127.0.0.1:$SERVER_PORT/health" >/dev/null 2>&1; then
            echo " ready"
            return 0
        fi
        # Check process is still alive
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            echo " SERVER DIED — see /tmp/bench-server.log"
            return 1
        fi
        printf "."
        sleep 2
    done
    echo " TIMEOUT after ${SERVER_WAIT_TIMEOUT}s"
    return 1
}

stop_server() {
    # Stop Docker containers
    docker stop $(docker ps -q --filter "publish=$SERVER_PORT") 2>/dev/null || true
    # Kill native processes
    pkill -f "llama-server.*port $SERVER_PORT" 2>/dev/null || true
    kill "$SERVER_PID" 2>/dev/null || true
    sleep 2
}

run_bench() {
    local label="$1"
    local out_csv="$2"
    echo "  [bench] Running perf test for: $label"
    # TODO: this inline python triplicates bench/test-server-perf.py and the
    #       inline bench in bench/tune-rocm7-vega.sh — consolidate into one
    #       parameterized script (URL, sizes, gen tokens, CSV out) used by all.
    # TODO: also record model load time (time from server start to /health OK)
    #       — it differs a lot between backends and is worth tracking.

    python3 - "$label" "$out_csv" "$SERVER_PORT" "${PROMPT_SIZES[@]}" <<'PYEOF'
import sys, json, urllib.request, urllib.error, time

label       = sys.argv[1]
out_csv     = sys.argv[2]
port        = sys.argv[3]
sizes       = [int(x) for x in sys.argv[4:]]
url         = f"http://127.0.0.1:{port}/completion"
base_words  = ("The quick brown fox jumped over the lazy dog. "
               "Here is some more text to fill up the context window. ").split()
GEN_TOKENS  = 50

rows = []
print(f"  {'Requested':>10} | {'Actual ctx':>10} | {'Prefill t/s':>12} | {'Decode t/s':>12}")
print("  " + "-" * 54)

for target in sizes:
    words  = (base_words * (target // len(base_words) + 1))[:target]
    prompt = " ".join(words)
    payload = json.dumps({"prompt": prompt, "n_predict": GEN_TOKENS, "temperature": 0.0}).encode()
    req     = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            result = json.loads(resp.read())
        t = result.get("timings", {})
        prefill = t.get("prompt_per_second", 0)
        decode  = t.get("predicted_per_second", 0)
        ctx     = t.get("prompt_n", target)
        print(f"  {target:>10} | {ctx:>10} | {prefill:>11.2f} | {decode:>11.2f}")
        rows.append((label, target, ctx, f"{prefill:.2f}", f"{decode:.2f}"))
    except Exception as e:
        print(f"  {target:>10} | ERROR: {e}")
        rows.append((label, target, "ERR", "ERR", "ERR"))

with open(out_csv, "w") as f:
    f.write("backend,requested,actual_ctx,prefill_tps,decode_tps\n")
    for r in rows:
        f.write(",".join(str(x) for x in r) + "\n")
PYEOF
}

print_summary() {
    echo ""
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  BENCHMARK SUMMARY"
    echo "  Context: $CONTEXT_SIZE  |  Date: $(date '+%Y-%m-%d %H:%M')"
    echo "════════════════════════════════════════════════════════════════════════"

    python3 - "$RESULTS_DIR" "${PROMPT_SIZES[@]}" <<'PYEOF'
import sys, os, glob

results_dir = sys.argv[1]
sizes       = [int(x) for x in sys.argv[2:]]

# Load all CSVs — filename encodes  ModelLabel__BackendLabel
data = {}   # {model: {backend: {requested: (prefill, decode)}}}
for f in sorted(glob.glob(os.path.join(results_dir, "*.csv"))):
    with open(f) as fh:
        lines = fh.read().strip().splitlines()[1:]   # skip header
    for line in lines:
        parts = line.split(",")
        full_label, req, ctx, pre, dec = parts[0], int(parts[1]), parts[2], parts[3], parts[4]
        if "__" in full_label:
            model_label, backend_label = full_label.split("__", 1)
        else:
            model_label, backend_label = "unknown", full_label
        data.setdefault(model_label, {}).setdefault(backend_label, {})[req] = (pre, dec)

if not data:
    print("  No results found.")
    sys.exit(0)

col_w = 34
for model_label, backends in data.items():
    print(f"\n  ── Model: {model_label} ──")
    for metric, col_idx in [("Prefill (t/s)", 0), ("Decode (t/s)", 1)]:
        print(f"\n  {metric}  (higher is better)")
        print(f"  {'Backend':<{col_w}} | " + " | ".join(f"{'~'+str(s)+' tok':>9}" for s in sizes))
        print("  " + "-" * col_w + "---" + "---+-----------" * len(sizes))
        for backend_label, entries in backends.items():
            vals = " | ".join(
                f"{entries[s][col_idx]:>9}" if s in entries else f"{'N/A':>9}"
                for s in sizes
            )
            print(f"  {backend_label:<{col_w}} | {vals}")
PYEOF
    echo ""
    echo "  Raw CSVs: $RESULTS_DIR/"
    echo "════════════════════════════════════════════════════════════════════════"
}

# =============================================================================
# ── MAIN ──────────────────────────────────────────────────────────────────────
# =============================================================================

mkdir -p "$RESULTS_DIR"

echo "════════════════════════════════════════════════════════════════════════"
echo "  run-all-benchmarks.sh"
echo "  Models:  ${#MODELS[@]}"
for m in "${MODELS[@]}"; do echo "    $(basename "$m")"; done
echo "  Context: $CONTEXT_SIZE   Prompt sizes: ${PROMPT_SIZES[*]}"
echo "  Backends: ${#ENABLED_BACKENDS[@]}"
echo "  Results: $RESULTS_DIR"
echo "════════════════════════════════════════════════════════════════════════"
echo ""

SERVER_PID=""
COMPLETED=0
FAILED=0

for CURRENT_MODEL in "${MODELS[@]}"; do
    MODEL_LABEL="$(basename "$CURRENT_MODEL" .gguf)"
    echo "════════════════════════════════════════════════════════════════════════"
    echo "  MODEL: $MODEL_LABEL"
    echo "════════════════════════════════════════════════════════════════════════"
    echo ""

    for entry in "${ENABLED_BACKENDS[@]}"; do
        IFS=: read -r label fa_flag start_fn <<< "$entry"
        # Prefix label with short model name so CSV filenames and summary are unambiguous
        full_label="${MODEL_LABEL}__${label}"
        echo "──────────────────────────────────────────────────────────────────────"
        echo "  Model:   $MODEL_LABEL"
        echo "  Backend: $label"

        SERVER_PID=""
        if ! "$start_fn" "$fa_flag"; then
            echo "  SKIPPING $label — start function returned error (check configuration above)"
            FAILED=$(( FAILED + 1 ))
            echo ""
            continue
        fi

        if wait_for_server "$label"; then
            csv_file="$RESULTS_DIR/$(echo "$full_label" | tr '/' '-').csv"
            run_bench "$full_label" "$csv_file"
            stop_server
            COMPLETED=$(( COMPLETED + 1 ))
        else
            echo "  SKIPPING bench for $label — server did not start"
            cat /tmp/bench-server.log | tail -20 | sed 's/^/    /'
            stop_server || true
            FAILED=$(( FAILED + 1 ))
        fi
        echo ""
    done
done

echo "Completed: $COMPLETED / $(( COMPLETED + FAILED )) backend×model combinations"
print_summary
