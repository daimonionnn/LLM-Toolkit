#!/bin/bash
# =============================================================================
# tune-rocm7-vega.sh — ROCm 7.2 / Vega 8 tuning sweep
#
# Benchmarks the working ROCm 7.2 Docker path (image: llama-rocm7-vega) across
# a set of runtime tweaks to find any that beat the baseline on the Vega 8 iGPU.
# Each config runs in a fresh container; results are written as CSV and printed
# as a markdown comparison table for docs/benchmarks.md.
#
# Background (why these levers): docs/ARCHITECTURE.md "Performance ceiling and
# tuning levers" — gfx900 has no hardware dp4a and decode is DDR4-bandwidth-bound.
#
# Usage:
#   ./bench/tune-rocm7-vega.sh
#   ./bench/tune-rocm7-vega.sh 2>&1 | tee /tmp/tune-rocm7-$(date +%Y%m%d-%H%M).log
#
# Env overrides:
#   MODEL=/path/to/model.gguf   (default: Qwen3.5-35B-A3B-Q4_K_M)
#   CTX=8192                    context size
#   IMAGE=llama-rocm7-vega      docker image
# =============================================================================

set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

MODEL="${MODEL:-$HOME/.lmstudio/models/lmstudio-community/Qwen3.5-35B-A3B-GGUF/Qwen3.5-35B-A3B-Q4_K_M.gguf}"
IMAGE="${IMAGE:-llama-rocm7-vega}"
CTX="${CTX:-8192}"
PORT=8080
WAIT_TIMEOUT=300          # 20 GB model load into GTT can take a few minutes
PROMPT_SIZES=(128 1024 4096)
GEN_TOKENS=50
RESULTS_DIR="/tmp/tune-rocm7-$(date +%Y%m%d-%H%M%S)"
RESULTS_CSV="$RESULTS_DIR/results.csv"

# ── Configs: "LABEL|extra llama-server flags|PERF" ───────────────────────────
# PERF=high → set GPU clocks high for that config (reset to auto afterwards).
CONFIGS=(
    "baseline|-fa 0|auto"
    "ub256|-fa 0 -b 2048 -ub 256|auto"
    "ub1024|-fa 0 -b 2048 -ub 1024|auto"
    "ub2048|-fa 0 -b 2048 -ub 2048|auto"
    "kcache-q8|-fa 0 -ctk q8_0|auto"
    "perflevel-high|-fa 0|high"
)

mkdir -p "$RESULTS_DIR"
echo "backend,requested,actual_ctx,prefill_tps,decode_tps" > "$RESULTS_CSV"

if [ ! -f "$MODEL" ]; then echo "✗ model not found: $MODEL"; exit 1; fi
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then echo "✗ docker image missing: $IMAGE"; exit 1; fi

MODEL_DIR="$(dirname "$MODEL")"
MODEL_NAME="$(basename "$MODEL")"

# ── Detect Vega 8 render node + card ─────────────────────────────────────────
# Detection lives in lib/vega8.sh so every script in this repo agrees. Both the
# render node and the card number move when GPUs are added or removed — this box
# has been renderD129, renderD130 and now renderD128 — so neither is hardcoded.
# Override with VEGA8_RENDER_NODE / VEGA8_CARD_DIR.
. "$SCRIPT_DIR/../lib/vega8.sh"
VEGA8_RENDER_NODE="$(vega8_render_node || true)"
VEGA8_CARD_PATH="$(vega8_card_dir || true)"
VEGA8_CARD="$(basename "${VEGA8_CARD_PATH:-unknown}")"
: "${VEGA8_RENDER_NODE:=/dev/dri/renderD128}"
echo "Vega 8 render node: $VEGA8_RENDER_NODE   card: $VEGA8_CARD"
PERF_PATH="${VEGA8_CARD_PATH:-/sys/class/drm/$VEGA8_CARD}/device/power_dpm_force_performance_level"

set_perf() {  # $1 = auto|high ; returns non-zero if it could not change the level
    [ -z "$VEGA8_CARD" ] && return 1
    if [ -w "$PERF_PATH" ]; then echo "$1" > "$PERF_PATH" 2>/dev/null
    else echo "$1" | sudo -n tee "$PERF_PATH" >/dev/null 2>&1; fi
    local now; now="$(cat "$PERF_PATH" 2>/dev/null)"
    echo "  GPU perf level → ${now:-unknown}"
    [ "$now" = "$1" ]
}

stop_container() {
    docker stop "$(docker ps -q --filter "ancestor=$IMAGE")" >/dev/null 2>&1 || true
    docker stop "$(docker ps -q --filter "publish=$PORT")"   >/dev/null 2>&1 || true
}
trap 'stop_container; set_perf auto' EXIT

run_config() {
    local label="$1" flags="$2" perf="$3"
    echo ""
    echo "──────────────────────────────────────────────────────────────"
    echo "  CONFIG: $label   flags: [$flags]   perf: $perf"
    echo "──────────────────────────────────────────────────────────────"
    if [ "$perf" = "high" ]; then
        if ! set_perf high; then
            echo "  ⊘ SKIP $label — cannot set GPU perf level (needs root; see notes after the run)"
            return 2
        fi
    fi
    stop_container; sleep 2

    # shellcheck disable=SC2086
    docker run --rm -d \
        --device=/dev/kfd --device="$VEGA8_RENDER_NODE" \
        --group-add=video --group-add=render \
        --ipc=host --security-opt seccomp=unconfined --ulimit memlock=-1 \
        -e ROCR_VISIBLE_DEVICES=0 -e HIP_VISIBLE_DEVICES=0 \
        -v "$MODEL_DIR:/models:ro" -p "$PORT:8080" \
        "$IMAGE" \
        --host 0.0.0.0 -m "/models/$MODEL_NAME" -ngl 99 -c "$CTX" --no-warmup $flags \
        >/dev/null 2>&1 || { echo "  ✗ docker run failed"; return 1; }

    # Wait for health
    local deadline=$(( $(date +%s) + WAIT_TIMEOUT ))
    printf "  loading "
    while (( $(date +%s) < deadline )); do
        curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo " ready"; break; }
        printf "."; sleep 3
    done
    if ! curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo " TIMEOUT — skipping $label"; stop_container; [ "$perf" = "high" ] && set_perf auto; return 1
    fi

    python3 - "$label" "$RESULTS_CSV" "$PORT" "$GEN_TOKENS" "${PROMPT_SIZES[@]}" <<'PY'
import sys, json, urllib.request, time
label, csv, port, gen = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
sizes = [int(x) for x in sys.argv[5:]]
base = ("The quick brown fox jumped over the lazy dog. "
        "Here is some more text to fill up the context window. ").split()
print(f"  {'req':>6} | {'ctx':>6} | {'prefill t/s':>12} | {'decode t/s':>11}")
with open(csv, "a") as f:
    for tgt in sizes:
        prompt = " ".join((base*(tgt//len(base)+1))[:tgt])
        payload = json.dumps({"prompt":prompt,"n_predict":gen,"temperature":0.0}).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", data=payload,
                                     headers={"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(req, timeout=600) as r: res = json.loads(r.read())
            t = res.get("timings", {})
            pre, dec, ctx = t.get("prompt_per_second",0), t.get("predicted_per_second",0), t.get("prompt_n",tgt)
            print(f"  {tgt:>6} | {ctx:>6} | {pre:>12.2f} | {dec:>11.2f}")
            f.write(f"{label},{tgt},{ctx},{pre:.2f},{dec:.2f}\n")
        except Exception as e:
            print(f"  {tgt:>6} | ERROR: {e}")
            f.write(f"{label},{tgt},ERR,ERR,ERR\n")
PY

    stop_container
    [ "$perf" = "high" ] && set_perf auto
    sleep 2
}

echo "════════════════════════════════════════════════════════════════"
echo "  ROCm 7.2 / Vega 8 tuning sweep"
echo "  Model:   $MODEL_NAME"
echo "  Context: $CTX   Prompts: ${PROMPT_SIZES[*]}   Configs: ${#CONFIGS[@]}"
echo "  Results: $RESULTS_DIR"
echo "════════════════════════════════════════════════════════════════"

for entry in "${CONFIGS[@]}"; do
    IFS='|' read -r label flags perf <<< "$entry"
    run_config "$label" "$flags" "$perf"
done

# ── Markdown comparison table ────────────────────────────────────────────────
echo ""
echo "  Markdown table → $RESULTS_DIR/table.md"
python3 - "$RESULTS_CSV" "${PROMPT_SIZES[@]}" > "$RESULTS_DIR/table.md" <<'PY'
import sys, csv
path = sys.argv[1]; sizes = [int(x) for x in sys.argv[2:]]
data = {}
with open(path) as f:
    for row in csv.DictReader(f):
        data.setdefault(row["backend"], {})[int(row["requested"])] = (row["prefill_tps"], row["decode_tps"])
def tbl(idx, name):
    hdr = " | ".join(f"~{s} tok" for s in sizes)
    print(f"\n**{name} (t/s)**\n")
    print(f"| Config | {hdr} |")
    print("|" + "---|"*(len(sizes)+1))
    for cfg, e in data.items():
        vals = " | ".join(e.get(s, ("N/A","N/A"))[idx] for s in sizes)
        print(f"| {cfg} | {vals} |")
tbl(0, "Prefill"); tbl(1, "Decode")
PY
cat "$RESULTS_DIR/table.md"
echo ""
echo "  Raw CSV: $RESULTS_CSV"
