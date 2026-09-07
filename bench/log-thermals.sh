#!/bin/bash
#
# Log CPU/iGPU thermals and clocks while a benchmark runs.
#
# Why this exists: this rig is an APU. The CPU cores, the Vega 8 iGPU and the
# memory controller all sit on one die, share one power/thermal budget and one
# DDR4 bus. A benchmark that pushes prefill hard heats the package, the SoC
# drops SCLK, and decode t/s sags — so a slow number can mean "thermally
# limited" rather than "this backend is slower". Without a thermal trace next
# to the throughput trace the two are indistinguishable, and docs/benchmarks.md
# compares runs recorded on different days.
#
# Reads sysfs directly (no lm-sensors, no rocm-smi, no root). The Vega 8 is
# found by PCI ID 0x1638 like every other script here, and its hwmon is taken
# from that card's own directory — hwmon* and card* indices are NOT stable
# across boots (this box moved card1 -> card0 on the 26.04 reinstall).
#
# Usage:
#   bench/log-thermals.sh                          # log until Ctrl-C
#   bench/log-thermals.sh -d 120                   # log for 120 s
#   bench/log-thermals.sh -- ./bench/bench-rocm.sh # log while a command runs
#   bench/log-thermals.sh -i 1 -o /tmp/run.csv -- llama-bench -m model.gguf
#
# Options:
#   -i, --interval SEC   sample interval (default: 2)
#   -d, --duration SEC   stop after SEC seconds (default: unlimited)
#   -o, --out FILE       CSV path (default: /tmp/bench-results-<ts>/thermals.csv)
#   -q, --quiet          CSV only, no live output
#   -h, --help           this help

set -euo pipefail

INTERVAL=2
DURATION=0
OUT=""
QUIET=false
CMD=()

while [ $# -gt 0 ]; do
    case "$1" in
        -i|--interval) INTERVAL="$2"; shift 2 ;;
        -d|--duration) DURATION="$2"; shift 2 ;;
        -o|--out)      OUT="$2";      shift 2 ;;
        -q|--quiet)    QUIET=true;    shift ;;
        -h|--help)     sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        --)            shift; CMD=("$@"); break ;;
        *) echo "Unknown option: $1 (try --help)" >&2; exit 1 ;;
    esac
done

# ─── Locate sensors ──────────────────────────────────────────────────────────

# Vega 8 by PCI device ID, same convention as the run/ and bench/ scripts.
GPU_DEV=""
for d in /sys/class/drm/card*/device; do
    [ "$(cat "$d/device" 2>/dev/null)" = "0x1638" ] && GPU_DEV="$d" && break
done
[ -z "$GPU_DEV" ] && echo "⚠  Vega 8 (PCI 0x1638) not found — GPU columns will be empty" >&2

# The GPU's own hwmon, not a global name match: with a dGPU present there is
# more than one 'amdgpu' hwmon and the numbering is arbitrary.
GPU_HWMON=""
if [ -n "$GPU_DEV" ]; then
    GPU_HWMON=$(ls -d "$GPU_DEV"/hwmon/hwmon* 2>/dev/null | head -1 || true)
fi

# CPU package temp — k10temp exposes Tctl, which on Cezanne (5700G) is the real
# die temperature with no offset. Tjmax is 95 °C.
CPU_TEMP_FILE=""
for h in /sys/class/hwmon/hwmon*; do
    [ "$(cat "$h/name" 2>/dev/null)" = "k10temp" ] || continue
    for l in "$h"/temp*_label; do
        [ -f "$l" ] || continue
        if [ "$(cat "$l")" = "Tctl" ]; then CPU_TEMP_FILE="${l%_label}_input"; break 2; fi
    done
    # Some kernels expose k10temp without labels; temp1 is Tctl there.
    [ -f "$h/temp1_input" ] && CPU_TEMP_FILE="$h/temp1_input" && break
done
[ -z "$CPU_TEMP_FILE" ] && echo "⚠  k10temp Tctl not found — CPU temp will be empty" >&2

# 95 °C is the 5700G's stock Tjmax. Some boards expose a user-settable throttle
# limit (ASRock B450: CPU Temperature Throttle), so allow overriding it —
# otherwise every run on a raised limit reports a false throttling warning.
TJMAX="${TJMAX:-95}"

# ─── Output path ─────────────────────────────────────────────────────────────
if [ -z "$OUT" ]; then
    OUT="/tmp/bench-results-$(date +%Y%m%d-%H%M%S)/thermals.csv"
fi
mkdir -p "$(dirname "$OUT")"

# ─── Helpers ─────────────────────────────────────────────────────────────────
# Every read is best-effort: a sensor going away mid-run must not kill the log.
rd() { cat "$1" 2>/dev/null || echo ""; }

cpu_mhz_avg() {
    awk '{s+=$1; n++} END{ if (n) printf "%.0f", s/n/1000; }' \
        /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null || echo ""
}

div() {  # value, divisor, decimals — empty in, empty out
    [ -z "$1" ] && { echo ""; return; }
    awk -v v="$1" -v d="$2" -v p="${3:-1}" 'BEGIN{ printf "%.*f", p, v/d }'
}

echo "─────────────────────────────────────────────────────────────"
echo "  Thermal log"
echo "─────────────────────────────────────────────────────────────"
echo "  CSV       : $OUT"
echo "  Interval  : ${INTERVAL}s"
[ "$DURATION" -gt 0 ] && echo "  Duration  : ${DURATION}s"
[ ${#CMD[@]} -gt 0 ] && echo "  Command   : ${CMD[*]}"
echo "  CPU sensor: ${CPU_TEMP_FILE:-none}"
echo "  GPU sensor: ${GPU_HWMON:-none}"
echo ""

# On a TTY each sample overwrites the previous one; otherwise emit plain lines.
if [ -t 1 ]; then LIVE_TTY=true;  LIVE_PREFIX='\r'
else               LIVE_TTY=false; LIVE_PREFIX=''
fi

echo "elapsed_s,cpu_tctl_c,gpu_edge_c,gpu_sclk_mhz,pkg_power_w,gpu_busy_pct,vram_used_mb,gtt_used_mb,cpu_mhz_avg" > "$OUT"

# ─── Run the workload, if one was given ──────────────────────────────────────
CMD_PID=""
CMD_RC=0
if [ ${#CMD[@]} -gt 0 ]; then
    "${CMD[@]}" &
    CMD_PID=$!
fi

STOP=false
trap 'STOP=true' INT TERM

START=$(date +%s)
while ! $STOP; do
    NOW=$(date +%s)
    ELAPSED=$(( NOW - START ))

    [ "$DURATION" -gt 0 ] && [ "$ELAPSED" -ge "$DURATION" ] && break
    if [ -n "$CMD_PID" ] && ! kill -0 "$CMD_PID" 2>/dev/null; then break; fi

    CPU_C=$(div "$(rd "$CPU_TEMP_FILE")" 1000 1)
    if [ -n "$GPU_HWMON" ]; then
        GPU_C=$(div "$(rd "$GPU_HWMON/temp1_input")" 1000 1)
        SCLK=$(div "$(rd "$GPU_HWMON/freq1_input")" 1000000 0)
        PKG_W=$(div "$(rd "$GPU_HWMON/power1_input")" 1000000 1)
    else
        GPU_C=""; SCLK=""; PKG_W=""
    fi
    BUSY=$(rd "$GPU_DEV/gpu_busy_percent")
    VRAM=$(div "$(rd "$GPU_DEV/mem_info_vram_used")" 1048576 0)
    GTT=$(div "$(rd "$GPU_DEV/mem_info_gtt_used")" 1048576 0)
    CPU_MHZ=$(cpu_mhz_avg)

    echo "${ELAPSED},${CPU_C},${GPU_C},${SCLK},${PKG_W},${BUSY},${VRAM},${GTT},${CPU_MHZ}" >> "$OUT"

    if ! $QUIET; then
        printf "%b  %4ss  CPU %5s°C %5sMHz | GPU %5s°C %5sMHz %4s%% | %5sW pkg | VRAM %6sMB  GTT %6sMB   " \
            "$LIVE_PREFIX" "$ELAPSED" "${CPU_C:--}" "${CPU_MHZ:--}" "${GPU_C:--}" "${SCLK:--}" \
            "${BUSY:--}" "${PKG_W:--}" "${VRAM:--}" "${GTT:--}"
        $LIVE_TTY || echo ""
    fi

    sleep "$INTERVAL"
done

if [ -n "$CMD_PID" ]; then
    wait "$CMD_PID" 2>/dev/null || CMD_RC=$?
fi
$QUIET || echo ""
echo ""

# ─── Summary ─────────────────────────────────────────────────────────────────
awk -F, -v tjmax="$TJMAX" '
NR == 1 { next }
{
    n++
    if ($2 != "") { ct[++cn] = $2+0; if ($2+0 > cmax) cmax = $2+0; csum += $2 }
    if ($3 != "") { if ($3+0 > gmax) gmax = $3+0; gsum += $3; gn++ }
    if ($4 != "") { sclk[++sn] = $4+0; busy[sn] = $6+0 }
    if ($5 != "") { psum += $5; pn++ }
    if ($7 != "" && $7+0 > vpeak) vpeak = $7+0
    if ($8 != "" && $8+0 > gtpeak) gtpeak = $8+0
    last = $1
}
END {
    if (n == 0) { print "  No samples recorded."; exit }
    printf "─────────────────────────────────────────────────────────────\n"
    printf "  %d samples over %d s\n", n, last
    printf "─────────────────────────────────────────────────────────────\n"
    if (cn) printf "  CPU Tctl    : max %.1f °C   avg %.1f °C   (Tjmax %d)\n", cmax, csum/cn, tjmax
    if (gn) printf "  GPU edge    : max %.1f °C   avg %.1f °C\n", gmax, gsum/gn
    if (pn) printf "  Pkg power   : avg %.1f W   (whole SoC — CPU+iGPU share this budget)\n", psum/pn
    if (vpeak)  printf "  VRAM peak   : %d MB\n", vpeak
    if (gtpeak) printf "  GTT peak    : %d MB\n", gtpeak

    # Compare SCLK in the first vs last quarter of the samples where the GPU was
    # actually loaded. A sustained drop there is the signature of SoC throttling.
    if (sn >= 8) {
        q = int(sn/4)
        for (i = 1; i <= q; i++)      if (busy[i] > 30) { a += sclk[i]; an++ }
        for (i = sn-q+1; i <= sn; i++) if (busy[i] > 30) { b += sclk[i]; bn++ }
        if (an && bn) {
            printf "  GPU SCLK    : %.0f MHz early -> %.0f MHz late\n", a/an, b/bn
            if (b/bn < 0.90 * a/an)
                printf "\n  ⚠  SCLK fell %.0f%% under sustained load — throughput numbers\n     from this run are thermally limited, not backend-limited.\n", (1 - (b/bn)/(a/an))*100
        }
    }
    print ""
    if (cmax >= tjmax - 5)
        printf "  ⚠  CPU peaked at %.1f °C, within 5 °C of Tjmax (%d) — expect clock\n     throttling. Treat this run as not comparable to a cool-start run.\n", cmax, tjmax
    else if (cmax >= tjmax - 10)
        printf "  ⚠  CPU peaked at %.1f °C — warm, watch for drift on longer runs.\n", cmax
    else if (cn)
        printf "  ✓  Thermals stayed clear of Tjmax — run is thermally valid.\n"
}
' "$OUT"

echo ""
echo "  Raw CSV: $OUT"
[ -n "$CMD_PID" ] && [ "$CMD_RC" -ne 0 ] && echo "  ⚠  Command exited with code $CMD_RC"
exit "$CMD_RC"
