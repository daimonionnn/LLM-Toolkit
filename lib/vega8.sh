# Shared Vega 8 (gfx90c) detection. Source it; do not execute it.
#
#   source "$SCRIPT_DIR/../lib/vega8.sh"
#
# Four things need finding, and each was previously reimplemented in three or
# four scripts. They drifted, and one of the copies carried a bug that silently
# ran benchmarks on the CPU for months (see vega8_rocm_index below).
#
# Every function honours an override so a machine this heuristic does not suit
# can be driven by hand:
#
#   VEGA8_PCI_ID        PCI device id to match       (default 0x1638, Cezanne/Renoir)
#   VEGA8_RENDER_NODE   /dev/dri/renderDXXX
#   VEGA8_CARD_DIR      /sys/class/drm/cardN
#   VEGA8_ROCM_DEVICE   0-based ROCm agent index
#   VEGA8_VULKAN_DEV    VulkanN
#
# Nothing here writes to stdout except the value being returned, so the
# functions are safe in command substitution.

VEGA8_PCI_ID="${VEGA8_PCI_ID:-0x1638}"

# ─── /sys/class/drm/cardN of the Vega 8 ──────────────────────────────────────
# card* and renderD* numbers are NOT stable across boots or hardware changes:
# this box moved card1 -> card0 on the 26.04 reinstall, and renderD129 ->
# renderD128 when the discrete GPUs left. Match on the PCI device id instead.
vega8_card_dir() {
    if [ -n "${VEGA8_CARD_DIR:-}" ]; then printf '%s\n' "$VEGA8_CARD_DIR"; return 0; fi
    local d
    for d in /sys/class/drm/card[0-9]*; do
        [ -e "$d/device/device" ] || continue
        if [ "$(cat "$d/device/device" 2>/dev/null)" = "$VEGA8_PCI_ID" ]; then
            printf '%s\n' "$d"; return 0
        fi
    done
    return 1
}

# ─── /sys/class/drm/cardN/device — the PCI device dir (sysfs knobs live here) ─
vega8_device_dir() {
    local card; card=$(vega8_card_dir) || return 1
    printf '%s/device\n' "$card"
}

# ─── /dev/dri/renderDXXX of the Vega 8 ───────────────────────────────────────
vega8_render_node() {
    if [ -n "${VEGA8_RENDER_NODE:-}" ]; then printf '%s\n' "$VEGA8_RENDER_NODE"; return 0; fi
    local d node
    for d in /sys/class/drm/renderD*/device; do
        [ -e "$d/device" ] || continue
        if [ "$(cat "$d/device" 2>/dev/null)" = "$VEGA8_PCI_ID" ]; then
            node=$(basename "$(dirname "$d")")
            printf '/dev/dri/%s\n' "$node"; return 0
        fi
    done
    return 1
}

# ─── PCI address, e.g. 0000:0a:00.0 (for DRI_PRIME and friends) ──────────────
vega8_pci_addr() {
    local d; d=$(vega8_device_dir) || return 1
    basename "$(readlink -f "$d")"
}

# ─── 0-based ROCm agent index (what ROCR_VISIBLE_DEVICES takes) ───────────────
# rocminfo prints each agent's "Name: gfxXXX" line BEFORE its "Device Type: GPU"
# line, so remember the last name seen and count GPUs in enumeration order.
#
# `print gpu+0`, never `print gpu`: when the Vega 8 is the first GPU its index
# is 0 and awk's `gpu` was never assigned, so a bare `print gpu` emits an EMPTY
# string. That becomes ROCR_VISIBLE_DEVICES="", which hides every GPU and falls
# back to the CPU while still reporting success. The bug was invisible until
# September 2026, when the discrete GPUs left and the Vega became index 0 for
# the first time -- until then it silently produced CPU benchmark numbers
# labelled as ROCm.
#
# Vega APUs are gfx900/gfx902/gfx909/gfx90c.
vega8_rocm_index() {
    if [ -n "${VEGA8_ROCM_DEVICE:-}" ]; then printf '%s\n' "$VEGA8_ROCM_DEVICE"; return 0; fi
    local bin="${ROCM_PATH:-/opt/rocm}/bin/rocminfo"
    [ -x "$bin" ] || bin="$(command -v rocminfo 2>/dev/null || true)"
    if [ -z "$bin" ]; then printf '0\n'; return 1; fi
    "$bin" 2>/dev/null | awk '
        $1 == "Name:" && $2 ~ /^gfx/  { name = $2 }
        /Device Type:[[:space:]]+GPU/ {
            if (name ~ /^gfx90[029c]$/) { print gpu+0; found = 1; exit }
            gpu++
        }
        END { if (!found) print 0 }
    '
}

# ─── llama.cpp Vulkan device id of the Vega 8 ────────────────────────────────
# $1: a llama.cpp binary that supports --list-devices (llama-server or
# llama-bench). RADV names the Cezanne/Renoir iGPU "RENOIR".
vega8_vulkan_dev() {
    if [ -n "${VEGA8_VULKAN_DEV:-}" ]; then printf '%s\n' "$VEGA8_VULKAN_DEV"; return 0; fi
    local bin="${1:-}" dev
    if [ -n "$bin" ] && [ -x "$bin" ]; then
        dev=$("$bin" --list-devices 2>/dev/null \
              | awk -F: '/RENOIR/ { gsub(/^[ \t]+/, "", $1); print $1; exit }')
    fi
    printf '%s\n' "${dev:-Vulkan0}"
}

# ─── hwmon directory of the Vega 8 ───────────────────────────────────────────
# The GPU's own hwmon, not a global name match: with a discrete GPU present
# there is more than one amdgpu hwmon and the indices are not stable.
vega8_hwmon_dir() {
    local card; card=$(vega8_card_dir) || return 1
    local h
    for h in "$card"/device/hwmon/hwmon[0-9]*; do
        [ -d "$h" ] && { printf '%s\n' "$h"; return 0; }
    done
    return 1
}
