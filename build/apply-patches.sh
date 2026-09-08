# Sourced by the build scripts, from inside the llama.cpp checkout.
# Applies every patch in patches/ in filename order.
#
# Runs after the pinned commit is checked out, because the build scripts reset
# the tree first. A patch that no longer applies is a hard error: the pin moved
# and the patch needs re-validating, and silently building without it would
# quietly undo a measured improvement.
_patch_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../patches" && pwd)"
if [ -d "$_patch_dir" ]; then
    for _p in "$_patch_dir"/*.patch; do
        [ -e "$_p" ] || continue
        _name="$(basename "$_p")"
        if git apply --check "$_p" 2>/dev/null; then
            git apply "$_p"
            echo "  ✓  applied $_name"
        elif git apply --reverse --check "$_p" 2>/dev/null; then
            echo "  ✓  $_name already applied"
        else
            echo "  ✗  $_name does not apply to $(git rev-parse --short HEAD)" >&2
            echo "     The llama.cpp pin moved. Re-validate the patch against the new" >&2
            echo "     commit and re-run its benchmark before continuing — see patches/README.md." >&2
            exit 1
        fi
    done
fi
