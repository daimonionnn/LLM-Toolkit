# Sourced by the build scripts. Sets LLAMA_CPP_REF from build/llama.cpp-ref.
# Kept separate so the Dockerfiles can read the same file via --build-arg.
_ref_file="$(dirname "${BASH_SOURCE[0]}")/llama.cpp-ref"
LLAMA_CPP_REF="$(grep -vE '^\s*(#|$)' "$_ref_file" | head -1 | tr -d '[:space:]')"
if [ -z "$LLAMA_CPP_REF" ]; then
    echo "✗  No commit found in $_ref_file" >&2
    exit 1
fi

# Must be a full 40-character SHA. GitHub's smart HTTP will not serve
# `git fetch --depth 1 origin <short-sha>` — it fails with "couldn't find remote
# ref", and a build script that ignores that silently compiles whatever the
# checkout already had. That happened on 2026-09-08 and produced a build
# labelled as one commit but containing another.
if ! printf '%s' "$LLAMA_CPP_REF" | grep -qE '^[0-9a-f]{40}$'; then
    echo "✗  $_ref_file must contain a full 40-character commit SHA, got: $LLAMA_CPP_REF" >&2
    echo "   Short SHAs cannot be fetched from GitHub." >&2
    exit 1
fi
