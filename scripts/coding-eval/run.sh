#!/usr/bin/env bash
# Run the coding eval harness with only the permissions it needs:
#   network:   the endpoint's host only
#   read:      this directory (problems, sandbox) and the output directory
#   write:     the output directory only
#   run:       deno (the sandbox evaluator) and bubblewrap only
#   env:       CODING_EVAL_TOKEN only
#
# usage: ./run.sh --endpoint http://127.0.0.1:8001 --model MODEL [--out DIR] [options]   (see --help)
#        CODING_EVAL_TOKEN=sk-... ./run.sh --endpoint https://openrouter.ai/api/v1 --model vendor/model
#        ./run.sh --self-check
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Get deno and bubblewrap from the repo's pinned nixpkgs if they are not on PATH.
if ! command -v deno >/dev/null || ! command -v bwrap >/dev/null; then
    if [ -n "${CODING_EVAL_IN_NIX:-}" ]; then
        echo "deno or bwrap still missing inside nix shell" >&2
        exit 1
    fi
    export CODING_EVAL_IN_NIX=1
    exec nix shell --inputs-from "$HERE/../.." nixpkgs#deno nixpkgs#bubblewrap -c "$0" "$@"
fi

DENO="$(readlink -f "$(command -v deno)")"
BWRAP="$(readlink -f "$(command -v bwrap)")"

endpoint=""
out="results"
regrade=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    case "${args[$i]}" in
        --endpoint) endpoint="${args[$((i + 1))]:-}" ;;
        --out) out="${args[$((i + 1))]:-}" ;;
        --regrade) regrade=1 ;;
    esac
done

# Check that bubblewrap can create namespaces here; fall back to deno-only isolation otherwise.
if ! "$BWRAP" --unshare-all --ro-bind /nix/store /nix/store --dev /dev "$DENO" --version >/dev/null 2>&1; then
    echo "warning: bubblewrap does not work here, running the sandbox with deno permissions only" >&2
    BWRAP=none
fi

perms=(--allow-read="$HERE" --allow-env=CODING_EVAL_TOKEN)
run_perm="$DENO"
[ "$BWRAP" != none ] && run_perm="$run_perm,$BWRAP"
perms+=(--allow-run="$run_perm")
# --regrade only reads and rewrites the results: no network access
if [ -n "$endpoint" ] && [ -z "$regrade" ]; then
    hostport="$(printf '%s' "$endpoint" | sed -E 's#^[A-Za-z][A-Za-z0-9+.-]*://##; s#/.*$##; s#^[^@]*@##')"
    perms+=(--allow-net="$hostport")
fi
if [ -n "$endpoint" ] || [ -n "$regrade" ]; then
    mkdir -p "$out"
    out_abs="$(cd "$out" && pwd)"
    perms+=(--allow-read="$HERE,$out_abs" --allow-write="$out_abs")
fi

exec "$DENO" run --no-prompt --no-config --no-lock "${perms[@]}" "$HERE/harness.js" --deno "$DENO" --bwrap "$BWRAP" "$@"
