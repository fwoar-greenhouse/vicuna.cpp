#!/usr/bin/env bash
# Compare coding eval label directories. Read-only: deno may only read the given directories.
# usage: ./compare.sh results/label-a results/label-b > compare.md
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v deno >/dev/null; then
    if [ -n "${CODING_EVAL_IN_NIX:-}" ]; then
        echo "deno still missing inside nix shell" >&2
        exit 1
    fi
    export CODING_EVAL_IN_NIX=1
    exec nix shell --inputs-from "$HERE/../.." nixpkgs#deno -c "$0" "$@"
fi

dirs=()
for d in "$@"; do dirs+=("$(cd "$d" && pwd)"); done
read_list="$(IFS=,; echo "${dirs[*]}")"
exec deno run --no-prompt --no-config --no-lock --allow-read="$read_list" "$HERE/compare.js" "${dirs[@]}"
