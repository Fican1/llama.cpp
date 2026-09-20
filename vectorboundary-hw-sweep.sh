#!/bin/bash
# VectorBoundary: real-hardware A/B sweep. Compares llama.cpp's upstream
# VLEN-matched schedule rule against this fork's learned dispatcher on
# whatever real RVV hardware this runs on. The core comparison (the two
# `run` calls below) only needs the one binary tree you already built.
#
# usage: bash vectorboundary-hw-sweep.sh /path/to/model-Q4_0.gguf [out.csv] [norepack_bin_dir]
set -eu
MODEL=${1:?usage: $0 MODEL.gguf [OUT.csv] [norepack_bin_dir]}
OUT=${2:-vb-hw-results.csv}
NOREPACK_BIN=${3:-}
BIN=$(dirname "$0")/build/bin
[ -x "$BIN/llama-bench" ] || { echo "build first: cmake -B build && cmake --build build -j --target llama-bench" >&2; exit 1; }

echo "config,build_commit,cpu_info,backends,model_filename,n_prompt,n_gen,avg_ts" > "$OUT"

run() {
    local name="$1" bindir="$2"; shift 2
    echo ">>> $name"
    env "$@" "$bindir/llama-bench" -m "$MODEL" -p 512 -n 128 -r 3 -o csv 2>/dev/null | tail -n +2 | while IFS=',' read -r -a f; do
        n=${#f[@]}
        build_commit=$(echo "${f[0]}" | tr -d '"')
        cpu_info=$(echo "${f[2]}" | tr -d '"')
        backends=$(echo "${f[4]}" | tr -d '"')
        model_filename=$(echo "${f[5]}" | tr -d '"')
        n_prompt=$(echo "${f[$((n-8))]}" | tr -d '"')
        n_gen=$(echo "${f[$((n-7))]}" | tr -d '"')
        avg_ts=$(echo "${f[$((n-2))]}" | tr -d '"')
        echo "$name,$build_commit,$cpu_info,$backends,$model_filename,$n_prompt,$n_gen,$avg_ts" >> "$OUT"
    done
}

# The two rows that matter: same binary, two env vars.
run "upstream-heuristic" "$BIN" GGML_RVV_DISPATCH=heuristic
run "learned-dispatch"   "$BIN"

# Optional third row: the non-repacked baseline both are trying to beat.
# llama-bench has its own arg parser and does NOT support -nr/--no-repack
# (that's a common_arg only wired up for the other tools), so this needs a
# second binary tree built with repacking compiled out:
#   cmake -B build-norepack -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DGGML_CPU_REPACK=OFF
#   cmake --build build-norepack -j --target llama-bench
# then re-run this script with that dir as the 3rd argument, e.g.:
#   bash vectorboundary-hw-sweep.sh model.gguf out.csv ./build-norepack/bin
if [ -n "$NOREPACK_BIN" ] && [ -x "$NOREPACK_BIN/llama-bench" ]; then
    run "norepack" "$NOREPACK_BIN"
else
    echo ">>> skipping norepack row (optional; see comment in this script for how to add it)"
fi

echo
echo "=== results ($OUT) ==="
column -s, -t "$OUT" 2>/dev/null || cat "$OUT"
echo
echo "Please attach $OUT plus 'uname -a' and 'cat /proc/cpuinfo | head -30' to a new issue at"
echo "https://github.com/Fican1/vectorboundary-artifact/issues/new (title: 'hardware result: <your board>')."
