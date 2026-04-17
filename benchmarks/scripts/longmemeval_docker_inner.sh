#!/usr/bin/env bash
# Run inside the benchmark Docker image (working_dir=/workspace).
# Do not call directly on the host — use run_longmemeval_docker.sh
set -euo pipefail
cd /workspace

mkdir -p data/longmemeval .benchmarks/runs

LM="data/longmemeval/longmemeval_s_cleaned.json"
if [[ ! -f "$LM" ]]; then
  wget -q -O "$LM" \
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
fi

: "${LONGMEMEVAL_OUTPUT:?LONGMEMEVAL_OUTPUT must be set (output JSON path)}"

exec python -m benchmarks.academic.longmemeval.adapter \
  --data-file "$LM" \
  --output "$LONGMEMEVAL_OUTPUT" \
  "$@"
