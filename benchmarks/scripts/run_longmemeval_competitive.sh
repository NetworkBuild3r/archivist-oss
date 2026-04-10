#!/usr/bin/env bash
# run_longmemeval_competitive.sh — Run LongMemEval ablation across pipeline
# variants and produce an ablation comparison table.
#
# Variants (additive):
#   1. vector_only        — baseline vector search only
#   2. full_pipeline      — BM25 + graph + temporal decay + rescue + adaptive
#   3. full_plus_topic    — above + topic-room routing
#   4. full_plus_rerank   — above + rerank
#
# Usage:
#   bash benchmarks/scripts/run_longmemeval_competitive.sh
#   bash benchmarks/scripts/run_longmemeval_competitive.sh --limit 50 --no-refine
#   bash benchmarks/scripts/run_longmemeval_competitive.sh --variants vector_only,full_pipeline
#
# Output:
#   .benchmarks/longmemeval_ablation.json

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
if [[ -f .env && ! -f /.dockerenv ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

LIMIT=0
REFINE=""
CURATOR="--run-curator"
VARIANTS=""
DATA_ROOT="data"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)      LIMIT="$2"; shift 2 ;;
    --no-refine)  REFINE="--no-refine"; shift ;;
    --no-curator) CURATOR=""; shift ;;
    --variants)   VARIANTS="--variants $2"; shift 2 ;;
    --data-root)  DATA_ROOT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

OUT_DIR=".benchmarks"
mkdir -p "$OUT_DIR" "$DATA_ROOT"

# ── Download dataset if missing ──────────────────────────────────────────────
LONGMEM_DIR="$DATA_ROOT/longmemeval"
LONGMEM_FILE="$LONGMEM_DIR/longmemeval_s_cleaned.json"
if [ ! -f "$LONGMEM_FILE" ]; then
  echo "Downloading LongMemEval dataset..."
  mkdir -p "$LONGMEM_DIR"
  wget -q -O "$LONGMEM_FILE" \
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" \
    || { echo "ERROR: Failed to download LongMemEval. Download manually."; exit 1; }
fi

# ── Run ablation ─────────────────────────────────────────────────────────────
echo "========================================"
echo "  LongMemEval Competitive Ablation"
echo "========================================"
echo "  Dataset:  $LONGMEM_FILE"
echo "  Limit:    ${LIMIT:-all}"
echo "  Curator:  ${CURATOR:-disabled}"
echo "  Refine:   ${REFINE:-enabled}"
echo "========================================"

ARGS="--data-file $LONGMEM_FILE --ablation --output $OUT_DIR/longmemeval_ablation.json $CURATOR $REFINE $VARIANTS"
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
  ARGS="$ARGS --limit $LIMIT"
fi

python -m benchmarks.academic.longmemeval.adapter $ARGS

echo ""
echo "Results written to: $OUT_DIR/longmemeval_ablation.json"
echo "Load into dashboard: open docs/benchmark-dashboard.html"
