#!/usr/bin/env bash
# Thin reference benchmarks: LongMemEval (Archivist adapter) + BEIR dense baseline.
#
# Prerequisites:
#   - Host: cp .env.example .env  (LLM_URL, EMBED_URL, QDRANT_URL, VECTOR_DIM, …)
#   - docker compose up -d qdrant
#   - pip install -r requirements.txt
#   - pip install -r requirements-benchmark.txt   # for BEIR only
#
# Usage:
#   bash benchmarks/scripts/run_thin_reference.sh
#   LIMIT_LM=50 LIMIT_BEIR=100 bash benchmarks/scripts/run_thin_reference.sh
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -f .env && ! -f /.dockerenv ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

LIMIT_LM="${LIMIT_LM:-20}"
LIMIT_BEIR="${LIMIT_BEIR:-50}"
OUT_LM="${OUT_LM:-.benchmarks/longmemeval_thin.json}"
OUT_BEIR="${OUT_BEIR:-.benchmarks/beir_nfcorpus_thin.json}"

mkdir -p .benchmarks data/longmemeval data/beir

echo "=============================================="
echo "  Thin reference benchmarks"
echo "  LongMemEval limit: $LIMIT_LM  → $OUT_LM"
echo "  BEIR queries:      $LIMIT_BEIR → $OUT_BEIR"
echo "=============================================="

# --- LongMemEval (full Archivist stack: Qdrant + index + retrieve + judge) ---
LM="data/longmemeval/longmemeval_s_cleaned.json"
if [[ ! -f "$LM" ]]; then
  echo "[download] LongMemEval S (cleaned)..."
  wget -q -O "$LM" \
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
fi

echo ""
echo "[1/2] LongMemEval adapter (limit=$LIMIT_LM, --no-refine for speed)..."
# shellcheck disable=SC2086
python -m benchmarks.academic.longmemeval.adapter \
  --data-file "$LM" \
  --limit "$LIMIT_LM" \
  --no-refine \
  --output "$OUT_LM"

echo ""
echo "[2/2] BEIR NFCorpus thin (dense baseline — not Archivist RLM)..."
python -c "import beir" 2>/dev/null || pip install -q -r requirements-benchmark.txt

python -m benchmarks.academic.beir_thin \
  --dataset nfcorpus \
  --limit-queries "$LIMIT_BEIR" \
  --output "$OUT_BEIR"

echo ""
echo "Done."
echo "  LongMemEval: $OUT_LM"
echo "  BEIR:        $OUT_BEIR"
