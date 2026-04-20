#!/usr/bin/env bash
# Thin reference benchmarks: LongMemEval (Archivist adapter) + BEIR dense baseline.
#
# Run from repo root (path is auto-detected). Reuses your existing .env — do not overwrite it.
#
# Prerequisites:
#   - .env with LLM_URL, EMBED_URL, QDRANT_URL, VECTOR_DIM (and keys if needed)
#   - Qdrant reachable at QDRANT_URL (e.g. docker compose up -d qdrant)
#   - Same Python you use for dev:  python -m pip install -r requirements.txt
#   - For BEIR: pip install -r requirements-benchmark.txt  OR  SKIP_BEIR=1
#
# Usage:
#   bash benchmarks/scripts/run_thin_reference.sh
#   LIMIT_LM=50 LIMIT_BEIR=100 bash benchmarks/scripts/run_thin_reference.sh
#   SKIP_BEIR=1 bash benchmarks/scripts/run_thin_reference.sh   # LongMemEval only
#   BENCHMARK_FAST=0 bash benchmarks/scripts/run_thin_reference.sh  # full pipeline (slow)
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Warning: no .env in $ROOT — create from .env.example or export LLM_URL/EMBED_URL/QDRANT_URL." >&2
fi

if [[ -f .env && ! -f /.dockerenv ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Fast profile (default): skip index-time reverse HyDE + other optional LLM paths.
# Set BENCHMARK_FAST=0 to use whatever is in .env (full fidelity, much slower).
if [[ "${BENCHMARK_FAST:-1}" != "0" ]]; then
  export REVERSE_HYDE_ENABLED=false
  export SYNTHETIC_QUESTIONS_ENABLED=false
  export QUERY_EXPANSION_ENABLED=false
  export TIERED_CONTEXT_ENABLED=false
  export CONTEXTUAL_AUGMENTATION_ENABLED=false
fi

LIMIT_LM="${LIMIT_LM:-20}"
LIMIT_BEIR="${LIMIT_BEIR:-50}"
OUT_LM="${OUT_LM:-.benchmarks/longmemeval_thin.json}"
OUT_BEIR="${OUT_BEIR:-.benchmarks/beir_nfcorpus_thin.json}"

mkdir -p .benchmarks data/longmemeval data/beir

echo "=============================================="
echo "  Thin reference benchmarks  (repo: $ROOT)"
echo "  LongMemEval limit: $LIMIT_LM  → $OUT_LM"
echo "  BEIR queries:      $LIMIT_BEIR → $OUT_BEIR"
echo "  BENCHMARK_FAST:    ${BENCHMARK_FAST:-1}  (0 = use .env only, no speed overrides)"
if [[ -n "${BENCHMARK_JUDGE_LLM_URL:-}" ]]; then
  echo "  Judge LLM:         ${BENCHMARK_JUDGE_LLM_URL}  (model: ${BENCHMARK_JUDGE_LLM_MODEL:-<unset>})"
else
  echo "  Judge LLM:         (same as LLM_URL / LLM_MODEL — BENCHMARK_JUDGE_LLM_URL empty)"
fi
echo "  Embeddings:       ${EMBED_URL:-<unset>}  model=${EMBED_MODEL:-<unset>}  VECTOR_DIM=${VECTOR_DIM:-<unset>}"
if [[ "${SKIP_BEIR:-0}" == "1" ]]; then
  echo "  BEIR:              skipped (SKIP_BEIR=1)"
fi
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

if [[ "${SKIP_BEIR:-0}" != "1" ]]; then
  echo ""
  echo "[2/2] BEIR NFCorpus thin (dense baseline — not Archivist RLM)..."
  python -c "import beir" 2>/dev/null || python -m pip install -q -r requirements-benchmark.txt

  python -m benchmarks.academic.beir_thin \
    --dataset nfcorpus \
    --limit-queries "$LIMIT_BEIR" \
    --output "$OUT_BEIR"
else
  echo ""
  echo "[2/2] BEIR skipped (SKIP_BEIR=1)"
fi

echo ""
echo "Done."
echo "  LongMemEval: $OUT_LM"
if [[ "${SKIP_BEIR:-0}" != "1" ]]; then
  echo "  BEIR:        $OUT_BEIR"
fi
