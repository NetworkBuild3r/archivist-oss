#!/usr/bin/env bash
# run_longmemeval_ablation.sh — LongMemEval 4-variant ablation ladder.
#
# Runs all four pipeline variants (vector_only → full_pipeline → full_plus_topic
# → full_plus_rerank) in BOTH evaluation modes (retrieval-only and full-pipeline)
# and prints a side-by-side comparison table.
#
# This is the script to run before publishing results.  It shows exactly what
# each pipeline stage adds and prevents cherry-picking a single mode.
#
# Prerequisites:
#   - .env with LLM_URL, EMBED_URL, QDRANT_URL, VECTOR_DIM (and API keys if needed)
#   - Qdrant reachable at QDRANT_URL  (docker compose up -d qdrant)
#   - python -m pip install -r requirements.txt
#
# Usage:
#   bash benchmarks/scripts/run_longmemeval_ablation.sh
#
# Override defaults:
#   LIMIT=50  bash ...                 # quick smoke (default: 0 = all 500 questions)
#   VARIANTS=vector_only,full_pipeline bash ...   # subset of variants
#   SKIP_FULL=1 bash ...               # only retrieval-only pass (faster)
#   OUT_DIR=.benchmarks/runs bash ...  # custom output directory
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Warning: no .env found — create from .env.example or export env vars manually." >&2
fi

if [[ -f .env && ! -f /.dockerenv ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# Fast profile: disable write-time LLM enrichment to keep indexing fast.
# Set BENCHMARK_FAST=0 if you need full-fidelity indexing.
if [[ "${BENCHMARK_FAST:-1}" != "0" ]]; then
  export REVERSE_HYDE_ENABLED=false
  export SYNTHETIC_QUESTIONS_ENABLED=false
  export QUERY_EXPANSION_ENABLED=false
  export TIERED_CONTEXT_ENABLED=false
  export CONTEXTUAL_AUGMENTATION_ENABLED=false
fi

# ── Config ───────────────────────────────────────────────────────────────────
LIMIT="${LIMIT:-0}"
VARIANTS="${VARIANTS:-vector_only,full_pipeline,full_plus_topic,full_plus_rerank}"
SEARCH_LIMIT="${SEARCH_LIMIT:-20}"
SKIP_FULL="${SKIP_FULL:-0}"

STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
GIT_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

OUT_DIR="${OUT_DIR:-.benchmarks/runs}"
mkdir -p "$OUT_DIR" .benchmarks

OUT_RO="${OUT_DIR}/longmemeval_ablation_retrieval_${STAMP}_${GIT_SHA}.json"
OUT_FP="${OUT_DIR}/longmemeval_ablation_full_${STAMP}_${GIT_SHA}.json"
LATEST_RO=".benchmarks/longmemeval_ablation_retrieval_latest.json"
LATEST_FP=".benchmarks/longmemeval_ablation_full_latest.json"

LM="data/longmemeval/longmemeval_s_cleaned.json"
mkdir -p data/longmemeval
if [[ ! -f "$LM" ]]; then
  echo "[download] LongMemEval S (cleaned)..."
  wget -q -O "$LM" \
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
fi

LIMIT_LABEL="all 500"
[[ "$LIMIT" -gt 0 ]] && LIMIT_LABEL="$LIMIT (smoke test — not for publication)"

echo "========================================================"
echo "  LongMemEval Ablation Ladder"
echo "  Git:       ${GIT_BRANCH} @ ${GIT_SHA}"
echo "  Variants:  $VARIANTS"
echo "  Questions: $LIMIT_LABEL"
echo "  Search-k:  $SEARCH_LIMIT"
echo "  FAST:      ${BENCHMARK_FAST:-1}"
if [[ -n "${BENCHMARK_JUDGE_LLM_URL:-}" ]]; then
  echo "  Judge:     ${BENCHMARK_JUDGE_LLM_URL}  (model: ${BENCHMARK_JUDGE_LLM_MODEL:-<unset>})"
else
  echo "  Judge:     (same as LLM_URL / LLM_MODEL)"
fi
echo "  Outputs:"
echo "    retrieval-only → $OUT_RO"
[[ "$SKIP_FULL" != "1" ]] && echo "    full-pipeline  → $OUT_FP"
echo "========================================================"

# Build common args
COMMON_ARGS=(
  --data-file "$LM"
  --ablation
  --variants "$VARIANTS"
  --search-limit "$SEARCH_LIMIT"
)
[[ "$LIMIT" -gt 0 ]] && COMMON_ARGS+=(--limit "$LIMIT")

# ── Pass 1: retrieval-only (no LLM refine) ───────────────────────────────────
echo ""
echo "[1/2] Retrieval-only pass (--mode retrieval-only)..."
python -m benchmarks.academic.longmemeval.adapter \
  "${COMMON_ARGS[@]}" \
  --mode retrieval-only \
  --output "$OUT_RO"

ln -sfn "runs/$(basename "$OUT_RO")" "$LATEST_RO"
echo "  → $OUT_RO"

# ── Pass 2: full-pipeline (LLM refine) ───────────────────────────────────────
if [[ "$SKIP_FULL" == "1" ]]; then
  echo ""
  echo "[2/2] Full-pipeline pass skipped (SKIP_FULL=1)."
else
  echo ""
  echo "[2/2] Full-pipeline pass (--mode full)..."
  python -m benchmarks.academic.longmemeval.adapter \
    "${COMMON_ARGS[@]}" \
    --mode full \
    --output "$OUT_FP"

  ln -sfn "runs/$(basename "$OUT_FP")" "$LATEST_FP"
  echo "  → $OUT_FP"
fi

# ── Combined comparison ───────────────────────────────────────────────────────
echo ""
if [[ "$SKIP_FULL" != "1" && -f "$OUT_RO" && -f "$OUT_FP" ]]; then
  python3 - "$OUT_RO" "$OUT_FP" "$LIMIT_LABEL" "$VARIANTS" <<'PYEOF'
import json, sys

ro_path, fp_path, limit_label, variants_str = sys.argv[1:5]

with open(ro_path) as f:
    ro_data = json.load(f)
with open(fp_path) as f:
    fp_data = json.load(f)

ro_variants = ro_data.get("variants", {})
fp_variants = fp_data.get("variants", {})
all_variants = list(dict.fromkeys(list(ro_variants.keys()) + list(fp_variants.keys())))

w = 22
fmt_acc = lambda v: f"{v:.1%}" if isinstance(v, (int, float)) else "—"
fmt_flt = lambda v: f"{v:.4f}" if isinstance(v, (int, float)) else "—"

print()
print("=" * 90)
print("  LongMemEval Ablation — Dual-Mode Summary")
print(f"  Questions: {limit_label}")
print("=" * 90)
print(f"  {'Variant':<{w}}  {'retrieval-only':^29}  {'full-pipeline':^29}")
print(f"  {'':^{w}}  {'Accuracy':>10}  {'R@5':>8}  {'NDCG@5':>8}  {'Accuracy':>10}  {'R@5':>8}  {'NDCG@5':>8}")
print(f"  {'-'*w}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*10}  {'-'*8}  {'-'*8}")

for vname in all_variants:
    ro_s = ro_variants.get(vname, {}).get("summary", {})
    fp_s = fp_variants.get(vname, {}).get("summary", {})
    print(
        f"  {vname:<{w}}"
        f"  {fmt_acc(ro_s.get('overall_accuracy')):>10}"
        f"  {fmt_flt(ro_s.get('retrieval', {}).get('recall@5')):>8}"
        f"  {fmt_flt(ro_s.get('retrieval', {}).get('ndcg@5')):>8}"
        f"  {fmt_acc(fp_s.get('overall_accuracy')):>10}"
        f"  {fmt_flt(fp_s.get('retrieval', {}).get('recall@5')):>8}"
        f"  {fmt_flt(fp_s.get('retrieval', {}).get('ndcg@5')):>8}"
    )

first = all_variants[0] if all_variants else None
last  = all_variants[-1] if len(all_variants) > 1 else None
if first and last and first != last:
    fo_acc = fp_variants.get(first, {}).get("summary", {}).get("overall_accuracy")
    la_acc = fp_variants.get(last,  {}).get("summary", {}).get("overall_accuracy")
    if isinstance(fo_acc, (int, float)) and isinstance(la_acc, (int, float)):
        delta = la_acc - fo_acc
        sign = "+" if delta >= 0 else ""
        print(f"\n  Pipeline lift ({first} → {last}, full-pipeline mode): {sign}{delta:.1%}")

print("=" * 90)
print("  Note: R@5 is stable across variants — same retrieval candidates, different synthesis.")
print("=" * 90)
PYEOF
fi

echo ""
echo "Ablation complete."
echo "  retrieval-only: $OUT_RO  (symlink: $LATEST_RO)"
[[ "$SKIP_FULL" != "1" ]] && echo "  full-pipeline:  $OUT_FP  (symlink: $LATEST_FP)"
echo ""
echo "To reproduce:"
echo "  bash benchmarks/scripts/run_longmemeval_ablation.sh"
echo "  # or for a quick smoke:"
echo "  LIMIT=20 bash benchmarks/scripts/run_longmemeval_ablation.sh"
