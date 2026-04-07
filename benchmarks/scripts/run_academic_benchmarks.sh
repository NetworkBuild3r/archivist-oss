#!/usr/bin/env bash
# run_academic_benchmarks.sh — Download datasets and run all three academic
# benchmark adapters (LongMemEval, LoCoMo, HaluMem) with consistent settings.
#
# Usage:
#   bash benchmarks/scripts/run_academic_benchmarks.sh [--limit N] [--no-curator] [--no-refine]
#
# From Docker:
#   docker compose run --rm --entrypoint /bin/bash benchmark \
#     benchmarks/scripts/run_academic_benchmarks.sh
#
# Output:
#   .benchmarks/longmemeval_results.json
#   .benchmarks/locomo_results.json
#   .benchmarks/halumem_results.json
#   .benchmarks/academic_scores.json  (merged summary)

set -euo pipefail

LIMIT=0
CURATOR="--run-curator"
REFINE=""
DATA_ROOT="data"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --limit)   LIMIT="$2"; shift 2 ;;
    --no-curator) CURATOR=""; shift ;;
    --no-refine)  REFINE="--no-refine"; shift ;;
    --data-root)  DATA_ROOT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

OUT_DIR=".benchmarks"
mkdir -p "$OUT_DIR" "$DATA_ROOT"

echo "========================================"
echo "  Archivist Academic Benchmark Suite"
echo "========================================"
echo "  Limit:    ${LIMIT:-all}"
echo "  Curator:  ${CURATOR:-disabled}"
echo "  Refine:   ${REFINE:-enabled}"
echo "  Data dir: $DATA_ROOT"
echo "========================================"

# ── 1. Download datasets if missing ──────────────────────────────────────────

echo ""
echo "[1/6] Checking datasets..."

# LongMemEval
LONGMEM_DIR="$DATA_ROOT/longmemeval"
LONGMEM_FILE="$LONGMEM_DIR/longmemeval_s_cleaned.json"
if [ ! -f "$LONGMEM_FILE" ]; then
  echo "  Downloading LongMemEval..."
  mkdir -p "$LONGMEM_DIR"
  wget -q -O "$LONGMEM_FILE" \
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json" \
    || echo "  ⚠ Failed to download LongMemEval (manual download required)"
  wget -q -O "$LONGMEM_DIR/longmemeval_oracle.json" \
    "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_oracle.json" \
    2>/dev/null || true
else
  echo "  ✓ LongMemEval already present"
fi

# LoCoMo
LOCOMO_DIR="$DATA_ROOT/locomo"
if [ ! -d "$LOCOMO_DIR" ] || [ -z "$(ls -A "$LOCOMO_DIR" 2>/dev/null)" ]; then
  echo "  Downloading LoCoMo..."
  git clone --depth=1 https://github.com/snap-research/locomo.git "$LOCOMO_DIR" \
    2>/dev/null || echo "  ⚠ Failed to clone LoCoMo (manual download required)"
else
  echo "  ✓ LoCoMo already present"
fi

# HaluMem
HALUMEM_DIR="$DATA_ROOT/halumem"
if [ ! -d "$HALUMEM_DIR" ] || [ -z "$(ls -A "$HALUMEM_DIR" 2>/dev/null)" ]; then
  echo "  Downloading HaluMem..."
  git clone --depth=1 https://github.com/MemTensor/HaluMem.git "$HALUMEM_DIR" \
    2>/dev/null || echo "  ⚠ Failed to clone HaluMem (manual download required)"
else
  echo "  ✓ HaluMem already present"
fi

# ── 2. Install evaluation deps ──────────────────────────────────────────────

echo ""
echo "[2/6] Checking dependencies..."
pip install -q rouge-score nltk 2>/dev/null || true

# ── 3. Run LongMemEval ──────────────────────────────────────────────────────

echo ""
echo "[3/6] Running LongMemEval..."
LONGMEM_OUT="$OUT_DIR/longmemeval_results.json"
LONGMEM_ARGS="--data-file $LONGMEM_FILE --output $LONGMEM_OUT $CURATOR $REFINE"
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
  LONGMEM_ARGS="$LONGMEM_ARGS --limit $LIMIT"
fi

if [ -f "$LONGMEM_FILE" ]; then
  python -m benchmarks.academic.longmemeval.adapter $LONGMEM_ARGS || {
    echo "  ⚠ LongMemEval failed (continuing)"
    echo '{"error":"adapter failed"}' > "$LONGMEM_OUT"
  }
else
  echo "  ⚠ LongMemEval dataset not found, skipping"
  echo '{"error":"dataset not found"}' > "$LONGMEM_OUT"
fi

# ── 4. Run LoCoMo ───────────────────────────────────────────────────────────

echo ""
echo "[4/6] Running LoCoMo..."
LOCOMO_OUT="$OUT_DIR/locomo_results.json"
LOCOMO_ARGS="--data-dir $LOCOMO_DIR --output $LOCOMO_OUT $CURATOR $REFINE"
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
  LOCOMO_ARGS="$LOCOMO_ARGS --limit $LIMIT"
fi

if [ -d "$LOCOMO_DIR" ] && [ -n "$(ls -A "$LOCOMO_DIR" 2>/dev/null)" ]; then
  python -m benchmarks.academic.locomo.adapter $LOCOMO_ARGS || {
    echo "  ⚠ LoCoMo failed (continuing)"
    echo '{"error":"adapter failed"}' > "$LOCOMO_OUT"
  }
else
  echo "  ⚠ LoCoMo dataset not found, skipping"
  echo '{"error":"dataset not found"}' > "$LOCOMO_OUT"
fi

# ── 5. Run HaluMem ──────────────────────────────────────────────────────────

echo ""
echo "[5/6] Running HaluMem..."
HALUMEM_OUT="$OUT_DIR/halumem_results.json"
HALUMEM_ARGS="--data-dir $HALUMEM_DIR --output $HALUMEM_OUT $CURATOR"
if [ "$LIMIT" -gt 0 ] 2>/dev/null; then
  HALUMEM_ARGS="$HALUMEM_ARGS --limit $LIMIT"
fi

if [ -d "$HALUMEM_DIR" ] && [ -n "$(ls -A "$HALUMEM_DIR" 2>/dev/null)" ]; then
  python -m benchmarks.academic.halumem.adapter $HALUMEM_ARGS || {
    echo "  ⚠ HaluMem failed (continuing)"
    echo '{"error":"adapter failed"}' > "$HALUMEM_OUT"
  }
else
  echo "  ⚠ HaluMem dataset not found, skipping"
  echo '{"error":"dataset not found"}' > "$HALUMEM_OUT"
fi

# ── 6. Merge results ────────────────────────────────────────────────────────

echo ""
echo "[6/6] Merging results..."

python3 - "$LONGMEM_OUT" "$LOCOMO_OUT" "$HALUMEM_OUT" "$OUT_DIR/academic_scores.json" <<'PYEOF'
import json, sys
from datetime import datetime

longmem_path, locomo_path, halumem_path, out_path = sys.argv[1:5]

merged = {
    "timestamp": datetime.now().isoformat(),
    "version": "v1.10.0",
    "benchmarks": {},
}

for name, path in [("LongMemEval", longmem_path), ("LoCoMo", locomo_path), ("HaluMem", halumem_path)]:
    try:
        with open(path) as f:
            data = json.load(f)
        if "error" not in data and "summary" in data:
            merged["benchmarks"][name] = data["summary"]
        elif "error" not in data:
            merged["benchmarks"][name] = data
        else:
            merged["benchmarks"][name] = {"status": "skipped", "reason": data.get("error", "unknown")}
    except Exception as e:
        merged["benchmarks"][name] = {"status": "error", "reason": str(e)}

with open(out_path, "w") as f:
    json.dump(merged, f, indent=2)

print("\n" + "=" * 60)
print("  Academic Benchmark Summary")
print("=" * 60)

for name, bench in merged["benchmarks"].items():
    if bench.get("status") in ("skipped", "error"):
        print(f"  {name:20s}  SKIPPED ({bench.get('reason', '')})")
        continue
    recall = bench.get("overall_keyword_recall", bench.get("overall_f1", bench.get("composite_score", 0)))
    questions = bench.get("evaluated_questions", bench.get("users_evaluated", "?"))
    print(f"  {name:20s}  score={recall:.4f}  n={questions}")

print("=" * 60)
print(f"\nMerged results: {out_path}")
PYEOF

echo ""
echo "✅ Academic benchmark suite complete."
echo "   Results in: $OUT_DIR/"
echo "   Load into dashboard: open docs/benchmark-dashboard.html"
