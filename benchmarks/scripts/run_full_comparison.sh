#!/usr/bin/env bash
# Full Comparison: MD Files (Context Stuffing) vs Archivist across all corpus scales.
#
# Generates all four corpora (small/medium/large/stress), runs LLM-backed curator
# extraction to populate the knowledge graph (required for v1.7 entity-anchored retrieval),
# runs context stuffing with real LLM calls, runs Archivist retrieval
# (vector_only + full_pipeline), and produces a comprehensive side-by-side comparison.
#
# Shows:
#   1. Where MD file stuffing degrades / overflows the context window.
#   2. Where Archivist entity-anchored retrieval excels over raw context stuffing.
#   3. Per-query-type breakdown (single_hop, multi_hop, temporal, needle, etc.).
#   4. Break-even table across scales.
#
# Usage (from repo root):
#   ./benchmarks/scripts/run_full_comparison.sh
#
# Recommended (inside Docker — Qdrant already running):
#   docker compose up -d qdrant
#   docker compose --profile benchmark build benchmark
#   docker compose --profile benchmark run --rm --entrypoint /bin/bash benchmark \
#     benchmarks/scripts/run_full_comparison.sh
#
# Environment overrides:
#   CONTEXT_BUDGET   — token budget for overflow detection (default: 32768)
#   VARIANTS         — comma-separated evaluate.py variants (default: vector_only,full_pipeline)
#   SCALES           — space-separated scales to run (default: "small medium large stress")
#   SKIP_CORPUS_GEN  — set to 1 to skip corpus generation if already present
#
# With --output, evaluate.py auto-writes per-scale checkpoints: .benchmarks/full_<scale>.run_state.json
# Pass extra evaluate.py flags after the script name, e.g.:
#   ./benchmarks/scripts/run_full_comparison.sh --progress-pct 5 --no-checkpoint
#
# Script-only flags (not passed to evaluate.py):
#   -d, --debug   — enable bash trace (set -x) for troubleshooting

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

EVAL_EXTRA=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        -d|--debug)
            set -x
            shift
            ;;
        *)
            EVAL_EXTRA+=("$1")
            shift
            ;;
    esac
done

FIXTURES_DIR="benchmarks/fixtures"
OUT_DIR=".benchmarks"
mkdir -p "$OUT_DIR"

CONTEXT_BUDGET="${CONTEXT_BUDGET:-32768}"
VARIANTS="${VARIANTS:-vector_only,full_pipeline}"
SCALES="${SCALES:-small medium large stress}"
SKIP_CORPUS_GEN="${SKIP_CORPUS_GEN:-0}"

echo "======================================================="
echo "  Full Benchmark: MD Files vs Archivist Memory System  "
echo "======================================================="
echo "  Context budget : ${CONTEXT_BUDGET} tokens"
echo "  LLM calls      : YES (stuffing + curator extraction)"
echo "  Variants       : ${VARIANTS}"
echo "  Scales         : ${SCALES}"
echo "  Output dir     : ${OUT_DIR}/"
echo "  Checkpoints    : ${OUT_DIR}/full_<scale>.run_state.json (use --no-checkpoint to disable)"
echo "======================================================="
echo ""

# ── 1. Generate corpora ─────────────────────────────────────────────────────
if [[ "$SKIP_CORPUS_GEN" == "1" ]]; then
    echo "[corpus] Skipping generation (SKIP_CORPUS_GEN=1)"
else
    echo "[corpus] Generating any missing corpora ..."
    for preset in $SCALES; do
        corpus_path="${FIXTURES_DIR}/corpus_${preset}"
        if [[ -d "${corpus_path}/agents" ]]; then
            echo "  corpus_${preset} already exists — skipping."
        else
            echo "  Generating corpus_${preset} ..."
            python benchmarks/fixtures/generate_corpus.py --preset "$preset" --corpus-only
            echo "  Done: corpus_${preset}"
        fi
    done
fi
echo ""

# ── 2. Per-scale runs ────────────────────────────────────────────────────────
for scale in $SCALES; do
    echo "-------------------------------------------------------"
    echo "  Scale: ${scale}"
    echo "-------------------------------------------------------"

    out_file="${OUT_DIR}/full_${scale}.json"

    python -m benchmarks.pipeline.evaluate \
        --memory-scale "$scale" \
        --variants "$VARIANTS" \
        --no-refine \
        --run-curator \
        --compare-stuffing \
        --stuffing-call-llm \
        --context-budget "$CONTEXT_BUDGET" \
        --print-slices \
        --output "$out_file" \
        "${EVAL_EXTRA[@]}"

    echo ""
    echo "  Results written: ${out_file}"
    echo ""
done

# ── 3. Merge all scale JSONs into a unified report ───────────────────────────
echo "-------------------------------------------------------"
echo "  Merging results across all scales ..."
echo "-------------------------------------------------------"
echo ""

python - <<'PYEOF'
import json
import os
import sys
import pathlib

root = pathlib.Path(".")
out_dir = root / ".benchmarks"
scales = os.environ.get("SCALES", "small medium large stress").split()

all_stuffing: list[dict] = []
all_archivist: list[dict] = []
scale_files: list[str] = []

for scale in scales:
    p = out_dir / f"full_{scale}.json"
    if not p.exists():
        print(f"  WARN: {p} not found, skipping", file=sys.stderr)
        continue
    data = json.loads(p.read_text())
    scale_files.append(str(p))

    # Support both single-scale (stuffing_summaries list) and scale-sweep (by_scale)
    for s in data.get("stuffing_summaries", []):
        s.setdefault("memory_scale", scale)
        all_stuffing.append(s)

    for s in data.get("summaries", []):
        s.setdefault("memory_scale", scale)
        all_archivist.append(s)

# Build combined JSON
combined = {
    "scales_run": scales,
    "scale_files": scale_files,
    "stuffing_summaries": all_stuffing,
    "archivist_summaries": all_archivist,
}

out_json = out_dir / "full_comparison.json"
out_json.write_text(json.dumps(combined, indent=2))
print(f"  Combined JSON: {out_json}")

# Build markdown report
lines = [
    "# Archivist vs MD Files (Context Stuffing) — Full Benchmark Report",
    "",
    f"Context budget: **{os.environ.get('CONTEXT_BUDGET', '32768')} tokens**  ",
    "LLM: real LLM calls for both stuffing answers and curator extraction.  ",
    "Archivist variants run: vector_only, full_pipeline.",
    "",
    "---",
    "",
]

for scale in scales:
    p = out_dir / f"full_{scale}.json"
    if not p.exists():
        continue
    data = json.loads(p.read_text())

    lines.append(f"## Scale: {scale}")
    lines.append("")

    if bt := data.get("breakeven_table"):
        lines.append("### Break-Even Table")
        lines.append("")
        lines.append(bt)
        lines.append("")

    if ft := data.get("full_comparison_table"):
        lines.append("### Per-Query-Type Breakdown")
        lines.append("")
        lines.append(ft)
        lines.append("")

    # Archivist variant table
    if summaries := data.get("summaries"):
        lines.append("### Archivist Variant Results")
        lines.append("")
        lines.append("| Variant | Recall@5 | Recall@10 | MRR | p50 ms | p95 ms | Tok/Q |")
        lines.append("|---------|----------|-----------|-----|--------|--------|-------|")
        for s in summaries:
            lines.append(
                f"| {s.get('variant','?')} "
                f"| {s.get('recall_at_5', 0):.4f} "
                f"| {s.get('recall_at_10', 0):.4f} "
                f"| {s.get('mrr', 0):.4f} "
                f"| {s.get('latency_p50_ms', 0):.0f} "
                f"| {s.get('latency_p95_ms', 0):.0f} "
                f"| {s.get('avg_tokens_per_query', 0):.0f} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("")

out_md = out_dir / "full_comparison.md"
out_md.write_text("\n".join(lines))
print(f"  Markdown report: {out_md}")
PYEOF

echo ""
echo "======================================================="
echo "  Done!  Full comparison report:"
echo "    JSON : .benchmarks/full_comparison.json"
echo "    MD   : .benchmarks/full_comparison.md"
echo "======================================================="
