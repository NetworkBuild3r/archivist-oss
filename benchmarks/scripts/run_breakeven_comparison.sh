#!/usr/bin/env bash
# Break-even comparison: context stuffing vs Archivist retrieval across all corpus scales.
#
# Generates any missing corpora (small/medium/large), then runs the pipeline benchmark
# with --compare-stuffing to produce the side-by-side table showing where context stuffing
# overflows the context window and Archivist retrieval becomes the only viable approach.
#
# Usage (from repo root):
#   ./benchmarks/scripts/run_breakeven_comparison.sh
#
# Run inside Docker (recommended — Qdrant and Tailscale already running):
#   docker compose up -d qdrant tailscale
#   docker compose --profile benchmark run --rm benchmark \
#     /bin/bash benchmarks/scripts/run_breakeven_comparison.sh
#
# Extra args are passed through to evaluate.py:
#   ./benchmarks/scripts/run_breakeven_comparison.sh --stuffing-call-llm   # actually call LLM for stuffing
#   ./benchmarks/scripts/run_breakeven_comparison.sh --limit 20            # shorter run
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

FIXTURES_DIR="benchmarks/fixtures"

# Context budget for overflow detection.
# Use a realistic agent context window — e.g. 32768 (common practical limit when
# system prompt, conversation history, and tools are also sharing the window).
# Pass --context-budget N to override.
CONTEXT_BUDGET=${BREAKEVEN_CONTEXT_BUDGET:-32768}

echo "=== Break-Even Benchmark: Context Stuffing vs Archivist ==="
echo "    Context budget: ${CONTEXT_BUDGET} tokens"
echo ""

# Generate any missing scale corpora (including stress for the hard overflow case).
for preset in small medium large stress; do
  if [[ ! -d "$FIXTURES_DIR/corpus_${preset}/agents" ]]; then
    echo "Generating corpus_${preset} ..."
    python benchmarks/fixtures/generate_corpus.py --preset "$preset" --corpus-only
    echo "  Done."
  else
    echo "corpus_${preset} already exists — skipping generation."
  fi
done

echo ""
echo "Running scale sweep (small/medium/large) with --compare-stuffing ..."
echo "(Token-count-only mode for stuffing by default; use --stuffing-call-llm for actual LLM answers.)"
echo ""

# Scale sweep (small/medium/large).
python -m benchmarks.pipeline.evaluate \
  --scale-sweep \
  --variants vector_only,full_pipeline \
  --no-refine \
  --compare-stuffing \
  --context-budget "$CONTEXT_BUDGET" \
  --print-slices \
  --output .benchmarks/breakeven_sweep.json \
  "$@"

echo ""
echo "Running stress corpus (1500+ files) ..."
echo ""

# Stress corpus — run separately (not in scale-sweep which only covers small/medium/large).
python -m benchmarks.pipeline.evaluate \
  --memory-scale stress \
  --variants vector_only,full_pipeline \
  --no-refine \
  --compare-stuffing \
  --context-budget "$CONTEXT_BUDGET" \
  --print-slices \
  --output .benchmarks/breakeven_stress.json \
  "$@"

echo ""
echo "Done. Results:"
echo "  .benchmarks/breakeven_sweep.json  (small/medium/large)"
echo "  .benchmarks/breakeven_stress.json (stress — ~1500 files)"
