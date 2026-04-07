#!/usr/bin/env bash
# Seed the knowledge graph from fixture glossary and run full_pipeline + full_pipeline_rerank.
# No LLM curator calls — uses deterministic entity population.
#
# Usage:
#   ./benchmarks/scripts/warm_graph.sh                    # default: medium corpus
#   ./benchmarks/scripts/warm_graph.sh --memory-scale large
#   ./benchmarks/scripts/warm_graph.sh large              # shorthand
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SCALE="medium"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --memory-scale) SCALE="$2"; shift 2 ;;
    -*) echo "Unknown flag: $1"; exit 1 ;;
    *) SCALE="$1"; shift ;;
  esac
done

echo "=== Warm-graph benchmark: scale=${SCALE} ==="
echo "Step 1: Generate corpus (if missing)"
python benchmarks/fixtures/generate_corpus.py --preset "$SCALE" --corpus-only 2>/dev/null || true
python benchmarks/fixtures/generate_corpus.py --write-questions 2>/dev/null || true

echo "Step 2: Run benchmark with --warm-graph"
python -m benchmarks.pipeline.evaluate \
    --memory-scale "$SCALE" \
    --warm-graph \
    --variants full_pipeline,full_pipeline_rerank \
    --no-refine \
    --print-slices \
    --output ".benchmarks/warm_graph_${SCALE}.json"

echo "=== Done. Results in .benchmarks/warm_graph_${SCALE}.json ==="
