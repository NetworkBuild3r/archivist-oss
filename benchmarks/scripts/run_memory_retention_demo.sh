#!/usr/bin/env bash
# Generate the stress corpus if missing, then run the pipeline benchmark with slice table.
# Same question mix as --memory-scale large; corpus has ~1.5k+ files + noise for haystack tests.
#
# Usage (from repo root):
#   ./benchmarks/scripts/run_memory_retention_demo.sh
#   ./benchmarks/scripts/run_memory_retention_demo.sh --limit 20   # extra args pass through
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ ! -d benchmarks/fixtures/corpus_stress/agents ]]; then
  echo "Generating stress corpus under benchmarks/fixtures/corpus_stress/ (first run may take several minutes)..."
  python benchmarks/fixtures/generate_corpus.py --preset stress --corpus-only
fi

exec python -m benchmarks.pipeline.evaluate \
  --memory-scale stress \
  --variants vector_only,full_pipeline \
  --no-refine \
  --print-slices \
  --output .benchmarks/stress_retention.json \
  "$@"
