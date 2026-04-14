#!/usr/bin/env bash
# Full comparison (MD-file stuffing vs Archivist across all corpus scales) inside Docker.
# Timestamped output under .benchmarks/runs/.
#
# Usage (from repo root):
#   bash benchmarks/scripts/run_full_comparison_docker.sh
#   bash benchmarks/scripts/run_full_comparison_docker.sh --progress-pct 5
#   bash benchmarks/scripts/run_full_comparison_docker.sh -d   # debug mode
#
# Extra args pass through to run_full_comparison.sh → evaluate.py.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

[[ -f .env ]] || { echo "Missing .env — copy from .env.example." >&2; exit 1; }

STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
GIT_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

echo "================================================"
echo "  Full Comparison (Docker)"
echo "  Git:  ${GIT_BRANCH} @ ${GIT_SHA}"
echo "  Time: ${STAMP}"
echo "  Args: $*"
echo "================================================"

docker compose up -d qdrant
docker compose --profile benchmark build benchmark

docker compose --profile benchmark run --rm \
  -e BENCHMARK_GIT_SHA="$GIT_SHA" \
  -e BENCHMARK_GIT_BRANCH="$GIT_BRANCH" \
  --entrypoint bash benchmark \
  benchmarks/scripts/run_full_comparison.sh "$@"

echo ""
echo "Done. Per-scale JSON in .benchmarks/full_*.json, merged in .benchmarks/full_comparison.json."
