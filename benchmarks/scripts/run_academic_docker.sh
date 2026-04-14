#!/usr/bin/env bash
# All three academic benchmarks (LongMemEval + LoCoMo + HaluMem) inside Docker.
# Timestamped output under .benchmarks/runs/, individual + merged JSON.
#
# Usage (from repo root):
#   bash benchmarks/scripts/run_academic_docker.sh
#   bash benchmarks/scripts/run_academic_docker.sh --limit 50 --no-curator
#
# Extra args pass through to run_academic_benchmarks.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

[[ -f .env ]] || { echo "Missing .env — copy from .env.example." >&2; exit 1; }

STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
GIT_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

echo "================================================"
echo "  Academic Suite (Docker)"
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
  benchmarks/scripts/run_academic_benchmarks.sh "$@"

echo ""
echo "Done. Results in .benchmarks/ (longmemeval_results, locomo_results, halumem_results, academic_scores)."
