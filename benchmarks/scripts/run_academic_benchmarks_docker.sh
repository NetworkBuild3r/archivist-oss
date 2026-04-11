#!/usr/bin/env bash
# Run the academic benchmark suite inside the compose "benchmark" container so
# LLM_URL / EMBED_URL (e.g. host.docker.internal) and QDRANT_URL=http://qdrant:6333
# resolve consistently on the compose network.
#
# Prerequisites:
#   cp .env.example .env   # keep .env local only — it is gitignored
#   docker compose up -d qdrant
#   Set LLM_URL, BENCHMARK_EMBED_URL, etc. in .env as needed.
#
# Usage (from repo root):
#   bash benchmarks/scripts/run_academic_benchmarks_docker.sh
#   bash benchmarks/scripts/run_academic_benchmarks_docker.sh --limit 50 --no-curator

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ ! -f .env ]]; then
  echo "Missing .env — copy from .env.example and configure LLM/EMBED/Qdrant." >&2
  exit 1
fi

exec docker compose --profile benchmark run --rm \
  --entrypoint bash \
  benchmark \
  benchmarks/scripts/run_academic_benchmarks.sh "$@"
