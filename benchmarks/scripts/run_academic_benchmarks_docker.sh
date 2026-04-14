#!/usr/bin/env bash
# Run the academic benchmark suite inside the compose "benchmark" container.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
[[ -f .env ]] || { echo "Missing .env — copy from .env.example." >&2; exit 1; }
docker compose up -d qdrant
docker compose --profile benchmark build benchmark
exec docker compose --profile benchmark run --rm \
  --entrypoint bash benchmark \
  benchmarks/scripts/run_academic_benchmarks.sh "$@"
