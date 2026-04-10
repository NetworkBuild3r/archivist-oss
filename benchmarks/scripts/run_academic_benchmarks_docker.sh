#!/usr/bin/env bash
# Run the academic benchmark suite inside the compose "benchmark" container so
# LLM_URL hostnames (e.g. openclaw-vllm-corp → Qwen via extra_hosts on tailscale)
# and QDRANT_URL=http://qdrant:6333 match the same network namespace as archivist.
#
# Prerequisites:
#   cp .env.example .env   # keep .env local only — it is gitignored
#   docker compose up -d qdrant tailscale
#   Set LLM_URL / OPENCLAW_VLLM_TSIP / BENCHMARK_EMBED_URL in .env as needed.
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
