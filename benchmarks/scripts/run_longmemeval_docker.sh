#!/usr/bin/env bash
# LongMemEval inside the compose "benchmark" container (Qdrant + host LLM/embed).
# Writes a timestamped JSON under .benchmarks/runs/ and updates .benchmarks/longmemeval_latest.json.
#
# Usage (from repo root):
#   bash benchmarks/scripts/run_longmemeval_docker.sh
#   bash benchmarks/scripts/run_longmemeval_docker.sh -d             # detached (survives logout)
#   bash benchmarks/scripts/run_longmemeval_docker.sh --limit 10 --no-refine
#   bash benchmarks/scripts/run_longmemeval_docker.sh --ablation --limit 20
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

[[ -f .env ]] || { echo "Missing .env — copy from .env.example and configure LLM/EMBED." >&2; exit 1; }

DETACH=false
EVAL_ARGS=()
for _a in "$@"; do
  case "$_a" in
    -d|--detach) DETACH=true ;;
    *) EVAL_ARGS+=("$_a") ;;
  esac
done

STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"
GIT_SHA="$(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo unknown)"
GIT_BRANCH="$(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo unknown)"

RUN_DIR=".benchmarks/runs"
mkdir -p "$RUN_DIR" "data/longmemeval"

OUT="${RUN_DIR}/longmemeval_${STAMP}_${GIT_SHA}.json"
LATEST=".benchmarks/longmemeval_latest.json"
CID_FILE=".benchmarks/longmemeval.cid"

echo "================================================"
echo "  LongMemEval (Docker benchmark profile)"
echo "  Output file:  $OUT"
echo "  Latest link:  $LATEST"
echo "  Git:          ${GIT_BRANCH} @ ${GIT_SHA}"
echo "  Detached:     $DETACH"
echo "================================================"

export BENCHMARK_GIT_SHA="$GIT_SHA"
export BENCHMARK_GIT_BRANCH="$GIT_BRANCH"

_INNER="/workspace/benchmarks/scripts/longmemeval_docker_inner.sh"

docker compose up -d qdrant
docker compose --profile benchmark build benchmark

_run() {
  # shellcheck disable=SC2086
  docker compose --profile benchmark run --rm \
    -e BENCHMARK_GIT_SHA="$GIT_SHA" \
    -e BENCHMARK_GIT_BRANCH="$GIT_BRANCH" \
    -e LONGMEMEVAL_OUTPUT="/workspace/$OUT" \
    --entrypoint bash benchmark \
    -c "exec bash $_INNER $(printf '%q ' "${EVAL_ARGS[@]}")"
}

if $DETACH; then
  CID=$(docker compose --profile benchmark run -d \
    -e BENCHMARK_GIT_SHA="$GIT_SHA" \
    -e BENCHMARK_GIT_BRANCH="$GIT_BRANCH" \
    -e LONGMEMEVAL_OUTPUT="/workspace/$OUT" \
    --entrypoint bash benchmark \
    -c "exec bash $_INNER $(printf '%q ' "${EVAL_ARGS[@]}")")
  echo "$CID" > "$CID_FILE"
  ln -sfn "runs/$(basename "$OUT")" "$LATEST"
  echo ""
  echo "Container started (detached): $CID"
  echo "Tail logs:   docker logs -f $CID"
  echo "Output file: $OUT  (written when complete)"
  echo "Latest link: $LATEST"
  echo "Stop early:  docker stop $CID"
  echo "CID saved:   $CID_FILE"
else
  _run
  ln -sfn "runs/$(basename "$OUT")" "$LATEST"
  echo ""
  echo "Done. Results: $OUT"
  echo "Latest symlink: $LATEST"
fi
