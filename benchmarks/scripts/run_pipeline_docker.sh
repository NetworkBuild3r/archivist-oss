#!/usr/bin/env bash
# Pipeline ablation benchmark (Recall@K, MRR, latency) inside Docker.
# Runs evaluate.py for a single --memory-scale (default: small).
# Timestamped output under .benchmarks/runs/.
#
# Usage (from repo root):
#   bash benchmarks/scripts/run_pipeline_docker.sh
#   bash benchmarks/scripts/run_pipeline_docker.sh -d                       # detached
#   bash benchmarks/scripts/run_pipeline_docker.sh --memory-scale medium
#   bash benchmarks/scripts/run_pipeline_docker.sh -d --memory-scale large --no-refine
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

[[ -f .env ]] || { echo "Missing .env — copy from .env.example." >&2; exit 1; }

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
mkdir -p "$RUN_DIR"

OUT="${RUN_DIR}/pipeline_${STAMP}_${GIT_SHA}.json"
LATEST=".benchmarks/pipeline_latest.json"
CID_FILE=".benchmarks/pipeline.cid"

echo "================================================"
echo "  Pipeline Benchmark (Docker)"
echo "  Output:   $OUT"
echo "  Latest:   $LATEST"
echo "  Git:      ${GIT_BRANCH} @ ${GIT_SHA}"
echo "  Detached: $DETACH"
echo "  Args:     ${EVAL_ARGS[*]+"${EVAL_ARGS[*]}"}"
echo "================================================"

EXTRA=""
[[ ${#EVAL_ARGS[@]} -gt 0 ]] && EXTRA=$(printf ' %q' "${EVAL_ARGS[@]}")

docker compose up -d qdrant
docker compose --profile benchmark build benchmark

_INNER='set -euo pipefail
exec python -m benchmarks.pipeline.evaluate \
  --output "$PIPELINE_OUTPUT" \
  '"$EXTRA"

if $DETACH; then
  CID=$(docker compose --profile benchmark run -d \
    -e BENCHMARK_GIT_SHA="$GIT_SHA" \
    -e BENCHMARK_GIT_BRANCH="$GIT_BRANCH" \
    -e PIPELINE_OUTPUT="$OUT" \
    --entrypoint bash benchmark \
    -c "$_INNER")
  echo "$CID" > "$CID_FILE"
  ln -sfn "runs/$(basename "$OUT")" "$LATEST"
  echo ""
  echo "Container started (detached): $CID"
  echo "Tail logs:   docker logs -f $CID"
  echo "Output file: $OUT  (written when complete)"
  echo "Stop early:  docker stop $CID"
  echo "CID saved:   $CID_FILE"
else
  # shellcheck disable=SC2086
  docker compose --profile benchmark run --rm \
    -e BENCHMARK_GIT_SHA="$GIT_SHA" \
    -e BENCHMARK_GIT_BRANCH="$GIT_BRANCH" \
    -e PIPELINE_OUTPUT="$OUT" \
    --entrypoint bash benchmark \
    -c "$_INNER"
  ln -sfn "runs/$(basename "$OUT")" "$LATEST"
  echo ""
  echo "Done. Results: $OUT"
  echo "Latest: $LATEST"
fi
