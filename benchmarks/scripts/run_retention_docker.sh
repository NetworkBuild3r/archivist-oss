#!/usr/bin/env bash
# Memory retention / stress test inside Docker (1500+ files corpus).
# Timestamped output under .benchmarks/runs/.
#
# Usage (from repo root):
#   bash benchmarks/scripts/run_retention_docker.sh             # foreground
#   bash benchmarks/scripts/run_retention_docker.sh -d          # detached (survives logout)
#   bash benchmarks/scripts/run_retention_docker.sh --limit 20
#   bash benchmarks/scripts/run_retention_docker.sh -d --limit 20
#
# Extra args (excluding -d/--detach) pass through to evaluate.py.
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
CID_DIR=".benchmarks"
mkdir -p "$RUN_DIR"

OUT="${RUN_DIR}/retention_${STAMP}_${GIT_SHA}.json"
LATEST=".benchmarks/retention_latest.json"
CID_FILE="${CID_DIR}/retention.cid"

echo "================================================"
echo "  Memory Retention / Stress (Docker)"
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

if $DETACH; then
  # -d is incompatible with --rm; container persists until it exits naturally.
  # shellcheck disable=SC2086
  CID=$(docker compose --profile benchmark run -d \
    -e BENCHMARK_GIT_SHA="$GIT_SHA" \
    -e BENCHMARK_GIT_BRANCH="$GIT_BRANCH" \
    -e RETENTION_OUTPUT="$OUT" \
    --entrypoint bash benchmark \
    -c 'set -euo pipefail
if [[ ! -d benchmarks/fixtures/corpus_stress/agents ]]; then
  echo "Generating stress corpus ..."
  python benchmarks/fixtures/generate_corpus.py --preset stress --corpus-only
fi
exec python -m benchmarks.pipeline.evaluate \
  --memory-scale stress \
  --variants vector_only,full_pipeline \
  --no-refine \
  --print-slices \
  --output "$RETENTION_OUTPUT" \
  '"$EXTRA")
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
  # shellcheck disable=SC2086
  docker compose --profile benchmark run --rm \
    -e BENCHMARK_GIT_SHA="$GIT_SHA" \
    -e BENCHMARK_GIT_BRANCH="$GIT_BRANCH" \
    -e RETENTION_OUTPUT="$OUT" \
    --entrypoint bash benchmark \
    -c 'set -euo pipefail
if [[ ! -d benchmarks/fixtures/corpus_stress/agents ]]; then
  echo "Generating stress corpus ..."
  python benchmarks/fixtures/generate_corpus.py --preset stress --corpus-only
fi
exec python -m benchmarks.pipeline.evaluate \
  --memory-scale stress \
  --variants vector_only,full_pipeline \
  --no-refine \
  --print-slices \
  --output "$RETENTION_OUTPUT" \
  '"$EXTRA"
  ln -sfn "runs/$(basename "$OUT")" "$LATEST"
  echo ""
  echo "Done. Results: $OUT"
  echo "Latest: $LATEST"
fi
