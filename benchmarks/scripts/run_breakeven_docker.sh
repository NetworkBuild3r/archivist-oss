#!/usr/bin/env bash
# Break-even comparison (context stuffing vs Archivist) inside Docker.
# Generates missing corpora, runs scale-sweep + stress, writes timestamped output.
#
# Usage (from repo root):
#   bash benchmarks/scripts/run_breakeven_docker.sh
#   bash benchmarks/scripts/run_breakeven_docker.sh -d                 # detached
#   bash benchmarks/scripts/run_breakeven_docker.sh --stuffing-call-llm
#   bash benchmarks/scripts/run_breakeven_docker.sh -d --limit 20
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
mkdir -p "$RUN_DIR"

SWEEP_OUT="${RUN_DIR}/breakeven_sweep_${STAMP}_${GIT_SHA}.json"
STRESS_OUT="${RUN_DIR}/breakeven_stress_${STAMP}_${GIT_SHA}.json"
LATEST_SWEEP=".benchmarks/breakeven_sweep_latest.json"
LATEST_STRESS=".benchmarks/breakeven_stress_latest.json"
CID_FILE=".benchmarks/breakeven.cid"

echo "================================================"
echo "  Break-Even Benchmark (Docker)"
echo "  Sweep output:   $SWEEP_OUT"
echo "  Stress output:  $STRESS_OUT"
echo "  Git:            ${GIT_BRANCH} @ ${GIT_SHA}"
echo "  Detached:       $DETACH"
echo "  Args:           ${EVAL_ARGS[*]+"${EVAL_ARGS[*]}"}"
echo "================================================"

EXTRA=""
[[ ${#EVAL_ARGS[@]} -gt 0 ]] && EXTRA=$(printf ' %q' "${EVAL_ARGS[@]}")
CONTEXT_BUDGET="${BREAKEVEN_CONTEXT_BUDGET:-32768}"

docker compose up -d qdrant
docker compose --profile benchmark build benchmark

_INNER='set -euo pipefail
for preset in small medium large stress; do
  if [[ ! -d "benchmarks/fixtures/corpus_${preset}/agents" ]]; then
    echo "Generating corpus_${preset} ..."
    python benchmarks/fixtures/generate_corpus.py --preset "$preset" --corpus-only
  fi
done
echo "Running scale sweep (small/medium/large) ..."
python -m benchmarks.pipeline.evaluate \
  --scale-sweep \
  --variants vector_only,full_pipeline \
  --no-refine \
  --compare-stuffing \
  --context-budget "$CONTEXT_BUDGET" \
  --print-slices \
  --output "$SWEEP_OUT" \
  '"$EXTRA"'
echo "Running stress corpus ..."
python -m benchmarks.pipeline.evaluate \
  --memory-scale stress \
  --variants vector_only,full_pipeline \
  --no-refine \
  --compare-stuffing \
  --context-budget "$CONTEXT_BUDGET" \
  --print-slices \
  --output "$STRESS_OUT" \
  '"$EXTRA"

if $DETACH; then
  CID=$(docker compose --profile benchmark run -d \
    -e BENCHMARK_GIT_SHA="$GIT_SHA" \
    -e BENCHMARK_GIT_BRANCH="$GIT_BRANCH" \
    -e SWEEP_OUT="$SWEEP_OUT" \
    -e STRESS_OUT="$STRESS_OUT" \
    -e CONTEXT_BUDGET="$CONTEXT_BUDGET" \
    --entrypoint bash benchmark \
    -c "$_INNER")
  echo "$CID" > "$CID_FILE"
  ln -sfn "runs/$(basename "$SWEEP_OUT")" "$LATEST_SWEEP"
  ln -sfn "runs/$(basename "$STRESS_OUT")" "$LATEST_STRESS"
  echo ""
  echo "Container started (detached): $CID"
  echo "Tail logs:   docker logs -f $CID"
  echo "Outputs written when complete:"
  echo "  Sweep:  $SWEEP_OUT"
  echo "  Stress: $STRESS_OUT"
  echo "Stop early:  docker stop $CID"
  echo "CID saved:   $CID_FILE"
else
  # shellcheck disable=SC2086
  docker compose --profile benchmark run --rm \
    -e BENCHMARK_GIT_SHA="$GIT_SHA" \
    -e BENCHMARK_GIT_BRANCH="$GIT_BRANCH" \
    -e SWEEP_OUT="$SWEEP_OUT" \
    -e STRESS_OUT="$STRESS_OUT" \
    -e CONTEXT_BUDGET="$CONTEXT_BUDGET" \
    --entrypoint bash benchmark \
    -c "$_INNER"
  ln -sfn "runs/$(basename "$SWEEP_OUT")" "$LATEST_SWEEP"
  ln -sfn "runs/$(basename "$STRESS_OUT")" "$LATEST_STRESS"
  echo ""
  echo "Done."
  echo "  Sweep:  $SWEEP_OUT"
  echo "  Stress: $STRESS_OUT"
fi
