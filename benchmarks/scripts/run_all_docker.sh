#!/usr/bin/env bash
# Run all benchmark suites (or a selected subset) inside Docker.
#
# Each suite runs in its own container (sequentially by default, or in parallel with --parallel).
# All output goes to timestamped files under .benchmarks/runs/.
#
# Usage (from repo root):
#   bash benchmarks/scripts/run_all_docker.sh                 # all 6 suites, sequential
#   bash benchmarks/scripts/run_all_docker.sh --parallel      # all 6 suites, background
#   bash benchmarks/scripts/run_all_docker.sh longmemeval ablation   # pick specific suites
#   bash benchmarks/scripts/run_all_docker.sh --parallel longmemeval pipeline retention
#
# Available suite names:
#   longmemeval   — LongMemEval single-run (timestamped)
#   ablation      — LongMemEval ablation across pipeline variants
#   academic      — All 3 academic benchmarks (LongMemEval + LoCoMo + HaluMem)
#   pipeline      — Pipeline retrieval (Recall@K, MRR, latency) — default small scale
#   breakeven     — Context-stuffing vs Archivist scale sweep
#   full          — Full comparison across all scales with LLM calls
#   retention     — Stress / memory retention test (1500+ files)
#
# To pass extra args to a specific suite, run it directly:
#   bash benchmarks/scripts/run_pipeline_docker.sh --memory-scale large --limit 40
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

PARALLEL=false
SUITES=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --parallel|-p) PARALLEL=true; shift ;;
    --help|-h)
      head -30 "$0" | grep '^#' | sed 's/^# \?//'
      exit 0
      ;;
    *) SUITES+=("$1"); shift ;;
  esac
done

if [[ ${#SUITES[@]} -eq 0 ]]; then
  SUITES=(longmemeval ablation academic pipeline breakeven full retention)
fi

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

declare -A SUITE_SCRIPT=(
  [longmemeval]="$SCRIPTS_DIR/run_longmemeval_docker.sh"
  [ablation]="$SCRIPTS_DIR/run_ablation_docker.sh"
  [academic]="$SCRIPTS_DIR/run_academic_docker.sh"
  [pipeline]="$SCRIPTS_DIR/run_pipeline_docker.sh"
  [breakeven]="$SCRIPTS_DIR/run_breakeven_docker.sh"
  [full]="$SCRIPTS_DIR/run_full_comparison_docker.sh"
  [retention]="$SCRIPTS_DIR/run_retention_docker.sh"
)

# Pre-build once so individual scripts don't each rebuild
echo "================================================"
echo "  Building benchmark image ..."
echo "================================================"
docker compose up -d qdrant
docker compose --profile benchmark build benchmark
echo ""

LOG_DIR=".benchmarks/logs"
mkdir -p "$LOG_DIR"
STAMP="$(date -u +"%Y%m%dT%H%M%SZ")"

PIDS=()
NAMES=()

for suite in "${SUITES[@]}"; do
  script="${SUITE_SCRIPT[$suite]:-}"
  if [[ -z "$script" ]]; then
    echo "Unknown suite: $suite (available: ${!SUITE_SCRIPT[*]})"
    exit 1
  fi

  if $PARALLEL; then
    log="${LOG_DIR}/${suite}_${STAMP}.log"
    echo "[parallel] Starting $suite → $log"
    bash "$script" > "$log" 2>&1 &
    PIDS+=($!)
    NAMES+=("$suite")
  else
    echo "================================================"
    echo "  Running: $suite"
    echo "================================================"
    bash "$script" || echo "WARNING: $suite exited non-zero"
    echo ""
  fi
done

if $PARALLEL && [[ ${#PIDS[@]} -gt 0 ]]; then
  echo ""
  echo "Waiting for ${#PIDS[@]} suites to finish ..."
  echo "  Logs: $LOG_DIR/"
  echo "  Monitor: tail -f $LOG_DIR/*_${STAMP}.log"
  echo ""

  FAILURES=0
  for i in "${!PIDS[@]}"; do
    if wait "${PIDS[$i]}"; then
      echo "  [done] ${NAMES[$i]}"
    else
      echo "  [FAIL] ${NAMES[$i]} (see ${LOG_DIR}/${NAMES[$i]}_${STAMP}.log)"
      FAILURES=$((FAILURES + 1))
    fi
  done

  echo ""
  if [[ $FAILURES -gt 0 ]]; then
    echo "$FAILURES suite(s) failed. Check logs in $LOG_DIR/"
  else
    echo "All suites complete."
  fi
fi

echo ""
echo "Results in .benchmarks/ and .benchmarks/runs/"
