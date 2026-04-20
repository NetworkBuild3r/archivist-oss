"""Smoke scenario — fast CI gate for store + search latency.

Runs store × 20 + search × 20 against a fresh SQLite database with fully
mocked Qdrant and embedding dependencies.  Target wall-clock time is under
60 seconds on a shared GitHub Actions runner.

Baseline gate
-------------
If ``--check-baselines`` is passed (default in CI), each operation's p99
latency is compared against ``benchmarks/enterprise/baselines.json``.  The
scenario exits non-zero if any p99 exceeds ``2 × baseline``.

Dry-run mode
------------
Pass ``--dry-run`` to skip all real I/O.  All timings will be synthetic 1 ms
values.  Use this to validate the harness itself without any database setup.

CLI usage::

    # Standard smoke run (SQLite)
    python -m benchmarks.enterprise.scenarios.smoke

    # Dry-run (no I/O, always passes)
    python -m benchmarks.enterprise.scenarios.smoke --dry-run

    # Disable baseline gate (smoke-only timing, no fail)
    python -m benchmarks.enterprise.scenarios.smoke --no-check-baselines

    # Custom output path
    python -m benchmarks.enterprise.scenarios.smoke \\
        --output .benchmarks/enterprise/smoke_sqlite.json
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import tempfile
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s  %(message)s",
)
logger = logging.getLogger("archivist.benchmark.smoke")

# ---------------------------------------------------------------------------
# Scenario defaults
# ---------------------------------------------------------------------------

_STORE_N = 20
_SEARCH_N = 20
_CONCURRENCY = 2
_BACKEND = "sqlite"


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------


async def run_smoke(
    *,
    dry_run: bool = False,
    check_baselines: bool = True,
    output_path: Path | None = None,
    store_n: int = _STORE_N,
    search_n: int = _SEARCH_N,
    concurrency: int = _CONCURRENCY,
) -> int:
    """Execute the smoke scenario and return an exit code.

    Args:
        dry_run: Skip real I/O; record synthetic 1 ms timings.
        check_baselines: Fail if any p99 exceeds 2× stored baseline.
        output_path: Where to write the JSON result.  Defaults to
            ``.benchmarks/enterprise/smoke_sqlite.json``.
        store_n: Number of store calls.
        search_n: Number of search calls.
        concurrency: Maximum concurrent calls per workload.

    Returns:
        Exit code: 0 = pass, 1 = baseline violation, 2 = unexpected error.
    """
    from benchmarks.enterprise.harness import (
        BackendFixture,
        BaselineViolation,
        write_json,
    )
    from benchmarks.enterprise.harness import (
        check_baselines as _check,
    )
    from benchmarks.enterprise.workloads.search import run_search_workload
    from benchmarks.enterprise.workloads.store import run_store_workload

    if output_path is None:
        output_path = Path(".benchmarks/enterprise/smoke_sqlite.json")

    baselines_path = Path(__file__).parent.parent / "baselines.json"

    logger.info(
        "smoke scenario: store×%d search×%d concurrency=%d backend=%s dry_run=%s",
        store_n,
        search_n,
        concurrency,
        _BACKEND,
        dry_run,
    )

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            db_path = Path(tmp_dir) / "smoke.db"

            async with BackendFixture(_BACKEND, db_path) as pool:
                logger.info("running store workload (%d calls)...", store_n)
                store_result = await run_store_workload(
                    pool,
                    n=store_n,
                    concurrency=concurrency,
                    backend=_BACKEND,
                    dry_run=dry_run,
                )

                logger.info("running search workload (%d calls)...", search_n)
                search_result = await run_search_workload(
                    pool,
                    n=search_n,
                    concurrency=concurrency,
                    backend=_BACKEND,
                    dry_run=dry_run,
                )

    except Exception as exc:
        logger.error("smoke scenario failed with unexpected error: %s", exc, exc_info=True)
        return 2

    results = [store_result, search_result]

    _print_summary(results)

    out = write_json(
        results,
        output_path,
        scenario="smoke",
        backend=_BACKEND,
        dry_run=dry_run,
    )
    logger.info("results written to %s", out)

    if check_baselines and baselines_path.exists() and not dry_run:
        try:
            _check(results, baselines_path, multiplier=2.0)
            logger.info("baseline gate passed")
        except BaselineViolation as exc:
            logger.error("baseline gate FAILED:\n%s", exc)
            return 1
    elif check_baselines and not baselines_path.exists():
        logger.warning("baselines.json not found at %s — skipping baseline gate", baselines_path)

    return 0


def _print_summary(results: list) -> None:
    """Print a formatted summary table to stdout.

    Args:
        results: List of ``BenchmarkResult`` objects.
    """
    print()
    print("=" * 72)
    print(
        f"{'Operation':<12} {'Backend':<10} {'p50ms':>7} {'p95ms':>7} {'p99ms':>7} {'ops/s':>8} {'err%':>6} {'n':>5}"
    )
    print("-" * 72)
    for r in results:
        recall = f"  recall@5={r.recall_at_5:.2f}" if r.recall_at_5 is not None else ""
        print(
            f"{r.name:<12} {r.backend:<10} {r.p50_ms:>7.1f} {r.p95_ms:>7.1f} "
            f"{r.p99_ms:>7.1f} {r.ops_per_sec:>8.2f} {r.error_rate * 100:>5.1f}% "
            f"{r.sample_count:>5}{recall}"
        )
    print("=" * 72)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.enterprise.scenarios.smoke",
        description="Archivist enterprise smoke benchmark (store + search, SQLite, mocked services)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Skip all real I/O; record synthetic 1 ms timings (harness validation only)",
    )
    p.add_argument(
        "--check-baselines",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fail if any p99 exceeds 2× stored baseline (default: on)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for JSON results (default: .benchmarks/enterprise/smoke_sqlite.json)",
    )
    p.add_argument("--store-n", type=int, default=_STORE_N, help="Number of store calls")
    p.add_argument("--search-n", type=int, default=_SEARCH_N, help="Number of search calls")
    p.add_argument("--concurrency", type=int, default=_CONCURRENCY, help="Concurrency level")
    p.add_argument("--verbose", "-v", action="store_true", help="Set log level to DEBUG")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the smoke scenario."""
    args = _parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    exit_code = asyncio.run(
        run_smoke(
            dry_run=args.dry_run,
            check_baselines=args.check_baselines,
            output_path=args.output,
            store_n=args.store_n,
            search_n=args.search_n,
            concurrency=args.concurrency,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
