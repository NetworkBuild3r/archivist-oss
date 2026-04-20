"""SQLite vs Postgres comparison scenario.

Runs identical store + search workloads against both the SQLite and Postgres
backends and emits a side-by-side comparison table in the JSON output.

Requirements
------------
- A running Postgres instance accessible via the ``DATABASE_URL`` env var
  (e.g. ``postgresql://archivist:archivist@localhost:5432/archivist``).
- The ``asyncpg`` package must be installed
  (``pip install "archivist-oss[postgres]"``).

The scenario does NOT require a live Qdrant instance; embedding and vector
search calls are mocked identically for both backends.

CLI usage::

    # Requires DATABASE_URL env var (real Postgres)
    DATABASE_URL=postgresql://archivist:archivist@localhost:5432/archivist \\
        python -m benchmarks.enterprise.scenarios.sqlite_vs_pg

    # Customise scale
    DATABASE_URL=... python -m benchmarks.enterprise.scenarios.sqlite_vs_pg \\
        --n 200 --concurrency-levels 1 5 10 \\
        --output .benchmarks/enterprise/comparison.json

    # Dry-run (both backends, no real I/O)
    python -m benchmarks.enterprise.scenarios.sqlite_vs_pg --dry-run
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
logger = logging.getLogger("archivist.benchmark.comparison")

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

_DEFAULT_N = 200
_DEFAULT_CONCURRENCY_LEVELS = [1, 5, 10]


# ---------------------------------------------------------------------------
# Comparison table formatter
# ---------------------------------------------------------------------------


def _build_comparison_table(
    sqlite_results: dict[str, BenchmarkResult],  # type: ignore[name-defined]  # noqa: F821
    pg_results: dict[str, BenchmarkResult],  # type: ignore[name-defined]  # noqa: F821
    concurrency: int,
) -> str:
    """Build a markdown comparison table from result dicts.

    Args:
        sqlite_results: Mapping from operation name to ``BenchmarkResult`` for SQLite.
        pg_results: Mapping from operation name to ``BenchmarkResult`` for Postgres.
        concurrency: The concurrency level used for this comparison.

    Returns:
        A formatted markdown table string.
    """
    lines = [
        f"## SQLite vs Postgres Comparison (concurrency={concurrency})",
        "",
        "| Metric | SQLite | Postgres | Delta |",
        "|--------|--------|----------|-------|",
    ]

    def _row(label: str, sqlite_val: float, pg_val: float, unit: str = "ms") -> str:
        if sqlite_val > 0:
            delta = (pg_val - sqlite_val) / sqlite_val * 100
            delta_str = f"{delta:+.1f}%"
        else:
            delta_str = "N/A"
        return f"| {label} | {sqlite_val:.1f} {unit} | {pg_val:.1f} {unit} | {delta_str} |"

    for op in ("store", "search"):
        sr = sqlite_results.get(op)
        pr = pg_results.get(op)
        if sr is None or pr is None:
            continue
        lines.append(f"| **{op}** | | | |")
        lines.append(_row(f"{op} p50", sr.p50_ms, pr.p50_ms))
        lines.append(_row(f"{op} p95", sr.p95_ms, pr.p95_ms))
        lines.append(_row(f"{op} p99", sr.p99_ms, pr.p99_ms))
        lines.append(_row(f"{op} throughput", sr.ops_per_sec, pr.ops_per_sec, unit="ops/s"))
        if sr.recall_at_5 is not None and pr.recall_at_5 is not None:
            lines.append(
                f"| {op} recall@5 | {sr.recall_at_5:.3f} | {pr.recall_at_5:.3f} | "
                f"{(pr.recall_at_5 - sr.recall_at_5):+.3f} |"
            )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main scenario
# ---------------------------------------------------------------------------


async def run_comparison(
    *,
    n: int = _DEFAULT_N,
    concurrency_levels: list[int] | None = None,
    dry_run: bool = False,
    output_path: Path | None = None,
    pg_dsn: str | None = None,
) -> int:
    """Run store + search on SQLite and Postgres and write a comparison report.

    Args:
        n: Number of calls per operation per backend per concurrency level.
        concurrency_levels: List of concurrency values to test.
        dry_run: Skip real I/O; all timings are synthetic 1 ms.
        output_path: Destination for the JSON report.
        pg_dsn: Optional Postgres DSN.  Falls back to ``DATABASE_URL`` env var.

    Returns:
        Exit code: 0 = success, 1 = error.
    """
    import os

    from benchmarks.enterprise.harness import BackendFixture, write_json
    from benchmarks.enterprise.workloads.search import run_search_workload
    from benchmarks.enterprise.workloads.store import run_store_workload

    if concurrency_levels is None:
        concurrency_levels = _DEFAULT_CONCURRENCY_LEVELS

    if output_path is None:
        output_path = Path(".benchmarks/enterprise/comparison.json")

    pg_dsn = pg_dsn or os.environ.get("DATABASE_URL", "")
    if not pg_dsn and not dry_run:
        logger.error(
            "DATABASE_URL env var is required for Postgres comparison. Set it or use --dry-run."
        )
        return 1

    all_results = []
    comparison_sections: list[str] = []

    for concurrency in concurrency_levels:
        logger.info("running comparison at concurrency=%d n=%d...", concurrency, n)

        sqlite_results_map: dict[str, object] = {}
        pg_results_map: dict[str, object] = {}

        # --- SQLite ---
        try:
            with tempfile.TemporaryDirectory() as tmp_dir:
                sqlite_db = Path(tmp_dir) / "comparison_sqlite.db"
                async with BackendFixture("sqlite", sqlite_db) as sqlite_pool:
                    logger.info("  [sqlite] store...")
                    s_store = await run_store_workload(
                        sqlite_pool,
                        n=n,
                        concurrency=concurrency,
                        backend="sqlite",
                        dry_run=dry_run,
                    )
                    logger.info("  [sqlite] search...")
                    s_search = await run_search_workload(
                        sqlite_pool,
                        n=n,
                        concurrency=concurrency,
                        backend="sqlite",
                        dry_run=dry_run,
                    )
            sqlite_results_map["store"] = s_store
            sqlite_results_map["search"] = s_search
            all_results.extend([s_store, s_search])

        except Exception as exc:
            logger.error("SQLite run failed at concurrency=%d: %s", concurrency, exc, exc_info=True)
            return 1

        # --- Postgres ---
        try:
            async with BackendFixture("postgres", "", pg_dsn=pg_dsn) as pg_pool:
                logger.info("  [postgres] store...")
                p_store = await run_store_workload(
                    pg_pool,
                    n=n,
                    concurrency=concurrency,
                    backend="postgres",
                    dry_run=dry_run,
                )
                logger.info("  [postgres] search...")
                p_search = await run_search_workload(
                    pg_pool,
                    n=n,
                    concurrency=concurrency,
                    backend="postgres",
                    dry_run=dry_run,
                )
            pg_results_map["store"] = p_store
            pg_results_map["search"] = p_search
            all_results.extend([p_store, p_search])

        except Exception as exc:
            logger.error(
                "Postgres run failed at concurrency=%d: %s", concurrency, exc, exc_info=True
            )
            return 1

        section = _build_comparison_table(
            sqlite_results_map,  # type: ignore[arg-type]
            pg_results_map,  # type: ignore[arg-type]
            concurrency,
        )
        comparison_sections.append(section)
        _print_section(section)

    full_table = "\n\n".join(comparison_sections)

    write_json(
        all_results,
        output_path,
        scenario="comparison",
        backend="sqlite+postgres",
        dry_run=dry_run,
        extra_meta={"concurrency_levels": concurrency_levels, "n_per_level": n},
        comparison_table=full_table,
    )
    logger.info("comparison results written to %s", output_path)
    return 0


def _print_section(section: str) -> None:
    """Print a comparison section to stdout.

    Args:
        section: Formatted markdown section string.
    """
    print()
    print(section)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m benchmarks.enterprise.scenarios.sqlite_vs_pg",
        description="Archivist enterprise SQLite vs Postgres comparison benchmark",
    )
    p.add_argument("--dry-run", action="store_true", default=False)
    p.add_argument(
        "--n",
        type=int,
        default=_DEFAULT_N,
        help="Number of calls per operation per backend per concurrency level",
    )
    p.add_argument(
        "--concurrency-levels",
        nargs="+",
        type=int,
        default=_DEFAULT_CONCURRENCY_LEVELS,
        metavar="C",
        help="Concurrency levels to test (default: 1 5 10)",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output path for JSON results",
    )
    p.add_argument(
        "--pg-dsn",
        default=None,
        help="Postgres DSN (overrides DATABASE_URL env var)",
    )
    p.add_argument("--verbose", "-v", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for the comparison scenario."""
    args = _parse_args(argv)
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    exit_code = asyncio.run(
        run_comparison(
            n=args.n,
            concurrency_levels=args.concurrency_levels,
            dry_run=args.dry_run,
            output_path=args.output,
            pg_dsn=args.pg_dsn,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
