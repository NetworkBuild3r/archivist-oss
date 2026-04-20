"""Enterprise benchmark harness — Phase 1.

Core components
---------------
- ``LatencyHistogram`` — streaming sorted-buffer percentile computation
- ``BenchmarkResult``  — typed result dataclass, JSON-serialisable
- ``BackendFixture``   — context manager that spins up SQLite or Postgres pool
- ``run_workload``     — drives N async calls at concurrency C
- ``write_json``       — writes result to ``.benchmarks/enterprise/``

Dry-run guard
-------------
Pass ``dry_run=True`` (or set ``DRY_RUN=1`` env var) to skip all real I/O and
return synthetic 1 ms timings.  This prevents accidental LLM / embed charges
in CI when the smoke job is misconfigured.

Usage::

    from benchmarks.enterprise.harness import BackendFixture, run_workload, write_json

    async def _store_one(pool) -> None:
        # perform one store operation using pool
        ...

    async with BackendFixture("sqlite", tmp_path / "bench.db") as pool:
        result = await run_workload(
            name="store",
            backend="sqlite",
            coro_factory=lambda: _store_one(pool),
            n=100,
            concurrency=5,
        )
    write_json(result, Path(".benchmarks/enterprise/smoke_sqlite.json"))
"""

from __future__ import annotations

import asyncio
import bisect
import json
import logging
import os
import time
from collections.abc import AsyncGenerator, Callable, Coroutine
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from subprocess import run as _run
from typing import Any

logger = logging.getLogger("archivist.benchmark.harness")

# ---------------------------------------------------------------------------
# Dry-run guard
# ---------------------------------------------------------------------------

_DRY_RUN_ENV: bool = os.environ.get("DRY_RUN", "").strip() in {"1", "true", "yes"}


def _is_dry_run(override: bool | None = None) -> bool:
    """Return ``True`` when dry-run mode is active.

    Precedence: explicit *override* argument > ``DRY_RUN`` env var.
    """
    if override is not None:
        return override
    return _DRY_RUN_ENV


# ---------------------------------------------------------------------------
# LatencyHistogram
# ---------------------------------------------------------------------------


class LatencyHistogram:
    """Streaming latency histogram using a sorted insertion buffer.

    Samples are stored in a sorted list so percentile queries are O(1).
    The first ``warm_up_count`` samples recorded are discarded to eliminate
    cold-start noise, matching the ``_warmup()`` pattern in
    ``test_outbox_throughput.py``.

    Args:
        warm_up_count: Number of initial samples to discard.  Default 2.
    """

    def __init__(self, warm_up_count: int = 2) -> None:
        self._warm_up_count = warm_up_count
        self._discarded: int = 0
        self._samples: list[float] = []

    def record(self, ms: float) -> None:
        """Record a single latency sample in milliseconds.

        Args:
            ms: Observed latency in milliseconds.
        """
        if self._discarded < self._warm_up_count:
            self._discarded += 1
            return
        bisect.insort(self._samples, ms)

    @property
    def count(self) -> int:
        """Number of samples retained (after warm-up discard)."""
        return len(self._samples)

    def p(self, percentile: float) -> float:
        """Return the *percentile*-th latency value in milliseconds.

        Args:
            percentile: Value between 0 and 100 (e.g. 99 for p99).

        Returns:
            Latency in milliseconds, or 0.0 if no samples recorded.
        """
        if not self._samples:
            return 0.0
        idx = int(len(self._samples) * percentile / 100)
        idx = min(idx, len(self._samples) - 1)
        return self._samples[idx]

    def mean(self) -> float:
        """Return the mean latency in milliseconds, or 0.0 if empty."""
        if not self._samples:
            return 0.0
        return sum(self._samples) / len(self._samples)

    def p50(self) -> float:
        """Return p50 latency in milliseconds."""
        return self.p(50)

    def p95(self) -> float:
        """Return p95 latency in milliseconds."""
        return self.p(95)

    def p99(self) -> float:
        """Return p99 latency in milliseconds."""
        return self.p(99)


# ---------------------------------------------------------------------------
# BenchmarkResult
# ---------------------------------------------------------------------------


@dataclass
class ResourceSnapshot:
    """Before/after process resource snapshot captured via psutil.

    Args:
        rss_mb_before: RSS memory in MiB at run start.
        rss_mb_after: RSS memory in MiB at run end.
        cpu_percent_mean: Mean CPU percentage sampled during run.
    """

    rss_mb_before: float = 0.0
    rss_mb_after: float = 0.0
    cpu_percent_mean: float = 0.0

    @property
    def rss_mb_delta(self) -> float:
        """Return RSS growth in MiB."""
        return self.rss_mb_after - self.rss_mb_before


@dataclass
class BenchmarkResult:
    """Single-operation benchmark result.

    Args:
        name: Operation name (e.g. "store", "search").
        backend: Backend name ("sqlite" or "postgres").
        p50_ms: p50 latency in milliseconds.
        p95_ms: p95 latency in milliseconds.
        p99_ms: p99 latency in milliseconds.
        mean_ms: Mean latency in milliseconds.
        ops_per_sec: Throughput in operations per second.
        error_rate: Fraction of calls that raised an exception (0.0–1.0).
        sample_count: Number of samples retained after warm-up.
        warm_up_count: Number of samples discarded during warm-up.
        concurrency: Concurrency level used for this run.
        recall_at_5: Recall@5 quality metric, or ``None`` if not measured.
        extra: Arbitrary extra key-value pairs (e.g. recall@10, NDCG).
    """

    name: str
    backend: str
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    ops_per_sec: float
    error_rate: float
    sample_count: int
    warm_up_count: int
    concurrency: int
    recall_at_5: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable dict representation."""
        d = asdict(self)
        d["rss_mb_delta"] = d.pop("extra", {}).get("rss_mb_delta")
        return d


# ---------------------------------------------------------------------------
# BackendFixture
# ---------------------------------------------------------------------------


@asynccontextmanager
async def BackendFixture(
    backend: str,
    db_path: Path | str,
    *,
    pg_dsn: str | None = None,
) -> AsyncGenerator[Any, None]:
    """Async context manager that provisions a benchmark-ready storage pool.

    For SQLite, initialises a ``SQLitePool`` against *db_path* and patches
    the global pool singleton for the duration of the ``async with`` block.

    For Postgres, creates an ``AsyncpgGraphBackend`` using *pg_dsn* (or the
    ``DATABASE_URL`` env var) and applies the schema.

    Args:
        backend: Either ``"sqlite"`` or ``"postgres"``.
        db_path: Path to a temporary SQLite file (ignored for Postgres).
        pg_dsn: Optional Postgres DSN.  Falls back to ``DATABASE_URL`` env var.

    Yields:
        The initialised pool/backend object.

    Raises:
        ValueError: If *backend* is not ``"sqlite"`` or ``"postgres"``.
        RuntimeError: If Postgres is requested but no DSN is available.
    """
    if backend not in {"sqlite", "postgres"}:
        raise ValueError(f"backend must be 'sqlite' or 'postgres', got {backend!r}")

    if backend == "sqlite":
        from tests.fixtures.schema import build_schema

        from archivist.storage import sqlite_pool as _sp

        pool = _sp.SQLitePool()
        await pool.initialize(str(db_path))

        _original_pool = _sp.pool
        _sp.pool = pool  # type: ignore[assignment]

        # Patch graph.SQLITE_PATH so that graph.get_db() and needle token
        # operations use the same temp db instead of the production /data path.
        import archivist.storage.graph as _graph

        _original_graph_path = _graph.SQLITE_PATH
        _original_env_path = os.environ.get("SQLITE_PATH")
        _graph.SQLITE_PATH = str(db_path)
        os.environ["SQLITE_PATH"] = str(db_path)

        try:
            async with pool.write() as conn:
                await build_schema(conn)

            # Initialise graph's own SQLite tables (needle_tokens, entities etc.)
            # in the same temp file so needle-token writes don't fail.  The
            # schema_guard's applied flag must be reset so it re-runs the DDL
            # against the fresh temp database.
            if hasattr(_graph, "_ensure_needle_registry") and hasattr(
                _graph._ensure_needle_registry, "reset"
            ):
                _graph._ensure_needle_registry.reset()
            with _graph.GRAPH_WRITE_LOCK:
                _graph_conn = _graph.get_db()
                try:
                    if hasattr(_graph, "_ensure_needle_registry"):
                        _graph._ensure_needle_registry()
                    _graph_conn.commit()
                finally:
                    _graph_conn.close()

            yield pool
        finally:
            _sp.pool = _original_pool  # type: ignore[assignment]
            _graph.SQLITE_PATH = _original_graph_path
            # Re-arm the schema guard so next caller re-runs DDL against its db.
            if hasattr(_graph, "_ensure_needle_registry") and hasattr(
                _graph._ensure_needle_registry, "reset"
            ):
                _graph._ensure_needle_registry.reset()
            if _original_env_path is not None:
                os.environ["SQLITE_PATH"] = _original_env_path
            else:
                os.environ.pop("SQLITE_PATH", None)
            await pool.close()

    else:  # postgres
        dsn = pg_dsn or os.environ.get("DATABASE_URL", "")
        if not dsn:
            raise RuntimeError(
                "Postgres benchmark requires DATABASE_URL env var or pg_dsn argument"
            )
        from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

        backend_obj = AsyncpgGraphBackend(dsn)
        await backend_obj.initialize()
        try:
            yield backend_obj
        finally:
            await backend_obj.close()


# ---------------------------------------------------------------------------
# run_workload
# ---------------------------------------------------------------------------


async def run_workload(
    *,
    name: str,
    backend: str,
    coro_factory: Callable[[], Coroutine[Any, Any, Any]],
    n: int,
    concurrency: int,
    dry_run: bool | None = None,
    warm_up_count: int = 2,
) -> BenchmarkResult:
    """Drive *n* calls of *coro_factory* at *concurrency* and return a result.

    Args:
        name: Operation label for the result (e.g. ``"store"``).
        backend: Backend label for the result (e.g. ``"sqlite"``).
        coro_factory: Zero-argument callable that returns a new coroutine each
            call.  Must be safe to call concurrently.
        n: Total number of calls to make.
        concurrency: Maximum number of in-flight coroutines at any time.
        dry_run: Override the global DRY_RUN setting.  When ``True``, skips
            real I/O and records synthetic 1 ms timings for all *n* calls.
        warm_up_count: Initial samples to discard.  Defaults to 2.

    Returns:
        A ``BenchmarkResult`` with latency percentiles and throughput.
    """
    is_dry = _is_dry_run(dry_run)

    # Resource sampling — optional psutil
    rss_before = _rss_mb()
    cpu_samples: list[float] = []

    histogram = LatencyHistogram(warm_up_count=warm_up_count)
    errors = 0
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_one() -> None:
        nonlocal errors
        async with semaphore:
            if is_dry:
                histogram.record(1.0)
                return
            t0 = time.perf_counter()
            try:
                await coro_factory()
            except Exception:
                errors += 1
                logger.debug("benchmark call raised", exc_info=True)
            finally:
                elapsed_ms = (time.perf_counter() - t0) * 1000
                histogram.record(elapsed_ms)

    wall_start = time.perf_counter()
    tasks = [asyncio.create_task(_run_one()) for _ in range(n)]

    # Poll CPU while tasks run (every 200 ms)
    async def _poll_cpu() -> None:
        import psutil

        proc = psutil.Process()
        while True:
            cpu_samples.append(proc.cpu_percent(interval=None))
            await asyncio.sleep(0.2)

    cpu_task = asyncio.create_task(_poll_cpu())
    await asyncio.gather(*tasks)
    cpu_task.cancel()

    wall_elapsed = time.perf_counter() - wall_start
    rss_after = _rss_mb()

    cpu_mean = sum(cpu_samples) / len(cpu_samples) if cpu_samples else 0.0

    total_calls = n
    throughput = total_calls / wall_elapsed if wall_elapsed > 0 else 0.0
    error_rate = errors / total_calls if total_calls > 0 else 0.0

    return BenchmarkResult(
        name=name,
        backend=backend,
        p50_ms=histogram.p50(),
        p95_ms=histogram.p95(),
        p99_ms=histogram.p99(),
        mean_ms=histogram.mean(),
        ops_per_sec=throughput,
        error_rate=error_rate,
        sample_count=histogram.count,
        warm_up_count=warm_up_count,
        concurrency=concurrency,
        extra={
            "rss_mb_before": rss_before,
            "rss_mb_after": rss_after,
            "rss_mb_delta": rss_after - rss_before,
            "cpu_percent_mean": cpu_mean,
            "wall_elapsed_s": wall_elapsed,
        },
    )


# ---------------------------------------------------------------------------
# JSON reporter
# ---------------------------------------------------------------------------


def write_json(
    results: list[BenchmarkResult],
    path: Path | str,
    *,
    scenario: str = "unknown",
    backend: str = "sqlite",
    dry_run: bool = False,
    extra_meta: dict[str, Any] | None = None,
    comparison_table: str | None = None,
) -> Path:
    """Write benchmark results to a JSON file following the .benchmarks/ contract.

    Creates parent directories as needed.  The output format is compatible with
    the existing ``benchmarks/pipeline/evaluate.py`` result structure so that
    tooling can read both.

    Args:
        results: List of ``BenchmarkResult`` objects to serialise.
        path: Destination file path.
        scenario: Scenario label (e.g. ``"smoke"``, ``"comparison"``).
        backend: Primary backend label for ``benchmark_meta``.
        dry_run: Record in metadata whether this was a dry-run.
        extra_meta: Additional key-value pairs merged into ``benchmark_meta``.
        comparison_table: Optional preformatted markdown comparison table.

    Returns:
        The resolved ``Path`` that was written.
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    git_sha = _git_sha()

    meta: dict[str, Any] = {
        "timestamp": datetime.now(UTC).isoformat(),
        "backend": backend,
        "scenario": scenario,
        "git_sha": git_sha,
        "dry_run": dry_run,
    }
    if extra_meta:
        meta.update(extra_meta)

    operations: dict[str, Any] = {}
    resource: dict[str, Any] = {}

    for r in results:
        op_dict: dict[str, Any] = {
            "p50_ms": round(r.p50_ms, 3),
            "p95_ms": round(r.p95_ms, 3),
            "p99_ms": round(r.p99_ms, 3),
            "mean_ms": round(r.mean_ms, 3),
            "ops_per_sec": round(r.ops_per_sec, 3),
            "error_rate": round(r.error_rate, 6),
            "sample_count": r.sample_count,
            "concurrency": r.concurrency,
        }
        if r.recall_at_5 is not None:
            op_dict["recall_at_5"] = round(r.recall_at_5, 4)
        operations[r.name] = op_dict

        # Aggregate resource stats (last writer wins per key)
        if r.extra:
            resource.setdefault("rss_mb_delta", r.extra.get("rss_mb_delta", 0.0))
            resource.setdefault("cpu_percent_mean", r.extra.get("cpu_percent_mean", 0.0))

    payload: dict[str, Any] = {
        "benchmark_meta": meta,
        "operations": operations,
        "resource": resource,
    }
    if comparison_table:
        payload["comparison_table"] = comparison_table

    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)

    logger.info("benchmark results written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Baseline regression gate
# ---------------------------------------------------------------------------


class BaselineViolation(Exception):
    """Raised when a benchmark result exceeds a stored baseline by too much."""


def check_baselines(
    results: list[BenchmarkResult],
    baselines_path: Path | str,
    *,
    multiplier: float = 2.0,
) -> None:
    """Assert that no result exceeds the stored baseline p99 by *multiplier*×.

    Args:
        results: Benchmark results to validate.
        baselines_path: Path to ``baselines.json``.
        multiplier: Maximum allowed ratio ``result.p99 / baseline.p99``.
            Defaults to ``2.0`` (any p99 more than 2× baseline fails).

    Raises:
        BaselineViolation: If any operation's p99 exceeds the threshold.
        FileNotFoundError: If *baselines_path* does not exist.
    """
    baselines_path = Path(baselines_path)
    baselines: dict[str, Any] = json.loads(baselines_path.read_text(encoding="utf-8"))

    violations: list[str] = []
    for result in results:
        key = result.name
        if key not in baselines:
            logger.debug("no baseline for operation %r — skipping gate", key)
            continue
        threshold_p99 = baselines[key].get("p99_ms", 0) * multiplier
        if result.p99_ms > threshold_p99:
            violations.append(
                f"{key}: p99 {result.p99_ms:.1f} ms exceeds {threshold_p99:.1f} ms "
                f"({multiplier}× baseline {baselines[key]['p99_ms']} ms)"
            )

    if violations:
        raise BaselineViolation(
            f"Baseline regression gate failed ({len(violations)} violation(s)):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rss_mb() -> float:
    """Return current process RSS in MiB, or 0.0 if psutil is unavailable."""
    try:
        import psutil

        return psutil.Process().memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def _git_sha() -> str:
    """Return the current HEAD git SHA (short), or ``"unknown"``."""
    try:
        result = _run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=3,
        )
        return result.stdout.strip() or "unknown"
    except Exception:
        return "unknown"
