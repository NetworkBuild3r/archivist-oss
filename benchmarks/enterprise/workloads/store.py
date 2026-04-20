"""Store workload for the enterprise benchmark suite.

Measures write-path latency and throughput for ``archivist_store`` operations.

The workload calls ``_handle_store`` directly (not via HTTP) with injectable
mocks for Qdrant and embedding so the benchmark measures the SQLite/Postgres
storage path without requiring live external services.

Mock injection seams
--------------------
- ``embed_text`` / ``embed_batch`` — patched to return a fixed vector.
- ``qdrant_client()`` — patched to return a ``MagicMock`` that accepts all
  ``upsert``, ``search``, ``scroll``, and ``count`` calls.
- ``CONFLICT_CHECK_ON_STORE`` config flag — forced ``False`` to skip the LLM
  dedup round-trip and measure the pure storage path.

Usage::

    from benchmarks.enterprise.workloads.store import run_store_workload

    result = await run_store_workload(
        pool=sqlite_pool,
        n=50,
        concurrency=5,
        backend="sqlite",
        dry_run=False,
    )
"""

from __future__ import annotations

import uuid
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from benchmarks.enterprise.harness import BenchmarkResult, run_workload

# ---------------------------------------------------------------------------
# Synthetic memory corpus
# ---------------------------------------------------------------------------

_MEMORY_TEXTS = [
    "The Kubernetes cluster uses 3 master nodes and 10 worker nodes.",
    "Redis is configured with AOF persistence and maxmemory=4gb policy=allkeys-lru.",
    "PostgreSQL primary runs on host db-01; replica on db-02 with streaming replication.",
    "Deployment pipeline: GitHub Actions → ECR push → Helm upgrade on EKS.",
    "Alert threshold for p99 API latency is 500 ms; pages on-call if exceeded for 5 min.",
    "Secrets are managed via AWS Secrets Manager; rotation is weekly for DB passwords.",
    "The embedding model is text-embedding-3-large with 1024 dimensions.",
    "Qdrant collection 'memories' uses HNSW m=16 ef_construct=100.",
    "Backup schedule: nightly full at 02:00 UTC, 6-hourly incrementals.",
    "The staging environment mirrors production but uses t3.medium instances.",
    "Agent ROG handles infrastructure tasks; Agent Alice handles code reviews.",
    "MCP server listens on port 8765; health endpoint is /health.",
    "FTS5 index is rebuilt nightly via curator; approximate token count is 2M.",
    "Rate limit for archivist_store is 100 req/s per namespace.",
    "Memory TTL for ephemeral retention is 7 days.",
    "The GRAPH_WRITE_LOCK serialises all SQLite DDL and DML writes.",
    "Outbox drain runs every 500 ms; max batch size is 50 events.",
    "Recall@5 on the LongMemEval benchmark is 0.87 (phase 6 baseline).",
    "Cross-agent memory sharing is capped at 10 results per query.",
    "Curator merges duplicate memories when cosine similarity exceeds 0.92.",
]


def _sample_text(idx: int) -> str:
    """Return a synthetic memory text for index *idx*."""
    return _MEMORY_TEXTS[idx % len(_MEMORY_TEXTS)]


def _sample_agent(idx: int, *, namespace_count: int = 1) -> tuple[str, str]:
    """Return ``(agent_id, namespace)`` for the given *idx*.

    Args:
        idx: Call index.
        namespace_count: Number of distinct namespaces to rotate through.

    Returns:
        A ``(agent_id, namespace)`` tuple.
    """
    ns_idx = idx % namespace_count
    namespace = f"bench_ns_{ns_idx}" if namespace_count > 1 else "bench"
    agent_id = f"bench_agent_{ns_idx}"
    return agent_id, namespace


# ---------------------------------------------------------------------------
# Mock factory
# ---------------------------------------------------------------------------


def _make_qdrant_mock() -> MagicMock:
    """Return a Qdrant client mock that accepts all benchmark calls."""
    client = MagicMock()
    client.upsert = MagicMock(return_value=MagicMock(operation_id=1))
    client.search = MagicMock(return_value=[])
    client.scroll = MagicMock(return_value=([], None))
    client.count = MagicMock(return_value=MagicMock(count=0))
    client.delete = MagicMock(return_value=MagicMock(operation_id=1))
    client.get_collections = MagicMock(return_value=MagicMock(collections=[]))
    client.collection_exists = MagicMock(return_value=True)
    client.get_collection = MagicMock(
        return_value=MagicMock(
            config=MagicMock(params=MagicMock(vectors=MagicMock(size=1024, distance="Cosine")))
        )
    )
    return client


_FAKE_VECTOR: list[float] = [0.01] * 1024


# ---------------------------------------------------------------------------
# Core workload
# ---------------------------------------------------------------------------


async def run_store_workload(
    pool: Any,
    *,
    n: int = 100,
    concurrency: int = 5,
    backend: str = "sqlite",
    namespace_count: int = 1,
    dry_run: bool = False,
    qdrant_override: Any | None = None,
    embed_override: Any | None = None,
) -> BenchmarkResult:
    """Run the store write-path benchmark.

    Calls ``_handle_store`` directly with mocked external services so the
    measured latency reflects the SQLite/Postgres storage layer only.

    Args:
        pool: An initialised SQLitePool or AsyncpgGraphBackend from
            ``BackendFixture``.  Not used directly but ensures the global pool
            singleton is patched before calls begin.
        n: Total number of store calls.
        concurrency: Maximum concurrent calls in flight.
        backend: Backend label for the result (``"sqlite"`` or ``"postgres"``).
        namespace_count: Number of distinct namespaces to distribute writes
            across.  1 = single namespace; >1 tests namespace fan-out.
        dry_run: When ``True``, skips all real I/O.
        qdrant_override: Optional pre-built Qdrant mock.  Defaults to a fresh
            ``_make_qdrant_mock()`` per run.
        embed_override: Optional embed callable ``async (text) -> list[float]``.
            Defaults to a fixed 1024-dim vector.

    Returns:
        A ``BenchmarkResult`` for the ``"store"`` operation.
    """
    qdrant_mock = qdrant_override if qdrant_override is not None else _make_qdrant_mock()
    embed_mock = (
        embed_override if embed_override is not None else AsyncMock(return_value=_FAKE_VECTOR)
    )
    embed_batch_mock = AsyncMock(return_value=[_FAKE_VECTOR])

    call_counter = [0]

    async def _one_store() -> None:
        idx = call_counter[0]
        call_counter[0] += 1
        agent_id, namespace = _sample_agent(idx, namespace_count=namespace_count)
        text = _sample_text(idx)
        unique_text = f"{text} [bench-{uuid.uuid4().hex[:8]}]"

        from archivist.app.handlers import tools_storage as _ts
        from archivist.storage import collection_router as _cr

        with (
            patch.object(_ts, "embed_text", embed_mock),
            patch.object(_ts, "embed_batch", embed_batch_mock),
            patch.object(_ts, "qdrant_client", return_value=qdrant_mock),
            patch.object(_cr, "qdrant_client", return_value=qdrant_mock),
            patch.object(_ts, "CONFLICT_CHECK_ON_STORE", False),
            patch.object(_ts, "CONFLICT_BLOCK_ON_STORE", False),
            # Disable background LLM tasks so benchmark measures storage path only
            patch("archivist.core.config.REVERSE_HYDE_ENABLED", False),
            patch("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False),
        ):
            await _ts._handle_store(
                {
                    "text": unique_text,
                    "agent_id": agent_id,
                    "namespace": namespace,
                    "force_skip_conflict_check": True,
                }
            )

    return await run_workload(
        name="store",
        backend=backend,
        coro_factory=_one_store,
        n=n,
        concurrency=concurrency,
        dry_run=dry_run,
    )
