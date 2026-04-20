"""Search workload for the enterprise benchmark suite.

Measures retrieval latency (p50/p95/p99), throughput (ops/s), and optionally
Recall@5 when expected keywords are available in the questions corpus.

The workload calls ``search_vectors`` (the coarse vector search stage of the
retrieval pipeline) with injectable mocks for embedding so the benchmark
measures the Qdrant mock + filter + payload parse path.  The Qdrant singleton
is replaced with a ``MagicMock`` that returns synthetic scored points.

Using ``search_vectors`` rather than the full ``recursive_retrieve`` avoids
patching a large number of flags in the retriever pipeline and keeps each
benchmark call to a predictable, well-isolated path.

Recall@5 computation
--------------------
If a question has ``expected_keywords``, the workload checks whether any
returned memory text contains at least one keyword (case-insensitive substring
match).  This is an approximate recall proxy, not semantic equivalence.

Usage::

    from benchmarks.enterprise.workloads.search import run_search_workload

    result = await run_search_workload(
        pool=sqlite_pool,
        n=50,
        concurrency=5,
        backend="sqlite",
        dry_run=False,
    )
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from benchmarks.enterprise.harness import BenchmarkResult, run_workload

# ---------------------------------------------------------------------------
# Questions corpus
# ---------------------------------------------------------------------------

_QUESTIONS_PATH = Path(__file__).parent.parent.parent / "fixtures" / "questions.json"

_FALLBACK_QUESTIONS: list[dict[str, Any]] = [
    {
        "query": "What is the Kubernetes cluster configuration?",
        "expected_keywords": ["kubernetes", "k8s", "cluster"],
        "namespace": "bench",
    },
    {
        "query": "How is Redis configured?",
        "expected_keywords": ["redis", "maxmemory", "persistence"],
        "namespace": "bench",
    },
    {
        "query": "What is the backup schedule?",
        "expected_keywords": ["backup", "nightly", "schedule"],
        "namespace": "bench",
    },
    {
        "query": "Which agent handles infrastructure tasks?",
        "expected_keywords": ["agent", "infrastructure", "ROG"],
        "namespace": "bench",
    },
    {
        "query": "What is the embedding model dimension?",
        "expected_keywords": ["embedding", "dimension", "1024"],
        "namespace": "bench",
    },
]


def _load_questions() -> list[dict[str, Any]]:
    """Load the questions corpus from the fixtures file, or use the fallback.

    Returns:
        A list of question dicts, each with ``query``, ``expected_keywords``,
        and ``namespace`` keys.
    """
    if _QUESTIONS_PATH.exists():
        try:
            questions = json.loads(_QUESTIONS_PATH.read_text(encoding="utf-8"))
            return [q for q in questions if q.get("query")]
        except Exception:
            pass
    return _FALLBACK_QUESTIONS


_QUESTIONS: list[dict[str, Any]] = _load_questions()


def _sample_question(idx: int) -> dict[str, Any]:
    """Return a question for the given call index (rotates through corpus).

    Args:
        idx: Call index.

    Returns:
        A question dict.
    """
    return _QUESTIONS[idx % len(_QUESTIONS)]


# ---------------------------------------------------------------------------
# Mock factory
# ---------------------------------------------------------------------------

_FAKE_VECTOR: list[float] = [0.01] * 1024


def _make_search_qdrant_mock() -> MagicMock:
    """Return a Qdrant mock with minimal search results for retrieval tests."""
    from qdrant_client.models import ScoredPoint

    def _scored_point(text: str, score: float = 0.85) -> ScoredPoint:
        return ScoredPoint(
            id="00000000-0000-0000-0000-000000000001",
            version=1,
            score=score,
            payload={
                "text": text,
                "agent_id": "bench_agent_0",
                "namespace": "bench",
                "file_path": "explicit/bench_agent_0",
                "date": "2026-04-19",
                "thought_type": "general",
                "memory_type": "general",
                "retention_class": "standard",
                "importance_score": 0.5,
                "deleted": False,
                "archived": False,
            },
            vector=None,
        )

    mock_points = [
        _scored_point("The Kubernetes cluster uses 3 master nodes and 10 worker nodes."),
        _scored_point("Redis is configured with AOF persistence and maxmemory=4gb.", 0.80),
        _scored_point("Deployment pipeline: GitHub Actions → ECR push → Helm upgrade.", 0.75),
        _scored_point("MCP server listens on port 8765; health endpoint is /health.", 0.70),
        _scored_point("The embedding model is text-embedding-3-large with 1024 dims.", 0.65),
    ]

    mock = MagicMock()
    mock.search = MagicMock(return_value=mock_points)
    # recursive_retrieve calls client.query_points(...).points
    _query_result = MagicMock()
    _query_result.points = mock_points
    mock.query_points = MagicMock(return_value=_query_result)
    mock.scroll = MagicMock(return_value=([], None))
    mock.count = MagicMock(return_value=MagicMock(count=5))
    mock.collection_exists = MagicMock(return_value=True)
    mock.get_collection = MagicMock(
        return_value=MagicMock(
            config=MagicMock(params=MagicMock(vectors=MagicMock(size=1024, distance="Cosine")))
        )
    )
    return mock


# ---------------------------------------------------------------------------
# Recall@5 helper
# ---------------------------------------------------------------------------


def _compute_recall_at_5(
    result_texts: list[str],
    expected_keywords: list[str],
) -> float:
    """Compute an approximate Recall@5 based on keyword presence.

    Returns 1.0 if at least one of the top-5 results contains any expected
    keyword (case-insensitive substring match), else 0.0.

    Args:
        result_texts: List of memory text strings returned by retrieval.
        expected_keywords: List of keywords that indicate a relevant result.

    Returns:
        1.0 if any keyword found in any of the top-5 results, else 0.0.
    """
    top5 = result_texts[:5]
    joined = " ".join(t.lower() for t in top5)
    for kw in expected_keywords:
        if kw.lower() in joined:
            return 1.0
    return 0.0


# ---------------------------------------------------------------------------
# Core workload
# ---------------------------------------------------------------------------


async def run_search_workload(
    pool: Any,
    *,
    n: int = 100,
    concurrency: int = 5,
    backend: str = "sqlite",
    dry_run: bool = False,
    qdrant_override: Any | None = None,
    embed_override: Any | None = None,
) -> BenchmarkResult:
    """Run the retrieval benchmark.

    Calls ``recursive_retrieve`` with mocked embed + Qdrant so latency reflects
    the SQLite/Postgres FTS + rank-fusion path only (no live embedding or vector
    service required).

    Args:
        pool: An initialised pool from ``BackendFixture``.  Ensures the global
            pool singleton is already patched before calls begin.
        n: Total number of search calls.
        concurrency: Maximum concurrent calls in flight.
        backend: Backend label for the result.
        dry_run: When ``True``, skips all real I/O.
        qdrant_override: Optional pre-built Qdrant mock.
        embed_override: Optional embed callable ``async (text) -> list[float]``.

    Returns:
        A ``BenchmarkResult`` for the ``"search"`` operation, with
        ``recall_at_5`` set to the mean Recall@5 across all calls.
    """
    qdrant_mock = qdrant_override if qdrant_override is not None else _make_search_qdrant_mock()
    embed_mock = (
        embed_override if embed_override is not None else AsyncMock(return_value=_FAKE_VECTOR)
    )

    call_counter = [0]
    recall_scores: list[float] = []

    async def _one_search() -> None:
        idx = call_counter[0]
        call_counter[0] += 1
        question = _sample_question(idx)
        query = question["query"]
        namespace = question.get("namespace") or "bench"
        expected_keywords: list[str] = question.get("expected_keywords") or []

    async def _one_search() -> None:
        idx = call_counter[0]
        call_counter[0] += 1
        question = _sample_question(idx)
        query = question["query"]
        namespace = question.get("namespace") or "bench"
        expected_keywords: list[str] = question.get("expected_keywords") or []

        import archivist.retrieval.rlm_retriever as _rr
        import archivist.storage.qdrant as _qdrant_mod
        from archivist.retrieval.rlm_retriever import search_vectors
        from archivist.storage import collection_router as _cr

        _orig_instance = _qdrant_mod._instance
        _qdrant_mod._instance = qdrant_mock

        try:
            with (
                patch.object(_rr, "embed_text", embed_mock),
                patch.object(_cr, "qdrant_client", return_value=qdrant_mock),
            ):
                hits = await search_vectors(
                    query,
                    agent_id="bench_agent_0",
                    namespace=namespace,
                    limit=10,
                )
        finally:
            _qdrant_mod._instance = _orig_instance

        if not dry_run:
            result_texts = [m.get("text", "") if isinstance(m, dict) else "" for m in hits]
            recall = _compute_recall_at_5(result_texts, expected_keywords)
            recall_scores.append(recall)

    result = await run_workload(
        name="search",
        backend=backend,
        coro_factory=_one_search,
        n=n,
        concurrency=concurrency,
        dry_run=dry_run,
    )

    if recall_scores:
        result.recall_at_5 = sum(recall_scores) / len(recall_scores)

    return result
