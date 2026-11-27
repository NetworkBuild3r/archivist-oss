"""Unit tests for Phase 3 auto-compress hook in rlm_retriever (MAJ-11, MAJ-13).

Exercises the real ``recursive_retrieve()`` overflow compression hook instead of
re-implementing the boolean guard inline.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [pytest.mark.unit]


def _make_hits(n: int = 5) -> list[dict]:
    return [
        {
            "id": f"mem-{i}",
            "score": 0.95 - i * 0.02,
            "text": f"Memory chunk {i} " + ("detail " * 40),
            "l0": f"L0 headline {i} " + ("word " * 15),
            "l1": f"L1 overview {i} " + ("word " * 25),
            "agent_id": "alice",
            "file_path": f"/notes/{i}.md",
            "date": "2026-04-01",
            "namespace": "test-ns",
        }
        for i in range(n)
    ]


async def _run_retrieve_with_packing(monkeypatch, *, auto_compress: bool, compact_mock: AsyncMock):
    """Run real recursive_retrieve(refine=False, max_tokens=...) with mocked I/O."""
    import archivist.retrieval.hot_cache as hot_cache
    import archivist.retrieval.rlm_retriever as rlm

    hot_cache.invalidate_all()
    hits = _make_hits(5)

    monkeypatch.setattr(rlm, "AUTO_COMPRESS_ENABLED", auto_compress)
    monkeypatch.setattr(rlm, "AUTO_COMPRESS_THRESHOLD", 0.5)
    monkeypatch.setattr(rlm, "RERANKER_ENABLED", False)
    monkeypatch.setattr(rlm, "BM25_ENABLED", False)
    monkeypatch.setattr(rlm, "GRAPH_RETRIEVAL_ENABLED", False)
    monkeypatch.setattr(rlm, "QUERY_EXPANSION_ENABLED", False)
    monkeypatch.setattr(rlm, "QUERY_CLASSIFICATION_ENABLED", False)
    monkeypatch.setattr(rlm, "TEMPORAL_DECAY_HALFLIFE_DAYS", 0)
    monkeypatch.setattr(rlm, "CONTEXT_MIN_FULL_RESULTS", 1)

    with (
        patch.object(rlm, "embed_batch", new=AsyncMock(return_value=[[0.1] * 8])),
        patch.object(rlm, "search_vectors", new=AsyncMock(return_value=hits)),
        patch.object(rlm, "lookup_needle_tokens", new=AsyncMock(return_value=[])),
        patch.object(rlm, "apply_hotness_to_results", new=AsyncMock(side_effect=lambda x: x)),
        patch.object(rlm, "get_outcome_adjustments", new=AsyncMock(return_value={})),
        patch.object(rlm, "_apply_rerank", new=AsyncMock(side_effect=lambda _q, x: x)),
        patch.object(rlm, "_maybe_iterative_retry", new=AsyncMock(return_value=None)),
        patch("archivist.write.compaction.compact_flat", new=compact_mock),
        patch(
            "archivist.storage.chunk_lifecycle.list_superseded_loser_ids",
            new=AsyncMock(return_value=set()),
        ),
    ):
        return await rlm.recursive_retrieve(
            "test auto compress query",
            agent_id="alice",
            namespace="test-ns",
            limit=5,
            refine=False,
            max_tokens=45,
            threshold=0.1,
            pack_policy="l2_first",
        )


class TestAutoCompressHookDisabled:
    """AUTO_COMPRESS_ENABLED=False (default) — hook must be a no-op."""

    async def test_no_compress_when_disabled(self, monkeypatch):
        compact_mock = AsyncMock(return_value="should not be called")
        res = await _run_retrieve_with_packing(
            monkeypatch, auto_compress=False, compact_mock=compact_mock
        )

        compact_mock.assert_not_awaited()
        ctx_status = res["retrieval_trace"]["context_status"]
        assert ctx_status["dropped_count"] > 0
        assert ctx_status["auto_compressed"] is False
        assert ctx_status["compress_savings_approx"] == 0
        assert all(not s.get("_is_synthetic_compress") for s in res["sources"])


class TestAutoCompressHookEnabled:
    """AUTO_COMPRESS_ENABLED=True — hook compacts the ACTUAL dropped items (MAJ-11)."""

    async def test_compress_compacts_dropped_not_kept_items(self, monkeypatch):
        compact_mock = AsyncMock(return_value="Compressed overflow summary.")
        res = await _run_retrieve_with_packing(
            monkeypatch, auto_compress=True, compact_mock=compact_mock
        )

        compact_mock.assert_awaited_once()
        overflow_pairs = compact_mock.await_args.args[0]
        compacted_ids = {pid for pid, _ in overflow_pairs}

        kept_ids = {s["id"] for s in res["sources"] if not s.get("_is_synthetic_compress")}
        assert kept_ids, "at least one kept item must remain in sources"
        assert compacted_ids, "dropped items must be passed to compact_flat"
        assert compacted_ids.isdisjoint(kept_ids), (
            f"auto-compress compacted already-kept items {compacted_ids & kept_ids} "
            f"instead of dropped items"
        )
        assert compacted_ids | kept_ids == {f"mem-{i}" for i in range(5)}

        synthetic = [s for s in res["sources"] if s.get("_is_synthetic_compress")]
        assert len(synthetic) == 1
        assert synthetic[0]["text"] == "Compressed overflow summary."
        ctx_status = res["retrieval_trace"]["context_status"]
        assert ctx_status["auto_compressed"] is True
        assert ctx_status["compress_savings_approx"] > 0
