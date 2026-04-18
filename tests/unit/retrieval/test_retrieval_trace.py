"""Unit tests for Phase 1 foundations — retrieval trace shape."""

import pytest

from rlm_retriever import _retrieval_trace

pytestmark = [pytest.mark.unit, pytest.mark.retrieval]


def test_retrieval_trace_keys():
    t = _retrieval_trace(
        vector_limit=64,
        coarse_count=10,
        deduped_count=8,
        threshold=0.65,
        after_threshold_count=5,
        after_rerank_count=5,
        parent_enriched=False,
        refinement_chunks=3,
    )
    assert t["coarse_hits"] == 10
    assert t["after_dedupe"] == 8
    assert t["threshold"] == 0.65
    assert "rerank_enabled" in t
    assert "chunks_sent_to_refinement" in t
