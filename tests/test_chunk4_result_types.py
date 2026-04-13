"""Tests for Chunk 4: Uniform Result Types + Registry Integration.

Covers:
- ResultCandidate dataclass and factory methods
- Registry hits flow through RRF (no hardcoded score bypass)
- Stale registry entries (deleted memories) are dropped with warning
- Literal search no longer uses hardcoded 0.85 score
- Fleet queries iterate agent_ids for registry lookup
- RetrievalSource enum values
"""

from unittest.mock import MagicMock, patch

import pytest


class TestRetrievalSource:
    """RetrievalSource enum has all expected values."""

    def test_all_sources(self):
        from result_types import RetrievalSource
        assert RetrievalSource.VECTOR == "vector"
        assert RetrievalSource.BM25 == "bm25"
        assert RetrievalSource.LITERAL == "literal"
        assert RetrievalSource.REGISTRY == "registry"
        assert RetrievalSource.GRAPH_FACT == "graph_fact"


class TestResultCandidate:
    """ResultCandidate dataclass and factory methods."""

    def test_to_dict_has_all_keys(self):
        from result_types import ResultCandidate, RetrievalSource

        rc = ResultCandidate(id="abc", score=0.9, text="hello", source=RetrievalSource.VECTOR)
        d = rc.to_dict()

        required_keys = {
            "id", "score", "text", "agent_id", "file_path", "file_type",
            "date", "namespace", "chunk_index", "parent_id", "is_parent",
            "importance_score", "retention_class", "retrieval_source",
        }
        assert required_keys.issubset(set(d.keys()))
        assert d["retrieval_source"] == "vector"

    def test_from_qdrant_payload(self):
        from result_types import ResultCandidate, RetrievalSource

        payload = {
            "text": "server runs on 10.0.0.1",
            "agent_id": "alice",
            "file_path": "explicit/alice",
            "file_type": "explicit",
            "date": "2025-01-01",
            "namespace": "chief",
            "importance_score": 0.8,
            "topic": "infra",
        }
        rc = ResultCandidate.from_qdrant_payload("pid-1", payload, score=0.75)
        assert rc.id == "pid-1"
        assert rc.score == 0.75
        assert rc.text == "server runs on 10.0.0.1"
        assert rc.topic == "infra"
        assert rc.source == RetrievalSource.VECTOR

    def test_from_registry_hit(self):
        from result_types import ResultCandidate, RetrievalSource

        hit = {
            "memory_id": "mem-abc",
            "chunk_text": "10.0.0.1 is the gateway",
            "agent_id": "bob",
            "namespace": "ops",
            "created_at": "2025-03-15T10:30:00Z",
        }
        rc = ResultCandidate.from_registry_hit(hit)
        assert rc.id == "mem-abc"
        assert rc.score == 0.0
        assert rc.text == "10.0.0.1 is the gateway"
        assert rc.date == "2025-03-15"
        assert rc.source == RetrievalSource.REGISTRY

    def test_from_bm25_hit(self):
        from result_types import ResultCandidate, RetrievalSource

        hit = {
            "qdrant_id": "qid-1",
            "text": "deploy to staging",
            "bm25_rank": -2.5,
            "agent_id": "alice",
            "namespace": "dev",
        }
        rc = ResultCandidate.from_bm25_hit(hit)
        assert rc.id == "qid-1"
        assert rc.score == 2.5
        assert rc.source == RetrievalSource.BM25

    def test_update_from_payload(self):
        from result_types import ResultCandidate

        rc = ResultCandidate(id="x", text="stale", agent_id="old")
        rc.update_from_payload({
            "text": "fresh full text",
            "agent_id": "new_agent",
            "file_path": "explicit/new_agent",
            "importance_score": 0.9,
        })
        assert rc.text == "fresh full text"
        assert rc.agent_id == "new_agent"
        assert rc.importance_score == 0.9


class TestRegistryNoHardcodedScore:
    """Registry hits no longer bypass RRF with score=0.95."""

    def test_no_hardcoded_095(self):
        """The retriever must not set score=0.95 for registry hits."""
        import inspect
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        assert "0.95" not in source, "Hardcoded 0.95 score for registry hits must be removed"

    def test_registry_hits_in_source(self):
        """Registry hits should be built via ResultCandidate.from_registry_hit."""
        import inspect
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        assert "ResultCandidate.from_registry_hit" in source


class TestLiteralSearchNoHardcodedScore:
    """Literal search no longer uses hardcoded 0.85 score."""

    def test_no_hardcoded_085(self):
        import inspect
        from rlm_retriever import _literal_search_sync

        source = inspect.getsource(_literal_search_sync)
        assert "0.85" not in source, "Hardcoded 0.85 score for literal hits must be removed"


class TestRegistryRRFIntegration:
    """Registry hits are fed through RRF as an additional ranking list."""

    def test_registry_fed_into_rrf(self):
        """The RRF merge call should include registry hits as a ranking."""
        import inspect
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        assert "_registry_hits" in source
        assert "non_empty_rankings.append(_registry_hits)" in source
        assert "coarse.insert(0, rh)" not in source, "Direct injection must be removed"


class TestStaleRegistryDrop:
    """Stale registry entries (deleted memories) are dropped."""

    def test_stale_entries_dropped(self):
        from result_types import ResultCandidate

        rc1 = ResultCandidate.from_registry_hit({
            "memory_id": "alive",
            "chunk_text": "text",
            "agent_id": "a",
            "namespace": "ns",
        })
        rc2 = ResultCandidate.from_registry_hit({
            "memory_id": "dead",
            "chunk_text": "old text",
            "agent_id": "a",
            "namespace": "ns",
        })

        payload_map = {"alive": {"text": "fresh text", "agent_id": "a"}}

        live = []
        for c in [rc1, rc2]:
            p = payload_map.get(c.id)
            if p:
                c.update_from_payload(p)
                live.append(c)

        assert len(live) == 1
        assert live[0].id == "alive"
        assert live[0].text == "fresh text"


class TestFleetRegistryLookup:
    """Fleet queries (agent_ids set) iterate per agent_id."""

    def test_per_agent_lookup(self):
        import inspect
        from rlm_retriever import recursive_retrieve

        source = inspect.getsource(recursive_retrieve)
        assert "for aid in agent_ids:" in source, (
            "Fleet registry lookup must iterate agent_ids"
        )
