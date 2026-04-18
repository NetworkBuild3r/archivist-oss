"""Unit tests for Phase 6 provenance types and propagation."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from provenance import ActorType, SourceTrace, default_confidence

# ── SourceTrace dataclass ─────────────────────────────────────────────────────


def test_source_trace_round_trip():
    st = SourceTrace(
        tool="archivist_store",
        session_id="sess-123",
        upstream_source="slack",
        parent_memory_id="mem-abc",
        extra={"channel": "#ops"},
    )
    d = st.to_dict()
    assert d["tool"] == "archivist_store"
    assert d["session_id"] == "sess-123"
    assert d["upstream_source"] == "slack"
    assert d["parent_memory_id"] == "mem-abc"
    assert d["extra"] == {"channel": "#ops"}

    restored = SourceTrace.from_dict(d)
    assert restored.tool == st.tool
    assert restored.session_id == st.session_id
    assert restored.upstream_source == st.upstream_source
    assert restored.parent_memory_id == st.parent_memory_id
    assert restored.extra == st.extra


def test_source_trace_empty_omits_keys():
    st = SourceTrace()
    d = st.to_dict()
    assert d == {}


def test_source_trace_from_dict_none():
    st = SourceTrace.from_dict(None)
    assert st.tool == ""
    assert st.session_id == ""


def test_source_trace_with_parent():
    st = SourceTrace(tool="archivist_store", session_id="s1")
    derived = st.with_parent("parent-id-123")
    assert derived.parent_memory_id == "parent-id-123"
    assert derived.tool == "archivist_store"
    assert derived.session_id == "s1"
    assert st.parent_memory_id == ""  # original unchanged


# ── ActorType enum ────────────────────────────────────────────────────────────


def test_actor_type_values():
    assert ActorType.AGENT.value == "agent"
    assert ActorType.HUMAN.value == "human"
    assert ActorType.SYSTEM.value == "system"
    assert ActorType.TOOL.value == "tool"


def test_actor_type_string_comparison():
    assert ActorType.AGENT == "agent"
    assert ActorType.SYSTEM == "system"


# ── default_confidence ────────────────────────────────────────────────────────


def test_default_confidence_by_actor_type():
    cmap = {"human": 1.0, "agent": 0.8, "system": 0.7, "tool": 0.7, "extracted": 0.5}
    assert default_confidence("human", cmap) == 1.0
    assert default_confidence("agent", cmap) == 0.8
    assert default_confidence("system", cmap) == 0.7
    assert default_confidence("extracted", cmap) == 0.5


def test_default_confidence_unknown_falls_back_to_agent():
    cmap = {"human": 1.0, "agent": 0.8}
    assert default_confidence("unknown_type", cmap) == 0.8


# ── ResultCandidate provenance fields ─────────────────────────────────────────


def test_result_candidate_from_qdrant_payload_provenance():
    from result_types import ResultCandidate, RetrievalSource

    payload = {
        "text": "hello world",
        "agent_id": "alice",
        "actor_id": "bob",
        "actor_type": "human",
        "confidence": 0.95,
        "source_trace": {"tool": "archivist_store", "session_id": "s1"},
    }
    rc = ResultCandidate.from_qdrant_payload("pt-1", payload, score=0.9)
    assert rc.actor_id == "bob"
    assert rc.actor_type == "human"
    assert rc.confidence == 0.95
    assert rc.source_trace["tool"] == "archivist_store"
    assert rc.source == RetrievalSource.VECTOR


def test_result_candidate_from_bm25_hit_provenance():
    from result_types import ResultCandidate, RetrievalSource

    hit = {
        "qdrant_id": "q1",
        "text": "some text",
        "agent_id": "alice",
        "actor_id": "file_indexer",
        "actor_type": "system",
        "bm25_rank": -5,
    }
    rc = ResultCandidate.from_bm25_hit(hit)
    assert rc.actor_id == "file_indexer"
    assert rc.actor_type == "system"
    assert rc.source == RetrievalSource.BM25


def test_result_candidate_from_registry_hit_provenance():
    from result_types import ResultCandidate, RetrievalSource

    hit = {
        "memory_id": "m1",
        "chunk_text": "192.168.1.1",
        "agent_id": "alice",
        "actor_id": "bob",
        "actor_type": "human",
        "namespace": "ops",
    }
    rc = ResultCandidate.from_registry_hit(hit)
    assert rc.actor_id == "bob"
    assert rc.actor_type == "human"
    assert rc.source == RetrievalSource.REGISTRY


def test_result_candidate_to_dict_includes_provenance():
    from result_types import ResultCandidate

    rc = ResultCandidate(
        id="p1",
        text="test",
        actor_id="bob",
        actor_type="human",
        confidence=0.9,
        source_trace={"tool": "test"},
    )
    d = rc.to_dict()
    assert d["actor_id"] == "bob"
    assert d["actor_type"] == "human"
    assert d["confidence"] == 0.9
    assert d["source_trace"]["tool"] == "test"


def test_result_candidate_update_from_payload_provenance():
    from result_types import ResultCandidate

    rc = ResultCandidate(id="p1", text="old")
    rc.update_from_payload(
        {
            "text": "new",
            "actor_id": "carol",
            "actor_type": "tool",
            "confidence": 0.6,
            "source_trace": {"tool": "file_indexer"},
        }
    )
    assert rc.actor_id == "carol"
    assert rc.actor_type == "tool"
    assert rc.confidence == 0.6
    assert rc.source_trace["tool"] == "file_indexer"


# ── resolve_actor ─────────────────────────────────────────────────────────────


def test_resolve_actor_defaults_to_agent_id():
    from handlers._common import resolve_actor

    actor_id, actor_type = resolve_actor({"agent_id": "alice"})
    assert actor_id == "alice"
    assert actor_type == "agent"


def test_resolve_actor_explicit():
    from handlers._common import resolve_actor

    actor_id, actor_type = resolve_actor(
        {
            "agent_id": "alice",
            "actor_id": "bob",
            "actor_type": "human",
        }
    )
    assert actor_id == "bob"
    assert actor_type == "human"


# ── Reranker _build_pair provenance context ───────────────────────────────────


def test_reranker_build_pair_includes_provenance():
    from reranker import _build_pair

    candidate = {
        "text": "hello world",
        "actor_type": "human",
        "confidence": 0.95,
    }
    passage = _build_pair("query", candidate)
    assert "Actor: human" in passage
    assert "Confidence: 0.95" in passage


def test_reranker_build_pair_no_provenance():
    from reranker import _build_pair

    candidate = {"text": "hello world"}
    passage = _build_pair("query", candidate)
    assert "Actor:" not in passage
    assert "Confidence:" not in passage


# ── Augment header ────────────────────────────────────────────────────────────


def test_augment_chunk_uses_actor_over_agent():
    from contextual_augment import augment_chunk

    result = augment_chunk("test text", agent_id="alice", actor_id="bob", actor_type="human")
    assert "Actor: bob (human)" in result
    assert "Agent: alice" not in result


def test_augment_chunk_falls_back_to_agent():
    from contextual_augment import augment_chunk

    result = augment_chunk("test text", agent_id="alice")
    assert "Agent: alice" in result


# ── Derived artifact provenance inheritance (synthetic questions) ─────────────


def test_source_trace_with_parent_for_synthetic_questions():
    """Synthetic question points inherit provenance with parent_memory_id set."""
    parent_trace = SourceTrace(tool="archivist_store", session_id="s1", upstream_source="slack")
    sq_trace = parent_trace.with_parent("pid-123")
    d = sq_trace.to_dict()
    assert d["parent_memory_id"] == "pid-123"
    assert d["tool"] == "archivist_store"
    assert d["session_id"] == "s1"
    assert d["upstream_source"] == "slack"


# ── Derived artifact provenance inheritance (reverse HyDE) ────────────────────


def test_source_trace_with_parent_for_reverse_hyde():
    """Reverse HyDE points inherit provenance with parent_memory_id set."""
    parent_trace = SourceTrace(tool="archivist_store", upstream_source="jira")
    rh_trace = parent_trace.with_parent("pid-456")
    d = rh_trace.to_dict()
    assert d["parent_memory_id"] == "pid-456"
    assert d["tool"] == "archivist_store"
    assert d["upstream_source"] == "jira"


# ── Graph retrieval confidence filtering ──────────────────────────────────────


async def test_build_entity_fact_results_filters_low_confidence(monkeypatch):
    """Facts below MIN_FACT_CONFIDENCE should be excluded from results."""
    from unittest.mock import AsyncMock

    import graph_retrieval as gr

    monkeypatch.setattr(gr, "MIN_FACT_CONFIDENCE", 0.5)

    high_conf_fact = {
        "fact_text": "Bob manages the ops cluster",
        "agent_id": "alice",
        "confidence": 0.9,
        "created_at": "2026-01-01T00:00:00Z",
        "retention_class": "standard",
        "source_file": "",
        "valid_from": "",
        "valid_until": "",
        "actor_id": "alice",
    }
    low_conf_fact = {
        "fact_text": "Bob might use vim",
        "agent_id": "alice",
        "confidence": 0.2,
        "created_at": "2026-01-01T00:00:00Z",
        "retention_class": "standard",
        "source_file": "",
        "valid_from": "",
        "valid_until": "",
        "actor_id": "alice",
    }

    monkeypatch.setattr(
        gr,
        "get_entity_facts_bulk",
        AsyncMock(return_value={1: [high_conf_fact, low_conf_fact]}),
    )

    entities = [{"id": 1, "name": "Bob", "retention_class": "standard"}]
    results = await gr.build_entity_fact_results(entities, min_score=0.7)

    assert len(results) == 1
    assert "manages" in results[0]["text"]
    assert results[0]["confidence"] == 0.9
