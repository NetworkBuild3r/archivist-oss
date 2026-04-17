"""Integration tests for Phase 6 provenance pipeline: store → index → retrieve → rerank.

These tests exercise the full pipeline path and verify provenance propagation
across all artifact types. They use mocks for Qdrant and LLM calls but test
real SQLite operations and the full handler logic.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def _isolate_sqlite(tmp_path, monkeypatch):
    """Point SQLite at a temp dir so tests don't mutate the real DB."""
    db_path = str(tmp_path / "test_graph.db")
    monkeypatch.setenv("SQLITE_PATH", db_path)
    import config

    monkeypatch.setattr(config, "SQLITE_PATH", db_path)
    import graph

    monkeypatch.setattr(graph, "SQLITE_PATH", db_path)
    graph.init_schema()


def _store_patches(mock_client):
    """Return a stack of context managers that mock all external dependencies."""
    return [
        patch(
            "handlers.tools_storage.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024
        ),
        patch(
            "handlers.tools_storage.embed_batch",
            new_callable=AsyncMock,
            return_value=[[0.1] * 1024] * 20,
        ),
        patch("conflict_detection.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024),
        patch(
            "handlers.tools_storage.check_for_conflicts",
            new_callable=AsyncMock,
            return_value=MagicMock(has_conflict=False),
        ),
        patch(
            "handlers.tools_storage.llm_adjudicated_dedup",
            new_callable=AsyncMock,
            return_value=None,
        ),
        patch("handlers.tools_storage.qdrant_client", return_value=mock_client),
        patch("handlers.tools_storage.ensure_collection", return_value="test_coll"),
        patch("handlers.tools_storage.register_memory_points_batch", lambda pts: None),
        patch("handlers.tools_storage.get_namespace_config", return_value=None),
        patch("handlers.tools_storage._rbac_gate", return_value=None),
        patch("audit.log_memory_event", new_callable=AsyncMock),
    ]


# ── Integration: store with provenance → verify Qdrant payload ────────────────


@pytest.mark.asyncio
async def test_store_with_provenance_sets_payload(monkeypatch):
    """archivist_store with provenance fields propagates to primary Qdrant point."""
    monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", False)
    monkeypatch.setattr("config.SYNTHETIC_QUESTIONS_ENABLED", False)
    monkeypatch.setattr("config.BM25_ENABLED", False)
    monkeypatch.setattr("config.CONTEXTUAL_AUGMENTATION_ENABLED", False)

    captured = []
    mock_client = MagicMock()
    mock_client.upsert = lambda collection_name, points: captured.extend(points)

    patches = _store_patches(mock_client)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
        patches[9],
        patches[10],
    ):
        from handlers.tools_storage import _handle_store

        await _handle_store(
            {
                "text": "Server 192.168.1.100 is the production gateway",
                "agent_id": "alice",
                "actor_id": "bob",
                "actor_type": "human",
                "confidence": 0.95,
                "source_trace": {"tool": "manual_entry", "upstream_source": "slack"},
            }
        )

    primary = [p for p in captured if p.payload.get("representation_type") == "chunk"]
    assert len(primary) >= 1
    p = primary[0]
    assert p.payload["actor_id"] == "bob"
    assert p.payload["actor_type"] == "human"
    assert p.payload["confidence"] == 0.95
    assert p.payload["source_trace"]["tool"] == "manual_entry"
    assert p.payload["source_trace"]["upstream_source"] == "slack"
    assert p.payload["agent_id"] == "alice"


@pytest.mark.asyncio
async def test_store_provenance_defaults(monkeypatch):
    """When no provenance fields provided, defaults are applied."""
    monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", False)
    monkeypatch.setattr("config.SYNTHETIC_QUESTIONS_ENABLED", False)
    monkeypatch.setattr("config.BM25_ENABLED", False)
    monkeypatch.setattr("config.CONTEXTUAL_AUGMENTATION_ENABLED", False)

    captured = []
    mock_client = MagicMock()
    mock_client.upsert = lambda collection_name, points: captured.extend(points)

    patches = _store_patches(mock_client)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
        patches[9],
        patches[10],
    ):
        from handlers.tools_storage import _handle_store

        await _handle_store(
            {
                "text": "Default provenance test",
                "agent_id": "alice",
            }
        )

    primary = [p for p in captured if p.payload.get("representation_type") == "chunk"]
    assert len(primary) >= 1
    p = primary[0]
    assert p.payload["actor_id"] == "alice"
    assert p.payload["actor_type"] == "agent"
    assert p.payload["confidence"] == 0.8
    assert p.payload["source_trace"]["tool"] == "archivist_store"


# ── Integration: SQLite provenance columns ────────────────────────────────────


@pytest.mark.asyncio
async def test_store_propagates_to_sqlite(monkeypatch):
    """Verify provenance reaches SQLite tables (facts, memory_chunks)."""
    monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", False)
    monkeypatch.setattr("config.SYNTHETIC_QUESTIONS_ENABLED", False)
    monkeypatch.setattr("config.BM25_ENABLED", True)
    monkeypatch.setattr("config.CONTEXTUAL_AUGMENTATION_ENABLED", False)

    mock_client = MagicMock()
    mock_client.upsert = MagicMock()

    patches = _store_patches(mock_client)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
        patches[9],
        patches[10],
    ):
        from handlers.tools_storage import _handle_store

        await _handle_store(
            {
                "text": "Host 10.0.0.1 runs the cache service for team-alpha",
                "agent_id": "alice",
                "actor_id": "bob",
                "actor_type": "human",
                "confidence": 0.9,
            }
        )

    import graph

    conn = graph.get_db()
    try:
        facts = conn.execute("SELECT * FROM facts WHERE actor_id = 'bob'").fetchall()
        assert len(facts) >= 1
        confidences = sorted({dict(f)["confidence"] for f in facts})
        assert 0.9 in confidences

        chunks = conn.execute("SELECT * FROM memory_chunks WHERE actor_type = 'human'").fetchall()
        assert len(chunks) >= 1
        for c in chunks:
            c = dict(c)
            assert c["actor_id"] == "bob"
            assert c["actor_type"] == "human"
    finally:
        conn.close()


# ── Integration: ResultCandidate round-trip ───────────────────────────────────


def test_result_candidate_qdrant_round_trip():
    """Provenance survives payload → ResultCandidate → dict → back."""
    from result_types import ResultCandidate

    payload = {
        "text": "test",
        "agent_id": "alice",
        "actor_id": "bob",
        "actor_type": "human",
        "confidence": 0.95,
        "source_trace": {"tool": "archivist_store", "parent_memory_id": "pid-1"},
    }
    rc = ResultCandidate.from_qdrant_payload("p1", payload, score=0.9)
    d = rc.to_dict()

    rc2 = ResultCandidate.from_qdrant_payload(d["id"], d, score=d["score"])
    assert rc2.actor_id == "bob"
    assert rc2.actor_type == "human"
    assert rc2.confidence == 0.95
    assert rc2.source_trace["parent_memory_id"] == "pid-1"


# ── Integration: reranker receives provenance ─────────────────────────────────


def test_reranker_passage_includes_provenance_from_candidate():
    """The cross-encoder passage text includes provenance context."""
    from reranker import _build_pair

    candidate = {
        "text": "Server 10.0.0.1 is the gateway",
        "parent_text": "Infrastructure notes",
        "actor_type": "system",
        "confidence": 0.7,
    }
    passage = _build_pair("what is the gateway IP?", candidate)
    assert "Actor: system" in passage
    assert "Confidence: 0.7" in passage
    assert "Server 10.0.0.1 is the gateway" in passage
    assert "Infrastructure notes" in passage


# ── Integration: micro-chunk inherits provenance ─────────────────────────────


@pytest.mark.asyncio
async def test_micro_chunks_inherit_provenance(monkeypatch):
    """Micro-chunk Qdrant points carry the same provenance as the parent."""
    monkeypatch.setattr("config.REVERSE_HYDE_ENABLED", False)
    monkeypatch.setattr("config.SYNTHETIC_QUESTIONS_ENABLED", False)
    monkeypatch.setattr("config.BM25_ENABLED", False)
    monkeypatch.setattr("config.CONTEXTUAL_AUGMENTATION_ENABLED", False)
    monkeypatch.setattr("config.MAX_MICRO_CHUNKS_PER_MEMORY", 5)

    captured = []
    mock_client = MagicMock()
    mock_client.upsert = lambda collection_name, points: captured.extend(points)

    patches = _store_patches(mock_client)
    with (
        patches[0],
        patches[1],
        patches[2],
        patches[3],
        patches[4],
        patches[5],
        patches[6],
        patches[7],
        patches[8],
        patches[9],
        patches[10],
    ):
        from handlers.tools_storage import _handle_store

        await _handle_store(
            {
                "text": "IP address 192.168.1.100 is assigned to the prod gateway server with UUID a1b2c3d4-e5f6-7890-abcd-ef1234567890",
                "agent_id": "alice",
                "actor_id": "ops-bot",
                "actor_type": "tool",
                "confidence": 0.85,
            }
        )

    micro_chunks = [p for p in captured if p.payload.get("parent_id")]
    if micro_chunks:
        for mc in micro_chunks:
            assert mc.payload["actor_id"] == "ops-bot"
            assert mc.payload["actor_type"] == "tool"
            assert mc.payload["confidence"] == 0.85
