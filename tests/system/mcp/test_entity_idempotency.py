"""System tests verifying that archivist_store handles duplicate entities gracefully.

These tests exercise the public MCP tool interface (``_handle_store``) rather
than internal graph functions, confirming that the full stack from tool call
through entity upsert never returns an error response when a duplicate entity
is encountered.

This is the regression guard for the 'archivist_store tool encountered a fetch
failure' symptom that caused agents to enter endless retry loops.

Run with::

    pytest tests/system/mcp/test_entity_idempotency.py -v -m system
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.system, pytest.mark.mcp]

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_MOCK_VEC = [0.0] * 1536


def _parse_response(result) -> dict:
    assert isinstance(result, list) and len(result) > 0, "Handler returned empty list"
    assert result[0].type == "text", f"Expected text response, got {result[0].type!r}"
    return json.loads(result[0].text)


def _assert_no_error(data: dict, context: str = "") -> None:
    prefix = f"[{context}] " if context else ""
    assert "error" not in data, f"{prefix}Handler returned error: {data}"
    assert data.get("stored") is True or data.get("status") == "success", (
        f"{prefix}Expected stored=True or status=success, got: {data}"
    )


# ---------------------------------------------------------------------------
# Fixture: patch all external I/O (matches pattern from test_integrity_storage)
# ---------------------------------------------------------------------------


@pytest.fixture
def _mock_externals(monkeypatch):
    """Patch Qdrant, embeddings, LLM, and RBAC so tests run without real backends."""
    mock_client = MagicMock()
    mock_client.upsert = MagicMock(return_value=None)
    mock_client.retrieve = MagicMock(return_value=[])

    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage._rbac_gate",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.qdrant_client",
        MagicMock(return_value=mock_client),
    )
    # ensure_collection is called synchronously — must NOT be AsyncMock
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.ensure_collection",
        MagicMock(return_value="test_collection"),
    )
    monkeypatch.setattr(
        "archivist.storage.collection_router.collection_for",
        MagicMock(return_value="test_collection"),
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.embed_text",
        AsyncMock(return_value=_MOCK_VEC),
    )
    monkeypatch.setattr(
        "archivist.features.embeddings.embed_text",
        AsyncMock(return_value=_MOCK_VEC),
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.embed_batch",
        AsyncMock(return_value=[_MOCK_VEC]),
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.check_for_conflicts",
        AsyncMock(return_value=MagicMock(has_conflict=False)),
    )
    monkeypatch.setattr(
        "archivist.write.conflict_detection.check_for_conflicts",
        AsyncMock(return_value=MagicMock(has_conflict=False)),
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.llm_adjudicated_dedup",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.register_memory_points_batch",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "archivist.storage.graph.register_memory_points_batch",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.compute_ttl",
        MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "archivist.write.indexer.compute_ttl",
        MagicMock(return_value=None),
    )
    monkeypatch.setattr(
        "archivist.core.audit.log_memory_event",
        AsyncMock(return_value=None),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.system
async def test_archivist_store_handles_duplicate_entity_gracefully(qa_pool, _mock_externals):
    """archivist_store called twice with the same explicit entity returns success both times.

    Regression test: before the ON CONFLICT DO UPDATE fix, the second call
    would raise sqlite3.IntegrityError and the handler would return an error
    response, causing agents to retry indefinitely.
    """
    from archivist.app.handlers.tools_storage import _handle_store

    r1 = await _handle_store(
        {
            "text": "brian manages the Kubernetes cluster",
            "agent_id": "agent-alpha",
            "entities": ["brian"],
            "namespace": "shared",
        }
    )
    data1 = _parse_response(r1)
    _assert_no_error(data1, "first store")

    r2 = await _handle_store(
        {
            "text": "brian deployed ArgoCD in the cluster",
            "agent_id": "agent-beta",
            "entities": ["brian"],
            "namespace": "shared",
        }
    )
    data2 = _parse_response(r2)
    _assert_no_error(data2, "second store (duplicate entity)")


@pytest.mark.system
async def test_archivist_store_concurrent_same_entity_all_succeed(qa_pool, _mock_externals):
    """Five concurrent archivist_store calls with the same entity must all return success.

    This is the agent-fleet migration scenario: multiple agents simultaneously
    store facts referencing the same entity.
    """
    from archivist.app.handlers.tools_storage import _handle_store

    async def store(agent_idx: int):
        return await _handle_store(
            {
                "text": f"agent-{agent_idx} knows about brian",
                "agent_id": f"agent-{agent_idx:03d}",
                "entities": ["brian"],
                "namespace": "shared",
            }
        )

    results = await asyncio.gather(*[store(i) for i in range(5)], return_exceptions=True)

    exceptions = [r for r in results if isinstance(r, Exception)]
    assert not exceptions, f"Concurrent store calls raised: {exceptions}"

    for idx, result in enumerate(results):
        data = _parse_response(result)  # type: ignore[arg-type]
        _assert_no_error(data, f"concurrent store #{idx}")


@pytest.mark.system
async def test_archivist_store_no_entity_still_succeeds(qa_pool, _mock_externals):
    """archivist_store without explicit entities (auto-extract path) must succeed."""
    from archivist.app.handlers.tools_storage import _handle_store

    result = await _handle_store(
        {
            "text": "The Grafana dashboard shows CPU spikes on node-3",
            "agent_id": "monitor-agent",
            "namespace": "shared",
        }
    )
    data = _parse_response(result)
    _assert_no_error(data, "no explicit entities")
