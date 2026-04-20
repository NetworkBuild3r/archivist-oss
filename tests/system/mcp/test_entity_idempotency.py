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
# Helpers
# ---------------------------------------------------------------------------


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
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _mock_qdrant_and_embed(monkeypatch):
    """Patch all external I/O so storage tests run without Qdrant, LLM, or RBAC."""
    import numpy as np

    mock_client = MagicMock()
    mock_client.upsert = MagicMock(return_value=None)
    mock_client.get_collection = MagicMock(return_value=MagicMock(points_count=0))

    # RBAC: bypass namespace access checks — these tests focus on entity
    # idempotency, not RBAC policy.  All other system tests do the same.
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage._rbac_gate",
        lambda *_a, **_kw: None,
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.qdrant_client",
        MagicMock(return_value=mock_client),
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.embed_text",
        AsyncMock(return_value=np.zeros(1536).tolist()),
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.embed_batch",
        AsyncMock(return_value=[np.zeros(1536).tolist()]),
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.check_for_conflicts",
        AsyncMock(return_value=MagicMock(has_conflict=False)),
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.llm_adjudicated_dedup",
        AsyncMock(return_value=None),
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.ensure_collection",
        AsyncMock(return_value="memories"),
    )
    monkeypatch.setattr(
        "archivist.app.handlers.tools_storage.register_memory_points_batch",
        AsyncMock(return_value=None),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.system
async def test_archivist_store_handles_duplicate_entity_gracefully(qa_pool, _mock_qdrant_and_embed):
    """archivist_store called twice with the same explicit entity returns success both times.

    Regression test: before the ON CONFLICT DO UPDATE fix, the second call
    would raise sqlite3.IntegrityError and the handler would return an error
    response, causing agents to retry indefinitely.
    """
    from archivist.app.handlers.tools_storage import _handle_store

    args_a = {
        "text": "brian manages the Kubernetes cluster",
        "agent_id": "agent-alpha",
        "entities": ["brian"],
        "namespace": "test-idempotency",
    }
    args_b = {
        "text": "brian deployed ArgoCD in the cluster",
        "agent_id": "agent-beta",
        "entities": ["brian"],
        "namespace": "test-idempotency",
    }

    r1 = await _handle_store(args_a)
    data1 = _parse_response(r1)
    _assert_no_error(data1, "first store")

    r2 = await _handle_store(args_b)
    data2 = _parse_response(r2)
    _assert_no_error(data2, "second store (duplicate entity)")


@pytest.mark.system
async def test_archivist_store_concurrent_same_entity_all_succeed(qa_pool, _mock_qdrant_and_embed):
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
                "namespace": "test-concurrent-store",
            }
        )

    results = await asyncio.gather(*[store(i) for i in range(5)], return_exceptions=True)

    exceptions = [r for r in results if isinstance(r, Exception)]
    assert not exceptions, f"Concurrent store calls raised: {exceptions}"

    for idx, result in enumerate(results):
        data = _parse_response(result)  # type: ignore[arg-type]
        _assert_no_error(data, f"concurrent store #{idx}")


@pytest.mark.system
async def test_archivist_store_no_entity_still_succeeds(qa_pool, _mock_qdrant_and_embed):
    """archivist_store without explicit entities (auto-extract path) must succeed."""
    from archivist.app.handlers.tools_storage import _handle_store

    result = await _handle_store(
        {
            "text": "The Grafana dashboard shows CPU spikes on node-3",
            "agent_id": "monitor-agent",
            "namespace": "test-autoextract",
        }
    )
    data = _parse_response(result)
    _assert_no_error(data, "no explicit entities")
