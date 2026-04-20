"""System tests for handler-level abort regression in archivist_store.

These tests specifically target the bug where ``_handle_store``'s
``except sqlite3.IntegrityError`` block called ``return conflict_resolved_response(...)``
on the *first* entity conflict, aborting the entire store pipeline (embedding,
Qdrant, FTS, audit) for ALL subsequent operations in that call — and all remaining
entities in the same batch.

The existing test_entity_idempotency.py tests confirm the happy path and basic
concurrency.  These tests push harder: 1, 5, and 20 concurrent agents storing
the same entity in the same namespace — the production fleet migration load.

Run with::

    pytest tests/system/mcp/test_entity_races.py -v -m system
"""

from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock, MagicMock

import pytest

pytestmark = [pytest.mark.system, pytest.mark.mcp]

_MOCK_VEC = [0.0] * 1536


def _parse_response(result) -> dict:
    assert isinstance(result, list) and len(result) > 0
    return json.loads(result[0].text)


def _assert_stored(data: dict, label: str = "") -> None:
    """Assert that the response indicates a successful store (not a conflict abort)."""
    ctx = f"[{label}] " if label else ""
    # conflict_resolved_response has status="conflict_resolved" and no memory_id.
    # That is the regression: the old code returned this from inside the entity loop,
    # which means the memory was never actually stored.
    assert data.get("status") != "conflict_resolved", (
        f"{ctx}BUG REGRESSION: _handle_store returned conflict_resolved_response, "
        "which means it aborted the entire store pipeline on the first entity "
        "conflict.  archivist_store must be idempotent — not aborting."
    )
    assert "error" not in data, f"{ctx}Handler returned unexpected error: {data}"


@pytest.fixture
def _mock_externals(monkeypatch):
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
# Parametrized concurrent store test
# ---------------------------------------------------------------------------


@pytest.mark.system
@pytest.mark.parametrize("concurrency", [1, 5, 20])
async def test_concurrent_store_same_entity_no_handler_abort(
    qa_pool, _mock_externals, concurrency
):
    """N concurrent archivist_store calls with the same entity must all succeed.

    Regression guard for the handler-level abort bug:
    - Old behaviour: the FIRST entity conflict caused ``return conflict_resolved_response(...)``
      inside ``_handle_store``, aborting the pipeline entirely.  Under concurrency,
      at least one agent would always hit this path.
    - Expected behaviour: all N concurrent stores return a success response.

    Parametrized over [1, 5, 20] concurrent agents to cover single-agent,
    small-fleet, and large-fleet migration scenarios.
    """
    from archivist.app.handlers.tools_storage import _handle_store

    async def store(agent_idx: int):
        return await _handle_store(
            {
                "text": f"agent-{agent_idx} observed the Kubernetes cluster health",
                "agent_id": f"fleet-agent-{agent_idx:03d}",
                "entities": ["kubernetes", "prometheus"],
                "namespace": f"concurrent-race-ns-{concurrency}",
            }
        )

    results = await asyncio.gather(
        *[store(i) for i in range(concurrency)], return_exceptions=True
    )

    exceptions = [r for r in results if isinstance(r, Exception)]
    assert not exceptions, (
        f"[concurrency={concurrency}] {len(exceptions)} concurrent store calls "
        f"raised exceptions: {exceptions}"
    )

    for idx, result in enumerate(results):
        data = _parse_response(result)  # type: ignore[arg-type]
        _assert_stored(data, f"concurrency={concurrency} agent={idx}")


# ---------------------------------------------------------------------------
# Multi-entity batch: IntegrityError on entity[0] must not skip entity[1+]
# ---------------------------------------------------------------------------


@pytest.mark.system
async def test_store_continues_after_first_entity_conflict(qa_pool, _mock_externals):
    """Storing the same multi-entity batch twice must not silently drop the second store.

    Regression guard: with the old ``return`` inside the entity loop, a batch of
    [entity-A, entity-B] where entity-A was already in the DB would:
      1. Hit the IntegrityError on entity-A
      2. Return conflict_resolved_response immediately
      3. Never process entity-B, never write the embedding, never touch Qdrant

    The fixed code continues the loop and processes all entities.
    """
    from archivist.app.handlers.tools_storage import _handle_store

    batch = {
        "text": "Multi-entity migration batch: argocd manages kubernetes deployments",
        "agent_id": "migration-orchestrator",
        "entities": ["argocd", "kubernetes", "helm"],
        "namespace": "multi-entity-ns",
    }

    r1 = await _handle_store(batch)
    data1 = _parse_response(r1)
    _assert_stored(data1, "first batch store")

    # Second store — all three entities now already exist in the DB.
    # This is the failing case under the old code.
    r2 = await _handle_store(
        {
            **batch,
            "text": "Migration retry: same entities, different memory text",
        }
    )
    data2 = _parse_response(r2)
    _assert_stored(data2, "second batch store (all entities pre-existing)")
