"""INIT-005/SPEC-005 — embed-deferred store ack path.

Covers:
- Defer on → store does not await primary embed_text
- Defer off → primary embed_text still called
- Outbox payload carries embed_deferred + embed_inputs
- Dead-Qdrant ack still fast (INIT-003 spirit / SM-003)
"""

from __future__ import annotations

import json
import time
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit]


@contextmanager
def _store_patches(*, embed_return=None):
    """Minimal patches for store without real Qdrant/embed/LLM."""
    if embed_return is None:
        embed_return = [0.1] * 1024
    mock_client = MagicMock()
    mock_client.upsert = MagicMock()
    mock_client.query_points = MagicMock(return_value=MagicMock(points=[]))
    stack = ExitStack()
    try:
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=embed_return,
            )
        )
        stack.enter_context(
            patch(
                "archivist.write.conflict_detection.embed_text",
                new_callable=AsyncMock,
                return_value=embed_return,
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.embed_batch",
                new_callable=AsyncMock,
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.llm_adjudicated_dedup",
                new_callable=AsyncMock,
                return_value=None,
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage._query_similar",
                new_callable=AsyncMock,
                return_value=(embed_return, []),
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.check_for_conflicts",
                new_callable=AsyncMock,
                return_value=MagicMock(
                    has_conflict=False,
                    max_similarity=0.0,
                    conflicting_ids=[],
                    recommendation="",
                ),
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage._extract_and_store_entities",
                new_callable=AsyncMock,
                return_value={},
            )
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_storage.qdrant_client", return_value=mock_client)
        )
        stack.enter_context(
            patch("archivist.write.conflict_detection.qdrant_client", return_value=mock_client)
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.ensure_collection",
                return_value="archivist_memories",
            )
        )
        stack.enter_context(patch("archivist.core.audit.log_memory_event", new_callable=AsyncMock))
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage._extract_needle_micro_chunks",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.get_namespace_for_agent",
                return_value="coach-ns",
            )
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_storage.get_namespace_config", return_value=None)
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_storage.require_rbac", return_value=None)
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_storage.pre_extract", return_value={})
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.extract_needle_entities",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.conflict_vec_for_primary_embed",
                return_value=None,
            )
        )
        yield mock_client
    finally:
        stack.close()


def _configure_store_flags(monkeypatch, *, embed_defer: bool) -> None:
    monkeypatch.setattr("archivist.core.config.OUTBOX_ENABLED", True)
    monkeypatch.setattr("archivist.core.config.ARCHIVIST_EMBED_DEFER", embed_defer)
    monkeypatch.setattr("archivist.core.config.CONFLICT_CHECK_ON_STORE", False)
    monkeypatch.setattr("archivist.app.handlers.tools_storage.CONFLICT_CHECK_ON_STORE", False)
    monkeypatch.setattr("archivist.core.config.CONFLICT_BLOCK_ON_STORE", False)
    monkeypatch.setattr("archivist.app.handlers.tools_storage.CONFLICT_BLOCK_ON_STORE", False)
    monkeypatch.setattr("archivist.core.config.REVERSE_HYDE_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.BM25_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.CONTEXTUAL_AUGMENTATION_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.TOPIC_ROUTING_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.DEDUP_LLM_ENABLED", False)


_STORE_ARGS = {
    "text": "Coach note: sleep debt rises after late meals",
    "agent_id": "coach-agent",
    "namespace": "coach-ns",
    "entities": ["SleepDebtEntity"],
    "actor_id": "coach-agent",
    "actor_type": "agent",
}


class TestEmbedDeferredStore:
    async def test_defer_on_skips_primary_embed_text(self, async_pool, monkeypatch):
        """ac-1: ARCHIVIST_EMBED_DEFER=true → ack does not await primary embed."""
        _configure_store_flags(monkeypatch, embed_defer=True)

        with _store_patches() as _client:
            import archivist.app.handlers.tools_storage as ts

            embed_mock = ts.embed_text
            result = await ts._handle_store(dict(_STORE_ARGS))

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        assert data.get("embed_deferred") is True
        embed_mock.assert_not_awaited()
        assert data["stage_timings"]["embed_ms"] >= 0

        async with async_pool.read() as conn:
            cur = await conn.execute(
                "SELECT payload FROM outbox WHERE status='pending' ORDER BY created_at"
            )
            row = await cur.fetchone()
            assert row is not None
            payload = json.loads(row["payload"] if isinstance(row, dict) else row[0])
            assert payload.get("embed_deferred") is True
            assert data["memory_id"] in (payload.get("embed_inputs") or {})
            # Vectors empty on deferred row until drain fills.
            assert payload["points"][0]["vector"] == []

    async def test_defer_off_calls_primary_embed_text(self, async_pool, monkeypatch):
        """ac-5: default / defer off keeps sync primary embed."""
        _configure_store_flags(monkeypatch, embed_defer=False)

        with _store_patches():
            import archivist.app.handlers.tools_storage as ts

            embed_mock = ts.embed_text
            result = await ts._handle_store(dict(_STORE_ARGS))

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        assert data.get("embed_deferred") is False
        embed_mock.assert_awaited()

        async with async_pool.read() as conn:
            cur = await conn.execute(
                "SELECT payload FROM outbox WHERE status='pending' ORDER BY created_at"
            )
            row = await cur.fetchone()
            assert row is not None
            payload = json.loads(row["payload"] if isinstance(row, dict) else row[0])
            assert payload.get("embed_deferred") is not True
            assert len(payload["points"][0]["vector"]) > 0

    async def test_dead_qdrant_ack_still_fast_with_defer(self, async_pool, monkeypatch):
        """ac-4 / SM-003: dead Qdrant must not stall store ack (INIT-003 spirit)."""
        _configure_store_flags(monkeypatch, embed_defer=True)

        dead = MagicMock()
        dead.upsert = MagicMock(side_effect=RuntimeError("qdrant unreachable"))
        dead.query_points = MagicMock(side_effect=RuntimeError("qdrant unreachable"))

        t0 = time.monotonic()
        with _store_patches() as mock_client:
            mock_client.upsert = dead.upsert
            mock_client.query_points = dead.query_points
            import archivist.app.handlers.tools_storage as ts

            result = await ts._handle_store(dict(_STORE_ARGS))
        elapsed_ms = (time.monotonic() - t0) * 1000

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        assert data.get("embed_deferred") is True
        # Soft INIT-003 spirit: well under client ~30s hang and STORE_ACK_BUDGET_MS*3.
        assert elapsed_ms < 4000, f"ack too slow under dead Qdrant: {elapsed_ms:.1f}ms"
        assert data["duration_ms"] < 4000

        async with async_pool.read() as conn:
            cur = await conn.execute("SELECT COUNT(*) AS c FROM outbox WHERE status='pending'")
            row = await cur.fetchone()
            assert int(row["c"] if isinstance(row, dict) else row[0]) >= 1
