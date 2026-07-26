"""INIT-003/SPEC-004 — fast durable store ack SLO under dead/slow Qdrant.

Covers:
- Outbox default-on: store succeeds when conflict query hangs/fails; outbox pending
- Fail-fast when graph pool is not initialized
- Ack path respects CONFLICT_QUERY_TIMEOUT_S (timeout mock + elapsed upper bound)
- OUTBOX_ENABLED remains True (not the latency "fix")
"""

from __future__ import annotations

import asyncio
import json
import time
from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.storage]


@contextmanager
def _store_patches(*, upsert_side_effect=None):
    """Common patches for a store call that does not need real Qdrant/embed."""
    mock_client = MagicMock()
    if upsert_side_effect is not None:
        mock_client.upsert.side_effect = upsert_side_effect
    else:
        mock_client.upsert = MagicMock()
    mock_client.query_points = MagicMock(return_value=MagicMock(points=[]))
    stack = ExitStack()
    try:
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
            )
        )
        stack.enter_context(
            patch(
                "archivist.write.conflict_detection.embed_text",
                new_callable=AsyncMock,
                return_value=[0.1] * 1024,
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
            patch("archivist.app.handlers.tools_storage.qdrant_client", return_value=mock_client)
        )
        stack.enter_context(
            patch("archivist.write.conflict_detection.qdrant_client", return_value=mock_client)
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.ensure_collection",
                return_value="test_col",
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
        yield mock_client
    finally:
        stack.close()


async def _count_outbox_pending(async_pool) -> int:
    async with async_pool.read() as conn:
        cur = await conn.execute("SELECT COUNT(*) AS c FROM outbox WHERE status='pending'")
        row = await cur.fetchone()
    if row is None:
        return 0
    return int(row["c"] if hasattr(row, "keys") else row[0])


class TestOutboxDefaultOn:
    def test_outbox_enabled_default_true(self):
        from archivist.core.config import OUTBOX_ENABLED

        assert OUTBOX_ENABLED is True


class TestConflictQueryTimeout:
    async def test_query_similar_fail_open_on_wait_for_timeout(self, monkeypatch):
        """Module-scoped wait_for TimeoutError → empty results; elapsed upper bound."""
        from archivist.write import conflict_detection as cd

        monkeypatch.setattr("archivist.core.config.CONFLICT_QUERY_TIMEOUT_S", 0.05)

        async def _fake_wait_for(awaitable, timeout=None):
            # Do not run the to_thread body (would sleep); fail-open immediately.
            if asyncio.iscoroutine(awaitable):
                awaitable.close()
            raise TimeoutError

        mock_client = MagicMock()
        mock_client.query_points.side_effect = lambda **_kw: time.sleep(30)

        t0 = time.monotonic()
        with (
            patch.object(cd, "embed_text", new_callable=AsyncMock, return_value=[0.1] * 8),
            patch.object(cd, "qdrant_client", return_value=mock_client),
            patch.object(cd, "collection_for", return_value="col"),
            patch.object(cd.asyncio, "wait_for", side_effect=_fake_wait_for),
        ):
            vec, results = await cd._query_similar("hello", "ns", timeout_s=0.05)
        elapsed = time.monotonic() - t0

        assert vec == [0.1] * 8
        assert results == []
        assert elapsed < 1.0

    async def test_query_similar_passes_timeout_to_wait_for(self, monkeypatch):
        from archivist.write import conflict_detection as cd

        seen: dict = {}

        async def _capture_wait_for(awaitable, timeout=None):
            seen["timeout"] = timeout
            if asyncio.iscoroutine(awaitable):
                awaitable.close()
            return []

        mock_client = MagicMock()
        with (
            patch.object(cd, "embed_text", new_callable=AsyncMock, return_value=[0.1] * 4),
            patch.object(cd, "qdrant_client", return_value=mock_client),
            patch.object(cd, "collection_for", return_value="col"),
            patch.object(cd.asyncio, "wait_for", side_effect=_capture_wait_for),
        ):
            await cd._query_similar("hello", "ns", timeout_s=0.37)

        assert seen["timeout"] == 0.37

    async def test_query_similar_fail_open_on_qdrant_error(self):
        from archivist.write import conflict_detection as cd

        mock_client = MagicMock()

        def _boom(**_kw):
            raise ConnectionError("Qdrant down")

        mock_client.query_points.side_effect = _boom

        with (
            patch.object(cd, "embed_text", new_callable=AsyncMock, return_value=[0.2] * 4),
            patch.object(cd, "qdrant_client", return_value=mock_client),
            patch.object(cd, "collection_for", return_value="col"),
        ):
            _, results = await cd._query_similar("hello", "ns", timeout_s=0.2)

        assert results == []


class TestStoreAckUnderDeadQdrant:
    async def test_store_succeeds_when_conflict_query_times_out(self, async_pool, monkeypatch):
        """With outbox on + conflict check on, budgeted similarity timeout must not block ack."""
        monkeypatch.setattr("archivist.core.config.OUTBOX_ENABLED", True)
        monkeypatch.setattr("archivist.core.config.CONFLICT_CHECK_ON_STORE", True)
        monkeypatch.setattr("archivist.core.config.CONFLICT_BLOCK_ON_STORE", False)
        monkeypatch.setattr("archivist.core.config.CONFLICT_QUERY_TIMEOUT_S", 0.05)
        # Hoisted imports in tools_storage — patch the bound names too.
        monkeypatch.setattr(
            "archivist.app.handlers.tools_storage.CONFLICT_CHECK_ON_STORE", True
        )
        monkeypatch.setattr(
            "archivist.app.handlers.tools_storage.CONFLICT_BLOCK_ON_STORE", False
        )
        monkeypatch.setattr("archivist.core.config.REVERSE_HYDE_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.BM25_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.CONTEXTUAL_AUGMENTATION_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.TOPIC_ROUTING_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.DEDUP_LLM_ENABLED", False)

        async def _fake_wait_for(awaitable, timeout=None):
            if asyncio.iscoroutine(awaitable):
                awaitable.close()
            raise TimeoutError

        pending_before = await _count_outbox_pending(async_pool)

        t0 = time.monotonic()
        with (
            _store_patches(
                upsert_side_effect=AssertionError("inline upsert must not run with outbox on")
            ) as mock_client,
            patch(
                "archivist.write.conflict_detection.asyncio.wait_for",
                side_effect=_fake_wait_for,
            ),
        ):
            from archivist.app.handlers.tools_storage import _handle_store

            result = await _handle_store(
                {
                    "text": "coach insight about sleep debt",
                    "agent_id": "coach-agent",
                    "namespace": "coach-ns",
                    "entities": ["sleep"],
                }
            )
        elapsed = time.monotonic() - t0

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        assert "memory_id" in data
        # Timeout-mock path: must stay well under coach ~4s / 15s stall thresholds.
        assert elapsed < 2.0, f"store ack took {elapsed:.2f}s under timeout mock"
        assert mock_client.upsert.call_count == 0

        pending_after = await _count_outbox_pending(async_pool)
        assert pending_after > pending_before, "expected pending outbox row after durable store"

    async def test_store_succeeds_when_conflict_query_fails(self, async_pool, monkeypatch):
        monkeypatch.setattr("archivist.core.config.OUTBOX_ENABLED", True)
        monkeypatch.setattr("archivist.core.config.CONFLICT_CHECK_ON_STORE", True)
        monkeypatch.setattr(
            "archivist.app.handlers.tools_storage.CONFLICT_CHECK_ON_STORE", True
        )
        monkeypatch.setattr("archivist.core.config.REVERSE_HYDE_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.BM25_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.CONTEXTUAL_AUGMENTATION_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.DEDUP_LLM_ENABLED", False)

        pending_before = await _count_outbox_pending(async_pool)
        with _store_patches() as mock_client:
            mock_client.query_points.side_effect = OSError("Qdrant unreachable")
            from archivist.app.handlers.tools_storage import _handle_store

            result = await _handle_store(
                {
                    "text": "another coach memory",
                    "agent_id": "coach-agent",
                    "namespace": "coach-ns",
                    "entities": ["focus"],
                }
            )

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        pending_after = await _count_outbox_pending(async_pool)
        assert pending_after > pending_before

    async def test_store_does_not_block_on_slow_qdrant_upsert(self, async_pool, monkeypatch):
        """Outbox path must not call client.upsert on the hot path (slow upsert irrelevant)."""
        monkeypatch.setattr("archivist.core.config.OUTBOX_ENABLED", True)
        monkeypatch.setattr("archivist.core.config.CONFLICT_CHECK_ON_STORE", False)
        monkeypatch.setattr("archivist.core.config.REVERSE_HYDE_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.BM25_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.CONTEXTUAL_AUGMENTATION_ENABLED", False)

        def _slow_upsert(*_a, **_k):
            time.sleep(30)
            raise AssertionError("upsert should not be awaited on outbox hot path")

        t0 = time.monotonic()
        with _store_patches(upsert_side_effect=_slow_upsert) as mock_client:
            from archivist.app.handlers.tools_storage import _handle_store

            result = await _handle_store(
                {
                    "text": "upsert hang must not matter",
                    "agent_id": "coach-agent",
                    "namespace": "coach-ns",
                    "entities": ["habit"],
                    "force_skip_conflict_check": True,
                }
            )
        elapsed = time.monotonic() - t0

        data = json.loads(result[0].text)
        assert data.get("stored") is True, data
        assert mock_client.upsert.call_count == 0
        assert elapsed < 2.0


class TestGraphPoolFailFast:
    async def test_store_fails_fast_when_pool_not_initialized(self, monkeypatch):
        monkeypatch.setattr("archivist.core.config.OUTBOX_ENABLED", True)
        monkeypatch.setattr("archivist.core.config.CONFLICT_CHECK_ON_STORE", False)
        monkeypatch.setattr("archivist.core.config.REVERSE_HYDE_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.BM25_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False)
        monkeypatch.setattr("archivist.core.config.CONTEXTUAL_AUGMENTATION_ENABLED", False)

        t0 = time.monotonic()
        with (
            _store_patches(),
            patch(
                "archivist.app.handlers.tools_storage.upsert_entity",
                new_callable=AsyncMock,
                side_effect=RuntimeError(
                    "SQLitePool is not initialized — call initialize_pool() first"
                ),
            ),
        ):
            from archivist.app.handlers.tools_storage import _handle_store

            result = await _handle_store(
                {
                    "text": "needs graph",
                    "agent_id": "coach-agent",
                    "namespace": "coach-ns",
                    "entities": ["x"],
                }
            )
        elapsed = time.monotonic() - t0

        data = json.loads(result[0].text)
        assert data.get("stored") is False
        assert data.get("error") == "graph_pool_unavailable"
        assert data.get("namespace") == "coach-ns"
        # Error payload must not invent other-namespace data.
        assert "other_namespace" not in data
        assert elapsed < 1.0
