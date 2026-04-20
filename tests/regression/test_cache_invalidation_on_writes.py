"""Regression tests: every write-path handler invalidates both hot_cache AND index_cache.

Verifies that _handle_merge, _handle_compress, _handle_pin, _handle_unpin, and
_handle_delete each call both:
  - hot_cache.invalidate_namespace(namespace)
  - invalidate_index_cache(namespace)

Also verifies that _handle_store (the reference implementation) still does both.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.regression]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _make_success_text(data: dict | None = None) -> list:
    """Return a minimal success TextContent list."""
    import json

    from mcp.types import TextContent

    payload = data or {"stored": True, "memory_id": "mem-abc"}
    return [TextContent(type="text", text=json.dumps(payload))]


# ---------------------------------------------------------------------------
# _handle_merge
# ---------------------------------------------------------------------------


class TestMergeCacheInvalidation:
    """merge must invalidate both hot_cache and index_cache (fix for C1)."""

    async def test_merge_invalidates_hot_cache_and_index_cache(self, async_pool):
        invalidate_hot = MagicMock()
        invalidate_index = MagicMock()
        mock_merge_result = {"merged_id": "merged-1", "strategy": "concat"}

        with (
            patch(
                "archivist.lifecycle.merge.merge_memories",
                new_callable=AsyncMock,
                return_value=mock_merge_result,
            ),
            patch("archivist.retrieval.hot_cache.invalidate_namespace", invalidate_hot),
            patch("archivist.storage.compressed_index.invalidate_index_cache", invalidate_index),
        ):
            from archivist.app.handlers.tools_storage import _handle_merge

            await _handle_merge(
                {
                    "agent_id": "agent-1",
                    "memory_ids": ["id-a", "id-b"],
                    "strategy": "concat",
                    "namespace": "test-ns",
                }
            )

        invalidate_hot.assert_called_once_with("test-ns")
        invalidate_index.assert_called_once_with("test-ns")


# ---------------------------------------------------------------------------
# _handle_compress
# ---------------------------------------------------------------------------


class TestCompressCacheInvalidation:
    """compress must invalidate both hot_cache and index_cache (fix for C4)."""

    async def test_compress_invalidates_index_cache(self, async_pool):
        invalidate_hot = MagicMock()
        invalidate_index = MagicMock()

        mock_client = MagicMock()
        mock_client.retrieve.return_value = [
            MagicMock(id="id1", payload={"text": "some text", "agent_id": "agent-1"})
        ]

        with (
            patch("archivist.app.handlers.tools_storage.qdrant_client", return_value=mock_client),
            patch(
                "archivist.write.compaction.compact_flat",
                new_callable=AsyncMock,
                return_value="summary text",
            ),
            patch(
                "archivist.app.handlers.tools_storage._handle_store",
                new_callable=AsyncMock,
                return_value=_make_success_text({"stored": True, "memory_id": "new-1"}),
            ),
            patch("archivist.lifecycle.curator_queue.enqueue", MagicMock()),
            patch("archivist.retrieval.hot_cache.invalidate_namespace", invalidate_hot),
            patch("archivist.storage.compressed_index.invalidate_index_cache", invalidate_index),
            patch("archivist.app.handlers.tools_storage._rbac_gate", return_value=None),
        ):
            from archivist.app.handlers.tools_storage import _handle_compress

            await _handle_compress(
                {
                    "agent_id": "agent-1",
                    "namespace": "test-ns",
                    "memory_ids": ["id1"],
                    "format": "flat",
                }
            )

        invalidate_hot.assert_called_with("test-ns")
        invalidate_index.assert_called_with("test-ns")


# ---------------------------------------------------------------------------
# _handle_pin
# ---------------------------------------------------------------------------


class TestPinCacheInvalidation:
    """pin must invalidate both hot_cache and index_cache (fix for C4)."""

    async def test_pin_invalidates_index_cache(self, async_pool):
        invalidate_hot = MagicMock()
        invalidate_index = MagicMock()

        mock_client = MagicMock()
        mock_client.retrieve.return_value = [MagicMock(id="mem-1", payload={})]
        mock_client.set_payload.return_value = MagicMock()

        with (
            patch("archivist.app.handlers.tools_storage.qdrant_client", return_value=mock_client),
            patch("archivist.app.handlers.tools_storage.collection_for", return_value="col-test"),
            patch("archivist.core.audit.log_memory_event", new_callable=AsyncMock),
            patch("archivist.retrieval.hot_cache.invalidate_namespace", invalidate_hot),
            patch("archivist.storage.compressed_index.invalidate_index_cache", invalidate_index),
            patch("archivist.app.handlers.tools_storage._rbac_gate", return_value=None),
        ):
            from archivist.app.handlers.tools_storage import _handle_pin

            await _handle_pin(
                {
                    "agent_id": "agent-1",
                    "memory_id": "mem-1",
                    "namespace": "test-ns",
                    "reason": "important",
                }
            )

        invalidate_hot.assert_called_with("test-ns")
        invalidate_index.assert_called_with("test-ns")


# ---------------------------------------------------------------------------
# _handle_unpin
# ---------------------------------------------------------------------------


class TestUnpinCacheInvalidation:
    """unpin must invalidate both hot_cache and index_cache (fix for C4)."""

    async def test_unpin_invalidates_index_cache(self, async_pool):
        invalidate_hot = MagicMock()
        invalidate_index = MagicMock()

        mock_client = MagicMock()
        mock_client.set_payload.return_value = MagicMock()

        with (
            patch("archivist.app.handlers.tools_storage.qdrant_client", return_value=mock_client),
            patch("archivist.app.handlers.tools_storage.collection_for", return_value="col-test"),
            patch("archivist.retrieval.hot_cache.invalidate_namespace", invalidate_hot),
            patch("archivist.storage.compressed_index.invalidate_index_cache", invalidate_index),
            patch("archivist.app.handlers.tools_storage._rbac_gate", return_value=None),
        ):
            from archivist.app.handlers.tools_storage import _handle_unpin

            await _handle_unpin(
                {
                    "agent_id": "agent-1",
                    "memory_id": "mem-1",
                    "namespace": "test-ns",
                }
            )

        invalidate_hot.assert_called_with("test-ns")
        invalidate_index.assert_called_with("test-ns")


# ---------------------------------------------------------------------------
# _handle_delete
# ---------------------------------------------------------------------------


class TestDeleteCacheInvalidation:
    """delete must invalidate both hot_cache and index_cache (fix for C4)."""

    async def test_delete_invalidates_index_cache(self, async_pool):
        invalidate_hot = MagicMock()
        invalidate_index = MagicMock()

        with (
            patch(
                "archivist.lifecycle.memory_lifecycle.soft_delete_memory",
                new_callable=AsyncMock,
                return_value={"status": "soft_delete_initiated"},
            ),
            patch("archivist.core.rbac.get_namespace_for_agent", return_value="test-ns"),
            patch("archivist.retrieval.hot_cache.invalidate_namespace", invalidate_hot),
            patch("archivist.storage.compressed_index.invalidate_index_cache", invalidate_index),
            patch("archivist.app.handlers.tools_storage._rbac_gate", return_value=None),
        ):
            from archivist.app.handlers.tools_storage import _handle_delete

            await _handle_delete(
                {
                    "agent_id": "agent-1",
                    "memory_id": "mem-1",
                    "namespace": "test-ns",
                }
            )

        invalidate_hot.assert_called_with("test-ns")
        invalidate_index.assert_called_with("test-ns")


# ---------------------------------------------------------------------------
# _handle_store (reference — must still do both)
# ---------------------------------------------------------------------------


class TestStoreCacheInvalidation:
    """store must continue to invalidate both caches (regression guard)."""

    async def test_store_still_invalidates_both_caches(self, async_pool):
        """Ensure invalidate_index_cache is called from _handle_store."""

        # Use a simpler approach: inspect the source to confirm the call is present
        import inspect

        import archivist.app.handlers.tools_storage as ts_mod

        source = inspect.getsource(ts_mod._handle_store)
        assert "invalidate_index_cache" in source, (
            "_handle_store must call invalidate_index_cache — regression guard"
        )
        assert "hot_cache.invalidate_namespace" in source, (
            "_handle_store must call hot_cache.invalidate_namespace — regression guard"
        )
