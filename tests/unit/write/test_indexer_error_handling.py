"""Regression tests for INIT-022/SPEC-001's indexer error-handling fixes.

  - C1 (Critical): ``_persist_points`` (the shared helper introduced by the
    M13 dedup) must re-raise on a ``MemoryTransaction`` failure instead of
    swallowing it -- previously ``index_file()`` reported success (returned
    ``len(points)``) even when nothing was actually persisted.
  - H8 (High): ``delete_file_points`` must log at ``warning`` level (not
    silently swallow) when the Qdrant delete-by-filter call fails, on both
    the outbox and legacy code paths.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.storage]


def _mock_txn_ctx(*, raise_on_executemany: Exception | None = None):
    """Return a MemoryTransaction async-context-manager mock.

    When ``raise_on_executemany`` is set, the transaction's ``executemany``
    call raises it -- simulating a real commit/write failure inside the
    transaction body.
    """
    txn = MagicMock()
    if raise_on_executemany is not None:
        txn.executemany = AsyncMock(side_effect=raise_on_executemany)
    else:
        txn.executemany = AsyncMock()
    txn.enqueue_qdrant_upsert = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=txn)
    txn.__aexit__ = AsyncMock(return_value=False)
    cm = MagicMock()
    cm.return_value = txn
    return cm


class TestPersistPointsReraisesOnFailure:
    """C1: a transaction failure must propagate, not be swallowed as success."""

    async def test_force_transaction_reraises(self):
        from archivist.write.indexer import _persist_points

        boom = RuntimeError("simulated commit failure")
        with (
            patch(
                "archivist.storage.transaction.MemoryTransaction",
                _mock_txn_ctx(raise_on_executemany=boom),
            ),
            patch("archivist.core.config.OUTBOX_ENABLED", False),
            pytest.raises(RuntimeError, match="simulated commit failure"),
        ):
            await _persist_points(
                "test-collection",
                points=[],
                mp_records=[],
                memory_id="mem-1",
                force_transaction=True,
            )

    async def test_outbox_enabled_reraises(self):
        from archivist.write.indexer import _persist_points

        boom = RuntimeError("simulated commit failure")
        with (
            patch(
                "archivist.storage.transaction.MemoryTransaction",
                _mock_txn_ctx(raise_on_executemany=boom),
            ),
            patch("archivist.core.config.OUTBOX_ENABLED", True),
            pytest.raises(RuntimeError, match="simulated commit failure"),
        ):
            await _persist_points(
                "test-collection",
                points=[],
                mp_records=[],
                memory_id="mem-1",
            )

    async def test_legacy_register_failure_reraises(self):
        """Non-outbox path: register_memory_points_batch failure must also propagate."""
        from archivist.write.indexer import _persist_points

        boom = RuntimeError("simulated register failure")
        with (
            patch("archivist.core.config.OUTBOX_ENABLED", False),
            patch("archivist.write.indexer.qdrant_client", return_value=MagicMock()),
            patch(
                "archivist.write.indexer.register_memory_points_batch",
                AsyncMock(side_effect=boom),
            ),
            pytest.raises(RuntimeError, match="simulated register failure"),
        ):
            await _persist_points(
                "test-collection",
                points=[],
                mp_records=[],
                memory_id="mem-1",
            )

    async def test_success_path_does_not_raise(self):
        """Sanity check: the happy path still completes without raising."""
        from archivist.write.indexer import _persist_points

        with (
            patch("archivist.storage.transaction.MemoryTransaction", _mock_txn_ctx()),
            patch("archivist.core.config.OUTBOX_ENABLED", True),
        ):
            await _persist_points(
                "test-collection",
                points=[],
                mp_records=[],
                memory_id="mem-1",
            )


class TestDeleteFilePointsLogsFailures:
    """H8: a Qdrant delete-by-filter failure must be logged, not silently discarded."""

    async def test_legacy_path_logs_warning_on_failure(self, caplog):
        from archivist.write import indexer as indexer_module

        boom = RuntimeError("simulated qdrant delete failure")
        mock_client = MagicMock()
        mock_client.delete.side_effect = boom

        with (
            patch("archivist.write.indexer.qdrant_client", return_value=mock_client),
            patch("archivist.core.config.OUTBOX_ENABLED", False),
            patch(
                "archivist.write.indexer.collections_for_query",
                return_value=["test-collection"],
            ),
            patch("archivist.write.indexer.BM25_ENABLED", False),
            caplog.at_level("WARNING", logger="archivist.indexer"),
        ):
            await indexer_module.delete_file_points("some/file.md")

        assert any(
            "Qdrant delete-by-filter failed" in r.message and r.levelname == "WARNING"
            for r in caplog.records
        )

    async def test_outbox_path_logs_warning_on_failure(self, caplog):
        from archivist.write import indexer as indexer_module

        boom = RuntimeError("simulated qdrant delete failure")

        with (
            patch("archivist.core.config.OUTBOX_ENABLED", True),
            patch(
                "archivist.storage.transaction.MemoryTransaction",
                _mock_txn_ctx(raise_on_executemany=None),
            ) as mock_txn_cls,
            patch(
                "archivist.write.indexer.collections_for_query",
                return_value=["test-collection"],
            ),
            patch("archivist.write.indexer.BM25_ENABLED", False),
            caplog.at_level("WARNING", logger="archivist.indexer"),
        ):
            mock_txn_cls.return_value.__aenter__.side_effect = boom
            from archivist.write import indexer as indexer_module

            await indexer_module.delete_file_points("some/file.md")

        assert any(
            "Qdrant delete-by-filter failed" in r.message and r.levelname == "WARNING"
            for r in caplog.records
        )
