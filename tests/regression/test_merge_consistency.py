import pytest

"""Regression tests for P2: merge.py data integrity fixes.

Verifies that after merge_memories():
  1. The merged point is written to the correct Qdrant collection (collection_for(ns),
     not the hardcoded QDRANT_COLLECTION constant).
  2. FTS (memory_chunks), needle_registry, and memory_points rows exist for merged_id.
  3. record_version is awaited and produces a version row.
  4. Partial deletion errors in the originals loop are caught without crashing merge.
"""

from unittest.mock import AsyncMock, MagicMock, patch

pytestmark = [pytest.mark.regression, pytest.mark.lifecycle]

def _make_mock_point(pid: str, namespace: str = "test-ns") -> MagicMock:
    pt = MagicMock()
    pt.id = pid
    pt.payload = {
        "text": f"fact about {pid}",
        "date": "2025-01-01",
        "team": "engineering",
        "namespace": namespace,
        "version": 1,
        "importance_score": 0.7,
        "consistency_level": "eventual",
    }
    return pt

def _mock_txn_ctx():
    """Return a no-op MemoryTransaction async context manager mock."""
    txn = MagicMock()
    txn.execute = AsyncMock()
    txn.executemany = AsyncMock()
    txn.upsert_fts_chunk = AsyncMock()
    txn.register_needle_tokens = AsyncMock()
    txn.enqueue_qdrant_upsert = MagicMock()
    txn.enqueue_qdrant_delete = MagicMock()
    txn.__aenter__ = AsyncMock(return_value=txn)
    txn.__aexit__ = AsyncMock(return_value=False)
    cm = MagicMock()
    cm.return_value = txn
    return cm

class TestMergeCollectionRouting:
    """merge_memories routes to collection_for(ns), not QDRANT_COLLECTION."""

    async def test_upsert_uses_collection_for_ns(self, async_pool):
        """Qdrant upsert must use collection_for(ns), not the global constant."""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = [
            _make_mock_point("id1"),
            _make_mock_point("id2"),
        ]

        upsert_calls = []

        def capture_upsert(**kwargs):
            upsert_calls.append(kwargs.get("collection_name"))
            return MagicMock(operation_id=1)

        mock_client.upsert.side_effect = capture_upsert

        with (
            patch("merge.qdrant_client", return_value=mock_client),
            patch("merge.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024),
            patch("merge.record_version", new_callable=AsyncMock, return_value=2),
            patch("archivist.storage.transaction.MemoryTransaction", _mock_txn_ctx()),
            patch("merge.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.delete_memory_complete", new_callable=AsyncMock),
            patch(
                "merge.collection_for",
                return_value="archivist_test_ns",
            ) as mock_coll_for,
        ):
            from merge import merge_memories

            await merge_memories(["id1", "id2"], "latest", "agent1", "test-ns")

        assert upsert_calls, "Qdrant upsert was never called"
        assert upsert_calls[0] == "archivist_test_ns", (
            f"upsert used wrong collection: {upsert_calls[0]!r} — "
            "should be collection_for(ns), not the hardcoded constant"
        )

class TestMergeSQLiteArtifacts:
    """After merge, FTS, needle, and memory_points rows exist for merged_id."""

    async def test_register_memory_points_called_for_merged_id(self, async_pool):
        """memory_points row must be written for merged_id via MemoryTransaction."""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = [
            _make_mock_point("src1"),
            _make_mock_point("src2"),
        ]

        txn_mock = _mock_txn_ctx()

        with (
            patch("merge.qdrant_client", return_value=mock_client),
            patch("merge.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024),
            patch("merge.record_version", new_callable=AsyncMock, return_value=2),
            patch("archivist.storage.transaction.MemoryTransaction", txn_mock),
            patch("merge.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.delete_memory_complete", new_callable=AsyncMock),
        ):
            from merge import merge_memories

            result = await merge_memories(["src1", "src2"], "latest", "agent1", "test-ns")

        merged_id = result["merged_id"]
        # Verify memory_points INSERT was executed via the transaction
        txn_instance = txn_mock.return_value
        assert txn_instance.execute.called or txn_instance.executemany.called, (
            f"merged_id {merged_id!r} — MemoryTransaction.execute/executemany never called, "
            "the merged point is invisible to memory_points lookup"
        )

    async def test_upsert_fts_chunk_called_for_merged_id(self, async_pool):
        """upsert_fts_chunk must be called for the merged_id via txn shim."""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = [
            _make_mock_point("fts1"),
            _make_mock_point("fts2"),
        ]

        txn_mock = _mock_txn_ctx()

        with (
            patch("merge.qdrant_client", return_value=mock_client),
            patch("merge.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024),
            patch("merge.record_version", new_callable=AsyncMock, return_value=2),
            patch("archivist.storage.transaction.MemoryTransaction", txn_mock),
            patch("merge.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.delete_memory_complete", new_callable=AsyncMock),
        ):
            from merge import merge_memories

            result = await merge_memories(["fts1", "fts2"], "concat", "agent1", "test-ns")

        merged_id = result["merged_id"]
        txn_instance = txn_mock.return_value
        # upsert_fts_chunk is now called as a txn shim
        assert txn_instance.upsert_fts_chunk.called, (
            f"merged_id {merged_id!r} — txn.upsert_fts_chunk never called; "
            "the merged point is invisible to BM25 search"
        )
        call_kwargs = txn_instance.upsert_fts_chunk.call_args[1]
        assert call_kwargs.get("qdrant_id") == merged_id, (
            f"upsert_fts_chunk called with qdrant_id={call_kwargs.get('qdrant_id')!r}, "
            f"expected {merged_id!r}"
        )

    async def test_register_needle_tokens_called_for_merged_id(self, async_pool):
        """register_needle_tokens must be called for the merged_id via txn shim."""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = [
            _make_mock_point("nee1"),
            _make_mock_point("nee2"),
        ]

        txn_mock = _mock_txn_ctx()

        with (
            patch("merge.qdrant_client", return_value=mock_client),
            patch("merge.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024),
            patch("merge.record_version", new_callable=AsyncMock, return_value=2),
            patch("archivist.storage.transaction.MemoryTransaction", txn_mock),
            patch("merge.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.delete_memory_complete", new_callable=AsyncMock),
        ):
            from merge import merge_memories

            result = await merge_memories(["nee1", "nee2"], "latest", "agent1", "test-ns")

        merged_id = result["merged_id"]
        txn_instance = txn_mock.return_value
        assert txn_instance.register_needle_tokens.called, (
            f"merged_id {merged_id!r} — txn.register_needle_tokens never called; "
            "the merged point is invisible to needle lookup"
        )
        call_args = txn_instance.register_needle_tokens.call_args
        # First positional arg is memory_id
        called_mid = call_args[0][0] if call_args[0] else call_args[1].get("memory_id")
        assert called_mid == merged_id, (
            f"register_needle_tokens called with memory_id={called_mid!r}, expected {merged_id!r}"
        )

class TestMergeVersionTracking:
    """record_version is awaited and the version is returned."""

    async def test_record_version_is_awaited(self, async_pool):
        """record_version must be called as a coroutine (not dropped)."""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = [
            _make_mock_point("v1"),
            _make_mock_point("v2"),
        ]

        version_calls = []

        async def capture_version(memory_id, *args, **kwargs):
            version_calls.append(memory_id)
            return 42

        with (
            patch("merge.qdrant_client", return_value=mock_client),
            patch("merge.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024),
            patch("merge.record_version", side_effect=capture_version),
            patch("archivist.storage.transaction.MemoryTransaction", _mock_txn_ctx()),
            patch("merge.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.delete_memory_complete", new_callable=AsyncMock),
        ):
            from merge import merge_memories

            result = await merge_memories(["v1", "v2"], "latest", "agent1", "test-ns")

        assert version_calls, (
            "record_version was never called — version tracking was silently dropped"
        )
        assert result["version"] == 42, (
            f"result version={result['version']!r} — expected 42 from record_version"
        )

class TestMergePartialDeletionGuard:
    """Partial deletion errors in originals loop are caught, not propagated."""

    async def test_partial_deletion_error_does_not_abort_merge(self, async_pool):
        """merge_memories catches PartialDeletionError from delete loop, returns result."""
        from cascade import PartialDeletionError
        from memory_lifecycle import DeleteResult

        mock_client = MagicMock()
        mock_client.retrieve.return_value = [
            _make_mock_point("del1"),
            _make_mock_point("del2"),
        ]

        failed_result = DeleteResult(
            memory_id="del1",
            failed_steps=["qdrant_primary"],
        )

        async def partial_fail(memory_id, *args, **kwargs):
            if memory_id == "del1":
                raise PartialDeletionError(failed_result)

        with (
            patch("merge.qdrant_client", return_value=mock_client),
            patch("merge.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024),
            patch("merge.record_version", new_callable=AsyncMock, return_value=3),
            patch("archivist.storage.transaction.MemoryTransaction", _mock_txn_ctx()),
            patch("merge.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.delete_memory_complete", side_effect=partial_fail),
        ):
            from merge import merge_memories


            # Should NOT raise — partial failure is caught and logged
            result = await merge_memories(["del1", "del2"], "latest", "agent1", "test-ns")

        assert "merged_id" in result, (
            "merge_memories raised instead of handling PartialDeletionError gracefully"
        )
