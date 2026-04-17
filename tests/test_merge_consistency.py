"""Regression tests for P2: merge.py data integrity fixes.

Verifies that after merge_memories():
  1. The merged point is written to the correct Qdrant collection (collection_for(ns),
     not the hardcoded QDRANT_COLLECTION constant).
  2. FTS (memory_chunks), needle_registry, and memory_points rows exist for merged_id.
  3. record_version is awaited and produces a version row.
  4. Partial deletion errors in the originals loop are caught without crashing merge.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


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
            patch("merge.register_memory_points_batch", new_callable=AsyncMock),
            patch("merge.upsert_fts_chunk", new_callable=AsyncMock),
            patch("merge.register_needle_tokens", new_callable=AsyncMock),
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
        """register_memory_points_batch must be called with merged_id."""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = [
            _make_mock_point("src1"),
            _make_mock_point("src2"),
        ]

        captured_batch = []

        async def capture_register(batch):
            captured_batch.extend(batch)

        with (
            patch("merge.qdrant_client", return_value=mock_client),
            patch("merge.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024),
            patch("merge.record_version", new_callable=AsyncMock, return_value=2),
            patch("merge.register_memory_points_batch", side_effect=capture_register),
            patch("merge.upsert_fts_chunk", new_callable=AsyncMock),
            patch("merge.register_needle_tokens", new_callable=AsyncMock),
            patch("merge.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.delete_memory_complete", new_callable=AsyncMock),
        ):
            from merge import merge_memories

            result = await merge_memories(["src1", "src2"], "latest", "agent1", "test-ns")

        merged_id = result["merged_id"]
        ids_registered = [row["memory_id"] for row in captured_batch]
        assert merged_id in ids_registered, (
            f"merged_id {merged_id!r} not in register_memory_points_batch call — "
            "the merged point is invisible to memory_points lookup"
        )

    async def test_upsert_fts_chunk_called_for_merged_id(self, async_pool):
        """upsert_fts_chunk must be called for the merged_id."""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = [
            _make_mock_point("fts1"),
            _make_mock_point("fts2"),
        ]

        fts_calls = []

        async def capture_fts(**kwargs):
            fts_calls.append(kwargs.get("qdrant_id"))

        with (
            patch("merge.qdrant_client", return_value=mock_client),
            patch("merge.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024),
            patch("merge.record_version", new_callable=AsyncMock, return_value=2),
            patch("merge.register_memory_points_batch", new_callable=AsyncMock),
            patch("merge.upsert_fts_chunk", side_effect=capture_fts),
            patch("merge.register_needle_tokens", new_callable=AsyncMock),
            patch("merge.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.delete_memory_complete", new_callable=AsyncMock),
        ):
            from merge import merge_memories

            result = await merge_memories(["fts1", "fts2"], "concat", "agent1", "test-ns")

        merged_id = result["merged_id"]
        assert merged_id in fts_calls, (
            f"merged_id {merged_id!r} never passed to upsert_fts_chunk — "
            "the merged point is invisible to BM25 search"
        )

    async def test_register_needle_tokens_called_for_merged_id(self, async_pool):
        """register_needle_tokens must be called for the merged_id."""
        mock_client = MagicMock()
        mock_client.retrieve.return_value = [
            _make_mock_point("nee1"),
            _make_mock_point("nee2"),
        ]

        needle_calls = []

        async def capture_needle(memory_id, *args, **kwargs):
            needle_calls.append(memory_id)

        with (
            patch("merge.qdrant_client", return_value=mock_client),
            patch("merge.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024),
            patch("merge.record_version", new_callable=AsyncMock, return_value=2),
            patch("merge.register_memory_points_batch", new_callable=AsyncMock),
            patch("merge.upsert_fts_chunk", new_callable=AsyncMock),
            patch("merge.register_needle_tokens", side_effect=capture_needle),
            patch("merge.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.delete_memory_complete", new_callable=AsyncMock),
        ):
            from merge import merge_memories

            result = await merge_memories(["nee1", "nee2"], "latest", "agent1", "test-ns")

        merged_id = result["merged_id"]
        assert merged_id in needle_calls, (
            f"merged_id {merged_id!r} never passed to register_needle_tokens — "
            "the merged point is invisible to needle lookup"
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
            patch("merge.register_memory_points_batch", new_callable=AsyncMock),
            patch("merge.upsert_fts_chunk", new_callable=AsyncMock),
            patch("merge.register_needle_tokens", new_callable=AsyncMock),
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
            patch("merge.register_memory_points_batch", new_callable=AsyncMock),
            patch("merge.upsert_fts_chunk", new_callable=AsyncMock),
            patch("merge.register_needle_tokens", new_callable=AsyncMock),
            patch("merge.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.delete_memory_complete", side_effect=partial_fail),
        ):
            from merge import merge_memories

            # Should NOT raise — partial failure is caught and logged
            result = await merge_memories(["del1", "del2"], "latest", "agent1", "test-ns")

        assert "merged_id" in result, (
            "merge_memories raised instead of handling PartialDeletionError gracefully"
        )
