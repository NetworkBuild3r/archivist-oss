"""Tests for Chunk 2: Delete Cascade.

Covers:
- delete_memory_complete cleans up all 7 artifact types
- archive_memory_complete flags all related Qdrant points
- delete_fts_chunks_by_qdrant_id removes FTS entries correctly
- Qdrant failures don't prevent SQLite cleanup (partial cascade)
- curator_queue drain calls lifecycle functions
- merge.py uses delete_memory_complete per original ID
"""

import asyncio
from dataclasses import asdict
from unittest.mock import AsyncMock, MagicMock, patch, call

import pytest


class TestDeleteResult:
    """DeleteResult dataclass behaves correctly."""

    def test_total_sums_all_fields(self):
        from memory_lifecycle import DeleteResult

        r = DeleteResult(
            memory_id="abc",
            qdrant_primary=1,
            qdrant_reverse_hyde=2,
            qdrant_micro_chunks=3,
            fts_entries=1,
            registry_tokens=5,
            entity_facts=2,
        )
        assert r.total == 14

    def test_defaults_are_zero(self):
        from memory_lifecycle import DeleteResult

        r = DeleteResult()
        assert r.total == 0
        assert r.memory_id == ""


class TestDeleteMemoryComplete:
    """delete_memory_complete cleans ALL artifact types."""

    @pytest.fixture
    def mock_qdrant(self):
        client = MagicMock()
        client.delete.return_value = MagicMock(operation_id=1)
        client.scroll.return_value = ([], None)
        with patch("memory_lifecycle.qdrant_client", return_value=client):
            yield client

    @pytest.fixture
    def mock_fts(self):
        with patch("memory_lifecycle.delete_fts_chunks_by_qdrant_id", return_value=1) as m:
            yield m

    @pytest.fixture
    def mock_needle(self):
        with patch("memory_lifecycle.delete_needle_tokens_by_memory", return_value=3) as m:
            yield m

    @pytest.fixture
    def mock_entity_facts(self):
        with patch("memory_lifecycle._delete_entity_facts_for_memory", return_value=2) as m:
            yield m

    @pytest.mark.asyncio
    async def test_calls_all_cleanup_steps(
        self, mock_qdrant, mock_fts, mock_needle, mock_entity_facts
    ):
        from memory_lifecycle import delete_memory_complete

        result = await delete_memory_complete("mem-123", "test-ns")

        assert result.memory_id == "mem-123"
        assert result.qdrant_primary == 1
        assert result.fts_entries == 1
        assert result.registry_tokens == 3
        assert result.entity_facts == 2

        assert mock_qdrant.delete.call_count == 3
        mock_fts.assert_called_once_with("mem-123")
        mock_needle.assert_called_once_with("mem-123")
        mock_entity_facts.assert_called_once_with("mem-123")

    @pytest.mark.asyncio
    async def test_cleans_up_child_fts_and_needle(
        self, mock_fts, mock_needle, mock_entity_facts
    ):
        """Micro-chunk and reverse HyDE FTS/needle rows are cleaned up."""
        client = MagicMock()
        client.delete.return_value = MagicMock(operation_id=1)

        mc1 = MagicMock(); mc1.id = "micro-1"
        mc2 = MagicMock(); mc2.id = "micro-2"
        rh1 = MagicMock(); rh1.id = "rhyde-1"

        def scroll_side_effect(**kwargs):
            filt = kwargs.get("scroll_filter")
            if filt and filt.must and filt.must[0].key == "parent_id":
                return ([mc1, mc2], None)
            elif filt and filt.must and filt.must[0].key == "source_memory_id":
                return ([rh1], None)
            return ([], None)

        client.scroll.side_effect = scroll_side_effect

        with patch("memory_lifecycle.qdrant_client", return_value=client):
            from memory_lifecycle import delete_memory_complete
            result = await delete_memory_complete("mem-parent", "ns")

        fts_ids = [c[0][0] for c in mock_fts.call_args_list]
        assert "mem-parent" in fts_ids
        assert "micro-1" in fts_ids
        assert "micro-2" in fts_ids
        assert "rhyde-1" in fts_ids
        assert mock_fts.call_count == 4

        needle_ids = [c[0][0] for c in mock_needle.call_args_list]
        assert "mem-parent" in needle_ids
        assert "micro-1" in needle_ids
        assert "micro-2" in needle_ids
        assert "rhyde-1" in needle_ids
        assert mock_needle.call_count == 4

    @pytest.mark.asyncio
    async def test_qdrant_failure_doesnt_block_sqlite_cleanup(
        self, mock_fts, mock_needle, mock_entity_facts
    ):
        """If Qdrant is down, SQLite cleanup still runs (for primary at minimum)."""
        client = MagicMock()
        client.delete.side_effect = Exception("Qdrant connection refused")
        client.scroll.side_effect = Exception("Qdrant connection refused")
        with patch("memory_lifecycle.qdrant_client", return_value=client):
            from memory_lifecycle import delete_memory_complete

            result = await delete_memory_complete("mem-fail", "ns")

        assert result.qdrant_primary == 0
        mock_fts.assert_called_once_with("mem-fail")
        mock_needle.assert_called_once_with("mem-fail")
        mock_entity_facts.assert_called_once_with("mem-fail")

    @pytest.mark.asyncio
    async def test_uses_collection_router(self, mock_fts, mock_needle, mock_entity_facts):
        """Respects namespace-to-collection routing."""
        client = MagicMock()
        client.delete.return_value = MagicMock(operation_id=0)
        client.scroll.return_value = ([], None)
        with patch("memory_lifecycle.qdrant_client", return_value=client), \
             patch("memory_lifecycle.collection_for", return_value="archivist_custom") as cf:
            from memory_lifecycle import delete_memory_complete

            await delete_memory_complete("m1", "custom-ns")

        cf.assert_called_once_with("custom-ns")
        assert client.delete.call_args_list[0][1]["collection_name"] == "archivist_custom"

    @pytest.mark.asyncio
    async def test_collection_override(self, mock_qdrant, mock_fts, mock_needle, mock_entity_facts):
        """Explicit collection= kwarg overrides routing."""
        from memory_lifecycle import delete_memory_complete

        await delete_memory_complete("m1", "ns", collection="override_col")

        assert mock_qdrant.delete.call_args_list[0][1]["collection_name"] == "override_col"


class TestArchiveMemoryComplete:
    """archive_memory_complete flags all related Qdrant points."""

    @pytest.mark.asyncio
    async def test_archives_primary_and_children(self):
        client = MagicMock()
        with patch("memory_lifecycle.qdrant_client", return_value=client), \
             patch("memory_lifecycle.collection_for", return_value="test_col"):
            from memory_lifecycle import archive_memory_complete

            count = await archive_memory_complete("m1", "ns")

        assert count >= 1
        assert client.set_payload.call_count == 3
        for c in client.set_payload.call_args_list:
            assert c[1]["payload"] == {"archived": True}
            assert c[1]["collection_name"] == "test_col"


class TestDeleteFtsChunksByQdrantId:
    """New graph.delete_fts_chunks_by_qdrant_id works correctly."""

    def test_deletes_matching_rows(self):
        from graph import get_db, init_schema, upsert_fts_chunk, delete_fts_chunks_by_qdrant_id

        upsert_fts_chunk("qid-1", "some text", "test.md", 0, "agent", "ns")
        upsert_fts_chunk("qid-2", "other text", "test.md", 1, "agent", "ns")

        deleted = delete_fts_chunks_by_qdrant_id("qid-1")
        assert deleted == 1

        conn = get_db()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id = 'qid-1'"
        ).fetchone()[0]
        conn.close()
        assert remaining == 0

    def test_returns_zero_for_missing_id(self):
        from graph import delete_fts_chunks_by_qdrant_id
        assert delete_fts_chunks_by_qdrant_id("nonexistent") == 0

    def test_leaves_other_rows_intact(self):
        from graph import get_db, upsert_fts_chunk, delete_fts_chunks_by_qdrant_id

        upsert_fts_chunk("keep-me", "important text", "f.md", 0, "a", "ns")
        upsert_fts_chunk("delete-me", "trash text", "f.md", 1, "a", "ns")

        delete_fts_chunks_by_qdrant_id("delete-me")

        conn = get_db()
        kept = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id = 'keep-me'"
        ).fetchone()[0]
        conn.close()
        assert kept == 1


class TestDeleteEntityFactsForMemory:
    """_delete_entity_facts_for_memory soft-deactivates linked facts."""

    def test_deactivates_matching_facts(self):
        from graph import get_db, upsert_entity, add_fact
        from memory_lifecycle import _delete_entity_facts_for_memory

        eid = upsert_entity("test-entity")
        add_fact(eid, "some fact", source_file="explicit/mem-xyz-agent", agent_id="agent")
        add_fact(eid, "other fact", source_file="explicit/mem-xyz-agent", agent_id="agent")
        add_fact(eid, "unrelated fact", source_file="explicit/other-agent", agent_id="other")

        count = _delete_entity_facts_for_memory("mem-xyz")
        assert count == 2

        conn = get_db()
        active = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE source_file LIKE '%mem-xyz%' AND is_active = 1"
        ).fetchone()[0]
        conn.close()
        assert active == 0

    def test_does_not_affect_unrelated_facts(self):
        from graph import get_db, upsert_entity, add_fact
        from memory_lifecycle import _delete_entity_facts_for_memory

        eid = upsert_entity("test-entity-2")
        add_fact(eid, "safe fact", source_file="explicit/safe-agent", agent_id="safe")

        _delete_entity_facts_for_memory("dangerous-id")

        conn = get_db()
        active = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE source_file LIKE '%safe%' AND is_active = 1"
        ).fetchone()[0]
        conn.close()
        assert active == 1


class TestCuratorQueueDrainAsync:
    """curator_queue.drain() is async and calls lifecycle functions."""

    @pytest.mark.asyncio
    async def test_drain_is_async_coroutine(self):
        from curator_queue import drain
        import inspect
        assert inspect.iscoroutinefunction(drain)

    @pytest.mark.asyncio
    async def test_delete_op_calls_lifecycle(self):
        from curator_queue import enqueue, drain

        with patch("curator_queue._apply_delete", new_callable=AsyncMock) as mock_del:
            enqueue("delete_memory", {"memory_ids": ["m1"], "namespace": "ns"})
            result = await drain(limit=10)

        assert len(result) == 1
        assert result[0]["status"] == "applied"
        mock_del.assert_called_once()

    @pytest.mark.asyncio
    async def test_archive_op_calls_lifecycle(self):
        from curator_queue import enqueue, drain

        with patch("curator_queue._apply_archive", new_callable=AsyncMock) as mock_arc:
            enqueue("archive_memory", {"memory_ids": ["m1"], "namespace": "ns"})
            result = await drain(limit=10)

        assert len(result) == 1
        assert result[0]["status"] == "applied"
        mock_arc.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_op_marked_failed(self):
        from curator_queue import enqueue, drain

        with patch("curator_queue._apply_delete", new_callable=AsyncMock, side_effect=Exception("boom")):
            enqueue("delete_memory", {"memory_ids": ["m1"]})
            result = await drain(limit=10)

        assert result[0]["status"] == "failed"


class TestMergeUsesLifecycle:
    """merge.py uses delete_memory_complete for each original memory."""

    @pytest.mark.asyncio
    async def test_merge_calls_delete_per_id(self):
        mock_client = MagicMock()
        mock_points = []
        for mid in ["id1", "id2"]:
            pt = MagicMock()
            pt.id = mid
            pt.payload = {
                "text": "test text",
                "date": "2025-01-01",
                "team": "test",
                "namespace": "ns",
                "version": 1,
                "importance_score": 0.5,
                "consistency_level": "eventual",
            }
            mock_points.append(pt)
        mock_client.retrieve.return_value = mock_points

        mock_del = AsyncMock()

        with patch("merge.qdrant_client", return_value=mock_client), \
             patch("merge.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024), \
             patch("merge.llm_query", new_callable=AsyncMock, return_value="merged text"), \
             patch("merge.record_version", return_value=2), \
             patch("merge.log_memory_event", new_callable=AsyncMock), \
             patch("memory_lifecycle.delete_memory_complete", mock_del):
            from merge import merge_memories

            result = await merge_memories(["id1", "id2"], "semantic", "agent", "ns")

        assert mock_del.call_count == 2


class TestMetricsExist:
    """New lifecycle metrics are defined."""

    def test_delete_complete_metric(self):
        import metrics as m
        assert hasattr(m, "DELETE_COMPLETE")
        assert "delete" in m.DELETE_COMPLETE.lower()

    def test_archive_complete_metric(self):
        import metrics as m
        assert hasattr(m, "ARCHIVE_COMPLETE")
        assert "archive" in m.ARCHIVE_COMPLETE.lower()
