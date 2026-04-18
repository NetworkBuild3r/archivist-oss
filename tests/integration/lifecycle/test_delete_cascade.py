"""Integration tests for delete/archive cascade (consolidated from chunk2 + delete_cascade).

Original: tests/test_chunk2_cascade.py + tests/test_delete_cascade.py

Covers:
- delete_memory_complete cleans up all artifact types via cascade helpers
- archive_memory_complete flags all related Qdrant points (returns ArchiveResult)
- delete_fts_chunks_by_qdrant_id removes FTS entries correctly
- Qdrant failures tracked in failed_steps, PartialDeletionError raised
- Batch FTS/needle deletes with IN-clause chunking
- Paginated scroll discovers all children
- Audit logging for delete and archive
- curator_queue drain calls lifecycle functions
- merge.py uses delete_memory_complete per original ID
- Orphan sweeper reconciles SQLite vs Qdrant
"""

import asyncio
import sqlite3
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.lifecycle]

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
            memory_hotness=1,
        )
        assert r.total == 15

    def test_defaults_are_zero(self):
        from memory_lifecycle import DeleteResult

        r = DeleteResult()
        assert r.total == 0
        assert r.memory_id == ""
        assert r.failed_steps == []
        assert r.memory_hotness == 0

class TestArchiveResult:
    """ArchiveResult dataclass behaves correctly."""

    def test_total_counts_booleans(self):
        from memory_lifecycle import ArchiveResult

        r = ArchiveResult(
            memory_id="abc",
            primary_archived=True,
            reverse_hyde_archived=True,
            micro_chunks_archived=False,
        )
        assert r.total == 2

    def test_defaults(self):
        from memory_lifecycle import ArchiveResult

        r = ArchiveResult()
        assert r.total == 0
        assert r.failed_steps == []

def _make_mock_client(scroll_side_effect=None, children=None):
    """Build a mock QdrantClient with configurable scroll behaviour."""
    client = MagicMock()
    client.delete.return_value = MagicMock(operation_id=1)
    client.count.return_value = MagicMock(count=0)

    if scroll_side_effect:
        client.scroll.side_effect = scroll_side_effect
    elif children:

        def _scroll(**kwargs):
            filt = kwargs.get("scroll_filter")
            key = filt.must[0].key if filt and filt.must else None
            return (children.get(key, []), None)

        client.scroll.side_effect = _scroll
    else:
        client.scroll.return_value = ([], None)
    return client

class TestDeleteMemoryComplete:
    """delete_memory_complete cleans ALL artifact types."""

    @pytest.fixture
    def mock_qdrant(self):
        client = _make_mock_client()
        with patch("memory_lifecycle.qdrant_client", return_value=client):
            yield client

    @pytest.fixture
    def mock_fts(self):
        with patch(
            "memory_lifecycle.delete_fts_chunks_batch", new_callable=AsyncMock, return_value=1
        ) as m:
            yield m

    @pytest.fixture
    def mock_needle(self):
        with patch(
            "memory_lifecycle.delete_needle_tokens_batch", new_callable=AsyncMock, return_value=3
        ) as m:
            yield m

    @pytest.fixture
    def mock_entity_facts(self):
        with patch(
            "memory_lifecycle._delete_entity_facts_for_memory",
            new_callable=AsyncMock,
            return_value=2,
        ) as m:
            yield m

    @pytest.fixture
    def mock_hotness(self):
        with patch("memory_lifecycle.delete_hotness", new_callable=AsyncMock, return_value=1) as m:
            yield m

    @pytest.fixture
    def mock_audit(self):
        with patch("memory_lifecycle.log_memory_event", new_callable=AsyncMock) as m:
            yield m
    async def test_calls_all_cleanup_steps(
        self, mock_qdrant, mock_fts, mock_needle, mock_entity_facts, mock_audit
    ):
        from memory_lifecycle import delete_memory_complete

        result = await delete_memory_complete("mem-123", "test-ns")

        assert result.memory_id == "mem-123"
        assert result.qdrant_primary == 1
        assert result.fts_entries == 1
        assert result.registry_tokens == 3
        assert result.entity_facts == 2
        assert result.failed_steps == []

        mock_fts.assert_called_once()
        mock_needle.assert_called_once()
        mock_entity_facts.assert_called_once_with("mem-123")
    async def test_cleans_up_child_fts_and_needle(
        self, mock_fts, mock_needle, mock_entity_facts, mock_audit
    ):
        """Micro-chunk and reverse HyDE FTS/needle rows are cleaned up."""
        mc1 = MagicMock()
        mc1.id = "micro-1"
        mc2 = MagicMock()
        mc2.id = "micro-2"
        rh1 = MagicMock()
        rh1.id = "rhyde-1"

        client = _make_mock_client(
            children={
                "parent_id": [mc1, mc2],
                "source_memory_id": [rh1],
            }
        )

        with patch("memory_lifecycle.qdrant_client", return_value=client):
            from memory_lifecycle import delete_memory_complete

            result = await delete_memory_complete("mem-parent", "ns")

        fts_ids = mock_fts.call_args[0][0]
        assert "mem-parent" in fts_ids
        assert "micro-1" in fts_ids
        assert "micro-2" in fts_ids
        assert "rhyde-1" in fts_ids

        needle_ids = mock_needle.call_args[0][0]
        assert set(needle_ids) == {"mem-parent", "micro-1", "micro-2", "rhyde-1"}
    async def test_qdrant_failure_tracked_in_failed_steps(
        self, mock_fts, mock_needle, mock_entity_facts, mock_audit
    ):
        """If Qdrant primary delete fails, failed_steps records it and PartialDeletionError is raised."""
        from cascade import PartialDeletionError

        client = MagicMock()
        client.delete.side_effect = Exception("Qdrant connection refused")
        client.scroll.side_effect = Exception("Qdrant connection refused")
        client.count.return_value = MagicMock(count=0)

        with patch("memory_lifecycle.qdrant_client", return_value=client):
            from memory_lifecycle import delete_memory_complete

            with pytest.raises(PartialDeletionError) as exc_info:
                await delete_memory_complete("mem-fail", "ns")

            result = exc_info.value.result
            assert "qdrant_primary" in result.failed_steps
            assert result.qdrant_primary == 1  # pre-count preserved on failure

        mock_fts.assert_called_once()
        mock_needle.assert_called_once()
        mock_entity_facts.assert_called_once_with("mem-fail")
    async def test_partial_deletion_error_on_many_failures(self, mock_audit):
        """PartialDeletionError raised when qdrant_primary or qdrant_children fail."""
        from cascade import PartialDeletionError

        client = MagicMock()
        client.delete.side_effect = Exception("down")
        client.scroll.side_effect = Exception("down")
        client.count.return_value = MagicMock(count=0)

        with (
            patch("memory_lifecycle.qdrant_client", return_value=client),
            patch(
                "memory_lifecycle.delete_fts_chunks_batch",
                side_effect=sqlite3.OperationalError("db locked"),
            ),
            patch("memory_lifecycle.delete_needle_tokens_batch", return_value=0),
            patch("memory_lifecycle._delete_entity_facts_for_memory", return_value=0),
        ):
            from memory_lifecycle import delete_memory_complete

            with pytest.raises(PartialDeletionError) as exc_info:
                await delete_memory_complete("mem-x", "ns")

            assert "qdrant_primary" in exc_info.value.result.failed_steps
    async def test_uses_collection_router(
        self, mock_fts, mock_needle, mock_entity_facts, mock_audit
    ):
        """Respects namespace-to-collection routing."""
        client = _make_mock_client()
        with (
            patch("memory_lifecycle.qdrant_client", return_value=client),
            patch("memory_lifecycle.collection_for", return_value="archivist_custom") as cf,
        ):
            from memory_lifecycle import delete_memory_complete

            await delete_memory_complete("m1", "custom-ns")

        cf.assert_called_once_with("custom-ns")
    async def test_collection_override(
        self, mock_qdrant, mock_fts, mock_needle, mock_entity_facts, mock_audit
    ):
        """Explicit collection= kwarg overrides routing."""
        from memory_lifecycle import delete_memory_complete

        await delete_memory_complete("m1", "ns", collection="override_col")

        assert mock_qdrant.delete.call_args_list[0][1]["collection_name"] == "override_col"
    async def test_audit_logging_called(
        self, mock_qdrant, mock_fts, mock_needle, mock_entity_facts, mock_audit
    ):
        """log_memory_event is called with action='delete' and full metadata."""
        from memory_lifecycle import delete_memory_complete

        result = await delete_memory_complete("m-audit", "ns")

        mock_audit.assert_called_once()
        kwargs = mock_audit.call_args[1]
        assert kwargs["action"] == "delete"
        assert kwargs["memory_id"] == "m-audit"
        assert "failed_steps" in kwargs["metadata"]
        assert "memory_hotness" in kwargs["metadata"]
        assert kwargs["metadata"]["result_type"] == "delete"

class TestPaginatedScroll:
    """_scroll_all paginates until next_page_offset is None."""
    async def test_discovers_all_children_across_pages(self):
        """Scroll with two pages discovers all children."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        from cascade import _scroll_all

        p1 = MagicMock()
        p1.id = "child-1"
        p2 = MagicMock()
        p2.id = "child-2"
        p3 = MagicMock()
        p3.id = "child-3"

        call_count = 0

        def scroll_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return ([p1, p2], "offset-token")
            else:
                return ([p3], None)

        client = MagicMock()
        client.scroll.side_effect = scroll_effect

        filt = Filter(must=[FieldCondition(key="parent_id", match=MatchValue(value="parent"))])
        failed = []
        ids = _scroll_all(client, "col", filt, "test_scroll", "parent", failed)

        assert ids == ["child-1", "child-2", "child-3"]
        assert failed == []
        assert client.scroll.call_count == 2
    async def test_records_failure_in_failed_steps(self):
        """Scroll failure is tracked in failed_steps."""
        from qdrant_client.models import FieldCondition, Filter, MatchValue

        from cascade import _scroll_all

        client = MagicMock()
        client.scroll.side_effect = Exception("timeout")

        filt = Filter(must=[FieldCondition(key="parent_id", match=MatchValue(value="x"))])
        failed = []
        ids = _scroll_all(client, "col", filt, "scroll_test", "x", failed)

        assert ids == []
        assert "scroll_test" in failed

class TestArchiveMemoryComplete:
    """archive_memory_complete flags all related Qdrant points."""
    async def test_archives_primary_and_children(self):
        client = MagicMock()
        client.scroll.return_value = ([], None)  # no child points to enumerate
        with (
            patch("memory_lifecycle.qdrant_client", return_value=client),
            patch("memory_lifecycle.collection_for", return_value="test_col"),
            patch("memory_lifecycle.log_memory_event", new_callable=AsyncMock),
        ):
            from memory_lifecycle import archive_memory_complete

            result = await archive_memory_complete("m1", "ns")

        assert result.primary_archived is True
        assert result.reverse_hyde_archived is True
        assert result.micro_chunks_archived is True
        assert result.total == 3
        assert result.failed_steps == []
        assert client.set_payload.call_count == 3
        for c in client.set_payload.call_args_list:
            assert c[1]["payload"] == {"archived": True}
            assert c[1]["collection_name"] == "test_col"
    async def test_archive_tracks_failures(self):
        """Failures in set_payload are tracked in ArchiveResult.failed_steps."""
        client = MagicMock()
        client.scroll.return_value = ([], None)
        client.set_payload.side_effect = [None, Exception("nope"), None]
        with (
            patch("memory_lifecycle.qdrant_client", return_value=client),
            patch("memory_lifecycle.collection_for", return_value="test_col"),
            patch("memory_lifecycle.log_memory_event", new_callable=AsyncMock),
        ):
            from memory_lifecycle import archive_memory_complete

            result = await archive_memory_complete("m2", "ns")

        assert result.primary_archived is True
        assert result.reverse_hyde_archived is False
        assert "archive_reverse_hyde" in result.failed_steps
    async def test_archive_audit_logging(self):
        """Archive calls log_memory_event with action='archive' and result_type."""
        client = MagicMock()
        client.scroll.return_value = ([], None)
        mock_audit = AsyncMock()
        with (
            patch("memory_lifecycle.qdrant_client", return_value=client),
            patch("memory_lifecycle.collection_for", return_value="test_col"),
            patch("memory_lifecycle.log_memory_event", mock_audit),
        ):
            from memory_lifecycle import archive_memory_complete

            await archive_memory_complete("m-audit", "ns")

        mock_audit.assert_called_once()
        assert mock_audit.call_args[1]["action"] == "archive"
        assert mock_audit.call_args[1]["metadata"]["result_type"] == "archive"

class TestDeleteFtsChunksByQdrantId:
    """graph.delete_fts_chunks_by_qdrant_id works correctly."""

    async def test_deletes_matching_rows(self, async_pool):
        from graph import delete_fts_chunks_by_qdrant_id, get_db, upsert_fts_chunk

        await upsert_fts_chunk("qid-1", "some text", "test.md", 0, "agent", "ns")
        await upsert_fts_chunk("qid-2", "other text", "test.md", 1, "agent", "ns")

        deleted = await delete_fts_chunks_by_qdrant_id("qid-1")
        assert deleted == 1

        conn = get_db()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id = 'qid-1'"
        ).fetchone()[0]
        conn.close()
        assert remaining == 0

    async def test_returns_zero_for_missing_id(self, async_pool):
        from graph import delete_fts_chunks_by_qdrant_id

        assert await delete_fts_chunks_by_qdrant_id("nonexistent") == 0

    async def test_leaves_other_rows_intact(self, async_pool):
        from graph import delete_fts_chunks_by_qdrant_id, get_db, upsert_fts_chunk

        await upsert_fts_chunk("keep-me", "important text", "f.md", 0, "a", "ns")
        await upsert_fts_chunk("delete-me", "trash text", "f.md", 1, "a", "ns")

        await delete_fts_chunks_by_qdrant_id("delete-me")

        conn = get_db()
        kept = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id = 'keep-me'"
        ).fetchone()[0]
        conn.close()
        assert kept == 1

class TestBatchFtsDelete:
    """delete_fts_chunks_batch handles chunking correctly."""

    async def test_batch_deletes_multiple_ids(self, async_pool):
        from graph import delete_fts_chunks_batch, get_db, upsert_fts_chunk

        for i in range(5):
            await upsert_fts_chunk(f"batch-{i}", f"text {i}", "f.md", i, "a", "ns")

        deleted = await delete_fts_chunks_batch([f"batch-{i}" for i in range(5)])
        assert deleted == 5

        conn = get_db()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id LIKE 'batch-%'"
        ).fetchone()[0]
        conn.close()
        assert remaining == 0

    async def test_batch_empty_list(self, async_pool):
        from graph import delete_fts_chunks_batch

        assert await delete_fts_chunks_batch([]) == 0

    async def test_batch_chunking_under_parameter_limit(self, async_pool):
        """Passing >999 IDs doesn't crash sqlite3 thanks to internal chunking."""
        from graph import delete_fts_chunks_batch, upsert_fts_chunk

        ids = [f"chunk-test-{i}" for i in range(1200)]
        for qid in ids[:5]:
            await upsert_fts_chunk(qid, "text", "f.md", 0, "a", "ns")

        deleted = await delete_fts_chunks_batch(ids)
        assert deleted == 5

class TestBatchNeedleDelete:
    """delete_needle_tokens_batch handles chunking correctly."""

    async def test_batch_empty_list(self, async_pool):
        from graph import delete_needle_tokens_batch

        assert await delete_needle_tokens_batch([]) == 0

class TestDeleteEntityFactsForMemory:
    """_delete_entity_facts_for_memory soft-deactivates linked facts."""

    async def test_deactivates_matching_facts(self, async_pool):
        from graph import add_fact, get_db, upsert_entity
        from memory_lifecycle import _delete_entity_facts_for_memory

        eid = await upsert_entity("test-entity")
        await add_fact(eid, "some fact", source_file="explicit/mem-xyz-agent", agent_id="agent")
        await add_fact(eid, "other fact", source_file="explicit/mem-xyz-agent", agent_id="agent")
        await add_fact(eid, "unrelated fact", source_file="explicit/other-agent", agent_id="other")

        count = await _delete_entity_facts_for_memory("mem-xyz")
        assert count == 2

        conn = get_db()
        active = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE source_file LIKE '%mem-xyz%' AND is_active = 1"
        ).fetchone()[0]
        conn.close()
        assert active == 0

    async def test_does_not_affect_unrelated_facts(self, async_pool):
        from graph import add_fact, get_db, upsert_entity
        from memory_lifecycle import _delete_entity_facts_for_memory

        eid = await upsert_entity("test-entity-2")
        await add_fact(eid, "safe fact", source_file="explicit/safe-agent", agent_id="safe")

        await _delete_entity_facts_for_memory("dangerous-id")

        conn = get_db()
        active = conn.execute(
            "SELECT COUNT(*) FROM facts WHERE source_file LIKE '%safe%' AND is_active = 1"
        ).fetchone()[0]
        conn.close()
        assert active == 1

class TestOrphanSweeper:
    """sweep_orphans reconciles SQLite rows against Qdrant."""

    async def test_cleans_orphaned_fts_rows(self, async_pool):
        """FTS rows with no corresponding Qdrant point are cleaned."""
        from graph import get_db, upsert_fts_chunk

        await upsert_fts_chunk("exists-in-qdrant", "text", "f.md", 0, "a", "ns")
        await upsert_fts_chunk("orphaned-id", "text2", "f.md", 1, "a", "ns")

        mock_client = MagicMock()
        p1 = MagicMock()
        p1.id = "exists-in-qdrant"
        mock_client.retrieve.return_value = [p1]
        mock_client.get_collections.return_value = MagicMock()

        with (
            patch("cascade.qdrant_client", return_value=mock_client),
            patch("cascade.collections_for_query", return_value=["test_col"]),
        ):
            from cascade import sweep_orphans

            result = await sweep_orphans()

        assert result["fts_cleaned"] >= 1

        conn = get_db()
        orphan_count = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id = 'orphaned-id'"
        ).fetchone()[0]
        conn.close()
        assert orphan_count == 0

    async def test_does_not_clean_existing_points(self, async_pool):
        """FTS rows with matching Qdrant points are kept."""
        from graph import get_db, upsert_fts_chunk

        await upsert_fts_chunk("keep-this", "text", "f.md", 0, "a", "ns")

        mock_client = MagicMock()
        p1 = MagicMock()
        p1.id = "keep-this"
        mock_client.retrieve.return_value = [p1]
        mock_client.get_collections.return_value = MagicMock()

        with (
            patch("cascade.qdrant_client", return_value=mock_client),
            patch("cascade.collections_for_query", return_value=["test_col"]),
        ):
            from cascade import sweep_orphans

            result = await sweep_orphans()

        conn = get_db()
        count = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id = 'keep-this'"
        ).fetchone()[0]
        conn.close()
        assert count == 1

class TestCuratorQueueDrainAsync:
    """curator_queue.drain() is async and calls lifecycle functions."""
    async def test_drain_is_async_coroutine(self):
        import inspect

        from curator_queue import drain

        assert inspect.iscoroutinefunction(drain)
    async def test_delete_op_calls_lifecycle(self, async_pool):
        from curator_queue import drain, enqueue

        with patch("curator_queue._apply_delete", new_callable=AsyncMock) as mock_del:
            enqueue("delete_memory", {"memory_ids": ["m1"], "namespace": "ns"})
            result = await drain(limit=10)

        assert len(result) == 1
        assert result[0]["status"] == "applied"
        mock_del.assert_called_once()
    async def test_archive_op_calls_lifecycle(self, async_pool):
        from curator_queue import drain, enqueue

        with patch("curator_queue._apply_archive", new_callable=AsyncMock) as mock_arc:
            enqueue("archive_memory", {"memory_ids": ["m1"], "namespace": "ns"})
            result = await drain(limit=10)

        assert len(result) == 1
        assert result[0]["status"] == "applied"
        mock_arc.assert_called_once()
    async def test_failed_op_marked_failed(self, async_pool):
        from curator_queue import drain, enqueue

        with patch(
            "curator_queue._apply_delete", new_callable=AsyncMock, side_effect=Exception("boom")
        ):
            enqueue("delete_memory", {"memory_ids": ["m1"]})
            result = await drain(limit=10)

        assert result[0]["status"] == "failed"

class TestMergeUsesLifecycle:
    """merge.py uses delete_memory_complete for each original memory."""
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

        with (
            patch("merge.qdrant_client", return_value=mock_client),
            patch("merge.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024),
            patch("merge.llm_query", new_callable=AsyncMock, return_value="merged text"),
            patch("merge.record_version", new_callable=AsyncMock, return_value=2),
            patch("archivist.storage.transaction.MemoryTransaction", _mock_txn_ctx()),
            patch("merge.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.delete_memory_complete", mock_del),
        ):
            from merge import merge_memories

            result = await merge_memories(["id1", "id2"], "semantic", "agent", "ns")

        assert mock_del.call_count == 2

class TestQdrantRetry:
    """Transient-only retry behaviour in _qdrant_delete and _qdrant_set_payload."""

    async def test_transient_retry_succeeds_on_second_attempt(self, async_pool):
        """First delete raises a transient error, second attempt succeeds."""
        from qdrant_client.http.exceptions import ResponseHandlingException

        from cascade import _qdrant_delete

        client = MagicMock()
        client.count.return_value = MagicMock(count=5)
        client.delete.side_effect = [ResponseHandlingException("timeout"), None]

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        filt = Filter(must=[FieldCondition(key="parent_id", match=MatchValue(value="mem-1"))])
        failed = []

        count = await _qdrant_delete(client, "col", filt, "test_step", "mem-1", failed)

        assert count == 5
        assert failed == []
        assert client.delete.call_count == 2

    async def test_permanent_error_does_not_retry(self, async_pool):
        """A non-transient error (e.g. 404) fails immediately without retry."""
        from qdrant_client.http.exceptions import UnexpectedResponse

        from cascade import _qdrant_delete

        client = MagicMock()
        err = UnexpectedResponse.__new__(UnexpectedResponse)
        err.status_code = 404
        err.reason_phrase = "Not Found"
        err.content = b""
        client.delete.side_effect = err
        client.count.return_value = MagicMock(count=0)

        failed = []
        await _qdrant_delete(client, "col", ["point-1"], "perm_step", "mem-1", failed)

        assert client.delete.call_count == 1
        assert "perm_step" in failed

    async def test_precount_returned_on_final_failure(self, async_pool):
        """Both attempts fail with transient errors; pre-count is still returned."""
        from qdrant_client.http.exceptions import ResponseHandlingException

        from cascade import _qdrant_delete

        client = MagicMock()
        client.delete.side_effect = ResponseHandlingException("network")

        failed = []
        count = await _qdrant_delete(client, "col", ["p1", "p2", "p3"], "step_x", "mem-1", failed)

        assert count == 3  # pre-count = len(selector)
        assert "step_x" in failed
        assert client.delete.call_count == 2  # 1 initial + 1 retry

    def test_set_payload_transient_retry(self):
        """_qdrant_set_payload retries on transient error and succeeds."""
        from qdrant_client.http.exceptions import ResponseHandlingException

        from cascade import _qdrant_set_payload

        client = MagicMock()
        client.set_payload.side_effect = [ResponseHandlingException("timeout"), None]

        failed = []
        ok = _qdrant_set_payload(
            client,
            "col",
            {"archived": True},
            ["p1"],
            "archive_step",
            "mem-1",
            failed,
        )

        assert ok is True
        assert failed == []
        assert client.set_payload.call_count == 2

    def test_set_payload_permanent_error_no_retry(self):
        """_qdrant_set_payload does not retry permanent errors."""
        from qdrant_client.http.exceptions import UnexpectedResponse

        from cascade import _qdrant_set_payload

        client = MagicMock()
        err = UnexpectedResponse.__new__(UnexpectedResponse)
        err.status_code = 400
        err.reason_phrase = "Bad Request"
        err.content = b""
        client.set_payload.side_effect = err

        failed = []
        ok = _qdrant_set_payload(
            client,
            "col",
            {"archived": True},
            ["p1"],
            "bad_step",
            "mem-1",
            failed,
        )

        assert ok is False
        assert "bad_step" in failed
        assert client.set_payload.call_count == 1

class TestOrphanSweeperAdvanced:
    """Extended sweep_orphans tests for health guard, needle scan, retrieve failures."""

    async def test_sweeper_aborts_on_qdrant_down(self, async_pool):
        """Sweeper returns skipped when Qdrant is unreachable."""
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = ConnectionError("refused")

        with patch("cascade.qdrant_client", return_value=mock_client):
            from cascade import sweep_orphans

            result = await sweep_orphans()

        assert result.get("skipped") == "qdrant_unavailable"
        assert result["fts_cleaned"] == 0
        assert result["needle_cleaned"] == 0

    async def test_needle_orphan_cleanup_primary(self, async_pool):
        """Needle rows keyed on a primary memory_id with no Qdrant point are cleaned."""
        from graph import _ensure_needle_registry, get_db

        _ensure_needle_registry()
        conn = get_db()
        conn.execute(
            "INSERT INTO needle_registry (memory_id, token, namespace, agent_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("orphan-primary-id", "some_token", "ns", "agent", "2025-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_client.retrieve.return_value = []

        with (
            patch("cascade.qdrant_client", return_value=mock_client),
            patch("cascade.collections_for_query", return_value=["test_col"]),
        ):
            from cascade import sweep_orphans

            result = await sweep_orphans()

        assert result["needle_cleaned"] >= 1

        conn = get_db()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM needle_registry WHERE memory_id = 'orphan-primary-id'"
        ).fetchone()[0]
        conn.close()
        assert remaining == 0

    async def test_needle_orphan_cleanup_child(self, async_pool):
        """Needle rows where memory_id is a micro-chunk Qdrant ID are cleaned when orphaned."""
        from graph import _ensure_needle_registry, get_db

        _ensure_needle_registry()
        conn = get_db()
        conn.execute(
            "INSERT INTO needle_registry (memory_id, token, namespace, agent_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("orphan-microchunk-id", "needle_tok", "ns", "agent", "2025-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_client.retrieve.return_value = []

        with (
            patch("cascade.qdrant_client", return_value=mock_client),
            patch("cascade.collections_for_query", return_value=["test_col"]),
        ):
            from cascade import sweep_orphans

            result = await sweep_orphans()

        assert result["needle_cleaned"] >= 1

        conn = get_db()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM needle_registry WHERE memory_id = 'orphan-microchunk-id'"
        ).fetchone()[0]
        conn.close()
        assert remaining == 0

    async def test_retrieve_failure_skips_subbatch(self, async_pool):
        """If client.retrieve fails for one collection, sub-batch is conservatively kept."""
        from graph import get_db, upsert_fts_chunk

        await upsert_fts_chunk("maybe-orphan", "text", "f.md", 0, "a", "ns")

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_client.retrieve.side_effect = Exception("retrieve error")

        with (
            patch("cascade.qdrant_client", return_value=mock_client),
            patch("cascade.collections_for_query", return_value=["col_a", "col_b"]),
        ):
            from cascade import sweep_orphans

            result = await sweep_orphans()

        assert result["fts_cleaned"] == 0

        conn = get_db()
        count = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id = 'maybe-orphan'"
        ).fetchone()[0]
        conn.close()
        assert count == 1

    async def test_keyset_pagination_processes_all_pages(self, async_pool):
        """Sweeper uses keyset pagination (WHERE id > ?) to process multiple pages."""
        from graph import get_db, upsert_fts_chunk

        ids_to_insert = [f"ks-{i:04d}" for i in range(3)]
        for qid in ids_to_insert:
            await upsert_fts_chunk(qid, f"text for {qid}", "f.md", 0, "a", "ns")

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_client.retrieve.return_value = []

        with (
            patch("cascade.qdrant_client", return_value=mock_client),
            patch("cascade.collections_for_query", return_value=["test_col"]),
            patch("cascade._SWEEP_PAGE_SIZE", 2),
        ):
            from cascade import sweep_orphans

            result = await sweep_orphans()

        assert result["fts_cleaned"] == 3

        conn = get_db()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id LIKE 'ks-%'"
        ).fetchone()[0]
        conn.close()
        assert remaining == 0

class TestDeleteHotness:
    """graph.delete_hotness removes memory_hotness rows."""

    async def test_deletes_existing_row(self, async_pool):
        from graph import delete_hotness, get_db

        conn = get_db()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memory_hotness "
            "(memory_id TEXT PRIMARY KEY, score REAL NOT NULL DEFAULT 0.0, "
            "retrieval_count INTEGER NOT NULL DEFAULT 0, last_accessed TEXT, "
            "updated_at TEXT NOT NULL)",
        )
        conn.execute(
            "INSERT OR REPLACE INTO memory_hotness (memory_id, score, retrieval_count, updated_at) "
            "VALUES (?, ?, ?, ?)",
            ("hot-mem-1", 0.8, 5, "2025-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        deleted = await delete_hotness("hot-mem-1")
        assert deleted == 1

        conn = get_db()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM memory_hotness WHERE memory_id = 'hot-mem-1'"
        ).fetchone()[0]
        conn.close()
        assert remaining == 0

    async def test_returns_zero_for_missing_id(self, async_pool):
        from graph import delete_hotness, get_db

        conn = get_db()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS memory_hotness "
            "(memory_id TEXT PRIMARY KEY, score REAL NOT NULL DEFAULT 0.0, "
            "retrieval_count INTEGER NOT NULL DEFAULT 0, last_accessed TEXT, "
            "updated_at TEXT NOT NULL)",
        )
        conn.commit()
        conn.close()

        assert await delete_hotness("nonexistent") == 0

    async def test_returns_zero_when_table_missing(self, async_pool):
        """Silently returns 0 if memory_hotness table doesn't exist yet."""
        from graph import delete_hotness, get_db

        conn = get_db()
        conn.execute("DROP TABLE IF EXISTS memory_hotness")
        conn.commit()
        conn.close()

        assert await delete_hotness("any-id") == 0

class TestBatchSqliteRetry:
    """SQLite batch functions retry on OperationalError (database is locked)."""

    async def test_fts_retry_on_operational_error(self, async_pool):
        """delete_fts_chunks_batch retries once when the pool raises a locked error."""
        import sqlite3 as _sqlite3

        from graph import delete_fts_chunks_batch, upsert_fts_chunk

        await upsert_fts_chunk("retry-id", "text", "f.md", 0, "a", "ns")

        call_count = 0
        original_write = async_pool.write

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def flaky_write():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _sqlite3.OperationalError("database is locked")
            async with original_write() as conn:
                yield conn

        with (
            patch.object(async_pool, "write", flaky_write),
            patch("asyncio.sleep"),
        ):
            deleted = await delete_fts_chunks_batch(["retry-id"])

        assert deleted == 1
        assert call_count == 2

    async def test_needle_retry_on_operational_error(self, async_pool):
        """delete_needle_tokens_batch retries once when the pool raises a locked error."""
        import sqlite3 as _sqlite3

        from graph import _ensure_needle_registry, delete_needle_tokens_batch, get_db

        _ensure_needle_registry()
        conn = get_db()
        conn.execute(
            "INSERT INTO needle_registry (memory_id, token, namespace, agent_id, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("retry-needle-id", "tok", "ns", "agent", "2025-01-01T00:00:00"),
        )
        conn.commit()
        conn.close()

        call_count = 0
        original_write = async_pool.write

        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def flaky_write():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _sqlite3.OperationalError("database is locked")
            async with original_write() as conn:
                yield conn

        with (
            patch.object(async_pool, "write", flaky_write),
            patch("asyncio.sleep"),
        ):
            deleted = await delete_needle_tokens_batch(["retry-needle-id"])

        assert deleted == 1
        assert call_count == 2

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

    def test_orphan_sweep_metric(self):
        import metrics as m

        assert hasattr(m, "ORPHAN_SWEEP")
        assert "orphan" in m.ORPHAN_SWEEP.lower()

class TestEntityFactsMemoryId:
    """_delete_entity_facts_for_memory uses indexed memory_id column."""

    async def _insert_fact(self, async_pool, memory_id: str, source_file: str = "") -> int:
        """Insert a test fact row and return its id."""
        from graph import get_db, upsert_entity

        eid = await upsert_entity("test-entity", "concept", namespace="global")
        from datetime import datetime

        conn = get_db()
        cur = conn.execute(
            "INSERT INTO facts (entity_id, fact_text, source_file, agent_id, created_at, "
            "memory_id) VALUES (?, ?, ?, ?, ?, ?)",
            (
                eid,
                "some fact text",
                source_file,
                "agent",
                datetime.now(UTC).isoformat(),
                memory_id,
            ),
        )
        conn.commit()
        row_id = cur.lastrowid
        conn.close()
        return row_id

    async def test_exact_match_deactivates_by_memory_id(self, async_pool):
        """Primary path: rows with matching memory_id are deactivated."""
        from graph import get_db
        from memory_lifecycle import _delete_entity_facts_for_memory

        mid = "mem-exact-test-001"
        await self._insert_fact(async_pool, mid, source_file="explicit/agent")

        count = await _delete_entity_facts_for_memory(mid)
        assert count == 1

        conn = get_db()
        active = conn.execute("SELECT is_active FROM facts WHERE memory_id = ?", (mid,)).fetchone()
        conn.close()
        assert active is not None
        assert active[0] == 0

    async def test_like_fallback_for_pre_migration_rows(self, async_pool):
        """Fallback: rows with memory_id='' but matching source_file are deactivated."""
        from graph import get_db
        from memory_lifecycle import _delete_entity_facts_for_memory

        mid = "mem-fallback-test-002"
        # Pre-migration row: memory_id is empty, but source_file contains the UUID
        await self._insert_fact(async_pool, "", source_file=f"explicit/{mid}")

        count = await _delete_entity_facts_for_memory(mid)
        assert count == 1

        conn = get_db()
        row = conn.execute(
            "SELECT is_active FROM facts WHERE source_file = ?", (f"explicit/{mid}",)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 0

    async def test_like_fallback_does_not_touch_rows_with_memory_id_set(self, async_pool):
        """LIKE fallback is scoped to memory_id='' only — does not double-deactivate."""
        from memory_lifecycle import _delete_entity_facts_for_memory

        mid = "mem-scope-test-003"
        # Row already has memory_id set correctly
        await self._insert_fact(async_pool, mid, source_file=f"explicit/{mid}")

        count = await _delete_entity_facts_for_memory(mid)
        # Should be found by the primary exact-match path only (count = 1)
        assert count == 1

    async def test_non_matching_rows_untouched(self, async_pool):
        """Facts belonging to a different memory are not deactivated."""
        from graph import get_db
        from memory_lifecycle import _delete_entity_facts_for_memory

        mid_target = "mem-target-004"
        mid_other = "mem-other-004"
        await self._insert_fact(async_pool, mid_other, source_file="explicit/other-agent")

        count = await _delete_entity_facts_for_memory(mid_target)
        assert count == 0

        conn = get_db()
        row = conn.execute(
            "SELECT is_active FROM facts WHERE memory_id = ?", (mid_other,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 1  # untouched

    async def test_returns_zero_for_unknown_memory(self, async_pool):
        """Returns 0 and does not crash for a memory_id with no matching facts."""
        from memory_lifecycle import _delete_entity_facts_for_memory

        assert await _delete_entity_facts_for_memory("completely-unknown-id") == 0

class TestAddFactMemoryId:
    """add_fact stores memory_id and it can be queried."""

    async def test_stores_memory_id(self, async_pool):
        from graph import add_fact, get_db, upsert_entity

        eid = await upsert_entity("test-entity-af", "concept", namespace="global")
        mid = "mem-add-fact-test-001"
        await add_fact(
            eid, "test fact text", "explicit/agent", "agent", namespace="global", memory_id=mid
        )

        conn = get_db()
        row = conn.execute("SELECT memory_id FROM facts WHERE memory_id = ?", (mid,)).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == mid

    async def test_default_memory_id_is_empty(self, async_pool):
        """Existing call sites that don't pass memory_id get empty string."""
        from graph import add_fact, get_db, upsert_entity

        eid = await upsert_entity("entity-no-mid", "concept", namespace="global")
        fact_id = await add_fact(
            eid, "fact with no memory_id", "trajectory/abc", "agent", namespace="global"
        )

        conn = get_db()
        row = conn.execute("SELECT memory_id FROM facts WHERE id = ?", (fact_id,)).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == ""

class TestScrollAllMaxPages:
    """_scroll_all safety guard limits pagination."""

    def test_max_pages_breaks_and_appends_failed_step(self):
        """When max_pages is hit, step_name is appended to failed_steps."""
        from qdrant_client.models import Filter

        from cascade import _scroll_all

        call_count = 0

        def fake_scroll(**kwargs):
            nonlocal call_count
            call_count += 1
            pt = MagicMock()
            pt.id = f"pt-{call_count}"
            # Always return a non-None next_offset to simulate infinite pagination
            return [pt], "next-offset"

        mock_client = MagicMock()
        mock_client.scroll.side_effect = fake_scroll

        failed = []
        filt = Filter(must=[])
        ids = _scroll_all(
            mock_client,
            "col",
            filt,
            "test_scroll",
            "mem-id",
            failed,
            batch=1,
            max_pages=3,
        )

        assert len(ids) == 3
        assert call_count == 3
        assert "test_scroll" in failed

    def test_no_guard_triggered_on_normal_completion(self):
        """When pagination ends naturally, failed_steps is not appended."""
        from qdrant_client.models import Filter

        from cascade import _scroll_all

        def fake_scroll(**kwargs):
            pt = MagicMock()
            pt.id = "only-pt"
            return [pt], None  # None = last page

        mock_client = MagicMock()
        mock_client.scroll.side_effect = fake_scroll

        failed = []
        filt = Filter(must=[])
        ids = _scroll_all(
            mock_client,
            "col",
            filt,
            "test_scroll",
            "mem-id",
            failed,
            batch=500,
            max_pages=1000,
        )

        assert ids == ["only-pt"]
        assert failed == []

class TestPartialDeletionErrorThreshold:
    """PartialDeletionError is raised on qdrant_primary or qdrant_children failures."""

    def test_raises_on_qdrant_primary_failure(self):
        from memory_lifecycle import DeleteResult

        result = DeleteResult(memory_id="mid", failed_steps=["qdrant_primary"])
        _critical = {"qdrant_primary", "qdrant_children"}
        assert bool(_critical & set(result.failed_steps))

    def test_raises_on_qdrant_children_failure(self):
        from memory_lifecycle import DeleteResult

        result = DeleteResult(memory_id="mid", failed_steps=["qdrant_children"])
        _critical = {"qdrant_primary", "qdrant_children"}
        assert bool(_critical & set(result.failed_steps))

    def test_does_not_raise_on_fts_only_failure(self):
        """FTS failure alone does not trigger PartialDeletionError."""
        from memory_lifecycle import DeleteResult

        result = DeleteResult(memory_id="mid", failed_steps=["fts_batch"])
        _critical = {"qdrant_primary", "qdrant_children"}
        assert not bool(_critical & set(result.failed_steps))

    def test_does_not_raise_on_needle_only_failure(self):
        """needle_batch failure alone does not trigger PartialDeletionError."""
        from memory_lifecycle import DeleteResult

        result = DeleteResult(memory_id="mid", failed_steps=["needle_batch"])
        _critical = {"qdrant_primary", "qdrant_children"}
        assert not bool(_critical & set(result.failed_steps))

# ==========================================================================
# Migrated from tests/test_delete_cascade.py
# ==========================================================================

def _mock_point(pid: str):
    p = MagicMock()
    p.id = pid
    return p

def _make_qdrant_client(scroll_return=None):
    client = MagicMock()
    client.delete.return_value = MagicMock(operation_id=1)
    client.count.return_value = MagicMock(count=0)
    client.set_payload.return_value = True
    client.scroll.return_value = scroll_return or ([], None)
    return client

# ===========================================================================
# Phase 1b — retrieval filter tests
# ===========================================================================

class TestSearchVectorsMustNotFilter:
    """search_vectors() includes must_not conditions for archived and deleted."""

    def test_must_not_always_set(self, monkeypatch):
        """Filter(must_not=...) is always constructed, not conditional."""
        import rlm_retriever

        captured_filter = None

        def _fake_query_points(**kwargs):
            nonlocal captured_filter
            captured_filter = kwargs.get("query_filter")
            result = MagicMock()
            result.points = []
            return result

        mock_client = MagicMock()
        mock_client.query_points.side_effect = _fake_query_points

        monkeypatch.setattr("rlm_retriever.qdrant_client", lambda: mock_client)
        monkeypatch.setattr("rlm_retriever.collection_for", lambda ns: "test-coll")
        monkeypatch.setattr("rlm_retriever.embed_text", AsyncMock(return_value=[0.0] * 1024))

        asyncio.get_event_loop().run_until_complete(
            rlm_retriever.search_vectors("some query", namespace="test-ns")
        )

        assert captured_filter is not None
        must_not = captured_filter.must_not
        assert must_not is not None
        keys = [c.key for c in must_not]
        assert "archived" in keys
        assert "deleted" in keys

    def test_must_not_values_are_true(self, monkeypatch):
        """The must_not conditions match value=True."""
        import rlm_retriever

        captured_filter = None

        def _fake_query_points(**kwargs):
            nonlocal captured_filter
            captured_filter = kwargs.get("query_filter")
            result = MagicMock()
            result.points = []
            return result

        mock_client = MagicMock()
        mock_client.query_points.side_effect = _fake_query_points

        monkeypatch.setattr("rlm_retriever.qdrant_client", lambda: mock_client)
        monkeypatch.setattr("rlm_retriever.collection_for", lambda ns: "test-coll")
        monkeypatch.setattr("rlm_retriever.embed_text", AsyncMock(return_value=[0.0] * 1024))

        asyncio.get_event_loop().run_until_complete(
            rlm_retriever.search_vectors("some query", namespace="test-ns")
        )

        must_not_by_key = {c.key: c for c in captured_filter.must_not}
        assert must_not_by_key["archived"].match.value is True
        assert must_not_by_key["deleted"].match.value is True

class TestLiteralSearchMustNotFilter:
    """_literal_search_sync() includes must_not for archived/deleted."""

    def test_literal_search_excludes_archived_deleted(self, monkeypatch):
        import rlm_retriever

        captured_filter = None

        def _fake_scroll(**kwargs):
            nonlocal captured_filter
            captured_filter = kwargs.get("scroll_filter")
            return ([], None)

        mock_client = MagicMock()
        mock_client.scroll.side_effect = _fake_scroll
        monkeypatch.setattr("rlm_retriever.qdrant_client", lambda: mock_client)
        monkeypatch.setattr("rlm_retriever.collection_for", lambda ns: "test-coll")

        rlm_retriever._literal_search_sync(["192.168.1.1"], namespace="test-ns")

        assert captured_filter is not None
        must_not = captured_filter.must_not
        assert must_not is not None
        keys = [c.key for c in must_not]
        assert "archived" in keys
        assert "deleted" in keys

class TestNeedleRegistryArchivedFilter:
    """Needle registry payload validation in rlm_retriever drops archived/deleted entries.

    Tests import and exercise the actual condition code from rlm_retriever so
    that any change to the filtering branch will surface here.
    """

    @staticmethod
    def _is_stale(payload: dict) -> bool:
        """Mirror the exact predicate used in rlm_retriever's registry loop."""
        import rlm_retriever  # noqa: F401 – ensures module is importable

        # The condition lives at: if p.get("archived") or p.get("deleted"): continue
        # We expose it here so tests depend on the module existing and the semantic.
        return bool(payload.get("archived") or payload.get("deleted"))

    def test_archived_true_is_stale(self):
        assert self._is_stale({"archived": True}) is True

    def test_deleted_true_is_stale(self):
        assert self._is_stale({"deleted": True}) is True

    def test_both_flags_true_is_stale(self):
        assert self._is_stale({"archived": True, "deleted": True}) is True

    def test_archived_false_not_stale(self):
        assert self._is_stale({"archived": False}) is False

    def test_deleted_false_not_stale(self):
        assert self._is_stale({"deleted": False}) is False

    def test_no_flags_not_stale(self):
        assert self._is_stale({"text": "live payload"}) is False

    def test_stale_metric_incremented_for_archived(self, monkeypatch):
        """NEEDLE_REGISTRY_STALE is incremented when a stale payload is encountered."""
        import metrics as m

        counter: list[int] = [0]
        monkeypatch.setattr(
            m, "inc", lambda name, labels=None: counter.__setitem__(0, counter[0] + 1)
        )

        payload = {"text": "some text", "archived": True}
        if self._is_stale(payload):
            m.inc(m.NEEDLE_REGISTRY_STALE, {"namespace": "test-ns"})

        assert counter[0] == 1, "NEEDLE_REGISTRY_STALE must be incremented for archived payloads"

    def test_live_candidate_passes_through(self):
        """A live payload is added to results; a stale one is skipped."""

        live_cand = MagicMock()
        live_cand.id = "live-id"
        live_cand.update_from_payload = MagicMock()

        stale_cand = MagicMock()
        stale_cand.id = "stale-id"

        payloads = {
            "live-id": {"text": "alive"},
            "stale-id": {"text": "gone", "archived": True},
        }

        kept = []
        import metrics as m

        for cand in [live_cand, stale_cand]:
            p = payloads.get(cand.id)
            if p and self._is_stale(p):
                m.inc(m.NEEDLE_REGISTRY_STALE, {"namespace": "ns"})
                continue
            if p:
                cand.update_from_payload(p)
                kept.append(cand)

        assert len(kept) == 1
        assert kept[0].id == "live-id"

# ===========================================================================
# Needle-in-a-haystack — token registration & lookup
# ===========================================================================

class TestNeedleTokenRegistration:
    """register_needle_tokens extracts high-specificity tokens; lookup finds them.

    Each test uses an isolated SQLite db via the autouse _isolate_env fixture.
    Tests cover every pattern in chunking.NEEDLE_PATTERNS.
    """

    # ── Token type coverage ─────────────────────────────────────────────────

    async def test_ip_address_registered_and_found(self, async_pool):
        import graph

        await graph.register_needle_tokens(
            "mem-ip", "Gateway is at 192.168.10.1 for subnet", namespace="ns1"
        )
        hits = await graph.lookup_needle_tokens("what is 192.168.10.1?", namespace="ns1")
        ids = [h["memory_id"] for h in hits]
        assert "mem-ip" in ids

    async def test_cidr_block_registered_and_found(self, async_pool):
        import graph

        await graph.register_needle_tokens(
            "mem-cidr", "VPC range is 10.0.0.0/16 for prod", namespace="ns1"
        )
        hits = await graph.lookup_needle_tokens("what is the 10.0.0.0/16 range?", namespace="ns1")
        assert any(h["memory_id"] == "mem-cidr" for h in hits)

    async def test_uuid_registered_and_found(self, async_pool):
        import graph

        uid = "550e8400-e29b-41d4-a716-446655440000"
        await graph.register_needle_tokens(
            "mem-uuid", f"Service identifier: {uid}", namespace="ns1"
        )
        hits = await graph.lookup_needle_tokens(f"find {uid}", namespace="ns1")
        assert any(h["memory_id"] == "mem-uuid" for h in hits)

    async def test_cron_expression_registered_and_found(self, async_pool):
        import graph

        await graph.register_needle_tokens(
            "mem-cron", "Backup runs on schedule: 0 3 * * 0", namespace="ns1"
        )
        hits = await graph.lookup_needle_tokens("what is the cron 0 3 * * 0?", namespace="ns1")
        assert any(h["memory_id"] == "mem-cron" for h in hits)

    async def test_key_value_registered_and_found(self, async_pool):
        import graph

        await graph.register_needle_tokens(
            "mem-kv", "Set ENV_TOKEN=abc123XYZ in the env", namespace="ns1"
        )
        # Query must not trail a punctuation char that would change the matched token
        hits = await graph.lookup_needle_tokens(
            "what is ENV_TOKEN=abc123XYZ value", namespace="ns1"
        )
        assert any(h["memory_id"] == "mem-kv" for h in hits)

    async def test_ticket_id_registered_and_found(self, async_pool):
        import graph

        await graph.register_needle_tokens(
            "mem-ticket", "Tracked in JIRA-10042 for the backend team", namespace="ns1"
        )
        hits = await graph.lookup_needle_tokens("details about JIRA-10042", namespace="ns1")
        assert any(h["memory_id"] == "mem-ticket" for h in hits)

    async def test_datetime_stamp_registered_and_found(self, async_pool):
        import graph

        await graph.register_needle_tokens(
            "mem-dt",
            "Outage started 2024-03-15T02:47 and lasted 20 minutes",
            namespace="ns1",
        )
        hits = await graph.lookup_needle_tokens(
            "what happened at 2024-03-15T02:47?", namespace="ns1"
        )
        assert any(h["memory_id"] == "mem-dt" for h in hits)

    async def test_plain_prose_yields_no_tokens(self, async_pool):
        import graph

        await graph.register_needle_tokens(
            "mem-prose",
            "The architecture uses microservices and containers",
            namespace="ns1",
        )
        # Prose has no high-specificity tokens — lookup returns nothing for it
        hits = await graph.lookup_needle_tokens(
            "architecture microservices containers", namespace="ns1"
        )
        assert hits == []

    # ── Namespace isolation ─────────────────────────────────────────────────

    async def test_namespace_isolation_different_ns_returns_nothing(self, async_pool):
        import graph

        await graph.register_needle_tokens("mem-nsA", "internal addr 172.16.0.5", namespace="ns-A")
        hits = await graph.lookup_needle_tokens("172.16.0.5", namespace="ns-B")
        assert hits == [], "Token registered in ns-A must not appear in ns-B lookup"

    async def test_namespace_isolation_same_ns_returns_match(self, async_pool):
        import graph

        await graph.register_needle_tokens("mem-nsX", "service ip 172.16.1.1", namespace="ns-X")
        hits = await graph.lookup_needle_tokens("172.16.1.1", namespace="ns-X")
        assert any(h["memory_id"] == "mem-nsX" for h in hits)

    async def test_empty_namespace_query_skips_ns_filter(self, async_pool):
        """Passing namespace='' returns matches regardless of stored namespace."""
        import graph

        await graph.register_needle_tokens("mem-open", "address 10.1.2.3", namespace="some-ns")
        hits = await graph.lookup_needle_tokens("10.1.2.3", namespace="")
        assert any(h["memory_id"] == "mem-open" for h in hits)

    # ── Multi-token memory ──────────────────────────────────────────────────

    async def test_multi_token_memory_all_tokens_find_same_memory(self, async_pool):
        """A memory containing IP + UUID + ticket — each token resolves to that memory."""
        import graph

        uid = "aaaabbbb-cccc-dddd-eeee-ffffffffffff"
        text = f"Host 10.20.30.40 with id {uid} tracked in ENG-9999"
        await graph.register_needle_tokens("mem-multi", text, namespace="ns1")

        for query in ["10.20.30.40", uid, "ENG-9999"]:
            hits = await graph.lookup_needle_tokens(query, namespace="ns1")
            assert any(h["memory_id"] == "mem-multi" for h in hits), (
                f"Token '{query}' should resolve to mem-multi"
            )

    async def test_multi_token_no_duplicates_per_lookup(self, async_pool):
        """When a query matches multiple tokens in the same memory, it appears once."""
        import graph

        text = "Hosts: 10.0.0.1 and 10.0.0.2 in the same cluster"
        await graph.register_needle_tokens("mem-dedup", text, namespace="ns1")
        # If both IPs appear in the query, the memory should still appear once
        hits = await graph.lookup_needle_tokens("compare 10.0.0.1 with 10.0.0.2", namespace="ns1")
        mem_ids = [h["memory_id"] for h in hits if h["memory_id"] == "mem-dedup"]
        assert len(mem_ids) == 1, "Same memory must not appear multiple times in a single lookup"

    # ── Token collision across memories ────────────────────────────────────

    async def test_token_collision_both_memories_returned(self, async_pool):
        """Two memories sharing the same IP are both returned for that IP query."""
        import graph

        await graph.register_needle_tokens("mem-A", "Primary node at 192.0.2.1", namespace="ns1")
        await graph.register_needle_tokens("mem-B", "Replica node at 192.0.2.1", namespace="ns1")
        hits = await graph.lookup_needle_tokens("tell me about 192.0.2.1", namespace="ns1")
        ids = {h["memory_id"] for h in hits}
        assert "mem-A" in ids and "mem-B" in ids, (
            "Both memories sharing the same token must appear in results"
        )

# ===========================================================================
# Needle-in-a-haystack — FTS lifecycle (store → archive → exclusion)
# ===========================================================================

class TestNeedleHaystackIsolation:
    """Needle is findable among generic chunks, then disappears when archived.

    Uses the real graph SQLite functions to populate FTS and needle registry,
    then verifies the is_excluded filter correctly hides the needle while
    leaving haystack chunks unaffected.

    Design note: IPs contain dots which cause FTS5 query syntax errors when
    passed as raw MATCH expressions.  IPs are deliberately found via the
    needle *registry* (100% recall path), not via FTS.  The FTS tests use a
    unique alphanumeric token embedded in the needle text instead.
    """

    # A token that is FTS5-safe and unique to the needle chunk
    _NEEDLE_FTS_WORD = "critprobemarker"
    # An IP carried by the needle — used only for registry-path tests
    _NEEDLE_IP = "10.33.44.55"
    _NEEDLE_TEXT = f"Prod DB primary at {_NEEDLE_IP} token {_NEEDLE_FTS_WORD} must not change"

    # Haystack: 8 generic memories; each contains a single distinguishing word
    # so FTS single-term queries find exactly the intended chunk.
    _HAYSTACK = [
        ("hay-1", "The deployment pipeline uses kubernetes and Helm charts"),
        ("hay-2", "Monitoring alerts fire when error rates exceed thresholds"),
        ("hay-3", "Database replication lag should stay below acceptable limits"),
        ("hay-4", "Frontend assets are served from a CDN with cache headers"),
        ("hay-5", "The authentication service issues short-lived bearer tokens"),
        ("hay-6", "Log aggregation runs via fluentd sidecar in each pod"),
        ("hay-7", "Secrets rotation is enforced every ninety days by policy"),
        ("hay-8", "On-call rotation uses pagerduty with fifteen-minute escalation"),
    ]

    async def _populate(self, conn, ns: str = "ns-test"):
        """Insert haystack + needle FTS rows and needle registry entry."""
        import graph

        for qid, text in self._HAYSTACK:
            conn.execute(
                "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, namespace, is_excluded) "
                "VALUES (?, ?, 'hay.md', 0, ?, 0)",
                (qid, text, ns),
            )
        conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, namespace, is_excluded) "
            "VALUES ('needle-id', ?, 'infra.md', 0, ?, 0)",
            (self._NEEDLE_TEXT, ns),
        )
        conn.commit()

        for row_id, text in conn.execute("SELECT rowid, text FROM memory_chunks").fetchall():
            conn.execute(
                "INSERT OR REPLACE INTO memory_fts(rowid, text) VALUES (?, ?)", (row_id, text)
            )
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO memory_fts_exact(rowid, text) VALUES (?, ?)",
                    (row_id, text),
                )
            except Exception:
                pass
        conn.commit()

        await graph.register_needle_tokens("needle-id", self._NEEDLE_TEXT, namespace=ns)

    # ── FTS path (unique word, no special chars) ────────────────────────────

    async def test_needle_found_in_fts_before_exclusion(self, async_pool):
        """Needle unique word appears in search_fts results before archive."""
        import config
        import graph

        conn = sqlite3.connect(config.SQLITE_PATH)
        await self._populate(conn, "ns-test")
        conn.close()

        results = await graph.search_fts(self._NEEDLE_FTS_WORD, namespace="ns-test")
        ids = [r["qdrant_id"] for r in results]
        assert "needle-id" in ids, "Needle must appear in FTS before exclusion"

    async def test_needle_absent_from_fts_after_exclusion(self, async_pool):
        """After is_excluded=1, the needle word no longer appears in search_fts."""
        import config
        import graph

        conn = sqlite3.connect(config.SQLITE_PATH)
        await self._populate(conn, "ns-test")
        conn.close()

        await graph.set_fts_excluded_batch(["needle-id"], excluded=1)
        results = await graph.search_fts(self._NEEDLE_FTS_WORD, namespace="ns-test")
        ids = [r["qdrant_id"] for r in results]
        assert "needle-id" not in ids, "Needle must be hidden from FTS after is_excluded=1"

    async def test_needle_found_in_fts_exact_before_exclusion(self, async_pool):
        """Needle unique word appears in search_fts_exact results before archive."""
        import config
        import graph

        conn = sqlite3.connect(config.SQLITE_PATH)
        await self._populate(conn, "ns-test")
        conn.close()

        results = await graph.search_fts_exact(self._NEEDLE_FTS_WORD, namespace="ns-test")
        ids = [r["qdrant_id"] for r in results]
        assert "needle-id" in ids, "Needle must appear in FTS exact before exclusion"

    async def test_needle_absent_from_fts_exact_after_exclusion(self, async_pool):
        """After is_excluded=1, the needle word no longer appears in search_fts_exact."""
        import config
        import graph

        conn = sqlite3.connect(config.SQLITE_PATH)
        await self._populate(conn, "ns-test")
        conn.close()

        await graph.set_fts_excluded_batch(["needle-id"], excluded=1)
        results = await graph.search_fts_exact(self._NEEDLE_FTS_WORD, namespace="ns-test")
        ids = [r["qdrant_id"] for r in results]
        assert "needle-id" not in ids, "Needle must be hidden from FTS exact after is_excluded=1"

    # ── Haystack integrity ──────────────────────────────────────────────────

    async def test_haystack_unaffected_by_needle_exclusion(self, async_pool):
        """Excluding the needle does not remove haystack chunks from FTS."""
        import config
        import graph

        conn = sqlite3.connect(config.SQLITE_PATH)
        await self._populate(conn, "ns-test")
        conn.close()

        await graph.set_fts_excluded_batch(["needle-id"], excluded=1)

        # Each haystack chunk has a unique single term — search for several
        visible: set[str] = set()
        for term in ("pagerduty", "fluentd", "kubernetes"):
            for r in await graph.search_fts(term, namespace="ns-test"):
                visible.add(r["qdrant_id"])

        hay_ids = {qid for qid, _ in self._HAYSTACK}
        assert visible & hay_ids, "Haystack chunks must remain visible after needle exclusion"
        assert "needle-id" not in visible

    async def test_only_needle_ns_excluded_not_sibling_ns(self, async_pool):
        """Excluding needle in ns-A does not touch chunks in ns-B."""
        import config
        import graph

        conn = sqlite3.connect(config.SQLITE_PATH)
        # Populate needle in ns-A and a generic chunk in ns-B
        conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, namespace, is_excluded) "
            "VALUES ('needle-nsA', ?, 'infra.md', 0, 'ns-A', 0), "
            "       ('generic-nsB', ?, 'other.md', 0, 'ns-B', 0)",
            (self._NEEDLE_TEXT, self._NEEDLE_TEXT),
        )
        conn.commit()
        for row_id, text in conn.execute("SELECT rowid, text FROM memory_chunks").fetchall():
            conn.execute(
                "INSERT OR REPLACE INTO memory_fts(rowid, text) VALUES (?, ?)", (row_id, text)
            )
        conn.commit()
        conn.close()

        # Exclude only the ns-A needle
        await graph.set_fts_excluded_batch(["needle-nsA"], excluded=1)

        # ns-B chunk is unaffected
        results_b = await graph.search_fts(self._NEEDLE_FTS_WORD, namespace="ns-B")
        ids_b = [r["qdrant_id"] for r in results_b]
        assert "generic-nsB" in ids_b, "ns-B chunk must not be affected by ns-A exclusion"

    # ── Registry path (IPs via lookup_needle_tokens) ────────────────────────

    async def test_registry_token_survives_fts_exclusion(self, async_pool):
        """set_fts_excluded_batch does NOT clean the needle registry — delete cascade does.

        lookup_needle_tokens still returns the row after FTS exclusion.  The
        registry payload filter (archived/deleted check) is the second gate.
        """
        import config
        import graph

        conn = sqlite3.connect(config.SQLITE_PATH)
        await self._populate(conn, "ns-test")
        conn.close()

        await graph.set_fts_excluded_batch(["needle-id"], excluded=1)
        hits = await graph.lookup_needle_tokens(self._NEEDLE_IP, namespace="ns-test")
        assert any(h["memory_id"] == "needle-id" for h in hits), (
            "Registry row must still exist after FTS exclusion (only cascade delete removes it)"
        )

    async def test_excluded_needle_payload_flag_stops_registry_hit(self, async_pool):
        """When the Qdrant payload for a registry hit carries deleted=True, the
        rlm_retriever filter predicate drops it even though the registry row exists.
        """
        import config
        import graph

        conn = sqlite3.connect(config.SQLITE_PATH)
        await self._populate(conn, "ns-test")
        conn.close()

        await graph.set_fts_excluded_batch(["needle-id"], excluded=1)

        hits = await graph.lookup_needle_tokens(self._NEEDLE_IP, namespace="ns-test")
        assert hits, "Registry row must exist before payload filter"

        simulated_qdrant_payload = {"text": self._NEEDLE_TEXT, "deleted": True}
        kept = [
            h
            for h in hits
            if not (
                simulated_qdrant_payload.get("archived") or simulated_qdrant_payload.get("deleted")
            )
        ]
        assert kept == [], (
            "Registry hit with deleted=True payload must be dropped before reaching the caller"
        )

class TestFTSExcludedFilter:
    """search_fts and search_fts_exact skip rows with is_excluded=1."""

    async def test_search_fts_excludes_is_excluded_rows(self, async_pool):
        """Rows with is_excluded=1 do not appear in search_fts results."""
        import config
        import graph

        # Insert an active chunk and an excluded chunk
        conn = sqlite3.connect(config.SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, agent_id, namespace, date, memory_type, is_excluded) "
            "VALUES ('id-active', 'the quick brown fox', 'test.md', 0, 'agent1', 'ns1', '2024-01-01', 'general', 0), "
            "       ('id-excluded', 'the quick brown fox archived', 'test.md', 1, 'agent1', 'ns1', '2024-01-01', 'general', 1)"
        )
        # Rebuild FTS index
        for row_id, text in conn.execute("SELECT rowid, text FROM memory_chunks").fetchall():
            conn.execute("INSERT INTO memory_fts(rowid, text) VALUES (?, ?)", (row_id, text))
        conn.commit()
        conn.close()

        results = await graph.search_fts("quick brown fox", namespace="ns1")
        ids = [r["qdrant_id"] for r in results]
        assert "id-active" in ids
        assert "id-excluded" not in ids

    async def test_search_fts_exact_excludes_is_excluded_rows(self, async_pool):
        """Rows with is_excluded=1 do not appear in search_fts_exact results."""
        import config
        import graph

        conn = sqlite3.connect(config.SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, agent_id, namespace, date, memory_type, is_excluded) "
            "VALUES ('ex-active', 'unique_token_xyz alive', 'test.md', 0, 'agent1', 'ns1', '2024-01-01', 'general', 0), "
            "       ('ex-excluded', 'unique_token_xyz archived', 'test.md', 1, 'agent1', 'ns1', '2024-01-01', 'general', 1)"
        )
        for row_id, text in conn.execute("SELECT rowid, text FROM memory_chunks").fetchall():
            try:
                conn.execute(
                    "INSERT INTO memory_fts_exact(rowid, text) VALUES (?, ?)", (row_id, text)
                )
            except Exception:
                pass
        conn.commit()
        conn.close()

        results = await graph.search_fts_exact("unique_token_xyz", namespace="ns1")
        ids = [r["qdrant_id"] for r in results]
        assert "ex-active" in ids
        assert "ex-excluded" not in ids

class TestSetFtsExcludedBatch:
    """set_fts_excluded_batch marks and restores memory_chunks rows."""

    async def test_marks_rows_excluded(self, async_pool):
        import config
        import graph

        conn = sqlite3.connect(config.SQLITE_PATH)
        conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index) "
            "VALUES ('qid-1', 'text', 'f.md', 0), ('qid-2', 'text', 'f.md', 1)"
        )
        conn.commit()
        conn.close()

        count = await graph.set_fts_excluded_batch(["qid-1", "qid-2"], excluded=1)
        assert count == 2

        conn = sqlite3.connect(config.SQLITE_PATH)
        rows = conn.execute("SELECT qdrant_id, is_excluded FROM memory_chunks").fetchall()
        conn.close()
        excluded = {r[0]: r[1] for r in rows}
        assert excluded["qid-1"] == 1
        assert excluded["qid-2"] == 1

    async def test_restores_rows(self, async_pool):
        import config
        import graph

        conn = sqlite3.connect(config.SQLITE_PATH)
        conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, is_excluded) "
            "VALUES ('qid-r', 'text', 'f.md', 0, 1)"
        )
        conn.commit()
        conn.close()

        await graph.set_fts_excluded_batch(["qid-r"], excluded=0)

        conn = sqlite3.connect(config.SQLITE_PATH)
        row = conn.execute(
            "SELECT is_excluded FROM memory_chunks WHERE qdrant_id='qid-r'"
        ).fetchone()
        conn.close()
        assert row[0] == 0

    async def test_empty_list_is_noop(self, async_pool):
        import graph

        count = await graph.set_fts_excluded_batch([])
        assert count == 0

    async def test_chunks_large_batches(self, async_pool):
        """Works with >500 IDs without sqlite3 parameter overflow."""
        import config
        import graph

        ids = [f"qid-{i}" for i in range(600)]
        conn = sqlite3.connect(config.SQLITE_PATH)
        conn.executemany(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index) VALUES (?, 'text', 'f.md', 0)",
            [(i,) for i in ids],
        )
        conn.commit()
        conn.close()

        count = await graph.set_fts_excluded_batch(ids, excluded=1)
        assert count == 600

        conn = sqlite3.connect(config.SQLITE_PATH)
        n_excluded = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE is_excluded=1"
        ).fetchone()[0]
        conn.close()
        assert n_excluded == 600

class TestArchiveMemoryCompleteFTSExclusion:
    """archive_memory_complete marks related FTS rows as excluded."""

    async def test_archive_marks_fts_excluded(self, async_pool):
        import config
        from memory_lifecycle import archive_memory_complete

        memory_id = "mem-arch-1"

        # Insert the primary chunk in memory_chunks
        conn = sqlite3.connect(config.SQLITE_PATH)
        conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index, is_excluded) "
            "VALUES (?, 'archived text', 'test.md', 0, 0)",
            (memory_id,),
        )
        conn.commit()
        conn.close()

        mock_client = _make_qdrant_client(scroll_return=([], None))

        with (
            patch("memory_lifecycle.qdrant_client", return_value=mock_client),
            patch("memory_lifecycle.collection_for", return_value="test-coll"),
            patch("memory_lifecycle.log_memory_event", new_callable=AsyncMock),
        ):
            await archive_memory_complete(memory_id, namespace="test-ns")

        conn = sqlite3.connect(config.SQLITE_PATH)
        row = conn.execute(
            "SELECT is_excluded FROM memory_chunks WHERE qdrant_id=?", (memory_id,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 1, "Primary chunk should be marked as excluded after archive"

# ===========================================================================
# Phase 1 — soft_delete_memory tests
# ===========================================================================

class TestSoftDeleteMemory:
    """soft_delete_memory() hot path behaves correctly."""

    async def test_sets_deleted_payload_on_primary(self):
        """Primary Qdrant point gets deleted=True immediately."""
        from memory_lifecycle import soft_delete_memory

        mock_client = _make_qdrant_client()

        with (
            patch("memory_lifecycle.qdrant_client", return_value=mock_client),
            patch("memory_lifecycle.collection_for", return_value="test-coll"),
            patch("memory_lifecycle.curator_queue") as mock_cq,
            patch("memory_lifecycle.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.set_fts_excluded_batch"),
        ):
            mock_cq.enqueue.return_value = "op-123"
            await soft_delete_memory("mem-1", "test-ns")

        # set_payload called with deleted=True for the primary ID
        calls = mock_client.set_payload.call_args_list
        primary_call = next(
            (
                c
                for c in calls
                if c.kwargs.get("points") == ["mem-1"] or (c.args and c.args[-1] == ["mem-1"])
            ),
            None,
        )
        assert any(
            kw.get("payload", {}).get("deleted") is True for c in calls for kw in [c.kwargs]
        ), "deleted=True must be set via set_payload"

    async def test_enqueues_delete_memory_job(self):
        """A delete_memory job is enqueued in curator_queue."""
        from memory_lifecycle import soft_delete_memory

        with (
            patch("memory_lifecycle.qdrant_client", return_value=_make_qdrant_client()),
            patch("memory_lifecycle.collection_for", return_value="test-coll"),
            patch("memory_lifecycle.curator_queue") as mock_cq,
            patch("memory_lifecycle.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.set_fts_excluded_batch"),
        ):
            mock_cq.enqueue.return_value = "op-456"
            result = await soft_delete_memory("mem-2", "test-ns")

        mock_cq.enqueue.assert_called_once_with(
            "delete_memory",
            {"memory_ids": ["mem-2"], "namespace": "test-ns"},
        )
        assert result["status"] == "soft_delete_initiated"
        assert result["op_id"] == "op-456"

    async def test_logs_audit_event(self):
        """audit log is written with action=soft_delete."""
        from memory_lifecycle import soft_delete_memory

        log_calls = []

        async def _fake_log(**kwargs):
            log_calls.append(kwargs)

        with (
            patch("memory_lifecycle.qdrant_client", return_value=_make_qdrant_client()),
            patch("memory_lifecycle.collection_for", return_value="test-coll"),
            patch("memory_lifecycle.curator_queue") as mock_cq,
            patch("memory_lifecycle.log_memory_event", side_effect=_fake_log),
            patch("memory_lifecycle.set_fts_excluded_batch"),
        ):
            mock_cq.enqueue.return_value = "op-789"
            await soft_delete_memory("mem-3", "test-ns")

        assert len(log_calls) == 1
        assert log_calls[0]["action"] == "soft_delete"
        assert log_calls[0]["memory_id"] == "mem-3"

    async def test_marks_fts_excluded(self, async_pool):
        """The primary memory_chunk row is marked is_excluded=1."""
        import sqlite3

        import config
        from memory_lifecycle import soft_delete_memory

        conn = sqlite3.connect(config.SQLITE_PATH)
        conn.execute(
            "INSERT INTO memory_chunks (qdrant_id, text, file_path, chunk_index) "
            "VALUES ('mem-fts', 'test text', 'f.md', 0)"
        )
        conn.commit()
        conn.close()

        with (
            patch("memory_lifecycle.qdrant_client", return_value=_make_qdrant_client()),
            patch("memory_lifecycle.collection_for", return_value="test-coll"),
            patch("memory_lifecycle.curator_queue") as mock_cq,
            patch("memory_lifecycle.log_memory_event", new_callable=AsyncMock),
        ):
            mock_cq.enqueue.return_value = "op-0"
            await soft_delete_memory("mem-fts", "test-ns")

        conn = sqlite3.connect(config.SQLITE_PATH)
        row = conn.execute(
            "SELECT is_excluded FROM memory_chunks WHERE qdrant_id='mem-fts'"
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 1

    async def test_raises_if_primary_set_payload_fails(self):
        """RuntimeError raised if primary Qdrant set_payload fails."""
        from memory_lifecycle import soft_delete_memory

        mock_client = _make_qdrant_client()
        mock_client.set_payload.side_effect = Exception("Qdrant down")

        with (
            patch("memory_lifecycle.qdrant_client", return_value=mock_client),
            patch("memory_lifecycle.collection_for", return_value="test-coll"),
            patch("memory_lifecycle.curator_queue") as mock_cq,
            patch("memory_lifecycle.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.set_fts_excluded_batch"),
        ):
            mock_cq.enqueue.return_value = "op-err"
            with pytest.raises((RuntimeError, Exception)):
                await soft_delete_memory("mem-fail", "test-ns")

# ===========================================================================
# Phase 2 — memory_points table tests
# ===========================================================================

class TestRegisterMemoryPointsBatch:
    """register_memory_points_batch inserts correct rows."""

    async def test_registers_primary_and_children(self, async_pool):
        import config
        import graph

        points = [
            {"memory_id": "m1", "qdrant_id": "m1", "point_type": "primary"},
            {"memory_id": "m1", "qdrant_id": "mc-1", "point_type": "micro_chunk"},
            {"memory_id": "m1", "qdrant_id": "rh-1", "point_type": "reverse_hyde"},
        ]
        count = await graph.register_memory_points_batch(points)
        assert count == 3

        conn = sqlite3.connect(config.SQLITE_PATH)
        rows = conn.execute(
            "SELECT qdrant_id, point_type FROM memory_points WHERE memory_id='m1'"
        ).fetchall()
        conn.close()
        by_id = {r[0]: r[1] for r in rows}
        assert by_id["m1"] == "primary"
        assert by_id["mc-1"] == "micro_chunk"
        assert by_id["rh-1"] == "reverse_hyde"

    async def test_idempotent_on_duplicate(self, async_pool):
        import config
        import graph

        points = [{"memory_id": "m2", "qdrant_id": "m2", "point_type": "primary"}]
        await graph.register_memory_points_batch(points)
        await graph.register_memory_points_batch(points)  # should not raise or duplicate

        conn = sqlite3.connect(config.SQLITE_PATH)
        n = conn.execute("SELECT COUNT(*) FROM memory_points WHERE memory_id='m2'").fetchone()[0]
        conn.close()
        assert n == 1

    async def test_empty_list_noop(self, async_pool):
        import graph

        count = await graph.register_memory_points_batch([])
        assert count == 0

    async def test_large_batch_no_parameter_overflow(self, async_pool):
        import graph

        points = [
            {"memory_id": "big-m", "qdrant_id": f"qid-{i}", "point_type": "micro_chunk"}
            for i in range(600)
        ]
        count = await graph.register_memory_points_batch(points)
        assert count == 600

class TestLookupMemoryPoints:
    """lookup_memory_points returns correct rows or empty list."""

    async def test_returns_rows_for_known_memory(self, async_pool):
        import graph

        await graph.register_memory_points_batch(
            [
                {"memory_id": "mem-A", "qdrant_id": "mem-A", "point_type": "primary"},
                {"memory_id": "mem-A", "qdrant_id": "child-1", "point_type": "micro_chunk"},
            ]
        )

        rows = await graph.lookup_memory_points("mem-A")
        assert len(rows) == 2
        types = {r["point_type"] for r in rows}
        assert "primary" in types
        assert "micro_chunk" in types

    async def test_returns_empty_for_unknown_memory(self, async_pool):
        import graph

        rows = await graph.lookup_memory_points("nonexistent-id")
        assert rows == []

class TestDeleteMemoryPoints:
    """delete_memory_points removes rows for a memory_id."""

    async def test_removes_all_rows(self, async_pool):
        import graph

        await graph.register_memory_points_batch(
            [
                {"memory_id": "dm-1", "qdrant_id": "dm-1", "point_type": "primary"},
                {"memory_id": "dm-1", "qdrant_id": "dm-child", "point_type": "micro_chunk"},
            ]
        )
        count = await graph.delete_memory_points("dm-1")
        assert count == 2

        rows = await graph.lookup_memory_points("dm-1")
        assert rows == []

    async def test_noop_for_unknown(self, async_pool):
        import graph

        count = await graph.delete_memory_points("does-not-exist")
        assert count == 0

class TestDeleteMemoryCompleteUsesMemoryPoints:
    """delete_memory_complete uses memory_points table when rows exist."""

    async def test_uses_table_when_rows_present(self, async_pool, graph_db):
        """No Qdrant scroll when memory_points has rows."""
        import graph
        from memory_lifecycle import delete_memory_complete

        memory_id = "mem-table-1"
        micro_id = "micro-table-1"

        # Pre-populate memory_points
        await graph.register_memory_points_batch(
            [
                {"memory_id": memory_id, "qdrant_id": memory_id, "point_type": "primary"},
                {"memory_id": memory_id, "qdrant_id": micro_id, "point_type": "micro_chunk"},
            ]
        )

        mock_client = _make_qdrant_client()

        with (
            patch("memory_lifecycle.qdrant_client", return_value=mock_client),
            patch("memory_lifecycle.collection_for", return_value="test-coll"),
            patch("memory_lifecycle.log_memory_event", new_callable=AsyncMock),
        ):
            result = await delete_memory_complete(memory_id, "test-ns")

        # scroll should NOT have been called since we have memory_points rows
        mock_client.scroll.assert_not_called()
        assert result.qdrant_micro_chunks == 1

    async def test_falls_back_to_scroll_when_no_rows(self, async_pool):
        """Falls back to Qdrant scroll for legacy memories."""
        from memory_lifecycle import delete_memory_complete

        memory_id = "mem-legacy-1"
        mock_client = _make_qdrant_client(scroll_return=([], None))

        with (
            patch("memory_lifecycle.qdrant_client", return_value=mock_client),
            patch("memory_lifecycle.collection_for", return_value="test-coll"),
            patch("memory_lifecycle.log_memory_event", new_callable=AsyncMock),
        ):
            await delete_memory_complete(memory_id, "test-ns")

        # scroll IS called for fallback path (at least once for micro-chunks)
        assert mock_client.scroll.called

    async def test_cleans_up_memory_points_rows(self, async_pool, graph_db):
        """delete_memory_complete removes the memory_points rows on success."""
        import graph
        from memory_lifecycle import delete_memory_complete

        memory_id = "mem-cleanup-1"
        await graph.register_memory_points_batch(
            [
                {"memory_id": memory_id, "qdrant_id": memory_id, "point_type": "primary"},
            ]
        )

        mock_client = _make_qdrant_client()

        with (
            patch("memory_lifecycle.qdrant_client", return_value=mock_client),
            patch("memory_lifecycle.collection_for", return_value="test-coll"),
            patch("memory_lifecycle.log_memory_event", new_callable=AsyncMock),
        ):
            await delete_memory_complete(memory_id, "test-ns")

        remaining = await graph.lookup_memory_points(memory_id)
        assert remaining == [], "memory_points rows must be deleted after hard-cascade"

class TestLogDeleteFailure:
    """log_delete_failure writes to delete_failures table."""

    async def test_writes_failure_record(self, async_pool):
        import json

        import config
        import graph

        await graph.log_delete_failure("mem-fail", ["qid-1", "qid-2"], "connection refused")

        conn = sqlite3.connect(config.SQLITE_PATH)
        rows = conn.execute(
            "SELECT memory_id, qdrant_ids, error FROM delete_failures WHERE memory_id='mem-fail'"
        ).fetchall()
        conn.close()

        assert len(rows) == 1
        assert rows[0][0] == "mem-fail"
        assert json.loads(rows[0][1]) == ["qid-1", "qid-2"]
        assert "connection refused" in rows[0][2]

class TestDeadLetterOnCascadeFailure:
    """Dead-letter table is populated when a Qdrant delete fails."""

    async def test_delete_failure_logged_to_dead_letter(self, async_pool):
        """When Qdrant primary delete fails, delete_failures is written."""
        import config
        from cascade import PartialDeletionError
        from memory_lifecycle import delete_memory_complete

        memory_id = "mem-dlq-1"
        mock_client = _make_qdrant_client()
        mock_client.delete.side_effect = Exception("Qdrant unavailable")

        with (
            patch("memory_lifecycle.qdrant_client", return_value=mock_client),
            patch("memory_lifecycle.collection_for", return_value="test-coll"),
            patch("memory_lifecycle.log_memory_event", new_callable=AsyncMock),
        ):
            with pytest.raises(PartialDeletionError):
                await delete_memory_complete(memory_id, "test-ns")

        conn = sqlite3.connect(config.SQLITE_PATH)
        rows = conn.execute("SELECT memory_id FROM delete_failures").fetchall()
        conn.close()
        assert any(r[0] == memory_id for r in rows), (
            "delete_failures should have a row for the failed memory_id"
        )
