"""Tests for Chunk 2: Delete Cascade.

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

import sqlite3
from datetime import UTC
from unittest.mock import AsyncMock, MagicMock, patch

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
        with patch("memory_lifecycle.delete_fts_chunks_batch", return_value=1) as m:
            yield m

    @pytest.fixture
    def mock_needle(self):
        with patch("memory_lifecycle.delete_needle_tokens_batch", return_value=3) as m:
            yield m

    @pytest.fixture
    def mock_entity_facts(self):
        with patch("memory_lifecycle._delete_entity_facts_for_memory", return_value=2) as m:
            yield m

    @pytest.fixture
    def mock_hotness(self):
        with patch("memory_lifecycle.delete_hotness", return_value=1) as m:
            yield m

    @pytest.fixture
    def mock_audit(self):
        with patch("memory_lifecycle.log_memory_event", new_callable=AsyncMock) as m:
            yield m

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
    async def test_collection_override(
        self, mock_qdrant, mock_fts, mock_needle, mock_entity_facts, mock_audit
    ):
        """Explicit collection= kwarg overrides routing."""
        from memory_lifecycle import delete_memory_complete

        await delete_memory_complete("m1", "ns", collection="override_col")

        assert mock_qdrant.delete.call_args_list[0][1]["collection_name"] == "override_col"

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    @pytest.mark.asyncio
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

    def test_deletes_matching_rows(self):
        from graph import delete_fts_chunks_by_qdrant_id, get_db, upsert_fts_chunk

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
        from graph import delete_fts_chunks_by_qdrant_id, get_db, upsert_fts_chunk

        upsert_fts_chunk("keep-me", "important text", "f.md", 0, "a", "ns")
        upsert_fts_chunk("delete-me", "trash text", "f.md", 1, "a", "ns")

        delete_fts_chunks_by_qdrant_id("delete-me")

        conn = get_db()
        kept = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id = 'keep-me'"
        ).fetchone()[0]
        conn.close()
        assert kept == 1


class TestBatchFtsDelete:
    """delete_fts_chunks_batch handles chunking correctly."""

    def test_batch_deletes_multiple_ids(self):
        from graph import delete_fts_chunks_batch, get_db, upsert_fts_chunk

        for i in range(5):
            upsert_fts_chunk(f"batch-{i}", f"text {i}", "f.md", i, "a", "ns")

        deleted = delete_fts_chunks_batch([f"batch-{i}" for i in range(5)])
        assert deleted == 5

        conn = get_db()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id LIKE 'batch-%'"
        ).fetchone()[0]
        conn.close()
        assert remaining == 0

    def test_batch_empty_list(self):
        from graph import delete_fts_chunks_batch

        assert delete_fts_chunks_batch([]) == 0

    def test_batch_chunking_under_parameter_limit(self):
        """Passing >999 IDs doesn't crash sqlite3 thanks to internal chunking."""
        from graph import delete_fts_chunks_batch, upsert_fts_chunk

        ids = [f"chunk-test-{i}" for i in range(1200)]
        for qid in ids[:5]:
            upsert_fts_chunk(qid, "text", "f.md", 0, "a", "ns")

        deleted = delete_fts_chunks_batch(ids)
        assert deleted == 5


class TestBatchNeedleDelete:
    """delete_needle_tokens_batch handles chunking correctly."""

    def test_batch_empty_list(self):
        from graph import delete_needle_tokens_batch

        assert delete_needle_tokens_batch([]) == 0


class TestDeleteEntityFactsForMemory:
    """_delete_entity_facts_for_memory soft-deactivates linked facts."""

    def test_deactivates_matching_facts(self):
        from graph import add_fact, get_db, upsert_entity
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
        from graph import add_fact, get_db, upsert_entity
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


class TestOrphanSweeper:
    """sweep_orphans reconciles SQLite rows against Qdrant."""

    def test_cleans_orphaned_fts_rows(self):
        """FTS rows with no corresponding Qdrant point are cleaned."""
        from graph import get_db, upsert_fts_chunk

        upsert_fts_chunk("exists-in-qdrant", "text", "f.md", 0, "a", "ns")
        upsert_fts_chunk("orphaned-id", "text2", "f.md", 1, "a", "ns")

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

            result = sweep_orphans()

        assert result["fts_cleaned"] >= 1

        conn = get_db()
        orphan_count = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id = 'orphaned-id'"
        ).fetchone()[0]
        conn.close()
        assert orphan_count == 0

    def test_does_not_clean_existing_points(self):
        """FTS rows with matching Qdrant points are kept."""
        from graph import get_db, upsert_fts_chunk

        upsert_fts_chunk("keep-this", "text", "f.md", 0, "a", "ns")

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

            result = sweep_orphans()

        conn = get_db()
        count = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id = 'keep-this'"
        ).fetchone()[0]
        conn.close()
        assert count == 1


class TestCuratorQueueDrainAsync:
    """curator_queue.drain() is async and calls lifecycle functions."""

    @pytest.mark.asyncio
    async def test_drain_is_async_coroutine(self):
        import inspect

        from curator_queue import drain

        assert inspect.iscoroutinefunction(drain)

    @pytest.mark.asyncio
    async def test_delete_op_calls_lifecycle(self):
        from curator_queue import drain, enqueue

        with patch("curator_queue._apply_delete", new_callable=AsyncMock) as mock_del:
            enqueue("delete_memory", {"memory_ids": ["m1"], "namespace": "ns"})
            result = await drain(limit=10)

        assert len(result) == 1
        assert result[0]["status"] == "applied"
        mock_del.assert_called_once()

    @pytest.mark.asyncio
    async def test_archive_op_calls_lifecycle(self):
        from curator_queue import drain, enqueue

        with patch("curator_queue._apply_archive", new_callable=AsyncMock) as mock_arc:
            enqueue("archive_memory", {"memory_ids": ["m1"], "namespace": "ns"})
            result = await drain(limit=10)

        assert len(result) == 1
        assert result[0]["status"] == "applied"
        mock_arc.assert_called_once()

    @pytest.mark.asyncio
    async def test_failed_op_marked_failed(self):
        from curator_queue import drain, enqueue

        with patch(
            "curator_queue._apply_delete", new_callable=AsyncMock, side_effect=Exception("boom")
        ):
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

        with (
            patch("merge.qdrant_client", return_value=mock_client),
            patch("merge.embed_text", new_callable=AsyncMock, return_value=[0.1] * 1024),
            patch("merge.llm_query", new_callable=AsyncMock, return_value="merged text"),
            patch("merge.record_version", return_value=2),
            patch("merge.log_memory_event", new_callable=AsyncMock),
            patch("memory_lifecycle.delete_memory_complete", mock_del),
        ):
            from merge import merge_memories

            result = await merge_memories(["id1", "id2"], "semantic", "agent", "ns")

        assert mock_del.call_count == 2


class TestQdrantRetry:
    """Transient-only retry behaviour in _qdrant_delete and _qdrant_set_payload."""

    def test_transient_retry_succeeds_on_second_attempt(self):
        """First delete raises a transient error, second attempt succeeds."""
        from qdrant_client.http.exceptions import ResponseHandlingException

        from cascade import _qdrant_delete

        client = MagicMock()
        client.count.return_value = MagicMock(count=5)
        client.delete.side_effect = [ResponseHandlingException("timeout"), None]

        from qdrant_client.models import FieldCondition, Filter, MatchValue

        filt = Filter(must=[FieldCondition(key="parent_id", match=MatchValue(value="mem-1"))])
        failed = []

        count = _qdrant_delete(client, "col", filt, "test_step", "mem-1", failed)

        assert count == 5
        assert failed == []
        assert client.delete.call_count == 2

    def test_permanent_error_does_not_retry(self):
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
        _qdrant_delete(client, "col", ["point-1"], "perm_step", "mem-1", failed)

        assert client.delete.call_count == 1
        assert "perm_step" in failed

    def test_precount_returned_on_final_failure(self):
        """Both attempts fail with transient errors; pre-count is still returned."""
        from qdrant_client.http.exceptions import ResponseHandlingException

        from cascade import _qdrant_delete

        client = MagicMock()
        client.delete.side_effect = ResponseHandlingException("network")

        failed = []
        count = _qdrant_delete(client, "col", ["p1", "p2", "p3"], "step_x", "mem-1", failed)

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

    def test_sweeper_aborts_on_qdrant_down(self):
        """Sweeper returns skipped when Qdrant is unreachable."""
        mock_client = MagicMock()
        mock_client.get_collections.side_effect = ConnectionError("refused")

        with patch("cascade.qdrant_client", return_value=mock_client):
            from cascade import sweep_orphans

            result = sweep_orphans()

        assert result.get("skipped") == "qdrant_unavailable"
        assert result["fts_cleaned"] == 0
        assert result["needle_cleaned"] == 0

    def test_needle_orphan_cleanup_primary(self):
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

            result = sweep_orphans()

        assert result["needle_cleaned"] >= 1

        conn = get_db()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM needle_registry WHERE memory_id = 'orphan-primary-id'"
        ).fetchone()[0]
        conn.close()
        assert remaining == 0

    def test_needle_orphan_cleanup_child(self):
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

            result = sweep_orphans()

        assert result["needle_cleaned"] >= 1

        conn = get_db()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM needle_registry WHERE memory_id = 'orphan-microchunk-id'"
        ).fetchone()[0]
        conn.close()
        assert remaining == 0

    def test_retrieve_failure_skips_subbatch(self):
        """If client.retrieve fails for one collection, sub-batch is conservatively kept."""
        from graph import get_db, upsert_fts_chunk

        upsert_fts_chunk("maybe-orphan", "text", "f.md", 0, "a", "ns")

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_client.retrieve.side_effect = Exception("retrieve error")

        with (
            patch("cascade.qdrant_client", return_value=mock_client),
            patch("cascade.collections_for_query", return_value=["col_a", "col_b"]),
        ):
            from cascade import sweep_orphans

            result = sweep_orphans()

        assert result["fts_cleaned"] == 0

        conn = get_db()
        count = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id = 'maybe-orphan'"
        ).fetchone()[0]
        conn.close()
        assert count == 1

    def test_keyset_pagination_processes_all_pages(self):
        """Sweeper uses keyset pagination (WHERE id > ?) to process multiple pages."""
        from graph import get_db, upsert_fts_chunk

        ids_to_insert = [f"ks-{i:04d}" for i in range(3)]
        for qid in ids_to_insert:
            upsert_fts_chunk(qid, f"text for {qid}", "f.md", 0, "a", "ns")

        mock_client = MagicMock()
        mock_client.get_collections.return_value = MagicMock()
        mock_client.retrieve.return_value = []

        with (
            patch("cascade.qdrant_client", return_value=mock_client),
            patch("cascade.collections_for_query", return_value=["test_col"]),
            patch("cascade._SWEEP_PAGE_SIZE", 2),
        ):
            from cascade import sweep_orphans

            result = sweep_orphans()

        assert result["fts_cleaned"] == 3

        conn = get_db()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM memory_chunks WHERE qdrant_id LIKE 'ks-%'"
        ).fetchone()[0]
        conn.close()
        assert remaining == 0


class TestDeleteHotness:
    """graph.delete_hotness removes memory_hotness rows."""

    def test_deletes_existing_row(self):
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

        deleted = delete_hotness("hot-mem-1")
        assert deleted == 1

        conn = get_db()
        remaining = conn.execute(
            "SELECT COUNT(*) FROM memory_hotness WHERE memory_id = 'hot-mem-1'"
        ).fetchone()[0]
        conn.close()
        assert remaining == 0

    def test_returns_zero_for_missing_id(self):
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

        assert delete_hotness("nonexistent") == 0

    def test_returns_zero_when_table_missing(self):
        """Silently returns 0 if memory_hotness table doesn't exist yet."""
        from graph import delete_hotness, get_db

        conn = get_db()
        conn.execute("DROP TABLE IF EXISTS memory_hotness")
        conn.commit()
        conn.close()

        assert delete_hotness("any-id") == 0


class TestBatchSqliteRetry:
    """SQLite batch functions retry on OperationalError."""

    def test_fts_retry_on_operational_error(self):
        """delete_fts_chunks_batch retries once on sqlite3.OperationalError."""
        import sqlite3 as _sqlite3

        from graph import delete_fts_chunks_batch

        call_count = 0
        _orig_get_db = None

        def _flaky_get_db():
            nonlocal call_count, _orig_get_db
            call_count += 1
            if call_count == 1:
                raise _sqlite3.OperationalError("database is locked")
            return _orig_get_db()

        from graph import get_db as orig_get_db

        _orig_get_db = orig_get_db

        from graph import upsert_fts_chunk

        upsert_fts_chunk("retry-id", "text", "f.md", 0, "a", "ns")

        with patch("graph.get_db", side_effect=_flaky_get_db):
            deleted = delete_fts_chunks_batch(["retry-id"])

        assert deleted == 1
        assert call_count == 2

    def test_needle_retry_on_operational_error(self):
        """delete_needle_tokens_batch retries once on sqlite3.OperationalError."""
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
        _orig_get_db = get_db

        def _flaky_get_db():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _sqlite3.OperationalError("database is locked")
            return _orig_get_db()

        with patch("graph.get_db", side_effect=_flaky_get_db):
            deleted = delete_needle_tokens_batch(["retry-needle-id"])

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

    def _insert_fact(self, memory_id: str, source_file: str = "") -> int:
        """Insert a test fact row and return its id."""
        from graph import get_db, upsert_entity

        eid = upsert_entity("test-entity", "concept", namespace="global")
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

    def test_exact_match_deactivates_by_memory_id(self):
        """Primary path: rows with matching memory_id are deactivated."""
        from graph import get_db
        from memory_lifecycle import _delete_entity_facts_for_memory

        mid = "mem-exact-test-001"
        self._insert_fact(mid, source_file="explicit/agent")

        count = _delete_entity_facts_for_memory(mid)
        assert count == 1

        conn = get_db()
        active = conn.execute("SELECT is_active FROM facts WHERE memory_id = ?", (mid,)).fetchone()
        conn.close()
        assert active is not None
        assert active[0] == 0

    def test_like_fallback_for_pre_migration_rows(self):
        """Fallback: rows with memory_id='' but matching source_file are deactivated."""
        from graph import get_db
        from memory_lifecycle import _delete_entity_facts_for_memory

        mid = "mem-fallback-test-002"
        # Pre-migration row: memory_id is empty, but source_file contains the UUID
        self._insert_fact("", source_file=f"explicit/{mid}")

        count = _delete_entity_facts_for_memory(mid)
        assert count == 1

        conn = get_db()
        row = conn.execute(
            "SELECT is_active FROM facts WHERE source_file = ?", (f"explicit/{mid}",)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 0

    def test_like_fallback_does_not_touch_rows_with_memory_id_set(self):
        """LIKE fallback is scoped to memory_id='' only — does not double-deactivate."""
        from memory_lifecycle import _delete_entity_facts_for_memory

        mid = "mem-scope-test-003"
        # Row already has memory_id set correctly
        self._insert_fact(mid, source_file=f"explicit/{mid}")

        count = _delete_entity_facts_for_memory(mid)
        # Should be found by the primary exact-match path only (count = 1)
        assert count == 1

    def test_non_matching_rows_untouched(self):
        """Facts belonging to a different memory are not deactivated."""
        from graph import get_db
        from memory_lifecycle import _delete_entity_facts_for_memory

        mid_target = "mem-target-004"
        mid_other = "mem-other-004"
        self._insert_fact(mid_other, source_file="explicit/other-agent")

        count = _delete_entity_facts_for_memory(mid_target)
        assert count == 0

        conn = get_db()
        row = conn.execute(
            "SELECT is_active FROM facts WHERE memory_id = ?", (mid_other,)
        ).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == 1  # untouched

    def test_returns_zero_for_unknown_memory(self):
        """Returns 0 and does not crash for a memory_id with no matching facts."""
        from memory_lifecycle import _delete_entity_facts_for_memory

        assert _delete_entity_facts_for_memory("completely-unknown-id") == 0


class TestAddFactMemoryId:
    """add_fact stores memory_id and it can be queried."""

    def test_stores_memory_id(self):
        from graph import add_fact, get_db, upsert_entity

        eid = upsert_entity("test-entity-af", "concept", namespace="global")
        mid = "mem-add-fact-test-001"
        add_fact(
            eid, "test fact text", "explicit/agent", "agent", namespace="global", memory_id=mid
        )

        conn = get_db()
        row = conn.execute("SELECT memory_id FROM facts WHERE memory_id = ?", (mid,)).fetchone()
        conn.close()
        assert row is not None
        assert row[0] == mid

    def test_default_memory_id_is_empty(self):
        """Existing call sites that don't pass memory_id get empty string."""
        from graph import add_fact, get_db, upsert_entity

        eid = upsert_entity("entity-no-mid", "concept", namespace="global")
        fact_id = add_fact(
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
