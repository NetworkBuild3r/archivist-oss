"""Unit tests for INIT-003/SPEC-007 supersede / suppress / delete lifecycle.

Covers the service-layer state machine and default-recall visibility helpers
that SPEC-005 / SPEC-006 will call. Qdrant is mocked; SQLite flags are real.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.lifecycle]


def _qdrant_client() -> MagicMock:
    client = MagicMock()
    client.set_payload = MagicMock(return_value=True)
    return client


async def _seed_chunk(memory_id: str, namespace: str, text: str = "body") -> None:
    from archivist.storage.graph import upsert_fts_chunk

    await upsert_fts_chunk(
        memory_id,
        text,
        "f.md",
        0,
        agent_id="agent",
        namespace=namespace,
    )


# ---------------------------------------------------------------------------
# Visibility pure helpers
# ---------------------------------------------------------------------------


class TestIsRecallVisible:
    def test_visible_by_default(self):
        from archivist.lifecycle.visibility import is_recall_visible

        assert is_recall_visible({"qdrant_id": "a", "is_suppressed": 0}) is True

    def test_suppressed_hidden(self):
        from archivist.lifecycle.visibility import is_recall_visible

        assert is_recall_visible({"qdrant_id": "a", "is_suppressed": 1}) is False
        assert (
            is_recall_visible({"qdrant_id": "a", "is_suppressed": 1}, include_suppressed=True)
            is True
        )

    def test_superseded_loser_via_flag(self):
        from archivist.lifecycle.visibility import is_recall_visible

        assert is_recall_visible({"qdrant_id": "old", "is_superseded": True}) is False
        assert is_recall_visible({"id": 1, "superseded_by": 9}) is False

    def test_superseded_loser_via_known_set(self):
        from archivist.lifecycle.visibility import is_recall_visible

        assert (
            is_recall_visible(
                {"qdrant_id": "old", "is_suppressed": 0},
                known_superseded_ids={"old"},
            )
            is False
        )
        assert (
            is_recall_visible(
                {"qdrant_id": "new", "is_suppressed": 0},
                known_superseded_ids={"old"},
            )
            is True
        )

    def test_deleted_hidden(self):
        from archivist.lifecycle.visibility import is_recall_visible

        assert is_recall_visible({"qdrant_id": "a", "deleted": True}) is False
        assert is_recall_visible({"qdrant_id": "a", "is_excluded": 1}) is False

    def test_sql_predicates_mention_suppress_and_supersede(self):
        from archivist.lifecycle.visibility import (
            recall_visible_sql_chunks,
            recall_visible_sql_facts,
        )

        chunks = recall_visible_sql_chunks("mc")
        facts = recall_visible_sql_facts("f")
        assert "is_suppressed" in chunks
        assert "supersedes_id" in chunks
        assert "is_suppressed" in facts
        assert "superseded_by" in facts


# ---------------------------------------------------------------------------
# Suppress
# ---------------------------------------------------------------------------


class TestSuppressMemory:
    async def test_suppress_hides_but_record_remains(self, async_pool):
        from archivist.lifecycle.correct import get_lifecycle_state, suppress_memory
        from archivist.lifecycle.visibility import is_recall_visible
        from archivist.storage.chunk_lifecycle import list_superseded_loser_ids

        await _seed_chunk("mem-s1", "ns-a", "keep me")

        with (
            patch("archivist.lifecycle.correct.qdrant_client", return_value=_qdrant_client()),
            patch("archivist.lifecycle.correct.collection_for", return_value="col"),
            patch("archivist.lifecycle.correct.log_memory_event", new_callable=AsyncMock),
        ):
            result = await suppress_memory("mem-s1", "ns-a", reason="stale tip")

        assert result["status"] == "suppressed"
        state = await get_lifecycle_state("mem-s1", "ns-a")
        assert state is not None
        assert state["is_suppressed"] == 1
        assert is_recall_visible(state) is False
        # Record still present (not hard-erased).
        assert state["qdrant_id"] == "mem-s1"
        assert await list_superseded_loser_ids("ns-a") == set()

    async def test_suppress_is_namespace_scoped(self, async_pool):
        """Wrong-namespace suppress must not flip flags (qdrant_id is globally unique)."""
        from archivist.lifecycle.correct import get_lifecycle_state, suppress_memory

        await _seed_chunk("mem-ns", "ns-a")

        with (
            patch("archivist.lifecycle.correct.qdrant_client", return_value=_qdrant_client()),
            patch("archivist.lifecycle.correct.collection_for", return_value="col"),
            patch("archivist.lifecycle.correct.log_memory_event", new_callable=AsyncMock),
        ):
            wrong = await suppress_memory("mem-ns", "ns-other")
            right = await suppress_memory("mem-ns", "ns-a")

        assert wrong["rows_updated"] == 0
        assert right["rows_updated"] == 1
        state = await get_lifecycle_state("mem-ns", "ns-a")
        assert state is not None and state["is_suppressed"] == 1
        assert await get_lifecycle_state("mem-ns", "ns-other") is None

    async def test_wrong_namespace_suppress_does_not_mutate_qdrant(self, async_pool):
        """INIT-003/SPEC-009 SEC-001: wrong-ns suppress must not call Qdrant payload."""
        from archivist.lifecycle.correct import suppress_memory

        await _seed_chunk("mem-sec001", "ns-a")
        qdrant_spy = AsyncMock(return_value=[])

        with (
            patch(
                "archivist.lifecycle.correct._best_effort_qdrant_payload",
                qdrant_spy,
            ),
            patch("archivist.lifecycle.correct.log_memory_event", new_callable=AsyncMock),
        ):
            wrong = await suppress_memory("mem-sec001", "ns-other")

        assert wrong["rows_updated"] == 0
        qdrant_spy.assert_not_awaited()

    async def test_same_namespace_suppress_mutates_qdrant(self, async_pool):
        """INIT-003/SPEC-009 SEC-001: durable hit still syncs Qdrant suppress flag."""
        from archivist.lifecycle.correct import suppress_memory

        await _seed_chunk("mem-sec001b", "ns-a")
        qdrant_spy = AsyncMock(return_value=[])

        with (
            patch(
                "archivist.lifecycle.correct._best_effort_qdrant_payload",
                qdrant_spy,
            ),
            patch("archivist.lifecycle.correct.log_memory_event", new_callable=AsyncMock),
        ):
            result = await suppress_memory("mem-sec001b", "ns-a")

        assert result["rows_updated"] == 1
        qdrant_spy.assert_awaited_once()
        assert qdrant_spy.await_args.args[0] == "mem-sec001b"
        assert qdrant_spy.await_args.args[1] == "ns-a"
        assert qdrant_spy.await_args.args[2] == {"is_suppressed": True}

    async def test_suppress_idempotent(self, async_pool):
        from archivist.lifecycle.correct import suppress_memory

        await _seed_chunk("mem-s2", "ns-a")

        with (
            patch("archivist.lifecycle.correct.qdrant_client", return_value=_qdrant_client()),
            patch("archivist.lifecycle.correct.collection_for", return_value="col"),
            patch("archivist.lifecycle.correct.log_memory_event", new_callable=AsyncMock),
        ):
            first = await suppress_memory("mem-s2", "ns-a")
            second = await suppress_memory("mem-s2", "ns-a")

        assert first["status"] == "suppressed"
        assert second["status"] == "suppressed"

    async def test_unsuppress_required_to_restore(self, async_pool):
        from archivist.lifecycle.correct import (
            get_lifecycle_state,
            suppress_memory,
            unsuppress_memory,
        )
        from archivist.lifecycle.visibility import is_recall_visible

        await _seed_chunk("mem-s3", "ns-a")

        with (
            patch("archivist.lifecycle.correct.qdrant_client", return_value=_qdrant_client()),
            patch("archivist.lifecycle.correct.collection_for", return_value="col"),
            patch("archivist.lifecycle.correct.log_memory_event", new_callable=AsyncMock),
        ):
            await suppress_memory("mem-s3", "ns-a")
            state = await get_lifecycle_state("mem-s3", "ns-a")
            assert is_recall_visible(state) is False
            await unsuppress_memory("mem-s3", "ns-a")
            state = await get_lifecycle_state("mem-s3", "ns-a")
            assert state is not None and state["is_suppressed"] == 0
            assert is_recall_visible(state) is True


# ---------------------------------------------------------------------------
# Supersede / correct
# ---------------------------------------------------------------------------


class TestSupersedeMemory:
    async def test_correct_hides_loser_keeps_winner(self, async_pool):
        from archivist.lifecycle.correct import (
            correct_memory,
            default_recall_rows,
            get_lifecycle_state,
        )

        await _seed_chunk("mem-old", "ns-a", "old fact")
        await _seed_chunk("mem-new", "ns-a", "corrected fact")

        with (
            patch("archivist.lifecycle.correct.qdrant_client", return_value=_qdrant_client()),
            patch("archivist.lifecycle.correct.collection_for", return_value="col"),
            patch("archivist.lifecycle.correct.log_memory_event", new_callable=AsyncMock),
        ):
            result = await correct_memory("mem-old", "mem-new", "ns-a")

        assert result["status"] == "corrected"
        winner = await get_lifecycle_state("mem-new", "ns-a")
        loser = await get_lifecycle_state("mem-old", "ns-a")
        assert winner is not None and winner["supersedes_id"] == "mem-old"
        assert loser is not None  # not hard-erased

        rows = [
            {"qdrant_id": "mem-old", "is_suppressed": 0, "is_excluded": 0},
            {"qdrant_id": "mem-new", "is_suppressed": 0, "is_excluded": 0},
        ]
        visible = await default_recall_rows(rows, "ns-a")
        ids = {r["qdrant_id"] for r in visible}
        assert "mem-old" not in ids
        assert "mem-new" in ids

    async def test_supersede_namespace_scoped(self, async_pool):
        """Wrong-namespace supersede must not set supersedes_id on the winner."""
        from archivist.lifecycle.correct import get_lifecycle_state, supersede_memory
        from archivist.storage.chunk_lifecycle import list_superseded_loser_ids

        await _seed_chunk("mem-old", "ns-a")
        await _seed_chunk("mem-new", "ns-a")

        with (
            patch("archivist.lifecycle.correct.qdrant_client", return_value=_qdrant_client()),
            patch("archivist.lifecycle.correct.collection_for", return_value="col"),
            patch("archivist.lifecycle.correct.log_memory_event", new_callable=AsyncMock),
        ):
            wrong = await supersede_memory("mem-old", "mem-new", "ns-other")
            right = await supersede_memory("mem-old", "mem-new", "ns-a")

        assert wrong["rows_updated"] == 0
        assert right["rows_updated"] == 1
        assert (await get_lifecycle_state("mem-new", "ns-a"))["supersedes_id"] == "mem-old"
        assert await list_superseded_loser_ids("ns-a") == {"mem-old"}
        assert await list_superseded_loser_ids("ns-other") == set()

    async def test_wrong_namespace_supersede_does_not_mutate_qdrant(self, async_pool):
        """INIT-003/SPEC-009 SEC-001: wrong-ns supersede must not touch Qdrant."""
        from archivist.lifecycle.correct import supersede_memory

        await _seed_chunk("mem-old-sec", "ns-a")
        await _seed_chunk("mem-new-sec", "ns-a")
        qdrant_spy = AsyncMock(return_value=[])

        with (
            patch(
                "archivist.lifecycle.correct._best_effort_qdrant_payload",
                qdrant_spy,
            ),
            patch("archivist.lifecycle.correct.log_memory_event", new_callable=AsyncMock),
        ):
            wrong = await supersede_memory("mem-old-sec", "mem-new-sec", "ns-other")

        assert wrong["rows_updated"] == 0
        qdrant_spy.assert_not_awaited()

    async def test_rejects_same_ids(self):
        from archivist.lifecycle.correct import supersede_memory

        with pytest.raises(ValueError, match="must differ"):
            await supersede_memory("x", "x", "ns")


# ---------------------------------------------------------------------------
# Delete (idempotent)
# ---------------------------------------------------------------------------


class TestDeleteMemory:
    async def test_delete_tombstones_via_soft_delete(self, async_pool):
        from archivist.lifecycle.correct import delete_memory

        await _seed_chunk("mem-d1", "ns-a")

        with (
            patch(
                "archivist.lifecycle.correct.soft_delete_memory",
                new_callable=AsyncMock,
                return_value={"status": "soft_delete_initiated", "op_id": "op-1"},
            ) as mock_soft,
            patch("archivist.lifecycle.correct.log_memory_event", new_callable=AsyncMock),
        ):
            result = await delete_memory("mem-d1", "ns-a")

        mock_soft.assert_awaited_once_with("mem-d1", "ns-a")
        assert result["status"] == "soft_delete_initiated"
        assert result["idempotent"] is False

    async def test_second_delete_idempotent(self, async_pool):
        from archivist.lifecycle.correct import delete_memory
        from archivist.storage.graph import set_fts_excluded_batch

        await _seed_chunk("mem-d2", "ns-a")
        await set_fts_excluded_batch(["mem-d2"], 1)

        with (
            patch(
                "archivist.lifecycle.correct.soft_delete_memory",
                new_callable=AsyncMock,
            ) as mock_soft,
            patch("archivist.lifecycle.correct.log_memory_event", new_callable=AsyncMock),
        ):
            result = await delete_memory("mem-d2", "ns-a")

        mock_soft.assert_not_awaited()
        assert result["status"] == "already_deleted"
        assert result["idempotent"] is True

    async def test_delete_hides_from_default_recall(self, async_pool):
        from archivist.lifecycle.visibility import is_recall_visible
        from archivist.storage.graph import set_fts_excluded_batch

        await _seed_chunk("mem-d3", "ns-a")
        await set_fts_excluded_batch(["mem-d3"], 1)
        from archivist.lifecycle.correct import get_lifecycle_state

        state = await get_lifecycle_state("mem-d3", "ns-a")
        assert is_recall_visible(state) is False


# ---------------------------------------------------------------------------
# Security: no secrets in logs / audit metadata
# ---------------------------------------------------------------------------


class TestNoSecretLogging:
    async def test_suppress_audit_has_no_text_body(self, async_pool):
        from archivist.lifecycle.correct import suppress_memory

        await _seed_chunk("mem-sec", "ns-a", text="password=hunter2 secret_token=abc")
        log_calls: list[dict] = []

        async def _capture(**kwargs):
            log_calls.append(kwargs)

        with (
            patch("archivist.lifecycle.correct.qdrant_client", return_value=_qdrant_client()),
            patch("archivist.lifecycle.correct.collection_for", return_value="col"),
            patch("archivist.lifecycle.correct.log_memory_event", side_effect=_capture),
        ):
            await suppress_memory("mem-sec", "ns-a", reason="ops")

        assert log_calls
        meta = log_calls[0]["metadata"]
        blob = str(meta) + str(log_calls[0].get("text_hash", ""))
        assert "hunter2" not in blob
        assert "secret_token" not in blob
        assert log_calls[0]["namespace"] == "ns-a"


# ---------------------------------------------------------------------------
# INIT-003/SPEC-009 — FTS suppress defense-in-depth (SEC-002)
# ---------------------------------------------------------------------------


class TestFtsSuppressWhere:
    def test_build_fts_where_excludes_suppressed(self):
        from archivist.storage.graph_fts import _build_fts_where

        clauses, _params = _build_fts_where("", "", "", "")
        assert "mc.is_suppressed = 0" in clauses
        assert "mc.is_excluded = 0" in clauses
