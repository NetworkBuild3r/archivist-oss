"""Integration tests: cascade delete cleans relationships and memory_versions.

Verifies that delete_memory_complete removes:
  1. memory_versions rows keyed to the deleted memory_id
  2. Orphaned relationship rows (where entity has no remaining active facts)

These were missing from the original cascade (H3, H4).
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.lifecycle]


# ---------------------------------------------------------------------------
# Helper: insert test rows directly via pool
# ---------------------------------------------------------------------------


async def _insert_entity(conn, name: str, namespace: str = "test-ns") -> int:
    now = datetime.now(UTC).isoformat()
    cur = await conn.execute(
        """INSERT INTO entities (name, entity_type, first_seen, last_seen, namespace)
           VALUES (?, 'person', ?, ?, ?)
           ON CONFLICT(name, namespace) DO UPDATE SET last_seen=excluded.last_seen
           RETURNING id""",
        (name, now, now, namespace),
    )
    row = await cur.fetchone()
    return row[0]


async def _insert_fact(conn, entity_id: int, memory_id: str) -> None:
    now = datetime.now(UTC).isoformat()
    await conn.execute(
        """INSERT INTO facts (entity_id, fact_text, source_file, agent_id, created_at,
                              namespace, memory_id, is_active)
           VALUES (?, 'test fact', 'test.md', 'agent1', ?, 'test-ns', ?, 1)""",
        (entity_id, now, memory_id),
    )


async def _insert_relationship(conn, src_id: int, tgt_id: int) -> int:
    now = datetime.now(UTC).isoformat()
    cur = await conn.execute(
        """INSERT INTO relationships
               (source_entity_id, target_entity_id, relation_type, evidence,
                confidence, created_at, updated_at)
           VALUES (?, ?, 'works_with', 'test', 1.0, ?, ?)
           RETURNING id""",
        (src_id, tgt_id, now, now),
    )
    row = await cur.fetchone()
    return row[0]


async def _insert_memory_version(conn, memory_id: str, version: int = 1) -> int:
    cur = await conn.execute(
        """INSERT INTO memory_versions
               (memory_id, version, agent_id, timestamp, text_hash, operation)
           VALUES (?, ?, 'agent1', '2025-01-01T00:00:00', 'abc123', 'store')
           RETURNING id""",
        (memory_id, version),
    )
    row = await cur.fetchone()
    return row[0]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCascadeDeleteVersionRows:
    """delete_memory_complete removes memory_versions rows (fix for H4)."""

    async def test_memory_versions_deleted_on_cascade(self, integration_pool):
        memory_id = "mem-version-test-001"

        async with integration_pool.write() as conn:
            await _insert_memory_version(conn, memory_id, version=1)
            await _insert_memory_version(conn, memory_id, version=2)
            await _insert_memory_version(conn, "other-mem-999", version=1)

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        with (
            patch(
                "archivist.lifecycle.memory_lifecycle.qdrant_client",
                return_value=mock_client,
            ),
            patch(
                "archivist.lifecycle.memory_lifecycle.collection_for",
                return_value="test_collection",
            ),
            patch("archivist.lifecycle.cascade._qdrant_delete", return_value=1),
            patch("archivist.lifecycle.cascade._qdrant_set_payload", return_value=1),
            patch("archivist.core.audit.log_memory_event", new_callable=AsyncMock),
            patch(
                "archivist.lifecycle.memory_lifecycle.lookup_memory_points",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            from archivist.lifecycle.memory_lifecycle import delete_memory_complete

            result = await delete_memory_complete(memory_id, "test-ns")

        assert result.version_rows == 2, (
            f"Expected 2 version rows deleted, got {result.version_rows}"
        )

        async with integration_pool.read() as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) FROM memory_versions WHERE memory_id = ?", (memory_id,)
            )
            row = await cur.fetchone()
            assert row[0] == 0, "memory_versions rows still present after cascade delete"

            cur2 = await conn.execute(
                "SELECT COUNT(*) FROM memory_versions WHERE memory_id = 'other-mem-999'"
            )
            row2 = await cur2.fetchone()
            assert row2[0] == 1, "unrelated memory_versions row was wrongly deleted"


class TestCascadeDeleteRelationshipRows:
    """delete_memory_complete removes orphaned relationships (fix for H3)."""

    async def test_orphaned_relationships_deleted_after_fact_deactivation(self, integration_pool):
        memory_id = "mem-rel-test-001"

        async with integration_pool.write() as conn:
            eid_a = await _insert_entity(conn, "Entity-A-Orphan-Rel")
            eid_b = await _insert_entity(conn, "Entity-B-Survivor-Rel")

            await _insert_fact(conn, eid_a, memory_id)
            await _insert_fact(conn, eid_b, memory_id)
            await _insert_fact(conn, eid_b, "other-mem-survivor")

            rel_id = await _insert_relationship(conn, eid_a, eid_b)

        mock_client = MagicMock()
        mock_client.scroll.return_value = ([], None)

        with (
            patch(
                "archivist.lifecycle.memory_lifecycle.qdrant_client",
                return_value=mock_client,
            ),
            patch(
                "archivist.lifecycle.memory_lifecycle.collection_for",
                return_value="test_collection",
            ),
            patch("archivist.lifecycle.cascade._qdrant_delete", return_value=1),
            patch("archivist.lifecycle.cascade._qdrant_set_payload", return_value=1),
            patch("archivist.core.audit.log_memory_event", new_callable=AsyncMock),
            patch(
                "archivist.lifecycle.memory_lifecycle.lookup_memory_points",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            from archivist.lifecycle.memory_lifecycle import delete_memory_complete

            result = await delete_memory_complete(memory_id, "test-ns")

        async with integration_pool.read() as conn:
            cur = await conn.execute(
                "SELECT COUNT(*) FROM facts WHERE entity_id = ? AND is_active = 1", (eid_a,)
            )
            row = await cur.fetchone()
            assert row[0] == 0, "Entity A still has active facts after cascade"

            cur2 = await conn.execute("SELECT COUNT(*) FROM relationships WHERE id = ?", (rel_id,))
            row2 = await cur2.fetchone()
            assert row2[0] == 0, "Orphaned relationship A->B was not deleted"

        assert result.relationship_rows >= 0


class TestDeleteResultIncludesNewFields:
    """DeleteResult.total includes relationship_rows and version_rows."""

    def test_total_includes_relationship_and_version_rows(self):
        from archivist.lifecycle.memory_lifecycle import DeleteResult

        r = DeleteResult(
            memory_id="abc",
            qdrant_primary=1,
            fts_entries=1,
            relationship_rows=3,
            version_rows=2,
        )
        assert r.total == 7
        assert r.relationship_rows == 3
        assert r.version_rows == 2

    def test_defaults_are_zero(self):
        from archivist.lifecycle.memory_lifecycle import DeleteResult

        r = DeleteResult()
        assert r.relationship_rows == 0
        assert r.version_rows == 0
        assert r.total == 0
