"""Integration tests for Memory-as-Product (INIT-001/SPEC-009).

Fork-then-read isolation and Postgres SQL artifact presence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.storage]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POSTGRES_SQL = _REPO_ROOT / "src" / "archivist" / "storage" / "schema_postgres.sql"


@pytest.mark.asyncio
async def test_fork_then_read_isolation(async_pool, tmp_path, monkeypatch, rbac_config):
    """Forked scope is readable in target and does not leak source qdrant_ids."""
    from archivist.storage import memory_product as mp
    from archivist.storage.backup_manager import _snapshot_dir, delete_snapshot

    backup = tmp_path / "backups"
    backup.mkdir()
    monkeypatch.setattr("archivist.storage.backup_manager.BACKUP_DIR", str(backup))
    monkeypatch.setattr("archivist.storage.memory_product._snapshot_dir", _snapshot_dir)
    monkeypatch.setattr("archivist.storage.memory_product.delete_snapshot", delete_snapshot)

    async with async_pool.write() as conn:
        await conn.execute(
            """
            INSERT INTO memory_chunks (
                qdrant_id, text, file_path, chunk_index, agent_id, namespace,
                date, memory_type, is_excluded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            ("src-qid-1", "secret-ish fact", "a.md", 0, "chief", "shared", "", "general"),
        )

    source = await mp.create_scope_snapshot(
        namespace="shared",
        caller_agent_id="chief",
        vectors=[{"id": "src-qid-1", "vector": [0.1, 0.2, 0.3], "payload": {}}],
    )
    forked = await mp.fork_from_snapshot(
        source_version_id=source.id,
        target_namespace="pipeline",
        caller_agent_id="gitbob",
        target_agent_id="gitbob",
        label="iso",
    )

    async with async_pool.read() as conn:
        cur = await conn.execute(
            "SELECT qdrant_id, text, namespace FROM memory_chunks WHERE namespace = ?",
            ("pipeline",),
        )
        rows = [dict(r) for r in await cur.fetchall()]
    assert len(rows) == 1
    assert rows[0]["text"] == "secret-ish fact"
    assert rows[0]["qdrant_id"] != "src-qid-1"
    assert rows[0]["namespace"] == "pipeline"
    assert forked.parent_version_id == source.id

    # Source id still present only in source ns
    async with async_pool.read() as conn:
        cur = await conn.execute(
            "SELECT namespace FROM memory_chunks WHERE qdrant_id = ?",
            ("src-qid-1",),
        )
        src_row = await cur.fetchone()
    assert src_row is not None
    assert src_row["namespace"] == "shared"


def test_postgres_sql_artifact_present():
    ddl = _POSTGRES_SQL.read_text(encoding="utf-8")
    assert "memory_scope_versions" in ddl
    assert "idx_msv_namespace_version" in ddl
