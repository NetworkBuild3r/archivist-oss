"""Integration tests for Memory-as-Product (INIT-001/SPEC-009; INIT-011/SPEC-004).

Fork-then-read isolation, snapshot→export→import round-trip, and Postgres
SQL artifact presence.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.storage]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POSTGRES_SQL = _REPO_ROOT / "src" / "archivist" / "storage" / "schema_postgres.sql"


def _map_backup_patches(monkeypatch, backup: Path) -> None:
    from archivist.storage.backup_manager import _snapshot_dir, delete_snapshot

    monkeypatch.setattr("archivist.storage.backup_manager.BACKUP_DIR", str(backup))
    monkeypatch.setattr("archivist.storage.memory_product._snapshot_dir", _snapshot_dir)
    monkeypatch.setattr("archivist.storage.memory_product.delete_snapshot", delete_snapshot)


@pytest.mark.asyncio
async def test_fork_then_read_isolation(async_pool, tmp_path, monkeypatch, rbac_config):
    """Forked scope is readable in target and does not leak source qdrant_ids."""
    from archivist.storage import memory_product as mp

    backup = tmp_path / "backups"
    backup.mkdir()
    _map_backup_patches(monkeypatch, backup)

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


@pytest.mark.asyncio
@pytest.mark.agentic_memory
async def test_snapshot_export_import_round_trip(async_pool, tmp_path, monkeypatch, rbac_config):
    """INIT-011/SPEC-004 ac-1: snapshot → export → import restores usable memories."""
    from archivist.storage import memory_product as mp

    backup = tmp_path / "backups"
    backup.mkdir()
    _map_backup_patches(monkeypatch, backup)

    marker = "MAP_ROUNDTRIP_MARKER_INIT011"
    async with async_pool.write() as conn:
        await conn.execute(
            """
            INSERT INTO memory_chunks (
                qdrant_id, text, file_path, chunk_index, agent_id, namespace,
                date, memory_type, is_excluded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            ("rt-qid-1", f"fact {marker}", "rt.md", 0, "chief", "shared", "", "general"),
        )

    snap = await mp.create_scope_snapshot(
        namespace="shared",
        caller_agent_id="chief",
        agent_id="chief",
        label="rt-snap",
        vectors=[{"id": "rt-qid-1", "vector": [0.2, 0.3], "payload": {}}],
    )
    exported = await mp.export_scope(
        namespace="shared",
        caller_agent_id="chief",
        agent_id="chief",
        version_id=snap.id,
        label="rt-export",
    )
    assert exported["archive_id"] == snap.archive_id
    assert Path(exported["path"]).is_dir()

    imported = await mp.import_scope(
        archive_id=exported["archive_id"],
        target_namespace="pipeline",
        caller_agent_id="gitbob",
        target_agent_id="gitbob",
        label="rt-import",
    )
    assert imported.operation == "import"
    assert imported.chunk_count == 1

    async with async_pool.read() as conn:
        cur = await conn.execute(
            "SELECT text, namespace, agent_id FROM memory_chunks "
            "WHERE namespace = ? AND agent_id = ?",
            ("pipeline", "gitbob"),
        )
        rows = [dict(r) for r in await cur.fetchall()]
    assert len(rows) == 1
    assert marker in rows[0]["text"]
    assert rows[0]["namespace"] == "pipeline"


@pytest.mark.asyncio
@pytest.mark.agentic_memory
async def test_fork_and_import_coexist(async_pool, tmp_path, monkeypatch, rbac_config):
    """INIT-011/SPEC-004 ac-2: fork path stays green alongside import."""
    from archivist.storage import memory_product as mp

    backup = tmp_path / "backups"
    backup.mkdir()
    _map_backup_patches(monkeypatch, backup)

    async with async_pool.write() as conn:
        await conn.execute(
            """
            INSERT INTO memory_chunks (
                qdrant_id, text, file_path, chunk_index, agent_id, namespace,
                date, memory_type, is_excluded
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            ("co-qid-1", "coexist fact", "c.md", 0, "chief", "shared", "", "general"),
        )

    source = await mp.create_scope_snapshot(
        namespace="shared",
        caller_agent_id="chief",
        agent_id="chief",
        vectors=[{"id": "co-qid-1", "vector": [1.0], "payload": {}}],
    )
    forked = await mp.fork_from_snapshot(
        source_version_id=source.id,
        target_namespace="pipeline",
        caller_agent_id="gitbob",
        target_agent_id="gitbob-fork",
        label="co-fork",
    )
    assert forked.operation == "fork"
    assert forked.chunk_count == 1

    # Import into a different agent scope under pipeline (empty target).
    imported = await mp.import_scope(
        archive_id=source.archive_id,
        target_namespace="pipeline",
        caller_agent_id="gitbob",
        target_agent_id="gitbob-import",
        label="co-import",
    )
    assert imported.operation == "import"
    assert imported.chunk_count == 1

    async with async_pool.read() as conn:
        cur = await conn.execute(
            "SELECT agent_id, COUNT(*) AS c FROM memory_chunks "
            "WHERE namespace = ? GROUP BY agent_id",
            ("pipeline",),
        )
        counts = {r["agent_id"]: int(r["c"]) for r in await cur.fetchall()}
    assert counts.get("gitbob-fork") == 1
    assert counts.get("gitbob-import") == 1


def test_postgres_sql_artifact_present():
    ddl = _POSTGRES_SQL.read_text(encoding="utf-8")
    assert "memory_scope_versions" in ddl
    assert "idx_msv_namespace_version" in ddl
