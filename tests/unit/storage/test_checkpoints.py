"""Unit tests for agent checkpoint store (INIT-001/SPEC-007).

Covers schema presence after init_schema, CRUD helpers, namespace scoping,
parent linking, and index DDL presence in both SQLite and Postgres SQL.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.storage]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POSTGRES_SQL = _REPO_ROOT / "src" / "archivist" / "storage" / "schema_postgres.sql"


def test_sqlite_schema_creates_agent_checkpoints_idempotently(tmp_path, monkeypatch):
    """ac-1: CREATE IF NOT EXISTS succeeds twice on SQLite."""
    db_path = str(tmp_path / "ckpt.db")
    monkeypatch.setenv("SQLITE_PATH", db_path)
    monkeypatch.setattr("archivist.core.config.SQLITE_PATH", db_path)
    monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "sqlite")

    from archivist.storage import graph_schema

    monkeypatch.setattr(graph_schema, "SQLITE_PATH", db_path, raising=False)
    graph_schema.init_schema()
    graph_schema.init_schema()  # idempotent

    conn = sqlite3.connect(db_path)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(agent_checkpoints)")}
        assert {
            "id",
            "agent_id",
            "session_id",
            "namespace",
            "parent_checkpoint_id",
            "payload",
            "blob_ref",
            "metadata",
            "created_at",
        } <= cols
        idx_sql = " ".join(
            (r[0] or "")
            for r in conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='agent_checkpoints'"
            ).fetchall()
        )
        assert "idx_checkpoints_agent_session_time" in idx_sql
        assert "idx_checkpoints_session_time" in idx_sql
        assert "idx_checkpoints_namespace" in idx_sql
    finally:
        conn.close()


def test_postgres_sql_artifact_defines_checkpoints_and_indexes():
    """ac-1 / ac-4: Postgres DDL artifact present with table + list indexes."""
    assert _POSTGRES_SQL.is_file()
    ddl = _POSTGRES_SQL.read_text(encoding="utf-8")
    assert "CREATE TABLE IF NOT EXISTS agent_checkpoints" in ddl
    assert "idx_checkpoints_agent_session_time" in ddl
    assert "idx_checkpoints_session_time" in ddl
    assert "idx_checkpoints_namespace" in ddl
    assert "idx_checkpoints_parent" in ddl
    assert "DROP TABLE IF EXISTS agent_checkpoints" in ddl  # rollback note in comments
    # Checkpoint table block must not reuse L0–L2 tier_label (GR-002)
    start = ddl.index("CREATE TABLE IF NOT EXISTS agent_checkpoints")
    end = ddl.index(";", start)
    assert "tier_label" not in ddl[start:end]


@pytest.mark.asyncio
async def test_create_get_list_and_parent_link(async_pool):
    """ac-2 / ac-3: CRUD + parent link + time-ordered list."""
    from archivist.storage import checkpoints as ckpt

    root = await ckpt.create_checkpoint(
        agent_id="agent-a",
        session_id="sess-1",
        namespace="ns-alpha",
        payload={"step": 0, "state": "boot"},
        metadata={"source": "unit"},
    )
    child = await ckpt.create_checkpoint(
        agent_id="agent-a",
        session_id="sess-1",
        namespace="ns-alpha",
        payload={"step": 1},
        parent_checkpoint_id=root.id,
    )

    got = await ckpt.get_checkpoint(root.id, namespace="ns-alpha")
    assert got is not None
    assert got.payload == {"step": 0, "state": "boot"}
    assert got.metadata == {"source": "unit"}
    assert got.parent_checkpoint_id is None

    # Wrong namespace → no leak
    assert await ckpt.get_checkpoint(root.id, namespace="ns-other") is None

    listed = await ckpt.list_checkpoints_by_session(
        agent_id="agent-a",
        session_id="sess-1",
        namespace="ns-alpha",
    )
    assert [r.id for r in listed] == [root.id, child.id]
    assert listed[1].parent_checkpoint_id == root.id

    # link_parent on a third row created without parent
    orphan = await ckpt.create_checkpoint(
        agent_id="agent-a",
        session_id="sess-1",
        namespace="ns-alpha",
        payload={"step": 2},
    )
    linked = await ckpt.link_parent(orphan.id, child.id, namespace="ns-alpha")
    assert linked is not None
    assert linked.parent_checkpoint_id == child.id


@pytest.mark.asyncio
async def test_create_requires_namespace(async_pool):
    from archivist.storage import checkpoints as ckpt

    with pytest.raises(ValueError, match="namespace"):
        await ckpt.create_checkpoint(
            agent_id="a",
            session_id="s",
            namespace="",
            payload={},
        )


@pytest.mark.asyncio
async def test_link_parent_rejects_self_and_missing(async_pool):
    from archivist.storage import checkpoints as ckpt

    row = await ckpt.create_checkpoint(
        agent_id="a",
        session_id="s",
        namespace="ns",
        payload={},
    )
    with pytest.raises(ValueError, match="own parent"):
        await ckpt.link_parent(row.id, row.id, namespace="ns")
    with pytest.raises(ValueError, match="parent checkpoint not found"):
        await ckpt.link_parent(row.id, "missing-id", namespace="ns")


@pytest.mark.asyncio
async def test_payload_not_logged_at_info(async_pool, caplog):
    """Security AC: checkpoint payloads are not logged at info level."""
    import logging

    from archivist.storage import checkpoints as ckpt

    secretish = {"token": "should-not-appear-in-info-logs"}
    with caplog.at_level(logging.INFO, logger="archivist.checkpoints"):
        await ckpt.create_checkpoint(
            agent_id="a",
            session_id="s",
            namespace="ns",
            payload=secretish,
        )
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert "should-not-appear-in-info-logs" not in joined
