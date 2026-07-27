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
    assert got.metadata.get("source") == "unit"
    assert got.metadata.get(ckpt.HITL_STATUS_KEY) == ckpt.HITL_STATUS_NONE
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


# ---------------------------------------------------------------------------
# INIT-012/SPEC-002 — branch + HITL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_branch_checkpoint_happy_path(async_pool):
    """ac-1: branch creates child with required parent in same namespace."""
    from archivist.storage import checkpoints as ckpt

    parent = await ckpt.create_checkpoint(
        agent_id="agent-a",
        session_id="sess-1",
        namespace="ns-alpha",
        payload={"step": 0, "secret": "do-not-log"},
    )
    child = await ckpt.branch_checkpoint(
        parent_checkpoint_id=parent.id,
        namespace="ns-alpha",
        agent_id="agent-a",
    )
    assert child.parent_checkpoint_id == parent.id
    assert child.agent_id == "agent-a"
    assert child.session_id == "sess-1"
    assert child.payload == parent.payload
    assert child.metadata.get(ckpt.HITL_STATUS_KEY) == ckpt.HITL_STATUS_NONE
    assert child.id != parent.id


@pytest.mark.asyncio
async def test_branch_checkpoint_fail_closed(async_pool):
    """ac-3: missing parent / wrong owner / cross-namespace → error."""
    from archivist.storage import checkpoints as ckpt

    parent = await ckpt.create_checkpoint(
        agent_id="agent-a",
        session_id="sess-1",
        namespace="ns-alpha",
        payload={"step": 0},
    )

    with pytest.raises(ckpt.CheckpointNotFoundError):
        await ckpt.branch_checkpoint(
            parent_checkpoint_id="missing-id",
            namespace="ns-alpha",
            agent_id="agent-a",
        )

    # Parent id exists but wrong namespace → not found (no leak)
    with pytest.raises(ckpt.CheckpointNotFoundError):
        await ckpt.branch_checkpoint(
            parent_checkpoint_id=parent.id,
            namespace="ns-other",
            agent_id="agent-a",
        )

    with pytest.raises(ckpt.CheckpointAuthzError):
        await ckpt.branch_checkpoint(
            parent_checkpoint_id=parent.id,
            namespace="ns-alpha",
            agent_id="agent-b",
        )


@pytest.mark.asyncio
async def test_hitl_interrupt_approve_resume_gate(async_pool):
    """ac-2: interrupt blocks resume until approve; approve is idempotent."""
    from archivist.storage import checkpoints as ckpt

    row = await ckpt.create_checkpoint(
        agent_id="agent-a",
        session_id="sess-1",
        namespace="ns-alpha",
        payload={"step": 1},
    )
    interrupted = await ckpt.interrupt_checkpoint(
        row.id,
        namespace="ns-alpha",
        agent_id="agent-a",
        reason="need human",
    )
    assert ckpt.hitl_status(interrupted) == ckpt.HITL_STATUS_INTERRUPTED
    assert interrupted.metadata.get(ckpt.HITL_REASON_KEY) == "need human"

    with pytest.raises(ckpt.CheckpointConflictError, match="interrupted"):
        ckpt.ensure_resume_allowed(interrupted, agent_id="agent-a")

    approved = await ckpt.approve_checkpoint(
        row.id,
        namespace="ns-alpha",
        agent_id="agent-a",
    )
    assert ckpt.hitl_status(approved) == ckpt.HITL_STATUS_APPROVED
    ckpt.ensure_resume_allowed(approved, agent_id="agent-a")  # no raise

    again = await ckpt.approve_checkpoint(
        row.id,
        namespace="ns-alpha",
        agent_id="agent-a",
    )
    assert again.id == approved.id
    assert ckpt.hitl_status(again) == ckpt.HITL_STATUS_APPROVED


@pytest.mark.asyncio
async def test_hitl_owner_bind_deny(async_pool):
    """ac-3: interrupt/approve wrong owner fail closed."""
    from archivist.storage import checkpoints as ckpt

    row = await ckpt.create_checkpoint(
        agent_id="agent-a",
        session_id="sess-1",
        namespace="ns-alpha",
        payload={},
    )
    with pytest.raises(ckpt.CheckpointAuthzError):
        await ckpt.interrupt_checkpoint(
            row.id,
            namespace="ns-alpha",
            agent_id="intruder",
        )
    await ckpt.interrupt_checkpoint(
        row.id,
        namespace="ns-alpha",
        agent_id="agent-a",
    )
    with pytest.raises(ckpt.CheckpointAuthzError):
        await ckpt.approve_checkpoint(
            row.id,
            namespace="ns-alpha",
            agent_id="intruder",
        )


@pytest.mark.asyncio
async def test_client_cannot_forge_hitl_status_on_create_or_branch(async_pool):
    """SEC-012-02: save/branch strip client HITL keys; only interrupt/approve set them."""
    from archivist.storage import checkpoints as ckpt

    forged = await ckpt.create_checkpoint(
        agent_id="agent-a",
        session_id="sess-1",
        namespace="ns-alpha",
        payload={"step": 0},
        metadata={"hitl_status": "approved", "label": "keep-me"},
    )
    assert forged.metadata.get(ckpt.HITL_STATUS_KEY) == ckpt.HITL_STATUS_NONE
    assert forged.metadata.get("label") == "keep-me"

    child = await ckpt.branch_checkpoint(
        parent_checkpoint_id=forged.id,
        namespace="ns-alpha",
        agent_id="agent-a",
        metadata={"hitl_status": "approved", "hitl_reason": "forged"},
    )
    assert child.metadata.get(ckpt.HITL_STATUS_KEY) == ckpt.HITL_STATUS_NONE
    assert ckpt.HITL_REASON_KEY not in child.metadata


@pytest.mark.asyncio
async def test_branch_hitl_payload_not_logged(async_pool, caplog):
    """ac-4 / security: branch + HITL info logs omit payload bodies."""
    import logging

    from archivist.storage import checkpoints as ckpt

    secret = "should-not-appear-in-info-logs-branch"
    parent = await ckpt.create_checkpoint(
        agent_id="agent-a",
        session_id="sess-1",
        namespace="ns-alpha",
        payload={"token": secret},
    )
    with caplog.at_level(logging.INFO, logger="archivist.checkpoints"):
        child = await ckpt.branch_checkpoint(
            parent_checkpoint_id=parent.id,
            namespace="ns-alpha",
            agent_id="agent-a",
        )
        await ckpt.interrupt_checkpoint(
            child.id,
            namespace="ns-alpha",
            agent_id="agent-a",
            reason="pause",
        )
        await ckpt.approve_checkpoint(
            child.id,
            namespace="ns-alpha",
            agent_id="agent-a",
        )
    joined = " ".join(r.getMessage() for r in caplog.records)
    assert secret not in joined
