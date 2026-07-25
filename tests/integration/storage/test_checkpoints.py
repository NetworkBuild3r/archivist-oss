"""Integration tests for agent checkpoints (INIT-001/SPEC-007).

SQLite path runs against the async pool. Postgres live tests are skipped when
``DATABASE_URL`` / Postgres is unavailable; the SQL artifact is still asserted.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.storage]

_POSTGRES_SQL = (
    Path(__file__).resolve().parents[3] / "src" / "archivist" / "storage" / "schema_postgres.sql"
)


@pytest.mark.asyncio
async def test_concurrent_create_same_session(async_pool):
    """Concurrent creates under one session remain consistent and listable."""
    from archivist.storage import checkpoints as ckpt

    async def _one(i: int):
        return await ckpt.create_checkpoint(
            agent_id="agent-concurrent",
            session_id="sess-c",
            namespace="ns-c",
            payload={"i": i},
            metadata={"batch": "concurrent"},
        )

    rows = await asyncio.gather(*[_one(i) for i in range(8)])
    assert len({r.id for r in rows}) == 8

    listed = await ckpt.list_checkpoints_by_session(
        agent_id="agent-concurrent",
        session_id="sess-c",
        namespace="ns-c",
        limit=50,
    )
    assert len(listed) == 8
    assert all(r.namespace == "ns-c" for r in listed)


@pytest.mark.asyncio
async def test_parent_chain_walk(async_pool):
    """Parent links form a walkable chain within a namespace."""
    from archivist.storage import checkpoints as ckpt

    c0 = await ckpt.create_checkpoint(
        agent_id="agent-chain",
        session_id="sess-chain",
        namespace="ns-chain",
        payload={"n": 0},
    )
    c1 = await ckpt.create_checkpoint(
        agent_id="agent-chain",
        session_id="sess-chain",
        namespace="ns-chain",
        payload={"n": 1},
        parent_checkpoint_id=c0.id,
    )
    c2 = await ckpt.create_checkpoint(
        agent_id="agent-chain",
        session_id="sess-chain",
        namespace="ns-chain",
        payload={"n": 2},
        parent_checkpoint_id=c1.id,
    )

    # Walk parents
    cur = await ckpt.get_checkpoint(c2.id, namespace="ns-chain")
    chain_ids: list[str] = []
    while cur is not None:
        chain_ids.append(cur.id)
        if cur.parent_checkpoint_id is None:
            break
        cur = await ckpt.get_checkpoint(cur.parent_checkpoint_id, namespace="ns-chain")
    assert chain_ids == [c2.id, c1.id, c0.id]


def test_postgres_sql_artifact_present_for_dual_backend():
    """ac-4: Postgres SQL present even when live Postgres is unavailable."""
    assert _POSTGRES_SQL.is_file()
    text = _POSTGRES_SQL.read_text(encoding="utf-8")
    assert "agent_checkpoints" in text


@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("ARCHIVIST_TEST_DATABASE_URL")
    and not (
        os.environ.get("GRAPH_BACKEND", "").lower() == "postgres" and os.environ.get("DATABASE_URL")
    ),
    reason="Postgres unavailable — set ARCHIVIST_TEST_DATABASE_URL to exercise live DDL",
)
async def test_postgres_schema_creates_agent_checkpoints_idempotently():
    """ac-1 (Postgres live): apply schema_postgres.sql twice when a DSN is configured."""
    import asyncpg

    dsn = os.environ.get("ARCHIVIST_TEST_DATABASE_URL") or os.environ["DATABASE_URL"]
    ddl = _POSTGRES_SQL.read_text(encoding="utf-8")
    conn = await asyncpg.connect(dsn)
    try:
        await conn.execute(ddl)
        await conn.execute(ddl)  # idempotent
        row = await conn.fetchrow(
            """
            SELECT 1 AS ok
            FROM information_schema.tables
            WHERE table_name = 'agent_checkpoints'
            """
        )
        assert row is not None
        idx = await conn.fetch(
            """
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'agent_checkpoints'
            """
        )
        names = {r["indexname"] for r in idx}
        assert "idx_checkpoints_agent_session_time" in names
        assert "idx_checkpoints_session_time" in names
    finally:
        await conn.close()
