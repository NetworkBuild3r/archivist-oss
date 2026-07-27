"""INIT-010/SPEC-004 — integration coverage for Diff #6 self-curation product paths.

Seeds real SQLite chunks/facts and exercises reconsolidation, relevance forget,
and contradiction resolve cycles behind ADR-010 safe flags (apply only when
explicitly opted in for the test).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.lifecycle]


async def _seed_l2(memory_id: str, namespace: str, agent_id: str, text: str) -> None:
    from archivist.storage.graph import upsert_fts_chunk

    await upsert_fts_chunk(
        memory_id,
        text,
        f"{memory_id}.md",
        0,
        agent_id=agent_id,
        namespace=namespace,
        importance=0.5,
        tier_label="l2",
    )


@pytest.mark.asyncio
async def test_reconsolidation_cycle_writes_l1_same_namespace(async_pool, monkeypatch):
    """ac-1: enabled+apply reconsolidation upserts L1 in source agent/namespace."""
    import archivist.lifecycle.reconsolidation as mod

    ns, agent = "ns-recon-int", "agent-recon"
    for i in range(3):
        await _seed_l2(f"recon-src-{i}", ns, agent, f"L2 note {i} about canary rollout.")

    monkeypatch.setattr(mod, "RECONSOLIDATION_ENABLED", True)
    monkeypatch.setattr(mod, "RECONSOLIDATION_DRY_RUN", False)
    monkeypatch.setattr(mod, "RECONSOLIDATION_MIN_CHUNKS", 3)
    monkeypatch.setattr(mod, "RECONSOLIDATION_MAX_GROUPS_PER_CYCLE", 5)

    with patch.object(
        mod,
        "_summarize_group",
        new=AsyncMock(return_value="Consolidated L1 overview of canary rollout notes."),
    ):
        out = await mod.reconsolidation_cycle(dry_run=False)

    assert out["enabled"] is True
    assert out["proposed"] >= 1
    assert out["applied"] >= 1
    assert out["dry_run"] is False

    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        rows = await conn.fetchall(
            "SELECT qdrant_id, namespace, agent_id, tier_label, memory_type "
            "FROM memory_chunks WHERE memory_type = 'reconsolidation' AND namespace = ?",
            (ns,),
        )
    assert rows
    row = rows[0]
    assert row["namespace"] == ns
    assert row["agent_id"] == agent
    assert row["tier_label"] == "l1"


@pytest.mark.asyncio
async def test_relevance_forget_cycle_suppresses_cold_chunk(async_pool, monkeypatch):
    """ac-2: relevance forget apply suppresses cold/low-importance chunk in-ns."""
    import archivist.lifecycle.relevance_forget as mod
    from archivist.storage.graph import upsert_fts_chunk
    from archivist.storage.sqlite_pool import pool

    ns, agent, mid = "ns-forget-int", "agent-forget", "forget-cold-1"
    await upsert_fts_chunk(
        mid,
        "obsolete low-value note",
        "cold.md",
        0,
        agent_id=agent,
        namespace=ns,
        importance=0.1,
        tier_label="l2",
    )
    old = (datetime.now(UTC) - timedelta(days=30)).isoformat()
    now = datetime.now(UTC).isoformat()
    async with pool.write() as conn:
        await conn.execute(
            "UPDATE memory_chunks SET created_at=? WHERE qdrant_id=? AND namespace=?",
            (old, mid, ns),
        )
        await conn.execute(
            "INSERT INTO memory_hotness "
            "(memory_id, score, retrieval_count, last_accessed, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(memory_id) DO UPDATE SET score=excluded.score",
            (mid, 0.01, 0, old, now),
        )

    monkeypatch.setattr(mod, "RELEVANCE_FORGET_ENABLED", True)
    monkeypatch.setattr(mod, "RELEVANCE_FORGET_DRY_RUN", False)
    monkeypatch.setattr(mod, "RELEVANCE_FORGET_MAX_PER_CYCLE", 10)
    monkeypatch.setattr(mod, "RELEVANCE_FORGET_HOTNESS_MAX", 0.05)
    monkeypatch.setattr(mod, "RELEVANCE_FORGET_IMPORTANCE_MAX", 0.4)
    monkeypatch.setattr(mod, "RELEVANCE_FORGET_MIN_AGE_DAYS", 7)

    with (
        patch(
            "archivist.lifecycle.correct._best_effort_qdrant_payload",
            new=AsyncMock(return_value=[]),
        ),
        patch.object(mod, "_qdrant_retention_class", new=AsyncMock(return_value=None)),
    ):
        out = await mod.relevance_forget_cycle(namespace=ns, dry_run=False)

    assert out["enabled"] is True
    assert out["proposed"] >= 1
    assert out["applied"] >= 1

    async with pool.read() as conn:
        row = await conn.fetchone(
            "SELECT is_suppressed FROM memory_chunks WHERE qdrant_id=? AND namespace=?",
            (mid, ns),
        )
    assert row is not None
    assert int(row["is_suppressed"] or 0) == 1


@pytest.mark.asyncio
async def test_resolve_cycle_dry_run_proposes_without_mutate(async_pool, monkeypatch):
    """ac-2: resolve cycle under ADR dry-run proposes and leaves facts active."""
    import archivist.lifecycle.contradiction_resolve as cr
    from archivist.storage.graph import add_fact, get_entity_facts, upsert_entity
    from archivist.storage.sqlite_pool import pool

    eid = await upsert_entity("svc-curation-cycle", "service", namespace="ns-resolve-cycle")
    fa = await add_fact(
        eid,
        "Feature flag alpha is enabled in staging",
        agent_id="alice",
        namespace="ns-resolve-cycle",
    )
    async with pool.write() as conn:
        await conn.execute(
            "UPDATE facts SET created_at=? WHERE id=?",
            ("2026-01-01T00:00:00+00:00", fa),
        )
    fb = await add_fact(
        eid,
        "Feature flag alpha is disabled after rollback",
        agent_id="bob",
        namespace="ns-resolve-cycle",
    )
    async with pool.write() as conn:
        await conn.execute(
            "UPDATE facts SET created_at=? WHERE id=?",
            ("2026-03-01T00:00:00+00:00", fb),
        )

    monkeypatch.setattr(cr, "CONTRADICTION_RESOLVE_ENABLED", True)
    monkeypatch.setattr(cr, "CONTRADICTION_RESOLVE_DRY_RUN", True)
    monkeypatch.setattr(cr, "CONTRADICTION_RESOLVE_MAX_PER_CYCLE", 10)

    out = await cr.resolve_contradictions_cycle(namespace="ns-resolve-cycle", dry_run=True)
    assert out["enabled"] is True
    assert out["dry_run"] is True
    assert out["proposed"] >= 1
    assert out["applied"] == 0

    facts = await get_entity_facts(eid)
    active = [f for f in facts if int(f.get("is_active") or 0) == 1]
    assert len(active) >= 2
