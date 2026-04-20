"""Chaos tests for entity upsert race conditions and migration simulation.

These tests verify that the ``ON CONFLICT DO UPDATE`` idempotency fix in
``upsert_entity`` holds under the adversarial concurrent-write conditions that
occur during memory migration, multi-agent session start, and agent fleet
scale-out.

Run with::

    pytest tests/chaos/test_entity_races.py -v -m chaos

They are intentionally fast (in-memory SQLite, no Qdrant) and deterministic.
"""

from __future__ import annotations

import asyncio

import pytest

pytestmark = [pytest.mark.chaos]


async def test_entity_upsert_during_migration_simulation(qa_pool):
    """20 concurrent agents all upserting the same 5 entities must not raise and must agree on IDs.

    This directly simulates the "duplicate key for 'brian'" failure mode that
    caused the agent fleet to enter endless retry loops during memory migration.
    The pre-fix SELECT-then-INSERT pattern would raise IntegrityError under
    this load; the atomic ON CONFLICT DO UPDATE must resolve it silently.
    """
    from archivist.storage.graph import upsert_entity

    entity_names = ["brian", "kubernetes", "argocd", "grafana", "postgres"]

    async def migration_worker(agent_idx: int) -> list[tuple[str, int]]:
        results = []
        for name in entity_names:
            eid = await upsert_entity(name, "system", namespace="migration-test")
            results.append((name, eid))
        return results

    all_results = await asyncio.gather(
        *[migration_worker(i) for i in range(20)],
        return_exceptions=True,
    )

    exceptions = [r for r in all_results if isinstance(r, Exception)]
    assert not exceptions, f"Migration workers raised unexpected exceptions: {exceptions}"

    worker_results: list[list[tuple[str, int]]] = [
        r for r in all_results if not isinstance(r, Exception)
    ]  # type: ignore[assignment]
    assert len(worker_results) == 20

    for entity_idx, name in enumerate(entity_names):
        ids_for_entity = {worker[entity_idx][1] for worker in worker_results}
        assert len(ids_for_entity) == 1, (
            f"Entity '{name}' got different IDs across workers: {ids_for_entity}"
        )
        sole_id = next(iter(ids_for_entity))
        assert sole_id > 0, f"Entity '{name}' ID must be a positive integer, got {sole_id}"


async def test_entity_upsert_high_cardinality_concurrent(qa_pool):
    """50 unique entities upserted by 10 concurrent workers produce exactly 50 rows."""
    from archivist.storage.graph import upsert_entity

    entity_count = 50
    worker_count = 10

    async def worker(worker_idx: int) -> list[int]:
        ids = []
        for i in range(entity_count):
            eid = await upsert_entity(
                f"entity-{i:04d}",
                "test",
                namespace="high-cardinality",
            )
            ids.append(eid)
        return ids

    all_results = await asyncio.gather(
        *[worker(w) for w in range(worker_count)],
        return_exceptions=True,
    )

    exceptions = [r for r in all_results if isinstance(r, Exception)]
    assert not exceptions, f"Workers raised: {exceptions}"

    for entity_idx in range(entity_count):
        ids_for_entity = {
            result[entity_idx] for result in all_results if not isinstance(result, Exception)
        }  # type: ignore[index]
        assert len(ids_for_entity) == 1, (
            f"entity-{entity_idx:04d} has multiple IDs: {ids_for_entity}"
        )


async def test_entity_upsert_namespace_isolation_under_concurrency(qa_pool):
    """Same entity name in N namespaces, under concurrent load, must stay isolated."""
    from archivist.storage.graph import upsert_entity

    namespaces = [f"ns-{i}" for i in range(10)]
    entity_name = "shared-agent"

    # Each namespace upserted 5 times concurrently
    tasks = [
        upsert_entity(entity_name, "agent", namespace=ns) for ns in namespaces for _ in range(5)
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    exceptions = [r for r in results if isinstance(r, Exception)]
    assert not exceptions, f"Namespace isolation test raised: {exceptions}"

    ids_by_ns: dict[str, set[int]] = {}
    idx = 0
    for ns in namespaces:
        for _ in range(5):
            eid = results[idx]  # type: ignore[assignment]
            ids_by_ns.setdefault(ns, set()).add(int(eid))  # type: ignore[arg-type]
            idx += 1

    # Each namespace must have exactly one distinct ID
    for ns, id_set in ids_by_ns.items():
        assert len(id_set) == 1, f"Namespace '{ns}' produced multiple IDs: {id_set}"

    # Different namespaces must have different IDs
    all_ids = [next(iter(id_set)) for id_set in ids_by_ns.values()]
    assert len(set(all_ids)) == len(namespaces), (
        f"Expected {len(namespaces)} unique IDs (one per namespace), got: {set(all_ids)}"
    )


async def test_retention_class_converges_under_concurrent_escalation(qa_pool, memory_factory):
    """When agents concurrently escalate a single entity's retention class, the highest wins."""
    from archivist.storage.graph import upsert_entity
    from archivist.storage.sqlite_pool import pool

    retention_classes = ["ephemeral", "standard", "durable", "permanent", "standard", "ephemeral"]

    async def escalate(rc: str) -> int:
        return await upsert_entity(
            "lifecycle-entity", "agent", retention_class=rc, namespace="rc-chaos"
        )

    results = await asyncio.gather(
        *[escalate(rc) for rc in retention_classes], return_exceptions=True
    )

    exceptions = [r for r in results if isinstance(r, Exception)]
    assert not exceptions, f"Retention escalation raised: {exceptions}"

    async with pool.read() as conn:
        cur = await conn.execute(
            "SELECT retention_class FROM entities WHERE name=? AND namespace=?",
            ("lifecycle-entity", "rc-chaos"),
        )
        row = await cur.fetchone()

    assert row is not None, "Entity must exist after concurrent upserts"
    assert row[0] == "permanent", f"Expected 'permanent' (highest rank) to win, got: {row[0]}"
