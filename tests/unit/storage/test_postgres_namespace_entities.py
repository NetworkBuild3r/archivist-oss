"""Regression tests for namespace-scoped Postgres entities.

Provenance: INIT-006/SPEC-002 (ported from local fix f252b24 onto the
INIT-001 graph module split — migration lives in graph_schema, upsert in
graph_entities).
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.storage]


def test_postgres_schema_declares_namespace_scoped_entity_unique_constraint():
    import archivist.storage.graph as graph

    schema = Path(graph.__file__).with_name("schema_postgres.sql").read_text()

    assert "CONSTRAINT entities_name_unique UNIQUE (name, namespace)" in schema
    assert "CONSTRAINT entities_name_unique UNIQUE (name)\n" not in schema


def test_postgres_entity_unique_migration_changes_name_only_constraint():
    import archivist.storage.graph_schema as graph_schema

    source = Path(graph_schema.__file__).read_text()

    assert "ALTER TABLE entities DROP CONSTRAINT entities_name_unique" in source
    assert (
        "ALTER TABLE entities ADD CONSTRAINT entities_name_unique UNIQUE (name, namespace)"
        in source
    )
    assert "_migrate_entity_unique_constraint_postgres" in source


@pytest.mark.asyncio
async def test_postgres_upsert_entity_is_atomic_and_namespace_scoped(monkeypatch):
    monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "postgres")
    monkeypatch.setattr("archivist.storage.graph_schema._is_postgres", lambda: True)
    monkeypatch.setattr("archivist.storage.graph_entities._is_postgres", lambda: True)

    calls: list[tuple[str, tuple[object, ...]]] = []

    class FakeConn:
        async def fetchval(self, sql: str, params: tuple[object, ...]):
            calls.append((sql, params))
            return 42

    from archivist.storage.graph import upsert_entity

    result = await upsert_entity(
        "athena",
        "agent",
        namespace="athena-identity",
        actor_id="athena",
        actor_type="agent",
        conn=FakeConn(),
    )

    assert result == 42
    assert len(calls) == 1
    sql, params = calls[0]
    assert "ON CONFLICT (name, namespace)" in sql
    assert "mention_count = entities.mention_count + 1" in sql
    assert params[0] == "athena"
    assert params[5] == "athena-identity"
