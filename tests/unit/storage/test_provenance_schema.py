"""Unit/schema tests for coach provenance + supersede/suppress DDL.

INIT-003/SPEC-002 — dual-backend init paths:
  - SQLite via ``graph_schema.init_schema()`` (fresh + upgrade migration)
  - Postgres via ``schema_postgres.sql`` artifact (content + live apply when DSN set)
"""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.storage]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_POSTGRES_SQL = _REPO_ROOT / "src" / "archivist" / "storage" / "schema_postgres.sql"
POSTGRES_DSN = os.getenv("POSTGRES_TEST_DSN", "")

# Coach provenance envelope columns required on store/search path tables.
_MC_PROVENANCE_COLS = {
    "source",
    "actor_id",
    "actor_type",
    "subject",
    "namespace",
    "confidence",
    "sensitivity",
    "purpose",
    "statement_kind",
    "created_at",
    "updated_at",
    "supersedes_id",
    "is_suppressed",
}
_FACTS_PROVENANCE_COLS = {
    "source",
    "actor_id",
    "subject",
    "namespace",
    "confidence",
    "sensitivity",
    "purpose",
    "statement_kind",
    "created_at",
    "updated_at",
    "superseded_by",
    "is_suppressed",
}
_MC_FILTER_INDEXES = {
    "idx_mc_subject",
    "idx_mc_purpose",
    "idx_mc_sensitivity",
    "idx_mc_statement_kind",
    "idx_mc_suppressed",
    "idx_mc_supersedes",
    "idx_mc_ns_suppressed",
    "idx_mc_created_at",
}
_FACTS_FILTER_INDEXES = {
    "idx_facts_subject",
    "idx_facts_purpose",
    "idx_facts_sensitivity",
    "idx_facts_statement_kind",
    "idx_facts_suppressed",
    "idx_facts_superseded_by",
}


def _sqlite_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}


def _sqlite_index_names(conn: sqlite3.Connection, table: str) -> set[str]:
    return {
        row[1]
        for row in conn.execute(f"PRAGMA index_list({table})")
        if row[1]  # name
    }


def test_sqlite_init_schema_creates_provenance_columns_idempotently(tmp_path, monkeypatch):
    """ac-1 / ac-4: fresh SQLite init exposes provenance + lifecycle columns twice."""
    db_path = str(tmp_path / "prov.db")
    monkeypatch.setenv("SQLITE_PATH", db_path)
    monkeypatch.setattr("archivist.core.config.SQLITE_PATH", db_path)
    monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "sqlite")

    from archivist.storage import graph_schema

    monkeypatch.setattr(graph_schema, "SQLITE_PATH", db_path, raising=False)
    graph_schema.init_schema()
    graph_schema.init_schema()

    conn = sqlite3.connect(db_path)
    try:
        mc_cols = _sqlite_columns(conn, "memory_chunks")
        facts_cols = _sqlite_columns(conn, "facts")
        assert mc_cols >= _MC_PROVENANCE_COLS
        assert facts_cols >= _FACTS_PROVENANCE_COLS

        mc_idx = _sqlite_index_names(conn, "memory_chunks")
        facts_idx = _sqlite_index_names(conn, "facts")
        assert mc_idx >= _MC_FILTER_INDEXES
        assert facts_idx >= _FACTS_FILTER_INDEXES
    finally:
        conn.close()


def test_sqlite_migrate_adds_provenance_to_legacy_memory_chunks(tmp_path, monkeypatch):
    """ac-1 / ac-2 / ac-4: upgrade path adds envelope columns to a pre-SPEC-002 table."""
    db_path = str(tmp_path / "legacy.db")
    monkeypatch.setenv("SQLITE_PATH", db_path)
    monkeypatch.setattr("archivist.core.config.SQLITE_PATH", db_path)
    monkeypatch.setattr("archivist.core.config.GRAPH_BACKEND", "sqlite")

    # Minimal pre-INIT-003/SPEC-002 memory_chunks + facts (previous head).
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(
            """
            CREATE TABLE memory_chunks (
                rowid INTEGER PRIMARY KEY,
                qdrant_id TEXT NOT NULL UNIQUE,
                text TEXT NOT NULL,
                file_path TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                agent_id TEXT NOT NULL DEFAULT '',
                namespace TEXT NOT NULL DEFAULT '',
                date TEXT NOT NULL DEFAULT '',
                memory_type TEXT NOT NULL DEFAULT 'general',
                is_excluded INTEGER NOT NULL DEFAULT 0,
                actor_id TEXT NOT NULL DEFAULT '',
                actor_type TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                entity_id INTEGER,
                fact_text TEXT NOT NULL,
                source_file TEXT,
                agent_id TEXT,
                created_at TEXT NOT NULL,
                superseded_by INTEGER,
                is_active INTEGER NOT NULL DEFAULT 1,
                namespace TEXT NOT NULL DEFAULT 'global',
                confidence REAL NOT NULL DEFAULT 1.0,
                provenance TEXT NOT NULL DEFAULT 'unknown',
                actor_id TEXT NOT NULL DEFAULT ''
            );
            CREATE TABLE entities (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT NOT NULL UNIQUE COLLATE NOCASE,
                entity_type TEXT NOT NULL DEFAULT 'unknown',
                first_seen TEXT NOT NULL,
                last_seen TEXT NOT NULL,
                mention_count INTEGER NOT NULL DEFAULT 1,
                metadata TEXT DEFAULT '{}',
                namespace TEXT NOT NULL DEFAULT 'global'
            );
            INSERT INTO memory_chunks (
                qdrant_id, text, file_path, chunk_index, namespace
            ) VALUES ('legacy-1', 'old text', 'a.md', 0, 'ns-a');
            INSERT INTO facts (fact_text, created_at, namespace)
            VALUES ('legacy fact', '2026-01-01T00:00:00Z', 'ns-a');
            """
        )
        conn.commit()
    finally:
        conn.close()

    from archivist.storage import graph_schema

    monkeypatch.setattr(graph_schema, "SQLITE_PATH", db_path, raising=False)
    graph_schema._migrate_schema()

    conn = sqlite3.connect(db_path)
    try:
        mc_cols = _sqlite_columns(conn, "memory_chunks")
        facts_cols = _sqlite_columns(conn, "facts")
        assert mc_cols >= _MC_PROVENANCE_COLS
        assert facts_cols >= _FACTS_PROVENANCE_COLS

        # Legacy rows remain readable; null-ish defaults apply (empty / standard).
        row = conn.execute(
            "SELECT source, subject, sensitivity, purpose, statement_kind, "
            "supersedes_id, is_suppressed, confidence, namespace "
            "FROM memory_chunks WHERE qdrant_id = 'legacy-1'"
        ).fetchone()
        assert row is not None
        assert row[0] == ""  # source
        assert row[1] == ""  # subject
        assert row[2] == "standard"
        assert row[3] == ""  # purpose
        assert row[4] == "user"
        assert row[5] == ""  # supersedes_id
        assert row[6] == 0  # is_suppressed
        assert row[7] == 1.0
        assert row[8] == "ns-a"  # namespace preserved

        # Supersede + suppress are persisted and queryable.
        conn.execute(
            "UPDATE memory_chunks SET statement_kind = 'inferred', "
            "subject = 'sleep', purpose = 'coaching', sensitivity = 'health', "
            "is_suppressed = 1 "
            "WHERE qdrant_id = 'legacy-1'"
        )
        conn.execute(
            "INSERT INTO memory_chunks ("
            "qdrant_id, text, file_path, chunk_index, namespace, "
            "supersedes_id, is_suppressed, statement_kind"
            ") VALUES ('corr-1', 'corrected', 'b.md', 0, 'ns-a', 'legacy-1', 0, 'user')"
        )
        conn.commit()

        suppressed = conn.execute(
            "SELECT qdrant_id FROM memory_chunks WHERE namespace = ? AND is_suppressed = 1",
            ("ns-a",),
        ).fetchall()
        assert [r[0] for r in suppressed] == ["legacy-1"]

        superseding = conn.execute(
            "SELECT qdrant_id FROM memory_chunks WHERE supersedes_id = ?",
            ("legacy-1",),
        ).fetchall()
        assert [r[0] for r in superseding] == ["corr-1"]

        # Defaults carry no secrets.
        defaults = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='memory_chunks'"
        ).fetchone()
        assert defaults is not None
        assert "password" not in defaults[0].lower()
        assert "secret" not in defaults[0].lower()
        assert "token" not in defaults[0].lower()
    finally:
        conn.close()


def test_postgres_sql_artifact_defines_provenance_and_indexes():
    """ac-1 / ac-2 / ac-3 / ac-4: Postgres DDL artifact includes columns + indexes + notes."""
    assert _POSTGRES_SQL.is_file()
    ddl = _POSTGRES_SQL.read_text(encoding="utf-8")

    assert "INIT-003/SPEC-002" in ddl
    assert "statement_kind" in ddl
    assert "is_suppressed" in ddl
    assert "supersedes_id" in ddl
    assert "DEFAULT 'user'" in ddl
    assert "DEFAULT 'standard'" in ddl
    assert "Previous head" in ddl
    assert "Rollback" in ddl

    for name in _MC_FILTER_INDEXES | _FACTS_FILTER_INDEXES:
        assert name in ddl

    # No secrets in defaults for envelope columns.
    assert "DEFAULT 'password'" not in ddl.lower()
    assert "api_key" not in ddl.lower()


@pytest.mark.asyncio
async def test_postgres_init_applies_provenance_when_dsn_available():
    """ac-4: live Postgres init path when POSTGRES_TEST_DSN is set; else skip."""
    if not POSTGRES_DSN:
        pytest.skip("POSTGRES_TEST_DSN not set — skipping Postgres live init")

    pytest.importorskip("asyncpg")
    from archivist.storage.asyncpg_backend import AsyncpgGraphBackend

    backend = AsyncpgGraphBackend()
    await backend.initialize(POSTGRES_DSN, min_size=1, max_size=2)
    try:
        ddl = _POSTGRES_SQL.read_text(encoding="utf-8")
        await backend.execute_ddl(ddl)

        async with backend.read() as conn:
            mc_rows = await conn.fetchall(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'memory_chunks'"
            )
            facts_rows = await conn.fetchall(
                "SELECT column_name FROM information_schema.columns WHERE table_name = 'facts'"
            )

        mc_cols = {r["column_name"] for r in mc_rows}
        facts_cols = {r["column_name"] for r in facts_rows}
        assert {
            "source",
            "subject",
            "confidence",
            "sensitivity",
            "purpose",
            "statement_kind",
            "created_at",
            "updated_at",
            "supersedes_id",
            "is_suppressed",
            "namespace",
            "actor_id",
        } <= mc_cols
        assert {
            "source",
            "subject",
            "sensitivity",
            "purpose",
            "statement_kind",
            "updated_at",
            "is_suppressed",
            "superseded_by",
            "namespace",
            "actor_id",
            "confidence",
        } <= facts_cols
    finally:
        await backend.close()
