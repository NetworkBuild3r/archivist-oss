#!/usr/bin/env python3
"""Migrate Archivist SQLite database to v1.11.0 schema.

Applies all schema changes introduced between v1.7 and v1.11:
  - namespace columns on entities, facts, relationships
  - memory_id column on facts
  - retention_class / aliases / valid_from / valid_until on entities/facts
  - provenance column on relationships
  - Namespace-scoped indexes
  - Rebuilds entities UNIQUE constraint from UNIQUE(name) → UNIQUE(name, namespace)

This script is idempotent — safe to run multiple times against the same database.
The same migrations run automatically on every service startup via graph.init_schema(),
so this script is only needed for pre-flight upgrades or auditing.

Usage:
    # Dry-run: show what would change without touching the DB
    python scripts/migrate_v1_11_0.py --dry-run

    # Check only: print current schema state and exit
    python scripts/migrate_v1_11_0.py --check

    # Apply to default path (/data/archivist/graph.db or $SQLITE_PATH)
    python scripts/migrate_v1_11_0.py

    # Apply to an explicit path
    python scripts/migrate_v1_11_0.py --db-path /path/to/graph.db

Exit codes:
    0  All migrations applied (or already up-to-date)
    1  One or more migrations failed
    2  --check found pending migrations (no changes made)
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys


# ---------------------------------------------------------------------------
# Schema definition
# ---------------------------------------------------------------------------

# Columns to add if missing. Matches _migrate_schema() in src/graph.py exactly.
COLUMN_MIGRATIONS: list[tuple[str, str, str]] = [
    ("facts",         "retention_class", "TEXT NOT NULL DEFAULT 'standard'"),
    ("entities",      "retention_class", "TEXT NOT NULL DEFAULT 'standard'"),
    ("entities",      "aliases",         "TEXT NOT NULL DEFAULT '[]'"),
    ("facts",         "valid_from",      "TEXT NOT NULL DEFAULT ''"),
    ("facts",         "valid_until",     "TEXT NOT NULL DEFAULT ''"),
    ("relationships", "provenance",      "TEXT NOT NULL DEFAULT 'unknown'"),
    ("entities",      "namespace",       "TEXT NOT NULL DEFAULT 'global'"),
    ("facts",         "namespace",       "TEXT NOT NULL DEFAULT 'global'"),
    ("relationships", "namespace",       "TEXT NOT NULL DEFAULT 'global'"),
    ("facts",         "memory_id",       "TEXT NOT NULL DEFAULT ''"),
]

INDEX_MIGRATIONS: list[str] = [
    "CREATE INDEX IF NOT EXISTS idx_facts_retention       ON facts(retention_class)",
    "CREATE INDEX IF NOT EXISTS idx_entities_retention    ON entities(retention_class)",
    "CREATE INDEX IF NOT EXISTS idx_facts_valid_from      ON facts(valid_from)",
    "CREATE INDEX IF NOT EXISTS idx_entities_namespace    ON entities(namespace)",
    "CREATE INDEX IF NOT EXISTS idx_facts_namespace       ON facts(namespace)",
    "CREATE INDEX IF NOT EXISTS idx_relationships_namespace ON relationships(namespace)",
    "CREATE INDEX IF NOT EXISTS idx_facts_memory_id       ON facts(memory_id)",
]

# Target DDL for entities after the UNIQUE-constraint rebuild.
_ENTITIES_NEW_DDL = """
    CREATE TABLE entities_new (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        name          TEXT    NOT NULL COLLATE NOCASE,
        entity_type   TEXT    NOT NULL DEFAULT 'unknown',
        first_seen    TEXT    NOT NULL,
        last_seen     TEXT    NOT NULL,
        mention_count INTEGER NOT NULL DEFAULT 1,
        metadata      TEXT             DEFAULT '{}',
        retention_class TEXT  NOT NULL DEFAULT 'standard',
        aliases       TEXT    NOT NULL DEFAULT '[]',
        namespace     TEXT    NOT NULL DEFAULT 'global',
        UNIQUE(name, namespace)
    )
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _needs_unique_constraint_rebuild(conn: sqlite3.Connection) -> bool:
    """Return True if entities still has the old UNIQUE(name) constraint."""
    if "namespace" not in _columns(conn, "entities"):
        return False

    # Check inline constraint in CREATE TABLE sql
    create_sql: str = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='entities'"
    ).fetchone()[0] or ""
    if "UNIQUE" in create_sql:
        after_unique = create_sql.split("UNIQUE", 1)[-1]
        if "namespace" in after_unique:
            return False

    # Check for a separate UNIQUE index on both columns
    idx_rows = conn.execute(
        "SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name='entities' AND sql IS NOT NULL"
    ).fetchall()
    if any("namespace" in (r[0] or "") for r in idx_rows):
        return False

    return True


def _check_pending(conn: sqlite3.Connection) -> dict[str, list[str]]:
    """Return a dict of pending changes (empty means up-to-date)."""
    pending: dict[str, list[str]] = {"columns": [], "unique_constraint": []}

    for table, column, _ in COLUMN_MIGRATIONS:
        if column not in _columns(conn, table):
            pending["columns"].append(f"{table}.{column}")

    if _needs_unique_constraint_rebuild(conn):
        pending["unique_constraint"].append("entities: UNIQUE(name) → UNIQUE(name, namespace)")

    return pending


# ---------------------------------------------------------------------------
# Migration steps
# ---------------------------------------------------------------------------

def _apply_columns(conn: sqlite3.Connection, dry_run: bool) -> bool:
    print("\nStep 1: Column migrations")
    ok = True
    for table, column, col_def in COLUMN_MIGRATIONS:
        if column in _columns(conn, table):
            print(f"  ✓  {table}.{column} — already present")
            continue
        label = f"ALTER TABLE {table} ADD COLUMN {column} {col_def}"
        print(f"  +  {label}")
        if dry_run:
            continue
        try:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {col_def}")
            conn.commit()
        except Exception as exc:
            print(f"     ✗  FAILED: {exc}")
            conn.rollback()
            ok = False
    return ok


def _apply_indexes(conn: sqlite3.Connection, dry_run: bool) -> bool:
    print("\nStep 2: Index migrations")
    ok = True
    for sql in INDEX_MIGRATIONS:
        idx_name = sql.split("idx_")[1].split(" ")[0] if "idx_" in sql else sql[:40]
        print(f"  +  idx_{idx_name}")
        if dry_run:
            continue
        try:
            conn.execute(sql)
            conn.commit()
        except Exception as exc:
            print(f"     ✗  FAILED: {exc}")
            ok = False
    return ok


def _apply_unique_constraint(conn: sqlite3.Connection, dry_run: bool) -> bool:
    print("\nStep 3: entities UNIQUE constraint rebuild")
    if not _needs_unique_constraint_rebuild(conn):
        print("  ✓  UNIQUE(name, namespace) — already in place")
        return True

    print("  ↺  Rebuilding entities table: UNIQUE(name) → UNIQUE(name, namespace)")
    if dry_run:
        print("     [dry-run] would rebuild entities table")
        return True

    try:
        # Safety: remove any leftover temp table from a previously failed run
        conn.execute("DROP TABLE IF EXISTS entities_new")

        conn.execute(_ENTITIES_NEW_DDL)
        conn.execute("""
            INSERT INTO entities_new
                (id, name, entity_type, first_seen, last_seen,
                 mention_count, metadata, retention_class, aliases, namespace)
            SELECT id, name, entity_type, first_seen, last_seen,
                   mention_count, metadata, retention_class, aliases, namespace
            FROM entities
        """)
        conn.execute("DROP TABLE entities")
        conn.execute("ALTER TABLE entities_new RENAME TO entities")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_entities_namespace ON entities(namespace)")
        conn.commit()
        print("  ✓  entities rebuilt with UNIQUE(name, namespace)")
        return True
    except Exception as exc:
        print(f"  ✗  FAILED: {exc}")
        conn.rollback()
        return False


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def resolve_db_path(cli_path: str | None) -> str:
    if cli_path:
        return cli_path
    env_path = os.getenv("SQLITE_PATH", "").strip()
    if env_path:
        return env_path
    return "/data/archivist/graph.db"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate Archivist SQLite DB to v1.11.0 schema.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--db-path",
        metavar="PATH",
        default=None,
        help="Path to graph.db (default: $SQLITE_PATH or /data/archivist/graph.db)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without modifying the database",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Print pending migrations and exit 2 if any are needed, 0 if up-to-date",
    )
    args = parser.parse_args()

    db_path = resolve_db_path(args.db_path)

    print("Archivist DB Migration — v1.11.0")
    print("=" * 50)
    print(f"Database : {db_path}")

    if not os.path.exists(db_path):
        print(f"\n✗  Database not found: {db_path}")
        print("   Pass --db-path or set $SQLITE_PATH.")
        return 1

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # ── Check mode ───────────────────────────────────────────────────────────
    if args.check:
        pending = _check_pending(conn)
        conn.close()
        total = sum(len(v) for v in pending.values())
        if total == 0:
            print("\n✅ Schema is up-to-date — no migrations needed.")
            return 0
        print(f"\n⚠  {total} pending migration(s):")
        for col in pending["columns"]:
            print(f"   + column  : {col}")
        for msg in pending["unique_constraint"]:
            print(f"   + rebuild : {msg}")
        return 2

    # ── Dry-run banner ───────────────────────────────────────────────────────
    if args.dry_run:
        print("\n[DRY-RUN] No changes will be written.\n")

    # ── Apply ────────────────────────────────────────────────────────────────
    results = [
        _apply_columns(conn, args.dry_run),
        _apply_indexes(conn, args.dry_run),
        _apply_unique_constraint(conn, args.dry_run),
    ]
    conn.close()

    print()
    if all(results):
        tag = "[dry-run] " if args.dry_run else ""
        print(f"✅ {tag}Migration to v1.11.0 completed successfully.")
        return 0
    else:
        print("❌ One or more migration steps failed — see output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
