"""SQLite-backed temporal knowledge graph for entity/relationship tracking."""

import sqlite3
import json
import os
import threading
from datetime import datetime, timezone

from config import SQLITE_PATH

# Serialize writes — WAL allows concurrent readers; writers must not race.
GRAPH_WRITE_LOCK = threading.Lock()


def _ensure_dir():
    os.makedirs(os.path.dirname(SQLITE_PATH), exist_ok=True)


def get_db() -> sqlite3.Connection:
    _ensure_dir()
    conn = sqlite3.connect(SQLITE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_schema():
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS entities (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE COLLATE NOCASE,
            entity_type TEXT NOT NULL DEFAULT 'unknown',
            first_seen TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            mention_count INTEGER NOT NULL DEFAULT 1,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_entities_name ON entities(name COLLATE NOCASE);
        CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(entity_type);

        CREATE TABLE IF NOT EXISTS relationships (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source_entity_id INTEGER NOT NULL REFERENCES entities(id),
            target_entity_id INTEGER NOT NULL REFERENCES entities(id),
            relation_type TEXT NOT NULL,
            evidence TEXT NOT NULL,
            agent_id TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            confidence REAL NOT NULL DEFAULT 1.0,
            UNIQUE(source_entity_id, target_entity_id, relation_type)
        );
        CREATE INDEX IF NOT EXISTS idx_rel_source ON relationships(source_entity_id);
        CREATE INDEX IF NOT EXISTS idx_rel_target ON relationships(target_entity_id);

        CREATE TABLE IF NOT EXISTS facts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_id INTEGER REFERENCES entities(id),
            fact_text TEXT NOT NULL,
            source_file TEXT,
            agent_id TEXT,
            created_at TEXT NOT NULL,
            superseded_by INTEGER REFERENCES facts(id),
            is_active INTEGER NOT NULL DEFAULT 1
        );
        CREATE INDEX IF NOT EXISTS idx_facts_entity ON facts(entity_id);
        CREATE INDEX IF NOT EXISTS idx_facts_active ON facts(is_active);

        CREATE TABLE IF NOT EXISTS curator_state (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS audit_log (
            id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            action TEXT NOT NULL,
            memory_id TEXT,
            namespace TEXT,
            text_hash TEXT,
            version INTEGER,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp);
        CREATE INDEX IF NOT EXISTS idx_audit_agent ON audit_log(agent_id);
        CREATE INDEX IF NOT EXISTS idx_audit_memory ON audit_log(memory_id);
        CREATE INDEX IF NOT EXISTS idx_audit_namespace ON audit_log(namespace);

        CREATE TABLE IF NOT EXISTS memory_versions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            version INTEGER NOT NULL,
            agent_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            text_hash TEXT NOT NULL,
            operation TEXT NOT NULL,
            parent_versions TEXT DEFAULT '[]'
        );
        CREATE INDEX IF NOT EXISTS idx_memver_memory ON memory_versions(memory_id);
        CREATE INDEX IF NOT EXISTS idx_memver_agent ON memory_versions(agent_id);
    """)
        conn.commit()
        conn.close()


def upsert_entity(name: str, entity_type: str = "unknown", agent_id: str = "") -> int:
    now = datetime.now(timezone.utc).isoformat()
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        cur = conn.execute("SELECT id, mention_count FROM entities WHERE name = ? COLLATE NOCASE", (name,))
        row = cur.fetchone()
        if row:
            conn.execute(
                "UPDATE entities SET last_seen=?, mention_count=mention_count+1 WHERE id=?",
                (now, row["id"]),
            )
            conn.commit()
            conn.close()
            return row["id"]
        cur = conn.execute(
            "INSERT INTO entities (name, entity_type, first_seen, last_seen) VALUES (?,?,?,?)",
            (name, entity_type, now, now),
        )
        conn.commit()
        eid = cur.lastrowid
        conn.close()
        return eid


def add_relationship(source_id: int, target_id: int, rel_type: str, evidence: str, agent_id: str = ""):
    now = datetime.now(timezone.utc).isoformat()
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            conn.execute(
                """INSERT INTO relationships (source_entity_id, target_entity_id, relation_type, evidence, agent_id, created_at, updated_at)
                   VALUES (?,?,?,?,?,?,?)
                   ON CONFLICT(source_entity_id, target_entity_id, relation_type)
                   DO UPDATE SET evidence=excluded.evidence, updated_at=excluded.updated_at, confidence=min(confidence+0.1, 1.0)""",
                (source_id, target_id, rel_type, evidence, agent_id, now, now),
            )
            conn.commit()
        finally:
            conn.close()


def add_fact(entity_id: int, fact_text: str, source_file: str = "", agent_id: str = "") -> int:
    now = datetime.now(timezone.utc).isoformat()
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        cur = conn.execute(
            "INSERT INTO facts (entity_id, fact_text, source_file, agent_id, created_at) VALUES (?,?,?,?,?)",
            (entity_id, fact_text, source_file, agent_id, now),
        )
        conn.commit()
        fid = cur.lastrowid
        conn.close()
        return fid


def search_entities(query: str, limit: int = 10) -> list[dict]:
    conn = get_db()
    cur = conn.execute(
        "SELECT * FROM entities WHERE name LIKE ? COLLATE NOCASE ORDER BY mention_count DESC LIMIT ?",
        (f"%{query}%", limit),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_entity_facts(entity_id: int) -> list[dict]:
    conn = get_db()
    cur = conn.execute(
        "SELECT * FROM facts WHERE entity_id=? AND is_active=1 ORDER BY created_at DESC",
        (entity_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_entity_relationships(entity_id: int) -> list[dict]:
    conn = get_db()
    cur = conn.execute(
        """SELECT r.*, e1.name AS source_name, e2.name AS target_name
           FROM relationships r
           JOIN entities e1 ON r.source_entity_id=e1.id
           JOIN entities e2 ON r.target_entity_id=e2.id
           WHERE r.source_entity_id=? OR r.target_entity_id=?
           ORDER BY r.updated_at DESC""",
        (entity_id, entity_id),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def get_curator_state(key: str) -> str | None:
    conn = get_db()
    cur = conn.execute("SELECT value FROM curator_state WHERE key=?", (key,))
    row = cur.fetchone()
    conn.close()
    return row["value"] if row else None


def set_curator_state(key: str, value: str):
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        conn.execute(
            "INSERT INTO curator_state (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
        conn.commit()
        conn.close()
