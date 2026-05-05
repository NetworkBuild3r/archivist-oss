"""Skill registry, version tracking, lessons learned, usage events, and health scoring.

Tracks the operational reality of skills (MCP tools) that agents use — which version,
what broke, what was learned. Designed for a world where skills are shared across
organizational boundaries via MCP connections.
"""

import json
import logging
import uuid
from datetime import UTC, datetime

from archivist.storage.graph import schema_guard

logger = logging.getLogger("archivist.skills")

_ensure_skill_schema = schema_guard("""
    CREATE TABLE IF NOT EXISTS skills (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        provider TEXT NOT NULL DEFAULT '',
        mcp_endpoint TEXT DEFAULT '',
        current_version TEXT NOT NULL DEFAULT '0.0.0',
        status TEXT NOT NULL DEFAULT 'active',
        description TEXT DEFAULT '',
        registered_by TEXT NOT NULL,
        registered_at TEXT NOT NULL,
        updated_at TEXT NOT NULL,
        metadata TEXT DEFAULT '{}',
        UNIQUE(name, provider)
    );
    CREATE INDEX IF NOT EXISTS idx_skills_name ON skills(name);
    CREATE INDEX IF NOT EXISTS idx_skills_provider ON skills(provider);
    CREATE INDEX IF NOT EXISTS idx_skills_status ON skills(status);

    CREATE TABLE IF NOT EXISTS skill_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        skill_id TEXT NOT NULL REFERENCES skills(id),
        version TEXT NOT NULL,
        changelog TEXT DEFAULT '',
        breaking_changes TEXT DEFAULT '',
        observed_at TEXT NOT NULL,
        reported_by TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'active',
        UNIQUE(skill_id, version)
    );
    CREATE INDEX IF NOT EXISTS idx_sv_skill ON skill_versions(skill_id);

    CREATE TABLE IF NOT EXISTS skill_lessons (
        id TEXT PRIMARY KEY,
        skill_id TEXT NOT NULL REFERENCES skills(id),
        lesson_type TEXT NOT NULL DEFAULT 'general',
        title TEXT NOT NULL,
        content TEXT NOT NULL,
        skill_version TEXT DEFAULT '',
        agent_id TEXT NOT NULL,
        created_at TEXT NOT NULL,
        upvotes INTEGER NOT NULL DEFAULT 0
    );
    CREATE INDEX IF NOT EXISTS idx_sl_skill ON skill_lessons(skill_id);
    CREATE INDEX IF NOT EXISTS idx_sl_type ON skill_lessons(lesson_type);

    CREATE TABLE IF NOT EXISTS skill_events (
        id TEXT PRIMARY KEY,
        skill_id TEXT NOT NULL REFERENCES skills(id),
        agent_id TEXT NOT NULL,
        event_type TEXT NOT NULL DEFAULT 'invocation',
        outcome TEXT NOT NULL DEFAULT 'unknown',
        skill_version TEXT DEFAULT '',
        duration_ms INTEGER,
        error_message TEXT DEFAULT '',
        trajectory_id TEXT DEFAULT '',
        created_at TEXT NOT NULL,
        metadata TEXT DEFAULT '{}'
    );
    CREATE INDEX IF NOT EXISTS idx_se_skill ON skill_events(skill_id);
    CREATE INDEX IF NOT EXISTS idx_se_agent ON skill_events(agent_id);
    CREATE INDEX IF NOT EXISTS idx_se_outcome ON skill_events(outcome);

    CREATE TABLE IF NOT EXISTS skill_relations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        skill_a TEXT NOT NULL REFERENCES skills(id),
        skill_b TEXT NOT NULL REFERENCES skills(id),
        relation_type TEXT NOT NULL,
        confidence REAL NOT NULL DEFAULT 1.0,
        evidence TEXT DEFAULT '',
        created_by TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS idx_sr_a ON skill_relations(skill_a);
    CREATE INDEX IF NOT EXISTS idx_sr_b ON skill_relations(skill_b);
    CREATE INDEX IF NOT EXISTS idx_sr_type ON skill_relations(relation_type);
""")


async def register_skill(
    name: str,
    provider: str = "",
    mcp_endpoint: str = "",
    version: str = "0.0.0",
    description: str = "",
    registered_by: str = "",
    metadata: dict | None = None,
) -> dict:
    """Register a new skill or update an existing one."""
    from archivist.storage.sqlite_pool import pool

    _ensure_skill_schema()
    now = datetime.now(UTC).isoformat()

    async with pool.write() as conn:
        row = await conn.fetchone(
            "SELECT id, current_version FROM skills WHERE name=? AND provider=?",
            (name, provider),
        )

        if row:
            skill_id = row["id"]
            old_version = row["current_version"]
            await conn.execute(
                """UPDATE skills SET current_version=?, mcp_endpoint=?, description=?,
                   status='active', updated_at=?, metadata=? WHERE id=?""",
                (version, mcp_endpoint, description, now, json.dumps(metadata or {}), skill_id),
            )
            if version != old_version:
                await conn.execute(
                    """INSERT INTO skill_versions
                       (skill_id, version, observed_at, reported_by)
                       VALUES (?,?,?,?)
                       ON CONFLICT(skill_id, version) DO NOTHING""",
                    (skill_id, version, now, registered_by),
                )
            return {"skill_id": skill_id, "action": "updated", "version": version}
        else:
            skill_id = str(uuid.uuid4())
            await conn.execute(
                """INSERT INTO skills
                   (id, name, provider, mcp_endpoint, current_version, description,
                    registered_by, registered_at, updated_at, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (
                    skill_id,
                    name,
                    provider,
                    mcp_endpoint,
                    version,
                    description,
                    registered_by,
                    now,
                    now,
                    json.dumps(metadata or {}),
                ),
            )
            await conn.execute(
                """INSERT INTO skill_versions
                   (skill_id, version, observed_at, reported_by)
                   VALUES (?,?,?,?)""",
                (skill_id, version, now, registered_by),
            )
            return {"skill_id": skill_id, "action": "registered", "version": version}


async def record_version(
    skill_id: str,
    version: str,
    changelog: str = "",
    breaking_changes: str = "",
    reported_by: str = "",
) -> dict:
    """Record a new version observation for a skill."""
    from archivist.storage.sqlite_pool import pool

    _ensure_skill_schema()
    now = datetime.now(UTC).isoformat()

    async with pool.write() as conn:
        await conn.execute(
            """INSERT INTO skill_versions (skill_id, version, changelog, breaking_changes, observed_at, reported_by)
               VALUES (?,?,?,?,?,?)
               ON CONFLICT(skill_id, version) DO UPDATE SET
               changelog=excluded.changelog, breaking_changes=excluded.breaking_changes""",
            (skill_id, version, changelog, breaking_changes, now, reported_by),
        )
        await conn.execute(
            "UPDATE skills SET current_version=?, updated_at=? WHERE id=?",
            (version, now, skill_id),
        )

    return {"skill_id": skill_id, "version": version, "recorded_at": now}


async def add_lesson(
    skill_id: str,
    title: str,
    content: str,
    lesson_type: str = "general",
    skill_version: str = "",
    agent_id: str = "",
) -> str:
    """Add a lesson learned to a skill."""
    from archivist.storage.sqlite_pool import pool

    _ensure_skill_schema()
    lesson_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    async with pool.write() as conn:
        await conn.execute(
            """INSERT INTO skill_lessons
               (id, skill_id, lesson_type, title, content, skill_version, agent_id, created_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (lesson_id, skill_id, lesson_type, title, content, skill_version, agent_id, now),
        )

    return lesson_id


async def get_lessons(skill_id: str, lesson_type: str = "", limit: int = 20) -> list[dict]:
    """Retrieve lessons learned for a skill."""
    from archivist.storage.sqlite_pool import pool

    _ensure_skill_schema()
    async with pool.read() as conn:
        if lesson_type:
            rows = await conn.fetchall(
                "SELECT * FROM skill_lessons WHERE skill_id=? AND lesson_type=? ORDER BY created_at DESC LIMIT ?",
                (skill_id, lesson_type, limit),
            )
        else:
            rows = await conn.fetchall(
                "SELECT * FROM skill_lessons WHERE skill_id=? ORDER BY created_at DESC LIMIT ?",
                (skill_id, limit),
            )
    return [dict(r) for r in rows]


async def log_skill_event(
    skill_id: str,
    agent_id: str,
    outcome: str,
    event_type: str = "invocation",
    skill_version: str = "",
    duration_ms: int | None = None,
    error_message: str = "",
    trajectory_id: str = "",
    metadata: dict | None = None,
) -> str:
    """Log a skill usage event (invocation, failure, etc.)."""
    from archivist.storage.sqlite_pool import pool

    _ensure_skill_schema()
    event_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()

    async with pool.write() as conn:
        if outcome == "failure" and not skill_version:
            row = await conn.fetchone("SELECT current_version FROM skills WHERE id=?", (skill_id,))
            if row:
                skill_version = row["current_version"]

        await conn.execute(
            """INSERT INTO skill_events
               (id, skill_id, agent_id, event_type, outcome, skill_version,
                duration_ms, error_message, trajectory_id, created_at, metadata)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (
                event_id,
                skill_id,
                agent_id,
                event_type,
                outcome,
                skill_version,
                duration_ms,
                error_message,
                trajectory_id,
                now,
                json.dumps(metadata or {}),
            ),
        )

    return event_id


async def get_skill_health(skill_id: str, window_days: int = 30) -> dict:
    """Compute health metrics for a skill from its event history."""
    from archivist.core.config import GRAPH_BACKEND
    from archivist.storage.sqlite_pool import pool

    _ensure_skill_schema()

    if (GRAPH_BACKEND or "sqlite").lower() == "postgres":
        window_expr = f"created_at >= NOW() - INTERVAL '{window_days} days'"
        window_params: list = []
    else:
        window_expr = "created_at >= datetime('now', ?)"
        window_params = [f"-{window_days} days"]

    async with pool.read() as conn:
        skill_row = await conn.fetchone("SELECT * FROM skills WHERE id=?", (skill_id,))
        if not skill_row:
            return {"error": "skill_not_found", "skill_id": skill_id}

        skill = dict(skill_row)

        outcome_rows = await conn.fetchall(
            f"SELECT outcome, COUNT(*) as cnt FROM skill_events "
            f"WHERE skill_id=? AND {window_expr} GROUP BY outcome",
            [skill_id] + window_params,
        )
        outcome_counts = {row["outcome"]: row["cnt"] for row in outcome_rows}

        total = sum(outcome_counts.values())
        successes = outcome_counts.get("success", 0)
        failures = outcome_counts.get("failure", 0)
        success_rate = successes / total if total > 0 else None

        last_success_row = await conn.fetchone(
            "SELECT created_at FROM skill_events WHERE skill_id=? AND outcome='success' ORDER BY created_at DESC LIMIT 1",
            (skill_id,),
        )
        last_success = last_success_row["created_at"] if last_success_row else None

        last_failure_row = await conn.fetchone(
            "SELECT created_at, error_message, skill_version FROM skill_events WHERE skill_id=? AND outcome='failure' ORDER BY created_at DESC LIMIT 1",
            (skill_id,),
        )
        last_failure = dict(last_failure_row) if last_failure_row else None

        lessons_row = await conn.fetchone(
            "SELECT COUNT(*) as cnt FROM skill_lessons WHERE skill_id=?",
            (skill_id,),
        )
        lessons_count = lessons_row["cnt"] if lessons_row else 0

        avg_row = await conn.fetchone(
            f"SELECT AVG(duration_ms) as avg_ms FROM skill_events "
            f"WHERE skill_id=? AND duration_ms IS NOT NULL AND {window_expr}",
            [skill_id] + window_params,
        )
        avg_duration_ms = round(avg_row["avg_ms"]) if avg_row and avg_row["avg_ms"] else None

        version_rows = await conn.fetchall(
            "SELECT version, breaking_changes, status, observed_at FROM skill_versions WHERE skill_id=? ORDER BY observed_at DESC LIMIT 5",
            (skill_id,),
        )
        versions = [dict(r) for r in version_rows]

    health = "healthy"
    if success_rate is not None and success_rate < 0.5:
        health = "degraded"
    elif success_rate is not None and success_rate < 0.8:
        health = "warning"
    if skill["status"] in ("broken", "deprecated"):
        health = skill["status"]

    return {
        "skill_id": skill_id,
        "name": skill["name"],
        "provider": skill["provider"],
        "current_version": skill["current_version"],
        "status": skill["status"],
        "health": health,
        "window_days": window_days,
        "total_events": total,
        "successes": successes,
        "failures": failures,
        "success_rate": round(success_rate, 3) if success_rate is not None else None,
        "last_success": last_success,
        "last_failure": last_failure,
        "lessons_count": lessons_count,
        "avg_duration_ms": avg_duration_ms,
        "recent_versions": versions,
    }


async def find_skill(name: str, provider: str = "") -> dict | None:
    """Look up a skill by name (and optionally provider)."""
    from archivist.storage.sqlite_pool import pool

    _ensure_skill_schema()
    async with pool.read() as conn:
        if provider:
            row = await conn.fetchone(
                "SELECT * FROM skills WHERE name=? AND provider=?",
                (name, provider),
            )
        else:
            row = await conn.fetchone(
                "SELECT * FROM skills WHERE name=? ORDER BY updated_at DESC LIMIT 1",
                (name,),
            )
    return dict(row) if row else None


async def list_skills(status: str = "", limit: int = 100) -> list[dict]:
    """Return all registered skills, optionally filtered by status."""
    from archivist.storage.sqlite_pool import pool

    _ensure_skill_schema()
    async with pool.read() as conn:
        if status:
            rows = await conn.fetchall(
                "SELECT * FROM skills WHERE status=? ORDER BY name LIMIT ?",
                (status, limit),
            )
        else:
            rows = await conn.fetchall("SELECT * FROM skills ORDER BY name LIMIT ?", (limit,))
    return [dict(r) for r in rows]


async def update_skill_status(skill_id: str, status: str) -> None:
    """Update a skill's status (active, deprecated, disabled)."""
    from archivist.storage.sqlite_pool import pool

    _ensure_skill_schema()
    now = datetime.now(UTC).isoformat()
    async with pool.write() as conn:
        await conn.execute(
            "UPDATE skills SET status=?, updated_at=? WHERE id=?",
            (status, now, skill_id),
        )


# ── Skill relation graph (v1.0) ─────────────────────────────────────────────

VALID_RELATION_TYPES = {"similar_to", "depend_on", "compose_with", "replaced_by"}


async def add_skill_relation(
    skill_a_id: str,
    skill_b_id: str,
    relation_type: str,
    confidence: float = 1.0,
    evidence: str = "",
    created_by: str = "",
) -> int:
    """Create or update a relation between two skills."""
    from archivist.storage.sqlite_pool import pool

    _ensure_skill_schema()
    if relation_type not in VALID_RELATION_TYPES:
        raise ValueError(
            f"Invalid relation_type: {relation_type}. Must be one of {VALID_RELATION_TYPES}"
        )

    now = datetime.now(UTC).isoformat()

    async with pool.write() as conn:
        existing = await conn.fetchone(
            "SELECT id FROM skill_relations WHERE skill_a=? AND skill_b=? AND relation_type=?",
            (skill_a_id, skill_b_id, relation_type),
        )

        if existing:
            await conn.execute(
                "UPDATE skill_relations SET confidence=?, evidence=?, created_by=?, created_at=? WHERE id=?",
                (confidence, evidence, created_by, now, existing["id"]),
            )
            return existing["id"]
        else:
            # Use RETURNING id for Postgres; fall back to a subsequent SELECT for SQLite
            # (asyncpg_backend translates ? → $N; AsyncpgConnection.execute returns status string)
            from archivist.core.config import GRAPH_BACKEND

            if (GRAPH_BACKEND or "sqlite").lower() == "postgres":
                row = await conn.fetchone(
                    """INSERT INTO skill_relations (skill_a, skill_b, relation_type, confidence, evidence, created_by, created_at)
                       VALUES (?,?,?,?,?,?,?) RETURNING id""",
                    (skill_a_id, skill_b_id, relation_type, confidence, evidence, created_by, now),
                )
                return row["id"] if row else 0
            else:
                cur = await conn.execute(
                    """INSERT INTO skill_relations (skill_a, skill_b, relation_type, confidence, evidence, created_by, created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (skill_a_id, skill_b_id, relation_type, confidence, evidence, created_by, now),
                )
                return cur.lastrowid


async def get_skill_relations(skill_id: str, depth: int = 1) -> list[dict]:
    """Get the relation graph for a skill. depth=1 for direct, depth>1 for multi-hop."""
    from archivist.storage.sqlite_pool import pool

    _ensure_skill_schema()

    visited: set[tuple] = set()
    frontier = {skill_id}
    all_relations: list[dict] = []

    async with pool.read() as conn:
        for _ in range(depth):
            if not frontier:
                break
            ids = list(frontier)
            placeholders = ",".join("?" for _ in ids)

            rows = await conn.fetchall(
                f"""SELECT sr.*, sa.name as skill_a_name, sb.name as skill_b_name
                    FROM skill_relations sr
                    JOIN skills sa ON sr.skill_a = sa.id
                    JOIN skills sb ON sr.skill_b = sb.id
                    WHERE sr.skill_a IN ({placeholders}) OR sr.skill_b IN ({placeholders})""",
                ids + ids,
            )

            next_frontier: set[str] = set()
            for r in rows:
                rel = dict(r)
                rel_key = (rel["skill_a"], rel["skill_b"], rel["relation_type"])
                if rel_key not in visited:
                    visited.add(rel_key)
                    all_relations.append(rel)
                    next_frontier.add(rel["skill_a"])
                    next_frontier.add(rel["skill_b"])

            frontier = next_frontier - {skill_id} - set(ids)

    return all_relations


async def get_skill_substitutes(skill_id: str) -> list[dict]:
    """Find skills that can substitute for this one (similar_to or replaced_by)."""
    from archivist.storage.sqlite_pool import pool

    _ensure_skill_schema()
    async with pool.read() as conn:
        rows = await conn.fetchall(
            """SELECT s.*, sr.relation_type, sr.confidence
               FROM skill_relations sr
               JOIN skills s ON (s.id = sr.skill_b AND sr.skill_a = ?)
                             OR (s.id = sr.skill_a AND sr.skill_b = ?)
               WHERE sr.relation_type IN ('similar_to', 'replaced_by')
               AND s.status = 'active'
               ORDER BY sr.confidence DESC""",
            (skill_id, skill_id),
        )
    return [dict(r) for r in rows]
