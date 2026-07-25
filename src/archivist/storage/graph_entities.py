"""Entity/relationship/fact CRUD and curator key-value state.

Split from the former monolithic ``storage/graph.py`` (INIT-001/SPEC-003).
Provenance: INIT-001/SPEC-003.
"""

from __future__ import annotations

import logging
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import aiosqlite

from archivist.storage.graph_schema import _is_postgres

logger = logging.getLogger("archivist.graph")

_RETENTION_RANK = {"ephemeral": 0, "standard": 1, "durable": 2, "permanent": 3}


async def upsert_entity(
    name: str,
    entity_type: str = "unknown",
    agent_id: str = "",
    retention_class: str = "standard",
    namespace: str = "global",
    actor_id: str = "",
    actor_type: str = "",
    conn: aiosqlite.Connection | None = None,
) -> int:
    """Insert or update an entity, returning its integer ID.

    Args:
        conn: Optional open ``aiosqlite.Connection``.  When provided the write
            joins the caller's transaction; when ``None`` a fresh
            ``pool.write()`` lock is acquired.
    """
    from archivist.storage.sqlite_pool import pool

    now = datetime.now(UTC).isoformat()

    async def _run(c: aiosqlite.Connection) -> int:
        cur = await c.execute(
            "SELECT id, mention_count, retention_class FROM entities WHERE name = ? AND namespace = ?",
            (name, namespace),
        )
        row = await cur.fetchone()
        if row:
            existing_rc = row["retention_class"] if "retention_class" in row.keys() else "standard"
            new_rc = (
                retention_class
                if _RETENTION_RANK.get(retention_class, 1) > _RETENTION_RANK.get(existing_rc, 1)
                else existing_rc
            )
            await c.execute(
                "UPDATE entities SET last_seen=?, mention_count=mention_count+1, retention_class=? WHERE id=?",
                (now, new_rc, row["id"]),
            )
            return row["id"]
        if _is_postgres():
            new_id = await c.fetchval(
                "INSERT INTO entities (name, entity_type, first_seen, last_seen, retention_class, namespace, actor_id, actor_type) "
                "VALUES (?,?,?,?,?,?,?,?) RETURNING id",
                (name, entity_type, now, now, retention_class, namespace, actor_id, actor_type),
            )
            return new_id
        cur2 = await c.execute(
            "INSERT INTO entities (name, entity_type, first_seen, last_seen, retention_class, namespace, actor_id, actor_type) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (name, entity_type, now, now, retention_class, namespace, actor_id, actor_type),
        )
        return cur2.lastrowid

    if conn is not None:
        return await _run(conn)
    async with pool.write() as c:
        return await _run(c)


async def add_relationship(
    source_id: int,
    target_id: int,
    rel_type: str,
    evidence: str,
    agent_id: str = "",
    provenance: str = "unknown",
    namespace: str = "global",
):
    """Insert or update a relationship between two entities."""
    from archivist.storage.sqlite_pool import pool

    now = datetime.now(UTC).isoformat()
    # SQLite uses MIN() as a scalar; Postgres uses LEAST() (MIN is aggregate-only there).
    # Qualify the table name to avoid column ambiguity in the DO UPDATE context.
    clamp_fn = "LEAST" if _is_postgres() else "MIN"
    async with pool.write() as conn:
        await conn.execute(
            f"""INSERT INTO relationships (source_entity_id, target_entity_id, relation_type,
               evidence, agent_id, created_at, updated_at, provenance, namespace)
               VALUES (?,?,?,?,?,?,?,?,?)
               ON CONFLICT(source_entity_id, target_entity_id, relation_type)
               DO UPDATE SET evidence=excluded.evidence, updated_at=excluded.updated_at,
               confidence={clamp_fn}(relationships.confidence+0.1, 1.0), provenance=excluded.provenance""",
            (source_id, target_id, rel_type, evidence, agent_id, now, now, provenance, namespace),
        )


def _word_set(text: str) -> set[str]:
    """Extract lowercase word tokens for overlap comparison."""
    return {w for w in text.lower().split() if len(w) >= 2}


_DATE_IN_PATH_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")


async def add_fact(
    entity_id: int,
    fact_text: str,
    source_file: str = "",
    agent_id: str = "",
    retention_class: str = "standard",
    valid_from: str = "",
    valid_until: str = "",
    namespace: str = "global",
    memory_id: str = "",
    confidence: float = 1.0,
    provenance: str = "unknown",
    actor_id: str = "",
    conn: aiosqlite.Connection | None = None,
) -> int:
    """Insert a new fact and auto-supersede overlapping existing facts.

    Args:
        conn: Optional open ``aiosqlite.Connection``.  When provided the write
            joins the caller's transaction; when ``None`` a fresh
            ``pool.write()`` lock is acquired.
    """
    from archivist.storage.sqlite_pool import pool

    now = datetime.now(UTC).isoformat()
    new_words = _word_set(fact_text)

    if not valid_from and source_file:
        _m = _DATE_IN_PATH_RE.search(source_file)
        if _m:
            valid_from = _m.group(1)

    async def _run(c: aiosqlite.Connection) -> int:
        if _is_postgres():
            fid = await c.fetchval(
                "INSERT INTO facts (entity_id, fact_text, source_file, agent_id, created_at, "
                "retention_class, valid_from, valid_until, namespace, memory_id, confidence, provenance, actor_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?) RETURNING id",
                (
                    entity_id,
                    fact_text,
                    source_file,
                    agent_id,
                    now,
                    retention_class,
                    valid_from,
                    valid_until,
                    namespace,
                    memory_id,
                    confidence,
                    provenance,
                    actor_id,
                ),
            )
        else:
            cur = await c.execute(
                "INSERT INTO facts (entity_id, fact_text, source_file, agent_id, created_at, "
                "retention_class, valid_from, valid_until, namespace, memory_id, confidence, provenance, actor_id) "
                "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (
                    entity_id,
                    fact_text,
                    source_file,
                    agent_id,
                    now,
                    retention_class,
                    valid_from,
                    valid_until,
                    namespace,
                    memory_id,
                    confidence,
                    provenance,
                    actor_id,
                ),
            )
            fid = cur.lastrowid

        if new_words:
            old_facts_cur = await c.execute(
                "SELECT id, fact_text FROM facts "
                "WHERE entity_id=? AND is_active=1 AND id!=? AND superseded_by IS NULL",
                (entity_id, fid),
            )
            old_facts = await old_facts_cur.fetchall()

            superseded_ids = []
            for old in old_facts:
                old_words = _word_set(old["fact_text"])
                if not old_words:
                    continue
                overlap = len(new_words & old_words) / max(len(old_words), 1)
                if overlap >= 0.6:
                    superseded_ids.append(old["id"])

            if superseded_ids:
                placeholders = ",".join("?" for _ in superseded_ids)
                await c.execute(
                    f"UPDATE facts SET superseded_by=? WHERE id IN ({placeholders})",
                    [fid] + superseded_ids,
                )

        return fid

    if conn is not None:
        return await _run(conn)
    async with pool.write() as c:
        return await _run(c)


async def invalidate_fact(fact_id: int, ended: str = ""):
    """Mark a fact as no longer valid by setting ``valid_until``.

    If *ended* is empty the current UTC date is used.
    """
    from archivist.storage.sqlite_pool import pool

    if not ended:
        ended = datetime.now(UTC).strftime("%Y-%m-%d")
    async with pool.write() as conn:
        await conn.execute(
            "UPDATE facts SET valid_until=? WHERE id=?",
            (ended, fact_id),
        )


async def supersede_fact(old_fact_id: int, new_fact_id: int):
    """Explicitly mark an old fact as superseded by a newer one."""
    from archivist.storage.sqlite_pool import pool

    async with pool.write() as conn:
        await conn.execute(
            "UPDATE facts SET superseded_by=? WHERE id=?",
            (new_fact_id, old_fact_id),
        )


def _normalize(text: str) -> str:
    """Lowercase, strip non-alphanumeric except hyphens/underscores."""
    return re.sub(r"[^\w\s\-]", "", text.lower()).strip()


async def search_entities(query: str, limit: int = 10, namespace: str = "") -> list[dict]:
    """Search entities by name or aliases (case-insensitive, normalized)."""
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        norm_q = _normalize(query)
        # On Postgres the entities.name column is citext so LIKE is already
        # case-insensitive; COLLATE NOCASE is stripped by the SQL translator.
        # On SQLite COLLATE NOCASE is kept for non-ASCII correctness.
        collate = "" if _is_postgres() else " COLLATE NOCASE"
        if namespace:
            cur = await conn.execute(
                f"SELECT * FROM entities "
                f"WHERE (name LIKE ?{collate} OR aliases LIKE ?{collate}) "
                f"AND namespace = ? "
                f"ORDER BY mention_count DESC LIMIT ?",
                (f"%{query}%", f"%{norm_q}%", namespace, limit),
            )
        else:
            cur = await conn.execute(
                f"SELECT * FROM entities "
                f"WHERE name LIKE ?{collate} OR aliases LIKE ?{collate} "
                f"ORDER BY mention_count DESC LIMIT ?",
                (f"%{query}%", f"%{norm_q}%", limit),
            )
        return [dict(r) for r in await cur.fetchall()]


async def add_entity_alias(entity_id: int, alias: str):
    """Add an alias to an entity (idempotent)."""
    import json as _json

    from archivist.storage.sqlite_pool import pool

    norm = _normalize(alias)
    if not norm:
        return
    async with pool.write() as conn:
        row = await (
            await conn.execute("SELECT aliases FROM entities WHERE id=?", (entity_id,))
        ).fetchone()
        if row:
            try:
                current = _json.loads(row["aliases"])
            except Exception:
                current = []
            if norm not in current:
                current.append(norm)
                await conn.execute(
                    "UPDATE entities SET aliases=? WHERE id=?",
                    (_json.dumps(current), entity_id),
                )


async def get_entity_by_id(entity_id: int) -> dict | None:
    """Return entity dict by primary key, or None if not found."""
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        row = await (
            await conn.execute("SELECT * FROM entities WHERE id=?", (entity_id,))
        ).fetchone()
        return dict(row) if row else None


async def get_entity_facts(
    entity_id: int, include_superseded: bool = False, as_of: str = ""
) -> list[dict]:
    """Get active facts for an entity.

    Non-superseded facts come first. Superseded facts are included only when
    ``include_superseded`` is True (useful for history views).

    When ``as_of`` is an ISO-date string (e.g. ``"2025-03-15"``), only facts
    whose validity window contains that date are returned.  Dateless facts
    (empty ``valid_from``) are always included.
    """
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        base = "SELECT * FROM facts WHERE entity_id=? AND is_active=1"
        params: list = [entity_id]

        if not include_superseded:
            base += " AND superseded_by IS NULL"

        if as_of:
            base += " AND (valid_from = '' OR valid_from <= ?)"
            params.append(as_of)
            base += " AND (valid_until = '' OR valid_until > ?)"
            params.append(as_of)

        if include_superseded:
            base += " ORDER BY (superseded_by IS NOT NULL), created_at DESC"
        else:
            base += " ORDER BY created_at DESC"

        cur = await conn.execute(base, params)
        results = []
        for r in await cur.fetchall():
            d = dict(r)
            d["is_current"] = d.get("superseded_by") is None
            results.append(d)
        return results


async def get_entity_relationships(entity_id: int) -> list[dict]:
    """Return all relationships involving the given entity."""
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        cur = await conn.execute(
            """SELECT r.*, e1.name AS source_name, e2.name AS target_name
               FROM relationships r
               JOIN entities e1 ON r.source_entity_id=e1.id
               JOIN entities e2 ON r.target_entity_id=e2.id
               WHERE r.source_entity_id=? OR r.target_entity_id=?
               ORDER BY r.updated_at DESC""",
            (entity_id, entity_id),
        )
        return [dict(r) for r in await cur.fetchall()]


async def get_entity_facts_bulk(entity_ids: list[int], as_of: str = "") -> dict[int, list[dict]]:
    """Fetch active, non-superseded facts for multiple entities in one query."""
    if not entity_ids:
        return {}
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        placeholders = ",".join("?" for _ in entity_ids)
        base = (
            f"SELECT * FROM facts WHERE entity_id IN ({placeholders}) "
            "AND is_active=1 AND superseded_by IS NULL"
        )
        params: list = list(entity_ids)
        if as_of:
            base += " AND (valid_from = '' OR valid_from <= ?)"
            params.append(as_of)
            base += " AND (valid_until = '' OR valid_until > ?)"
            params.append(as_of)
        base += " ORDER BY entity_id, created_at DESC"
        cur = await conn.execute(base, params)
        result: dict[int, list[dict]] = {eid: [] for eid in entity_ids}
        for r in await cur.fetchall():
            d = dict(r)
            d["is_current"] = True
            result.setdefault(d["entity_id"], []).append(d)
        return result


async def get_entity_relationships_bulk(entity_ids: list[int]) -> dict[int, list[dict]]:
    """Fetch relationships for multiple entities in one query."""
    if not entity_ids:
        return {}
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        placeholders = ",".join("?" for _ in entity_ids)
        params = list(entity_ids) + list(entity_ids)
        cur = await conn.execute(
            f"""SELECT r.*, e1.name AS source_name, e2.name AS target_name
                FROM relationships r
                JOIN entities e1 ON r.source_entity_id=e1.id
                JOIN entities e2 ON r.target_entity_id=e2.id
                WHERE r.source_entity_id IN ({placeholders})
                   OR r.target_entity_id IN ({placeholders})
                ORDER BY r.updated_at DESC""",
            params,
        )
        result: dict[int, list[dict]] = {eid: [] for eid in entity_ids}
        for r in await cur.fetchall():
            d = dict(r)
            if d["source_entity_id"] in result:
                result[d["source_entity_id"]].append(d)
            if d["target_entity_id"] in result and d["target_entity_id"] != d["source_entity_id"]:
                result[d["target_entity_id"]].append(d)
        return result


async def get_curator_state(key: str) -> str | None:
    """Read a single key from the curator_state table."""
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        cur = await conn.execute("SELECT value FROM curator_state WHERE key=?", (key,))
        row = await cur.fetchone()
        return row["value"] if row else None


async def set_curator_state(key: str, value: str):
    """Upsert a key/value pair in the curator_state table."""
    from archivist.storage.sqlite_pool import pool

    async with pool.write() as conn:
        await conn.execute(
            "INSERT INTO curator_state (key, value) VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )
