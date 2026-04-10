"""Namespace memory inventory — counts by memory_type, top entities, fleet tips flag.

Used by Stage 0 memory awareness (v1.6). Cached per namespace with TTL;
invalidated alongside hot_cache on namespace writes.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from graph import get_db

logger = logging.getLogger("archivist.inventory")

INVENTORY_TTL_SECONDS = 60
_INVENTORY_CACHE_MAX = 256

_lock = threading.Lock()
_cache: dict[str, tuple[float, "NamespaceInventory"]] = {}


@dataclass(frozen=True)
class NamespaceInventory:
    namespace: str
    total_memories: int
    by_type: dict[str, int]
    top_entities: list[str]
    has_fleet_tips: bool
    cached_at: float


def invalidate(namespace: str) -> None:
    """Drop cached inventory for one namespace (empty string clears default bucket)."""
    with _lock:
        _cache.pop(namespace, None)
    try:
        import query_classifier

        query_classifier.invalidate_cache(namespace)
    except Exception:
        pass


def invalidate_all() -> None:
    with _lock:
        _cache.clear()
    try:
        import query_classifier

        query_classifier.invalidate_all_cache()
    except Exception:
        pass


def _fetch_top_entities(conn, namespace: str, limit: int = 10) -> list[str]:
    """Entity names linked to this namespace via facts + memory_chunks file paths."""
    if not namespace:
        return []
    try:
        cur = conn.execute(
            """
            SELECT DISTINCT e.name, e.mention_count
            FROM entities e
            JOIN facts f ON f.entity_id = e.id AND f.is_active = 1
            JOIN memory_chunks mc ON mc.file_path = f.source_file AND mc.namespace = ?
            ORDER BY e.mention_count DESC
            LIMIT ?
            """,
            (namespace, limit),
        )
        rows = cur.fetchall()
        if rows:
            return [r["name"] for r in rows]
    except Exception as e:
        logger.debug("Top entities query failed for %s: %s", namespace, e)

    try:
        cur = conn.execute(
            """
            SELECT name, mention_count FROM entities
            WHERE mention_count >= 2
            ORDER BY mention_count DESC
            LIMIT ?
            """,
            (limit,),
        )
        return [r["name"] for r in cur.fetchall()]
    except Exception as e:
        logger.debug("Top entities fallback failed: %s", e)
        return []


def _fetch_inventory(namespace: str) -> NamespaceInventory:
    conn = get_db()
    try:
        by_type: dict[str, int] = {}
        if namespace:
            cur = conn.execute(
                """
                SELECT memory_type, COUNT(*) AS cnt
                FROM memory_chunks
                WHERE namespace = ?
                GROUP BY memory_type
                """,
                (namespace,),
            )
            for row in cur.fetchall():
                by_type[row["memory_type"] or "general"] = row["cnt"]
        else:
            cur = conn.execute(
                """
                SELECT memory_type, COUNT(*) AS cnt
                FROM memory_chunks
                GROUP BY memory_type
                """
            )
            for row in cur.fetchall():
                by_type[row["memory_type"] or "general"] = row["cnt"]

        total = sum(by_type.values())
        top_entities = _fetch_top_entities(conn, namespace, limit=10)

        has_fleet = False
        try:
            row = conn.execute(
                "SELECT COUNT(*) AS c FROM tips WHERE agent_id = 'fleet' AND archived = 0"
            ).fetchone()
            has_fleet = row and row["c"] and row["c"] > 0
        except Exception:
            pass

        return NamespaceInventory(
            namespace=namespace,
            total_memories=total,
            by_type=by_type,
            top_entities=top_entities,
            has_fleet_tips=has_fleet,
            cached_at=time.monotonic(),
        )
    finally:
        conn.close()


def get_inventory(namespace: str) -> NamespaceInventory:
    """Return inventory for namespace, using TTL cache."""
    now = time.monotonic()
    with _lock:
        hit = _cache.get(namespace)
        if hit is not None:
            ts, inv = hit
            if now - ts <= INVENTORY_TTL_SECONDS:
                return inv
    fresh = _fetch_inventory(namespace)
    with _lock:
        _cache[namespace] = (now, fresh)
        if len(_cache) > _INVENTORY_CACHE_MAX:
            expired = [k for k, (ts, _) in _cache.items()
                       if now - ts > INVENTORY_TTL_SECONDS]
            for k in expired:
                del _cache[k]
            if len(_cache) > _INVENTORY_CACHE_MAX:
                oldest = sorted(_cache.items(), key=lambda kv: kv[1][0])
                for k, _ in oldest[:len(oldest) // 2]:
                    del _cache[k]
    return fresh
