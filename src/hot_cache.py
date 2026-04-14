"""Per-agent LRU hot cache with TTL — the middle layer of Archivist's three-tier
memory hierarchy (session/ephemeral → hot cache → long-term Qdrant+SQLite).

The cache stores recent retrieval results keyed by (agent_id, query_hash) so that
repeated or similar lookups within a session skip the full RLM pipeline.

Invalidation:
  - TTL expiry (configurable via HOT_CACHE_TTL_SECONDS)
  - Explicit eviction when a write to the same namespace is detected
  - LRU eviction when per-agent capacity is exceeded
"""

import hashlib
import logging
import threading
import time
from collections import OrderedDict

from config import HOT_CACHE_ENABLED, HOT_CACHE_MAX_PER_AGENT, HOT_CACHE_TTL_SECONDS

logger = logging.getLogger("archivist.cache")

_lock = threading.Lock()

# agent_id → OrderedDict[(cache_key) → (timestamp, value)]
_agent_caches: dict[str, OrderedDict] = {}
# namespace → set of (agent_id, cache_key) for O(1) invalidation
_ns_index: dict[str, set[tuple[str, str]]] = {}
_MAX_AGENT_IDS = 256
_last_agent_prune = 0.0


def _cache_key(query: str, namespace: str = "", tier: str = "l2",
               memory_type: str = "", extra: str = "") -> str:
    raw = f"{query}|{namespace}|{tier}|{memory_type}|{extra}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def _remove_from_ns_index(agent_id: str, key: str, value: dict) -> None:
    """Remove a single (agent_id, key) pair from the namespace index. Caller holds _lock."""
    ns = value.get("_cache_namespace", "")
    if ns:
        bucket = _ns_index.get(ns)
        if bucket is not None:
            bucket.discard((agent_id, key))
            if not bucket:
                del _ns_index[ns]


def get(agent_id: str, query: str, namespace: str = "", tier: str = "l2",
        memory_type: str = "", extra: str = "") -> dict | None:
    """Return cached result or None on miss. Moves hit to front (LRU)."""
    if not HOT_CACHE_ENABLED:
        return None

    key = _cache_key(query, namespace, tier, memory_type, extra)
    with _lock:
        cache = _agent_caches.get(agent_id)
        if cache is None:
            return None
        entry = cache.get(key)
        if entry is None:
            return None
        ts, value = entry
        if time.time() - ts > HOT_CACHE_TTL_SECONDS:
            _remove_from_ns_index(agent_id, key, value)
            del cache[key]
            return None
        cache.move_to_end(key)
        return value


def _prune_stale_agents_locked(now: float) -> None:
    """Remove agent_id entries where all cached results have expired."""
    global _last_agent_prune
    if now - _last_agent_prune < 60:
        return
    _last_agent_prune = now
    empty = [aid for aid, cache in _agent_caches.items()
             if not cache or all(now - ts > HOT_CACHE_TTL_SECONDS for ts, _ in cache.values())]
    for aid in empty:
        cache = _agent_caches.pop(aid, None)
        if cache:
            for key, (_, value) in cache.items():
                _remove_from_ns_index(aid, key, value)


def put(agent_id: str, query: str, result: dict, namespace: str = "",
        tier: str = "l2", memory_type: str = "", extra: str = "") -> None:
    """Store a result in the hot cache, evicting LRU entries if needed."""
    if not HOT_CACHE_ENABLED:
        return

    key = _cache_key(query, namespace, tier, memory_type, extra)
    now = time.time()
    with _lock:
        cache = _agent_caches.setdefault(agent_id, OrderedDict())

        old_entry = cache.get(key)
        if old_entry is not None:
            _remove_from_ns_index(agent_id, key, old_entry[1])

        cache[key] = (now, result)
        cache.move_to_end(key)

        result_ns = result.get("_cache_namespace", "")
        if result_ns:
            _ns_index.setdefault(result_ns, set()).add((agent_id, key))

        while len(cache) > HOT_CACHE_MAX_PER_AGENT:
            evicted_key, (_, evicted_val) = cache.popitem(last=False)
            _remove_from_ns_index(agent_id, evicted_key, evicted_val)

        if len(_agent_caches) > _MAX_AGENT_IDS:
            _prune_stale_agents_locked(now)


def invalidate_namespace(namespace: str) -> int:
    """Evict all cache entries whose namespace matches. O(1) index lookup."""
    if not HOT_CACHE_ENABLED:
        return 0

    evicted = 0
    with _lock:
        entries = _ns_index.pop(namespace, None)
        if entries:
            for agent_id, key in entries:
                cache = _agent_caches.get(agent_id)
                if cache is not None and key in cache:
                    del cache[key]
                    evicted += 1
    if evicted:
        logger.debug("Cache invalidation: evicted %d entries for namespace %s", evicted, namespace)
    try:
        import namespace_inventory

        namespace_inventory.invalidate(namespace)
    except Exception:
        pass
    return evicted


def invalidate_agent(agent_id: str) -> int:
    """Clear the entire hot cache for one agent."""
    with _lock:
        cache = _agent_caches.pop(agent_id, None)
        if not cache:
            return 0
        for key, (_, value) in cache.items():
            _remove_from_ns_index(agent_id, key, value)
        return len(cache)


def invalidate_all() -> int:
    """Clear all caches unconditionally."""
    with _lock:
        total = sum(len(c) for c in _agent_caches.values())
        _agent_caches.clear()
        _ns_index.clear()
    try:
        import namespace_inventory

        namespace_inventory.invalidate_all()
    except Exception:
        pass
    return total


# Alias for callers that want to be explicit about bypassing guards.
force_invalidate_all = invalidate_all


def stats() -> dict:
    """Return cache statistics for monitoring."""
    with _lock:
        now = time.time()
        per_agent = {}
        total_entries = 0
        total_expired = 0
        for agent_id, cache in _agent_caches.items():
            alive = 0
            expired = 0
            for key, (ts, value) in cache.items():
                if now - ts > HOT_CACHE_TTL_SECONDS:
                    expired += 1
                else:
                    alive += 1
            per_agent[agent_id] = {"alive": alive, "expired": expired}
            total_entries += alive
            total_expired += expired
        return {
            "enabled": HOT_CACHE_ENABLED,
            "max_per_agent": HOT_CACHE_MAX_PER_AGENT,
            "ttl_seconds": HOT_CACHE_TTL_SECONDS,
            "total_entries": total_entries,
            "total_expired_pending": total_expired,
            "agents": len(per_agent),
            "per_agent": per_agent,
        }
