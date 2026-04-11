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

from config import HOT_CACHE_ENABLED, HOT_CACHE_MAX_PER_AGENT, HOT_CACHE_TTL_SECONDS, WRITE_FENCE_WINDOW_S

logger = logging.getLogger("archivist.cache")

_lock = threading.Lock()

# agent_id → OrderedDict[(cache_key) → (timestamp, value)]
_agent_caches: dict[str, OrderedDict] = {}
_MAX_AGENT_IDS = 256
_last_agent_prune = 0.0

# Write-fence: track recent writes per namespace for read-your-own-write.
# Process-local — in multi-worker uvicorn, each worker maintains its own dict.
# Cross-worker coherence requires shared state (Redis) if workers > 1.
_recent_writes: dict[str, float] = {}  # namespace -> monotonic timestamp


def mark_write(namespace: str) -> None:
    """Record that a write just happened for *namespace*.

    Called at the TOP of the store path (before Qdrant upsert) so the search
    path can skip the cache during the write-fence window.

    Prunes stale entries to prevent unbounded growth.
    """
    now = time.monotonic()
    with _lock:
        _recent_writes[namespace] = now
        stale = [k for k, v in _recent_writes.items() if now - v > WRITE_FENCE_WINDOW_S * 10]
        for k in stale:
            del _recent_writes[k]


def namespace_recently_written(namespace: str, window_s: float | None = None) -> bool:
    """Return True if *namespace* had a write within the last *window_s* seconds.

    For fleet-wide queries (``namespace=""``), returns True if ANY namespace
    was written recently, since a fleet-wide cached result could contain data
    from any namespace.
    """
    if window_s is None:
        window_s = WRITE_FENCE_WINDOW_S
    now = time.monotonic()
    with _lock:
        if not namespace:
            return any((now - ts) < window_s for ts in _recent_writes.values())
        return (now - _recent_writes.get(namespace, 0)) < window_s


def _cache_key(query: str, namespace: str = "", tier: str = "l2",
               memory_type: str = "", extra: str = "") -> str:
    raw = f"{query}|{namespace}|{tier}|{memory_type}|{extra}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


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
        del _agent_caches[aid]


def put(agent_id: str, query: str, result: dict, namespace: str = "",
        tier: str = "l2", memory_type: str = "", extra: str = "") -> None:
    """Store a result in the hot cache, evicting LRU entries if needed."""
    if not HOT_CACHE_ENABLED:
        return

    key = _cache_key(query, namespace, tier, memory_type, extra)
    now = time.time()
    with _lock:
        cache = _agent_caches.setdefault(agent_id, OrderedDict())
        cache[key] = (now, result)
        cache.move_to_end(key)
        while len(cache) > HOT_CACHE_MAX_PER_AGENT:
            cache.popitem(last=False)
        if len(_agent_caches) > _MAX_AGENT_IDS:
            _prune_stale_agents_locked(now)


def invalidate_namespace(namespace: str) -> int:
    """Evict all cache entries whose namespace matches OR were fleet-wide.

    A fleet-wide cached result (empty ``_cache_namespace``) may contain stale
    data from any namespace, so writes must also evict those entries.
    """
    if not HOT_CACHE_ENABLED:
        return 0

    evicted = 0
    with _lock:
        for agent_id, cache in _agent_caches.items():
            to_remove = []
            for key, (ts, value) in cache.items():
                cached_ns = value.get("_cache_namespace", "")
                if cached_ns == namespace or cached_ns == "":
                    to_remove.append(key)
            for key in to_remove:
                del cache[key]
                evicted += 1
    if evicted:
        logger.debug("Cache invalidation: evicted %d entries for namespace=%s (incl. fleet-wide)", evicted, namespace)
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
        return len(cache) if cache else 0


def invalidate_all() -> int:
    """Clear all caches."""
    with _lock:
        total = sum(len(c) for c in _agent_caches.values())
        _agent_caches.clear()
    try:
        import namespace_inventory

        namespace_inventory.invalidate_all()
    except Exception:
        pass
    return total


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
