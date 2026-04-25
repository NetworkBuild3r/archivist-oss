"""Ephemeral in-process session memory tier for Archivist.

``SessionStore`` provides fast, TTL-scoped key/value memory for a single agent
session.  It lives entirely in the Python process — no DB, no Qdrant, no I/O.

Lifecycle
---------
1. Agent puts working notes during a task: ``store.put(agent_id, session_id, key, value)``
2. Agent reads them back: ``store.get(agent_id, session_id, key)``
3. At session end, ``flush()`` returns all live entries so the caller can
   optionally persist high-value ones to the durable graph store.
4. ``promote()`` marks one entry as "persist me" and returns the text —
   the caller then stores it via the normal ``archivist_store`` pipeline.

Thread-safety
-------------
A single ``threading.Lock`` guards the internal dict.  All methods are
synchronous (no ``async``).  The store is a process-level singleton accessed
via ``get_session_store()``.

Token budget interaction
------------------------
During retrieval the packer can query the store for ephemeral entries that
are relevant to the current query.  These are injected as ``tier='ephemeral'``
results before the adaptive packer runs so they consume the first slice of
budget (highest freshness priority).
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

logger = logging.getLogger("archivist.session_store")


@dataclass
class _Entry:
    value: str
    created_at: float
    ttl_seconds: int
    promoted: bool = False

    def is_expired(self, now: float) -> bool:
        return (now - self.created_at) > self.ttl_seconds


class SessionStore:
    """In-process ephemeral memory scoped to one agent session.

    All operations are O(1) or O(n_entries_for_session).  The store is
    capacity-capped: when ``max_entries`` is reached, the oldest entry
    (by creation time) across all sessions is evicted.
    """

    def __init__(self, max_entries: int = 512, default_ttl_seconds: int = 3600) -> None:
        self._max_entries = max_entries
        self._default_ttl = default_ttl_seconds
        # { (agent_id, session_id, key): _Entry }
        self._data: dict[tuple[str, str, str], _Entry] = {}
        self._lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def put(
        self,
        agent_id: str,
        session_id: str,
        key: str,
        value: str,
        ttl_seconds: int | None = None,
    ) -> None:
        """Store *value* under *key* for the given agent/session pair.

        Overwrites any existing entry with the same key.  Evicts the
        oldest entry across all sessions when at capacity.
        """
        ttl = ttl_seconds if ttl_seconds is not None else self._default_ttl
        composite = (agent_id, session_id, key)
        entry = _Entry(value=value, created_at=time.monotonic(), ttl_seconds=ttl)
        with self._lock:
            self._data[composite] = entry
            self._maybe_evict()

    def get(self, agent_id: str, session_id: str, key: str) -> str | None:
        """Return the value for *key*, or ``None`` if absent or expired."""
        composite = (agent_id, session_id, key)
        with self._lock:
            entry = self._data.get(composite)
            if entry is None:
                return None
            if entry.is_expired(time.monotonic()):
                del self._data[composite]
                return None
            return entry.value

    def delete(self, agent_id: str, session_id: str, key: str) -> bool:
        """Remove *key*.  Returns True if it existed."""
        composite = (agent_id, session_id, key)
        with self._lock:
            return self._data.pop(composite, None) is not None

    def flush(self, agent_id: str, session_id: str) -> list[dict]:
        """Return all live entries for the session and remove them from the store.

        Each dict has: ``key``, ``value``, ``age_seconds``, ``promoted``.
        Expired entries are silently dropped.
        """
        now = time.monotonic()
        result: list[dict] = []
        prefix = (agent_id, session_id)
        with self._lock:
            keys_to_remove = [k for k in self._data if k[:2] == prefix]
            for k in keys_to_remove:
                entry = self._data.pop(k)
                if not entry.is_expired(now):
                    result.append(
                        {
                            "key": k[2],
                            "value": entry.value,
                            "age_seconds": int(now - entry.created_at),
                            "promoted": entry.promoted,
                        }
                    )
        return result

    def promote(self, agent_id: str, session_id: str, key: str) -> str | None:
        """Mark *key* as 'persist me' and return its text.

        The caller should subsequently store the returned text via
        ``archivist_store`` to escalate it to the durable memory tier.
        Returns ``None`` if the key does not exist or is expired.
        """
        composite = (agent_id, session_id, key)
        with self._lock:
            entry = self._data.get(composite)
            if entry is None or entry.is_expired(time.monotonic()):
                return None
            entry.promoted = True
            return entry.value

    def list_keys(self, agent_id: str, session_id: str) -> list[str]:
        """Return live (non-expired) keys for a session, in insertion order."""
        now = time.monotonic()
        prefix = (agent_id, session_id)
        with self._lock:
            return [
                k[2] for k, e in self._data.items() if k[:2] == prefix and not e.is_expired(now)
            ]

    def size(self) -> int:
        """Total live + possibly-expired entry count (no TTL sweep)."""
        with self._lock:
            return len(self._data)

    def clear_session(self, agent_id: str, session_id: str) -> int:
        """Remove all entries for a session.  Returns the count removed."""
        prefix = (agent_id, session_id)
        with self._lock:
            keys = [k for k in self._data if k[:2] == prefix]
            for k in keys:
                del self._data[k]
            return len(keys)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _maybe_evict(self) -> None:
        """Evict the oldest entry when over capacity.  Caller must hold the lock."""
        if len(self._data) <= self._max_entries:
            return
        oldest_key = min(self._data, key=lambda k: self._data[k].created_at)
        del self._data[oldest_key]
        logger.debug("SessionStore: evicted oldest entry %s", oldest_key)


# ---------------------------------------------------------------------------
# Process-level singleton
# ---------------------------------------------------------------------------

_store_instance: SessionStore | None = None
_store_lock = threading.Lock()


def get_session_store() -> SessionStore:
    """Return the process-level ``SessionStore`` singleton.

    Initialised lazily from ``config.SESSION_STORE_MAX_ENTRIES`` and
    ``config.SESSION_STORE_TTL_SECONDS`` on first call.
    """
    global _store_instance
    if _store_instance is not None:
        return _store_instance
    with _store_lock:
        if _store_instance is None:
            from archivist.core.config import SESSION_STORE_MAX_ENTRIES, SESSION_STORE_TTL_SECONDS

            _store_instance = SessionStore(
                max_entries=SESSION_STORE_MAX_ENTRIES,
                default_ttl_seconds=SESSION_STORE_TTL_SECONDS,
            )
            logger.info(
                "SessionStore initialised: max_entries=%d ttl=%ds",
                SESSION_STORE_MAX_ENTRIES,
                SESSION_STORE_TTL_SECONDS,
            )
    return _store_instance
