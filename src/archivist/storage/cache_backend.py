"""Pluggable cache backend — in-process memory (default) or Redis/Valkey.

Provides a unified interface for the hot cache and embedding cache so
the rest of the codebase doesn't need to know which backend is active.

Config:
  CACHE_BACKEND=memory  (default, in-process OrderedDict)
  CACHE_BACKEND=redis   (shared across pods, survives restarts)
  REDIS_URL=redis://localhost:6379/0
"""

import json
import logging
import threading
import time
from collections import OrderedDict
from typing import Any

from archivist.core.config import CACHE_BACKEND, REDIS_KEY_PREFIX, REDIS_URL

logger = logging.getLogger("archivist.cache_backend")


class CacheBackend:
    """Abstract cache interface."""

    def get(self, key: str) -> Any | None:
        raise NotImplementedError

    def put(self, key: str, value: Any, ttl_seconds: int = 600) -> None:
        raise NotImplementedError

    def delete(self, key: str) -> bool:
        raise NotImplementedError

    def delete_pattern(self, pattern: str) -> int:
        raise NotImplementedError

    def clear(self) -> int:
        raise NotImplementedError

    def size(self) -> int:
        raise NotImplementedError


class MemoryBackend(CacheBackend):
    """Thread-safe in-process LRU cache with TTL."""

    def __init__(self, max_entries: int = 4096):
        self._store: OrderedDict[str, tuple[float, int, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._max = max_entries

    def get(self, key: str) -> Any | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            ts, ttl, value = entry
            if time.monotonic() - ts > ttl:
                self._store.pop(key, None)
                return None
            self._store.move_to_end(key)
            return value

    def put(self, key: str, value: Any, ttl_seconds: int = 600) -> None:
        with self._lock:
            self._store[key] = (time.monotonic(), ttl_seconds, value)
            self._store.move_to_end(key)
            while len(self._store) > self._max:
                self._store.popitem(last=False)

    def delete(self, key: str) -> bool:
        with self._lock:
            return self._store.pop(key, None) is not None

    def delete_pattern(self, pattern: str) -> int:
        prefix = pattern.rstrip("*")
        with self._lock:
            keys = [k for k in self._store if k.startswith(prefix)]
            for k in keys:
                del self._store[k]
            return len(keys)

    def clear(self) -> int:
        with self._lock:
            n = len(self._store)
            self._store.clear()
            return n

    def size(self) -> int:
        with self._lock:
            return len(self._store)


class RedisBackend(CacheBackend):
    """Redis/Valkey-backed cache — shared across pods, native TTL."""

    def __init__(self):
        self._client = None
        self._prefix = REDIS_KEY_PREFIX
        self._client_lock = threading.Lock()

    def _get_client(self):
        if self._client is not None:
            return self._client
        with self._client_lock:
            if self._client is not None:
                return self._client
            try:
                import redis

                client = redis.Redis.from_url(
                    REDIS_URL,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_timeout=5,
                    retry_on_timeout=True,
                )
                client.ping()
                self._client = client
                logger.info("Connected to Redis at %s", REDIS_URL)
            except ImportError:
                logger.error("redis package not installed — pip install redis")
                raise
            except Exception as e:
                logger.error("Failed to connect to Redis: %s", e)
                raise
        return self._client

    def _key(self, key: str) -> str:
        return f"{self._prefix}{key}"

    def get(self, key: str) -> Any | None:
        try:
            raw = self._get_client().get(self._key(key))
            if raw is None:
                return None
            return json.loads(raw)
        except Exception as e:
            logger.debug("Redis GET failed: %s", e)
            return None

    def put(self, key: str, value: Any, ttl_seconds: int = 600) -> None:
        try:
            self._get_client().setex(
                self._key(key),
                ttl_seconds,
                json.dumps(value),
            )
        except Exception as e:
            logger.debug("Redis SET failed: %s", e)

    def delete(self, key: str) -> bool:
        try:
            return self._get_client().delete(self._key(key)) > 0
        except Exception:
            return False

    def delete_pattern(self, pattern: str) -> int:
        try:
            client = self._get_client()
            full_pattern = self._key(pattern)
            keys = []
            cursor = 0
            while True:
                cursor, batch = client.scan(cursor, match=full_pattern, count=200)
                keys.extend(batch)
                if cursor == 0:
                    break
            if keys:
                return client.delete(*keys)
            return 0
        except Exception:
            return 0

    def clear(self) -> int:
        return self.delete_pattern("*")

    def size(self) -> int:
        try:
            client = self._get_client()
            count = 0
            cursor = 0
            while True:
                cursor, batch = client.scan(cursor, match=self._key("*"), count=200)
                count += len(batch)
                if cursor == 0:
                    break
            return count
        except Exception:
            return 0


# ── Singleton ────────────────────────────────────────────────────────────────
_instance: CacheBackend | None = None
_instance_lock = threading.Lock()


def get_cache_backend() -> CacheBackend:
    """Return the configured cache backend singleton."""
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is not None:
            return _instance
        if CACHE_BACKEND == "redis":
            try:
                _instance = RedisBackend()
                return _instance
            except Exception:
                logger.warning("Redis unavailable — falling back to in-memory cache")
        _instance = MemoryBackend()
        return _instance
