"""Embedding client — calls any OpenAI-compatible /v1/embeddings endpoint.

Supports NVIDIA NIM endpoints which require extra fields (input_type, truncate).
Set EMBED_PROVIDER=nvidia in .env to enable NVIDIA-specific payload.

Uses a long-lived httpx.AsyncClient so TCP connections (and TLS sessions)
are reused across calls instead of paying setup cost per embedding.

Includes an in-process LRU+TTL cache so repeated / multi-query searches
avoid redundant embedding API round-trips (~50-200ms each).
"""

import asyncio
import logging
import os
import threading
import time
from collections import OrderedDict

import httpx

import archivist.core.health as health
import archivist.core.metrics as m
from archivist.core.config import EMBED_API_KEY, EMBED_MODEL, EMBED_URL
from archivist.core.observability import slow_embed_check

logger = logging.getLogger("archivist.embeddings")

_MAX_RETRIES = 3
_RETRY_DELAYS = [1, 2, 4]
_MAX_EMBED_CHARS = 2400
_IS_NVIDIA = os.getenv("EMBED_PROVIDER", "").lower() == "nvidia" or "nvidia" in EMBED_URL.lower()

_embed_client: httpx.AsyncClient | None = None

# ── Embedding vector cache (v1.10) ──────────────────────────────────────────
_EMBED_CACHE_MAX = 2048
_EMBED_CACHE_TTL = 3600  # 1 hour
_embed_cache: OrderedDict[str, tuple[float, tuple[float, ...]]] = OrderedDict()
_embed_cache_lock = threading.Lock()
# Backward-compatible aliases for the metric name strings (same as m.EMBED_CACHE_*).
EMBED_CACHE_HITS = m.EMBED_CACHE_HIT
EMBED_CACHE_MISSES = m.EMBED_CACHE_MISS


def _cache_key(text: str, model: str) -> str:
    import hashlib

    return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()[:24]


def _cache_get(text: str, model: str) -> tuple[float, ...] | None:
    key = _cache_key(text, model)
    with _embed_cache_lock:
        entry = _embed_cache.get(key)
        if entry is None:
            return None
        ts, vec = entry
        if time.monotonic() - ts > _EMBED_CACHE_TTL:
            _embed_cache.pop(key, None)
            return None
        _embed_cache.move_to_end(key)
        return vec


def _cache_put(text: str, model: str, vec: list[float]) -> None:
    key = _cache_key(text, model)
    with _embed_cache_lock:
        _embed_cache[key] = (time.monotonic(), tuple(vec))
        _embed_cache.move_to_end(key)
        while len(_embed_cache) > _EMBED_CACHE_MAX:
            _embed_cache.popitem(last=False)


def _get_embed_client() -> httpx.AsyncClient:
    global _embed_client
    if _embed_client is None or _embed_client.is_closed:
        _embed_client = httpx.AsyncClient(
            timeout=60,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _embed_client


async def embed_text(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Embed a single text string, returning a float vector.

    Results are cached in-process (LRU + 1h TTL) so multi-query
    expansion, adaptive-widen, and topic-fallback avoid redundant
    API calls.
    """
    if len(text) > _MAX_EMBED_CHARS:
        text = text[:_MAX_EMBED_CHARS]

    cached = _cache_get(text, model)
    if cached is not None:
        m.inc(m.EMBED_CACHE_HIT)
        # Cache stores immutable tuples; Qdrant query APIs expect list (not tuple).
        return list(cached)

    m.inc(m.EMBED_CACHE_MISS)
    client = _get_embed_client()
    for attempt in range(_MAX_RETRIES):
        try:
            t_embed = time.monotonic()
            payload: dict = {"model": model, "input": text}
            if _IS_NVIDIA:
                payload["input_type"] = "passage"
                payload["truncate"] = "END"
                payload["encoding_format"] = "float"

            headers: dict[str, str] = {"Content-Type": "application/json"}
            if EMBED_API_KEY:
                headers["Authorization"] = f"Bearer {EMBED_API_KEY}"
            resp = await client.post(
                f"{EMBED_URL}/v1/embeddings",
                headers=headers,
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
            result = data["data"][0]["embedding"]
            dur_ms = (time.monotonic() - t_embed) * 1000
            m.observe(m.EMBED_DURATION, dur_ms)
            slow_embed_check(dur_ms)
            health.register("embeddings", healthy=True)
            _cache_put(text, model, result)
            return result
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[attempt]
                logger.warning(
                    "Embed attempt %d failed: %s — retrying in %ds", attempt + 1, e, delay
                )
                await asyncio.sleep(delay)
            else:
                health.register("embeddings", healthy=False, detail=str(e))
                logger.error("Embed failed after %d attempts: %s", _MAX_RETRIES, e)
                raise


async def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed multiple texts in a single API call (true batch).

    Falls back to individual parallel calls if the batch request fails
    (e.g. provider doesn't support array input).
    """
    if not texts:
        return []
    if len(texts) == 1:
        return [await embed_text(texts[0], model)]

    truncated = [t[:_MAX_EMBED_CHARS] for t in texts]

    cached_results: dict[int, list[float]] = {}
    uncached_indices: list[int] = []
    uncached_texts: list[str] = []
    for i, t in enumerate(truncated):
        cached = _cache_get(t, model)
        if cached is not None:
            m.inc(m.EMBED_CACHE_HIT)
            cached_results[i] = list(cached)
        else:
            uncached_indices.append(i)
            uncached_texts.append(t)

    if not uncached_texts:
        return [cached_results[i] for i in range(len(texts))]

    try:
        client = _get_embed_client()
        t_embed = time.monotonic()
        payload: dict = {"model": model, "input": uncached_texts}
        if _IS_NVIDIA:
            payload["input_type"] = "passage"
            payload["truncate"] = "END"
            payload["encoding_format"] = "float"

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if EMBED_API_KEY:
            headers["Authorization"] = f"Bearer {EMBED_API_KEY}"

        resp = await client.post(
            f"{EMBED_URL}/v1/embeddings",
            headers=headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

        embeddings = sorted(data["data"], key=lambda x: x["index"])
        dur_ms = (time.monotonic() - t_embed) * 1000
        m.observe(m.EMBED_DURATION, dur_ms)
        health.register("embeddings", healthy=True)

        for j, emb in enumerate(embeddings):
            vec = emb["embedding"]
            orig_idx = uncached_indices[j]
            cached_results[orig_idx] = vec
            _cache_put(uncached_texts[j], model, vec)
            m.inc(m.EMBED_CACHE_MISS)

        return [cached_results[i] for i in range(len(texts))]

    except Exception as e:
        logger.warning("Batch embed failed (%s), falling back to individual calls", e)
        individual = list(await asyncio.gather(*(embed_text(t, model) for t in texts)))
        return individual
