"""Embedding client — calls any OpenAI-compatible /v1/embeddings endpoint.

Supports NVIDIA NIM endpoints which require extra fields (input_type, truncate).
Set EMBED_PROVIDER=nvidia in .env to enable NVIDIA-specific payload.

Uses a long-lived httpx.AsyncClient so TCP connections (and TLS sessions)
are reused across calls instead of paying setup cost per embedding.
"""

import asyncio
import logging
import os
import time
import httpx
from config import EMBED_URL, EMBED_API_KEY, EMBED_MODEL
import health
import metrics as m
from observability import slow_embed_check

logger = logging.getLogger("archivist.embeddings")

_MAX_RETRIES = 3
_RETRY_DELAYS = [1, 2, 4]
_MAX_EMBED_CHARS = 1200
_IS_NVIDIA = os.getenv("EMBED_PROVIDER", "").lower() == "nvidia" or "nvidia" in EMBED_URL.lower()

_embed_client: httpx.AsyncClient | None = None


def _get_embed_client() -> httpx.AsyncClient:
    global _embed_client
    if _embed_client is None or _embed_client.is_closed:
        _embed_client = httpx.AsyncClient(
            timeout=60,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _embed_client


async def embed_text(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Embed a single text string, returning a float vector."""
    if len(text) > _MAX_EMBED_CHARS:
        text = text[:_MAX_EMBED_CHARS]
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
            return result
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[attempt]
                logger.warning("Embed attempt %d failed: %s — retrying in %ds", attempt + 1, e, delay)
                await asyncio.sleep(delay)
            else:
                health.register("embeddings", healthy=False, detail=str(e))
                logger.error("Embed failed after %d attempts: %s", _MAX_RETRIES, e)
                raise


async def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed multiple texts in parallel using asyncio.gather."""
    return list(await asyncio.gather(*(embed_text(t, model) for t in texts)))
