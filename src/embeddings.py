"""Embedding client — calls any OpenAI-compatible /v1/embeddings endpoint.

Supports NVIDIA NIM endpoints which require extra fields (input_type, truncate).
Set EMBED_PROVIDER=nvidia in .env to enable NVIDIA-specific payload.
"""

import asyncio
import logging
import os
import httpx
from config import EMBED_URL, EMBED_API_KEY, EMBED_MODEL
import health

logger = logging.getLogger("archivist.embeddings")

_MAX_RETRIES = 3
_RETRY_DELAYS = [1, 2, 4]
_MAX_EMBED_CHARS = 1200
_IS_NVIDIA = os.getenv("EMBED_PROVIDER", "").lower() == "nvidia" or "nvidia" in EMBED_URL.lower()


async def embed_text(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Embed a single text string, returning a float vector."""
    if len(text) > _MAX_EMBED_CHARS:
        text = text[:_MAX_EMBED_CHARS]
    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
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
