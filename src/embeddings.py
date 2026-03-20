"""Embedding client — calls any OpenAI-compatible /v1/embeddings endpoint."""

import asyncio
import logging
import httpx
from config import EMBED_URL, LLM_API_KEY, EMBED_MODEL

logger = logging.getLogger("archivist.embeddings")

_MAX_RETRIES = 3
_RETRY_DELAYS = [1, 2, 4]
_MAX_EMBED_CHARS = 1200


async def embed_text(text: str, model: str = EMBED_MODEL) -> list[float]:
    """Embed a single text string, returning a float vector."""
    if len(text) > _MAX_EMBED_CHARS:
        text = text[:_MAX_EMBED_CHARS]
    for attempt in range(_MAX_RETRIES):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                resp = await client.post(
                    f"{EMBED_URL}/v1/embeddings",
                    headers={
                        "Authorization": f"Bearer {LLM_API_KEY}",
                        "Content-Type": "application/json",
                    },
                    json={"model": model, "input": text},
                )
                resp.raise_for_status()
                data = resp.json()
                return data["data"][0]["embedding"]
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[attempt]
                logger.warning("Embed attempt %d failed: %s — retrying in %ds", attempt + 1, e, delay)
                await asyncio.sleep(delay)
            else:
                logger.error("Embed failed after %d attempts: %s", _MAX_RETRIES, e)
                raise


async def embed_batch(texts: list[str], model: str = EMBED_MODEL) -> list[list[float]]:
    """Embed multiple texts sequentially. For high throughput, consider batching at the API level."""
    results = []
    for text in texts:
        results.append(await embed_text(text, model))
    return results
