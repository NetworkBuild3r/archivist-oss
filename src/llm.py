"""LLM client — calls any OpenAI-compatible /v1/chat/completions endpoint."""

import asyncio
import logging
import time

import httpx
from config import LLM_URL, LLM_MODEL, LLM_API_KEY
import health
import metrics as m

logger = logging.getLogger("archivist.llm")

_MAX_RETRIES = 3
_RETRY_DELAYS = [1, 2, 4]


async def llm_query(
    prompt: str,
    system: str = "",
    model: str = LLM_MODEL,
    max_tokens: int = 1024,
    json_mode: bool = False,
) -> str:
    """Send a prompt to the configured LLM and return the response text."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    body: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
    }
    if json_mode:
        body["response_format"] = {"type": "json_object"}

    m.inc(m.LLM_CALL)
    t0 = time.monotonic()
    for attempt in range(_MAX_RETRIES):
        try:
            headers: dict[str, str] = {}
            if LLM_API_KEY:
                headers["Authorization"] = f"Bearer {LLM_API_KEY}"
            async with httpx.AsyncClient(timeout=120) as client:
                try:
                    resp = await client.post(
                        f"{LLM_URL}/v1/chat/completions",
                        json=body,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    result = resp.json()["choices"][0]["message"]["content"]
                    m.observe(m.LLM_DURATION, (time.monotonic() - t0) * 1000)
                    health.register("llm", healthy=True)
                    return result
                except httpx.HTTPStatusError as exc:
                    if json_mode and exc.response.status_code in (400, 422):
                        logger.debug("json_mode unsupported by provider, retrying without it")
                        body.pop("response_format", None)
                        resp = await client.post(
                            f"{LLM_URL}/v1/chat/completions",
                            json=body,
                            headers=headers,
                        )
                        resp.raise_for_status()
                        result = resp.json()["choices"][0]["message"]["content"]
                        m.observe(m.LLM_DURATION, (time.monotonic() - t0) * 1000)
                        health.register("llm", healthy=True)
                        return result
                    raise
        except Exception as e:
            if attempt < _MAX_RETRIES - 1:
                delay = _RETRY_DELAYS[attempt]
                logger.warning("LLM attempt %d failed: %s — retrying in %ds", attempt + 1, e, delay)
                await asyncio.sleep(delay)
            else:
                m.inc(m.LLM_ERROR)
                health.register("llm", healthy=False, detail=str(e))
                logger.error("LLM failed after %d attempts: %s", _MAX_RETRIES, e)
                raise
