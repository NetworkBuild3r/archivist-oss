"""LLM client — calls any OpenAI-compatible /v1/chat/completions endpoint.

Uses a long-lived httpx.AsyncClient so TCP connections (and TLS sessions)
are reused across calls instead of paying setup cost per request.
"""

import asyncio
import logging
import time

import httpx
from config import LLM_URL, LLM_MODEL, LLM_API_KEY
import health
import metrics as m
from observability import slow_llm_check

logger = logging.getLogger("archivist.llm")

_MAX_RETRIES = 3
_RETRY_DELAYS = [1, 2, 4]

_llm_client: httpx.AsyncClient | None = None


def _message_text(data: dict) -> str:
    """Extract assistant text from an OpenAI-style chat completion JSON body.

    Providers may return ``content: null`` (JSON ``null`` → Python ``None``).
    ``dict.get("content", "")`` does **not** substitute when the key exists with a null value,
    which led to ``None`` answers and empty LongMemEval ``hypothesis`` fields downstream.
    Some Ollama models also populate ``reasoning`` when ``content`` is empty.
    """
    choice0 = (data.get("choices") or [{}])[0] or {}
    msg = choice0.get("message") or {}
    raw = msg.get("content")
    if raw is None:
        out = ""
    else:
        out = str(raw)
    if not out.strip():
        reasoning = msg.get("reasoning")
        if reasoning and str(reasoning).strip():
            rtxt = str(reasoning).strip()
            if len(rtxt) > 8000:
                rtxt = rtxt[-8000:]
            logger.warning(
                "LLM message.content empty; using message.reasoning (truncated to %d chars)",
                len(rtxt),
            )
            out = rtxt
    return out


def _get_llm_client() -> httpx.AsyncClient:
    global _llm_client
    if _llm_client is None or _llm_client.is_closed:
        _llm_client = httpx.AsyncClient(
            timeout=120,
            limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
        )
    return _llm_client


async def llm_query(
    prompt: str,
    system: str = "",
    model: str = LLM_MODEL,
    max_tokens: int = 1024,
    json_mode: bool = False,
    stage: str = "",
    url: str = "",
    api_key: str | None = None,
) -> str:
    """Send a prompt to the configured LLM and return the response text.

    ``url`` and ``api_key`` override the module-level LLM_URL / LLM_API_KEY
    defaults, allowing callers (e.g. the curator) to use a different endpoint
    without changing global config.  Pass ``api_key=""`` to explicitly send
    no key (e.g. Ollama doesn't require one).
    """
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

    effective_url = url or LLM_URL
    effective_key = LLM_API_KEY if api_key is None else api_key

    m.inc(m.LLM_CALL)
    t0 = time.monotonic()
    client = _get_llm_client()
    for attempt in range(_MAX_RETRIES):
        try:
            headers: dict[str, str] = {}
            if effective_key:
                headers["Authorization"] = f"Bearer {effective_key}"
            try:
                resp = await client.post(
                    f"{effective_url}/v1/chat/completions",
                    json=body,
                    headers=headers,
                )
                resp.raise_for_status()
                result = _message_text(resp.json())
                _dur_ms = (time.monotonic() - t0) * 1000
                _labels = {"stage": stage} if stage else None
                m.observe(m.LLM_DURATION, _dur_ms, _labels)
                slow_llm_check(_dur_ms)
                health.register("llm", healthy=True)
                return result
            except httpx.HTTPStatusError as exc:
                if json_mode and exc.response.status_code in (400, 422):
                    logger.debug("json_mode unsupported by provider, retrying without it")
                    body.pop("response_format", None)
                    resp = await client.post(
                        f"{effective_url}/v1/chat/completions",
                        json=body,
                        headers=headers,
                    )
                    resp.raise_for_status()
                    result = _message_text(resp.json())
                    _dur_ms = (time.monotonic() - t0) * 1000
                    _labels = {"stage": stage} if stage else None
                    m.observe(m.LLM_DURATION, _dur_ms, _labels)
                    slow_llm_check(_dur_ms)
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
