"""LLM client — calls any OpenAI-compatible /v1/chat/completions endpoint."""

import logging

import httpx
from config import LLM_URL, LLM_MODEL, LLM_API_KEY

logger = logging.getLogger("archivist.llm")


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

    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(
                f"{LLM_URL}/v1/chat/completions",
                json=body,
                headers={"Authorization": f"Bearer {LLM_API_KEY}"},
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]
        except httpx.HTTPStatusError as exc:
            if json_mode and exc.response.status_code in (400, 422):
                logger.debug("json_mode unsupported by provider, retrying without it")
                body.pop("response_format", None)
                resp = await client.post(
                    f"{LLM_URL}/v1/chat/completions",
                    json=body,
                    headers={"Authorization": f"Bearer {LLM_API_KEY}"},
                )
                resp.raise_for_status()
                return resp.json()["choices"][0]["message"]["content"]
            raise
