"""OpenAI-compatible REST adapter over httpx (OpenAI, xAI, Ollama, vLLM, LiteLLM)."""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any

import httpx

from archivist.crsi.config import ModelRoleConfig
from archivist.crsi.guards.secrets import SecretRedactor
from archivist.crsi.llm.base import LLMError, LLMRequest, LLMResponse
from archivist.crsi.llm.budget import (
    estimate_cost_usd,
    estimate_request_prompt_tokens,
    estimate_tokens,
)

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com"


def normalize_openai_base_url(base_url: str) -> str:
    """Strip trailing slashes and a trailing ``/v1`` so ``/v1/chat/completions`` never doubles."""
    cleaned = base_url.strip().rstrip("/")
    if cleaned.endswith("/v1"):
        cleaned = cleaned[:-3].rstrip("/")
    return cleaned


class OpenAICompatibleAdapter:
    """Async ``httpx`` adapter for ``/v1/chat/completions`` endpoints."""

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        redactor: SecretRedactor | None = None,
    ) -> None:
        self._environ = environ
        self._transport = transport
        self._redactor = redactor if redactor is not None else SecretRedactor()

    async def complete(
        self,
        request: LLMRequest,
        role_config: ModelRoleConfig,
    ) -> LLMResponse:
        base = normalize_openai_base_url(role_config.base_url or DEFAULT_OPENAI_BASE_URL)
        url = f"{base}/v1/chat/completions"

        api_key = role_config.resolve_api_key(self._environ)
        if api_key:
            self._redactor.register_secret(api_key)
        elif role_config.is_paid_provider():
            env_desc = role_config.api_key_env or "(unset api_key_env)"
            raise LLMError(
                f"Missing API key for openai_compat role={request.role!r}: "
                f"environment variable {env_desc} is not set."
            )

        headers: dict[str, str] = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload: dict[str, Any] = {
            "model": role_config.model,
            "messages": [{"role": m.role, "content": m.content} for m in request.messages],
            "max_tokens": (
                request.max_tokens if request.max_tokens is not None else role_config.max_tokens
            ),
            "temperature": (
                request.temperature if request.temperature is not None else role_config.temperature
            ),
        }
        if request.json_mode:
            payload["response_format"] = {"type": "json_object"}

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=role_config.timeout_s,
            ) as client:
                resp = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            safe = self._redactor.redact_text(str(exc))
            raise LLMError(f"openai_compat HTTP transport error: {safe}") from None
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code >= 400:
            safe_body = self._redactor.redact_text(resp.text[:500])
            raise LLMError(f"openai_compat returned HTTP {resp.status_code}: {safe_body}")

        data = resp.json()
        try:
            content = str(data["choices"][0]["message"]["content"])
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Malformed openai_compat response payload: {exc}") from None

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("prompt_tokens", estimate_request_prompt_tokens(request)))
        completion_tokens = int(usage.get("completion_tokens", estimate_tokens(content)))
        cost = (
            estimate_cost_usd(prompt_tokens, completion_tokens, role_config)
            if role_config.is_paid_provider()
            else 0.0
        )
        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=cost,
            model=str(data.get("model", role_config.model)),
            provider="openai_compat",
            latency_ms=round(latency_ms, 2),
        )
