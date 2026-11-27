"""Anthropic Messages API adapter over httpx (/v1/messages)."""

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

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicAdapter:
    """Async ``httpx`` adapter for Anthropic ``/v1/messages``."""

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
        base = (role_config.base_url or DEFAULT_ANTHROPIC_BASE_URL).strip().rstrip("/")
        if base.endswith("/v1"):
            base = base[:-3].rstrip("/")
        url = f"{base}/v1/messages"

        api_key = role_config.resolve_api_key(self._environ)
        if not api_key:
            env_desc = role_config.api_key_env or "(unset api_key_env)"
            raise LLMError(
                f"Missing API key for anthropic role={request.role!r}: "
                f"environment variable {env_desc} is not set."
            )
        self._redactor.register_secret(api_key)

        system_parts: list[str] = []
        chat_messages: list[dict[str, str]] = []
        for msg in request.messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                chat_messages.append({"role": msg.role, "content": msg.content})

        if not chat_messages:
            chat_messages = [{"role": "user", "content": "\n\n".join(system_parts)}]
            system_parts = []

        payload: dict[str, Any] = {
            "model": role_config.model,
            "max_tokens": (
                request.max_tokens if request.max_tokens is not None else role_config.max_tokens
            ),
            "temperature": (
                request.temperature if request.temperature is not None else role_config.temperature
            ),
            "messages": chat_messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)

        headers = {
            "Content-Type": "application/json",
            "x-api-key": api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
        }

        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(
                transport=self._transport,
                timeout=role_config.timeout_s,
            ) as client:
                resp = await client.post(url, headers=headers, json=payload)
        except httpx.HTTPError as exc:
            safe = self._redactor.redact_text(str(exc))
            raise LLMError(f"anthropic HTTP transport error: {safe}") from None
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code >= 400:
            safe_body = self._redactor.redact_text(resp.text[:500])
            raise LLMError(f"anthropic returned HTTP {resp.status_code}: {safe_body}")

        data = resp.json()
        blocks = data.get("content")
        if not isinstance(blocks, list) or not blocks:
            raise LLMError("Malformed anthropic response payload: missing content blocks")
        text_parts = [
            str(block.get("text", ""))
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        content = "".join(text_parts)

        usage = data.get("usage") or {}
        prompt_tokens = int(usage.get("input_tokens", estimate_request_prompt_tokens(request)))
        completion_tokens = int(usage.get("output_tokens", estimate_tokens(content)))
        cost = estimate_cost_usd(prompt_tokens, completion_tokens, role_config)
        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=cost,
            model=str(data.get("model", role_config.model)),
            provider="anthropic",
            latency_ms=round(latency_ms, 2),
        )
