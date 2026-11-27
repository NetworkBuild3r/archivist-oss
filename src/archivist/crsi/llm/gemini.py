"""Google Gemini REST generateContent adapter over httpx."""

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

DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"


class GoogleGeminiAdapter:
    """Async ``httpx`` adapter for Gemini ``v1beta/models/{model}:generateContent``."""

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
        base = (role_config.base_url or DEFAULT_GEMINI_BASE_URL).strip().rstrip("/")
        url = f"{base}/v1beta/models/{role_config.model}:generateContent"

        api_key = role_config.resolve_api_key(self._environ)
        if not api_key:
            env_desc = role_config.api_key_env or "(unset api_key_env)"
            raise LLMError(
                f"Missing API key for gemini role={request.role!r}: "
                f"environment variable {env_desc} is not set."
            )
        self._redactor.register_secret(api_key)

        system_parts: list[str] = []
        contents: list[dict[str, Any]] = []
        for msg in request.messages:
            if msg.role == "system":
                system_parts.append(msg.content)
            else:
                gemini_role = "model" if msg.role == "assistant" else "user"
                contents.append({"role": gemini_role, "parts": [{"text": msg.content}]})

        if not contents:
            contents = [{"role": "user", "parts": [{"text": "\n\n".join(system_parts)}]}]
            system_parts = []

        gen_cfg: dict[str, Any] = {
            "maxOutputTokens": (
                request.max_tokens if request.max_tokens is not None else role_config.max_tokens
            ),
            "temperature": (
                request.temperature if request.temperature is not None else role_config.temperature
            ),
        }
        if request.json_mode:
            gen_cfg["responseMimeType"] = "application/json"

        payload: dict[str, Any] = {
            "contents": contents,
            "generationConfig": gen_cfg,
        }
        if system_parts:
            payload["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}

        # Pass key via header (never URL query parameter) so logs/URLs cannot leak it.
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": api_key,
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
            raise LLMError(f"gemini HTTP transport error: {safe}") from None
        latency_ms = (time.perf_counter() - t0) * 1000.0

        if resp.status_code >= 400:
            safe_body = self._redactor.redact_text(resp.text[:500])
            raise LLMError(f"gemini returned HTTP {resp.status_code}: {safe_body}")

        data = resp.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
            content = "".join(str(p.get("text", "")) for p in parts if isinstance(p, dict))
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"Malformed gemini response payload: {exc}") from None

        usage = data.get("usageMetadata") or {}
        prompt_tokens = int(usage.get("promptTokenCount", estimate_request_prompt_tokens(request)))
        completion_tokens = int(usage.get("candidatesTokenCount", estimate_tokens(content)))
        cost = estimate_cost_usd(prompt_tokens, completion_tokens, role_config)
        return LLMResponse(
            content=content,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            estimated_cost_usd=cost,
            model=role_config.model,
            provider="gemini",
            latency_ms=round(latency_ms, 2),
        )
