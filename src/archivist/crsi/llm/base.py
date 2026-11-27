"""Core LLM request/response models, adapter protocol, and role-routed client."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

import httpx
from pydantic import BaseModel, ConfigDict, Field

from archivist.crsi.config import CrsiConfig, ModelRoleConfig, ProviderName
from archivist.crsi.guards.secrets import SecretRedactor

LLMRole = Literal["policy", "proposer", "selector"]
MessageRole = Literal["system", "user", "assistant"]


class LLMError(RuntimeError):
    """Sanitized runtime error raised by LLM adapters."""


class Message(BaseModel):
    """Single chat message passed to an LLM adapter."""

    model_config = ConfigDict(extra="forbid")

    role: MessageRole
    content: str


class LLMRequest(BaseModel):
    """Uniform model-agnostic completion request."""

    model_config = ConfigDict(extra="forbid")

    role: LLMRole
    messages: list[Message] = Field(min_length=1)
    json_mode: bool = False
    max_tokens: int | None = Field(default=None, ge=1)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMResponse(BaseModel):
    """Uniform model-agnostic completion response."""

    model_config = ConfigDict(extra="forbid")

    content: str
    prompt_tokens: int = Field(default=0, ge=0)
    completion_tokens: int = Field(default=0, ge=0)
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    model: str
    provider: str
    latency_ms: float = Field(default=0.0, ge=0.0)


@runtime_checkable
class ProviderAdapter(Protocol):
    """Provider-specific transport adapter (fake, OpenAI-compat, Anthropic, Gemini)."""

    async def complete(
        self,
        request: LLMRequest,
        role_config: ModelRoleConfig,
    ) -> LLMResponse: ...


@runtime_checkable
class LLMClient(Protocol):
    """High-level role-routed LLM client interface."""

    async def complete(self, request: LLMRequest) -> LLMResponse: ...


class RoutedLLMClient:
    """Role-routed LLM client enforcing SpendTracker gates and secret redaction."""

    def __init__(
        self,
        config: CrsiConfig,
        *,
        adapters: Mapping[ProviderName, ProviderAdapter] | None = None,
        environ: Mapping[str, str] | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
        redactor: SecretRedactor | None = None,
    ) -> None:
        from archivist.crsi.llm.anthropic import AnthropicAdapter
        from archivist.crsi.llm.budget import SpendTracker
        from archivist.crsi.llm.fake import FakeScriptedAdapter
        from archivist.crsi.llm.gemini import GoogleGeminiAdapter
        from archivist.crsi.llm.openai_compat import OpenAICompatibleAdapter

        self.config = config
        self.redactor = redactor if redactor is not None else config.build_redactor(environ=environ)
        self.spend_tracker = SpendTracker(config.budget)

        default_adapters: dict[ProviderName, ProviderAdapter] = {
            "fake": FakeScriptedAdapter(),
            "openai_compat": OpenAICompatibleAdapter(
                environ=environ,
                transport=transport,
                redactor=self.redactor,
            ),
            "anthropic": AnthropicAdapter(
                environ=environ,
                transport=transport,
                redactor=self.redactor,
            ),
            "gemini": GoogleGeminiAdapter(
                environ=environ,
                transport=transport,
                redactor=self.redactor,
            ),
        }
        if adapters:
            default_adapters.update(adapters)
        self._adapters: dict[ProviderName, ProviderAdapter] = default_adapters

    def register_adapter(self, provider: ProviderName, adapter: ProviderAdapter) -> None:
        """Override the adapter for a specific provider."""
        self._adapters[provider] = adapter

    def get_adapter(self, provider: ProviderName) -> ProviderAdapter:
        """Return the registered adapter for ``provider``."""
        return self._adapters[provider]

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Route ``request`` by role, atomically reserve budget, and invoke the adapter."""
        role_cfg = self.config.models.for_role(request.role)
        reservation = self.spend_tracker.reserve_before_call(role_cfg, request)
        adapter = self._adapters[role_cfg.provider]

        try:
            response = await adapter.complete(request, role_cfg)
        except LLMError:
            self.spend_tracker.release_reservation(reservation)
            raise
        except BaseException as exc:
            self.spend_tracker.release_reservation(reservation)
            if isinstance(exc, Exception):
                safe_msg = self.redactor.redact_text(str(exc))
                raise LLMError(
                    f"LLM call failed for role={request.role!r} provider={role_cfg.provider!r}: "
                    f"{safe_msg}"
                ) from None
            raise

        self.spend_tracker.commit_reservation(
            reservation,
            role=request.role,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            cost_usd=response.estimated_cost_usd,
        )
        return response

    def complete_sync(self, request: LLMRequest) -> LLMResponse:
        """Synchronous helper for non-async callers (CLI, sync harness loops)."""
        return asyncio.run(self.complete(request))
