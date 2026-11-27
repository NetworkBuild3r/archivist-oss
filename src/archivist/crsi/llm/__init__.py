"""Model-agnostic LLM client, spend gate, FakeScriptedAdapter, and httpx provider adapters."""

from __future__ import annotations

from archivist.crsi.llm.anthropic import AnthropicAdapter
from archivist.crsi.llm.base import (
    LLMClient,
    LLMError,
    LLMRequest,
    LLMResponse,
    LLMRole,
    Message,
    MessageRole,
    ProviderAdapter,
    RoutedLLMClient,
)
from archivist.crsi.llm.budget import (
    BudgetExhaustedError,
    PaidApiNotEnabledError,
    SpendTracker,
    estimate_cost_usd,
    estimate_tokens,
)
from archivist.crsi.llm.fake import FakeScriptedAdapter, ScriptHandler
from archivist.crsi.llm.gemini import GoogleGeminiAdapter
from archivist.crsi.llm.openai_compat import (
    OpenAICompatibleAdapter,
    normalize_openai_base_url,
)

__all__ = [
    "AnthropicAdapter",
    "BudgetExhaustedError",
    "FakeScriptedAdapter",
    "GoogleGeminiAdapter",
    "LLMClient",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "LLMRole",
    "Message",
    "MessageRole",
    "OpenAICompatibleAdapter",
    "PaidApiNotEnabledError",
    "ProviderAdapter",
    "RoutedLLMClient",
    "ScriptHandler",
    "SpendTracker",
    "estimate_cost_usd",
    "estimate_tokens",
    "normalize_openai_base_url",
]
