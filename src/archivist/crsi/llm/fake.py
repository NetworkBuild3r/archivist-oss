"""Deterministic scripted fake LLM adapter for zero-cost CI and toy scenarios."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable, Mapping, Sequence

from archivist.crsi.config import ModelRoleConfig
from archivist.crsi.llm.base import LLMRequest, LLMResponse, LLMRole
from archivist.crsi.llm.budget import estimate_request_prompt_tokens, estimate_tokens

ScriptHandler = Callable[[LLMRequest, ModelRoleConfig], str]

_DEFAULT_ROLE_RESPONSES: dict[LLMRole, str] = {
    "policy": '{"action": "finish", "answer": "done"}',
    "proposer": '{"candidates": []}',
    "selector": '{"selected": "NONE", "reason": "default fake selector"}',
}


class FakeScriptedAdapter:
    """Deterministic in-process adapter supporting queued scripts or dynamic handlers."""

    def __init__(
        self,
        *,
        responses: Mapping[LLMRole, Sequence[str]] | None = None,
        handlers: Mapping[LLMRole, ScriptHandler] | None = None,
        repeat_last: bool = True,
    ) -> None:
        self._queues: dict[LLMRole, deque[str]] = {
            role: deque(items) for role, items in (responses or {}).items()
        }
        self._handlers: dict[LLMRole, ScriptHandler] = dict(handlers or {})
        self._repeat_last = repeat_last
        self.call_log: list[LLMRequest] = []

    def set_handler(self, role: LLMRole, handler: ScriptHandler) -> None:
        """Register or replace a deterministic callback for ``role``."""
        self._handlers[role] = handler

    def enqueue(self, role: LLMRole, *items: str) -> None:
        """Append scripted responses to the FIFO queue for ``role``."""
        queue = self._queues.setdefault(role, deque())
        queue.extend(items)

    async def complete(
        self,
        request: LLMRequest,
        role_config: ModelRoleConfig,
    ) -> LLMResponse:
        self.call_log.append(request)
        role = request.role

        if role in self._handlers:
            content = self._handlers[role](request, role_config)
        elif self._queues.get(role):
            queue = self._queues[role]
            content = queue.popleft()
            if not queue and self._repeat_last:
                queue.append(content)
        else:
            content = _DEFAULT_ROLE_RESPONSES[role]

        return LLMResponse(
            content=content,
            prompt_tokens=estimate_request_prompt_tokens(request),
            completion_tokens=estimate_tokens(content),
            estimated_cost_usd=0.0,
            model=role_config.model,
            provider="fake",
            latency_ms=0.0,
        )
