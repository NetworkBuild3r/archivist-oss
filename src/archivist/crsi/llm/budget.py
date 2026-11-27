"""Pre-call spend tracking, paid-API opt-in gate, and call-count enforcement."""

from __future__ import annotations

import threading
from typing import Any

from archivist.crsi.config import BudgetConfig, ModelRoleConfig
from archivist.crsi.llm.base import LLMRequest, LLMRole


class PaidApiNotEnabledError(RuntimeError):
    """Raised when a paid provider is invoked without explicit opt-in and budget."""


class BudgetExhaustedError(RuntimeError):
    """Raised when max_llm_calls or max_spend_usd would be exceeded."""


def estimate_tokens(text: str) -> int:
    """Deterministic lightweight token estimator (chars // 4 heuristic)."""
    if not text:
        return 0
    return max(1, (len(text) + 3) // 4)


def estimate_request_prompt_tokens(request: LLMRequest) -> int:
    """Estimate total prompt tokens across all messages in ``request``."""
    return sum(estimate_tokens(msg.content) + 4 for msg in request.messages)


def estimate_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    role_config: ModelRoleConfig,
) -> float:
    """Calculate USD cost using configured per-1M input/output token rates."""
    input_cost = (prompt_tokens / 1_000_000.0) * role_config.input_usd_per_1m
    output_cost = (completion_tokens / 1_000_000.0) * role_config.output_usd_per_1m
    return round(input_cost + output_cost, 8)


class SpendTracker:
    """Thread-safe pre-call gate and post-call ledger for LLM calls and USD spend."""

    def __init__(self, budget: BudgetConfig) -> None:
        self.budget = budget
        self._lock = threading.Lock()
        self._calls_used: int = 0
        self._spent_usd: float = 0.0
        self._prompt_tokens: int = 0
        self._completion_tokens: int = 0
        self._calls_by_role: dict[str, int] = {"policy": 0, "proposer": 0, "selector": 0}

    @property
    def calls_used(self) -> int:
        with self._lock:
            return self._calls_used

    @property
    def spent_usd(self) -> float:
        with self._lock:
            return self._spent_usd

    @property
    def prompt_tokens(self) -> int:
        with self._lock:
            return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        with self._lock:
            return self._completion_tokens

    def check_before_call(
        self,
        role_config: ModelRoleConfig,
        request: LLMRequest,
    ) -> float:
        """Enforce paid-API opt-in, call cap, and spend cap BEFORE any network I/O.

        Returns the estimated upper-bound USD cost for this call.
        """
        is_paid = role_config.is_paid_provider()
        if is_paid and (not self.budget.allow_paid_apis or self.budget.max_spend_usd <= 0.0):
            raise PaidApiNotEnabledError(
                f"Paid LLM provider {role_config.provider!r} ({role_config.model!r}) is blocked: "
                "set budget.allow_paid_apis=True and budget.max_spend_usd > 0.0 to opt in."
            )

        est_prompt = estimate_request_prompt_tokens(request)
        est_completion = (
            request.max_tokens if request.max_tokens is not None else role_config.max_tokens
        )
        est_cost = estimate_cost_usd(est_prompt, est_completion, role_config) if is_paid else 0.0

        with self._lock:
            if self._calls_used >= self.budget.max_llm_calls:
                raise BudgetExhaustedError(
                    f"LLM call budget exhausted ({self._calls_used}/{self.budget.max_llm_calls}); "
                    "limit set by budget.max_llm_calls."
                )
            if is_paid and (self._spent_usd + est_cost > self.budget.max_spend_usd + 1e-12):
                raise BudgetExhaustedError(
                    f"Projected LLM spend (${self._spent_usd + est_cost:.6f}) would exceed "
                    f"budget.max_spend_usd (${self.budget.max_spend_usd:.6f})."
                )
        return est_cost

    def record_call(
        self,
        *,
        role: LLMRole,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        """Record a completed LLM call's token counts and cost."""
        with self._lock:
            self._calls_used += 1
            self._prompt_tokens += max(0, prompt_tokens)
            self._completion_tokens += max(0, completion_tokens)
            self._spent_usd = round(self._spent_usd + max(0.0, cost_usd), 8)
            self._calls_by_role[role] = self._calls_by_role.get(role, 0) + 1

    def snapshot(self) -> dict[str, Any]:
        """Serialize current tracker counters for resumable run checkpoints."""
        with self._lock:
            return {
                "calls_used": self._calls_used,
                "spent_usd": self._spent_usd,
                "prompt_tokens": self._prompt_tokens,
                "completion_tokens": self._completion_tokens,
                "calls_by_role": dict(self._calls_by_role),
            }

    def restore_snapshot(self, data: dict[str, Any]) -> None:
        """Restore persisted counters when resuming an interrupted CRSI run."""
        with self._lock:
            self._calls_used = int(data.get("calls_used", 0))
            self._spent_usd = float(data.get("spent_usd", 0.0))
            self._prompt_tokens = int(data.get("prompt_tokens", 0))
            self._completion_tokens = int(data.get("completion_tokens", 0))
            raw_roles = data.get("calls_by_role", {})
            if isinstance(raw_roles, dict):
                self._calls_by_role = {str(k): int(v) for k, v in raw_roles.items()}
