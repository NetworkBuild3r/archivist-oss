"""Pre-call spend tracking, paid-API opt-in gate, and call-count enforcement."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any

from archivist.crsi.config import BudgetConfig, ModelRoleConfig
from archivist.crsi.llm.base import LLMRequest, LLMRole


class PaidApiNotEnabledError(RuntimeError):
    """Raised when a paid provider is invoked without explicit opt-in and budget."""


class BudgetExhaustedError(RuntimeError):
    """Raised when max_llm_calls or max_spend_usd would be exceeded."""


@dataclass(frozen=True)
class BudgetReservation:
    """In-flight call & spend reservation held atomically across async HTTP I/O."""

    estimated_cost_usd: float
    is_paid: bool


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
        self._in_flight_calls: int = 0
        self._spent_usd: float = 0.0
        self._reserved_usd: float = 0.0
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

    def _estimate_call_cost(
        self,
        role_config: ModelRoleConfig,
        request: LLMRequest,
    ) -> tuple[bool, float]:
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
        return is_paid, est_cost

    def check_before_call(
        self,
        role_config: ModelRoleConfig,
        request: LLMRequest,
    ) -> float:
        """Read-only pre-call budget check (prefer ``reserve_before_call`` for concurrency)."""
        is_paid, est_cost = self._estimate_call_cost(role_config, request)
        with self._lock:
            if (self._calls_used + self._in_flight_calls) >= self.budget.max_llm_calls:
                raise BudgetExhaustedError(
                    f"LLM call budget exhausted ({self._calls_used + self._in_flight_calls}/"
                    f"{self.budget.max_llm_calls}); limit set by budget.max_llm_calls."
                )
            projected = self._spent_usd + self._reserved_usd + est_cost
            if is_paid and (projected > self.budget.max_spend_usd + 1e-12):
                raise BudgetExhaustedError(
                    f"Projected LLM spend (${projected:.6f}) would exceed "
                    f"budget.max_spend_usd (${self.budget.max_spend_usd:.6f})."
                )
        return est_cost

    def reserve_before_call(
        self,
        role_config: ModelRoleConfig,
        request: LLMRequest,
    ) -> BudgetReservation:
        """Atomically verify and reserve call & spend capacity before async/threaded I/O."""
        is_paid, est_cost = self._estimate_call_cost(role_config, request)
        with self._lock:
            if (self._calls_used + self._in_flight_calls) >= self.budget.max_llm_calls:
                raise BudgetExhaustedError(
                    f"LLM call budget exhausted ({self._calls_used + self._in_flight_calls}/"
                    f"{self.budget.max_llm_calls}); limit set by budget.max_llm_calls."
                )
            projected = self._spent_usd + self._reserved_usd + est_cost
            if is_paid and (projected > self.budget.max_spend_usd + 1e-12):
                raise BudgetExhaustedError(
                    f"Projected LLM spend (${projected:.6f}) would exceed "
                    f"budget.max_spend_usd (${self.budget.max_spend_usd:.6f})."
                )
            self._in_flight_calls += 1
            if is_paid:
                self._reserved_usd = round(self._reserved_usd + est_cost, 8)
        return BudgetReservation(estimated_cost_usd=est_cost, is_paid=is_paid)

    def commit_reservation(
        self,
        reservation: BudgetReservation,
        *,
        role: LLMRole,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
    ) -> None:
        """Atomically release ``reservation`` and record actual token/cost usage."""
        with self._lock:
            self._in_flight_calls = max(0, self._in_flight_calls - 1)
            if reservation.is_paid:
                self._reserved_usd = max(
                    0.0, round(self._reserved_usd - reservation.estimated_cost_usd, 8)
                )
            self._calls_used += 1
            self._prompt_tokens += max(0, prompt_tokens)
            self._completion_tokens += max(0, completion_tokens)
            self._spent_usd = round(self._spent_usd + max(0.0, cost_usd), 8)
            self._calls_by_role[role] = self._calls_by_role.get(role, 0) + 1

    def release_reservation(self, reservation: BudgetReservation) -> None:
        """Release an uncompleted reservation after a failed/cancelled LLM call."""
        with self._lock:
            self._in_flight_calls = max(0, self._in_flight_calls - 1)
            if reservation.is_paid:
                self._reserved_usd = max(
                    0.0, round(self._reserved_usd - reservation.estimated_cost_usd, 8)
                )

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
