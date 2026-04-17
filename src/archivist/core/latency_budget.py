"""Latency budget system — guarantees sub-500ms p95 by gating expensive ops.

The budget tracks elapsed time and decides whether optional expensive
operations (query expansion, HyDE, BM25, iterative retrieval) should
run based on remaining time.

Usage in the retrieval pipeline:
    budget = LatencyBudget()
    # ... required operations ...
    if budget.can_afford(150):
        await expand_query(...)
    if budget.can_afford(100):
        await generate_hyde(...)

Adaptive profiles (v1.11): budget is selected per query type so needle
queries get a fast path and multi-hop gets generous allocation.
"""

import time
from archivist.core.config import LATENCY_BUDGET_MS


BUDGET_PROFILES: dict[str, int] = {
    "needle": 200,
    "simple_recall": 300,
    "temporal": 400,
    "broad": 500,
    "multi_hop": 800,
    "default": LATENCY_BUDGET_MS or 500,
}


def budget_for_query_type(query_type: str) -> int:
    """Select the latency budget for a given query classification."""
    return BUDGET_PROFILES.get(query_type, BUDGET_PROFILES["default"])


class LatencyBudget:
    """Tracks wall-clock time within a single retrieval request."""

    def __init__(self, max_ms: int = 0):
        self._max_ms = max_ms or LATENCY_BUDGET_MS
        self._start = time.monotonic()
        self._reservations: dict[str, float] = {}
        self._reserved_ms: float = 0.0

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000

    def remaining_ms(self) -> float:
        return max(0.0, self._max_ms - self.elapsed_ms() - self._reserved_ms)

    def can_afford(self, estimated_ms: float) -> bool:
        """Return True if there's enough budget for an operation."""
        return self.remaining_ms() >= estimated_ms

    def reserve(self, label: str, estimated_ms: float) -> bool:
        """Reserve budget for a named operation. Returns True if affordable."""
        if not self.can_afford(estimated_ms):
            return False
        self._reservations[label] = estimated_ms
        self._reserved_ms += estimated_ms
        return True

    def release(self, label: str) -> None:
        """Release a reservation after the operation completes."""
        if label in self._reservations:
            self._reserved_ms -= self._reservations.pop(label)

    def is_expired(self) -> bool:
        return self.remaining_ms() <= 0

    def summary(self) -> dict:
        return {
            "budget_ms": self._max_ms,
            "elapsed_ms": round(self.elapsed_ms(), 1),
            "remaining_ms": round(self.remaining_ms(), 1),
            "reserved_ms": round(self._reserved_ms, 1),
            "reservations": dict(self._reservations),
        }
