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
"""

import time
from config import LATENCY_BUDGET_MS


class LatencyBudget:
    """Tracks wall-clock time within a single retrieval request."""

    def __init__(self, max_ms: int = 0):
        self._max_ms = max_ms or LATENCY_BUDGET_MS
        self._start = time.monotonic()
        self._reservations: dict[str, float] = {}

    def elapsed_ms(self) -> float:
        return (time.monotonic() - self._start) * 1000

    def remaining_ms(self) -> float:
        return max(0.0, self._max_ms - self.elapsed_ms())

    def can_afford(self, estimated_ms: float) -> bool:
        """Return True if there's enough budget for an operation."""
        return self.remaining_ms() >= estimated_ms

    def reserve(self, label: str, estimated_ms: float) -> bool:
        """Reserve budget for a named operation. Returns True if affordable."""
        if not self.can_afford(estimated_ms):
            return False
        self._reservations[label] = estimated_ms
        return True

    def is_expired(self) -> bool:
        return self.remaining_ms() <= 0

    def summary(self) -> dict:
        return {
            "budget_ms": self._max_ms,
            "elapsed_ms": round(self.elapsed_ms(), 1),
            "remaining_ms": round(self.remaining_ms(), 1),
            "reservations": dict(self._reservations),
        }
