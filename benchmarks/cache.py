"""Deterministic SHA-256 disk cache and hard USD spend cap for benchmarks.

Used by LongMemEval (`adapter.py`) and the Wave-1 ONNX ablation harness
(`ablation_onnx.py`) so embeddings, LLM answers, and judge verdicts are cached
across reruns (`<$0.50` per cached rerun) and live API spend can never exceed
the configured ceiling (`<= $5.00` for `INIT-023/SPEC-004`).
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TypeVar

T = TypeVar("T")


class BudgetExceededError(RuntimeError):
    """Raised before issuing an API call that would exceed the configured USD cap."""

    def __init__(self, spent_usd: float, projected_usd: float, cap_usd: float) -> None:
        self.spent_usd = spent_usd
        self.projected_usd = projected_usd
        self.cap_usd = cap_usd
        super().__init__(
            f"Benchmark spend cap exceeded: spent=${spent_usd:.4f}, "
            f"projected=${projected_usd:.4f} > cap=${cap_usd:.4f}"
        )


# Conservative per-1M-token pricing table (USD) used for pre-call budget gating.
# Unknown models fall back to a conservative upper bound so spend stays strictly bounded.
MODEL_PRICING_PER_1M: dict[str, tuple[float, float]] = {
    # (input_usd_per_1m, output_usd_per_1m)
    "text-embedding-3-small": (0.02, 0.0),
    "text-embedding-3-large": (0.13, 0.0),
    "text-embedding-ada-002": (0.10, 0.0),
    "gpt-4o-2024-08-06": (2.50, 10.00),
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4o-mini-2024-07-18": (0.15, 0.60),
}
_DEFAULT_PRICING_PER_1M: tuple[float, float] = (2.50, 10.00)


def estimate_call_usd(model: str, input_tokens: int, output_tokens: int = 0) -> float:
    """Estimate USD cost for a single embedding or chat-completion call."""
    key = (model or "").strip().lower()
    if (
        key.startswith(("local:", "onnx:"))
        or "bge-" in key
        or "nomic-" in key
        or "arctic-" in key
        or "minilm" in key
    ):
        return 0.0
    in_rate, out_rate = MODEL_PRICING_PER_1M.get(key, _DEFAULT_PRICING_PER_1M)
    return (max(0, input_tokens) / 1_000_000.0) * in_rate + (
        max(0, output_tokens) / 1_000_000.0
    ) * out_rate


@dataclass
class SpendTracker:
    """Tracks cumulative benchmark API spend and enforces a hard USD cap."""

    max_budget_usd: float = 5.0
    spent_usd: float = 0.0
    calls_by_stage: dict[str, int] = field(default_factory=dict)
    usd_by_stage: dict[str, float] = field(default_factory=dict)

    def check_and_reserve(
        self,
        *,
        stage: str,
        model: str,
        input_tokens: int,
        output_tokens: int = 0,
    ) -> float:
        """Raise :class:`BudgetExceededError` if this call would breach the cap.

        Records the estimated cost immediately when allowed so concurrent or
        sequential callers cannot overshoot ``max_budget_usd``.
        """
        cost = estimate_call_usd(model, input_tokens, output_tokens)
        projected = self.spent_usd + cost
        if self.max_budget_usd > 0 and projected > self.max_budget_usd + 1e-9:
            raise BudgetExceededError(self.spent_usd, projected, self.max_budget_usd)
        self.spent_usd = projected
        self.calls_by_stage[stage] = self.calls_by_stage.get(stage, 0) + 1
        self.usd_by_stage[stage] = round(self.usd_by_stage.get(stage, 0.0) + cost, 6)
        return cost

    def summary(self) -> dict[str, Any]:
        return {
            "max_budget_usd": round(self.max_budget_usd, 4),
            "estimated_usd": round(self.spent_usd, 6),
            "calls_by_stage": dict(self.calls_by_stage),
            "usd_by_stage": {k: round(v, 6) for k, v in self.usd_by_stage.items()},
        }


def sha256_key(namespace: str, payload: Any) -> str:
    """Compute a deterministic SHA-256 hex digest for *(namespace, payload)*."""
    canonical = json.dumps(
        {"ns": namespace, "payload": payload},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class DiskCache:
    """Content-addressed SHA-256 JSON disk cache with atomic writes.

    Each entry is stored at ``<cache_dir>/<namespace>/<digest[:2]>/<digest>.json``.
    Writes go to a sibling temporary file and replace atomically via
    ``os.replace()`` so interrupted runs never leave truncated cache files.
    """

    def __init__(self, cache_dir: str | Path) -> None:
        self.root = Path(cache_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits: int = 0
        self.misses: int = 0

    def _entry_path(self, namespace: str, digest: str) -> Path:
        safe_ns = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in namespace)
        return self.root / safe_ns / digest[:2] / f"{digest}.json"

    def get(self, namespace: str, payload: Any) -> tuple[bool, Any]:
        """Return ``(True, value)`` on cache hit or ``(False, None)`` on miss."""
        digest = sha256_key(namespace, payload)
        path = self._entry_path(namespace, digest)
        if not path.is_file():
            self.misses += 1
            return False, None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            self.hits += 1
            return True, data["value"]
        except (OSError, ValueError, KeyError):
            self.misses += 1
            return False, None

    def set(self, namespace: str, payload: Any, value: Any) -> str:
        """Atomically persist *value* under the SHA-256 key for *(namespace, payload)*."""
        digest = sha256_key(namespace, payload)
        path = self._entry_path(namespace, digest)
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = json.dumps({"digest": digest, "value": value}, ensure_ascii=False)
        fd, tmp_name = tempfile.mkstemp(prefix=".tmp_cache_", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(serialized)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_name, path)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        return digest

    async def get_or_compute_async(
        self,
        namespace: str,
        payload: Any,
        compute: Callable[[], Awaitable[T]],
    ) -> T:
        """Return cached value or await *compute()*, persist atomically, and return."""
        found, value = self.get(namespace, payload)
        if found:
            return value  # type: ignore[no-any-return]
        computed = await compute()
        self.set(namespace, payload, computed)
        return computed

    def stats(self) -> dict[str, int]:
        return {"hits": self.hits, "misses": self.misses}
