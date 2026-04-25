"""Tier-aware context packing for token-budgeted retrieval assembly.

Replaces the greedy first-fit truncation in ``rlm_retriever.py`` with a
3-pass adaptive algorithm that maximises coverage per token:

1. L0 pass  — include the shortest (headline) tier for every result until
               ``L0_BUDGET_SHARE`` of the budget is consumed.
2. L1 pass  — upgrade results to their L1 (overview) tier, best-score first,
               while budget remains.
3. L2 pass  — upgrade the top-K results to full-content (L2) tier.

The ``l0_first`` and ``l2_first`` policies degenerate the algorithm to
single-pass behaviour for agents that prefer a simpler strategy.

All callers should import from this module:

    from archivist.retrieval.context_packer import pack_context, PackedContext
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from archivist.utils.tokenizer import count_tokens
from archivist.write.tiering import select_tier

logger = logging.getLogger("archivist.context_packer")

# Tier ordering for upgrade passes (lowest token cost → highest)
_TIER_ORDER = ("l0", "l1", "l2")


@dataclass
class PackedContext:
    """Result of a tier-aware packing run."""

    sources: list[dict]
    """Packed result dicts — each has a ``tier_text`` key with the chosen tier text."""

    total_tokens: int
    """Actual tokens consumed by the packed sources."""

    budget_tokens: int
    """The max_tokens budget that was applied."""

    over_budget: bool
    """True when the full result set would have exceeded the budget."""

    dropped_count: int
    """Number of results that were excluded entirely due to budget exhaustion."""

    tier_distribution: dict[str, int] = field(default_factory=dict)
    """Breakdown of how many results were packed at each tier level."""

    token_savings_pct: float = 0.0
    """Estimated savings vs returning every result at L2 (0–100)."""

    naive_tokens: int = 0
    """Total tokens if every result were returned at L2 (baseline for savings calculation)."""


def pack_context(
    results: list[dict],
    max_tokens: int,
    tier_policy: str = "adaptive",
    l0_budget_share: float = 0.30,
    min_full_results: int = 3,
) -> PackedContext:
    """Pack *results* into a token budget using the specified tier policy.

    Args:
        results: Ranked list of retrieval result dicts.  Each dict should have
            score-ordered ranking already applied.  Dicts may optionally carry
            ``l0``, ``l1``, and ``l2`` / ``text`` tier text already populated
            by the write pipeline.
        max_tokens: Hard upper bound on total tokens in the packed output.
        tier_policy: One of ``"adaptive"`` (default), ``"l0_first"``, or
            ``"l2_first"``.
        l0_budget_share: Fraction of ``max_tokens`` reserved for the L0 pass
            in adaptive mode (default 0.30).
        min_full_results: Minimum number of results that should be upgraded to
            their best available tier regardless of budget.

    Returns:
        :class:`PackedContext` with packed sources and metadata.
    """
    if not results or max_tokens <= 0:
        return PackedContext(
            sources=[],
            total_tokens=0,
            budget_tokens=max_tokens,
            over_budget=False,
            dropped_count=0,
            tier_distribution={},
            token_savings_pct=0.0,
        )

    policy = tier_policy.lower()
    if policy == "l2_first":
        return _pack_greedy(results, max_tokens, prefer_tier="l2", min_full=min_full_results)
    if policy == "l0_first":
        return _pack_greedy(results, max_tokens, prefer_tier="l0", min_full=min_full_results)
    return _pack_adaptive(results, max_tokens, l0_budget_share, min_full_results)


# ---------------------------------------------------------------------------
# Internal packing implementations
# ---------------------------------------------------------------------------


def _best_tier_text(result: dict, tier: str) -> str:
    """Return the best available tier text for *result* at or below *tier*."""
    order = _TIER_ORDER
    idx = order.index(tier) if tier in order else len(order) - 1
    for t in order[: idx + 1]:
        text = select_tier(result, t)
        if text:
            return text
    return select_tier(result, "l2") or ""


def _naive_l2_tokens(results: list[dict]) -> int:
    """Total tokens if every result were returned at L2 (worst case baseline)."""
    return sum(count_tokens(select_tier(r, "l2") or "") for r in results)


def _pack_greedy(
    results: list[dict],
    max_tokens: int,
    prefer_tier: str,
    min_full: int,
) -> PackedContext:
    """Single-pass greedy packing: include results at *prefer_tier* until budget exhausted."""
    budget = 0
    packed: list[dict] = []
    tier_dist: dict[str, int] = {}
    dropped = 0
    naive_total = _naive_l2_tokens(results)

    for i, r in enumerate(results):
        tier = prefer_tier
        text = _best_tier_text(r, tier)
        toks = count_tokens(text)
        if budget + toks > max_tokens and i >= min_full:
            dropped += 1
            continue
        out = dict(r)
        out["tier_text"] = text
        out["_packed_tier"] = tier
        packed.append(out)
        budget += toks
        tier_dist[tier] = tier_dist.get(tier, 0) + 1

    savings = round((1 - budget / naive_total) * 100, 1) if naive_total > 0 else 0.0
    return PackedContext(
        sources=packed,
        total_tokens=budget,
        budget_tokens=max_tokens,
        over_budget=dropped > 0 or budget >= max_tokens,
        dropped_count=dropped,
        tier_distribution=tier_dist,
        token_savings_pct=savings,
        naive_tokens=naive_total,
    )


def _pack_adaptive(
    results: list[dict],
    max_tokens: int,
    l0_budget_share: float,
    min_full: int,
) -> PackedContext:
    """3-pass adaptive packing: L0 → L1 upgrade → L2 upgrade for top-K."""
    l0_cap = int(max_tokens * l0_budget_share)
    naive_total = _naive_l2_tokens(results)

    # Pass 1: include L0 (or best available below L0) for all results.
    slots: list[dict] = []
    budget = 0
    for r in results:
        text = _best_tier_text(r, "l0")
        toks = count_tokens(text)
        if budget + toks <= l0_cap or len(slots) < min_full:
            slots.append({"result": r, "tier": "l0", "text": text, "tokens": toks})
            budget += toks
        else:
            slots.append({"result": r, "tier": None, "text": "", "tokens": 0})

    remaining_cap = max_tokens - budget

    # Pass 2: upgrade to L1 in score order while budget allows.
    for slot in slots:
        if remaining_cap <= 0:
            break
        if slot["tier"] != "l0":
            continue
        l1_text = _best_tier_text(slot["result"], "l1")
        l1_toks = count_tokens(l1_text)
        delta = l1_toks - slot["tokens"]
        if delta <= remaining_cap:
            slot["tier"] = "l1"
            slot["text"] = l1_text
            slot["tokens"] = l1_toks
            budget += delta
            remaining_cap -= delta

    # Pass 3: upgrade to L2 for top-min_full results in score order.
    upgraded = 0
    for slot in slots:
        if upgraded >= min_full:
            break
        if slot["tier"] not in ("l1", "l0"):
            continue
        l2_text = _best_tier_text(slot["result"], "l2")
        l2_toks = count_tokens(l2_text)
        delta = l2_toks - slot["tokens"]
        if delta <= remaining_cap:
            slot["tier"] = "l2"
            slot["text"] = l2_text
            slot["tokens"] = l2_toks
            budget += delta
            remaining_cap -= delta
            upgraded += 1

    # Assemble output — drop slots that have no text (budget entirely exhausted)
    packed: list[dict] = []
    dropped = 0
    tier_dist: dict[str, int] = {}
    for slot in slots:
        if not slot["text"]:
            dropped += 1
            continue
        out = dict(slot["result"])
        out["tier_text"] = slot["text"]
        out["_packed_tier"] = slot["tier"] or "l2"
        packed.append(out)
        t = slot["tier"] or "l2"
        tier_dist[t] = tier_dist.get(t, 0) + 1

    savings = round((1 - budget / naive_total) * 100, 1) if naive_total > 0 else 0.0
    return PackedContext(
        sources=packed,
        total_tokens=budget,
        budget_tokens=max_tokens,
        over_budget=dropped > 0 or budget >= max_tokens,
        dropped_count=dropped,
        tier_distribution=tier_dist,
        token_savings_pct=savings,
        naive_tokens=naive_total,
    )
