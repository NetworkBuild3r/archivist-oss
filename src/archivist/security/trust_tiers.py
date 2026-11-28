"""Attested trust tiers and request-context provenance (INIT-024/SPEC-010).

Trust hierarchy (strictly ordered):
    ``T0_HUMAN > T1_AGENT > T2_EXTERNAL > T3_QUARANTINED``

Trust tiers are derived from verified channel/actor provenance (``RequestContext``)
and are never self-escalatable via MCP tool arguments. Monotonicity prevents
lower-trust writes or consolidated outputs from superseding higher-trust facts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class TrustTier(str, Enum):
    """Attested provenance tier for stored memories."""

    T0_HUMAN = "T0_HUMAN"
    T1_AGENT = "T1_AGENT"
    T2_EXTERNAL = "T2_EXTERNAL"
    T3_QUARANTINED = "T3_QUARANTINED"


_TRUST_RANK: dict[TrustTier, int] = {
    TrustTier.T0_HUMAN: 3,
    TrustTier.T1_AGENT: 2,
    TrustTier.T2_EXTERNAL: 1,
    TrustTier.T3_QUARANTINED: 0,
}

_HUMAN_ACTOR_TYPES = frozenset({"verified_human", "human", "cli_human", "cli_pin"})
_AGENT_ACTOR_TYPES = frozenset({"verified_local", "verified_agent", "agent", "local_daemon"})
_EXTERNAL_CHANNELS = frozenset({"import", "web", "external", "tool_output", "webhook"})


class TrustTierViolationError(PermissionError):
    """Raised when a lower-trust write attempts to supersede a higher-trust memory."""

    def __init__(self, challenger: TrustTier, incumbent: TrustTier) -> None:
        self.challenger = challenger
        self.incumbent = incumbent
        super().__init__(
            f"Trust-tier monotonicity violation: {challenger.value} cannot supersede "
            f"{incumbent.value}."
        )


def trust_tier_rank(tier: TrustTier | str) -> int:
    """Return numeric authority rank (higher = more trusted)."""
    parsed = tier if isinstance(tier, TrustTier) else TrustTier(str(tier))
    return _TRUST_RANK[parsed]


def can_supersede(challenger: TrustTier | str, incumbent: TrustTier | str) -> bool:
    """Return True iff *challenger* is allowed to supersede *incumbent*.

    ``T3_QUARANTINED`` can never supersede any memory (not even another
    quarantined memory). Otherwise, ``rank(challenger) >= rank(incumbent)``.
    """
    c = challenger if isinstance(challenger, TrustTier) else TrustTier(str(challenger))
    i = incumbent if isinstance(incumbent, TrustTier) else TrustTier(str(incumbent))
    if c is TrustTier.T3_QUARANTINED:
        return False
    return _TRUST_RANK[c] >= _TRUST_RANK[i]


def assert_monotonic_supersession(
    challenger: TrustTier | str,
    incumbent: TrustTier | str,
) -> None:
    """Raise :class:`TrustTierViolationError` if supersession violates monotonicity."""
    c = challenger if isinstance(challenger, TrustTier) else TrustTier(str(challenger))
    i = incumbent if isinstance(incumbent, TrustTier) else TrustTier(str(incumbent))
    if not can_supersede(c, i):
        raise TrustTierViolationError(c, i)


def min_trust_tier(tiers: Iterable[TrustTier | str]) -> TrustTier:
    """Return the lowest trust tier among *tiers* (fail-closed to ``T2_EXTERNAL`` if empty)."""
    parsed = [t if isinstance(t, TrustTier) else TrustTier(str(t)) for t in tiers]
    if not parsed:
        return TrustTier.T2_EXTERNAL
    return min(parsed, key=lambda t: _TRUST_RANK[t])


def is_retrievable_by_default(tier: TrustTier | str) -> bool:
    """Return False for ``T3_QUARANTINED`` so poisoned memories stay out of default recall."""
    parsed = tier if isinstance(tier, TrustTier) else TrustTier(str(tier))
    return parsed is not TrustTier.T3_QUARANTINED


def derive_trust_tier(*, actor_type: str, channel: str = "mcp") -> TrustTier:
    """Derive attested :class:`TrustTier` strictly from verified provenance.

    Never accepts caller-supplied trust overrides.
    """
    norm_channel = (channel or "").strip().lower()
    if norm_channel in _EXTERNAL_CHANNELS:
        return TrustTier.T2_EXTERNAL

    norm_actor = (actor_type or "").strip().lower()
    if norm_actor in _HUMAN_ACTOR_TYPES:
        return TrustTier.T0_HUMAN
    if norm_actor in _AGENT_ACTOR_TYPES:
        return TrustTier.T1_AGENT
    return TrustTier.T2_EXTERNAL


@dataclass(frozen=True)
class RequestContext:
    """Immutable caller identity derived from OS peer credentials or auth layer."""

    principal_id: str
    actor_type: str = "verified_local"
    tenant_id: str = "local"
    source_client: str = "unknown"
    channel: str = "mcp"
    uid: int | None = None

    @property
    def trust_tier(self) -> TrustTier:
        """Attested trust tier derived from ``actor_type`` and ``channel``."""
        return derive_trust_tier(actor_type=self.actor_type, channel=self.channel)

    @classmethod
    def for_local_uid(
        cls,
        uid: int,
        *,
        source_client: str = "stdio",
        channel: str = "mcp",
        actor_type: str = "verified_local",
    ) -> RequestContext:
        """Construct a local-daemon RequestContext bound to an OS UID."""
        return cls(
            principal_id=f"local:{uid}",
            actor_type=actor_type,
            tenant_id="local",
            source_client=source_client,
            channel=channel,
            uid=uid,
        )
