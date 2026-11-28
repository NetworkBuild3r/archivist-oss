"""Archivist v3 core security layer (INIT-024/SPEC-010).

Provides the linear-time ``google-re2`` AI Memory Firewall, attested trust
tiers (``T0_HUMAN`` .. ``T3_QUARANTINED``), monotonicity validation, and
structural ``<archivist_memory>`` read framing. Strictly decoupled from
``src/archivist/crsi/`` (GR-LAYER-001).
"""

from archivist.security.firewall import (
    MAX_FIREWALL_INPUT_BYTES,
    SECRET_PATTERN_NAMES,
    FirewallBlockedError,
    FirewallFinding,
    FirewallPolicy,
    FirewallResult,
    MemoryFirewall,
    default_firewall,
    frame_memories,
    frame_memory,
    screen_and_sanitize,
)
from archivist.security.trust_tiers import (
    RequestContext,
    TrustTier,
    TrustTierViolationError,
    assert_monotonic_supersession,
    can_supersede,
    derive_trust_tier,
    is_retrievable_by_default,
    min_trust_tier,
    trust_tier_rank,
)

__all__ = [
    "MAX_FIREWALL_INPUT_BYTES",
    "SECRET_PATTERN_NAMES",
    "FirewallBlockedError",
    "FirewallFinding",
    "FirewallPolicy",
    "FirewallResult",
    "MemoryFirewall",
    "RequestContext",
    "TrustTier",
    "TrustTierViolationError",
    "assert_monotonic_supersession",
    "can_supersede",
    "default_firewall",
    "derive_trust_tier",
    "frame_memories",
    "frame_memory",
    "is_retrievable_by_default",
    "min_trust_tier",
    "screen_and_sanitize",
    "trust_tier_rank",
]
