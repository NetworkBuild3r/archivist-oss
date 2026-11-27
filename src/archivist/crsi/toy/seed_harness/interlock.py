"""Editable toy interlock configuration wired into the seed harness."""

from __future__ import annotations

from archivist.crsi.harness.interlock import (
    CompletionAdvisor,
    ExecutionGatingInterlock,
    QualityFeasibilityCheck,
)

INTERLOCK_ENABLED = False
QUALITY_CHECK_ENABLED = False
V2_RECOVERY_GUIDANCE_ENABLED = False


def build_interlock() -> ExecutionGatingInterlock:
    """Build the ExecutionGatingInterlock plus any additive CompletionAdvisors."""
    advisors: list[CompletionAdvisor] = []
    if QUALITY_CHECK_ENABLED:
        advisors.append(QualityFeasibilityCheck(block_once=True))
    return ExecutionGatingInterlock(
        enabled=INTERLOCK_ENABLED,
        min_required_executions=1,
        max_rejections=3,
        advisors=advisors,
    )
