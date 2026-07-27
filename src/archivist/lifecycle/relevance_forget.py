"""Relevance-based forget — Diff #6 product path (INIT-010/SPEC-003).

Selects cold / low-importance ``memory_chunks`` using hotness + age/TTL signals
and either audits a proposal (dry-run) or suppresses them from default recall
via ``suppress_memory`` (namespace-scoped — not coach delete / raw TTL alone).

Safe defaults (ADR-010 / GR-SAFE-001): ``RELEVANCE_FORGET_ENABLED=false``,
``RELEVANCE_FORGET_DRY_RUN=true``. No net-new core MCP tools.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from archivist.core.audit import log_memory_event
from archivist.core.config import (
    RELEVANCE_FORGET_DRY_RUN,
    RELEVANCE_FORGET_ENABLED,
    RELEVANCE_FORGET_HOTNESS_MAX,
    RELEVANCE_FORGET_IMPORTANCE_MAX,
    RELEVANCE_FORGET_MAX_PER_CYCLE,
    RELEVANCE_FORGET_MIN_AGE_DAYS,
)

logger = logging.getLogger("archivist.relevance_forget")

# Pin / permanent retention floor — mirrors archivist_pin importance_score=1.0
# and excludes durable-class payloads from auto-suppress (SEC-010-01).
_PROTECTED_IMPORTANCE_MIN = 0.9
_PROTECTED_RETENTION = frozenset({"durable", "permanent"})


@dataclass
class ForgetProposal:
    """One relevance-forget candidate (proposed or applied)."""

    memory_id: str
    namespace: str
    agent_id: str
    hotness: float | None = None
    importance: float = 0.5
    dry_run: bool = True
    applied: bool = False
    rule: str = "cold_low_importance"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "memory_id": self.memory_id,
            "namespace": self.namespace,
            "agent_id": self.agent_id,
            "hotness": self.hotness,
            "importance": self.importance,
            "dry_run": self.dry_run,
            "applied": self.applied,
            "rule": self.rule,
            "metadata": dict(self.metadata),
        }


async def _candidate_chunks(
    *,
    limit: int,
    hotness_max: float,
    importance_max: float,
    min_age_days: int,
    namespace: str = "",
) -> list[ForgetProposal]:
    """Return cold, low-importance chunks eligible for relevance forget."""
    from archivist.storage.sqlite_pool import pool

    if limit <= 0:
        return []

    age_cutoff = (datetime.now(UTC) - timedelta(days=max(min_age_days, 0))).isoformat()
    now_iso = datetime.now(UTC).isoformat()
    ns_clause = ""
    params: list[Any] = [importance_max, hotness_max, age_cutoff, now_iso]
    if namespace:
        ns_clause = "AND mc.namespace = ?"
        params.append(namespace)
    params.append(limit)

    sql = f"""
        SELECT mc.qdrant_id, mc.namespace, mc.agent_id, mc.importance,
               h.score AS hotness, mc.ttl_at, mc.created_at
        FROM memory_chunks mc
        LEFT JOIN memory_hotness h ON h.memory_id = mc.qdrant_id
        WHERE COALESCE(mc.is_suppressed, 0) = 0
          AND COALESCE(mc.is_excluded, 0) = 0
          AND mc.qdrant_id NOT LIKE 'reconsolidate:%'
          AND mc.namespace != ''
          AND COALESCE(mc.importance, 0.5) <= ?
          AND (
                (h.score IS NOT NULL AND h.score <= ?)
             OR (
                  h.score IS NULL
                  AND (
                        (mc.created_at IS NOT NULL AND mc.created_at != ''
                         AND mc.created_at < ?)
                     OR (mc.ttl_at IS NOT NULL AND mc.ttl_at != ''
                         AND mc.ttl_at < ?)
                  )
             )
          )
          {ns_clause}
        ORDER BY COALESCE(h.score, 0.0) ASC, COALESCE(mc.importance, 0.5) ASC
        LIMIT ?
    """

    async with pool.read() as conn:
        rows = await conn.fetchall(sql, tuple(params))

    out: list[ForgetProposal] = []
    for row in rows:
        mid = row["qdrant_id"] if isinstance(row, dict) else row[0]
        ns = row["namespace"] if isinstance(row, dict) else row[1]
        aid = row["agent_id"] if isinstance(row, dict) else row[2]
        imp = float(row["importance"] if isinstance(row, dict) else row[3] or 0.5)
        hot_raw = row["hotness"] if isinstance(row, dict) else row[4]
        hot = float(hot_raw) if hot_raw is not None else None
        ttl = row["ttl_at"] if isinstance(row, dict) else row[5]
        rule = "cold_low_importance"
        if ttl and str(ttl) < now_iso:
            rule = "ttl_expired_cold"
        elif hot is None:
            rule = "unscored_aged_low_importance"
        out.append(
            ForgetProposal(
                memory_id=str(mid),
                namespace=str(ns),
                agent_id=str(aid or ""),
                hotness=hot,
                importance=imp,
                rule=rule,
                metadata={"ttl_at": ttl or ""},
            )
        )
    return out


async def _qdrant_retention_class(memory_id: str, namespace: str) -> str | None:
    """Best-effort Qdrant retention_class lookup (pin lives on vector payload)."""
    try:
        from archivist.storage.collection_router import collection_for
        from archivist.storage.qdrant import qdrant_client

        client = qdrant_client()
        coll = collection_for(namespace)
        points = client.retrieve(collection_name=coll, ids=[memory_id], with_payload=True)
        if not points:
            return None
        payload = getattr(points[0], "payload", None) or {}
        rc = payload.get("retention_class")
        return str(rc) if rc else None
    except Exception as e:
        logger.debug(
            "relevance_forget retention lookup failed id=%s ns=%s: %s",
            memory_id,
            namespace,
            e,
        )
        return None


async def is_protected_from_forget(
    memory_id: str,
    namespace: str,
    *,
    importance: float | None = None,
) -> tuple[bool, str]:
    """Return (protected, reason) for pin / durable / high-importance memories.

    SEC-010-01: archivist_pin sets Qdrant retention_class=permanent; forget must
    not suppress those rows even if SQLite importance lagged.
    """
    if importance is not None and importance >= _PROTECTED_IMPORTANCE_MIN:
        return True, "high_importance"
    rc = await _qdrant_retention_class(memory_id, namespace)
    if rc and rc.lower() in _PROTECTED_RETENTION:
        return True, f"retention_{rc.lower()}"
    return False, ""


async def apply_relevance_forget(
    proposal: ForgetProposal,
    *,
    dry_run: bool | None = None,
) -> ForgetProposal:
    """Audit and optionally suppress a cold memory (namespace-scoped)."""
    effective_dry = RELEVANCE_FORGET_DRY_RUN if dry_run is None else dry_run
    proposal.dry_run = effective_dry

    protected, protect_reason = await is_protected_from_forget(
        proposal.memory_id,
        proposal.namespace,
        importance=proposal.importance,
    )
    if protected:
        proposal.applied = False
        proposal.metadata["skipped"] = protect_reason
        await log_memory_event(
            agent_id=proposal.agent_id or "system:relevance_forget",
            action="relevance_forget_skipped",
            memory_id=proposal.memory_id,
            namespace=proposal.namespace,
            text_hash=proposal.memory_id,
            metadata={
                "reason": protect_reason,
                "hotness": proposal.hotness,
                "importance": proposal.importance,
                "dry_run": effective_dry,
            },
        )
        logger.info(
            "relevance_forget skipped=1 reason=%s ns=%s id=%s",
            protect_reason,
            proposal.namespace,
            proposal.memory_id,
        )
        return proposal

    audit_meta = {
        "hotness": proposal.hotness,
        "importance": proposal.importance,
        "rule": proposal.rule,
        "dry_run": effective_dry,
        **{k: v for k, v in proposal.metadata.items() if k != "skipped"},
    }

    if effective_dry:
        await log_memory_event(
            agent_id=proposal.agent_id or "system:relevance_forget",
            action="relevance_forget_proposed",
            memory_id=proposal.memory_id,
            namespace=proposal.namespace,
            text_hash=proposal.memory_id,
            metadata=audit_meta,
        )
        proposal.applied = False
        logger.info(
            "relevance_forget dry_run=1 ns=%s id=%s hotness=%s",
            proposal.namespace,
            proposal.memory_id,
            proposal.hotness,
        )
        return proposal

    from archivist.lifecycle.correct import suppress_memory

    await suppress_memory(
        proposal.memory_id,
        proposal.namespace,
        agent_id=proposal.agent_id or "system:relevance_forget",
        reason=f"relevance_forget:{proposal.rule}",
    )
    await log_memory_event(
        agent_id=proposal.agent_id or "system:relevance_forget",
        action="relevance_forget_applied",
        memory_id=proposal.memory_id,
        namespace=proposal.namespace,
        text_hash=proposal.memory_id,
        metadata=audit_meta,
    )
    proposal.applied = True
    logger.info(
        "relevance_forget applied=1 ns=%s id=%s",
        proposal.namespace,
        proposal.memory_id,
    )
    return proposal


async def relevance_forget_cycle(
    *,
    dry_run: bool | None = None,
    max_forget: int | None = None,
    namespace: str = "",
) -> dict[str, Any]:
    """Curator hook: propose/apply relevance forget. No-op when master flag is off."""
    if not RELEVANCE_FORGET_ENABLED:
        return {
            "enabled": False,
            "proposed": 0,
            "applied": 0,
            "dry_run": True,
            "results": [],
        }

    cap = RELEVANCE_FORGET_MAX_PER_CYCLE if max_forget is None else max_forget
    candidates = await _candidate_chunks(
        limit=cap,
        hotness_max=RELEVANCE_FORGET_HOTNESS_MAX,
        importance_max=RELEVANCE_FORGET_IMPORTANCE_MAX,
        min_age_days=RELEVANCE_FORGET_MIN_AGE_DAYS,
        namespace=namespace,
    )

    results: list[ForgetProposal] = []
    effective_dry = RELEVANCE_FORGET_DRY_RUN if dry_run is None else dry_run
    for proposal in candidates:
        try:
            applied = await apply_relevance_forget(proposal, dry_run=effective_dry)
            results.append(applied)
        except Exception as e:
            logger.warning(
                "relevance_forget apply failed ns=%s id=%s: %s",
                proposal.namespace,
                proposal.memory_id,
                e,
            )

    applied_n = sum(1 for r in results if r.applied)
    logger.info(
        "relevance_forget.cycle proposed=%d applied=%d dry_run=%s",
        len(results),
        applied_n,
        effective_dry,
    )
    return {
        "enabled": True,
        "proposed": len(results),
        "applied": applied_n,
        "dry_run": effective_dry,
        "results": [r.to_dict() for r in results],
    }
