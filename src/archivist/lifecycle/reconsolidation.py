"""Hierarchical reconsolidation — Diff #6 product path (INIT-010/SPEC-002).

Groups eligible L2 ``memory_chunks`` by ``(namespace, agent_id)``, summarizes via
existing ``generate_tiers`` / compaction helpers, and either audits a proposal
(dry-run) or writes an L1 summary chunk scoped to the same agent+namespace.

Safe defaults (ADR-010 / GR-SAFE-001): ``RECONSOLIDATION_ENABLED=false``,
``RECONSOLIDATION_DRY_RUN=true``. No net-new core MCP tools.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from archivist.core.audit import log_memory_event
from archivist.core.config import (
    RECONSOLIDATION_DRY_RUN,
    RECONSOLIDATION_ENABLED,
    RECONSOLIDATION_MAX_CHUNKS_PER_GROUP,
    RECONSOLIDATION_MAX_GROUPS_PER_CYCLE,
    RECONSOLIDATION_MIN_CHUNKS,
)

logger = logging.getLogger("archivist.reconsolidation")


@dataclass
class ReconsolidationProposal:
    """One group reconsolidation outcome (proposed or applied)."""

    namespace: str
    agent_id: str
    source_qdrant_ids: list[str]
    summary_text: str = ""
    summary_qdrant_id: str = ""
    dry_run: bool = True
    applied: bool = False
    rule: str = "l2_group_to_l1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "namespace": self.namespace,
            "agent_id": self.agent_id,
            "source_qdrant_ids": list(self.source_qdrant_ids),
            "summary_text": self.summary_text,
            "summary_qdrant_id": self.summary_qdrant_id,
            "dry_run": self.dry_run,
            "applied": self.applied,
            "rule": self.rule,
            "metadata": dict(self.metadata),
        }


def _summary_id(namespace: str, agent_id: str, source_ids: list[str]) -> str:
    digest = hashlib.sha256(
        f"{namespace}|{agent_id}|{'|'.join(sorted(source_ids))}".encode()
    ).hexdigest()[:16]
    return f"reconsolidate:{namespace}:{agent_id}:{digest}"


async def _candidate_groups(
    *,
    min_chunks: int,
    max_groups: int,
    max_chunks_per_group: int,
) -> list[dict[str, Any]]:
    """Return groups of L2 chunks eligible for reconsolidation."""
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        # Find (namespace, agent_id) pairs with enough active L2 chunks.
        pairs = await conn.fetchall(
            """
            SELECT namespace, agent_id, COUNT(*) AS n
            FROM memory_chunks
            WHERE COALESCE(tier_label, 'l2') = 'l2'
              AND COALESCE(is_suppressed, 0) = 0
              AND COALESCE(is_excluded, 0) = 0
              AND qdrant_id NOT LIKE 'reconsolidate:%'
            GROUP BY namespace, agent_id
            HAVING COUNT(*) >= ?
            ORDER BY n DESC
            LIMIT ?
            """,
            (min_chunks, max_groups),
        )

    groups: list[dict[str, Any]] = []
    async with pool.read() as conn:
        for row in pairs:
            ns = row["namespace"] if isinstance(row, dict) else row[0]
            aid = row["agent_id"] if isinstance(row, dict) else row[1]
            chunks = await conn.fetchall(
                """
                SELECT qdrant_id, text, agent_id, namespace, date, memory_type
                FROM memory_chunks
                WHERE namespace = ? AND agent_id = ?
                  AND COALESCE(tier_label, 'l2') = 'l2'
                  AND COALESCE(is_suppressed, 0) = 0
                  AND COALESCE(is_excluded, 0) = 0
                  AND qdrant_id NOT LIKE 'reconsolidate:%'
                ORDER BY COALESCE(created_at, date, '') ASC, qdrant_id ASC
                LIMIT ?
                """,
                (ns, aid, max_chunks_per_group),
            )
            if len(chunks) < min_chunks:
                continue
            groups.append(
                {
                    "namespace": ns or "",
                    "agent_id": aid or "",
                    "chunks": [dict(c) for c in chunks],
                }
            )
    return groups


async def _summarize_group(chunks: list[dict[str, Any]]) -> str:
    """Build L1 overview text via existing tiering helper (or flat compact fallback)."""
    from archivist.write.tiering import generate_tiers

    parts: list[str] = []
    for c in chunks:
        text = (c.get("text") or "").strip()
        if text:
            parts.append(text[:1200])
    combined = "\n\n---\n\n".join(parts)
    if not combined.strip():
        return ""
    tiers = await generate_tiers(combined[:8000])
    summary = (tiers.get("l1") or tiers.get("l0") or combined[:500]).strip()
    return summary


async def apply_reconsolidation(
    proposal: ReconsolidationProposal,
    *,
    dry_run: bool | None = None,
) -> ReconsolidationProposal:
    """Audit and optionally write an L1 summary chunk for a proposal."""
    from archivist.storage.graph_fts import upsert_fts_chunk

    effective_dry = RECONSOLIDATION_DRY_RUN if dry_run is None else dry_run
    proposal.dry_run = effective_dry
    if not proposal.summary_qdrant_id:
        proposal.summary_qdrant_id = _summary_id(
            proposal.namespace, proposal.agent_id, proposal.source_qdrant_ids
        )

    audit_meta = {
        "summary_qdrant_id": proposal.summary_qdrant_id,
        "source_qdrant_ids": proposal.source_qdrant_ids,
        "rule": proposal.rule,
        "dry_run": effective_dry,
        "summary_chars": len(proposal.summary_text or ""),
    }

    if effective_dry:
        await log_memory_event(
            agent_id=proposal.agent_id or "system:reconsolidation",
            action="reconsolidation_proposed",
            memory_id=proposal.summary_qdrant_id,
            namespace=proposal.namespace,
            text_hash=proposal.summary_qdrant_id,
            metadata=audit_meta,
        )
        proposal.applied = False
        logger.info(
            "reconsolidation dry_run=1 ns=%s agent=%s sources=%d",
            proposal.namespace,
            proposal.agent_id,
            len(proposal.source_qdrant_ids),
        )
        return proposal

    if not (proposal.summary_text or "").strip():
        raise ValueError("reconsolidation apply requires non-empty summary_text")

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    await upsert_fts_chunk(
        qdrant_id=proposal.summary_qdrant_id,
        text=proposal.summary_text,
        file_path=f"reconsolidate:{proposal.summary_qdrant_id}",
        chunk_index=0,
        agent_id=proposal.agent_id,
        namespace=proposal.namespace,
        date=today,
        memory_type="reconsolidation",
        actor_id="system:reconsolidation",
        actor_type="system",
        importance=0.6,
        tier_label="l1",
    )
    await log_memory_event(
        agent_id=proposal.agent_id or "system:reconsolidation",
        action="reconsolidation_applied",
        memory_id=proposal.summary_qdrant_id,
        namespace=proposal.namespace,
        text_hash=proposal.summary_qdrant_id,
        metadata=audit_meta,
    )
    proposal.applied = True
    logger.info(
        "reconsolidation applied=1 ns=%s agent=%s id=%s",
        proposal.namespace,
        proposal.agent_id,
        proposal.summary_qdrant_id,
    )
    return proposal


async def reconsolidation_cycle(
    *,
    dry_run: bool | None = None,
    max_groups: int | None = None,
) -> dict[str, Any]:
    """Curator hook: reconsolidate up to N L2 groups. No-op when master flag is off."""
    if not RECONSOLIDATION_ENABLED:
        return {
            "enabled": False,
            "proposed": 0,
            "applied": 0,
            "dry_run": True,
            "results": [],
        }

    cap = RECONSOLIDATION_MAX_GROUPS_PER_CYCLE if max_groups is None else max_groups
    effective_dry = RECONSOLIDATION_DRY_RUN if dry_run is None else dry_run
    groups = await _candidate_groups(
        min_chunks=RECONSOLIDATION_MIN_CHUNKS,
        max_groups=max(cap, 1),
        max_chunks_per_group=RECONSOLIDATION_MAX_CHUNKS_PER_GROUP,
    )

    results: list[ReconsolidationProposal] = []
    for group in groups[:cap]:
        chunks = group["chunks"]
        source_ids = [str(c.get("qdrant_id") or "") for c in chunks if c.get("qdrant_id")]
        try:
            summary = await _summarize_group(chunks)
        except Exception as exc:
            logger.warning(
                "reconsolidation summarize failed ns=%s agent=%s: %s",
                group["namespace"],
                group["agent_id"],
                exc,
            )
            continue
        if not summary:
            continue
        proposal = ReconsolidationProposal(
            namespace=group["namespace"],
            agent_id=group["agent_id"],
            source_qdrant_ids=source_ids,
            summary_text=summary,
            dry_run=effective_dry,
        )
        try:
            applied = await apply_reconsolidation(proposal, dry_run=effective_dry)
            results.append(applied)
        except Exception as exc:
            logger.warning(
                "reconsolidation apply failed ns=%s agent=%s: %s",
                group["namespace"],
                group["agent_id"],
                exc,
            )

    applied_n = sum(1 for r in results if r.applied)
    logger.info(
        "reconsolidation.cycle proposed=%d applied=%d dry_run=%s",
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
