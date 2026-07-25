"""Phase 8 reflection hooks — structured tips/memories from trajectory outcomes.

INIT-001/SPEC-006: Deterministic reflection (no live LLM required) writes a
tip artifact keyed by trajectory. Master switch ``REFLECTION_ENABLED`` defaults
off; when enabled, ``REFLECTION_DRY_RUN`` defaults true. Namespace scoping is
preserved via tip/memory metadata and audit rows.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any

from archivist.core.audit import log_memory_event
from archivist.core.config import REFLECTION_DRY_RUN, REFLECTION_ENABLED, REFLECTION_MAX_PER_CYCLE

logger = logging.getLogger("archivist.lifecycle.reflection")


@dataclass
class ReflectionArtifact:
    """Structured reflection produced from a trajectory outcome."""

    trajectory_id: str
    agent_id: str
    namespace: str
    category: str
    tip_text: str
    context: str
    tip_id: str | None = None
    dry_run: bool = True
    applied: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_reflection_from_outcome(
    *,
    trajectory_id: str,
    agent_id: str,
    task_description: str,
    outcome: str,
    outcome_score: float | None,
    namespace: str = "global",
    actions: list | str | None = None,
) -> ReflectionArtifact:
    """Deterministic tip/memory text from a trajectory outcome (no LLM)."""
    outcome_norm = (outcome or "unknown").strip().lower()
    if outcome_norm in ("failure", "failed", "error"):
        category = "recovery"
        lesson = "Avoid repeating the failing approach; verify preconditions and rollback paths."
    elif outcome_norm in ("success", "ok", "completed"):
        category = "strategy"
        lesson = "Reuse the successful approach for similar tasks."
    else:
        category = "optimization"
        lesson = "Record what was attempted and clarify success criteria next time."

    score_part = f" (score={outcome_score})" if outcome_score is not None else ""
    tip_text = (
        f"Reflection on '{(task_description or 'task')[:200]}': "
        f"outcome={outcome_norm}{score_part}. {lesson}"
    )

    action_summary = ""
    if isinstance(actions, list) and actions:
        action_summary = f"Actions logged: {len(actions)}."
    elif isinstance(actions, str) and actions.strip():
        try:
            parsed = json.loads(actions)
            if isinstance(parsed, list):
                action_summary = f"Actions logged: {len(parsed)}."
        except json.JSONDecodeError:
            action_summary = "Actions present."

    context = " ".join(
        p
        for p in (
            f"trajectory_id={trajectory_id}",
            f"namespace={namespace}",
            action_summary,
        )
        if p
    )

    return ReflectionArtifact(
        trajectory_id=trajectory_id,
        agent_id=agent_id or "unknown",
        namespace=namespace or "global",
        category=category,
        tip_text=tip_text,
        context=context,
        metadata={"source": "deterministic_reflection", "outcome": outcome_norm},
    )


async def _existing_reflection_tip(trajectory_id: str) -> str | None:
    """Idempotency: one reflection tip per trajectory."""
    from archivist.core.trajectory import _ensure_trajectory_schema
    from archivist.storage.sqlite_pool import pool

    _ensure_trajectory_schema()
    async with pool.read() as conn:
        row = await conn.fetchone(
            "SELECT id FROM tips WHERE trajectory_id=? AND category=? LIMIT 1",
            (trajectory_id, "reflection"),
        )
    if row is None:
        return None
    return str(row["id"] if isinstance(row, dict) else row[0])


async def write_reflection(
    artifact: ReflectionArtifact,
    *,
    dry_run: bool | None = None,
) -> ReflectionArtifact:
    """Persist a reflection tip (or dry-run) with audit trail."""
    from archivist.core.trajectory import _ensure_trajectory_schema
    from archivist.storage.sqlite_pool import pool

    effective_dry = REFLECTION_DRY_RUN if dry_run is None else dry_run
    artifact.dry_run = effective_dry

    existing = await _existing_reflection_tip(artifact.trajectory_id)
    if existing:
        artifact.tip_id = existing
        artifact.applied = False
        artifact.metadata["idempotent_skip"] = True
        logger.info(
            "reflection.skip=idempotent trajectory_id=%s tip_id=%s",
            artifact.trajectory_id,
            existing,
        )
        return artifact

    # Store under category "reflection" while preserving strategy/recovery signal in metadata.
    tip_category = "reflection"
    artifact.metadata["lesson_category"] = artifact.category

    audit_meta = {
        "trajectory_id": artifact.trajectory_id,
        "category": tip_category,
        "lesson_category": artifact.category,
        "tip_text": artifact.tip_text[:500],
        "namespace": artifact.namespace,
        "dry_run": effective_dry,
    }

    if effective_dry:
        await log_memory_event(
            agent_id=artifact.agent_id,
            action="reflection_proposed",
            memory_id=f"reflect:{artifact.trajectory_id}",
            namespace=artifact.namespace,
            text_hash=artifact.trajectory_id,
            metadata=audit_meta,
        )
        artifact.applied = False
        logger.info(
            "reflection.dry_run=1 trajectory_id=%s ns=%s category=%s",
            artifact.trajectory_id,
            artifact.namespace,
            artifact.category,
        )
        return artifact

    _ensure_trajectory_schema()
    tip_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    async with pool.write() as conn:
        await conn.execute(
            """INSERT INTO tips (id, trajectory_id, agent_id, category, tip_text, context, created_at)
               VALUES (?,?,?,?,?,?,?)""",
            (
                tip_id,
                artifact.trajectory_id,
                artifact.agent_id,
                tip_category,
                artifact.tip_text,
                artifact.context,
                now,
            ),
        )

    artifact.tip_id = tip_id
    artifact.applied = True
    await log_memory_event(
        agent_id=artifact.agent_id,
        action="reflection_written",
        memory_id=tip_id,
        namespace=artifact.namespace,
        text_hash=artifact.trajectory_id,
        metadata={**audit_meta, "tip_id": tip_id},
    )
    logger.info(
        "reflection.applied=1 tip_id=%s trajectory_id=%s ns=%s",
        tip_id,
        artifact.trajectory_id,
        artifact.namespace,
    )
    return artifact


async def reflect_from_trajectory(
    trajectory_id: str,
    *,
    namespace: str = "global",
    dry_run: bool | None = None,
) -> ReflectionArtifact | None:
    """Load a trajectory and write a structured reflection artifact."""
    from archivist.core.trajectory import _ensure_trajectory_schema
    from archivist.storage.sqlite_pool import pool

    _ensure_trajectory_schema()
    async with pool.read() as conn:
        row = await conn.fetchone("SELECT * FROM trajectories WHERE id=?", (trajectory_id,))
    if not row:
        logger.warning("reflection.trajectory_missing id=%s", trajectory_id)
        return None

    traj = dict(row)
    meta = {}
    try:
        meta = json.loads(traj.get("metadata") or "{}")
    except json.JSONDecodeError:
        meta = {}
    ns = namespace or meta.get("namespace") or "global"

    artifact = build_reflection_from_outcome(
        trajectory_id=trajectory_id,
        agent_id=traj.get("agent_id", ""),
        task_description=traj.get("task_description", ""),
        outcome=traj.get("outcome", "unknown"),
        outcome_score=traj.get("outcome_score"),
        namespace=ns,
        actions=traj.get("actions"),
    )
    return await write_reflection(artifact, dry_run=dry_run)


async def reflection_cycle(
    *,
    namespace: str = "",
    max_reflections: int | None = None,
    dry_run: bool | None = None,
) -> dict[str, Any]:
    """Curator hook: reflect on recent trajectories without existing reflection tips."""
    if not REFLECTION_ENABLED:
        return {"enabled": False, "proposed": 0, "applied": 0, "dry_run": True, "results": []}

    from archivist.core.trajectory import _ensure_trajectory_schema
    from archivist.storage.sqlite_pool import pool

    _ensure_trajectory_schema()
    cap = REFLECTION_MAX_PER_CYCLE if max_reflections is None else max_reflections
    async with pool.read() as conn:
        rows = await conn.fetchall(
            """
            SELECT t.id FROM trajectories t
            WHERE NOT EXISTS (
                SELECT 1 FROM tips tip
                WHERE tip.trajectory_id = t.id AND tip.category = 'reflection'
            )
            ORDER BY t.created_at DESC
            LIMIT ?
            """,
            (cap,),
        )

    results: list[ReflectionArtifact] = []
    for row in rows:
        tid = str(row["id"] if isinstance(row, dict) else row[0])
        art = await reflect_from_trajectory(
            tid,
            namespace=namespace or "global",
            dry_run=dry_run,
        )
        if art is not None:
            results.append(art)

    applied = sum(1 for r in results if r.applied)
    effective_dry = REFLECTION_DRY_RUN if dry_run is None else dry_run
    logger.info(
        "reflection.cycle proposed=%d applied=%d dry_run=%s",
        len(results),
        applied,
        effective_dry,
    )
    return {
        "enabled": True,
        "proposed": len(results),
        "applied": applied,
        "dry_run": effective_dry,
        "results": [r.to_dict() for r in results],
    }
