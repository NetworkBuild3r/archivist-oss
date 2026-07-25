"""Phase 8 contradiction resolution — propose/apply supersede, merge, keep-both.

INIT-001/SPEC-006: Hybrid resolver. Deterministic rules run first; optional LLM
adjudication is behind ``CONTRADICTION_RESOLVE_LLM_ENABLED`` (default off).
Master switch ``CONTRADICTION_RESOLVE_ENABLED`` defaults off; when enabled,
``CONTRADICTION_RESOLVE_DRY_RUN`` defaults true so mutations stay opt-in.
Resolutions never hard-delete facts — only supersede — and always audit.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from archivist.core.audit import log_memory_event
from archivist.core.config import (
    CONTRADICTION_RESOLVE_DRY_RUN,
    CONTRADICTION_RESOLVE_ENABLED,
    CONTRADICTION_RESOLVE_LLM_ENABLED,
    CONTRADICTION_RESOLVE_MAX_PER_CYCLE,
    CURATOR_LLM_API_KEY,
    CURATOR_LLM_MODEL,
    CURATOR_LLM_URL,
    LLM_MODEL,
    LLM_URL,
)

logger = logging.getLogger("archivist.lifecycle.resolve")

ResolutionAction = Literal["supersede", "merge", "keep_both"]

_OPPOSING = [
    ("enabled", "disabled"),
    ("active", "inactive"),
    ("yes", "no"),
    ("true", "false"),
    ("success", "failure"),
    ("up", "down"),
    ("running", "stopped"),
    ("allow", "deny"),
    ("open", "closed"),
]

_SECRET_RE = re.compile(
    r"(?i)\b(api[_-]?key|token|password|secret|bearer|authorization)\b\s*[:=]\s*\S+"
)

_LLM_SYSTEM = (
    "You are a knowledge-graph conflict adjudicator. Given two conflicting facts "
    "about the same entity, choose one action:\n"
    '- "supersede": one fact replaces the other (name winner_fact_id).\n'
    '- "merge": combine into one richer fact (provide merge_text).\n'
    '- "keep_both": both remain valid (different perspectives or times).\n'
    "Return ONLY JSON: "
    '{"action":"supersede|merge|keep_both","winner_fact_id":null|int,'
    '"loser_fact_id":null|int,"merge_text":"","reason":"..."}'
)


@dataclass
class ConflictPair:
    """Two namespace-scoped facts that appear to contradict each other."""

    entity_id: int
    namespace: str
    fact_a: dict[str, Any]
    fact_b: dict[str, Any]
    trigger: str


@dataclass
class ResolutionProposal:
    """Documented resolution decision for a conflict pair."""

    action: ResolutionAction
    entity_id: int
    namespace: str
    fact_a_id: int
    fact_b_id: int
    winner_fact_id: int | None
    loser_fact_id: int | None
    merge_text: str | None
    reason: str
    rule: str
    trigger: str
    dry_run: bool = True
    applied: bool = False
    resolution_id: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def redact_sensitive(text: str) -> str:
    """Strip high-sensitivity credential markers from text before LLM prompts."""
    return _SECRET_RE.sub(r"\1=[REDACTED]", text or "")


def _word_set(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9_]+", (text or "").lower()) if len(w) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _parse_ts(value: str) -> datetime:
    if not value:
        return datetime.min.replace(tzinfo=UTC)
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return dt
    except ValueError:
        return datetime.min.replace(tzinfo=UTC)


def _resolution_id(entity_id: int, fact_a_id: int, fact_b_id: int, action: str) -> str:
    """Stable id for idempotent apply — same pair+action yields same key."""
    a, b = sorted((int(fact_a_id), int(fact_b_id)))
    raw = f"{entity_id}:{a}:{b}:{action}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def find_opposing_trigger(text_a: str, text_b: str) -> str | None:
    """Return opposing-keyword trigger if present, else None."""
    a_lower = (text_a or "").lower()
    b_lower = (text_b or "").lower()
    for pos, neg in _OPPOSING:
        if (pos in a_lower and neg in b_lower) or (neg in a_lower and pos in b_lower):
            return f"{pos}/{neg}"
    return None


def detect_conflict_pairs(
    facts: list[dict[str, Any]], *, namespace: str = ""
) -> list[ConflictPair]:
    """Find contradicting fact pairs within an entity's fact list.

    Namespace scoping: when ``namespace`` is set, only facts in that namespace
    are considered; pairs whose namespaces disagree are skipped.
    """
    scoped = facts
    if namespace:
        scoped = [f for f in facts if (f.get("namespace") or "global") == namespace]

    pairs: list[ConflictPair] = []
    for i, a in enumerate(scoped):
        for b in scoped[i + 1 :]:
            ns_a = a.get("namespace") or "global"
            ns_b = b.get("namespace") or "global"
            if ns_a != ns_b:
                continue
            if a.get("agent_id") == b.get("agent_id"):
                continue
            if a.get("superseded_by") is not None or b.get("superseded_by") is not None:
                continue
            trigger = find_opposing_trigger(a.get("fact_text", ""), b.get("fact_text", ""))
            if not trigger:
                # Near-duplicate cross-agent facts also count as conflicts.
                if (
                    _jaccard(_word_set(a.get("fact_text", "")), _word_set(b.get("fact_text", "")))
                    < 0.7
                ):
                    continue
                trigger = "near_duplicate"
            pairs.append(
                ConflictPair(
                    entity_id=int(a.get("entity_id") or b.get("entity_id") or 0),
                    namespace=ns_a,
                    fact_a=a,
                    fact_b=b,
                    trigger=trigger,
                )
            )
    return pairs


def propose_resolution_rules(pair: ConflictPair) -> ResolutionProposal:
    """Deterministic supersede / merge / keep_both rules (no LLM)."""
    a, b = pair.fact_a, pair.fact_b
    a_id, b_id = int(a["id"]), int(b["id"])
    a_ts, b_ts = _parse_ts(a.get("created_at", "")), _parse_ts(b.get("created_at", ""))
    newer, older = (a, b) if a_ts >= b_ts else (b, a)
    newer_id, older_id = int(newer["id"]), int(older["id"])

    overlap = _jaccard(_word_set(a.get("fact_text", "")), _word_set(b.get("fact_text", "")))

    if pair.trigger == "near_duplicate" or overlap >= 0.85:
        merge_text = _merge_texts(a.get("fact_text", ""), b.get("fact_text", ""))
        return ResolutionProposal(
            action="merge",
            entity_id=pair.entity_id,
            namespace=pair.namespace,
            fact_a_id=a_id,
            fact_b_id=b_id,
            winner_fact_id=None,
            loser_fact_id=None,
            merge_text=merge_text,
            reason="Near-duplicate cross-agent facts; merge into single canonical fact.",
            rule="near_duplicate_merge",
            trigger=pair.trigger,
            resolution_id=_resolution_id(pair.entity_id, a_id, b_id, "merge"),
        )

    if pair.trigger and "/" in pair.trigger and a_ts != b_ts:
        return ResolutionProposal(
            action="supersede",
            entity_id=pair.entity_id,
            namespace=pair.namespace,
            fact_a_id=a_id,
            fact_b_id=b_id,
            winner_fact_id=newer_id,
            loser_fact_id=older_id,
            merge_text=None,
            reason=(
                f"Opposing keywords ({pair.trigger}); newer fact "
                f"({newer_id}) supersedes older ({older_id})."
            ),
            rule="temporal_supersede",
            trigger=pair.trigger,
            resolution_id=_resolution_id(pair.entity_id, a_id, b_id, "supersede"),
        )

    # Same-time opposing claims or ambiguous — keep both with audit trail.
    return ResolutionProposal(
        action="keep_both",
        entity_id=pair.entity_id,
        namespace=pair.namespace,
        fact_a_id=a_id,
        fact_b_id=b_id,
        winner_fact_id=None,
        loser_fact_id=None,
        merge_text=None,
        reason="Ambiguous or contemporaneous conflict; retain both perspectives.",
        rule="ambiguous_keep_both",
        trigger=pair.trigger,
        resolution_id=_resolution_id(pair.entity_id, a_id, b_id, "keep_both"),
    )


def _merge_texts(text_a: str, text_b: str) -> str:
    """Combine two fact texts without naive duplication."""
    a, b = (text_a or "").strip(), (text_b or "").strip()
    if not a:
        return b
    if not b or a.lower() == b.lower():
        return a
    if a.lower() in b.lower():
        return b
    if b.lower() in a.lower():
        return a
    return f"{a} | {b}"


def _coerce_pair_fact_ids(
    winner: Any, loser: Any, fact_a_id: int, fact_b_id: int
) -> tuple[int | None, int | None]:
    """Keep only winner/loser ids that name the two facts of this pair.

    Fact text is attacker-influenced content, so an LLM verdict must never be
    able to point supersede at a row outside the pair it was asked about.
    """
    allowed = {fact_a_id, fact_b_id}

    def _pick(value: Any) -> int | None:
        try:
            candidate = int(value)
        except (TypeError, ValueError):
            return None
        return candidate if candidate in allowed else None

    w, lo = _pick(winner), _pick(loser)
    if w is not None and w == lo:
        return None, None
    return w, lo


async def propose_resolution_llm(pair: ConflictPair) -> ResolutionProposal | None:
    """Optional LLM adjudication. Returns None on disable/failure (falls back to rules)."""
    if not CONTRADICTION_RESOLVE_LLM_ENABLED:
        return None

    from archivist.features.llm import llm_query

    prompt = (
        f"Entity id: {pair.entity_id}\nNamespace: {pair.namespace}\n"
        f"Trigger: {pair.trigger}\n\n"
        f"Fact A (id={pair.fact_a.get('id')}, agent={pair.fact_a.get('agent_id')}, "
        f"date={pair.fact_a.get('created_at')}):\n"
        f"{redact_sensitive(pair.fact_a.get('fact_text', '')[:500])}\n\n"
        f"Fact B (id={pair.fact_b.get('id')}, agent={pair.fact_b.get('agent_id')}, "
        f"date={pair.fact_b.get('created_at')}):\n"
        f"{redact_sensitive(pair.fact_b.get('fact_text', '')[:500])}\n"
    )
    try:
        raw = await llm_query(
            prompt,
            system=_LLM_SYSTEM,
            max_tokens=512,
            json_mode=True,
            model=CURATOR_LLM_MODEL or LLM_MODEL,
            url=CURATOR_LLM_URL or LLM_URL,
            api_key=CURATOR_LLM_API_KEY,
            stage="contradiction_resolve",
        )
        data = json.loads(raw.strip().strip("`").strip())
        if not isinstance(data, dict):
            return None
        action = data.get("action", "keep_both")
        if action not in ("supersede", "merge", "keep_both"):
            action = "keep_both"
        a_id, b_id = int(pair.fact_a["id"]), int(pair.fact_b["id"])
        winner, loser = _coerce_pair_fact_ids(
            data.get("winner_fact_id"), data.get("loser_fact_id"), a_id, b_id
        )
        if action == "supersede" and (winner is None or loser is None):
            # The model named a fact outside this pair — untrusted fact text can
            # steer adjudication, so fall back to deterministic rules rather than
            # superseding an arbitrary row (INIT-001/SPEC-012, SEC-012-03).
            logger.warning(
                "contradiction.resolve llm_supersede_rejected entity=%s pair=(%s,%s)",
                pair.entity_id,
                a_id,
                b_id,
            )
            return None
        return ResolutionProposal(
            action=action,  # type: ignore[arg-type]
            entity_id=pair.entity_id,
            namespace=pair.namespace,
            fact_a_id=a_id,
            fact_b_id=b_id,
            winner_fact_id=winner,
            loser_fact_id=loser,
            merge_text=data.get("merge_text") or None,
            reason=str(data.get("reason") or "LLM adjudication"),
            rule="llm_adjudicate",
            trigger=pair.trigger,
            resolution_id=_resolution_id(pair.entity_id, a_id, b_id, action),
            metadata={"llm": True},
        )
    except Exception as e:
        logger.warning("LLM contradiction resolve failed (falling back to rules): %s", e)
        return None


async def propose_resolution(pair: ConflictPair) -> ResolutionProposal:
    """Rules first; optional LLM override when flag enabled."""
    llm_prop = await propose_resolution_llm(pair)
    if llm_prop is not None:
        return llm_prop
    return propose_resolution_rules(pair)


async def _already_applied(resolution_id: str) -> bool:
    """Idempotency: skip if an apply audit already exists for this resolution_id."""
    from archivist.storage.sqlite_pool import pool

    try:
        async with pool.read() as conn:
            row = await conn.fetchone(
                "SELECT id FROM audit_log WHERE action=? AND memory_id=? LIMIT 1",
                ("contradiction_resolved", resolution_id),
            )
        return row is not None
    except Exception:
        return False


async def apply_resolution(
    proposal: ResolutionProposal,
    *,
    dry_run: bool | None = None,
    agent_id: str = "system:contradiction_resolve",
) -> ResolutionProposal:
    """Apply a proposal (or dry-run). Always writes an audit entry for the decision."""
    from archivist.storage.graph import add_fact, supersede_fact

    effective_dry = CONTRADICTION_RESOLVE_DRY_RUN if dry_run is None else dry_run
    proposal.dry_run = effective_dry

    if not proposal.resolution_id:
        proposal.resolution_id = _resolution_id(
            proposal.entity_id, proposal.fact_a_id, proposal.fact_b_id, proposal.action
        )

    if await _already_applied(proposal.resolution_id):
        proposal.applied = False
        proposal.metadata["idempotent_skip"] = True
        logger.info(
            "contradiction.resolve skip=idempotent resolution_id=%s action=%s ns=%s",
            proposal.resolution_id,
            proposal.action,
            proposal.namespace,
        )
        return proposal

    audit_meta = {
        "resolution_id": proposal.resolution_id,
        "action": proposal.action,
        "entity_id": proposal.entity_id,
        "namespace": proposal.namespace,
        "fact_a_id": proposal.fact_a_id,
        "fact_b_id": proposal.fact_b_id,
        "winner_fact_id": proposal.winner_fact_id,
        "loser_fact_id": proposal.loser_fact_id,
        "merge_text": proposal.merge_text,
        "reason": proposal.reason,
        "rule": proposal.rule,
        "trigger": proposal.trigger,
        "dry_run": effective_dry,
    }

    if effective_dry:
        await log_memory_event(
            agent_id=agent_id,
            action="contradiction_resolve_proposed",
            memory_id=proposal.resolution_id,
            namespace=proposal.namespace,
            text_hash=proposal.resolution_id,
            metadata=audit_meta,
        )
        proposal.applied = False
        logger.info(
            "contradiction.resolve dry_run=1 action=%s entity=%s ns=%s rule=%s",
            proposal.action,
            proposal.entity_id,
            proposal.namespace,
            proposal.rule,
        )
        return proposal

    try:
        if proposal.action == "supersede":
            winner, loser = _coerce_pair_fact_ids(
                proposal.winner_fact_id,
                proposal.loser_fact_id,
                proposal.fact_a_id,
                proposal.fact_b_id,
            )
            if winner is None or loser is None:
                raise ValueError(
                    "supersede requires winner_fact_id and loser_fact_id "
                    "drawn from this conflict pair"
                )
            await supersede_fact(int(loser), int(winner))
        elif proposal.action == "merge":
            merge_text = proposal.merge_text or ""
            if not merge_text:
                raise ValueError("merge requires merge_text")
            # Prefer metadata from the newer of the two source facts.
            newer_id = proposal.winner_fact_id or max(proposal.fact_a_id, proposal.fact_b_id)
            new_id = await add_fact(
                proposal.entity_id,
                merge_text,
                source_file=f"resolve:{proposal.resolution_id}",
                agent_id=agent_id,
                namespace=proposal.namespace,
                provenance="extracted",
            )
            await supersede_fact(proposal.fact_a_id, new_id)
            await supersede_fact(proposal.fact_b_id, new_id)
            proposal.winner_fact_id = new_id
            proposal.metadata["merged_from"] = [proposal.fact_a_id, proposal.fact_b_id, newer_id]
            audit_meta["winner_fact_id"] = new_id
        # keep_both: audit only — no graph mutation

        await log_memory_event(
            agent_id=agent_id,
            action="contradiction_resolved",
            memory_id=proposal.resolution_id,
            namespace=proposal.namespace,
            text_hash=proposal.resolution_id,
            metadata=audit_meta,
        )
        proposal.applied = True
        logger.info(
            "contradiction.resolve applied=1 action=%s entity=%s ns=%s rule=%s",
            proposal.action,
            proposal.entity_id,
            proposal.namespace,
            proposal.rule,
        )
    except Exception as e:
        logger.error(
            "contradiction.resolve failed action=%s entity=%s error=%s",
            proposal.action,
            proposal.entity_id,
            e,
        )
        await log_memory_event(
            agent_id=agent_id,
            action="contradiction_resolve_failed",
            memory_id=proposal.resolution_id,
            namespace=proposal.namespace,
            text_hash=proposal.resolution_id,
            metadata={**audit_meta, "error": str(e)},
        )
        raise

    return proposal


async def resolve_entity_contradictions(
    entity_id: int,
    *,
    namespace: str = "",
    dry_run: bool | None = None,
) -> list[ResolutionProposal]:
    """Detect + propose + apply resolutions for one entity (namespace-scoped)."""
    from archivist.storage.graph import get_entity_facts

    facts = await get_entity_facts(entity_id)
    # Attach entity_id onto fact dicts for pair builder when SELECT * omits join.
    for f in facts:
        f.setdefault("entity_id", entity_id)

    pairs = detect_conflict_pairs(facts, namespace=namespace)
    results: list[ResolutionProposal] = []
    for pair in pairs:
        if not pair.entity_id:
            pair.entity_id = entity_id
        proposal = await propose_resolution(pair)
        applied = await apply_resolution(proposal, dry_run=dry_run)
        results.append(applied)
    return results


async def _candidate_entity_ids(*, namespace: str = "", limit: int = 50) -> list[int]:
    """Entities with multi-agent active facts (cheap SQL prefilter)."""
    from archivist.storage.sqlite_pool import pool

    async with pool.read() as conn:
        if namespace:
            rows = await conn.fetchall(
                """
                SELECT entity_id
                FROM facts
                WHERE is_active=1 AND superseded_by IS NULL AND namespace=?
                GROUP BY entity_id
                HAVING COUNT(*) >= 2 AND COUNT(DISTINCT agent_id) >= 2
                LIMIT ?
                """,
                (namespace, limit),
            )
        else:
            rows = await conn.fetchall(
                """
                SELECT entity_id
                FROM facts
                WHERE is_active=1 AND superseded_by IS NULL
                GROUP BY entity_id
                HAVING COUNT(*) >= 2 AND COUNT(DISTINCT agent_id) >= 2
                LIMIT ?
                """,
                (limit,),
            )
    return [int(r["entity_id"] if isinstance(r, dict) else r[0]) for r in rows]


async def resolve_contradictions_cycle(
    *,
    namespace: str = "",
    max_resolutions: int | None = None,
    dry_run: bool | None = None,
) -> dict[str, Any]:
    """Curator hook: resolve up to N conflicts. No-op when master flag is off."""
    if not CONTRADICTION_RESOLVE_ENABLED:
        return {"enabled": False, "proposed": 0, "applied": 0, "dry_run": True, "results": []}

    cap = CONTRADICTION_RESOLVE_MAX_PER_CYCLE if max_resolutions is None else max_resolutions
    entity_ids = await _candidate_entity_ids(namespace=namespace, limit=max(cap * 2, 10))
    results: list[ResolutionProposal] = []
    for eid in entity_ids:
        if len(results) >= cap:
            break
        batch = await resolve_entity_contradictions(eid, namespace=namespace, dry_run=dry_run)
        results.extend(batch)
        if len(results) >= cap:
            results = results[:cap]
            break

    applied = sum(1 for r in results if r.applied)
    effective_dry = CONTRADICTION_RESOLVE_DRY_RUN if dry_run is None else dry_run
    logger.info(
        "contradiction.resolve.cycle proposed=%d applied=%d dry_run=%s ns=%s",
        len(results),
        applied,
        effective_dry,
        namespace or "*",
    )
    return {
        "enabled": True,
        "proposed": len(results),
        "applied": applied,
        "dry_run": effective_dry,
        "results": [r.to_dict() for r in results],
    }
