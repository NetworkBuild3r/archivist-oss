"""INIT-006/SPEC-002 — test-only agentic memory eval harness (ADR-006).

MemoryArena-*inspired* CI helpers: store-flag config, synthetic namespace map,
RBAC patches for search (mocks do not change production defaults), fake-embed
store patches, and a discrete action oracle.

GR-LAYER-001: action selection lives here under ``tests/`` only — not a
production agent runtime / MCP tool.
"""

from __future__ import annotations

import json
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# Discrete action ids for the test-only oracle (ADR-006).
ACTION_REFUSE = "refuse"
ACTION_ORDER_EXPRESS = "order_express"
ACTION_NEEDS_CLARIFICATION = "needs_clarification"

# Cue substring that maps to ACTION_ORDER_EXPRESS when present in memory text.
EXPRESS_ELIGIBLE_CUE = "EXPRESS_ELIGIBLE"
# Contradictory cue — when both eligible + ineligible appear in active evidence,
# oracle must not invent a single resolution (ADR-006 / SPEC-004).
EXPRESS_INELIGIBLE_CUE = "EXPRESS_INELIGIBLE"

# INIT-007/SPEC-004 — tip/procedure cue (must appear in get_context tips[], not
# only in memories / index TOC, when require_tip_evidence=True).
PROCEDURE_EXPRESS_CUE = "PROCEDURE_EXPRESS"

# Synthetic agent → namespace map (fail-closed: unknown agent_id → "").
_AGENT_NAMESPACE: dict[str, str] = {
    "agentic-agent-a": "agentic-ns-a",
    "agentic-agent-b": "agentic-ns-b",
    "agentic-agent": "agentic-ns",
}


@dataclass(frozen=True)
class AgenticSession:
    """One logical session identity for multi-session scenarios (SPEC-003+)."""

    agent_id: str
    namespace: str

    @classmethod
    def from_agent(cls, agent_id: str) -> AgenticSession:
        ns = namespace_for_agent(agent_id)
        if not ns:
            raise ValueError(f"unknown synthetic agent_id={agent_id!r} (fail-closed)")
        return cls(agent_id=agent_id, namespace=ns)


def namespace_for_agent(agent_id: str) -> str:
    """Map synthetic agent → namespace; unknown ids return empty (fail-closed)."""
    return _AGENT_NAMESPACE.get(agent_id, "")


def configure_agentic_store_flags(monkeypatch) -> None:
    """Disable optional pre-ack enrichment for deterministic CI stores."""
    monkeypatch.setattr("archivist.core.config.OUTBOX_ENABLED", True)
    monkeypatch.setattr("archivist.core.config.CONFLICT_CHECK_ON_STORE", False)
    monkeypatch.setattr("archivist.app.handlers.tools_storage.CONFLICT_CHECK_ON_STORE", False)
    monkeypatch.setattr("archivist.core.config.REVERSE_HYDE_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.BM25_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.SYNTHETIC_QUESTIONS_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.CONTEXTUAL_AUGMENTATION_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.TOPIC_ROUTING_ENABLED", False)
    monkeypatch.setattr("archivist.core.config.DEDUP_LLM_ENABLED", False)
    # ADR-005 default — keep defer off unless a scenario opts in.
    monkeypatch.setattr("archivist.core.config.ARCHIVIST_EMBED_DEFER", False)


@contextmanager
def allow_search_rbac():
    """Patch search-path RBAC for synthetic agentic agents only.

    Production RBAC modules are untouched; patches are process-local to the
    ``with`` block (same pattern as coach_core evals).
    """
    with ExitStack() as stack:
        stack.enter_context(
            patch("archivist.app.handlers.tools_search.require_rbac", return_value=None)
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_search.filter_agents_for_read",
                side_effect=lambda caller, ids: (list(ids), []),
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_search.get_namespace_for_agent",
                side_effect=namespace_for_agent,
            )
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_search.is_permissive_mode", return_value=True)
        )
        yield


@contextmanager
def store_patches(*, embed_dim: int = 1024, namespace: str | None = None):
    """Fake embed + stub Qdrant + store-path RBAC mocks (GR-EVAL-002).

    Patches are local to the ``with`` block — production RBAC defaults unchanged.
    """
    fake_vec = [0.11] * embed_dim
    mock_client = MagicMock()
    mock_client.upsert = MagicMock()
    mock_client.query_points = MagicMock(return_value=MagicMock(points=[]))
    ns = namespace or "agentic-ns"
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.embed_text",
                new_callable=AsyncMock,
                return_value=fake_vec,
            )
        )
        stack.enter_context(
            patch(
                "archivist.write.conflict_detection.embed_text",
                new_callable=AsyncMock,
                return_value=fake_vec,
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.embed_batch",
                new_callable=AsyncMock,
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.llm_adjudicated_dedup",
                new_callable=AsyncMock,
                return_value=None,
            )
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_storage.qdrant_client", return_value=mock_client)
        )
        stack.enter_context(
            patch("archivist.write.conflict_detection.qdrant_client", return_value=mock_client)
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.ensure_collection",
                return_value="test_col",
            )
        )
        stack.enter_context(patch("archivist.core.audit.log_memory_event", new_callable=AsyncMock))
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage._extract_needle_micro_chunks",
                return_value=[],
            )
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.get_namespace_for_agent",
                return_value=ns,
            )
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_storage.get_namespace_config", return_value=None)
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_storage.require_rbac", return_value=None)
        )
        stack.enter_context(
            patch("archivist.app.handlers.tools_storage.pre_extract", return_value={})
        )
        stack.enter_context(
            patch(
                "archivist.app.handlers.tools_storage.extract_needle_entities",
                return_value=[],
            )
        )
        yield mock_client


def _active_memories(memories: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    """Drop suppressed / empty-text rows (stale encoding via is_suppressed)."""
    if not memories:
        return []
    active: list[dict[str, Any]] = []
    for m in memories:
        if not isinstance(m, dict):
            continue
        if m.get("is_suppressed"):
            continue
        text = m.get("text")
        if isinstance(text, str) and text.strip():
            active.append(m)
    return active


def memories_text_blob(memories: list[dict[str, Any]] | None) -> str:
    """Concatenate **active** memory texts for oracle matching."""
    parts = [
        m["text"]
        for m in _active_memories(memories)
        if isinstance(m.get("text"), str) and str(m.get("text")).strip()
    ]
    return "\n".join(parts)


def choose_action(
    memories: list[dict[str, Any]] | None = None,
    *,
    context: dict[str, Any] | None = None,
    tips: list[str] | None = None,
    require_tip_evidence: bool = False,
) -> str:
    """Test-only action oracle: map retrieved memories/tips → discrete action id.

    Rules (SPEC-002…004 / INIT-007/SPEC-004):
    - empty / missing **active** evidence → ``refuse`` (empty-OK)
    - suppressed rows (``is_suppressed``) are ignored (stale / superseded)
    - both ``EXPRESS_ELIGIBLE`` and ``EXPRESS_INELIGIBLE`` in active text →
      ``needs_clarification`` (do not invent a merge)
    - else any active ``EXPRESS_ELIGIBLE`` → ``order_express``
    - otherwise → ``needs_clarification``

    When ``require_tip_evidence=True`` (procedure→action scenarios):
    - ``PROCEDURE_EXPRESS_CUE`` must appear in **tips** from get_context.
    - Memories / index TOC alone must **not** unlock ``order_express``.
    - Missing tip cue → ``refuse``.

    ``context`` may supply ``memories`` / ``tips`` when the get_context pack is
    passed whole. Index TOC / markdown alone must never be passed as ``memories``.
    """
    mems = memories
    tip_list = tips
    if isinstance(context, dict):
        if mems is None:
            raw = context.get("memories")
            mems = raw if isinstance(raw, list) else None
        if tip_list is None:
            raw_tips = context.get("tips")
            tip_list = raw_tips if isinstance(raw_tips, list) else None

    tip_blob = "\n".join(t for t in (tip_list or []) if isinstance(t, str) and t.strip())

    if require_tip_evidence:
        # Procedure path: tips are the only admissible evidence for express.
        if PROCEDURE_EXPRESS_CUE not in tip_blob:
            return ACTION_REFUSE
        return ACTION_ORDER_EXPRESS

    blob = memories_text_blob(mems)
    if not blob.strip():
        return ACTION_REFUSE
    has_eligible = EXPRESS_ELIGIBLE_CUE in blob
    has_ineligible = EXPRESS_INELIGIBLE_CUE in blob
    if has_eligible and has_ineligible:
        return ACTION_NEEDS_CLARIFICATION
    if has_eligible:
        return ACTION_ORDER_EXPRESS
    return ACTION_NEEDS_CLARIFICATION


async def seed_tip(
    *,
    session: AgenticSession,
    tip_text: str,
    category: str = "strategy",
    context: str = "",
    archived: int = 0,
) -> str:
    """Insert a tips-table row for procedure evals (INIT-007/SPEC-004).

    Direct SQLite seed — core profile does not expose log_trajectory (GR-PROD-002).
    Uses columns present on both fixture ``qa_pool`` schema and trajectory guard.
    Returns tip id.
    """
    import uuid
    from datetime import UTC, datetime

    from archivist.core.trajectory import _ensure_trajectory_schema
    from archivist.storage.sqlite_pool import pool

    _ensure_trajectory_schema()
    tip_id = str(uuid.uuid4())
    traj_id = str(uuid.uuid4())
    now = datetime.now(UTC).isoformat()
    async with pool.write() as conn:
        # Fixture qa_pool trajectories is a slim schema (no task_fingerprint).
        await conn.execute(
            """INSERT INTO trajectories
               (id, agent_id, session_id, task_description, outcome, created_at)
               VALUES (?,?,?,?,?,?)""",
            (
                traj_id,
                session.agent_id,
                "agentic-proc",
                "procedure eval seed",
                "success",
                now,
            ),
        )
        await conn.execute(
            """INSERT INTO tips
               (id, trajectory_id, agent_id, category, tip_text, context,
                archived, created_at, usage_count)
               VALUES (?,?,?,?,?,?,?,?,0)""",
            (
                tip_id,
                traj_id,
                session.agent_id,
                category,
                tip_text,
                context,
                int(archived),
                now,
            ),
        )
    return tip_id


async def store_memory(
    *,
    text: str,
    session: AgenticSession,
    entities: list[str] | None = None,
    provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Store via ``archivist_store`` under fake embed / stub Qdrant patches.

    Session identity always wins over provenance keys (SEC Low-001).
    """
    from archivist.app.handlers._registry import dispatch_tool

    args: dict[str, Any] = {
        **(provenance or {}),
        "text": text,
        "entities": entities or ["AgenticEntity"],
        # Force session identity last — provenance must not override tenant.
        "agent_id": session.agent_id,
        "namespace": session.namespace,
    }
    with store_patches(namespace=session.namespace):
        result = await dispatch_tool("archivist_store", args)
    return json.loads(result[0].text)


async def load_chunk(pool, memory_id: str) -> dict[str, Any] | None:
    """Load a ``memory_chunks`` row by id (Session B evidence source)."""
    async with pool.read() as conn:
        cur = await conn.execute(
            "SELECT qdrant_id, text, namespace, source, subject, purpose, "
            "sensitivity, statement_kind, confidence, agent_id, "
            "is_suppressed, supersedes_id FROM memory_chunks WHERE qdrant_id = ?",
            (memory_id,),
        )
        row = await cur.fetchone()
    return dict(row) if row is not None else None


async def chunks_for_namespace(pool, namespace: str) -> list[dict[str, Any]]:
    """List chunk rows for a namespace (isolation / omit-store controls)."""
    async with pool.read() as conn:
        cur = await conn.execute(
            "SELECT qdrant_id, text, namespace, source, subject, purpose, "
            "sensitivity, statement_kind, confidence, agent_id, "
            "is_suppressed, supersedes_id FROM memory_chunks WHERE namespace = ?",
            (namespace,),
        )
        rows = await cur.fetchall()
    return [dict(r) for r in rows]


def row_to_hit(row: dict[str, Any], *, score: float = 0.91) -> dict[str, Any]:
    """Normalize a SQLite chunk row into a retrieval hit dict."""
    mid = row.get("qdrant_id") or row.get("id") or ""
    return {
        "id": mid,
        "text": row.get("text") or "",
        "score": score,
        "namespace": row.get("namespace") or "",
        "source": row.get("source") or "",
        "subject": row.get("subject") or "",
        "purpose": row.get("purpose") or "",
        "sensitivity": row.get("sensitivity") or "standard",
        "statement_kind": row.get("statement_kind") or "",
        "confidence": row.get("confidence"),
        "agent_id": row.get("agent_id") or "",
        "is_suppressed": bool(row.get("is_suppressed")),
        "supersedes_id": row.get("supersedes_id") or "",
        "date": row.get("date") or "",
    }


@contextmanager
def allow_context_rbac():
    """Patch get_context RBAC for synthetic agentic sessions only."""
    with patch("archivist.app.handlers.tools_context.require_rbac", return_value=None):
        yield


async def get_context_with_hits(
    *,
    session: AgenticSession,
    hits: list[dict[str, Any]],
    task_description: str = "decide return action",
    include_tips: bool = False,
) -> dict[str, Any]:
    """Session B: ``archivist_get_context`` over provenance-bearing hits (SQLite CI).

    Patches ``get_relevant_context`` so CI needs no live Qdrant (GR-EVAL-002).
    Applies ``apply_prerank_filters`` for the session namespace (SEC Low-002).

    INIT-007/SPEC-004: when ``include_tips=True``, load tips via real
    ``search_tips`` + ``tip_rows_to_strings`` (SPEC-002/003 path) into the
    mocked context pack — not invented oracle strings.
    """
    from archivist.app.handlers._registry import dispatch_tool
    from archivist.core.trajectory import search_tips, tip_rows_to_strings
    from archivist.retrieval.context_api import ContextChunk, RelevantContext
    from archivist.retrieval.retrieval_filters import (
        apply_prerank_filters,
        build_stable_memories,
    )

    filtered = apply_prerank_filters(hits, namespace=session.namespace)
    memories = build_stable_memories(filtered)
    sources = [
        ContextChunk(
            memory_id=str(h.get("id") or ""),
            text=str(h.get("text") or ""),
            score=float(h.get("score") or 0.0),
            tier="l2",
            file_path="",
            date=str(h.get("date") or ""),
            agent_id=str(h.get("agent_id") or session.agent_id),
            namespace=str(h.get("namespace") or session.namespace),
            subject=str(h.get("subject") or ""),
            purpose=str(h.get("purpose") or ""),
            sensitivity=str(h.get("sensitivity") or "standard"),
            source=str(h.get("source") or ""),
            confidence=h.get("confidence"),
            statement_kind=str(h.get("statement_kind") or ""),
        )
        for h in filtered
    ]
    tips: list[str] = []
    if include_tips:
        tip_rows = await search_tips(
            agent_id=session.agent_id,
            limit=5,
            query=task_description or "",
            record_usage=False,
        )
        tips = tip_rows_to_strings(tip_rows)
    ctx = RelevantContext(
        answer="",
        sources=sources,
        graph_facts=[],
        tips=tips,
        total_tokens=max(1, sum(len(str(h.get("text") or "")) // 4 for h in filtered)),
        budget_tokens=8000,
        over_budget=False,
        tier_distribution={"l2": len(sources)},
        token_savings_pct=0.0,
        provenance=[str(h.get("id") or "") for h in filtered if h.get("id")],
        pack_policy="adaptive",
        memories=memories,
    )
    with (
        patch(
            "archivist.retrieval.context_api.get_relevant_context",
            new=AsyncMock(return_value=ctx),
        ),
        allow_context_rbac(),
    ):
        out = await dispatch_tool(
            "archivist_get_context",
            {
                "agent_id": session.agent_id,
                "task_description": task_description,
                "namespace": session.namespace,
                "subject": "returns",
                "purpose": "agentic_eval",
                "include_tips": include_tips,
            },
        )
    return json.loads(out[0].text)
