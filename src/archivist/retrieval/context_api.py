"""High-level context assembly API for Archivist — Phase 4.

Provides ``get_relevant_context()`` as the canonical single-call entry point
for any agent that needs memory.  Instead of making three separate tool calls
(archivist_search + archivist_tips + archivist_context_check), agents call this
one function and receive a token-budgeted ``RelevantContext`` struct.

Also provides ``create_handoff_packet()`` for structured agent-to-agent transfer.

Public API
----------
    from archivist.retrieval.context_api import get_relevant_context, RelevantContext
    from archivist.retrieval.context_api import create_handoff_packet, HandoffPacket
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime

from archivist.core.trajectory import search_tips
from archivist.retrieval.context_packer import pack_context
from archivist.retrieval.rlm_retriever import recursive_retrieve
from archivist.retrieval.session_store import get_session_store
from archivist.utils.tokenizer import count_tokens

logger = logging.getLogger("archivist.context_api")


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ContextChunk:
    """A single packed memory chunk in a RelevantContext."""

    memory_id: str
    text: str
    score: float
    tier: str  # 'l0' | 'l1' | 'l2' | 'ephemeral'
    file_path: str
    date: str
    agent_id: str


@dataclass
class RelevantContext:
    """Token-budgeted assembled context for an agent query."""

    answer: str
    """LLM synthesis when available; empty string otherwise."""

    sources: list[ContextChunk]
    """Packed, token-budgeted memory chunks."""

    graph_facts: list[str]
    """Entity facts from the knowledge graph relevant to the query."""

    tips: list[str]
    """Procedural tips for the query's agent."""

    total_tokens: int
    """Total tokens consumed by sources (answer not counted)."""

    budget_tokens: int
    """The max_tokens budget that was applied."""

    over_budget: bool
    """True when the full result set would have exceeded the budget."""

    tier_distribution: dict[str, int]
    """Breakdown of how many chunks were packed at each tier level."""

    token_savings_pct: float
    """Estimated savings vs returning all results at L2 (0–100)."""

    provenance: list[str]
    """Memory IDs of all sources used."""

    pack_policy: str
    """The tier-packing policy that was applied."""


@dataclass
class HandoffPacket:
    """Structured context bundle for agent-to-agent task transfer."""

    from_agent: str
    to_agent: str
    session_summary: str
    """Narrative summary of what the sending agent accomplished."""

    active_goals: list[str]
    """Task descriptions from trajectories with outcome='in_progress'."""

    open_questions: list[str]
    """Tips in category='recovery' that may need follow-up."""

    key_memory_ids: list[str]
    """Top hotness + outcome-adjusted memory IDs to seed the receiving agent."""

    knowledge_snapshot: dict
    """Top entities and facts from the knowledge graph."""

    token_count: int
    created_at: str

    ephemeral_notes: list[dict] = field(default_factory=list)
    """Flushed SessionStore entries from the sending agent's session."""


# ---------------------------------------------------------------------------
# get_relevant_context()
# ---------------------------------------------------------------------------


async def get_relevant_context(
    agent_id: str,
    task_description: str,
    max_tokens: int = 8000,
    namespace: str = "",
    tier_policy: str = "adaptive",
    include_graph: bool = True,
    include_tips: bool = True,
    include_hotness: bool = True,
    extra_memory_ids: list[str] | None = None,
) -> RelevantContext:
    """Assemble minimal, token-budgeted context for *agent_id* and *task_description*.

    This is the canonical single-call replacement for:
      ``archivist_search`` + ``archivist_tips`` + ``archivist_context_check``

    The function:
    1. Calls the full retrieval pipeline (vector + BM25 + graph + hotness).
    2. Optionally fetches procedural tips for the agent.
    3. Optionally injects extra memory IDs (e.g. from a handoff packet).
    4. Returns a ``RelevantContext`` struct with packing metadata.

    Args:
        agent_id: The agent requesting context.
        task_description: Free-form description of the current task / query.
        max_tokens: Hard budget for packed sources (default 8000).
        namespace: Limit retrieval to this namespace (empty = all).
        tier_policy: One of 'adaptive' (default), 'l0_first', 'l2_first'.
        include_graph: When True, include entity-fact strings from KG.
        include_tips: When True, include relevant procedural tips.
        include_hotness: When True, hotness scoring is applied to results.
        extra_memory_ids: Additional memory IDs to inject as context
            (e.g. from HandoffPacket.key_memory_ids).

    Returns:
        :class:`RelevantContext` ready to format as a system prompt prefix.
    """
    from archivist.core.config import CONTEXT_L0_BUDGET_SHARE, CONTEXT_MIN_FULL_RESULTS

    # --- Step 1: run the full retrieval pipeline ---
    retrieval_result = await recursive_retrieve(
        query=task_description,
        agent_id=agent_id,
        namespace=namespace,
        max_tokens=max_tokens,
        refine=False,
        tier=tier_policy if tier_policy in ("l0", "l1", "l2") else "l2",
    )

    raw_sources = retrieval_result.get("sources", [])
    answer = retrieval_result.get("answer", "")
    ctx_status = retrieval_result.get("retrieval_trace", {}).get("context_status", {})
    over_budget = retrieval_result.get("over_budget", False)
    tier_dist = ctx_status.get("tier_distribution", {})
    savings_pct = ctx_status.get("token_savings_pct", 0.0)

    # --- Step 2: inject extra_memory_ids if provided ---
    if extra_memory_ids:
        existing_ids = {str(r.get("id", r.get("qdrant_id", ""))) for r in raw_sources}
        for mid in extra_memory_ids:
            if mid not in existing_ids:
                raw_sources.append(
                    {
                        "id": mid,
                        "qdrant_id": mid,
                        "text": f"[pinned memory: {mid}]",
                        "tier_text": f"[pinned memory: {mid}]",
                        "score": 0.0,
                        "file_path": "",
                        "date": "",
                        "agent_id": agent_id,
                        "_packed_tier": "l2",
                    }
                )

    # --- Step 3: re-pack if we have extra IDs or need a fresh budget pass ---
    if extra_memory_ids and raw_sources:
        packed = pack_context(
            raw_sources,
            max_tokens=max_tokens,
            tier_policy=tier_policy,
            l0_budget_share=CONTEXT_L0_BUDGET_SHARE,
            min_full_results=CONTEXT_MIN_FULL_RESULTS,
        )
        packed_sources = packed.sources
        total_tokens = packed.total_tokens
        over_budget = packed.over_budget
        tier_dist = packed.tier_distribution
        savings_pct = packed.token_savings_pct
    else:
        packed_sources = raw_sources
        total_tokens = sum(
            count_tokens(r.get("tier_text") or r.get("text", "")) for r in packed_sources
        )

    # --- Step 4: extract graph facts ---
    graph_facts: list[str] = []
    if include_graph:
        try:
            rt = retrieval_result.get("retrieval_trace", {})
            raw_facts = rt.get("graph_context", [])
            for item in raw_facts:
                if isinstance(item, dict):
                    fact_text = item.get("text") or item.get("fact") or ""
                elif isinstance(item, str):
                    fact_text = item
                else:
                    fact_text = str(item)
                if fact_text:
                    graph_facts.append(fact_text.strip())
        except Exception as e:
            logger.debug("graph_facts extraction failed: %s", e)

    # --- Step 5: fetch procedural tips ---
    tips: list[str] = []
    if include_tips:
        try:
            tip_rows = await search_tips(agent_id=agent_id, limit=5)
            tips = [
                r.get("content") or r.get("tip", "")
                for r in tip_rows
                if r.get("content") or r.get("tip")
            ]
        except Exception as e:
            logger.debug("tips fetch failed: %s", e)

    # --- Build output ---
    chunks = [
        ContextChunk(
            memory_id=str(r.get("id", r.get("qdrant_id", ""))),
            text=r.get("tier_text") or r.get("text", ""),
            score=float(r.get("score", 0.0)),
            tier=r.get("_packed_tier", "l2"),
            file_path=r.get("file_path", ""),
            date=r.get("date", ""),
            agent_id=r.get("agent_id", ""),
        )
        for r in packed_sources
    ]

    provenance = [c.memory_id for c in chunks if c.memory_id]

    return RelevantContext(
        answer=answer,
        sources=chunks,
        graph_facts=graph_facts,
        tips=tips,
        total_tokens=total_tokens,
        budget_tokens=max_tokens,
        over_budget=over_budget,
        tier_distribution=tier_dist,
        token_savings_pct=float(savings_pct),
        provenance=provenance,
        pack_policy=tier_policy,
    )


def format_context_for_prompt(ctx: RelevantContext, include_tips: bool = True) -> str:
    """Render a ``RelevantContext`` as a compact system-prompt prefix.

    Suitable for injection at the top of an agent's context window before
    its task instructions.
    """
    lines: list[str] = []

    if ctx.answer:
        lines.append(f"## Memory Answer\n{ctx.answer}")

    if ctx.sources:
        lines.append("## Relevant Memories")
        for i, chunk in enumerate(ctx.sources, 1):
            meta = f"[{chunk.tier.upper()}]"
            if chunk.date:
                meta += f" {chunk.date}"
            if chunk.agent_id:
                meta += f" (agent: {chunk.agent_id})"
            lines.append(f"{i}. {meta}\n{chunk.text}")

    if ctx.graph_facts:
        lines.append("## Knowledge Graph Facts")
        lines.extend(f"- {f}" for f in ctx.graph_facts[:10])

    if include_tips and ctx.tips:
        lines.append("## Procedural Tips")
        lines.extend(f"- {t}" for t in ctx.tips)

    lines.append(
        f"\n---\n_Context: {ctx.total_tokens}/{ctx.budget_tokens} tokens "
        f"({ctx.token_savings_pct:.1f}% savings vs full-L2)_"
    )

    return "\n\n".join(lines)


# ---------------------------------------------------------------------------
# create_handoff_packet()
# ---------------------------------------------------------------------------


async def create_handoff_packet(
    agent_id: str,
    session_id: str,
    receiving_agent_id: str,
    max_tokens: int = 4000,
    namespace: str = "",
) -> HandoffPacket:
    """Package goal + progress + key memories for agent-to-agent transfer.

    Args:
        agent_id: The sending agent's ID.
        session_id: The session being handed off.
        receiving_agent_id: The agent that will receive this packet.
        max_tokens: Token budget for the key_memory_ids context window.
        namespace: Namespace scope for memory retrieval.

    Returns:
        :class:`HandoffPacket` ready for ``archivist_receive_handoff``.
    """
    from archivist.core.hotness import get_hotness_scores
    from archivist.storage.sqlite_pool import pool

    now_iso = datetime.now(UTC).isoformat()

    # --- 1. Session narrative summary ---
    session_summary = ""
    try:
        from archivist.core.trajectory import session_end_summary

        summary_result = await session_end_summary(agent_id=agent_id, session_id=session_id)
        session_summary = summary_result.get("summary", "")
    except Exception as e:
        logger.debug("handoff session_summary failed: %s", e)

    # --- 2. Active goals from trajectories ---
    active_goals: list[str] = []
    try:
        from archivist.core.trajectory import _ensure_trajectory_schema

        _ensure_trajectory_schema()
        rows = await pool.fetchall(
            "SELECT task_description FROM trajectories "
            "WHERE agent_id=? AND outcome='in_progress' ORDER BY created_at DESC LIMIT 10",
            (agent_id,),
        )
        active_goals = [r["task_description"] for r in rows if r["task_description"]]
    except Exception as e:
        logger.debug("handoff active_goals failed: %s", e)

    # --- 3. Recovery tips ---
    open_questions: list[str] = []
    try:
        tip_rows = await search_tips(agent_id=agent_id, category="recovery", limit=10)
        open_questions = [
            r.get("content") or r.get("tip", "")
            for r in tip_rows
            if r.get("content") or r.get("tip")
        ]
    except Exception as e:
        logger.debug("handoff tips failed: %s", e)

    # --- 4. Top hotness memory IDs ---
    key_memory_ids: list[str] = []
    try:
        rows = await pool.fetchall(
            "SELECT memory_id FROM memory_hotness ORDER BY score DESC LIMIT 20",
        )
        candidates = [r["memory_id"] for r in rows]
        scores = await get_hotness_scores(candidates)
        key_memory_ids = sorted(candidates, key=lambda mid: scores.get(mid, 0.0), reverse=True)[:10]
    except Exception as e:
        logger.debug("handoff key_memory_ids failed: %s", e)

    # --- 5. Knowledge snapshot: top entities + facts ---
    knowledge_snapshot: dict = {"entities": [], "facts": []}
    try:
        entity_rows = await pool.fetchall(
            "SELECT name, entity_type, mention_count FROM entities "
            "WHERE namespace=? OR namespace='global' "
            "ORDER BY mention_count DESC LIMIT 15",
            (namespace,),
        )
        knowledge_snapshot["entities"] = [
            {"name": r["name"], "type": r["entity_type"], "mentions": r["mention_count"]}
            for r in entity_rows
        ]
        fact_rows = await pool.fetchall(
            "SELECT subject, predicate, object FROM facts "
            "WHERE confidence >= 0.7 ORDER BY created_at DESC LIMIT 20",
        )
        knowledge_snapshot["facts"] = [
            f"{r['subject']} {r['predicate']} {r['object']}" for r in fact_rows
        ]
    except Exception as e:
        logger.debug("handoff knowledge_snapshot failed: %s", e)

    # --- 6. Flush ephemeral session store ---
    ephemeral_notes: list[dict] = []
    try:
        ss = get_session_store()
        ephemeral_notes = ss.flush(agent_id, session_id)
    except Exception as e:
        logger.debug("handoff session_store flush failed: %s", e)

    # --- Build packet and measure tokens ---
    packet_text = (
        f"{session_summary}\n"
        + "\n".join(active_goals)
        + "\n".join(open_questions)
        + " ".join(key_memory_ids)
    )
    token_count = count_tokens(packet_text)

    return HandoffPacket(
        from_agent=agent_id,
        to_agent=receiving_agent_id,
        session_summary=session_summary,
        active_goals=active_goals,
        open_questions=open_questions,
        key_memory_ids=key_memory_ids,
        knowledge_snapshot=knowledge_snapshot,
        token_count=token_count,
        created_at=now_iso,
        ephemeral_notes=ephemeral_notes,
    )


async def receive_handoff_packet(
    packet: HandoffPacket,
    receiving_agent_id: str,
    session_id: str,
) -> dict:
    """Inject a HandoffPacket as ephemeral session memory for the receiving agent.

    Stores the session summary and active goals in the SessionStore so they
    are immediately available via ``get_relevant_context()`` for the new session.

    Returns a summary dict describing what was injected.
    """
    ss = get_session_store()
    injected: list[str] = []

    if packet.session_summary:
        ss.put(receiving_agent_id, session_id, "handoff_summary", packet.session_summary)
        injected.append("handoff_summary")

    for i, goal in enumerate(packet.active_goals):
        key = f"handoff_goal_{i}"
        ss.put(receiving_agent_id, session_id, key, goal)
        injected.append(key)

    for i, q in enumerate(packet.open_questions):
        key = f"handoff_recovery_{i}"
        ss.put(receiving_agent_id, session_id, key, q)
        injected.append(key)

    for note in packet.ephemeral_notes:
        key = f"handoff_note_{note.get('key', '')}"
        ss.put(receiving_agent_id, session_id, key, note.get("value", ""))
        injected.append(key)

    return {
        "from_agent": packet.from_agent,
        "to_agent": receiving_agent_id,
        "session_id": session_id,
        "injected_keys": injected,
        "key_memory_ids": packet.key_memory_ids,
        "knowledge_snapshot": packet.knowledge_snapshot,
        "token_count": packet.token_count,
    }
