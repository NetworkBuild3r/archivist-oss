"""Trajectory logging, decision attribution, tips, and outcome-aware retrieval.

Inspired by arXiv:2603.10600 (Trajectory-Informed Memory). Agents log execution
trajectories (actions + outcomes), which are analyzed to extract actionable tips
(strategy / recovery / optimization). Memories linked to successful trajectories
get retrieval boosts; those linked to failures get warnings.
"""

import json
import logging
import uuid
from datetime import datetime, timezone

from graph import get_db, GRAPH_WRITE_LOCK, upsert_entity, add_fact
from config import OUTCOME_RETRIEVAL_BOOST, OUTCOME_RETRIEVAL_PENALTY
from llm import llm_query

logger = logging.getLogger("archivist.trajectory")

_SCHEMA_APPLIED = False

_ATTRIBUTION_SYSTEM = (
    "You are a post-mortem analyst. Given an agent's execution trajectory "
    "(actions taken and final outcome), identify which specific memories were "
    "most influential in the decisions made. Return a JSON array of objects: "
    '[{"memory_id": "...", "influence": "high|medium|low", "reasoning": "..."}]. '
    "Return ONLY the JSON array, no wrapping."
)

_TIP_SYSTEM = (
    "You are a meta-learning assistant. Given an agent's execution trajectory and outcome, "
    "extract 1-3 actionable tips that could help in similar future situations. "
    "Categorize each tip as: strategy (general approach), recovery (error handling), "
    "or optimization (efficiency improvement). Return a JSON array of objects: "
    '[{"category": "strategy|recovery|optimization", "tip": "...", "context": "..."}]. '
    "Return ONLY the JSON array, no wrapping."
)

_SESSION_SYSTEM = (
    "You are a session summarizer for an AI agent's memory system. "
    "Given a list of actions and memories from a work session, produce a concise "
    "durable summary (3-6 sentences) capturing: key decisions made, outcomes, "
    "new information learned, and any unresolved issues. Be factual and specific."
)


def _ensure_trajectory_schema():
    global _SCHEMA_APPLIED
    if _SCHEMA_APPLIED:
        return
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS trajectories (
            id TEXT PRIMARY KEY,
            agent_id TEXT NOT NULL,
            session_id TEXT,
            task_description TEXT NOT NULL,
            actions TEXT NOT NULL DEFAULT '[]',
            outcome TEXT NOT NULL DEFAULT 'unknown',
            outcome_score REAL,
            memory_ids_used TEXT DEFAULT '[]',
            created_at TEXT NOT NULL,
            metadata TEXT DEFAULT '{}'
        );
        CREATE INDEX IF NOT EXISTS idx_traj_agent ON trajectories(agent_id);
        CREATE INDEX IF NOT EXISTS idx_traj_session ON trajectories(session_id);
        CREATE INDEX IF NOT EXISTS idx_traj_outcome ON trajectories(outcome);

        CREATE TABLE IF NOT EXISTS tips (
            id TEXT PRIMARY KEY,
            trajectory_id TEXT NOT NULL REFERENCES trajectories(id),
            agent_id TEXT NOT NULL,
            category TEXT NOT NULL,
            tip_text TEXT NOT NULL,
            context TEXT,
            negative_example TEXT,
            archived INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            usage_count INTEGER NOT NULL DEFAULT 0,
            last_used_at TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_tips_agent ON tips(agent_id);
        CREATE INDEX IF NOT EXISTS idx_tips_category ON tips(category);
        CREATE INDEX IF NOT EXISTS idx_tips_archived ON tips(archived);

        CREATE TABLE IF NOT EXISTS annotations (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            annotation_type TEXT NOT NULL DEFAULT 'note',
            content TEXT NOT NULL,
            quality_score REAL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ann_memory ON annotations(memory_id);
        CREATE INDEX IF NOT EXISTS idx_ann_agent ON annotations(agent_id);

        CREATE TABLE IF NOT EXISTS ratings (
            id TEXT PRIMARY KEY,
            memory_id TEXT NOT NULL,
            agent_id TEXT NOT NULL,
            rating INTEGER NOT NULL,
            context TEXT,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_ratings_memory ON ratings(memory_id);
        CREATE INDEX IF NOT EXISTS idx_ratings_agent ON ratings(agent_id);

        CREATE TABLE IF NOT EXISTS memory_outcomes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id TEXT NOT NULL,
            trajectory_id TEXT NOT NULL REFERENCES trajectories(id),
            influence TEXT NOT NULL DEFAULT 'medium',
            outcome TEXT NOT NULL,
            outcome_score REAL,
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_mo_memory ON memory_outcomes(memory_id);
        """)
        conn.commit()
        conn.close()
    _SCHEMA_APPLIED = True


async def log_trajectory(
    agent_id: str,
    task_description: str,
    actions: list[dict],
    outcome: str,
    outcome_score: float | None = None,
    memory_ids_used: list[str] | None = None,
    session_id: str = "",
    metadata: dict | None = None,
) -> dict:
    """Log an execution trajectory and auto-extract tips via LLM."""
    _ensure_trajectory_schema()

    traj_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            conn.execute(
                """INSERT INTO trajectories
                   (id, agent_id, session_id, task_description, actions, outcome, outcome_score,
                    memory_ids_used, created_at, metadata)
                   VALUES (?,?,?,?,?,?,?,?,?,?)""",
                (traj_id, agent_id, session_id, task_description,
                 json.dumps(actions), outcome, outcome_score,
                 json.dumps(memory_ids_used or []), now,
                 json.dumps(metadata or {})),
            )
            conn.commit()
        finally:
            conn.close()

    eid = upsert_entity(agent_id, "agent")
    add_fact(eid, f"Trajectory [{outcome}]: {task_description[:200]}", f"trajectory/{traj_id}", agent_id)

    return {"trajectory_id": traj_id, "agent_id": agent_id, "outcome": outcome}


async def attribute_decisions(trajectory_id: str) -> list[dict]:
    """Use LLM to identify which memories most influenced decisions in a trajectory."""
    _ensure_trajectory_schema()

    conn = get_db()
    cur = conn.execute("SELECT * FROM trajectories WHERE id=?", (trajectory_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return []

    traj = dict(row)
    mem_ids = json.loads(traj.get("memory_ids_used", "[]"))
    if not mem_ids:
        return []

    prompt = (
        f"Task: {traj['task_description']}\n"
        f"Outcome: {traj['outcome']} (score: {traj.get('outcome_score', 'N/A')})\n"
        f"Actions: {traj['actions']}\n"
        f"Memory IDs consulted: {json.dumps(mem_ids)}\n\n"
        "Which memories were most influential?"
    )

    try:
        raw = await llm_query(prompt, system=_ATTRIBUTION_SYSTEM, max_tokens=512)
        attributions = json.loads(raw.strip())
    except Exception as e:
        logger.warning("Decision attribution LLM failed: %s", e)
        attributions = [{"memory_id": mid, "influence": "unknown", "reasoning": "LLM unavailable"} for mid in mem_ids]

    now = datetime.now(timezone.utc).isoformat()
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            for a in attributions:
                conn.execute(
                    """INSERT INTO memory_outcomes (memory_id, trajectory_id, influence, outcome, outcome_score, created_at)
                       VALUES (?,?,?,?,?,?)""",
                    (a.get("memory_id", ""), trajectory_id, a.get("influence", "medium"),
                     traj["outcome"], traj.get("outcome_score"), now),
                )
            conn.commit()
        finally:
            conn.close()

    return attributions


async def extract_tips(trajectory_id: str) -> list[dict]:
    """Extract actionable tips from a trajectory via LLM analysis."""
    _ensure_trajectory_schema()

    conn = get_db()
    cur = conn.execute("SELECT * FROM trajectories WHERE id=?", (trajectory_id,))
    row = cur.fetchone()
    conn.close()

    if not row:
        return []

    traj = dict(row)
    prompt = (
        f"Task: {traj['task_description']}\n"
        f"Outcome: {traj['outcome']} (score: {traj.get('outcome_score', 'N/A')})\n"
        f"Actions taken: {traj['actions']}\n"
    )

    try:
        raw = await llm_query(prompt, system=_TIP_SYSTEM, max_tokens=512)
        tips_data = json.loads(raw.strip())
    except Exception as e:
        logger.warning("Tip extraction LLM failed: %s", e)
        return []

    now = datetime.now(timezone.utc).isoformat()
    stored_tips = []
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            for t in tips_data:
                tip_id = str(uuid.uuid4())
                conn.execute(
                    """INSERT INTO tips (id, trajectory_id, agent_id, category, tip_text, context, created_at)
                       VALUES (?,?,?,?,?,?,?)""",
                    (tip_id, trajectory_id, traj["agent_id"],
                     t.get("category", "strategy"), t.get("tip", ""),
                     t.get("context", ""), now),
                )
                stored_tips.append({"tip_id": tip_id, **t})
            conn.commit()
        finally:
            conn.close()

    return stored_tips


def get_outcome_adjustments(memory_ids: list[str]) -> dict[str, float]:
    """Return score adjustments for memories based on their outcome history.

    Memories linked to successful trajectories get a positive boost;
    those linked to failures get a negative penalty.
    """
    _ensure_trajectory_schema()
    if not memory_ids:
        return {}

    conn = get_db()
    placeholders = ",".join("?" for _ in memory_ids)
    cur = conn.execute(
        f"""SELECT memory_id, outcome, outcome_score, influence
            FROM memory_outcomes
            WHERE memory_id IN ({placeholders})""",
        memory_ids,
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    adjustments: dict[str, float] = {}
    for r in rows:
        mid = r["memory_id"]
        adj = adjustments.get(mid, 0.0)
        influence_mult = {"high": 1.0, "medium": 0.6, "low": 0.3}.get(r["influence"], 0.5)
        if r["outcome"] == "success":
            adj += OUTCOME_RETRIEVAL_BOOST * influence_mult
        elif r["outcome"] == "failure":
            adj -= OUTCOME_RETRIEVAL_PENALTY * influence_mult
        adjustments[mid] = adj

    return adjustments


def search_tips(agent_id: str, category: str = "", limit: int = 10) -> list[dict]:
    """Retrieve non-archived tips for an agent, optionally filtered by category."""
    _ensure_trajectory_schema()
    conn = get_db()
    if category:
        cur = conn.execute(
            "SELECT * FROM tips WHERE agent_id=? AND category=? AND archived=0 ORDER BY created_at DESC LIMIT ?",
            (agent_id, category, limit),
        )
    else:
        cur = conn.execute(
            "SELECT * FROM tips WHERE agent_id=? AND archived=0 ORDER BY created_at DESC LIMIT ?",
            (agent_id, limit),
        )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def add_annotation(memory_id: str, agent_id: str, content: str,
                   annotation_type: str = "note", quality_score: float | None = None) -> str:
    """Add an annotation to a memory point."""
    _ensure_trajectory_schema()
    ann_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            conn.execute(
                """INSERT INTO annotations (id, memory_id, agent_id, annotation_type, content, quality_score, created_at)
                   VALUES (?,?,?,?,?,?,?)""",
                (ann_id, memory_id, agent_id, annotation_type, content, quality_score, now),
            )
            conn.commit()
        finally:
            conn.close()
    return ann_id


def get_annotations(memory_id: str) -> list[dict]:
    """Get all annotations for a memory point."""
    _ensure_trajectory_schema()
    conn = get_db()
    cur = conn.execute(
        "SELECT * FROM annotations WHERE memory_id=? ORDER BY created_at DESC",
        (memory_id,),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def add_rating(memory_id: str, agent_id: str, rating: int, context: str = "") -> str:
    """Rate a memory (positive = +1, negative = -1)."""
    _ensure_trajectory_schema()
    rating_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    with GRAPH_WRITE_LOCK:
        conn = get_db()
        try:
            conn.execute(
                "INSERT INTO ratings (id, memory_id, agent_id, rating, context, created_at) VALUES (?,?,?,?,?,?)",
                (rating_id, memory_id, agent_id, max(-1, min(1, rating)), context, now),
            )
            conn.commit()
        finally:
            conn.close()
    return rating_id


def get_rating_summary(memory_id: str) -> dict:
    """Get aggregate rating stats for a memory point."""
    _ensure_trajectory_schema()
    conn = get_db()
    cur = conn.execute(
        "SELECT COUNT(*) as total, SUM(CASE WHEN rating > 0 THEN 1 ELSE 0 END) as up, "
        "SUM(CASE WHEN rating < 0 THEN 1 ELSE 0 END) as down FROM ratings WHERE memory_id=?",
        (memory_id,),
    )
    row = dict(cur.fetchone())
    conn.close()
    return {"memory_id": memory_id, "total": row["total"] or 0, "up": row["up"] or 0, "down": row["down"] or 0}


async def session_end_summary(agent_id: str, session_id: str) -> dict:
    """Generate a durable summary from a session's trajectories and store it as a memory."""
    _ensure_trajectory_schema()

    conn = get_db()
    cur = conn.execute(
        "SELECT * FROM trajectories WHERE agent_id=? AND session_id=? ORDER BY created_at ASC",
        (agent_id, session_id),
    )
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()

    if not rows:
        return {"error": "no_trajectories", "session_id": session_id}

    actions_text = "\n".join(
        f"- [{t['outcome']}] {t['task_description'][:200]}"
        for t in rows
    )

    prompt = f"Agent: {agent_id}\nSession: {session_id}\n\nActivities:\n{actions_text}"

    try:
        summary = await llm_query(prompt, system=_SESSION_SYSTEM, max_tokens=400)
        summary = summary.strip()
    except Exception as e:
        logger.warning("Session summary LLM failed: %s", e)
        summary = f"Session {session_id}: {len(rows)} tasks completed."

    return {
        "session_id": session_id,
        "agent_id": agent_id,
        "trajectory_count": len(rows),
        "summary": summary,
    }


# ── Tip consolidation (v1.0) ────────────────────────────────────────────────

_CONSOLIDATION_SYSTEM = (
    "You are a tip consolidation assistant. Given a cluster of similar operational tips, "
    "merge them into ONE canonical tip that captures all the useful information. "
    "Include a 'negative_example' field with an explicit anti-pattern — what NOT to do. "
    "Resolve conflicts: tips from successful trajectories take priority over failure-linked ones.\n\n"
    "Return a JSON object: "
    '{"category": "strategy|recovery|optimization", '
    '"tip": "the consolidated tip text", '
    '"context": "when this applies", '
    '"negative_example": "what NOT to do and why"}. '
    "Return ONLY the JSON object."
)

_CONTRASTIVE_CONSOLIDATION_SYSTEM = (
    "You consolidate operational tips that came from DIFFERENT agents working on related topics. "
    "Extract invariant guidance that all agents' experience supports. "
    "Identify violation patterns that distinguish success from failure across agents. "
    "Discard agent-specific tool preferences, phrasing style, or architectural assumptions "
    "unless all agents agree they matter.\n\n"
    "Return a JSON object: "
    '{"category": "strategy|recovery|optimization", '
    '"tip": "the consolidated tip text", '
    '"context": "when this applies", '
    '"negative_example": "what NOT to do and why", '
    '"invariant_source": "cross-agent"}. '
    "Return ONLY the JSON object."
)


async def consolidate_tips(budget: int = 20) -> dict:
    """Cluster similar tips and merge clusters via LLM. Budget-capped.

    Returns summary of consolidation: clusters found, merged, budget used.
    """
    import re
    from config import CURATOR_TIP_BUDGET
    from embeddings import embed_text
    import metrics as m
    import curator_queue

    _ensure_trajectory_schema()
    budget = budget or CURATOR_TIP_BUDGET

    conn = get_db()
    cur = conn.execute(
        "SELECT id, agent_id, category, tip_text, context FROM tips WHERE archived=0"
    )
    all_tips = [dict(r) for r in cur.fetchall()]
    conn.close()

    if len(all_tips) < 3:
        return {"clusters_found": 0, "consolidated": 0, "budget_used": 0}

    embeddings = []
    for tip in all_tips:
        vec = await embed_text(tip["tip_text"])
        embeddings.append(vec)

    clusters = _cluster_tips(all_tips, embeddings, threshold=0.85)
    mergeable = [c for c in clusters if len(c) >= 3]

    consolidated_count = 0
    budget_used = 0

    for cluster in mergeable:
        if budget_used >= budget:
            break

        agent_ids = {t.get("agent_id") or "" for t in cluster}
        agent_ids.discard("")
        contrastive = len(agent_ids) >= 2
        tips_text = "\n\n".join(
            f"[agent={t.get('agent_id', '')}] [{t['category']}] {t['tip_text']}\nContext: {t.get('context', '')}"
            for t in cluster
        )
        prompt = f"TIPS TO CONSOLIDATE ({len(cluster)} tips):\n\n{tips_text}"
        system = _CONTRASTIVE_CONSOLIDATION_SYSTEM if contrastive else _CONSOLIDATION_SYSTEM

        try:
            m.inc(m.CURATOR_LLM_CALLS)
            budget_used += 1
            raw = await llm_query(prompt, system=system, max_tokens=512, json_mode=True)
            raw = raw.strip()
            raw = re.sub(r"^```(?:json)?\n?", "", raw)
            raw = re.sub(r"\n?```$", "", raw)
            merged = json.loads(raw)
        except Exception as e:
            logger.warning("Tip consolidation LLM failed: %s", e)
            continue

        original_ids = [t["id"] for t in cluster]
        fleet_agent = "fleet" if contrastive else cluster[0].get("agent_id", "curator")
        curator_queue.enqueue("consolidate_tips", {
            "consolidated_tip": {
                "agent_id": fleet_agent,
                "category": merged.get("category", "strategy"),
                "tip": merged.get("tip", ""),
                "context": merged.get("context", ""),
                "negative_example": merged.get("negative_example", ""),
            },
            "original_tip_ids": original_ids,
        })
        m.inc(m.CURATOR_TIP_CONSOLIDATIONS)
        consolidated_count += 1

    return {
        "tips_total": len(all_tips),
        "clusters_found": len(mergeable),
        "consolidated": consolidated_count,
        "budget_used": budget_used,
        "budget_limit": budget,
    }


def _cluster_tips(tips: list[dict], embeddings: list[list[float]], threshold: float = 0.85) -> list[list[dict]]:
    """Simple greedy clustering by cosine similarity."""
    import math

    def cosine_sim(a, b):
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    n = len(tips)
    assigned = [False] * n
    clusters = []

    for i in range(n):
        if assigned[i]:
            continue
        cluster = [tips[i]]
        assigned[i] = True
        for j in range(i + 1, n):
            if assigned[j]:
                continue
            sim = cosine_sim(embeddings[i], embeddings[j])
            if sim >= threshold:
                cluster.append(tips[j])
                assigned[j] = True
        clusters.append(cluster)

    return clusters
