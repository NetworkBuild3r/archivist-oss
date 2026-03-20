"""Graph-augmented retrieval — hybrid vector + KG, temporal decay, multi-hop, contradictions.

Called from rlm_retriever when GRAPH_RETRIEVAL_ENABLED is set.
"""

import math
import logging
from datetime import datetime, timezone

from graph import search_entities, get_entity_facts, get_entity_relationships
from config import GRAPH_RETRIEVAL_WEIGHT, MULTI_HOP_DEPTH, TEMPORAL_DECAY_HALFLIFE_DAYS

logger = logging.getLogger("archivist.graph_retrieval")


def extract_entity_mentions(query: str) -> list[dict]:
    """Find entities from the KG whose names appear in the query."""
    words = set(query.lower().split())
    results = []
    for word in words:
        if len(word) < 3:
            continue
        found = search_entities(word, limit=3)
        results.extend(found)
    seen = set()
    deduped = []
    for e in results:
        if e["id"] not in seen:
            seen.add(e["id"])
            deduped.append(e)
    return deduped


def graph_context_for_entities(entity_ids: list[int], depth: int = 1) -> list[dict]:
    """Gather facts and relationships for a set of entities up to `depth` hops."""
    visited: set[int] = set()
    frontier = list(entity_ids)
    context_items: list[dict] = []

    for hop in range(depth):
        next_frontier: list[int] = []
        for eid in frontier:
            if eid in visited:
                continue
            visited.add(eid)

            facts = get_entity_facts(eid)
            for f in facts:
                context_items.append({
                    "type": "fact",
                    "entity_id": eid,
                    "text": f["fact_text"],
                    "agent_id": f.get("agent_id", ""),
                    "created_at": f.get("created_at", ""),
                    "source_file": f.get("source_file", ""),
                    "hop": hop,
                })

            rels = get_entity_relationships(eid)
            for r in rels:
                context_items.append({
                    "type": "relationship",
                    "source": r.get("source_name", ""),
                    "target": r.get("target_name", ""),
                    "relation": r["relation_type"],
                    "evidence": r.get("evidence", ""),
                    "agent_id": r.get("agent_id", ""),
                    "hop": hop,
                })
                other = r["target_entity_id"] if r["source_entity_id"] == eid else r["source_entity_id"]
                if other not in visited:
                    next_frontier.append(other)

        frontier = next_frontier
        if not frontier:
            break

    return context_items


def apply_temporal_decay(results: list[dict], halflife_days: int | None = None) -> list[dict]:
    """Multiply each result's score by an exponential decay factor based on its date.

    Newer memories score higher; very old memories are down-weighted but never zeroed.
    """
    hl = halflife_days or TEMPORAL_DECAY_HALFLIFE_DAYS
    if hl <= 0:
        return results

    now = datetime.now(timezone.utc).date()
    ln2 = math.log(2)

    for r in results:
        date_str = r.get("date", "")
        if not date_str:
            continue
        try:
            d = datetime.strptime(date_str[:10], "%Y-%m-%d").date()
            age_days = max((now - d).days, 0)
            decay = math.exp(-ln2 * age_days / hl)
            r["original_score"] = r.get("score", 0)
            r["score"] = r.get("score", 0) * decay
            r["temporal_decay"] = round(decay, 4)
        except (ValueError, TypeError):
            pass

    results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return results


def merge_graph_context_into_results(
    vector_results: list[dict],
    graph_items: list[dict],
    weight: float | None = None,
) -> list[dict]:
    """Blend graph-sourced context into vector results.

    Graph items that match an existing vector hit (same source file) boost its score.
    New graph-only items are appended as synthetic hits.
    """
    w = weight if weight is not None else GRAPH_RETRIEVAL_WEIGHT

    file_index: dict[str, int] = {}
    for i, r in enumerate(vector_results):
        fp = r.get("file_path", "")
        if fp and fp not in file_index:
            file_index[fp] = i

    added: set[str] = set()
    for gi in graph_items:
        src = gi.get("source_file", "")
        text = gi.get("text", gi.get("evidence", ""))
        if not text:
            continue

        if src and src in file_index:
            idx = file_index[src]
            vector_results[idx]["score"] = vector_results[idx].get("score", 0) + w * 0.5
            if "graph_context" not in vector_results[idx]:
                vector_results[idx]["graph_context"] = []
            vector_results[idx]["graph_context"].append(text[:300])
        else:
            dedup_key = f"{gi.get('type', '')}:{text[:80]}"
            if dedup_key in added:
                continue
            added.add(dedup_key)
            vector_results.append({
                "id": "",
                "score": w,
                "text": text[:500],
                "agent_id": gi.get("agent_id", ""),
                "file_path": src,
                "file_type": "graph",
                "date": gi.get("created_at", "")[:10] if gi.get("created_at") else "",
                "team": "",
                "namespace": "",
                "chunk_index": 0,
                "parent_id": None,
                "is_parent": False,
                "graph_hop": gi.get("hop", 0),
            })

    vector_results.sort(key=lambda x: x.get("score", 0), reverse=True)
    return vector_results


def detect_contradictions(entity_id: int) -> list[dict]:
    """Find potentially contradicting facts about the same entity.

    Uses simple heuristic: facts from different agents about the same entity
    where one supersedes the other, or facts that contain opposing keywords.
    """
    facts = get_entity_facts(entity_id)
    if len(facts) < 2:
        return []

    _OPPOSING = [
        ("enabled", "disabled"), ("active", "inactive"), ("yes", "no"),
        ("true", "false"), ("success", "failure"), ("up", "down"),
        ("running", "stopped"), ("allow", "deny"), ("open", "closed"),
    ]

    contradictions: list[dict] = []
    for i, a in enumerate(facts):
        for b in facts[i + 1:]:
            if a.get("agent_id") == b.get("agent_id"):
                continue
            a_lower = a["fact_text"].lower()
            b_lower = b["fact_text"].lower()
            for pos, neg in _OPPOSING:
                if (pos in a_lower and neg in b_lower) or (neg in a_lower and pos in b_lower):
                    contradictions.append({
                        "fact_a": a["fact_text"],
                        "fact_a_agent": a.get("agent_id", ""),
                        "fact_a_date": a.get("created_at", ""),
                        "fact_b": b["fact_text"],
                        "fact_b_agent": b.get("agent_id", ""),
                        "fact_b_date": b.get("created_at", ""),
                        "trigger": f"{pos}/{neg}",
                    })
                    break

    return contradictions
