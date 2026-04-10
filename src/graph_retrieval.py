"""Graph-augmented retrieval — hybrid vector + KG, temporal decay, multi-hop, contradictions.

Called from rlm_retriever when GRAPH_RETRIEVAL_ENABLED is set.
"""

import math
import logging
from datetime import datetime, timezone

from graph import (
    search_entities, get_entity_facts, get_entity_relationships,
    get_entity_facts_bulk, get_entity_relationships_bulk,
    get_entity_by_id, _normalize,
)
from config import (
    GRAPH_RETRIEVAL_WEIGHT, TEMPORAL_DECAY_HALFLIFE_DAYS,
    MAX_ENTITY_FACT_INJECTIONS, ENTITY_SPECIFICITY_MAX_MENTIONS,
)

logger = logging.getLogger("archivist.graph_retrieval")


_STOPWORDS = frozenset({
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "shall",
    "should", "may", "might", "must", "can", "could", "not", "no", "nor",
    "and", "but", "or", "if", "then", "else", "when", "where", "why",
    "how", "what", "which", "who", "whom", "this", "that", "these",
    "those", "it", "its", "my", "your", "our", "their", "he", "she",
    "we", "they", "me", "him", "her", "us", "them", "i", "you",
    "of", "in", "to", "for", "with", "on", "at", "from", "by", "as",
    "into", "about", "between", "through", "during", "before", "after",
    "above", "below", "up", "down", "out", "off", "over", "under",
    "find", "get", "tell", "show", "give", "know", "use", "used",
    "exact", "all", "any", "each", "every", "both", "few", "more",
    "most", "other", "some", "such", "only", "own", "same", "so",
    "than", "too", "very", "just", "also", "now",
})


def extract_entity_mentions(query: str) -> list[dict]:
    """Find entities from the KG whose names appear in the query.

    Uses N-gram expansion (1, 2, 3-word windows) to match multi-word
    entities like "Argo CD", "hot cache", or "Kubernetes cluster".
    Longer phrases are tried first so multi-word matches take priority.

    Applies a specificity filter: single-word stopwords and high-mention-count
    generic entities are skipped to prevent graph injection dilution (v1.8).
    """
    tokens = query.lower().split()
    results = []
    seen: set[int] = set()

    for n in (3, 2, 1):
        for i in range(len(tokens) - n + 1):
            phrase = " ".join(tokens[i:i + n])
            if len(phrase) < 2:
                continue
            # Skip single stopwords — they match too many entities
            if n == 1 and phrase in _STOPWORDS:
                continue
            found = search_entities(phrase, limit=3)
            for e in found:
                if e["id"] in seen:
                    continue
                # Specificity guard: skip generic, high-frequency entities
                # matched by a short single-word phrase (e.g. "production",
                # "backup"). Exact name matches bypass this check.
                if n == 1 and e.get("mention_count", 0) > ENTITY_SPECIFICITY_MAX_MENTIONS:
                    if _normalize(e["name"]) != _normalize(phrase):
                        continue
                seen.add(e["id"])
                results.append(e)

    return results


_MAX_GRAPH_CONTEXT_ITEMS = 50


def graph_context_for_entities(entity_ids: list[int], depth: int = 1) -> list[dict]:
    """Gather facts and relationships for a set of entities up to `depth` hops.

    Uses bulk queries per hop to avoid N+1 DB round-trips.
    Capped at ``_MAX_GRAPH_CONTEXT_ITEMS`` total items to prevent
    graph-heavy queries from flooding the coarse result set.
    """
    visited: set[int] = set()
    frontier = list(entity_ids)
    context_items: list[dict] = []

    for hop in range(depth):
        unvisited = [eid for eid in frontier if eid not in visited]
        if not unvisited:
            break
        visited.update(unvisited)

        all_facts = get_entity_facts_bulk(unvisited)
        all_rels = get_entity_relationships_bulk(unvisited)

        next_frontier: list[int] = []
        for eid in unvisited:
            for f in all_facts.get(eid, []):
                context_items.append({
                    "type": "fact",
                    "entity_id": eid,
                    "text": f["fact_text"],
                    "agent_id": f.get("agent_id", ""),
                    "created_at": f.get("created_at", ""),
                    "source_file": f.get("source_file", ""),
                    "hop": hop,
                })
                if len(context_items) >= _MAX_GRAPH_CONTEXT_ITEMS:
                    return context_items

            for r in all_rels.get(eid, []):
                context_items.append({
                    "type": "relationship",
                    "source": r.get("source_name", ""),
                    "target": r.get("target_name", ""),
                    "relation": r["relation_type"],
                    "evidence": r.get("evidence", ""),
                    "agent_id": r.get("agent_id", ""),
                    "hop": hop,
                })
                if len(context_items) >= _MAX_GRAPH_CONTEXT_ITEMS:
                    return context_items
                other = r["target_entity_id"] if r["source_entity_id"] == eid else r["source_entity_id"]
                if other not in visited:
                    next_frontier.append(other)

        frontier = next_frontier

    return context_items


def apply_temporal_decay(
    results: list[dict],
    halflife_days: int | None = None,
    temporal_intent: str = "neutral",
) -> list[dict]:
    """Multiply each result's score by an exponential decay factor based on its date.

    Newer memories score higher; very old memories are down-weighted but never zeroed.

    When ``temporal_intent`` is ``"historical"``, the halflife is multiplied by
    TEMPORAL_HISTORICAL_HALFLIFE_MULTIPLIER (default 10×) so dated documents
    are barely decayed — the query *wants* old facts.  When ``"recency"``,
    normal halflife applies.  Documents without a ``content_date`` (i.e. the
    date was inferred from indexing time, not from the filename) are never
    decayed — they represent reference material with unknown event date.
    """
    from config import TEMPORAL_INTENT_ENABLED, TEMPORAL_HISTORICAL_HALFLIFE_MULTIPLIER

    hl = halflife_days or TEMPORAL_DECAY_HALFLIFE_DAYS
    if hl <= 0:
        return results

    if TEMPORAL_INTENT_ENABLED and temporal_intent == "historical":
        hl = int(hl * TEMPORAL_HISTORICAL_HALFLIFE_MULTIPLIER)

    now = datetime.now(timezone.utc).date()
    ln2 = math.log(2)

    for r in results:
        content_date = r.get("content_date", "")
        if not content_date:
            continue
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


_MAX_GRAPH_SYNTHETIC_HITS = 20


def merge_graph_context_into_results(
    vector_results: list[dict],
    graph_items: list[dict],
    weight: float | None = None,
) -> list[dict]:
    """Blend graph-sourced context into vector results.

    Graph items that match an existing vector hit (same source file) boost its score.
    New graph-only items are appended as synthetic hits (capped at
    ``_MAX_GRAPH_SYNTHETIC_HITS`` to prevent graph-heavy queries from
    dominating the coarse result set).
    """
    w = weight if weight is not None else GRAPH_RETRIEVAL_WEIGHT

    file_index: dict[str, int] = {}
    for i, r in enumerate(vector_results):
        fp = r.get("file_path", "")
        if fp and fp not in file_index:
            file_index[fp] = i

    added: set[str] = set()
    n_synthetic = 0
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
            if n_synthetic >= _MAX_GRAPH_SYNTHETIC_HITS:
                continue
            dedup_key = f"{gi.get('type', '')}:{text[:80]}"
            if dedup_key in added:
                continue
            added.add(dedup_key)
            n_synthetic += 1
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


def get_entity_brief(entity_id: int, as_of: str = "") -> dict | None:
    """Build a structured summary card for an entity.

    Reusable by MCP tools (archivist_entity_brief, archivist_recall) and
    internal retrieval logic.

    When *as_of* is provided only facts valid at that date are included.
    """
    entity = get_entity_by_id(entity_id)
    if not entity:
        return None

    facts = get_entity_facts(entity_id, as_of=as_of)
    rels = get_entity_relationships(entity_id)

    return {
        "entity": {
            "id": entity["id"],
            "name": entity["name"],
            "type": entity.get("entity_type", "unknown"),
            "retention_class": entity.get("retention_class", "standard"),
            "mention_count": entity.get("mention_count", 0),
            "first_seen": entity.get("first_seen", ""),
            "last_seen": entity.get("last_seen", ""),
            "aliases": entity.get("aliases", "[]"),
        },
        "facts": [
            {
                "text": f["fact_text"],
                "agent_id": f.get("agent_id", ""),
                "source_file": f.get("source_file", ""),
                "created_at": f.get("created_at", ""),
                "retention_class": f.get("retention_class", "standard"),
                "valid_from": f.get("valid_from", ""),
                "valid_until": f.get("valid_until", ""),
            }
            for f in facts
        ],
        "relationships": [
            {
                "source": r.get("source_name", ""),
                "target": r.get("target_name", ""),
                "relation": r["relation_type"],
                "evidence": r.get("evidence", ""),
                "confidence": r.get("confidence", 1.0),
            }
            for r in rels
        ],
        "fact_count": len(facts),
        "relationship_count": len(rels),
    }


def build_entity_fact_results(
    entities: list[dict],
    min_score: float = 0.70,
    max_injected: int | None = None,
    as_of: str = "",
) -> list[dict]:
    """Convert entity graph facts into synthetic retrieval results.

    These bypass the vector similarity threshold so that known entity
    facts are guaranteed to appear in search results when the entity
    is mentioned in the query.

    When *as_of* is provided only facts valid at that date are included.

    Capped to ``max_injected`` (default from config) to prevent
    dilution of vector results for queries that match many entities.

    Uses a single bulk query for all entity facts instead of N queries.
    """
    cap = max_injected if max_injected is not None else MAX_ENTITY_FACT_INJECTIONS
    results: list[dict] = []
    seen_texts: set[str] = set()

    eids = [e["id"] for e in entities]
    all_facts = get_entity_facts_bulk(eids, as_of=as_of)
    entity_map = {e["id"]: e for e in entities}

    for eid in eids:
        entity = entity_map[eid]
        retention = entity.get("retention_class", "standard")
        score_boost = 0.05 if retention in ("durable", "permanent") else 0.0

        for f in all_facts.get(eid, []):
            text_key = f["fact_text"][:100]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)

            results.append({
                "id": "",
                "score": min_score + score_boost,
                "text": f"[{entity['name']}] {f['fact_text']}",
                "l0": "",
                "l1": "",
                "agent_id": f.get("agent_id", ""),
                "file_path": f.get("source_file", ""),
                "file_type": "entity_fact",
                "date": (f.get("created_at", "") or "")[:10],
                "team": "",
                "namespace": "",
                "chunk_index": 0,
                "parent_id": None,
                "is_parent": False,
                "importance_score": 1.0 if retention == "permanent" else 0.7,
                "retention_class": f.get("retention_class", retention),
                "entity_name": entity["name"],
                "entity_id": eid,
                "valid_from": f.get("valid_from", ""),
                "valid_until": f.get("valid_until", ""),
            })

    if cap > 0 and len(results) > cap:
        results.sort(key=lambda r: r["score"], reverse=True)
        results = results[:cap]

    return results


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
