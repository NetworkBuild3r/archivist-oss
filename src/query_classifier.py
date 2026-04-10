"""Inventory-aware memory_type classification for search queries (v1.6).

v1.10: extended with subcategory classification and query-only heuristic
fallback when namespace inventory is unavailable.
"""

from __future__ import annotations

import hashlib
import logging
import re
import threading
import time

import llm as llm_mod
from namespace_inventory import NamespaceInventory

logger = logging.getLogger("archivist.query_classifier")

VALID_TYPES = ("experience", "skill", "general")

VALID_SUBCATEGORIES: dict[str, tuple[str, ...]] = {
    "experience": ("incident", "session", "timeline", "change", "deployment"),
    "skill": ("procedure", "command", "playbook", "architecture", "config"),
    "general": ("definition", "reference", "opinion", "status"),
}

SUBCATEGORY_TO_TOPIC: dict[str, str] = {
    "deployment": "cicd",
    "incident": "incident",
    "architecture": "architecture",
    "config": "architecture",
    "command": "cicd",
    "playbook": "cicd",
}

CLASSIFIER_CACHE_TTL_SECONDS = 120
_CLASSIFIER_CACHE_MAX_ENTRIES = 2048
_lock = threading.Lock()
_classifier_cache: dict[tuple[str, str], tuple[float, str, str]] = {}


def _sweep_expired_locked(now: float) -> None:
    """Remove expired entries under the lock. Called when the cache is too large."""
    expired = [k for k, (ts, _, _) in _classifier_cache.items()
               if now - ts > CLASSIFIER_CACHE_TTL_SECONDS]
    for k in expired:
        del _classifier_cache[k]
    if len(_classifier_cache) > _CLASSIFIER_CACHE_MAX_ENTRIES:
        # Still too large after sweeping expired: drop oldest half
        by_age = sorted(_classifier_cache.items(), key=lambda kv: kv[1][0])
        for k, _ in by_age[:len(by_age) // 2]:
            del _classifier_cache[k]


def _query_hash(query: str) -> str:
    return hashlib.sha256(query.strip().encode()).hexdigest()[:32]


def _invalidate_classifier_cache_for_namespace(namespace: str) -> None:
    with _lock:
        keys = [k for k in _classifier_cache if k[0] == namespace]
        for k in keys:
            del _classifier_cache[k]


def invalidate_cache(namespace: str) -> None:
    """Call when namespace memory set changes (same events as inventory)."""
    _invalidate_classifier_cache_for_namespace(namespace)


def invalidate_all_cache() -> None:
    with _lock:
        _classifier_cache.clear()


_CLASSIFY_SYSTEM_V2 = (
    "You classify a search query for an agent memory system.\n"
    "Step 1: Pick the primary type — experience | skill | general | unknown\n"
    "Step 2: Pick a subcategory from this list:\n"
    "  experience: incident, session, timeline, change, deployment\n"
    "  skill: procedure, command, playbook, architecture, config\n"
    "  general: definition, reference, opinion, status\n"
    "Return exactly: type/subcategory (e.g. experience/incident)\n"
    "If unsure of subcategory, return type only (e.g. experience)\n"
    "Return only the classification, no punctuation or explanation."
)

# ---------------------------------------------------------------------------
# Lightweight regex-based subcategory heuristics (no LLM)
# ---------------------------------------------------------------------------

_SUBCATEGORY_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"\b(?:incident|outage|downtime|postmortem|on-call)\b", re.I), "experience", "incident"),
    (re.compile(r"\b(?:deploy|deployment|release|rollout|shipped)\b", re.I), "experience", "deployment"),
    (re.compile(r"\b(?:session|meeting|standup|retro)\b", re.I), "experience", "session"),
    (re.compile(r"\b(?:timeline|chronolog|history|what happened)\b", re.I), "experience", "timeline"),
    (re.compile(r"\b(?:changed|migration|upgrade|switch)\b", re.I), "experience", "change"),
    (re.compile(r"\b(?:how to|how do|procedure|steps to|guide|tutorial)\b", re.I), "skill", "procedure"),
    (re.compile(r"\b(?:command|cli|kubectl|docker|run)\b", re.I), "skill", "command"),
    (re.compile(r"\b(?:playbook|runbook|checklist)\b", re.I), "skill", "playbook"),
    (re.compile(r"\b(?:architecture|design|pattern|diagram)\b", re.I), "skill", "architecture"),
    (re.compile(r"\b(?:config|configuration|setting|parameter|env)\b", re.I), "skill", "config"),
    (re.compile(r"\b(?:what is|definition|meaning|explain)\b", re.I), "general", "definition"),
    (re.compile(r"\b(?:status|current|state|health)\b", re.I), "general", "status"),
]


def classify_query_heuristic(query: str) -> tuple[str, str]:
    """Fast regex-based classification without LLM. Returns (type, subcategory)."""
    from collections import Counter
    type_votes: Counter = Counter()
    sub_votes: Counter = Counter()

    for pat, ptype, psub in _SUBCATEGORY_PATTERNS:
        hits = len(pat.findall(query))
        if hits:
            type_votes[ptype] += hits
            sub_votes[psub] += hits

    if not type_votes:
        return ("", "")

    best_type = type_votes.most_common(1)[0][0]
    best_sub = sub_votes.most_common(1)[0][0]

    valid_subs = VALID_SUBCATEGORIES.get(best_type, ())
    if best_sub not in valid_subs:
        best_sub = ""

    return (best_type, best_sub)


async def classify_query(
    query: str,
    inventory: NamespaceInventory | None = None,
) -> str:
    """Return memory_type to filter on, or "" to search all types.

    Uses inventory counts when available: skips small namespaces, uses skew
    heuristic, then optional LLM.  Falls back to regex heuristics when no
    inventory is provided.
    """
    if inventory is not None:
        total = inventory.total_memories
        if total < 50:
            return ""

        by = {k: v for k, v in inventory.by_type.items() if v > 0}
        if not by:
            return ""

        if len(by) == 1:
            return next(iter(by.keys()))

        max_type = max(by, key=by.get)
        max_count = by[max_type]
        if max_count / total >= 0.90:
            return max_type

        ns = inventory.namespace
    else:
        by = {}
        ns = ""

    qhash = _query_hash(query)
    cache_key = (ns, qhash)
    now = time.monotonic()
    with _lock:
        ent = _classifier_cache.get(cache_key)
        if ent is not None:
            ts, val, _sub = ent
            if now - ts <= CLASSIFIER_CACHE_TTL_SECONDS:
                return val

    if inventory is not None and by:
        dist = ", ".join(f"{k}: {v}" for k, v in sorted(by.items()))
        prompt = (
            f"Namespace memory counts — {dist}\n"
            f"Total chunks: {inventory.total_memories}\n\n"
            f"Query: {query[:2000]}\n"
        )

        try:
            raw = await llm_mod.llm_query(prompt, system=_CLASSIFY_SYSTEM_V2, max_tokens=32)
        except Exception as e:
            logger.warning("Query classification LLM failed: %s", e)
            return ""

        token = _parse_classification(raw)
        primary = token[0] if token else ""
        subcategory = token[1] if len(token) > 1 else ""

        if primary not in VALID_TYPES:
            return ""

        if by.get(primary, 0) == 0:
            return ""

        with _lock:
            _classifier_cache[cache_key] = (now, primary, subcategory)
            if len(_classifier_cache) > _CLASSIFIER_CACHE_MAX_ENTRIES:
                _sweep_expired_locked(now)

        return primary
    else:
        primary, _sub = classify_query_heuristic(query)
        if primary and primary in VALID_TYPES:
            with _lock:
                _classifier_cache[cache_key] = (now, primary, _sub)
                if len(_classifier_cache) > _CLASSIFIER_CACHE_MAX_ENTRIES:
                    _sweep_expired_locked(now)
            return primary
        return ""


async def classify_query_full(
    query: str,
    inventory: NamespaceInventory | None = None,
) -> tuple[str, str]:
    """Return (memory_type, subcategory) for a query.

    Like ``classify_query`` but also returns the subcategory for
    downstream routing (e.g. subcategory→topic mapping).
    """
    if inventory is not None:
        total = inventory.total_memories
        if total < 50:
            ht, hs = classify_query_heuristic(query)
            return (ht, hs) if ht in VALID_TYPES else ("", "")

        by = {k: v for k, v in inventory.by_type.items() if v > 0}
        if not by:
            ht, hs = classify_query_heuristic(query)
            return (ht, hs) if ht in VALID_TYPES else ("", "")
        ns = inventory.namespace
    else:
        by = {}
        ns = ""

    qhash = _query_hash(query)
    cache_key = (ns, qhash)
    now = time.monotonic()
    with _lock:
        ent = _classifier_cache.get(cache_key)
        if ent is not None:
            ts, val, sub = ent
            if now - ts <= CLASSIFIER_CACHE_TTL_SECONDS:
                return (val, sub)

    if inventory is not None and by:
        dist = ", ".join(f"{k}: {v}" for k, v in sorted(by.items()))
        prompt = (
            f"Namespace memory counts — {dist}\n"
            f"Total chunks: {inventory.total_memories}\n\n"
            f"Query: {query[:2000]}\n"
        )

        try:
            raw = await llm_mod.llm_query(prompt, system=_CLASSIFY_SYSTEM_V2, max_tokens=32)
        except Exception as e:
            logger.warning("Query classification LLM failed: %s", e)
            return classify_query_heuristic(query)

        token = _parse_classification(raw)
        primary = token[0] if token else ""
        subcategory = token[1] if len(token) > 1 else ""

        if primary not in VALID_TYPES:
            return classify_query_heuristic(query)

        if by.get(primary, 0) == 0:
            return classify_query_heuristic(query)

        with _lock:
            _classifier_cache[cache_key] = (now, primary, subcategory)
            if len(_classifier_cache) > _CLASSIFIER_CACHE_MAX_ENTRIES:
                _sweep_expired_locked(now)

        return (primary, subcategory)
    else:
        ht, hs = classify_query_heuristic(query)
        if ht and ht in VALID_TYPES:
            with _lock:
                _classifier_cache[cache_key] = (now, ht, hs)
                if len(_classifier_cache) > _CLASSIFIER_CACHE_MAX_ENTRIES:
                    _sweep_expired_locked(now)
            return (ht, hs)
        return ("", "")


def _parse_classification(raw: str | None) -> list[str]:
    """Parse LLM response like 'experience/incident' into ['experience', 'incident']."""
    if not raw:
        return []
    cleaned = re.sub(r"[^a-z/_ ]", "", raw.strip().lower())
    parts = cleaned.split("/", 1)
    parts = [re.sub(r"[^a-z_]", "", p.split()[0] if p.split() else "") for p in parts]
    return [p for p in parts if p]
