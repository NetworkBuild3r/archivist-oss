"""Inventory-aware memory_type classification for search queries (v1.6)."""

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

CLASSIFIER_CACHE_TTL_SECONDS = 120
_lock = threading.Lock()
_classifier_cache: dict[tuple[str, str], tuple[float, str]] = {}


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


_CLASSIFY_SYSTEM = (
    "You classify a search query for an agent memory system. "
    "Given the distribution of memory types in this namespace, pick the single best type "
    "for retrieval filtering.\n"
    "Reply with exactly one token, lowercase: experience | skill | general | unknown.\n"
    "- experience: past events, incidents, what happened, sessions, timelines\n"
    "- skill: how-to, procedures, commands, playbooks, reusable know-how\n"
    "- general: definitions, notes, mixed or unclear\n"
    "- unknown: cannot decide\n"
    "Return only the token, no punctuation or explanation."
)


async def classify_query(query: str, inventory: NamespaceInventory) -> str:
    """
    Return memory_type to filter on, or "" to search all types.

    Uses inventory counts: skips small namespaces, uses skew heuristic, then optional LLM.
    """
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

    qhash = _query_hash(query)
    cache_key = (inventory.namespace, qhash)
    now = time.monotonic()
    with _lock:
        ent = _classifier_cache.get(cache_key)
        if ent is not None:
            ts, val = ent
            if now - ts <= CLASSIFIER_CACHE_TTL_SECONDS:
                return val

    dist = ", ".join(f"{k}: {v}" for k, v in sorted(by.items()))
    prompt = (
        f"Namespace memory counts — {dist}\n"
        f"Total chunks: {total}\n\n"
        f"Query: {query[:2000]}\n"
    )

    try:
        raw = await llm_mod.llm_query(prompt, system=_CLASSIFY_SYSTEM, max_tokens=16)
    except Exception as e:
        logger.warning("Query classification LLM failed: %s", e)
        return ""

    token = (raw or "").strip().lower()
    token = re.sub(r"[^a-z_]", "", token.split()[0] if token else "")
    if token not in VALID_TYPES:
        return ""

    if by.get(token, 0) == 0:
        return ""

    with _lock:
        _classifier_cache[cache_key] = (now, token)

    return token
