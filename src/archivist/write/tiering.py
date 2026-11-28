"""Tiered context abstraction — L0 (abstract), L1 (overview), L2 (full text).

On ingest, optionally generates compact L0 (~100 tokens) and L1 (~500 tokens)
summaries stored alongside the full L2 text. When ``TIERED_CONTEXT_ENABLED`` is
False (the v3 zero-LLM default), deterministic extractive helpers produce L0
headlines and query-centered L1 windows with zero LLM/network calls.
"""

from __future__ import annotations

import logging
import math
import re

from archivist.core.config import L0_MAX_TOKENS, L1_MAX_TOKENS, TIERED_CONTEXT_ENABLED
from archivist.features.llm import llm_query
from archivist.utils.tokenizer import count_tokens

logger = logging.getLogger("archivist.tiering")

_L0_SYSTEM = (
    "Summarize the following text in one short sentence (~20 words). "
    "Capture the single most important idea. Return ONLY the sentence, no preamble."
)

_L1_SYSTEM = (
    "Produce a concise overview of the following text in 2-4 sentences. "
    "Include key entities, dates, and actionable facts. "
    "Return ONLY the overview, no preamble or markdown."
)

_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_TOKEN_RE = re.compile(r"[\w./:-]{2,}")
_HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s+")

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "did",
        "do",
        "does",
        "for",
        "from",
        "has",
        "have",
        "how",
        "in",
        "is",
        "it",
        "of",
        "on",
        "or",
        "that",
        "the",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "with",
    }
)


def _truncate_chars_to_tokens(text: str, max_tokens: int) -> str:
    """Binary-search a character prefix whose token count is ``<= max_tokens``."""
    if not text or max_tokens <= 0:
        return ""
    lo, hi = 1, len(text)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid].strip()
        if not candidate or count_tokens(candidate) <= max_tokens:
            if candidate:
                best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate *text* on a word boundary so ``count_tokens(result) <= max_tokens``."""
    cleaned = text.strip()
    if not cleaned or max_tokens <= 0:
        return ""
    if count_tokens(cleaned) <= max_tokens:
        return cleaned

    words = cleaned.split()
    if not words:
        return _truncate_chars_to_tokens(cleaned, max_tokens)

    lo, hi = 1, len(words)
    best = ""
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = " ".join(words[:mid])
        if count_tokens(candidate) <= max_tokens:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    if best:
        return best
    # Single word/run (e.g., CJK or emoji) exceeds max_tokens — binary-search characters.
    return _truncate_chars_to_tokens(cleaned, max_tokens)


def extract_l0_deterministic(text: str, max_tokens: int = 100) -> str:
    """Extract a deterministic, zero-LLM L0 headline/first-sentence summary.

    Uses the first markdown heading and/or first sentence, capped strictly at
    *max_tokens*. Performs zero network or LLM calls.
    """
    cleaned = (text or "").strip()
    if not cleaned or max_tokens <= 0:
        return ""

    lines = [ln.strip() for ln in cleaned.splitlines() if ln.strip()]
    if not lines:
        return ""

    first_line = lines[0]
    is_heading = bool(_HEADING_PREFIX_RE.match(first_line))
    heading = _HEADING_PREFIX_RE.sub("", first_line).strip() if is_heading else ""

    if is_heading and len(lines) > 1:
        rest = " ".join(lines[1:])
        sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(rest) if s.strip()]
        lead = f"{heading}: {sentences[0]}" if sentences else heading
        return _truncate_to_tokens(lead, max_tokens)

    sentences = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
    first = _HEADING_PREFIX_RE.sub("", sentences[0]).strip() if sentences else cleaned
    return _truncate_to_tokens(first, max_tokens)


def _extract_query_terms(query: str) -> list[str]:
    raw = [tok.lower() for tok in _TOKEN_RE.findall(query or "")]
    if not raw:
        return []
    filtered = [tok for tok in raw if tok not in _STOPWORDS]
    return filtered or raw


def _window_around_single_segment(segment: str, terms: set[str], max_tokens: int) -> str:
    """Slice a single long segment around the first matching query term, keeping the anchor."""
    words = segment.split()
    if not words or max_tokens <= 0:
        return ""
    anchor = 0
    if terms:
        for idx, word in enumerate(words):
            low = word.lower()
            if any(t in low for t in terms):
                anchor = idx
                break

    if count_tokens(words[anchor]) > max_tokens:
        return _truncate_to_tokens(words[anchor], max_tokens)

    # Expand outward symmetrically around *anchor* while staying strictly within max_tokens.
    left = anchor
    right = anchor
    while True:
        expanded = False
        if right + 1 < len(words):
            cand = " ".join(words[left : right + 2])
            if count_tokens(cand) <= max_tokens:
                right += 1
                expanded = True
        if left - 1 >= 0:
            cand = " ".join(words[left - 1 : right + 1])
            if count_tokens(cand) <= max_tokens:
                left -= 1
                expanded = True
        if not expanded:
            break
    return " ".join(words[left : right + 1])


def extract_l1_query_window(text: str, query: str, max_tokens: int = 300) -> str:
    """Extract a deterministic BM25/term-centered L1 snippet around query terms.

    When *query* is empty or has no term overlap with *text*, returns leading
    sentences up to *max_tokens*. Performs zero network or LLM calls.
    """
    cleaned = (text or "").strip()
    if not cleaned or max_tokens <= 0:
        return ""
    if count_tokens(cleaned) <= max_tokens:
        return cleaned

    segments = [s.strip() for s in _SENTENCE_SPLIT_RE.split(cleaned) if s.strip()]
    if not segments:
        return _truncate_to_tokens(cleaned, max_tokens)

    q_terms = _extract_query_terms(query)
    if len(segments) == 1:
        return _window_around_single_segment(segments[0], set(q_terms), max_tokens)

    if not q_terms:
        chosen: list[str] = []
        used = 0
        for seg in segments:
            seg_toks = count_tokens(seg)
            if chosen and used + seg_toks > max_tokens:
                break
            if not chosen and seg_toks > max_tokens:
                return _truncate_to_tokens(seg, max_tokens)
            chosen.append(seg)
            used += seg_toks
        return _truncate_to_tokens(" ".join(chosen), max_tokens)

    # Document-internal IDF + BM25-like TF saturation across segments.
    q_unique = set(q_terms)
    seg_tokens_list = [[tok.lower() for tok in _TOKEN_RE.findall(seg)] for seg in segments]
    n_segs = len(segments)
    df: dict[str, int] = {}
    for toks in seg_tokens_list:
        for t in set(toks) & q_unique:
            df[t] = df.get(t, 0) + 1

    query_lower = (query or "").strip().lower()
    scores: list[float] = []
    for seg, toks in zip(segments, seg_tokens_list):
        if not toks:
            scores.append(0.0)
            continue
        tf_counts: dict[str, int] = {}
        for tok in toks:
            if tok in q_unique:
                tf_counts[tok] = tf_counts.get(tok, 0) + 1
        score = 0.0
        for term, tf in tf_counts.items():
            idf = math.log(1.0 + (n_segs / (1.0 + df.get(term, 0))))
            score += idf * (tf / (tf + 1.2))
        if len(query_lower) >= 4 and query_lower in seg.lower():
            score += 2.0
        scores.append(score)

    best_idx = max(range(n_segs), key=lambda i: scores[i])
    if scores[best_idx] <= 0.0:
        # No query terms matched — return leading sentences.
        return extract_l1_query_window(cleaned, "", max_tokens=max_tokens)

    anchor_toks = count_tokens(segments[best_idx])
    if anchor_toks > max_tokens:
        return _window_around_single_segment(segments[best_idx], q_unique, max_tokens)

    # Expand a contiguous window around the highest-scoring anchor sentence
    # in O(N) using cached per-segment token counts rather than re-tokenizing
    # the growing joined prefix on every step.
    seg_tok_cache: dict[int, int] = {best_idx: anchor_toks}

    def _seg_tokens(idx: int) -> int:
        cached = seg_tok_cache.get(idx)
        if cached is None:
            cached = count_tokens(segments[idx])
            seg_tok_cache[idx] = cached
        return cached

    left = best_idx
    right = best_idx
    used_tokens = anchor_toks
    while True:
        expanded = False
        can_right = right + 1 < n_segs
        can_left = left - 1 >= 0
        # Prefer expanding toward the higher-scoring neighbor.
        neighbors: list[tuple[float, str]] = []
        if can_right:
            neighbors.append((scores[right + 1], "right"))
        if can_left:
            neighbors.append((scores[left - 1], "left"))
        neighbors.sort(key=lambda item: item[0], reverse=True)

        for _, direction in neighbors:
            next_idx = right + 1 if direction == "right" else left - 1
            # +1 token headroom for the joining space between segments.
            added = _seg_tokens(next_idx) + 1
            if used_tokens + added <= max_tokens:
                if direction == "right":
                    right = next_idx
                else:
                    left = next_idx
                used_tokens += added
                expanded = True
                break
        if not expanded:
            break

    return _truncate_to_tokens(" ".join(segments[left : right + 1]), max_tokens)


async def generate_tiers(text: str) -> dict:
    """Return {"l0": str, "l1": str, "l2": str} for a chunk of text.

    When tiering is disabled or LLM fails, l0 and l1 use deterministic
    zero-LLM extractive summaries.
    """
    l2 = text

    if not TIERED_CONTEXT_ENABLED or len(text.strip()) < 80:
        return {
            "l0": extract_l0_deterministic(text, max_tokens=L0_MAX_TOKENS),
            "l1": extract_l1_query_window(text, "", max_tokens=L1_MAX_TOKENS),
            "l2": l2,
        }

    try:
        l0 = await llm_query(text[:3000], system=_L0_SYSTEM, max_tokens=60)
        l0 = l0.strip()
    except Exception as e:
        logger.warning("L0 generation failed, falling back to deterministic L0: %s", e)
        l0 = extract_l0_deterministic(text, max_tokens=L0_MAX_TOKENS)

    try:
        l1 = await llm_query(text[:4000], system=_L1_SYSTEM, max_tokens=200)
        l1 = l1.strip()
    except Exception as e:
        logger.warning("L1 generation failed, falling back to deterministic L1: %s", e)
        l1 = extract_l1_query_window(text, "", max_tokens=L1_MAX_TOKENS)

    return {"l0": l0, "l1": l1, "l2": l2}


def select_tier(hit: dict, tier: str = "l2", query: str = "") -> str:
    """Extract the requested tier text from a search hit's payload.

    Falls back deterministically to extractive L0/L1 from full text (``l2`` or
    ``text``) when pre-computed tier text is absent.
    """
    full_text = hit.get("l2") or hit.get("text", "")
    if tier == "l0":
        return hit.get("l0") or extract_l0_deterministic(full_text, max_tokens=L0_MAX_TOKENS)
    if tier == "l1":
        return hit.get("l1") or extract_l1_query_window(full_text, query, max_tokens=L1_MAX_TOKENS)
    return full_text
