"""Context window management — token counting, budget checks, and message splitting.

Provides the logic behind the archivist_context_check MCP tool.
"""

import logging

from archivist.utils.tokenizer import count_message_tokens, count_tokens

logger = logging.getLogger("archivist.context")


def check_context(
    messages: list[dict],
    budget_tokens: int,
    reserve_from_tail: int = 2000,
) -> dict:
    """Analyze a message list against a token budget.

    Returns a dict with:
      - total_tokens: estimated token count of all messages
      - budget_tokens: the target budget
      - over_budget: True if total > budget
      - budget_used_pct: percentage of budget consumed
      - hint: "ok" | "compress" (>70%) | "critical" (>90%)
      - messages_to_compact: count of older messages that should be compacted
      - messages_to_keep: count of recent messages to preserve
      - compact_tokens: tokens in the compact group
      - keep_tokens: tokens in the keep group
    """
    if not messages:
        return {
            "total_tokens": 0,
            "budget_tokens": budget_tokens,
            "over_budget": False,
            "budget_used_pct": 0.0,
            "hint": "ok",
            "messages_to_compact": 0,
            "messages_to_keep": 0,
            "compact_tokens": 0,
            "keep_tokens": 0,
        }

    total = count_message_tokens(messages)
    pct = round(total / budget_tokens * 100, 1) if budget_tokens > 0 else 0.0

    if pct <= 70:
        hint = "ok"
    elif pct <= 90:
        hint = "compress"
    else:
        hint = "critical"

    compact_count = 0
    compact_toks = 0
    keep_toks = 0

    if total > budget_tokens:
        split_idx = _find_split_index(messages, reserve_from_tail)
        compact_count = split_idx
        compact_toks = _count_slice_tokens(messages[:split_idx])
        keep_toks = _count_slice_tokens(messages[split_idx:])

    return {
        "total_tokens": total,
        "budget_tokens": budget_tokens,
        "over_budget": total > budget_tokens,
        "budget_used_pct": pct,
        "hint": hint,
        "messages_to_compact": compact_count,
        "messages_to_keep": len(messages) - compact_count,
        "compact_tokens": compact_toks,
        "keep_tokens": keep_toks,
    }


def _find_split_index(messages: list[dict], reserve_tokens: int) -> int:
    """Walk backwards from the end, accumulating tokens until reserve is met.

    Returns the index where messages[:idx] should be compacted and
    messages[idx:] should be kept. Preserves complete user/assistant pairs
    by not splitting in the middle of a pair.
    """
    accumulated = 0
    split = len(messages)

    for i in range(len(messages) - 1, -1, -1):
        content = messages[i].get("content", "")
        if isinstance(content, str):
            msg_toks = count_tokens(content) + 4
        elif isinstance(content, list):
            msg_toks = 4
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    msg_toks += count_tokens(part.get("text", ""))
        else:
            msg_toks = 4
        if accumulated + msg_toks > reserve_tokens and i < len(messages) - 1:
            split = i + 1
            break
        accumulated += msg_toks
    else:
        split = 0

    if 0 < split < len(messages):
        if messages[split].get("role") == "assistant" and split > 0:
            split -= 1

    return split


def _count_slice_tokens(messages: list[dict]) -> int:
    """Count tokens in a sub-list of messages."""
    return count_message_tokens(messages) if messages else 0


def check_memories_budget(
    memory_texts: list[str],
    budget_tokens: int,
) -> dict:
    """Check a list of memory texts against a token budget.

    Simpler than check_context — just counts tokens and suggests compaction.
    """
    total = sum(count_tokens(t) for t in memory_texts)
    pct = round(total / budget_tokens * 100, 1) if budget_tokens > 0 else 0.0

    if pct <= 70:
        hint = "ok"
    elif pct <= 90:
        hint = "compress"
    else:
        hint = "critical"

    return {
        "total_tokens": total,
        "budget_tokens": budget_tokens,
        "over_budget": total > budget_tokens,
        "budget_used_pct": pct,
        "hint": hint,
        "memory_count": len(memory_texts),
    }
