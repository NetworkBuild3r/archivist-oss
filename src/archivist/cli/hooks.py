"""Zero-LLM deterministic passive hook runners (INIT-024/SPEC-009).

Provides `<100ms` local hook handlers for IDE lifecycle events:
- ``SessionStart``: read-only ~300-token compact project summary bootstrap
- ``PreCompact`` / ``Stop``: deterministic heuristic extraction of decisions,
  bug fixes, and user corrections — screened through the ``<5ms`` Memory
  Firewall with deterministic deduplication and ZERO synchronous LLM calls.
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

from archivist.app.handlers.scope_resolver import GLOBAL_SCOPE, resolve_write_scope
from archivist.security.trust_tiers import RequestContext
from archivist.storage.local_store import LocalMemoryStore

# Deterministic high-signal patterns for passive coding-session memory capture.
_CAPTURE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "decision",
        re.compile(
            r"(?i)(?:\bdecision:|\b(?:we decided|decided to|architecture decision|chosen approach|"
            r"standardize on)\b|\buse .+ instead of\b)"
        ),
    ),
    (
        "preference",
        re.compile(
            r"(?i)(?:\b(?:correction|convention):|\b(?:always use|never use|please always|"
            r"don't use|do not use)\b|\bprefer .+ over\b|\bactually,)"
        ),
    ),
    (
        "tip",
        re.compile(
            r"(?i)(?:\b(?:workaround|gotcha|caveat|pitfall):|\b(?:root cause|fixed by|"
            r"lesson learned|watch out for)\b)"
        ),
    ),
)

_MAX_CANDIDATES_PER_HOOK = 12
_MIN_CANDIDATE_CHARS = 24
_MAX_CANDIDATE_CHARS = 800


def extract_passive_candidates(raw_payload: str) -> list[tuple[str, str]]:
    """Deterministically extract ``(kind, sentence)`` tuples from transcript/hook input.

    Accepts either plain text/markdown or JSON hook payloads (Cursor/Claude Code
    ``transcript``/``messages``/``content`` fields). Performs ZERO LLM calls.
    """
    text = (raw_payload or "").strip()
    if not text:
        return []

    # Unwrap JSON hook payloads if provided on stdin.
    if text.startswith(("{", "[")):
        try:
            parsed = json.loads(text)
            chunks: list[str] = []
            if isinstance(parsed, dict):
                for key in ("content", "text", "summary", "transcript", "last_assistant_message"):
                    val = parsed.get(key)
                    if isinstance(val, str) and val.strip():
                        chunks.append(val.strip())
                msgs = parsed.get("messages")
                if isinstance(msgs, list):
                    for msg in msgs:
                        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
                            chunks.append(msg["content"].strip())
            elif isinstance(parsed, list):
                for item in parsed:
                    if isinstance(item, str):
                        chunks.append(item.strip())
                    elif isinstance(item, dict) and isinstance(item.get("content"), str):
                        chunks.append(item["content"].strip())
            if chunks:
                text = "\n".join(chunks)
        except json.JSONDecodeError:
            pass

    lines = [ln.strip().lstrip("-*• ").strip() for ln in text.splitlines() if ln.strip()]
    results: list[tuple[str, str]] = []
    seen: set[str] = set()

    for line in lines:
        if len(line) < _MIN_CANDIDATE_CHARS:
            continue
        snippet = line[:_MAX_CANDIDATE_CHARS]
        for kind, pat in _CAPTURE_PATTERNS:
            if pat.search(snippet):
                norm_key = snippet.lower()
                if norm_key not in seen:
                    seen.add(norm_key)
                    results.append((kind, snippet))
                break
        if len(results) >= _MAX_CANDIDATES_PER_HOOK:
            break

    return results


def run_session_start_hook(
    store: LocalMemoryStore,
    *,
    scope: str | None = None,
    max_tokens: int = 300,
) -> dict[str, Any]:
    """Read-only ``SessionStart`` hook returning a compact ``~300``-token summary."""
    t0 = time.perf_counter()
    resolved_scope = resolve_write_scope(scope)
    scopes = [resolved_scope]
    if resolved_scope != GLOBAL_SCOPE:
        scopes.append(GLOBAL_SCOPE)

    ctx = store.build_context(
        "project architecture decisions conventions gotchas preferences",
        scopes=scopes,
        max_tokens=max_tokens,
        limit=6,
    )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "hook": "SessionStart",
        "scope": resolved_scope,
        "summary": ctx["context"],
        "sources_count": len(ctx["sources"]),
        "total_tokens": ctx["total_tokens"],
        "budget_tokens": max_tokens,
        "elapsed_ms": elapsed_ms,
        "llm_calls": 0,
    }


def run_passive_capture_hook(
    store: LocalMemoryStore,
    *,
    hook_name: str,
    payload_text: str,
    scope: str | None = None,
    ctx: RequestContext | None = None,
) -> dict[str, Any]:
    """Deterministic zero-LLM ``PreCompact`` / ``Stop`` capture through Memory Firewall."""
    t0 = time.perf_counter()
    resolved_scope = resolve_write_scope(scope)
    candidates = extract_passive_candidates(payload_text)

    saved_ids: list[str] = []
    dedup_count = 0
    redacted_total = 0
    quarantined_count = 0

    for kind, text in candidates:
        saved = store.save_memory(
            text,
            scope=resolved_scope,
            kind=kind,
            tags=["passive-hook", hook_name.lower()],
            ctx=ctx,
            deduplicate=True,
        )
        if saved.deduplicated:
            dedup_count += 1
        else:
            saved_ids.append(saved.memory_id)
        redacted_total += saved.redacted_count
        if saved.quarantined:
            quarantined_count += 1

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        "hook": hook_name,
        "scope": resolved_scope,
        "candidates_extracted": len(candidates),
        "saved_count": len(saved_ids),
        "saved_ids": saved_ids,
        "deduplicated_count": dedup_count,
        "redacted_count": redacted_total,
        "quarantined_count": quarantined_count,
        "elapsed_ms": elapsed_ms,
        "llm_calls": 0,
    }
