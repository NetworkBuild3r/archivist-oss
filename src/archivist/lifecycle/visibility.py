"""Default-recall visibility helpers for suppress / supersede / delete.

INIT-003/SPEC-007 — public predicates consumed by SPEC-005 retrieval paths.

Default recall excludes:
  - suppressed rows (``is_suppressed``)
  - superseded losers (pointed-to by a winner ``supersedes_id``, or
    ``superseded_by`` / ``is_superseded`` on the row itself)
  - soft-deleted / excluded tombstones (``deleted`` / ``is_excluded``)
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

# SQL fragments for embedding in retrieval WHERE clauses (SPEC-005).
# Alias placeholders use {alias}; callers format with their table alias.

RECALL_VISIBLE_SQL_CHUNKS = (
    "({alias}.is_suppressed = 0 OR {alias}.is_suppressed IS NULL) "
    "AND ({alias}.is_excluded = 0 OR {alias}.is_excluded IS NULL) "
    "AND NOT EXISTS ("
    "SELECT 1 FROM memory_chunks _w "
    "WHERE _w.supersedes_id = {alias}.qdrant_id "
    "AND _w.namespace = {alias}.namespace "
    "AND (_w.is_excluded = 0 OR _w.is_excluded IS NULL)"
    ")"
)

RECALL_VISIBLE_SQL_FACTS = (
    "({alias}.is_suppressed = 0 OR {alias}.is_suppressed IS NULL) "
    "AND ({alias}.superseded_by IS NULL) "
    "AND ({alias}.is_active = 1 OR {alias}.is_active IS NULL)"
)


def recall_visible_sql_chunks(alias: str = "mc") -> str:
    """Return SQL AND-clause excluding suppressed + superseded losers for chunks."""
    return RECALL_VISIBLE_SQL_CHUNKS.format(alias=alias)


def recall_visible_sql_facts(alias: str = "f") -> str:
    """Return SQL AND-clause excluding suppressed + superseded facts."""
    return RECALL_VISIBLE_SQL_FACTS.format(alias=alias)


def _truthy(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int | float):
        return value != 0
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _get(row: Mapping[str, Any] | Any, *keys: str) -> Any:
    if isinstance(row, Mapping):
        for key in keys:
            if key in row:
                return row[key]
        return None
    for key in keys:
        if hasattr(row, key):
            return getattr(row, key)
    return None


def _row_id(row: Mapping[str, Any] | Any) -> str | None:
    value = _get(row, "qdrant_id", "memory_id", "id")
    if value is None or value == "":
        return None
    return str(value)


def is_recall_visible(
    row: Mapping[str, Any] | Any,
    *,
    include_suppressed: bool = False,
    include_superseded: bool = False,
    include_deleted: bool = False,
    known_superseded_ids: set[str] | frozenset[str] | None = None,
) -> bool:
    """Return whether *row* / payload should appear in default recall.

    Pure helper — no I/O. Pass ``known_superseded_ids`` (from
    ``list_superseded_loser_ids``) when the row itself lacks a loser flag
    but another winner points at it via ``supersedes_id``.
    """
    if not include_suppressed and _truthy(_get(row, "is_suppressed")):
        return False

    if not include_deleted and (_truthy(_get(row, "deleted")) or _truthy(_get(row, "is_excluded"))):
        return False

    if not include_superseded:
        if _truthy(_get(row, "is_superseded")):
            return False
        superseded_by = _get(row, "superseded_by")
        if superseded_by is not None and superseded_by != "" and superseded_by != 0:
            return False
        rid = _row_id(row)
        if known_superseded_ids is not None and rid is not None and rid in known_superseded_ids:
            return False

    return True


def filter_recall_visible(
    rows: Iterable[Mapping[str, Any] | Any],
    *,
    include_suppressed: bool = False,
    include_superseded: bool = False,
    include_deleted: bool = False,
    known_superseded_ids: set[str] | frozenset[str] | None = None,
) -> list[Any]:
    """Filter an iterable of rows/payloads to those visible under default recall."""
    return [
        row
        for row in rows
        if is_recall_visible(
            row,
            include_suppressed=include_suppressed,
            include_superseded=include_superseded,
            include_deleted=include_deleted,
            known_superseded_ids=known_superseded_ids,
        )
    ]
