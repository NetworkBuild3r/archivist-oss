"""Thin facade over the split graph storage modules (INIT-001/SPEC-003).

``storage/graph.py`` used to be a ~1800 LOC monolith mixing schema init,
entity/fact CRUD, FTS helpers, needle-registry, and Qdrant-point bookkeeping.
It is now decomposed into focused modules:

- :mod:`archivist.storage.graph_schema` — connection helpers, DDL, migrations
- :mod:`archivist.storage.graph_fts` — FTS5/tsvector chunk upsert, delete, search
- :mod:`archivist.storage.graph_entities` — entity/relationship/fact CRUD
- :mod:`archivist.storage.graph_needles` — deterministic needle-token registry
- :mod:`archivist.storage.graph_points` — hotness cleanup, Qdrant point tracking

This module re-exports every public and internal name so existing callers
(``from archivist.storage.graph import X`` / ``import archivist.storage.graph
as graph``) keep working unchanged for at least one release cycle (und-1,
aud-1). New code should prefer importing directly from the focused module.

Provenance: INIT-001/SPEC-003.
"""

from __future__ import annotations

import logging

# Kept importable as ``graph.m`` for backward-compat test monkeypatching of
# ``archivist.storage.graph.m.observe`` / ``.inc`` — same shared module object
# as ``graph_fts.m``, so patches here are visible to graph_fts's internals too.
import archivist.core.metrics as m  # noqa: F401

# Kept importable as ``graph.SQLITE_PATH`` for backward-compat test
# monkeypatching; production code should read it from ``archivist.core.config``
# (get_db()/_ensure_dir() in graph_schema.py already do so at call time).
from archivist.core.config import SQLITE_PATH
from archivist.storage.graph_entities import (
    _DATE_IN_PATH_RE,
    _RETENTION_RANK,
    _normalize,
    _word_set,
    add_entity_alias,
    add_fact,
    add_relationship,
    get_curator_state,
    get_entity_by_id,
    get_entity_facts,
    get_entity_facts_bulk,
    get_entity_relationships,
    get_entity_relationships_bulk,
    invalidate_fact,
    search_entities,
    set_curator_state,
    supersede_fact,
    upsert_entity,
)
from archivist.storage.graph_fts import (
    _build_fts_where,
    _delete_fts_rows_async,
    _run_fts_query,
    _search_fts_exact_postgres,
    _search_fts_exact_sqlite,
    _search_fts_postgres,
    _search_fts_postgres_family,
    _search_fts_sqlite,
    _search_fts_sqlite_family,
    _upsert_fts_chunk_postgres,
    _upsert_fts_chunk_sqlite,
    delete_fts_chunks_batch,
    delete_fts_chunks_by_file,
    delete_fts_chunks_by_qdrant_id,
    search_fts,
    search_fts_exact,
    set_fts_excluded_batch,
    upsert_fts_chunk,
)
from archivist.storage.graph_needles import (
    _ensure_needle_registry,
    delete_needle_tokens_batch,
    delete_needle_tokens_by_memory,
    lookup_needle_tokens,
    register_needle_tokens,
)
from archivist.storage.graph_points import (
    delete_hotness,
    delete_memory_points,
    log_delete_failure,
    lookup_memory_points,
    register_memory_points_batch,
)
from archivist.storage.graph_schema import (
    _BATCH_CHUNK,
    GRAPH_WRITE_LOCK,
    _ensure_dir,
    _init_fts5,
    _is_postgres,
    _migrate_entity_unique_constraint,
    _migrate_schema,
    db_conn,
    get_db,
    init_schema,
    init_schema_async,
    schema_guard,
)
from archivist.utils.chunking import NEEDLE_PATTERNS

logger = logging.getLogger("archivist.graph")

__all__ = [
    "GRAPH_WRITE_LOCK",
    "NEEDLE_PATTERNS",
    "SQLITE_PATH",
    "_BATCH_CHUNK",
    "_DATE_IN_PATH_RE",
    "_RETENTION_RANK",
    "_build_fts_where",
    "_delete_fts_rows_async",
    "_ensure_dir",
    "_ensure_needle_registry",
    "_init_fts5",
    "_is_postgres",
    "_migrate_entity_unique_constraint",
    "_migrate_schema",
    "_normalize",
    "_run_fts_query",
    "_search_fts_exact_postgres",
    "_search_fts_exact_sqlite",
    "_search_fts_postgres",
    "_search_fts_postgres_family",
    "_search_fts_sqlite",
    "_search_fts_sqlite_family",
    "_upsert_fts_chunk_postgres",
    "_upsert_fts_chunk_sqlite",
    "_word_set",
    "add_entity_alias",
    "add_fact",
    "add_relationship",
    "db_conn",
    "delete_fts_chunks_batch",
    "delete_fts_chunks_by_file",
    "delete_fts_chunks_by_qdrant_id",
    "delete_hotness",
    "delete_memory_points",
    "delete_needle_tokens_batch",
    "delete_needle_tokens_by_memory",
    "get_curator_state",
    "get_db",
    "get_entity_by_id",
    "get_entity_facts",
    "get_entity_facts_bulk",
    "get_entity_relationships",
    "get_entity_relationships_bulk",
    "init_schema",
    "init_schema_async",
    "invalidate_fact",
    "log_delete_failure",
    "lookup_memory_points",
    "lookup_needle_tokens",
    "register_memory_points_batch",
    "register_needle_tokens",
    "schema_guard",
    "search_entities",
    "search_fts",
    "search_fts_exact",
    "set_curator_state",
    "set_fts_excluded_batch",
    "supersede_fact",
    "upsert_entity",
    "upsert_fts_chunk",
]
