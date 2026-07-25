"""Memory/entity lineage aggregation for observability (INIT-001/SPEC-011).

Builds lineage edges from provenance (source_trace), versioning, audit_log,
and retrieval_logs. Payloads intentionally omit secrets and full memory text.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from archivist.storage.sqlite_pool import pool

logger = logging.getLogger("archivist.lineage")

# Memory ids are UUID-ish / opaque tokens — reject path/injection shapes.
_SAFE_MEMORY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:@-]{0,127}$")
# Entity names may include spaces; still reject control chars / path separators.
_SAFE_ENTITY_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 ._/@:+-]{0,127}$")


def validate_memory_id(memory_id: str) -> str | None:
    """Return a normalized memory id, or ``None`` when invalid."""
    rid = (memory_id or "").strip()
    if not rid or not _SAFE_MEMORY_ID_RE.match(rid):
        return None
    return rid


def _escape_like(value: str) -> str:
    """Escape LIKE metacharacters so an id can never act as a search pattern.

    ``_`` is a legal memory-id character but is also a single-character LIKE
    wildcard, which would otherwise let a caller enumerate rows they never
    named (INIT-001/SPEC-012, SEC-012-04).
    """
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def validate_entity_id(entity_id: str) -> str | None:
    """Return a normalized entity id/name, or ``None`` when invalid."""
    rid = (entity_id or "").strip()
    if not rid or not _SAFE_ENTITY_ID_RE.match(rid):
        return None
    if "/" in rid or "\\" in rid or ".." in rid:
        return None
    return rid


# Backward-compatible alias used by early drafts / tests.
validate_resource_id = validate_memory_id


def _edge(
    *,
    edge_type: str,
    from_id: str,
    to_id: str,
    actor_id: str = "",
    actor_type: str = "",
    timestamp: str = "",
    relation: str = "",
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "edge_type": edge_type,
        "from_id": from_id,
        "to_id": to_id,
        "actor_id": actor_id or "",
        "actor_type": actor_type or "",
        "timestamp": timestamp or "",
        "relation": relation or edge_type,
        "metadata": metadata or {},
    }


async def resolve_memory_namespace(memory_id: str) -> str:
    """Best-effort namespace for RBAC from SQLite, then Qdrant payload."""
    async with pool.read() as conn:
        row = await conn.fetchone(
            "SELECT namespace FROM memory_chunks WHERE qdrant_id = ? LIMIT 1",
            (memory_id,),
        )
        if row and row["namespace"]:
            return str(row["namespace"])

        audit = await conn.fetchone(
            "SELECT namespace FROM audit_log WHERE memory_id = ? "
            "AND namespace IS NOT NULL AND namespace != '' "
            "ORDER BY timestamp DESC LIMIT 1",
            (memory_id,),
        )
        if audit and audit["namespace"]:
            return str(audit["namespace"])

    try:
        import asyncio

        from archivist.core.config import QDRANT_COLLECTION
        from archivist.storage.qdrant import qdrant_client

        client = qdrant_client()
        points = await asyncio.to_thread(
            client.retrieve,
            collection_name=QDRANT_COLLECTION,
            ids=[memory_id],
            with_payload=True,
        )
        if points:
            ns = (points[0].payload or {}).get("namespace") or ""
            return str(ns)
    except Exception as e:
        logger.debug("lineage.resolve_memory_namespace qdrant miss: %s", e)

    return ""


async def _provenance_edges(memory_id: str) -> list[dict[str, Any]]:
    """Source-trace edges from Qdrant payload (allowlisted keys only)."""
    edges: list[dict[str, Any]] = []
    try:
        import asyncio

        from archivist.core.config import QDRANT_COLLECTION
        from archivist.core.provenance import SourceTrace
        from archivist.storage.qdrant import qdrant_client

        client = qdrant_client()
        points = await asyncio.to_thread(
            client.retrieve,
            collection_name=QDRANT_COLLECTION,
            ids=[memory_id],
            with_payload=True,
        )
        if not points:
            return edges
        payload = points[0].payload or {}
        trace = SourceTrace.from_dict(payload.get("source_trace") or {})
        actor_id = str(payload.get("actor_id") or payload.get("agent_id") or "")
        actor_type = str(payload.get("actor_type") or "")
        if trace.parent_memory_id:
            edges.append(
                _edge(
                    edge_type="provenance_parent",
                    from_id=trace.parent_memory_id,
                    to_id=memory_id,
                    actor_id=actor_id,
                    actor_type=actor_type,
                    relation="derived_from",
                    metadata={
                        k: v
                        for k, v in {
                            "tool": trace.tool,
                            "upstream_source": trace.upstream_source,
                            "session_id": trace.session_id,
                        }.items()
                        if v
                    },
                )
            )
        elif trace.tool or trace.upstream_source:
            edges.append(
                _edge(
                    edge_type="provenance_source",
                    from_id=trace.upstream_source or trace.tool or "unknown",
                    to_id=memory_id,
                    actor_id=actor_id,
                    actor_type=actor_type,
                    relation="originated_from",
                    metadata={
                        k: v
                        for k, v in {
                            "tool": trace.tool,
                            "upstream_source": trace.upstream_source,
                            "session_id": trace.session_id,
                        }.items()
                        if v
                    },
                )
            )
    except Exception as e:
        logger.debug("lineage._provenance_edges failed: %s", e)
    return edges


async def _version_edges(memory_id: str, limit: int) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    try:
        async with pool.read() as conn:
            rows = await conn.fetchall(
                """SELECT version, agent_id, timestamp, operation, parent_versions
                   FROM memory_versions
                   WHERE memory_id = ?
                   ORDER BY version DESC
                   LIMIT ?""",
                (memory_id, limit),
            )
        for row in rows:
            parents_raw = row["parent_versions"] or "[]"
            try:
                parents = json.loads(parents_raw) if isinstance(parents_raw, str) else parents_raw
            except (TypeError, json.JSONDecodeError):
                parents = []
            ver = row["version"]
            parent_label = ",".join(str(p) for p in parents) if parents else "0"
            edges.append(
                _edge(
                    edge_type="version",
                    from_id=f"{memory_id}@v{parent_label}",
                    to_id=f"{memory_id}@v{ver}",
                    actor_id=str(row["agent_id"] or ""),
                    actor_type="agent",
                    timestamp=str(row["timestamp"] or ""),
                    relation=str(row["operation"] or "version"),
                    metadata={"version": ver, "parent_versions": parents},
                )
            )
    except Exception as e:
        logger.debug("lineage._version_edges failed: %s", e)
    return edges


async def _audit_edges(memory_id: str, limit: int) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    try:
        from archivist.core.audit import get_audit_trail

        entries = await get_audit_trail(memory_id, limit=limit)
        for entry in entries:
            # Never echo audit metadata wholesale — it may contain caller context.
            edges.append(
                _edge(
                    edge_type="audit",
                    from_id=str(entry.get("agent_id") or "unknown"),
                    to_id=memory_id,
                    actor_id=str(entry.get("agent_id") or ""),
                    actor_type="agent",
                    timestamp=str(entry.get("timestamp") or ""),
                    relation=str(entry.get("action") or "audit"),
                    metadata={
                        k: entry[k]
                        for k in ("version", "namespace", "text_hash")
                        if entry.get(k) not in (None, "")
                    },
                )
            )
    except Exception as e:
        logger.debug("lineage._audit_edges failed: %s", e)
    return edges


async def _retrieval_edges(memory_id: str, limit: int) -> list[dict[str, Any]]:
    """Edges from retrieval_logs whose trace JSON mentions the memory id."""
    edges: list[dict[str, Any]] = []
    try:
        async with pool.read() as conn:
            # LIKE against JSON text is approximate but avoids dialect-specific JSON path ops.
            rows = await conn.fetchall(
                """SELECT id, agent_id, query, namespace, created_at, result_count
                   FROM retrieval_logs
                   WHERE retrieval_trace LIKE ? ESCAPE '\\'
                   ORDER BY created_at DESC
                   LIMIT ?""",
                (f"%{_escape_like(memory_id)}%", limit),
            )
        for row in rows:
            edges.append(
                _edge(
                    edge_type="retrieval",
                    from_id=str(row["id"]),
                    to_id=memory_id,
                    actor_id=str(row["agent_id"] or ""),
                    actor_type="agent",
                    timestamp=str(row["created_at"] or ""),
                    relation="retrieved_in",
                    metadata={
                        "namespace": row["namespace"] or "",
                        "result_count": row["result_count"],
                        # Truncate query for metrics; never treat as secret dump.
                        "query_preview": (row["query"] or "")[:120],
                    },
                )
            )
    except Exception as e:
        logger.debug("lineage._retrieval_edges failed: %s", e)
    return edges


async def _entity_edges(entity_id: str, limit: int, namespace: str = "") -> list[dict[str, Any]]:
    """Lineage-ish edges for an entity from facts / relationships tables.

    When *namespace* is set, every row is scoped to it. Entity names are global,
    so without this filter an entity lookup would return other tenants' facts and
    relationships (INIT-001/SPEC-012, SEC-012-02).
    """
    edges: list[dict[str, Any]] = []
    try:
        async with pool.read() as conn:
            # entities.id is INTEGER; also resolve by case-insensitive name.
            if entity_id.isdigit():
                ent = await conn.fetchone(
                    "SELECT id, name FROM entities WHERE id = ?"
                    + (" AND namespace = ?" if namespace else "")
                    + " LIMIT 1",
                    (int(entity_id), namespace) if namespace else (int(entity_id),),
                )
            else:
                ent = await conn.fetchone(
                    "SELECT id, name FROM entities WHERE name = ? COLLATE NOCASE"
                    + (" AND namespace = ?" if namespace else "")
                    + " LIMIT 1",
                    (entity_id, namespace) if namespace else (entity_id,),
                )
            if not ent:
                return edges
            eid = ent["id"]
            fact_rows = await conn.fetchall(
                """SELECT id, agent_id, fact_text, created_at, memory_id
                   FROM facts WHERE entity_id = ? AND is_active = 1"""
                + (" AND namespace = ?" if namespace else "")
                + " ORDER BY created_at DESC LIMIT ?",
                (eid, namespace, limit) if namespace else (eid, limit),
            )
            for row in fact_rows:
                edges.append(
                    _edge(
                        edge_type="entity_fact",
                        from_id=str(row["agent_id"] or "unknown"),
                        to_id=str(eid),
                        actor_id=str(row["agent_id"] or ""),
                        actor_type="agent",
                        timestamp=str(row["created_at"] or ""),
                        relation="asserted_fact",
                        metadata={
                            "fact_id": row["id"],
                            "memory_id": row["memory_id"] or "",
                            "fact_preview": (row["fact_text"] or "")[:120],
                        },
                    )
                )
            rel_rows = await conn.fetchall(
                """SELECT id, source_entity_id, target_entity_id, relation_type, agent_id
                   FROM relationships
                   WHERE (source_entity_id = ? OR target_entity_id = ?)"""
                + (" AND namespace = ?" if namespace else "")
                + " LIMIT ?",
                (eid, eid, namespace, limit) if namespace else (eid, eid, limit),
            )
            for row in rel_rows:
                edges.append(
                    _edge(
                        edge_type="entity_relationship",
                        from_id=str(row["source_entity_id"]),
                        to_id=str(row["target_entity_id"]),
                        actor_id=str(row["agent_id"] or ""),
                        actor_type="agent",
                        relation=str(row["relation_type"] or "related_to"),
                        metadata={"relationship_id": row["id"]},
                    )
                )
    except Exception as e:
        logger.debug("lineage._entity_edges failed: %s", e)
    return edges


async def build_memory_lineage(
    memory_id: str,
    *,
    limit: int = 50,
    namespace: str | None = None,
) -> dict[str, Any]:
    """Aggregate lineage edges for a memory id. Empty when no sources exist.

    Pass *namespace* when the caller has already resolved and authorized the
    owning namespace, to avoid a second lookup.
    """
    lim = max(1, min(int(limit or 50), 200))
    if namespace is None:
        namespace = await resolve_memory_namespace(memory_id)

    version_e = await _version_edges(memory_id, lim)
    audit_e = await _audit_edges(memory_id, lim)
    prov_e = await _provenance_edges(memory_id)
    retr_e = await _retrieval_edges(memory_id, lim)

    edges = version_e + audit_e + prov_e + retr_e
    sources = []
    if version_e:
        sources.append("versions")
    if audit_e:
        sources.append("audit")
    if prov_e:
        sources.append("provenance")
    if retr_e:
        sources.append("retrieval_logs")

    return {
        "resource_type": "memory",
        "resource_id": memory_id,
        "namespace": namespace,
        "edge_count": len(edges),
        "edges": edges,
        "sources": sources,
    }


async def build_entity_lineage(
    entity_id: str,
    *,
    namespace: str = "",
    limit: int = 50,
) -> dict[str, Any]:
    """Aggregate lineage edges for an entity id/name.

    Entity names are global; *namespace* is the authorized RBAC scope and is
    applied as a hard filter on the entity, its facts, and its relationships.
    """
    lim = max(1, min(int(limit or 50), 200))
    edges = await _entity_edges(entity_id, lim, namespace=namespace or "")
    return {
        "resource_type": "entity",
        "resource_id": entity_id,
        "namespace": namespace or "",
        "edge_count": len(edges),
        "edges": edges,
        "sources": ["facts", "relationships"] if edges else [],
    }
