"""MCP tool handlers — Memory as a Product (Diff #4).

Thin wrappers around ``storage.memory_product`` snapshot / fork / export /
import / list / get. Visible on **ops** and **full** only (GR-PROD-002 —
never on core). Archives use opaque ``archive_id`` / ``version_id``; path
confinement stays in the service layer.

Provenance: INIT-011/SPEC-003; service: INIT-001/SPEC-009 + INIT-011/SPEC-002.
"""

from __future__ import annotations

import logging
from typing import Any

from mcp.types import TextContent, Tool

from archivist.storage.backup_manager import SnapshotPathError
from archivist.storage.memory_product import (
    MemoryProductAuthzError,
    MemoryProductConflictError,
    MemoryProductError,
    MemoryProductNotFoundError,
    ScopeVersionRecord,
    create_scope_snapshot,
    export_scope,
    fork_from_snapshot,
    get_scope_version,
    import_scope,
    list_scope_versions,
)

from ._common import (
    error_response,
    require_caller,
    require_rbac,
    resolve_caller,
    success_response,
)

logger = logging.getLogger("archivist.mcp")

_MAP_TOOL_NAMES = (
    "archivist_map_list",
    "archivist_map_get",
    "archivist_map_snapshot",
    "archivist_map_fork",
    "archivist_map_export",
    "archivist_map_import",
)

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS: list[Tool] = [
    Tool(
        name="archivist_map_list",
        description=(
            "List Memory-as-Product scope versions for a namespace "
            "(optionally filtered by agent_id). Ops/full profile only."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Namespace to list versions for (required).",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Optional agent scope filter.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Max versions to return (default 50).",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Caller identity for RBAC (required).",
                },
            },
            "required": ["namespace", "caller_agent_id"],
        },
    ),
    Tool(
        name="archivist_map_get",
        description=("Get one Memory-as-Product scope version by version_id. Ops/full only."),
        inputSchema={
            "type": "object",
            "properties": {
                "version_id": {
                    "type": "string",
                    "description": "Scope version id (required).",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Caller identity for RBAC (required).",
                },
            },
            "required": ["version_id", "caller_agent_id"],
        },
    ),
    Tool(
        name="archivist_map_snapshot",
        description=(
            "Create a versioned snapshot archive of a memory scope under BACKUP_DIR. Ops/full only."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Source namespace (required).",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Optional agent scope filter for the snapshot.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional label for the version/archive.",
                },
                "parent_version_id": {
                    "type": "string",
                    "description": "Optional parent version id for lineage.",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Caller identity for RBAC (required).",
                },
            },
            "required": ["namespace", "caller_agent_id"],
        },
    ),
    Tool(
        name="archivist_map_fork",
        description=(
            "Fork a scope version into a target namespace with lineage. "
            "Requires read on source and write on target. Ops/full only."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "source_version_id": {
                    "type": "string",
                    "description": "Source scope version id (required).",
                },
                "target_namespace": {
                    "type": "string",
                    "description": "Destination namespace (required).",
                },
                "target_agent_id": {
                    "type": "string",
                    "description": "Optional destination agent scope.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional label for the fork.",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Caller identity for RBAC (required).",
                },
            },
            "required": ["source_version_id", "target_namespace", "caller_agent_id"],
        },
    ),
    Tool(
        name="archivist_map_export",
        description=(
            "Export a memory scope (or existing version) to BACKUP_DIR with a manifest. "
            "Ops/full only."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "namespace": {
                    "type": "string",
                    "description": "Namespace to export (required).",
                },
                "agent_id": {
                    "type": "string",
                    "description": "Optional agent scope filter.",
                },
                "version_id": {
                    "type": "string",
                    "description": "Optional existing version id to export.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional export label.",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Caller identity for RBAC (required).",
                },
            },
            "required": ["namespace", "caller_agent_id"],
        },
    ),
    Tool(
        name="archivist_map_import",
        description=(
            "Import an archive under BACKUP_DIR into a target namespace (fail-closed "
            "if target scope nonempty). Ops/full only."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "archive_id": {
                    "type": "string",
                    "description": "Opaque archive id under BACKUP_DIR (required).",
                },
                "target_namespace": {
                    "type": "string",
                    "description": "Destination namespace (required).",
                },
                "target_agent_id": {
                    "type": "string",
                    "description": "Optional destination agent scope.",
                },
                "label": {
                    "type": "string",
                    "description": "Optional import label.",
                },
                "caller_agent_id": {
                    "type": "string",
                    "description": "Caller identity for RBAC (required).",
                },
            },
            "required": ["archive_id", "target_namespace", "caller_agent_id"],
        },
    ),
]


def _record_to_dict(record: ScopeVersionRecord) -> dict[str, Any]:
    return {
        "id": record.id,
        "source_namespace": record.source_namespace,
        "source_agent_id": record.source_agent_id,
        "version": record.version,
        "label": record.label,
        "parent_version_id": record.parent_version_id,
        "chunk_count": record.chunk_count,
        "point_count": record.point_count,
        "archive_id": record.archive_id,
        "operation": record.operation,
        "created_by": record.created_by,
        "created_at": record.created_at,
        "lineage": record.lineage,
    }


def _map_error(exc: BaseException) -> list[TextContent]:
    """Map MaP service errors to structured MCP payloads (no stack leak)."""
    if isinstance(exc, MemoryProductAuthzError):
        return error_response({"error": "access_denied", "reason": str(exc)})
    if isinstance(exc, SnapshotPathError):
        return error_response({"error": "invalid_archive_id", "reason": str(exc)})
    if isinstance(exc, MemoryProductNotFoundError):
        return error_response({"error": "not_found", "reason": str(exc)})
    if isinstance(exc, MemoryProductConflictError):
        return error_response({"error": "conflict", "reason": str(exc)})
    if isinstance(exc, MemoryProductError):
        return error_response({"error": "memory_product_error", "reason": str(exc)})
    if isinstance(exc, ValueError):
        return error_response({"error": "invalid_request", "reason": str(exc)})
    return error_response({"error": "internal_error", "reason": "map operation failed"})


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _handle_map_list(arguments: dict) -> list[TextContent]:
    namespace = (arguments.get("namespace") or "").strip()
    agent_id = (arguments.get("agent_id") or "").strip()
    caller = resolve_caller(arguments)
    if err := require_caller(caller):
        return err
    if not namespace:
        return error_response({"error": "namespace is required"})

    limit_raw = arguments.get("limit", 50)
    try:
        limit = int(limit_raw)
    except (TypeError, ValueError):
        return error_response({"error": "invalid_request", "reason": "limit must be an integer"})

    logger.info(
        "archivist_map_list caller=%s namespace=%s agent_id=%s limit=%s",
        caller,
        namespace,
        agent_id or "-",
        limit,
    )
    try:
        records = await list_scope_versions(
            namespace=namespace,
            caller_agent_id=caller,
            agent_id=agent_id,
            limit=limit,
        )
    except Exception as exc:
        # MemoryProduct* + SnapshotPathError are covered by these bases.
        if isinstance(exc, MemoryProductError | ValueError):
            return _map_error(exc)
        logger.exception("archivist_map_list failed")
        return _map_error(exc)

    return success_response(
        {"versions": [_record_to_dict(r) for r in records], "count": len(records)},
        default=str,
    )


async def _handle_map_get(arguments: dict) -> list[TextContent]:
    version_id = (arguments.get("version_id") or "").strip()
    caller = resolve_caller(arguments)
    if err := require_caller(caller):
        return err
    if not version_id:
        return error_response({"error": "version_id is required"})

    logger.info("archivist_map_get caller=%s version_id=%s", caller, version_id)
    try:
        record = await get_scope_version(version_id)
        if record is None:
            return error_response(
                {"error": "not_found", "reason": f"version '{version_id}' not found"}
            )
        if denied := require_rbac(caller, "read", record.source_namespace):
            return denied
    except Exception as exc:
        # MemoryProduct* + SnapshotPathError are covered by these bases.
        if isinstance(exc, MemoryProductError | ValueError):
            return _map_error(exc)
        logger.exception("archivist_map_get failed")
        return _map_error(exc)

    return success_response({"version": _record_to_dict(record)}, default=str)


async def _handle_map_snapshot(arguments: dict) -> list[TextContent]:
    namespace = (arguments.get("namespace") or "").strip()
    agent_id = (arguments.get("agent_id") or "").strip()
    label = (arguments.get("label") or "").strip()
    parent_version_id = (arguments.get("parent_version_id") or "").strip() or None
    caller = resolve_caller(arguments)
    if err := require_caller(caller):
        return err
    if not namespace:
        return error_response({"error": "namespace is required"})

    logger.info(
        "archivist_map_snapshot caller=%s namespace=%s agent_id=%s label=%s",
        caller,
        namespace,
        agent_id or "-",
        label or "-",
    )
    try:
        record = await create_scope_snapshot(
            namespace=namespace,
            caller_agent_id=caller,
            agent_id=agent_id,
            label=label,
            parent_version_id=parent_version_id,
        )
    except Exception as exc:
        # MemoryProduct* + SnapshotPathError are covered by these bases.
        if isinstance(exc, MemoryProductError | ValueError):
            return _map_error(exc)
        logger.exception("archivist_map_snapshot failed")
        return _map_error(exc)

    logger.info(
        "archivist_map_snapshot ok version_id=%s archive_id=%s chunks=%s",
        record.id,
        record.archive_id,
        record.chunk_count,
    )
    return success_response({"version": _record_to_dict(record)}, default=str)


async def _handle_map_fork(arguments: dict) -> list[TextContent]:
    source_version_id = (arguments.get("source_version_id") or "").strip()
    target_namespace = (arguments.get("target_namespace") or "").strip()
    target_agent_id = (arguments.get("target_agent_id") or "").strip()
    label = (arguments.get("label") or "").strip()
    caller = resolve_caller(arguments)
    if err := require_caller(caller):
        return err
    if not source_version_id:
        return error_response({"error": "source_version_id is required"})
    if not target_namespace:
        return error_response({"error": "target_namespace is required"})

    logger.info(
        "archivist_map_fork caller=%s source_version_id=%s target_ns=%s",
        caller,
        source_version_id,
        target_namespace,
    )
    try:
        record = await fork_from_snapshot(
            source_version_id=source_version_id,
            target_namespace=target_namespace,
            caller_agent_id=caller,
            target_agent_id=target_agent_id,
            label=label,
        )
    except Exception as exc:
        # MemoryProduct* + SnapshotPathError are covered by these bases.
        if isinstance(exc, MemoryProductError | ValueError):
            return _map_error(exc)
        logger.exception("archivist_map_fork failed")
        return _map_error(exc)

    logger.info(
        "archivist_map_fork ok version_id=%s archive_id=%s chunks=%s",
        record.id,
        record.archive_id,
        record.chunk_count,
    )
    return success_response({"version": _record_to_dict(record)}, default=str)


async def _handle_map_export(arguments: dict) -> list[TextContent]:
    namespace = (arguments.get("namespace") or "").strip()
    agent_id = (arguments.get("agent_id") or "").strip()
    version_id = (arguments.get("version_id") or "").strip() or None
    label = (arguments.get("label") or "").strip()
    caller = resolve_caller(arguments)
    if err := require_caller(caller):
        return err
    if not namespace:
        return error_response({"error": "namespace is required"})

    logger.info(
        "archivist_map_export caller=%s namespace=%s version_id=%s",
        caller,
        namespace,
        version_id or "-",
    )
    try:
        result = await export_scope(
            namespace=namespace,
            caller_agent_id=caller,
            agent_id=agent_id,
            version_id=version_id,
            label=label,
        )
    except Exception as exc:
        # MemoryProduct* + SnapshotPathError are covered by these bases.
        if isinstance(exc, MemoryProductError | ValueError):
            return _map_error(exc)
        logger.exception("archivist_map_export failed")
        return _map_error(exc)

    # Return ids/paths/manifest metadata — omit raw ``bytes`` blob from MCP.
    payload = {
        "path": result.get("path"),
        "archive_id": result.get("archive_id"),
        "version_id": result.get("version_id"),
        "manifest": result.get("manifest"),
        "chunk_count": result.get("chunk_count"),
        "point_count": result.get("point_count"),
    }
    logger.info(
        "archivist_map_export ok archive_id=%s version_id=%s path=%s",
        payload["archive_id"],
        payload["version_id"],
        payload["path"],
    )
    return success_response(payload, default=str)


async def _handle_map_import(arguments: dict) -> list[TextContent]:
    archive_id = (arguments.get("archive_id") or "").strip()
    target_namespace = (arguments.get("target_namespace") or "").strip()
    target_agent_id = (arguments.get("target_agent_id") or "").strip()
    label = (arguments.get("label") or "").strip()
    caller = resolve_caller(arguments)
    if err := require_caller(caller):
        return err
    if not archive_id:
        return error_response({"error": "archive_id is required"})
    if not target_namespace:
        return error_response({"error": "target_namespace is required"})

    logger.info(
        "archivist_map_import caller=%s archive_id=%s target_ns=%s",
        caller,
        archive_id,
        target_namespace,
    )
    try:
        record = await import_scope(
            archive_id=archive_id,
            target_namespace=target_namespace,
            caller_agent_id=caller,
            target_agent_id=target_agent_id,
            label=label,
        )
    except Exception as exc:
        # MemoryProduct* + SnapshotPathError are covered by these bases.
        if isinstance(exc, MemoryProductError | ValueError):
            return _map_error(exc)
        logger.exception("archivist_map_import failed")
        return _map_error(exc)

    logger.info(
        "archivist_map_import ok version_id=%s archive_id=%s chunks=%s",
        record.id,
        record.archive_id,
        record.chunk_count,
    )
    return success_response({"version": _record_to_dict(record)}, default=str)


HANDLERS: dict[str, object] = {
    "archivist_map_list": _handle_map_list,
    "archivist_map_get": _handle_map_get,
    "archivist_map_snapshot": _handle_map_snapshot,
    "archivist_map_fork": _handle_map_fork,
    "archivist_map_export": _handle_map_export,
    "archivist_map_import": _handle_map_import,
}

__all__ = ["HANDLERS", "TOOLS", "_MAP_TOOL_NAMES"]
