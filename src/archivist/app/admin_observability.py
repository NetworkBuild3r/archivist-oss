"""HTTP handlers for Diff #8 observability billboard (INIT-013/SPEC-002).

Thin wrappers around ``app.lineage`` / ``core.audit`` for browser JSON —
parity with MCP ``archivist_memory_lineage`` / ``archivist_audit_trail`` authz.
"""

from __future__ import annotations

import logging
from typing import Any

from starlette.requests import Request
from starlette.responses import JSONResponse

from archivist.core.rbac import check_access, get_namespace_for_agent, is_permissive_mode

logger = logging.getLogger("archivist.admin")

_MAX_LIMIT = 200
_DEFAULT_LIMIT = 50


def _parse_limit(raw: str | None) -> int:
    try:
        lim = int(raw or _DEFAULT_LIMIT)
    except (TypeError, ValueError):
        lim = _DEFAULT_LIMIT
    return max(1, min(lim, _MAX_LIMIT))


def _caller_from_params(params: Any) -> str:
    agent_id = (params.get("agent_id") or "").strip()
    return (params.get("caller_agent_id") or "").strip() or agent_id


def _rbac_denied_response(agent_id: str, action: str, namespace: str) -> JSONResponse | None:
    policy = check_access(agent_id, action, namespace)
    if policy.allowed:
        return None
    payload: dict[str, Any] = {"error": "access_denied", "reason": policy.reason}
    if policy.hint:
        payload["hint"] = policy.hint
    return JSONResponse(payload, status_code=403)


async def handle_lineage(request: Request) -> JSONResponse:
    """GET /admin/lineage — memory or entity lineage edges (INIT-013/SPEC-002)."""
    from archivist.app.lineage import (
        build_entity_lineage,
        build_memory_lineage,
        resolve_memory_namespace,
        validate_entity_id,
        validate_memory_id,
    )

    params = request.query_params
    caller = _caller_from_params(params)
    memory_id_raw = (params.get("memory_id") or "").strip()
    entity_id_raw = (params.get("entity_id") or "").strip()
    namespace = (params.get("namespace") or "").strip()
    limit = _parse_limit(params.get("limit"))

    if memory_id_raw and entity_id_raw:
        return JSONResponse(
            {"error": "invalid_arguments", "reason": "Provide memory_id or entity_id, not both."},
            status_code=400,
        )
    if not memory_id_raw and not entity_id_raw:
        return JSONResponse(
            {"error": "invalid_arguments", "reason": "memory_id or entity_id is required."},
            status_code=400,
        )
    if not caller and not is_permissive_mode():
        return JSONResponse(
            {
                "error": "invalid_arguments",
                "reason": "agent_id (or caller_agent_id) is required under RBAC.",
            },
            status_code=400,
        )

    try:
        if memory_id_raw:
            memory_id = validate_memory_id(memory_id_raw)
            if not memory_id:
                return JSONResponse(
                    {"error": "invalid_id", "reason": "memory_id failed validation."},
                    status_code=400,
                )
            if is_permissive_mode():
                result = await build_memory_lineage(memory_id, limit=limit)
                ns = namespace or result.get("namespace") or ""
            else:
                ns = await resolve_memory_namespace(memory_id)
                if not ns:
                    return JSONResponse(
                        {
                            "error": "access_denied",
                            "reason": "Unable to resolve memory namespace for RBAC.",
                        },
                        status_code=403,
                    )
                if namespace and namespace != ns:
                    return JSONResponse(
                        {
                            "error": "access_denied",
                            "reason": "memory does not belong to the requested namespace.",
                        },
                        status_code=403,
                    )
                if denied := _rbac_denied_response(caller, "read", ns):
                    return denied
                result = await build_memory_lineage(memory_id, limit=limit, namespace=ns)
            result["namespace"] = ns
            return JSONResponse(result)

        entity_id = validate_entity_id(entity_id_raw)
        if not entity_id:
            return JSONResponse(
                {"error": "invalid_id", "reason": "entity_id failed validation."},
                status_code=400,
            )
        if not namespace:
            namespace = get_namespace_for_agent(caller) if caller else ""
        if namespace and not is_permissive_mode():
            if denied := _rbac_denied_response(caller, "read", namespace):
                return denied
        elif not namespace and not is_permissive_mode():
            return JSONResponse(
                {
                    "error": "access_denied",
                    "reason": "namespace is required for entity lineage under RBAC.",
                },
                status_code=403,
            )
        result = await build_entity_lineage(entity_id, namespace=namespace, limit=limit)
        return JSONResponse(result)
    except Exception:
        logger.exception("GET /admin/lineage failed")
        return JSONResponse(
            {"error": "internal_error", "reason": "lineage failed"}, status_code=500
        )


async def handle_audit(request: Request) -> JSONResponse:
    """GET /admin/audit — audit trail by memory_id or agent_id (INIT-013/SPEC-002)."""
    from archivist.core.audit import get_agent_activity, get_audit_trail

    params = request.query_params
    memory_id = (params.get("memory_id") or "").strip()
    agent_id = (params.get("agent_id") or params.get("target_agent") or "").strip()
    limit = _parse_limit(params.get("limit"))

    try:
        if memory_id:
            entries = await get_audit_trail(memory_id, limit=limit)
        else:
            # Empty agent_id → recent activity (same as MCP audit_trail default).
            entries = await get_agent_activity(agent_id, limit=limit)
        return JSONResponse({"entries": entries, "count": len(entries)})
    except Exception:
        logger.exception("GET /admin/audit failed")
        return JSONResponse({"error": "internal_error", "reason": "audit failed"}, status_code=500)
