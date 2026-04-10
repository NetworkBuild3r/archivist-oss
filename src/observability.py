"""Request correlation and optional slow-path logging for observability.

HTTP clients can send ``X-Request-ID``; it is propagated into a contextvar so
MCP tool handlers and dependency calls can include it in logs. OpenTelemetry
is optional: set ``OTEL_EXPORTER_OTLP_ENDPOINT`` and install OTel packages to
enable tracing; otherwise helpers are no-ops.
"""

from __future__ import annotations

import logging
import uuid
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, Iterator

from config import SLOW_EMBED_MS, SLOW_LLM_MS, SLOW_QDRANT_MS

logger = logging.getLogger("archivist.observability")

_request_id: ContextVar[str] = ContextVar("request_id", default="")


def get_request_id() -> str:
    return _request_id.get()


def set_request_id_from_scope(scope: dict) -> Token[str]:
    """Parse ASGI scope headers for ``X-Request-ID`` or generate a short id."""
    raw_headers = scope.get("headers") or []
    headers = {k.decode("latin-1").lower(): v.decode("latin-1") for k, v in raw_headers}
    rid = (headers.get("x-request-id") or "").strip()
    if not rid:
        rid = uuid.uuid4().hex[:12]
    return _request_id.set(rid)


def reset_request_id(token: Token[str]) -> None:
    _request_id.reset(token)


def log_slow_path(step: str, duration_ms: float, threshold_ms: float) -> None:
    """Emit one warning if threshold is enabled and duration exceeds it."""
    if threshold_ms <= 0 or duration_ms < threshold_ms:
        return
    rid = get_request_id()
    suffix = f" request_id={rid}" if rid else ""
    logger.warning(
        "slow_path step=%s duration_ms=%.1f%s",
        step,
        duration_ms,
        suffix,
    )


def slow_embed_check(duration_ms: float) -> None:
    log_slow_path("embed", duration_ms, SLOW_EMBED_MS)


def slow_qdrant_check(duration_ms: float) -> None:
    log_slow_path("qdrant_query", duration_ms, SLOW_QDRANT_MS)


def slow_llm_check(duration_ms: float) -> None:
    log_slow_path("llm", duration_ms, SLOW_LLM_MS)


# ── Optional OpenTelemetry (no hard dependency) ───────────────────────────────

_UNSET = object()
_tracer: Any = _UNSET


def _init_tracer() -> Any:
    """Resolve the OTel tracer exactly once; returns None when OTel is absent."""
    global _tracer
    if _tracer is not _UNSET:
        return _tracer
    import os

    if not os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "").strip():
        _tracer = None
        return None
    try:
        from opentelemetry import trace  # type: ignore

        _tracer = trace.get_tracer("archivist", "1.0.0")
    except Exception:
        _tracer = None
    return _tracer


@contextmanager
def tool_span(name: str) -> Iterator[None]:
    """Optional span around an MCP tool call when OTel is configured."""
    tr = _init_tracer()
    if tr is None:
        yield
        return
    with tr.start_as_current_span(f"mcp.tool.{name}"):
        yield
