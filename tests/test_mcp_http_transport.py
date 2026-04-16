"""Tests for MCP HTTP transport wiring in main.py.

Covers both transport layers:
  - Streamable HTTP  POST /mcp        (primary, MCP spec ≥2025-03)
  - Legacy SSE       GET  /mcp/sse    (backward-compat for OpenClaw ≤v2026.4.8)

Also covers the auth middleware OpenClaw placeholder acceptance.
"""

from contextlib import asynccontextmanager

import pytest
from starlette.routing import Route, Mount


def test_app_registers_streamable_http_route():
    import main

    route = next(
        (
            route
            for route in main.app.routes
            if isinstance(route, Route) and route.path == "/mcp"
        ),
        None,
    )

    assert route is not None
    assert {"GET", "POST", "DELETE"} <= route.methods
    assert route.endpoint is main.streamable_http_app


@pytest.mark.asyncio
async def test_streamable_http_app_delegates_to_session_manager(monkeypatch):
    import main

    called = {}

    class FakeManager:
        async def handle_request(self, scope, receive, send):
            called["scope"] = scope
            called["receive"] = receive
            called["send"] = send

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(_message):
        return None

    monkeypatch.setattr(main, "streamable_http_session_manager", FakeManager())

    scope = {"type": "http", "path": "/mcp"}
    await main.streamable_http_app(scope, _receive, _send)

    assert called["scope"] is scope
    assert called["receive"] is _receive
    assert called["send"] is _send


@pytest.mark.asyncio
async def test_sse_app_connects_transport_and_runs_server(monkeypatch):
    import main

    called = []

    @asynccontextmanager
    async def fake_connect_sse(scope, receive, send):
        called.append(("connect", scope, receive, send))
        yield ("read-stream", "write-stream")

    async def fake_run(read_stream, write_stream, options):
        called.append(("run", read_stream, write_stream, options))

    monkeypatch.setattr(main.sse_transport, "connect_sse", fake_connect_sse)
    monkeypatch.setattr(main.server, "run", fake_run)
    monkeypatch.setattr(main.server, "create_initialization_options", lambda: "init-options")

    async def _receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def _send(_message):
        return None

    scope = {"type": "http", "path": "/mcp/sse"}
    await main.sse_app(scope, _receive, _send)

    assert called == [
        ("connect", scope, _receive, _send),
        ("run", "read-stream", "write-stream", "init-options"),
    ]


@pytest.mark.asyncio
async def test_lifespan_initializes_and_clears_session_manager(monkeypatch):
    import main

    events = []

    class FakeManager:
        @asynccontextmanager
        async def run(self):
            events.append("manager_enter")
            try:
                yield
            finally:
                events.append("manager_exit")

    async def fake_startup():
        events.append("startup")

    fake_manager = FakeManager()

    monkeypatch.setattr(main, "_create_streamable_http_session_manager", lambda: fake_manager)
    monkeypatch.setattr(main, "_startup", fake_startup)
    monkeypatch.setattr(main, "_background_tasks", [])

    assert main.streamable_http_session_manager is None

    async with main.lifespan(main.app):
        events.append("inside")
        assert main.streamable_http_session_manager is fake_manager

    assert events == ["manager_enter", "startup", "inside", "manager_exit"]
    assert main.streamable_http_session_manager is None


# ── Phase 6.5 — OpenClaw Compatibility ───────────────────────────────────────


def test_app_registers_sse_get_route():
    """Legacy SSE GET /mcp/sse must be registered (MCP_SSE_ENABLED defaults true)."""
    import main

    route = next(
        (r for r in main.app.routes if isinstance(r, Route) and r.path == "/mcp/sse"),
        None,
    )
    assert route is not None, "/mcp/sse route not found — MCP_SSE_ENABLED may be false"
    assert "GET" in route.methods
    assert route.endpoint is main.sse_app


def test_app_registers_sse_messages_mount():
    """Legacy SSE POST /mcp/messages/ must be mounted for the SSE transport."""
    import main

    mount = next(
        (r for r in main.app.routes if isinstance(r, Mount) and r.path == "/mcp/messages"),
        None,
    )
    assert mount is not None, "/mcp/messages/ mount not found — MCP_SSE_ENABLED may be false"


def test_sse_transport_is_not_none_when_enabled():
    """When MCP_SSE_ENABLED is true, sse_transport must be a real SseServerTransport."""
    import main

    assert main.sse_transport is not None, (
        "sse_transport is None — expected a live SseServerTransport when MCP_SSE_ENABLED=true"
    )


@pytest.mark.asyncio
async def test_auth_middleware_accepts_actual_key(monkeypatch):
    """Standard Bearer token with the actual key must be accepted."""
    import main
    from starlette.testclient import TestClient

    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "correct-key", raising=False)
    monkeypatch.setattr("main.ARCHIVIST_API_KEY", "correct-key", raising=False)

    middleware = main.ArchivistAuthMiddleware(app=None)

    accepted = {}

    async def call_next(req):
        accepted["called"] = True
        from starlette.responses import JSONResponse
        return JSONResponse({"ok": True})

    from starlette.testclient import TestClient
    from starlette.datastructures import Headers
    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer correct-key")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    response = await middleware.dispatch(request, call_next)
    assert accepted.get("called"), "call_next was not invoked — valid Bearer token was rejected"


@pytest.mark.asyncio
async def test_auth_middleware_accepts_openclaw_placeholder(monkeypatch):
    """OpenClaw compatibility: literal 'Bearer ${ARCHIVIST_API_KEY}' must be accepted with a warning."""
    import main
    import logging

    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "some-secret-key", raising=False)

    middleware = main.ArchivistAuthMiddleware(app=None)
    accepted = {}

    async def call_next(req):
        accepted["called"] = True
        from starlette.responses import JSONResponse
        return JSONResponse({"ok": True})

    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/mcp/sse",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer ${ARCHIVIST_API_KEY}")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    # Warning is emitted via logger.warning(), not Python warnings.warn() —
    # just dispatch and assert call_next was invoked (no 401).
    response = await middleware.dispatch(request, call_next)
    assert accepted.get("called"), (
        "call_next was not invoked — OpenClaw literal placeholder '${ARCHIVIST_API_KEY}' was rejected"
    )


@pytest.mark.asyncio
async def test_auth_middleware_rejects_unknown_token(monkeypatch):
    """An unrecognized Bearer token must still return 401."""
    import main

    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "real-secret", raising=False)

    middleware = main.ArchivistAuthMiddleware(app=None)
    rejected = {}

    async def call_next(req):
        rejected["called"] = True
        from starlette.responses import JSONResponse
        return JSONResponse({"ok": True})

    from starlette.requests import Request

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer wrong-key")],
    }

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    request = Request(scope, receive)
    response = await middleware.dispatch(request, call_next)
    assert not rejected.get("called"), "call_next was invoked for an invalid Bearer token"
    assert response.status_code == 401
