"""Tests for MCP HTTP transport wiring in main.py.

Covers both transport layers:
  - Streamable HTTP  POST /mcp        (primary, MCP spec ≥2025-03)
  - Legacy SSE       GET  /mcp/sse    (backward-compat for OpenClaw ≤v2026.4.8)

Also covers the auth middleware OpenClaw placeholder acceptance.

Auth middleware tests use the pure-ASGI interface (scope/receive/send) because
ArchivistAuthMiddleware is no longer a BaseHTTPMiddleware subclass — using
BaseHTTPMiddleware was the root cause of the "Session not found" bug (MCP SDK
issue #883).
"""

from contextlib import asynccontextmanager

import pytest
from starlette.routing import Mount, Route

pytestmark = [pytest.mark.integration, pytest.mark.mcp]


def test_app_registers_streamable_http_route():
    import main

    route = next(
        (r for r in main._inner_app.routes if isinstance(r, Route) and r.path == "/mcp"),
        None,
    )

    assert route is not None
    assert {"GET", "POST", "DELETE"} <= route.methods
    assert route.endpoint is main.streamable_http_app


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

    async with main.lifespan(main._inner_app):
        events.append("inside")
        assert main.streamable_http_session_manager is fake_manager

    assert events == ["manager_enter", "startup", "inside", "manager_exit"]
    assert main.streamable_http_session_manager is None


# ── Phase 6.5 — OpenClaw Compatibility ───────────────────────────────────────


def test_app_registers_sse_get_route():
    """Legacy SSE GET /mcp/sse must be registered (MCP_SSE_ENABLED defaults true)."""
    import main

    route = next(
        (r for r in main._inner_app.routes if isinstance(r, Route) and r.path == "/mcp/sse"),
        None,
    )
    assert route is not None, "/mcp/sse route not found — MCP_SSE_ENABLED may be false"
    assert "GET" in route.methods
    assert route.endpoint is main.sse_app


def test_app_registers_sse_messages_mount():
    """Legacy SSE POST /mcp/messages/ must be mounted for the SSE transport."""
    import main

    mount = next(
        (r for r in main._inner_app.routes if isinstance(r, Mount) and r.path == "/mcp/messages"),
        None,
    )
    assert mount is not None, "/mcp/messages/ mount not found — MCP_SSE_ENABLED may be false"


def test_sse_transport_is_not_none_when_enabled():
    """When MCP_SSE_ENABLED is true, sse_transport must be a real SseServerTransport."""
    import main

    assert main.sse_transport is not None, (
        "sse_transport is None — expected a live SseServerTransport when MCP_SSE_ENABLED=true"
    )


# ── Auth middleware — pure ASGI interface ─────────────────────────────────────
# The middleware uses raw ASGI (scope/receive/send) rather than BaseHTTPMiddleware
# so tests call __call__ directly, capturing sends via a mock send callable.


async def test_auth_middleware_accepts_actual_key(monkeypatch):
    """Standard Bearer token with the actual key must be accepted."""
    import main as main_mod_local

    monkeypatch.setattr(main_mod_local, "ARCHIVIST_API_KEY", "correct-key", raising=False)

    downstream_called = False

    async def downstream(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        pass

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer correct-key")],
    }
    await main_mod_local.ArchivistAuthMiddleware(app=downstream)(scope, receive, send)
    assert downstream_called, "downstream was not called — valid Bearer token was rejected"


async def test_auth_middleware_accepts_openclaw_placeholder(monkeypatch):
    """OpenClaw compat: literal 'Bearer ${ARCHIVIST_API_KEY}' must be accepted with a warning."""
    import main as main_mod_local

    monkeypatch.setattr(main_mod_local, "ARCHIVIST_API_KEY", "some-secret-key", raising=False)

    downstream_called = False

    async def downstream(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        pass

    scope = {
        "type": "http",
        "method": "GET",
        "path": "/mcp/sse",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer ${ARCHIVIST_API_KEY}")],
    }
    await main_mod_local.ArchivistAuthMiddleware(app=downstream)(scope, receive, send)
    assert downstream_called, (
        "downstream was not called — OpenClaw literal '${ARCHIVIST_API_KEY}' placeholder was rejected"
    )


async def test_auth_middleware_rejects_unknown_token(monkeypatch):
    """An unrecognized Bearer token must return 401 without calling downstream."""
    import main as main_mod_local

    monkeypatch.setattr(main_mod_local, "ARCHIVIST_API_KEY", "real-secret", raising=False)

    downstream_called = False
    sent_messages: list = []

    async def downstream(scope, receive, send):
        nonlocal downstream_called
        downstream_called = True

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    async def send(msg):
        sent_messages.append(msg)

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "query_string": b"",
        "headers": [(b"authorization", b"Bearer wrong-key")],
    }
    await main_mod_local.ArchivistAuthMiddleware(app=downstream)(scope, receive, send)
    assert not downstream_called, "downstream was called despite an invalid Bearer token"

    status = next(
        (m["status"] for m in sent_messages if m.get("type") == "http.response.start"), None
    )
    assert status == 401, f"expected 401, got {status}"


# ── Session persistence regression test ──────────────────────────────────────


async def test_session_id_survives_initialize_then_tool_call(monkeypatch):
    """Verify that the mcp-session-id header returned by initialize is accepted
    on a subsequent request — i.e. the session is not dropped between calls.

    This is the core regression test for MCP SDK issue #883: when
    BaseHTTPMiddleware wraps the ASGI stack the session is cleaned up after
    the first request, so the second call returns "Session not found".
    With a pure ASGI middleware the session must survive across calls.
    """
    import main

    SESSION_ID = "test-session-abc123"

    # Track which session IDs the fake manager has seen
    sessions: dict[str, bool] = {}
    initialized_session: list[str] = []

    class FakeSessionManager:
        async def handle_request(self, scope, receive, send):
            headers = dict(scope.get("headers", []))
            incoming_session = headers.get(b"mcp-session-id", b"").decode()

            if not incoming_session:
                # Simulate initialize: create a new session and return its ID
                sessions[SESSION_ID] = True
                initialized_session.append(SESSION_ID)
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [
                            (b"mcp-session-id", SESSION_ID.encode()),
                            (b"content-type", b"application/json"),
                        ],
                    }
                )
                await send({"type": "http.response.body", "body": b"{}", "more_body": False})
            elif incoming_session in sessions:
                # Session known: serve the tool call
                await send(
                    {
                        "type": "http.response.start",
                        "status": 200,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"result": "ok"}',
                        "more_body": False,
                    }
                )
            else:
                # Session not found — this is the bug we're guarding against
                await send(
                    {
                        "type": "http.response.start",
                        "status": 404,
                        "headers": [(b"content-type", b"application/json")],
                    }
                )
                await send(
                    {
                        "type": "http.response.body",
                        "body": b'{"error": "Session not found"}',
                        "more_body": False,
                    }
                )

    monkeypatch.setattr(main, "streamable_http_session_manager", FakeSessionManager())
    # Disable API key auth so we can test the transport layer in isolation
    monkeypatch.setattr(main, "ARCHIVIST_API_KEY", "", raising=False)

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}

    init_statuses: list[int] = []
    tool_statuses: list[int] = []

    async def init_send(msg):
        if msg.get("type") == "http.response.start":
            init_statuses.append(msg["status"])

    async def tool_send(msg):
        if msg.get("type") == "http.response.start":
            tool_statuses.append(msg["status"])

    # Request 1: initialize (no session ID)
    init_scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "query_string": b"",
        "headers": [(b"content-type", b"application/json")],
    }
    await main.app(init_scope, receive, init_send)
    assert init_statuses == [200], f"initialize returned unexpected status: {init_statuses}"
    assert initialized_session, "no session was created by initialize"

    # Request 2: tool call with the session ID returned by initialize
    tool_scope = {
        "type": "http",
        "method": "POST",
        "path": "/mcp",
        "query_string": b"",
        "headers": [
            (b"content-type", b"application/json"),
            (b"mcp-session-id", SESSION_ID.encode()),
        ],
    }
    await main.app(tool_scope, receive, tool_send)
    assert tool_statuses == [200], (
        f"tool call returned {tool_statuses} — session was lost between requests "
        f"(BaseHTTPMiddleware regression, MCP SDK issue #883)"
    )


# ── Guard against re-introducing BaseHTTPMiddleware ──────────────────────────


def test_base_http_middleware_not_imported_in_main():
    """Ensure BaseHTTPMiddleware is never re-introduced into main.py.

    BaseHTTPMiddleware is incompatible with the MCP SDK's Streamable HTTP
    transport and causes every MCP tool call to return "Session not found"
    (MCP SDK issue #883).  If this test fails, the regression has been
    re-introduced.
    """
    import importlib
    import sys

    # Remove any cached module so we read the source fresh
    mod_name = "main"
    if mod_name in sys.modules:
        del sys.modules[mod_name]

    # Read the source directly instead of importing to avoid side-effects
    import ast
    import importlib.util

    spec = importlib.util.find_spec(mod_name)
    assert spec is not None and spec.origin is not None, "cannot locate main module"

    source = open(spec.origin).read()  # noqa: SIM115
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            node_src = ast.get_source_segment(source, node) or ""
            assert "BaseHTTPMiddleware" not in node_src, (
                "main.py imports BaseHTTPMiddleware — this breaks MCP session management "
                "(MCP SDK issue #883).  Use a pure ASGI middleware instead."
            )
