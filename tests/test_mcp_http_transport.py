"""Tests for MCP HTTP transport wiring in main.py."""

from contextlib import asynccontextmanager

import pytest
from starlette.routing import Route


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
