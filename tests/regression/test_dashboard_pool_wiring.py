"""Regression tests for the dashboard/tools_admin stale pool-singleton import bug.

Before ``INIT-026/SPEC-001``, ``dashboard.py`` and ``tools_admin.py`` each imported
``pool`` at MODULE level (``from archivist.storage.sqlite_pool import pool``),
binding the name once — at module-import time — to the placeholder
``SQLiteGraphBackend()`` instance defined in ``sqlite_pool.py``, which is never
itself initialized. Real startup (``main.py``) rebinds
``archivist.storage.sqlite_pool.pool`` to a freshly initialized backend *after*
``dashboard.py``/``tools_admin.py`` have already loaded — a module-level import in
a *consumer* module can never observe a later rebind of the name it copied a
reference from. Every call to ``pool.read()``/``pool.write()`` in those two files
raised ``RuntimeError: SQLitePool is not initialized`` on the default (SQLite)
backend, 100% reproducibly, breaking ``GET /admin/dashboard`` and the
``archivist_health_dashboard`` / ``archivist_savings_dashboard`` MCP tools.

``_reimport_dashboard_before_pool_init`` below reproduces that exact ordering
deterministically — reload ``dashboard``/``tools_admin`` while
``sqlite_pool.pool`` is still the pristine, uninitialized placeholder, THEN
initialize + rebind the real pool — rather than relying on Python's module cache
and whatever order other test files happened to import these modules in (which
would make the tests pass or fail depending on test *collection* order, not on
whether the bug is actually fixed).
"""

from __future__ import annotations

import importlib
import json
import pathlib
import re

import pytest

pytestmark = [pytest.mark.regression, pytest.mark.storage]


@pytest.fixture
async def dashboard_pool(tmp_path, monkeypatch):
    """Deterministically reproduce real startup ordering, then initialize the pool.

    1. Reset ``sqlite_pool.pool`` to a pristine, uninitialized instance.
    2. Reload ``dashboard``/``tools_admin`` so their imports re-run against that
       pristine state — exactly like process startup, where these modules load
       long before ``main.py``'s async startup handler runs.
    3. Only THEN initialize a real backend and rebind ``sqlite_pool.pool`` to it
       (mirrors ``main.py``'s ``_pool_module.pool = backend``).

    Yields the initialized backend. Callers still re-import
    ``dashboard``/``tools_admin`` (already reloaded) to get the live module
    objects for the functions under test.
    """
    import archivist.app.dashboard as dashboard_module
    import archivist.app.handlers.tools_admin as tools_admin_module
    import archivist.storage.graph as graph
    from archivist.storage import sqlite_pool as _pool_mod

    monkeypatch.setattr(_pool_mod, "pool", _pool_mod.SQLiteGraphBackend())
    importlib.reload(dashboard_module)
    importlib.reload(tools_admin_module)

    db_path = str(tmp_path / "test_dashboard_pool.db")
    real_backend = _pool_mod.SQLiteGraphBackend()
    await real_backend.initialize(db_path)
    monkeypatch.setattr(_pool_mod, "pool", real_backend)
    # get_db()/init_schema() import SQLITE_PATH from config at call time — patching
    # only graph.SQLITE_PATH (a re-export) does not redirect DDL (INIT-002/SPEC-003).
    monkeypatch.setattr("archivist.core.config.SQLITE_PATH", db_path)
    monkeypatch.setattr(graph, "SQLITE_PATH", db_path)
    graph.init_schema()

    yield real_backend

    await real_backend.close()


class TestDashboardPoolWiring:
    """``dashboard.build_dashboard()`` must resolve the pool initialized at
    (simulated) startup time, not a stale reference captured when the module
    was first imported."""

    async def test_build_dashboard_succeeds_with_initialized_pool(self, dashboard_pool):
        import archivist.app.dashboard as dashboard_module

        result = await dashboard_module.build_dashboard(window_days=7)

        assert isinstance(result, dict)
        assert "generated_at" in result
        assert "conflicts" in result
        assert "retrieval" in result
        assert "skills" in result

    async def test_hotness_heatmap_returns_seeded_row(self, dashboard_pool):
        """Content-based assertion, not just "did not raise": ``_hotness_heatmap``
        wraps its own ``pool.read()`` in a broad ``try/except`` that returns ``[]``
        on ANY failure — including the pre-fix ``RuntimeError`` — so a bare
        "does it raise" check would pass even against the buggy code. Seeding a
        real row and asserting it comes back is what actually discriminates
        pre-fix (silently swallowed -> ``[]``) from post-fix (real data)."""
        async with dashboard_pool.write() as conn:
            await conn.execute(
                "INSERT INTO memory_hotness (memory_id, score, retrieval_count) VALUES (?, ?, ?)",
                ("regression-test-memory-1", 0.9, 3),
            )

        import archivist.app.dashboard as dashboard_module

        result = await dashboard_module._hotness_heatmap(top_n=5)

        assert len(result) == 1, f"expected the seeded row, got: {result}"
        assert result[0]["memory_id"] == "regression-test-memory-1"
        assert result[0]["score"] == 0.9


class TestDashboardMCPToolsPoolWiring:
    """The MCP admin tools that read ``pool`` (directly or via ``dashboard.py``)
    must return a success response, not raise, once the pool is initialized."""

    async def test_health_dashboard_tool_succeeds(self, dashboard_pool):
        import archivist.app.handlers.tools_admin as tools_admin_module

        response = await tools_admin_module._handle_health_dashboard({"window_days": 7})

        payload = json.loads(response[0].text)
        assert "error" not in payload
        assert "generated_at" in payload

    async def test_savings_dashboard_tool_succeeds(self, dashboard_pool):
        """This handler does its own ``pool.read()`` directly in
        ``tools_admin.py`` (not only via ``dashboard.py``) — the second of the
        two files this bug affected."""
        import archivist.app.handlers.tools_admin as tools_admin_module

        response = await tools_admin_module._handle_savings_dashboard(
            {"window_days": 7, "heatmap_top_n": 10}
        )

        payload = json.loads(response[0].text)
        assert "error" not in payload
        assert payload["window_days"] == 7
        assert "token_savings" in payload
        assert "tier_distribution" in payload
        assert payload["hotness_heatmap"] == []


def test_no_module_level_pool_import_anywhere():
    """Structural guard (ac-5): no file may reintroduce a module-level
    ``from archivist.storage.sqlite_pool import pool``. Every consumer must
    import it inside the function that uses it, so it always resolves the pool
    object live at call time — see ``dashboard.build_dashboard()`` for the
    pattern every other consumer in this codebase already followed
    (``INIT-026/SPEC-001``)."""
    src_root = pathlib.Path(__file__).resolve().parents[2] / "src" / "archivist"
    pattern = re.compile(r"^from archivist\.storage\.sqlite_pool import pool\b", re.MULTILINE)

    offenders = sorted(
        str(path.relative_to(src_root))
        for path in src_root.rglob("*.py")
        if pattern.search(path.read_text())
    )

    assert offenders == [], (
        "Module-level `from archivist.storage.sqlite_pool import pool` found in: "
        f"{offenders}. Import it inside the function that uses it instead."
    )
