"""Root conftest — shared fixtures for Archivist tests.

Sets up in-memory SQLite, mock LLM, isolated config, and the asyncio >50ms
event-loop stall watchdog (INIT-025/SPEC-012).
"""

import asyncio
import inspect
import os
from unittest.mock import AsyncMock, patch

import pytest
from tests.fixtures.event_loop_watchdog import (
    DEFAULT_STALL_THRESHOLD_MS,
    EventLoopStallError,
    EventLoopStallWatchdog,
)


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register event-loop stall watchdog CLI options."""
    env_enabled = os.getenv("ARCHIVIST_STALL_WATCHDOG", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    env_threshold = float(
        os.getenv("ARCHIVIST_STALL_THRESHOLD_MS", str(DEFAULT_STALL_THRESHOLD_MS))
    )
    parser.addoption(
        "--stall-watchdog",
        action="store_true",
        default=env_enabled,
        help="Fail async tests when any asyncio event-loop callback blocks for >50ms.",
    )
    parser.addoption(
        "--stall-threshold-ms",
        action="store",
        type=float,
        default=env_threshold,
        help="Threshold in milliseconds for the asyncio event-loop stall watchdog (default: 50ms).",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Attach the event-loop stall watchdog to async tests when enabled or marked performance."""
    watchdog_enabled = bool(config.getoption("--stall-watchdog", default=False))
    for item in items:
        if not isinstance(item, pytest.Function):
            continue
        if item.get_closest_marker("allow_event_loop_stall") is not None:
            continue
        if "event_loop_watchdog" in item.fixturenames:
            continue
        if not inspect.iscoroutinefunction(getattr(item, "obj", None)):
            continue
        if watchdog_enabled or item.get_closest_marker("performance") is not None:
            item.fixturenames.append("_auto_stall_watchdog")


@pytest.fixture
async def event_loop_watchdog(request: pytest.FixtureRequest):
    """Fail the async test if any event-loop callback stalls beyond threshold (default 50ms)."""
    threshold_ms = float(
        request.config.getoption("--stall-threshold-ms", default=DEFAULT_STALL_THRESHOLD_MS)
    )
    watchdog = EventLoopStallWatchdog(threshold_ms=threshold_ms)
    watchdog.start()
    try:
        yield watchdog
        # Yield once so the callback finishing the test body completes and logs any stall.
        await asyncio.sleep(0)
    finally:
        stalls = watchdog.stop()
    if stalls:
        raise EventLoopStallError(stalls, threshold_ms)


@pytest.fixture
async def _auto_stall_watchdog(request: pytest.FixtureRequest):
    """Auto-attached stall watchdog for async tests when --stall-watchdog or performance is active."""
    threshold_ms = float(
        request.config.getoption("--stall-threshold-ms", default=DEFAULT_STALL_THRESHOLD_MS)
    )
    watchdog = EventLoopStallWatchdog(threshold_ms=threshold_ms)
    watchdog.start()
    try:
        yield watchdog
        await asyncio.sleep(0)
    finally:
        stalls = watchdog.stop()
    if stalls:
        raise EventLoopStallError(stalls, threshold_ms)


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Isolate every test from real config and filesystem state."""
    db_path = str(tmp_path / "graph.db")
    monkeypatch.setenv("MEMORY_ROOT", str(tmp_path / "memories"))
    monkeypatch.setenv("SQLITE_PATH", db_path)
    monkeypatch.setenv("NAMESPACES_CONFIG_PATH", "")
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("LLM_URL", "http://localhost:4000")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBED_URL", "http://localhost:4000")
    monkeypatch.setenv("BM25_ENABLED", "false")
    os.makedirs(str(tmp_path / "memories"), exist_ok=True)

    import archivist.core.config as config
    import archivist.core.rbac as rbac
    import archivist.storage.graph as graph

    monkeypatch.setattr(config, "SQLITE_PATH", db_path)
    monkeypatch.setattr(config, "NAMESPACES_CONFIG_PATH", "")
    monkeypatch.setattr(rbac, "NAMESPACES_CONFIG_PATH", "")
    rbac._config = None
    rbac._permissive_fallback = False
    monkeypatch.setattr(graph, "SQLITE_PATH", db_path)
    graph.init_schema()

    _schema_guards = [
        ("archivist.storage.graph", "_ensure_needle_registry"),
        ("archivist.core.trajectory", "_ensure_trajectory_schema"),
        ("archivist.lifecycle.curator_queue", "_ensure_schema"),
        ("archivist.retrieval.retrieval_log", "_ensure_schema"),
        ("archivist.core.hotness", "_ensure_schema"),
        ("archivist.core.audit", "_ensure_audit_schema"),
    ]
    import importlib

    for mod_name, guard_attr in _schema_guards:
        try:
            mod = importlib.import_module(mod_name)
            guard = getattr(mod, guard_attr, None)
            if guard is not None and hasattr(guard, "reset"):
                guard.reset()
        except ImportError:
            pass


@pytest.fixture
def graph_db(tmp_path, monkeypatch):
    """Provide an initialized in-memory-like SQLite graph database.

    Patches graph.SQLITE_PATH to use a temp file, then calls init_schema().
    Returns the path for direct inspection.
    """
    db_path = str(tmp_path / "test_graph.db")
    monkeypatch.setenv("SQLITE_PATH", db_path)

    import importlib

    import archivist.core.config as config

    importlib.reload(config)
    monkeypatch.setattr("archivist.core.config.SQLITE_PATH", db_path)

    import archivist.storage.graph as graph

    monkeypatch.setattr(graph, "SQLITE_PATH", db_path)
    graph.init_schema()
    return db_path


@pytest.fixture
async def async_pool(tmp_path, monkeypatch):
    """Async fixture: initialize the SQLitePool singleton for async graph tests.

    Tears down the pool after each test to prevent state bleed between tests.
    Requires ``pytest-asyncio`` with asyncio_mode = "auto" or explicit marking.
    """
    db_path = str(tmp_path / "test_async.db")
    monkeypatch.setenv("SQLITE_PATH", db_path)

    import importlib

    import archivist.core.config as config

    importlib.reload(config)
    monkeypatch.setattr("archivist.core.config.SQLITE_PATH", db_path)

    import archivist.storage.graph as graph
    from archivist.storage import sqlite_pool as _pool_mod

    monkeypatch.setattr(graph, "SQLITE_PATH", db_path)
    monkeypatch.setattr(_pool_mod, "pool", _pool_mod.SQLitePool())
    await _pool_mod.pool.initialize(db_path)
    graph.init_schema()

    yield _pool_mod.pool

    await _pool_mod.pool.close()


@pytest.fixture
def rbac_config(tmp_path, monkeypatch):
    """Write a test namespaces.yaml and load RBAC from it."""
    config_yaml = tmp_path / "namespaces.yaml"
    config_yaml.write_text("""\
namespaces:
  - id: chief
    read: [chief, gitbob, all_readers]
    write: [chief]
    consistency: strong
  - id: pipeline
    read: [gitbob, grafgreg, chief]
    write: [gitbob, grafgreg]
  - id: deployer
    read: [argo, kubekate, chief]
    write: [argo, kubekate]
  - id: shared
    read: [all]
    write: [all]

agent_namespaces:
  chief: chief
  gitbob: pipeline
  grafgreg: pipeline
  argo: deployer
  kubekate: deployer
""")
    monkeypatch.setenv("NAMESPACES_CONFIG_PATH", str(config_yaml))

    import importlib

    import archivist.core.rbac as rbac

    importlib.reload(rbac)
    rbac._config = None
    rbac._permissive_fallback = False
    rbac.load_config(str(config_yaml))
    try:
        yield rbac
    finally:
        rbac.NAMESPACES_CONFIG_PATH = ""
        rbac._config = None
        rbac._permissive_fallback = False


@pytest.fixture
def mock_llm():
    """Patch llm.llm_query to return a canned response without HTTP calls."""
    with patch("archivist.features.llm.llm_query", new_callable=AsyncMock) as mock:
        mock.return_value = "Mocked LLM response."
        yield mock
