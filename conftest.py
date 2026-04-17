"""Root conftest — shared fixtures for Archivist tests.

Sets up in-memory SQLite, mock LLM, and isolated config for unit tests.
"""

import os
from unittest.mock import AsyncMock, patch

import pytest


@pytest.fixture(autouse=True)
def _isolate_env(monkeypatch, tmp_path):
    """Isolate every test from real config and filesystem state."""
    db_path = str(tmp_path / "graph.db")
    monkeypatch.setenv("MEMORY_ROOT", str(tmp_path / "memories"))
    monkeypatch.setenv("SQLITE_PATH", db_path)
    monkeypatch.setenv("NAMESPACES_CONFIG_PATH", str(tmp_path / "ns.yaml"))
    monkeypatch.setenv("QDRANT_URL", "http://localhost:6333")
    monkeypatch.setenv("LLM_URL", "http://localhost:4000")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("EMBED_URL", "http://localhost:4000")
    monkeypatch.setenv("BM25_ENABLED", "false")
    os.makedirs(str(tmp_path / "memories"), exist_ok=True)

    import config
    import graph

    monkeypatch.setattr(config, "SQLITE_PATH", db_path)
    monkeypatch.setattr(graph, "SQLITE_PATH", db_path)
    graph.init_schema()

    _schema_guards = [
        ("graph", "_ensure_needle_registry"),
        ("trajectory", "_ensure_trajectory_schema"),
        ("skills", "_ensure_skill_schema"),
        ("curator_queue", "_ensure_schema"),
        ("retrieval_log", "_ensure_schema"),
        ("hotness", "_ensure_schema"),
        ("audit", "_ensure_audit_schema"),
    ]
    for mod_name, guard_attr in _schema_guards:
        try:
            mod = __import__(mod_name)
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

    import config

    importlib.reload(config)
    monkeypatch.setattr("config.SQLITE_PATH", db_path)

    import graph

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

    import config

    importlib.reload(config)
    monkeypatch.setattr("config.SQLITE_PATH", db_path)

    import graph
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

    import rbac

    importlib.reload(rbac)
    rbac._config = None
    rbac._permissive_fallback = False
    rbac.load_config(str(config_yaml))
    return rbac


@pytest.fixture
def mock_llm():
    """Patch llm.llm_query to return a canned response without HTTP calls."""
    with patch("llm.llm_query", new_callable=AsyncMock) as mock:
        mock.return_value = "Mocked LLM response."
        yield mock
