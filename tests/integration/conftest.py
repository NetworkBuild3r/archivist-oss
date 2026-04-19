"""Integration-tier conftest.py — shared fixtures for integration tests.

Provides:
- ``integration_pool`` — async SQLitePool with full schema for integration tests
- ``mock_vector_backend`` — VectorBackend-protocol mock
- ``memory_factory`` — deterministic memory payload generator
- ``skip_on_postgres`` — pytest mark that skips tests incompatible with the
  Postgres backend (e.g. SQLite FTS5-only tests)

Root-level fixtures (``_isolate_env``, ``graph_db``, ``async_pool``,
``rbac_config``, ``mock_llm``) are inherited from the project root conftest.py.
"""

from __future__ import annotations

import os

import pytest
from tests.fixtures.factories import MemoryFactory
from tests.fixtures.mocks import make_vector_backend_mock
from tests.fixtures.schema import build_schema

# ---------------------------------------------------------------------------
# Backend detection helpers
# ---------------------------------------------------------------------------

_IS_POSTGRES = os.environ.get("GRAPH_BACKEND", "sqlite").strip().lower() == "postgres"

#: Decorator that skips a test when the Postgres backend is active.
#:
#: Use on tests that rely on SQLite FTS5 virtual tables (``memory_fts``,
#: ``memory_fts_exact``) or ``graph.get_db()`` — neither of which exists or
#: behaves the same on the asyncpg backend.
#:
#: Example::
#:
#:     @skip_on_postgres
#:     async def test_sqlite_fts_specific(async_pool): ...
skip_on_postgres = pytest.mark.skipif(
    _IS_POSTGRES,
    reason="SQLite FTS5 virtual tables are not present on the Postgres backend",
)


@pytest.fixture
async def integration_pool(tmp_path, monkeypatch):
    """Full-schema async pool for integration tests.

    Replaces the duplicate pool+schema setup that was in tests/qa/conftest.py.
    Patches ``archivist.storage.sqlite_pool.pool`` for the duration of the test.
    """
    from archivist.storage import graph as _graph
    from archivist.storage import sqlite_pool as _sp

    p = _sp.SQLitePool()
    db_path = str(tmp_path / "integration.db")
    await p.initialize(db_path)

    original_pool = _sp.pool
    monkeypatch.setattr(_sp, "pool", p)

    for attr in dir(_graph):
        obj = getattr(_graph, attr, None)
        if hasattr(obj, "reset") and callable(obj.reset):
            obj.reset()
    if hasattr(_graph, "_ensure_needle_registry"):
        _graph._ensure_needle_registry.applied = True  # type: ignore[attr-defined]

    async with p.write() as conn:
        await build_schema(conn)

    yield p

    monkeypatch.setattr(_sp, "pool", original_pool)
    await p.close()


@pytest.fixture
async def qa_pool(tmp_path, monkeypatch):
    """Alias for the pool fixture matching the fixture name from tests/qa/conftest.py.

    Integration tests migrated from tests/qa/ retain the ``qa_pool`` name.
    Provides the same full-schema isolated pool as ``integration_pool``.
    """
    from archivist.storage import graph as _graph
    from archivist.storage import sqlite_pool as _sp

    p = _sp.SQLitePool()
    db_path = str(tmp_path / "qa_integration.db")
    await p.initialize(db_path)

    original_pool = _sp.pool
    monkeypatch.setattr(_sp, "pool", p)

    for attr in dir(_graph):
        obj = getattr(_graph, attr, None)
        if hasattr(obj, "reset") and callable(obj.reset):
            obj.reset()
    if hasattr(_graph, "_ensure_needle_registry"):
        _graph._ensure_needle_registry.applied = True  # type: ignore[attr-defined]

    async with p.write() as conn:
        await build_schema(conn)

    yield p

    monkeypatch.setattr(_sp, "pool", original_pool)
    await p.close()


@pytest.fixture
def mock_vector_backend():
    """VectorBackend-compatible mock. No real Qdrant calls."""
    return make_vector_backend_mock()


@pytest.fixture
def memory_factory():
    """Stateful factory returning deterministic memory payload dicts."""
    return MemoryFactory()
