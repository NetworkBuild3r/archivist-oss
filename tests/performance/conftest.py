"""Performance-tier conftest.py — shared fixtures for performance benchmarks.

Performance tests use mocked I/O and assert wall-clock timing budgets.
They are excluded from default CI runs and deselected during development
(use ``pytest -m performance`` to run them explicitly).

Root-level fixtures are inherited from project root conftest.py.
"""

from __future__ import annotations

import pytest
from tests.fixtures.mocks import make_vector_backend_mock
from tests.fixtures.schema import build_schema


@pytest.fixture(scope="session", autouse=True)
def _warmup_lazy_imports() -> None:
    """Pre-import heavy optional modules before timed performance tests and stall checks."""
    try:
        import qdrant_client.models  # noqa: F401
    except ImportError:
        return


@pytest.fixture
async def qa_pool(tmp_path, monkeypatch):
    """Pool fixture matching the name from tests/qa/conftest.py.

    Performance tests were migrated from tests/qa/ and retain the ``qa_pool`` name.
    """
    from archivist.storage import sqlite_pool as _sp

    p = _sp.SQLitePool()
    db_path = str(tmp_path / "perf.db")
    await p.initialize(db_path)

    original_pool = _sp.pool
    monkeypatch.setattr(_sp, "pool", p)

    async with p.write() as conn:
        await build_schema(conn)

    yield p

    monkeypatch.setattr(_sp, "pool", original_pool)
    await p.close()


@pytest.fixture
async def perf_pool(tmp_path, monkeypatch):
    """Lightweight SQLitePool for performance tests."""
    from archivist.storage import sqlite_pool as _sp

    p = _sp.SQLitePool()
    db_path = str(tmp_path / "perf.db")
    await p.initialize(db_path)

    original_pool = _sp.pool
    monkeypatch.setattr(_sp, "pool", p)

    async with p.write() as conn:
        await build_schema(conn)

    yield p

    monkeypatch.setattr(_sp, "pool", original_pool)
    await p.close()


@pytest.fixture
def mock_vector_backend():
    """VectorBackend-compatible mock with no real Qdrant calls."""
    return make_vector_backend_mock()
