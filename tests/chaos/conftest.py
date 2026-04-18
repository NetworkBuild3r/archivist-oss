"""Chaos-tier conftest.py — shared fixtures for chaos/fault-injection tests.

Chaos tests exercise adversarial scenarios: concurrent writers, I/O
failures, database corruption, timeout-under-load. They may share the
qa_pool fixture pattern from the system tier.

Root-level fixtures are inherited from project root conftest.py.
"""

from __future__ import annotations

import pytest
from tests.fixtures.factories import MemoryFactory
from tests.fixtures.mocks import make_vector_backend_mock
from tests.fixtures.schema import build_schema


@pytest.fixture
async def qa_pool(tmp_path, monkeypatch):
    """Alias for the test pool fixture — matches the fixture name from tests/qa/conftest.py.

    Chaos tests were migrated from tests/qa/ and retain the ``qa_pool`` name.
    """
    from archivist.storage import sqlite_pool as _sp

    p = _sp.SQLitePool()
    db_path = str(tmp_path / "chaos.db")
    await p.initialize(db_path)

    original_pool = _sp.pool
    monkeypatch.setattr(_sp, "pool", p)

    async with p.write() as conn:
        await build_schema(conn)

    yield p

    monkeypatch.setattr(_sp, "pool", original_pool)
    await p.close()


@pytest.fixture
async def chaos_pool(tmp_path, monkeypatch):
    """Isolated SQLitePool for chaos tests — same construction as qa_pool."""
    from archivist.storage import sqlite_pool as _sp

    p = _sp.SQLitePool()
    db_path = str(tmp_path / "chaos.db")
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


@pytest.fixture
def memory_factory():
    """Stateful factory returning deterministic memory payload dicts."""
    return MemoryFactory()
