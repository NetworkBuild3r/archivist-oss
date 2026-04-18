"""System-tier conftest.py — shared fixtures for system/MCP handler tests.

Provides:
- ``qa_pool`` — isolated async SQLitePool with full schema, mirrors what
  was previously in tests/qa/conftest.py
- ``mock_vector_backend`` — VectorBackend-protocol mock (no real Qdrant)
- ``memory_factory`` — deterministic memory payload generator
- ``_enable_outbox`` (autouse) — forces OUTBOX_ENABLED=True
"""

from __future__ import annotations

import pytest
from tests.fixtures.factories import MemoryFactory
from tests.fixtures.mocks import make_vector_backend_mock
from tests.fixtures.schema import build_schema


@pytest.fixture
async def qa_pool(tmp_path, monkeypatch):
    """Fresh SQLitePool per test, backed by a temp-file DB with full schema.

    Patches ``archivist.storage.sqlite_pool.pool`` so MemoryTransaction and
    OutboxProcessor use this isolated pool.  Loop-aware lock is used, so no
    explicit lock monkeypatching is required.
    """
    from archivist.storage import graph as _graph
    from archivist.storage import sqlite_pool as _sp

    p = _sp.SQLitePool()
    db_path = str(tmp_path / "qa_test.db")
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


@pytest.fixture(autouse=True)
def _enable_outbox(monkeypatch):
    """Force OUTBOX_ENABLED=True for every system-tier test."""
    import archivist.core.config as _cfg

    monkeypatch.setattr(_cfg, "OUTBOX_ENABLED", True)
    monkeypatch.setattr(_cfg, "OUTBOX_MAX_RETRIES", 5)
    monkeypatch.setattr(_cfg, "OUTBOX_BATCH_SIZE", 50)
    monkeypatch.setattr(_cfg, "OUTBOX_DRAIN_INTERVAL", 1)


@pytest.fixture
def mock_vector_backend():
    """VectorBackend-compatible mock. No real Qdrant calls."""
    return make_vector_backend_mock()


@pytest.fixture
def memory_factory():
    """Stateful factory returning deterministic memory payload dicts."""
    return MemoryFactory()
