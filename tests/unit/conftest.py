"""Unit-tier conftest.py — shared fixtures for unit tests.

Unit tests must not touch real I/O.  This conftest provides lightweight
helpers only — no real SQLite pools or Qdrant clients.

Root-level fixtures (``_isolate_env``, ``mock_llm``) are inherited from
the project root conftest.py.
"""

from __future__ import annotations

import pytest
from tests.fixtures.factories import MemoryFactory
from tests.fixtures.mocks import make_embed_mock, make_vector_backend_mock


@pytest.fixture
def memory_factory():
    """Stateful factory returning deterministic memory payload dicts."""
    return MemoryFactory()


@pytest.fixture
def vector_backend():
    """VectorBackend-compatible mock with no real Qdrant calls."""
    return make_vector_backend_mock()


@pytest.fixture
def embed_fn():
    """Async mock embedding function returning a 1024-float vector."""
    return make_embed_mock(dim=1024)
