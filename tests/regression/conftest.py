"""Regression-tier conftest.py — shared fixtures for regression tests.

Regression tests pin specific bug-fix behaviours. They may use both
unit-style (no I/O) and light integration fixtures depending on what
the original bug involved.

Inherits root-level fixtures from project root conftest.py.
"""

from __future__ import annotations

import pytest
from tests.fixtures.factories import MemoryFactory


@pytest.fixture
def memory_factory():
    """Stateful factory returning deterministic memory payload dicts."""
    return MemoryFactory()
