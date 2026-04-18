"""Contract tests for _get_graph_write_lock() and pool.write() loop-safety.

Gap that prompted this file
---------------------------
The QA conftest.py previously monkeypatched ``GRAPH_WRITE_LOCK_ASYNC`` with a
fresh ``asyncio.Lock()`` for every test.  That workaround hid the root bug: the
module-level lock was created at import time and became stale when
``pytest-asyncio`` (asyncio_mode="auto") gave each test a fresh event loop.

The downstream symptom was ``test_sqlite_pool.py::test_concurrent_writes_serialized``
failing in CI with::

    RuntimeError: <asyncio.locks.Lock object at 0x...> is bound to a different event loop

The conftest monkeypatch masked the failure for the QA suite itself but did
nothing to protect the existing ``tests/test_sqlite_pool.py`` tests, which
don't use the QA conftest.  CI runs the whole ``tests/`` folder; the hidden bug
surfaced there even after the QA suite passed in isolation.

The fix — a lazy ``_get_graph_write_lock()`` accessor that recreates the lock
when the running loop differs from the lock's bound loop — is tested here
explicitly so any regression is caught locally before push.
"""

from __future__ import annotations

import asyncio

import pytest

# ---------------------------------------------------------------------------
# _get_graph_write_lock() unit contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_graph_write_lock_returns_a_lock():
    """_get_graph_write_lock() returns an asyncio.Lock inside a running loop."""
    from archivist.storage.sqlite_pool import _get_graph_write_lock

    lock = _get_graph_write_lock()
    assert isinstance(lock, asyncio.Lock)


@pytest.mark.asyncio
async def test_get_graph_write_lock_is_idempotent_within_same_loop():
    """Multiple calls within the same event loop return the *same* object."""
    from archivist.storage import sqlite_pool as _sp

    # Force creation of the lock against the current loop.
    lock1 = _sp._get_graph_write_lock()
    # Acquire it so its ``_loop`` attribute is populated.
    async with lock1:
        pass
    lock2 = _sp._get_graph_write_lock()
    assert lock1 is lock2, "Lock object should be reused within the same event loop"


@pytest.mark.asyncio
async def test_get_graph_write_lock_rebinds_when_loop_changes():
    """When the module-level lock belongs to a *different* loop the accessor
    creates a new lock bound to the current loop.

    This is the exact scenario that caused the CI failure: the lock was
    created (and its ``_loop`` set) by a previous test's event loop, then the
    next test's loop was different but ``_get_graph_write_lock()`` would have
    returned the stale lock.  We simulate this by:
    1. Creating a lock and manually setting its ``_loop`` to a *closed* loop.
    2. Calling ``_get_graph_write_lock()`` from the current (different) loop.
    3. Asserting that a new lock is returned.
    """
    from archivist.storage import sqlite_pool as _sp

    current_loop = asyncio.get_running_loop()

    # Build a dummy "stale" lock whose _loop points to a different (closed) loop.
    stale_loop = asyncio.new_event_loop()
    stale_loop.close()
    stale_lock = asyncio.Lock()
    object.__setattr__(stale_lock, "_loop", stale_loop)

    # Inject the stale lock as the module-level singleton.
    original = _sp._GRAPH_WRITE_LOCK
    _sp._GRAPH_WRITE_LOCK = stale_lock
    try:
        fresh_lock = _sp._get_graph_write_lock()
        assert fresh_lock is not stale_lock, (
            "_get_graph_write_lock() should return a NEW lock when the existing "
            "lock is bound to a different event loop"
        )
        assert isinstance(fresh_lock, asyncio.Lock)
        # Verify the fresh lock is usable in the current loop.
        async with fresh_lock:
            pass
        # Verify the module-level variable was updated.
        assert _sp._GRAPH_WRITE_LOCK is fresh_lock
    finally:
        _sp._GRAPH_WRITE_LOCK = original


# ---------------------------------------------------------------------------
# pool.write() cross-test isolation (the actual CI regression scenario)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pool_write_acquires_lock_without_monkeypatch(tmp_path):
    """pool.write() works with *no* monkeypatching of _GRAPH_WRITE_LOCK.

    The old QA conftest masked the event-loop bug by monkeypatching the lock.
    This test deliberately does NOT monkeypatch anything, verifying the fix
    works at the source rather than relying on test-level workarounds.
    """
    from archivist.storage.sqlite_pool import SQLitePool

    p = SQLitePool()
    await p.initialize(str(tmp_path / "no_patch.db"))
    try:
        async with p.write() as conn:
            await conn.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
            await conn.execute("INSERT INTO t VALUES (1)")
        async with p.read() as conn:
            cur = await conn.execute("SELECT v FROM t")
            row = await cur.fetchone()
        assert row[0] == 1
    finally:
        await p.close()


@pytest.mark.asyncio
async def test_pool_write_concurrent_serialized_no_monkeypatch(tmp_path):
    """Concurrent writes via pool.write() are serialized correctly.

    This is ``test_sqlite_pool.py::test_concurrent_writes_serialized``
    duplicated here to give the QA suite direct ownership of the regression.
    It would have caught the CI failure had it been present before.
    """
    p_local = __import__("archivist.storage.sqlite_pool", fromlist=["SQLitePool"]).SQLitePool()
    await p_local.initialize(str(tmp_path / "concurrent.db"))

    async with p_local.write() as conn:
        await conn.execute("CREATE TABLE IF NOT EXISTS counter (n INTEGER)")
        await conn.execute("INSERT INTO counter VALUES (0)")

    async def increment() -> None:
        async with p_local.write() as conn:
            cur = await conn.execute("SELECT n FROM counter")
            row = await cur.fetchone()
            await conn.execute("UPDATE counter SET n = ?", (row[0] + 1,))

    await asyncio.gather(*[increment() for _ in range(20)])

    async with p_local.read() as conn:
        cur = await conn.execute("SELECT n FROM counter")
        row = await cur.fetchone()
    assert row[0] == 20
    await p_local.close()


# ---------------------------------------------------------------------------
# Legacy GRAPH_WRITE_LOCK_ASYNC alias sanity check
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_legacy_alias_is_a_lock():
    """GRAPH_WRITE_LOCK_ASYNC is still exported and is an asyncio.Lock.

    External callers (backup_manager, any custom scripts) may still import
    this name.  Ensure it remains importable and is the correct type.
    """
    from archivist.storage.sqlite_pool import GRAPH_WRITE_LOCK_ASYNC

    assert isinstance(GRAPH_WRITE_LOCK_ASYNC, asyncio.Lock), (
        "GRAPH_WRITE_LOCK_ASYNC must remain an asyncio.Lock for backward compatibility"
    )
