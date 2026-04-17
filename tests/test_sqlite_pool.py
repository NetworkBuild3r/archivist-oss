"""Tests for the SQLitePool async connection layer (v1.12)."""

import pytest


@pytest.mark.asyncio
async def test_pool_initialize_and_close(tmp_path):
    """Pool opens and closes cleanly."""
    from archivist.storage.sqlite_pool import SQLitePool

    p = SQLitePool()
    db = str(tmp_path / "test.db")
    await p.initialize(db)
    assert p._conn is not None
    await p.close()
    assert p._conn is None


@pytest.mark.asyncio
async def test_pool_initialize_idempotent(tmp_path):
    """Calling initialize() twice does not raise or create a second connection."""
    from archivist.storage.sqlite_pool import SQLitePool

    p = SQLitePool()
    db = str(tmp_path / "test.db")
    await p.initialize(db)
    conn_before = p._conn
    await p.initialize(db)
    assert p._conn is conn_before
    await p.close()


@pytest.mark.asyncio
async def test_write_commits_on_success(tmp_path):
    """pool.write() auto-commits on clean exit."""
    from archivist.storage.sqlite_pool import SQLitePool

    p = SQLitePool()
    await p.initialize(str(tmp_path / "test.db"))

    async with p.write() as conn:
        await conn.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")
        await conn.execute("INSERT INTO t VALUES (42)")

    async with p.read() as conn:
        cur = await conn.execute("SELECT v FROM t")
        row = await cur.fetchone()
    assert row[0] == 42
    await p.close()


@pytest.mark.asyncio
async def test_write_rolls_back_on_exception(tmp_path):
    """pool.write() rolls back on exception and re-raises."""
    from archivist.storage.sqlite_pool import SQLitePool

    p = SQLitePool()
    await p.initialize(str(tmp_path / "test.db"))

    async with p.write() as conn:
        await conn.execute("CREATE TABLE IF NOT EXISTS t (v INTEGER)")

    with pytest.raises(RuntimeError, match="intentional"):
        async with p.write() as conn:
            await conn.execute("INSERT INTO t VALUES (99)")
            raise RuntimeError("intentional")

    async with p.read() as conn:
        cur = await conn.execute("SELECT COUNT(*) FROM t")
        row = await cur.fetchone()
    assert row[0] == 0
    await p.close()


@pytest.mark.asyncio
async def test_read_raises_when_not_initialized():
    """pool.read() raises RuntimeError before initialize()."""
    from archivist.storage.sqlite_pool import SQLitePool

    p = SQLitePool()
    with pytest.raises(RuntimeError, match="not initialized"):
        async with p.read() as _:
            pass


@pytest.mark.asyncio
async def test_write_raises_when_not_initialized():
    """pool.write() raises RuntimeError before initialize()."""
    from archivist.storage.sqlite_pool import SQLitePool

    p = SQLitePool()
    with pytest.raises(RuntimeError, match="not initialized"):
        async with p.write() as _:
            pass


@pytest.mark.asyncio
async def test_concurrent_writes_serialized(tmp_path):
    """Concurrent writes complete without data corruption (lock serialization)."""
    import asyncio

    from archivist.storage.sqlite_pool import SQLitePool

    p = SQLitePool()
    await p.initialize(str(tmp_path / "test.db"))

    async with p.write() as conn:
        await conn.execute("CREATE TABLE IF NOT EXISTS counter (n INTEGER)")
        await conn.execute("INSERT INTO counter VALUES (0)")

    async def increment():
        async with p.write() as conn:
            cur = await conn.execute("SELECT n FROM counter")
            row = await cur.fetchone()
            await conn.execute("UPDATE counter SET n = ?", (row[0] + 1,))

    await asyncio.gather(*[increment() for _ in range(20)])

    async with p.read() as conn:
        cur = await conn.execute("SELECT n FROM counter")
        row = await cur.fetchone()
    assert row[0] == 20
    await p.close()
