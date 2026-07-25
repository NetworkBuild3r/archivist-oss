"""Unit tests for INIT-022/SPEC-004's event-loop offload fixes in ``cascade.py``.

Covers the three acceptance criteria that motivated this spec:

- ac-1 (``H2``): ``_qdrant_delete`` offloads its blocking core
  (``_qdrant_delete_sync``) via ``asyncio.to_thread`` so a concurrent
  coroutine can make progress while a slow Qdrant delete is in-flight.
- ac-2 (``M1``): ``sweep_orphans()`` offloads its blocking
  ``client.get_collections`` / ``client.retrieve`` calls the same way.
- ac-3 (``M4``): ``_delete_sqlite_artifacts`` (in ``memory_lifecycle.py``)
  genuinely honors its ``txn`` parameter — see
  ``tests/unit/lifecycle/test_memory_lifecycle.py`` for the parity test.

These are concurrency-proof tests, not merely call-shape assertions: each
test starts a "ticker" background task that increments a counter on a tight
``asyncio.sleep(0)`` loop, then runs the offloaded call with a *synchronous*
``time.sleep`` stand-in for the blocking Qdrant SDK call. If the blocking
call were not actually offloaded to a thread, it would run directly on the
event loop and starve the ticker for its full duration — so a healthy tick
count is direct evidence the event loop stayed responsive.
"""

import asyncio
import time
from contextlib import asynccontextmanager
from unittest.mock import MagicMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.lifecycle]

# How long the simulated blocking Qdrant call sleeps for, and how often the
# ticker wakes up. Chosen so a healthy offload comfortably produces >= 5
# ticks while keeping the test fast (~0.2s wall-clock).
_SLOW_CALL_SECONDS = 0.2
_TICK_INTERVAL_SECONDS = 0.01
_MIN_EXPECTED_TICKS = 5


async def _run_with_ticker(coro):
    """Run *coro* concurrently with a ticker task; return (result, tick_count).

    The ticker increments a counter on every ``asyncio.sleep(0.01)`` cycle.
    A low tick count means something blocked the event loop during *coro*.
    """
    tick_count = 0
    stop = False

    async def ticker():
        nonlocal tick_count
        while not stop:
            tick_count += 1
            await asyncio.sleep(_TICK_INTERVAL_SECONDS)

    ticker_task = asyncio.create_task(ticker())
    try:
        result = await coro
    finally:
        stop = True
        await ticker_task

    return result, tick_count


class TestQdrantDeleteOffload:
    """ac-1 (H2): _qdrant_delete offloads its blocking core via asyncio.to_thread."""

    async def test_concurrent_task_progresses_during_slow_delete(self):
        """A slow client.delete() must not block a concurrently-scheduled coroutine."""
        from archivist.lifecycle.cascade import _qdrant_delete

        client = MagicMock()
        client.count.return_value = MagicMock(count=0)

        def _slow_delete(**kwargs):
            # Synchronous, blocking sleep — mirrors the real qdrant_client
            # SDK, which is 100% synchronous. If _qdrant_delete failed to
            # offload this via asyncio.to_thread, this sleep would run
            # directly on the event loop.
            time.sleep(_SLOW_CALL_SECONDS)

        client.delete.side_effect = _slow_delete

        failed: list[str] = []
        result, tick_count = await _run_with_ticker(
            _qdrant_delete(client, "col", ["p1"], "qdrant_primary", "mem-1", failed)
        )

        assert failed == []
        assert result == 1  # len(["p1"])
        assert tick_count >= _MIN_EXPECTED_TICKS, (
            f"expected the event loop to stay responsive during the slow "
            f"delete (>= {_MIN_EXPECTED_TICKS} ticks), got {tick_count} — "
            "this indicates the blocking call is not offloaded via "
            "asyncio.to_thread"
        )

    async def test_blocking_baseline_would_starve_the_ticker(self):
        """Control case: calling the sync core directly on the loop starves the ticker.

        Proves the ticker methodology itself is sound — i.e. that a *real*
        blocking call in this test harness does suppress ticks — so the
        healthy-offload assertion above is meaningful and not a tautology.
        """
        from archivist.lifecycle.cascade import _qdrant_delete_sync

        client = MagicMock()
        client.count.return_value = MagicMock(count=0)
        client.delete.side_effect = lambda **kwargs: time.sleep(_SLOW_CALL_SECONDS)

        failed: list[str] = []

        async def _blocking_directly_on_loop():
            # Deliberately NOT wrapped in asyncio.to_thread — this is the
            # bug this spec fixed, reproduced here as a control.
            return _qdrant_delete_sync(client, "col", ["p1"], "qdrant_primary", "mem-1", failed)

        _, tick_count = await _run_with_ticker(_blocking_directly_on_loop())

        assert tick_count <= 1, (
            "control case should starve the ticker (no cooperative yield "
            f"points during the blocking sleep), got {tick_count} ticks"
        )


class TestSweepOrphansOffload:
    """ac-2 (M1): sweep_orphans() offloads get_collections/retrieve via asyncio.to_thread."""

    @staticmethod
    def _patch_empty_pool(monkeypatch):
        """Patch archivist.storage.sqlite_pool.pool.read to yield no rows.

        Keeps this test I/O-free: sweep_orphans's keyset-pagination loops
        exit immediately (empty result set), so the only meaningful work is
        the offloaded Qdrant calls under test.
        """
        from archivist.storage import sqlite_pool as sp

        class _EmptyCursor:
            async def fetchall(self):
                return []

        class _EmptyConn:
            async def execute(self, *args, **kwargs):
                return _EmptyCursor()

        @asynccontextmanager
        async def _empty_read():
            yield _EmptyConn()

        monkeypatch.setattr(sp.pool, "read", _empty_read)

    async def test_concurrent_task_progresses_during_slow_get_collections(self, monkeypatch):
        """A slow client.get_collections() must not block a concurrent coroutine."""
        self._patch_empty_pool(monkeypatch)

        client = MagicMock()

        def _slow_get_collections():
            time.sleep(_SLOW_CALL_SECONDS)
            return MagicMock()

        client.get_collections.side_effect = _slow_get_collections

        with (
            patch("archivist.lifecycle.cascade.qdrant_client", return_value=client),
            patch("archivist.lifecycle.cascade.collections_for_query", return_value=[]),
        ):
            from archivist.lifecycle.cascade import sweep_orphans

            result, tick_count = await _run_with_ticker(sweep_orphans())

        assert result == {"fts_cleaned": 0, "needle_cleaned": 0}
        assert tick_count >= _MIN_EXPECTED_TICKS, (
            f"expected the event loop to stay responsive during the slow "
            f"get_collections call (>= {_MIN_EXPECTED_TICKS} ticks), got "
            f"{tick_count} — this indicates the blocking call is not "
            "offloaded via asyncio.to_thread"
        )

    async def test_concurrent_task_progresses_during_slow_retrieve(self, monkeypatch):
        """A slow client.retrieve() (Phase 1 pagination loop) must not block the loop."""
        from archivist.storage import sqlite_pool as sp

        # One page of a single orphan-candidate qdrant_id, then an empty page
        # to end pagination. needle_registry stays empty so Phase 2 is a
        # no-op and only Phase 1's retrieve() call is exercised.
        pages = [[("candidate-id",)], []]

        class _Cursor:
            def __init__(self, rows):
                self._rows = rows

            async def fetchall(self):
                return self._rows

        class _Conn:
            async def execute(self, *args, **kwargs):
                return _Cursor(pages.pop(0) if pages else [])

        @asynccontextmanager
        async def _paged_read():
            yield _Conn()

        monkeypatch.setattr(sp.pool, "read", _paged_read)

        client = MagicMock()
        client.get_collections.return_value = MagicMock()

        def _slow_retrieve(**kwargs):
            time.sleep(_SLOW_CALL_SECONDS)
            return []

        client.retrieve.side_effect = _slow_retrieve

        with (
            patch("archivist.lifecycle.cascade.qdrant_client", return_value=client),
            patch("archivist.lifecycle.cascade.collections_for_query", return_value=["test_col"]),
            patch("archivist.lifecycle.cascade.delete_fts_chunks_batch", return_value=0),
            patch("archivist.lifecycle.cascade.delete_needle_tokens_batch", return_value=0),
        ):
            from archivist.lifecycle.cascade import sweep_orphans

            _, tick_count = await _run_with_ticker(sweep_orphans())

        assert tick_count >= _MIN_EXPECTED_TICKS, (
            f"expected the event loop to stay responsive during the slow "
            f"retrieve call (>= {_MIN_EXPECTED_TICKS} ticks), got "
            f"{tick_count} — this indicates client.retrieve is not "
            "offloaded via asyncio.to_thread"
        )
