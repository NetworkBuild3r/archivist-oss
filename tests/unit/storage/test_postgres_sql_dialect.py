"""Postgres-compatibility guards for the wake-up, hotness and curator refresh SQL.

These run without a Postgres server (CI has none), so they check the statements
Archivist would send *after* the asyncpg translation layer, plus the logging
contract of the curator refresh.  The statements are executed against a real
Postgres by the ``postgres`` variants in
``tests/integration/storage/test_dual_backend.py`` (set ``POSTGRES_TEST_DSN``).

What is covered here: SQL text shape (no DISTINCT + non-selected ORDER BY, no
boolean compared to an integer column, aggregate MAX kept intact by the translator)
and the once-per-cycle warning.  What is not: the server accepting the statements.
"""

from __future__ import annotations

import logging
import re
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.storage]


class _RecordingConn:
    """Connection stand-in that records SQL and returns empty results."""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    async def fetchall(self, sql, params=()):
        self._sink.append(sql)
        return []

    async def fetchone(self, sql, params=()):
        self._sink.append(sql)
        return None


class _RecordingPool:
    def __init__(self) -> None:
        self.sql: list[str] = []

    @asynccontextmanager
    async def read(self):
        yield _RecordingConn(self.sql)


@pytest.mark.parametrize("agent_id", ["wake-agent", ""])
async def test_wake_up_identity_statement_is_valid_on_postgres(monkeypatch, agent_id):
    """The identity query groups instead of using DISTINCT with a hidden ORDER BY key."""
    import archivist.storage.sqlite_pool as sqlite_pool
    from archivist.storage.asyncpg_backend import _translate_sql
    from archivist.storage.compressed_index import build_wake_up_context

    pool = _RecordingPool()
    monkeypatch.setattr(sqlite_pool, "pool", pool)

    await build_wake_up_context("ns", agent_id=agent_id)

    identity = [s for s in pool.sql if "e.retention_class IN ('permanent', 'durable')" in s]
    assert len(identity) == 1
    translated = _translate_sql(identity[0])

    # Postgres: "for SELECT DISTINCT, ORDER BY expressions must appear in select list".
    assert "DISTINCT" not in translated.upper()
    # Postgres: aggregate must survive translation (GREATEST(col) is not an aggregate).
    assert "MAX(e.mention_count) AS mc" in translated
    assert "GREATEST" not in translated
    assert re.search(
        r"GROUP BY e\.name, e\.entity_type, e\.retention_class ORDER BY mc DESC LIMIT 10$",
        translated,
    )


def test_hotness_query_postgres_compares_integer_and_text_columns_correctly():
    """cache_hit is INTEGER and created_at is TEXT in the Postgres schema."""
    from archivist.core.hotness import _retrieval_logs_query

    sql = _retrieval_logs_query(True)

    assert "cache_hit = 0" in sql  # "integer = boolean" otherwise
    assert not re.search(r"\b(TRUE|FALSE)\b", sql, re.IGNORECASE)
    # TEXT column vs timestamptz raises "operator does not exist"; compare as ISO text.
    assert "created_at > to_char(" in sql


def test_hotness_query_sqlite_unchanged():
    from archivist.core.hotness import _retrieval_logs_query

    sql = _retrieval_logs_query(False)

    assert "cache_hit = 0" in sql
    assert "created_at > datetime('now', '-7 days')" in sql


async def test_hotness_query_failure_is_logged_at_warning(monkeypatch, caplog):
    """A failing retrieval_logs query must not be hidden at debug level."""
    import archivist.core.hotness as hotness
    import archivist.storage.sqlite_pool as sqlite_pool

    class _BoomConn:
        async def fetchall(self, sql, params=()):
            if "retrieval_logs" in sql:
                raise RuntimeError("operator does not exist")
            return []

    class _BoomPool:
        @asynccontextmanager
        async def read(self):
            yield _BoomConn()

        @asynccontextmanager
        async def write(self):
            yield _BoomConn()

    monkeypatch.setattr(sqlite_pool, "pool", _BoomPool())
    with caplog.at_level(logging.DEBUG, logger=hotness.logger.name):
        await hotness.batch_update_hotness()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "retrieval_logs query failed" in warnings[0].getMessage()


class _PairsConn:
    def __init__(self, rows):
        self._rows = rows

    async def fetchall(self, sql, params=()):
        return self._rows


def _pairs_pool(rows):
    class _Pool:
        @asynccontextmanager
        async def read(self):
            yield _PairsConn(rows)

    return _Pool()


async def test_curator_wake_up_refresh_warns_once_per_cycle_with_failure_count(monkeypatch, caplog):
    import archivist.lifecycle.curator as curator
    import archivist.storage.sqlite_pool as sqlite_pool

    rows = [{"namespace": f"ns{i}", "agent_id": "a"} for i in range(5)]
    monkeypatch.setattr(sqlite_pool, "pool", _pairs_pool(rows))

    async def _cache(ns, agent_id=""):
        if ns in {"ns1", "ns2", "ns3"}:
            raise RuntimeError("boom for " + ns)

    with (
        patch.object(curator, "cache_wake_up", new=AsyncMock(side_effect=_cache)),
        caplog.at_level(logging.DEBUG, logger=curator.logger.name),
    ):
        refreshed = await curator._refresh_wake_up_caches()

    assert refreshed == 2
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert "failed for 3 of 5" in message
    assert "boom for ns1" in message  # first failure is the sample
    # No per-pair debug lines.
    assert not [r for r in caplog.records if "Wake-up cache failed for" in r.getMessage()]


async def test_curator_wake_up_refresh_silent_when_all_succeed(monkeypatch, caplog):
    import archivist.lifecycle.curator as curator
    import archivist.storage.sqlite_pool as sqlite_pool

    rows = [{"namespace": "ns0", "agent_id": "a"}, {"namespace": "ns0", "agent_id": "a"}]
    monkeypatch.setattr(sqlite_pool, "pool", _pairs_pool(rows))

    with (
        patch.object(curator, "cache_wake_up", new=AsyncMock(return_value={})),
        caplog.at_level(logging.WARNING, logger=curator.logger.name),
    ):
        refreshed = await curator._refresh_wake_up_caches()

    assert refreshed == 1  # duplicate pair is refreshed once
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
