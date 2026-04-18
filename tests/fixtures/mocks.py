"""Reusable mock builders for the Archivist test suite.

All external-service mocks (Qdrant, vector backend, LLM, embeddings) are
built here so tests remain decoupled from implementation details of the mocked
interfaces.

Usage::

    from tests.fixtures.mocks import make_vector_backend_mock, make_qdrant_client_mock
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock


def make_vector_backend_mock() -> MagicMock:
    """Return a ``VectorBackend``-protocol-compatible mock.

    All async methods return safe defaults (None / empty list).  Tests can
    override individual attributes as needed.
    """
    backend = MagicMock()
    backend.upsert = AsyncMock(return_value=None)
    backend.delete = AsyncMock(return_value=None)
    backend.delete_by_filter = AsyncMock(return_value=None)
    backend.set_payload = AsyncMock(return_value=None)
    backend.retrieve = AsyncMock(return_value=[])
    backend.ensure_collection = AsyncMock(return_value=None)
    return backend


def make_qdrant_client_mock(
    *,
    search_results: list | None = None,
    scroll_pages: list[tuple] | None = None,
) -> MagicMock:
    """Return a Qdrant client mock configured for common test scenarios.

    Args:
        search_results: List of mock scored points returned by ``.search()``.
            Defaults to an empty list.
        scroll_pages: List of ``(points, next_offset)`` tuples for paginated
            ``.scroll()`` calls.  Defaults to a single empty page.
    """
    client = MagicMock()
    client.delete = MagicMock(return_value=MagicMock(operation_id=1))
    client.set_payload = MagicMock(return_value=True)
    client.count = MagicMock(return_value=MagicMock(count=0))
    client.search = AsyncMock(return_value=search_results or [])
    client.upsert = AsyncMock(return_value=None)

    if scroll_pages:
        _pages = list(scroll_pages)
        _idx = [0]

        def _scroll(**kwargs):
            i = _idx[0]
            result = _pages[i] if i < len(_pages) else ([], None)
            _idx[0] += 1
            return result

        client.scroll = MagicMock(side_effect=_scroll)
    else:
        client.scroll = MagicMock(return_value=([], None))

    return client


def make_llm_mock(response: str = "Mocked LLM response.") -> AsyncMock:
    """Return an ``AsyncMock`` for ``llm.llm_query`` returning *response*."""
    mock = AsyncMock(return_value=response)
    return mock


def make_embed_mock(dim: int = 1024) -> AsyncMock:
    """Return an ``AsyncMock`` for embedding functions returning a *dim*-float vector."""
    return AsyncMock(return_value=[0.1] * dim)


async def count_table(pool, table: str) -> int:
    """Count all rows in *table* using *pool*.

    Helper shared by integration and system test modules.
    """
    async with pool.read() as conn:
        cur = await conn.execute(f"SELECT COUNT(*) FROM {table}")
        row = await cur.fetchone()
        return row[0]


async def count_outbox(pool, status: str | None = None) -> int:
    """Count outbox rows, optionally filtered by *status*."""
    async with pool.read() as conn:
        if status:
            cur = await conn.execute("SELECT COUNT(*) FROM outbox WHERE status=?", (status,))
        else:
            cur = await conn.execute("SELECT COUNT(*) FROM outbox")
        row = await cur.fetchone()
        return row[0]


async def reset_outbox_backoff(pool) -> None:
    """Force all pending outbox events to be immediately retryable."""
    async with pool.write() as conn:
        await conn.execute(
            "UPDATE outbox SET last_attempt='2000-01-01T00:00:00+00:00' WHERE status='pending'"
        )
