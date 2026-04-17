"""MemoryTransaction — atomic SQLite artefacts + Qdrant outbox enqueue (Phase 3).

``MemoryTransaction`` is an async context manager that wraps a single
``pool.write()`` transaction.  Inside the transaction body callers:

* Execute SQLite writes directly on the yielded ``aiosqlite.Connection`` (or
  via the helper methods delegating to existing graph functions).
* Enqueue Qdrant operations via ``txn.enqueue_*`` methods; these accumulate
  in-memory and are flushed into the ``outbox`` table atomically with all
  other SQLite writes at the moment ``__aexit__`` commits the transaction.

If any exception propagates out of the ``async with`` block, ``pool.write()``
rolls back *everything* — neither the SQLite artefacts nor the outbox rows are
written.  This is the core guarantee: no partial cross-store state.

Usage::

    from archivist.storage.transaction import MemoryTransaction

    async with MemoryTransaction() as txn:
        await txn.conn.execute("INSERT INTO memory_chunks ...", (...,))
        txn.enqueue_qdrant_upsert(
            collection="archivist_memories",
            points=[point_struct],
            memory_id="abc-123",
        )
    # On clean exit: SQLite artefacts + outbox rows committed together.
    # The OutboxProcessor will pick up the outbox row and upsert to Qdrant.

When ``OUTBOX_ENABLED=False`` (default for rollout safety) the context manager
falls through: ``enqueue_*`` calls are no-ops and callers are expected to
perform the Qdrant operations inline as they did before Phase 3.

Internal design
---------------
The ``MemoryTransaction`` class does NOT nest ``pool.write()`` — it enters
the write-lock once, exposes the live connection as ``txn.conn``, accumulates
events in ``txn._events``, and on clean ``__aexit__`` inserts all outbox rows
before the implicit commit.

This avoids a subtle deadlock: if a nested call inside the transaction tried
to acquire ``pool.write()`` again it would block forever (asyncio.Lock is not
re-entrant).  Callers must pass ``txn.conn`` through to any helper that
normally calls ``pool.write()`` internally, or use the ``txn.execute`` /
``txn.executemany`` shims provided here.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Any

import aiosqlite

from archivist.storage.outbox import EventType, OutboxEvent

logger = logging.getLogger("archivist.transaction")


class MemoryTransaction:
    """Async context manager for atomic SQLite + outbox writes.

    Attributes:
        conn: The live ``aiosqlite.Connection`` while inside the ``async with``
            block.  Callers can execute arbitrary SQL on this connection; all
            writes participate in the single enclosing transaction.

    Args:
        enabled: If ``False``, the context manager still yields but
            ``enqueue_*`` methods become no-ops.  Defaults to the value of
            ``OUTBOX_ENABLED`` from config.

    Raises:
        RuntimeError: If the pool has not been initialised before entering.
    """

    def __init__(self, enabled: bool | None = None) -> None:
        if enabled is None:
            from archivist.core.config import OUTBOX_ENABLED

            enabled = OUTBOX_ENABLED
        self._enabled = enabled
        self._events: list[OutboxEvent] = []
        self.conn: aiosqlite.Connection | None = None
        self._write_ctx = None  # the pool.write() async context manager

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> MemoryTransaction:
        from archivist.storage.sqlite_pool import pool

        self._write_ctx = pool.write()
        self.conn = await self._write_ctx.__aenter__()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> bool | None:
        if exc_type is None and self._enabled and self._events:
            # Flush all accumulated outbox rows atomically with the transaction.
            try:
                await self._flush_events()
            except Exception as flush_err:
                # If flushing fails, roll back everything via __aexit__(exc).
                logger.error("MemoryTransaction._flush_events failed: %s", flush_err)
                await self._write_ctx.__aexit__(type(flush_err), flush_err, None)
                raise

        # Delegate commit (or rollback on exception) to pool.write().
        return await self._write_ctx.__aexit__(exc_type, exc_val, exc_tb)

    # ------------------------------------------------------------------
    # SQL shims (execute on the held connection without re-acquiring lock)
    # ------------------------------------------------------------------

    async def execute(
        self,
        sql: str,
        params: tuple[Any, ...] | list[Any] = (),
    ) -> aiosqlite.Cursor:
        """Execute *sql* on the transaction connection.

        Args:
            sql: SQL statement to execute.
            params: Positional parameters.

        Returns:
            The ``aiosqlite.Cursor`` returned by ``execute``.

        Raises:
            RuntimeError: If called outside of ``async with`` block.
        """
        if self.conn is None:
            raise RuntimeError("MemoryTransaction.execute called outside async with block")
        return await self.conn.execute(sql, params)

    async def executemany(
        self,
        sql: str,
        params: list[tuple[Any, ...]],
    ) -> None:
        """Execute *sql* for each row in *params* on the transaction connection.

        Args:
            sql: SQL statement to execute.
            params: Sequence of parameter tuples.

        Raises:
            RuntimeError: If called outside of ``async with`` block.
        """
        if self.conn is None:
            raise RuntimeError("MemoryTransaction.executemany called outside async with block")
        await self.conn.executemany(sql, params)

    async def fetchall(
        self,
        sql: str,
        params: tuple[Any, ...] | list[Any] = (),
    ) -> list[Any]:
        """Execute *sql* and return all rows.

        Args:
            sql: SQL SELECT statement.
            params: Positional parameters.

        Returns:
            List of row objects (``aiosqlite.Row``).
        """
        if self.conn is None:
            raise RuntimeError("MemoryTransaction.fetchall called outside async with block")
        cursor = await self.conn.execute(sql, params)
        return await cursor.fetchall()

    # ------------------------------------------------------------------
    # Graph-helper shims (run on the held connection, no extra lock)
    # ------------------------------------------------------------------

    async def upsert_fts_chunk(
        self,
        qdrant_id: str,
        text: str,
        file_path: str,
        chunk_index: int,
        agent_id: str = "",
        namespace: str = "",
        date: str = "",
        memory_type: str = "general",
        actor_id: str = "",
        actor_type: str = "",
    ) -> None:
        """Upsert an FTS5 chunk using the transaction's open connection.

        Delegates to ``graph.upsert_fts_chunk`` with ``conn=self.conn`` so the
        FTS5 write joins this transaction atomically.

        Raises:
            RuntimeError: If called outside of ``async with`` block.
        """
        if self.conn is None:
            raise RuntimeError("MemoryTransaction.upsert_fts_chunk called outside async with block")
        from archivist.storage.graph import upsert_fts_chunk as _upsert_fts

        await _upsert_fts(
            qdrant_id=qdrant_id,
            text=text,
            file_path=file_path,
            chunk_index=chunk_index,
            agent_id=agent_id,
            namespace=namespace,
            date=date,
            memory_type=memory_type,
            actor_id=actor_id,
            actor_type=actor_type,
            conn=self.conn,
        )

    async def register_needle_tokens(
        self,
        memory_id: str,
        text: str,
        namespace: str = "",
        agent_id: str = "",
        actor_id: str = "",
        actor_type: str = "",
    ) -> None:
        """Register needle tokens using the transaction's open connection.

        Delegates to ``graph.register_needle_tokens`` with ``conn=self.conn``
        so needle inserts join this transaction atomically.

        Raises:
            RuntimeError: If called outside of ``async with`` block.
        """
        if self.conn is None:
            raise RuntimeError(
                "MemoryTransaction.register_needle_tokens called outside async with block"
            )
        from archivist.storage.graph import register_needle_tokens as _reg_needle

        await _reg_needle(
            memory_id=memory_id,
            text=text,
            namespace=namespace,
            agent_id=agent_id,
            actor_id=actor_id,
            actor_type=actor_type,
            conn=self.conn,
        )

    async def upsert_entity(self, *args: Any, **kwargs: Any) -> int:
        """Upsert an entity using the transaction's open connection.

        All positional and keyword arguments are forwarded to
        ``graph.upsert_entity``; ``conn=self.conn`` is injected automatically.

        Raises:
            RuntimeError: If called outside of ``async with`` block.
        """
        if self.conn is None:
            raise RuntimeError("MemoryTransaction.upsert_entity called outside async with block")
        from archivist.storage.graph import upsert_entity as _upsert_entity

        return await _upsert_entity(*args, **kwargs, conn=self.conn)

    async def add_fact(self, *args: Any, **kwargs: Any) -> int:
        """Insert a fact using the transaction's open connection.

        All positional and keyword arguments are forwarded to
        ``graph.add_fact``; ``conn=self.conn`` is injected automatically.

        Raises:
            RuntimeError: If called outside of ``async with`` block.
        """
        if self.conn is None:
            raise RuntimeError("MemoryTransaction.add_fact called outside async with block")
        from archivist.storage.graph import add_fact as _add_fact

        return await _add_fact(*args, **kwargs, conn=self.conn)

    # ------------------------------------------------------------------
    # Outbox event enqueue helpers
    # ------------------------------------------------------------------

    def enqueue_qdrant_upsert(
        self,
        collection: str,
        points: list[Any],
        memory_id: str = "",
    ) -> None:
        """Enqueue a ``QDRANT_UPSERT`` event for the given *points*.

        ``PointStruct`` objects are serialised to dicts via their ``.dict()``
        method (Pydantic v1) or ``model_dump()`` (Pydantic v2).  The outbox
        processor will reconstruct them on the other side.

        Args:
            collection: Target Qdrant collection name.
            points: List of ``PointStruct`` objects to upsert.
            memory_id: Optional memory ID for audit/logging correlation.
        """
        if not self._enabled:
            return
        serialised = []
        for p in points:
            if hasattr(p, "model_dump"):
                serialised.append(p.model_dump())
            elif hasattr(p, "dict"):
                serialised.append(p.dict())
            else:
                serialised.append(p)
        self._events.append(
            OutboxEvent(
                event_type=EventType.QDRANT_UPSERT,
                payload={
                    "collection": collection,
                    "points": serialised,
                    "memory_id": memory_id,
                },
            )
        )

    def enqueue_qdrant_delete(
        self,
        collection: str,
        ids: list[str],
        memory_id: str = "",
    ) -> None:
        """Enqueue a ``QDRANT_DELETE`` event for the given point *ids*.

        Args:
            collection: Target Qdrant collection name.
            ids: List of point UUIDs to delete.
            memory_id: Optional memory ID for audit/logging correlation.
        """
        if not self._enabled:
            return
        self._events.append(
            OutboxEvent(
                event_type=EventType.QDRANT_DELETE,
                payload={"collection": collection, "ids": ids, "memory_id": memory_id},
            )
        )

    def enqueue_qdrant_delete_filter(
        self,
        collection: str,
        filt_dict: dict[str, Any],
        memory_id: str = "",
    ) -> None:
        """Enqueue a ``QDRANT_DELETE_FILTER`` event.

        Args:
            collection: Target Qdrant collection name.
            filt_dict: Filter expression as a plain dict (``Filter.dict()``).
            memory_id: Optional memory ID for audit/logging correlation.
        """
        if not self._enabled:
            return
        self._events.append(
            OutboxEvent(
                event_type=EventType.QDRANT_DELETE_FILTER,
                payload={
                    "collection": collection,
                    "filter": filt_dict,
                    "memory_id": memory_id,
                },
            )
        )

    def enqueue_qdrant_set_payload(
        self,
        collection: str,
        payload: dict[str, Any],
        ids: list[str],
        memory_id: str = "",
    ) -> None:
        """Enqueue a ``QDRANT_SET_PAYLOAD`` event.

        Args:
            collection: Target Qdrant collection name.
            payload: Payload fields to set on the matched points.
            ids: List of point IDs to update.
            memory_id: Optional memory ID for audit/logging correlation.
        """
        if not self._enabled:
            return
        self._events.append(
            OutboxEvent(
                event_type=EventType.QDRANT_SET_PAYLOAD,
                payload={
                    "collection": collection,
                    "payload": payload,
                    "ids": ids,
                    "memory_id": memory_id,
                },
            )
        )

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    async def _flush_events(self) -> None:
        """Insert all accumulated outbox events into the ``outbox`` table.

        This is called inside the open transaction (same connection, same
        commit) so the outbox rows and all SQLite artefacts land atomically.
        """
        if not self._events or self.conn is None:
            return
        rows = [
            (
                ev.id,
                ev.event_type.value,
                ev.payload_json(),
                "pending",
                0,
                None,
                ev.created_at,
                None,
            )
            for ev in self._events
        ]
        await self.conn.executemany(
            """
            INSERT INTO outbox (id, event_type, payload, status, retry_count,
                                last_attempt, created_at, error)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        logger.debug(
            "MemoryTransaction._flush_events: queued %d outbox event(s)",
            len(self._events),
        )
