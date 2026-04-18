"""Transactional outbox for Archivist cross-store writes (Phase 3).

Architecture
------------
Every Qdrant mutation that must be atomic with a SQLite write is first
serialised as a JSON *outbox event* and committed to the ``outbox`` table
inside the same ``pool.write()`` transaction that persists the SQLite
artefacts.  A lightweight background task (``OutboxProcessor``) then drains
the table, applies each event to Qdrant idempotently, and marks it
``applied``.

This converts a two-phase write with an unrecoverable crash window into:

    1. **Single SQLite commit** — SQLite artefacts + outbox rows land
       atomically.  If this commit fails, nothing is written anywhere.
    2. **Async background apply** — Qdrant is updated independently, with
       exponential-backoff retry.  If the process crashes between step 1 and
       step 2, the outbox row survives and the next drain picks it up.

Event flow
----------
``pending`` → (processor claims) → ``processing``
             → (Qdrant call succeeds) → ``applied``
             → (Qdrant call fails, retries < max) → ``pending`` (backoff)
             → (retries == max) → ``dead`` + ``delete_failures`` audit row

Crash safety (Phase 3 enterprise hardening)
-------------------------------------------
A periodic ``_sweep_orphaned_processing()`` call resets any rows that have
been stuck in ``processing`` longer than ``OUTBOX_ORPHAN_TIMEOUT_SECONDS``
(default 60 s) back to ``pending``.  This recovers from process crashes that
occur between the batch-claim UPDATE and the ``_mark_applied`` / ``_mark_dead``
call.

Usage (for ``MemoryTransaction`` — not called by application code directly)::

    processor = OutboxProcessor(vector_backend)
    await processor.drain()        # called from drain_loop()
    await processor.drain_loop()   # runs forever, called from app/main.py
"""

from __future__ import annotations

import asyncio
import enum
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import archivist.core.metrics as m

logger = logging.getLogger("archivist.outbox")


# ---------------------------------------------------------------------------
# Event types
# ---------------------------------------------------------------------------


class EventType(str, enum.Enum):
    """Qdrant operation types stored in the outbox.

    Each variant maps to a specific ``VectorBackend`` method.  The string
    value is persisted to the database — **do not rename existing values**.
    """

    QDRANT_UPSERT = "qdrant_upsert"
    """Upsert one or more ``PointStruct`` objects into a collection."""

    QDRANT_DELETE = "qdrant_delete"
    """Delete points by a list of IDs from a collection."""

    QDRANT_DELETE_FILTER = "qdrant_delete_filter"
    """Delete all points matching a Qdrant ``Filter`` expression."""

    QDRANT_SET_PAYLOAD = "qdrant_set_payload"
    """Set payload fields on a list of points."""


# ---------------------------------------------------------------------------
# Event dataclass
# ---------------------------------------------------------------------------


@dataclass
class OutboxEvent:
    """An outbox event ready to be persisted to the ``outbox`` table.

    Attributes:
        event_type: The operation to apply to the vector backend.
        payload: JSON-serialisable dict; schema depends on *event_type*.
            ``QDRANT_UPSERT``   → ``{"collection": str, "points": [PointStruct.dict(), ...]}``
            ``QDRANT_DELETE``   → ``{"collection": str, "ids": [str, ...]}``
            ``QDRANT_DELETE_FILTER`` → ``{"collection": str, "filter": Filter.dict()}``
            ``QDRANT_SET_PAYLOAD``   → ``{"collection": str, "payload": dict, "ids": [str, ...]}``
        id: UUID v4, auto-generated.
        created_at: ISO 8601 UTC timestamp, auto-generated.
    """

    event_type: EventType
    payload: dict[str, Any]
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def payload_json(self) -> str:
        """Return *payload* as a compact JSON string for DB storage."""
        return json.dumps(self.payload, separators=(",", ":"))


# ---------------------------------------------------------------------------
# Outbox processor
# ---------------------------------------------------------------------------


class OutboxProcessor:
    """Background task that drains the ``outbox`` table and applies events.

    The processor polls ``outbox`` for ``pending`` rows on every
    ``OUTBOX_DRAIN_INTERVAL``-second tick.  Each batch is limited to
    ``OUTBOX_BATCH_SIZE`` rows to bound individual drain latency.  Events are
    marked ``processing`` before the Qdrant call so that concurrent drain
    calls never double-apply an event.

    Crash safety
    ------------
    ``_sweep_orphaned_processing()`` periodically resets rows that have been
    stuck in ``processing`` longer than ``OUTBOX_ORPHAN_TIMEOUT_SECONDS`` back
    to ``pending``.  This handles the crash window between the batch-claim
    UPDATE and ``_mark_applied`` / ``_mark_dead``.  The sweep runs on startup
    (drain cycle 0) and every ``OUTBOX_ORPHAN_SWEEP_EVERY_N`` cycles thereafter.

    Retention
    ---------
    ``_prune_applied()`` deletes ``applied`` rows older than
    ``OUTBOX_RETENTION_DAYS`` days to keep the table small.  It runs on the
    same cadence as the orphan sweep.

    Idempotency
    -----------
    * ``QDRANT_UPSERT`` — Qdrant upserts are idempotent by point ID.
    * ``QDRANT_DELETE`` — Delete of an already-deleted point returns success.
    * ``QDRANT_DELETE_FILTER`` — Safe to replay; already-deleted points are
      no-ops.
    * ``QDRANT_SET_PAYLOAD`` — Idempotent (last write wins).

    Args:
        vector_backend: A ``VectorBackend`` implementation.  Typically a
            ``QdrantVectorBackend`` instance.
    """

    def __init__(self, vector_backend: Any) -> None:
        self._backend = vector_backend
        self._drain_cycle: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def drain(self) -> int:
        """Fetch and process up to ``OUTBOX_BATCH_SIZE`` pending events.

        On every ``OUTBOX_ORPHAN_SWEEP_EVERY_N``-th call (and on the first
        call), runs ``_sweep_orphaned_processing()`` and ``_prune_applied()``
        before draining.

        Returns:
            Number of events successfully applied in this drain cycle.
        """
        from archivist.core.config import (
            OUTBOX_BATCH_SIZE,
            OUTBOX_MAX_RETRIES,
            OUTBOX_ORPHAN_SWEEP_EVERY_N,
        )
        from archivist.storage.sqlite_pool import pool

        t0 = time.monotonic()

        if self._drain_cycle % OUTBOX_ORPHAN_SWEEP_EVERY_N == 0:
            recovered = await self._sweep_orphaned_processing()
            pruned = await self._prune_applied()
            if recovered:
                logger.warning(
                    "outbox.sweep: recovered %d orphaned 'processing' events",
                    recovered,
                    extra={"recovered": recovered, "drain_cycle": self._drain_cycle},
                )
            if pruned:
                logger.info(
                    "outbox.prune: deleted %d expired 'applied' rows",
                    pruned,
                    extra={"pruned": pruned, "drain_cycle": self._drain_cycle},
                )

        self._drain_cycle += 1

        now_iso = datetime.now(UTC).isoformat()
        # Back-off: only retry events whose last_attempt is far enough in the
        # past.  We compute the threshold inside the SQL query via a fixed
        # drift; the OutboxProcessor handles precise per-event back-off by
        # comparing last_attempt locally.
        async with pool.write() as conn:
            cursor = await conn.execute(
                """
                SELECT id, event_type, payload, retry_count, last_attempt
                FROM outbox
                WHERE status = 'pending'
                  AND (last_attempt IS NULL OR last_attempt < ?)
                ORDER BY created_at ASC
                LIMIT ?
                """,
                (now_iso, OUTBOX_BATCH_SIZE),
            )
            rows: list[Any] = list(await cursor.fetchall())

            if not rows:
                elapsed_ms = (time.monotonic() - t0) * 1000
                m.observe(m.OUTBOX_DRAIN_DURATION, elapsed_ms)
                return 0

            # Claim the batch atomically — set status='processing'.
            ids = [r[0] for r in rows]
            placeholders = ",".join("?" * len(ids))
            await conn.execute(
                f"UPDATE outbox SET status='processing', last_attempt=? WHERE id IN ({placeholders})",
                [now_iso, *ids],
            )

        applied = 0
        dead_count = 0
        for row in rows:
            event_id = row[0]
            retry_count = row[3] or 0
            try:
                await self._apply_event(
                    event_id=event_id,
                    event_type=EventType(row[1]),
                    payload=json.loads(row[2]),
                )
                await self._mark_applied(event_id)
                applied += 1
                m.inc(m.OUTBOX_APPLIED_TOTAL)
            except Exception as exc:
                new_retry = retry_count + 1
                logger.warning(
                    "outbox.drain: event failed",
                    extra={
                        "event_id": event_id,
                        "event_type": row[1],
                        "attempt": new_retry,
                        "error": str(exc),
                    },
                )
                if new_retry >= OUTBOX_MAX_RETRIES:
                    await self._mark_dead(
                        event_id,
                        payload=json.loads(row[2]),
                        error=str(exc),
                    )
                    dead_count += 1
                    m.inc(m.OUTBOX_DEAD_LETTER_TOTAL)
                else:
                    backoff = min(2**new_retry * 0.5, 60.0)
                    next_attempt = datetime.fromtimestamp(
                        datetime.now(UTC).timestamp() + backoff, tz=UTC
                    ).isoformat()
                    await self._mark_failed(event_id, str(exc), new_retry, next_attempt)

        elapsed_ms = (time.monotonic() - t0) * 1000
        m.observe(m.OUTBOX_DRAIN_DURATION, elapsed_ms)

        if applied or dead_count:
            logger.info(
                "outbox.drain_complete",
                extra={
                    "applied": applied,
                    "dead": dead_count,
                    "batch_size": len(rows),
                    "duration_ms": round(elapsed_ms, 2),
                    "drain_cycle": self._drain_cycle,
                },
            )

        return applied

    async def drain_loop(self) -> None:
        """Run ``drain()`` forever on a ``OUTBOX_DRAIN_INTERVAL``-second tick.

        Designed to be started as an ``asyncio.create_task`` during app
        startup.  Logs but does not re-raise exceptions so the loop never
        silently dies.
        """
        from archivist.core.config import OUTBOX_DRAIN_INTERVAL

        logger.info(
            "OutboxProcessor drain loop started",
            extra={"interval_seconds": OUTBOX_DRAIN_INTERVAL},
        )
        while True:
            try:
                await self.drain()
            except Exception as exc:
                logger.error(
                    "outbox.drain_loop unhandled error",
                    extra={"error": str(exc)},
                    exc_info=True,
                )
            await asyncio.sleep(OUTBOX_DRAIN_INTERVAL)

    async def _sweep_orphaned_processing(self) -> int:
        """Reset events stuck in 'processing' past the orphan timeout to 'pending'.

        A process crash between the batch-claim UPDATE (``status='processing'``)
        and ``_mark_applied`` / ``_mark_dead`` leaves rows permanently orphaned
        because ``drain()`` only queries ``WHERE status='pending'``.  This
        method recovers those rows by resetting them after
        ``OUTBOX_ORPHAN_TIMEOUT_SECONDS`` have elapsed since ``last_attempt``.

        Each recovery increments ``retry_count`` so that events that repeatedly
        become stuck (e.g. persistent Qdrant failures after requeue) still
        advance toward ``OUTBOX_MAX_RETRIES`` and can be dead-lettered.

        Returns:
            Number of orphaned events reset to 'pending'.
        """
        from archivist.core.config import OUTBOX_ORPHAN_TIMEOUT_SECONDS
        from archivist.storage.sqlite_pool import pool

        cutoff = (
            datetime.now(UTC) - timedelta(seconds=OUTBOX_ORPHAN_TIMEOUT_SECONDS)
        ).isoformat()

        async with pool.write() as conn:
            cursor = await conn.execute(
                """
                SELECT id FROM outbox
                WHERE status = 'processing'
                  AND last_attempt < ?
                """,
                (cutoff,),
            )
            stuck: list[Any] = list(await cursor.fetchall())
            if not stuck:
                return 0

            ids = [r[0] for r in stuck]
            placeholders = ",".join("?" * len(ids))
            now_iso = datetime.now(UTC).isoformat()
            await conn.execute(
                f"""
                UPDATE outbox
                SET status = 'pending',
                    retry_count = retry_count + 1,
                    last_attempt = ?,
                    error = 'recovered by orphan sweep'
                WHERE id IN ({placeholders})
                """,
                [now_iso, *ids],
            )

        recovered = len(stuck)
        m.inc(m.OUTBOX_RECOVERY_COUNT, value=recovered)
        for row in stuck:
            logger.warning(
                "outbox.sweep: recovered orphaned event",
                extra={"event_id": row[0], "orphan_timeout_seconds": OUTBOX_ORPHAN_TIMEOUT_SECONDS},
            )
        return recovered

    async def _prune_applied(self) -> int:
        """Delete 'applied' outbox rows older than ``OUTBOX_RETENTION_DAYS``.

        Pruning is capped at 1 000 rows per call to bound write-lock hold time.
        If more rows are eligible they will be cleaned up on subsequent sweeps.

        Returns:
            Number of rows deleted.
        """
        from archivist.core.config import OUTBOX_RETENTION_DAYS
        from archivist.storage.sqlite_pool import pool

        cutoff = (datetime.now(UTC) - timedelta(days=OUTBOX_RETENTION_DAYS)).isoformat()

        async with pool.write() as conn:
            # Use a sub-select with LIMIT so we never hold the write lock for
            # an unbounded DELETE on a large table.
            cursor = await conn.execute(
                """
                DELETE FROM outbox
                WHERE id IN (
                    SELECT id FROM outbox
                    WHERE status = 'applied'
                      AND last_attempt < ?
                    LIMIT 1000
                )
                """,
                (cutoff,),
            )
            pruned = cursor.rowcount or 0

        if pruned:
            m.inc(m.OUTBOX_PRUNED_TOTAL, value=pruned)
        return pruned

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _apply_event(
        self,
        event_id: str,
        event_type: EventType,
        payload: dict[str, Any],
    ) -> None:
        """Route *event_type* to the appropriate ``VectorBackend`` call.

        Args:
            event_id: Outbox row UUID (for logging only).
            event_type: The operation to perform.
            payload: Decoded JSON payload from the outbox row.

        Raises:
            ValueError: If *event_type* is unrecognised.
            Any exception raised by the vector backend is propagated.
        """
        collection = payload["collection"]

        if event_type == EventType.QDRANT_UPSERT:
            from qdrant_client.models import PointStruct

            raw_points: list[dict[str, Any]] = payload["points"]
            points = [
                PointStruct(
                    id=p["id"],
                    vector=p["vector"],
                    payload=p.get("payload"),
                )
                for p in raw_points
            ]
            await self._backend.upsert(collection, points)

        elif event_type == EventType.QDRANT_DELETE:
            ids: list[str] = payload["ids"]
            await self._backend.delete(collection, ids)

        elif event_type == EventType.QDRANT_DELETE_FILTER:
            from qdrant_client.models import Filter

            filt = Filter(**payload["filter"])
            await self._backend.delete_by_filter(collection, filt)

        elif event_type == EventType.QDRANT_SET_PAYLOAD:
            await self._backend.set_payload(
                collection,
                payload["payload"],
                payload["ids"],
            )

        else:
            raise ValueError(f"Unknown EventType: {event_type!r}")

        logger.debug("outbox._apply_event: %s %s applied", event_type, event_id)

    async def _mark_applied(self, event_id: str) -> None:
        """Set status='applied' for *event_id*."""
        from archivist.storage.sqlite_pool import pool

        async with pool.write() as conn:
            await conn.execute(
                "UPDATE outbox SET status='applied', last_attempt=? WHERE id=?",
                (datetime.now(UTC).isoformat(), event_id),
            )

    async def _mark_failed(
        self,
        event_id: str,
        error: str,
        retry_count: int,
        next_attempt: str,
    ) -> None:
        """Reset status to 'pending' with updated retry_count and last_attempt.

        The next drain will skip this event until ``last_attempt`` is in the
        past (exponential back-off).
        """
        from archivist.storage.sqlite_pool import pool

        async with pool.write() as conn:
            await conn.execute(
                """
                UPDATE outbox
                SET status='pending',
                    retry_count=?,
                    last_attempt=?,
                    error=?
                WHERE id=?
                """,
                (retry_count, next_attempt, error[:2000], event_id),
            )

    async def _mark_dead(
        self,
        event_id: str,
        payload: dict[str, Any],
        error: str,
    ) -> None:
        """Set status='dead' and write an audit row to ``delete_failures``.

        The ``delete_failures`` table is the permanent audit log introduced in
        Phase 2.  Writing here preserves the existing failure-inspection UX.

        Args:
            event_id: Outbox row UUID.
            payload: Decoded payload (used to extract qdrant_ids for audit row).
            error: Last error message.
        """
        from archivist.storage.sqlite_pool import pool

        now_iso = datetime.now(UTC).isoformat()

        # Extract qdrant_ids for the audit row (best-effort).
        qdrant_ids: list[str] = payload.get("ids", [])
        memory_id: str = payload.get("memory_id", event_id)

        async with pool.write() as conn:
            await conn.execute(
                "UPDATE outbox SET status='dead', error=?, last_attempt=? WHERE id=?",
                (error[:2000], now_iso, event_id),
            )
            if qdrant_ids:
                df_id = str(uuid.uuid4())
                await conn.execute(
                    """
                    INSERT OR IGNORE INTO delete_failures
                        (id, memory_id, qdrant_ids, error, created_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        df_id,
                        memory_id,
                        json.dumps(qdrant_ids),
                        f"outbox dead-letter after max retries: {error[:500]}",
                        now_iso,
                    ),
                )
        logger.error(
            "outbox._mark_dead: event %s moved to dead-letter (memory_id=%s, ids=%s)",
            event_id,
            memory_id,
            qdrant_ids[:5],
        )


# ---------------------------------------------------------------------------
# Module-level processor singleton (lazy — set by main.py after startup)
# ---------------------------------------------------------------------------

_processor: OutboxProcessor | None = None


def get_processor() -> OutboxProcessor | None:
    """Return the module-level ``OutboxProcessor`` singleton, or ``None``."""
    return _processor


def set_processor(processor: OutboxProcessor) -> None:
    """Register the module-level singleton.  Called once from ``app/main.py``."""
    global _processor
    _processor = processor
