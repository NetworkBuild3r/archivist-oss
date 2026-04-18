"""Backend Protocol interfaces for Archivist storage abstraction (Phase 3).

Defines thin ``typing.Protocol`` contracts for the two heterogeneous stores
used by Archivist — a vector store and a relational/graph store — plus a
concrete ``QdrantVectorBackend`` adapter that wraps the existing
``QdrantClient`` singleton.

Design principles
-----------------
* **Protocol-only**: no base classes, no ABC, no runtime overhead.  Structural
  subtyping means any class that exposes the right method signatures satisfies
  the protocol without explicit inheritance.
* **Async throughout**: all mutation methods are ``async def`` so callers never
  block the event loop.  The ``QdrantVectorBackend`` adapter dispatches to the
  synchronous ``QdrantClient`` via ``asyncio.to_thread`` for each call —
  identical to how callers used to call the client inline.
* **PostgreSQL migration path**: implementing ``AsyncpgGraphBackend`` and
  injecting it wherever ``SQLitePool`` is used today would satisfy
  ``GraphBackend`` without touching any write-path business logic.

Usage::

    from archivist.storage.backends import QdrantVectorBackend, VectorBackend
    from archivist.storage.qdrant import qdrant_client

    backend: VectorBackend = QdrantVectorBackend(qdrant_client())
    await backend.upsert("my_collection", [point])
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Protocol, runtime_checkable

from qdrant_client.models import Filter, PointStruct, UpdateResult

logger = logging.getLogger("archivist.backends")


# ---------------------------------------------------------------------------
# Vector-store Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class VectorBackend(Protocol):
    """Thin protocol over any vector store.

    All methods are async; implementations are responsible for thread-offloading
    if the underlying SDK is synchronous.
    """

    async def upsert(
        self,
        collection: str,
        points: list[PointStruct],
    ) -> None:
        """Upsert one or more points into *collection*.

        Idempotent — upserting the same point twice is a safe no-op.
        """
        ...

    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> None:
        """Delete points by *ids* from *collection*.

        404-style responses (point not found) must be treated as success so
        that outbox retry loops are idempotent.
        """
        ...

    async def delete_by_filter(
        self,
        collection: str,
        filt: Filter,
    ) -> None:
        """Delete all points matching *filt* from *collection*."""
        ...

    async def set_payload(
        self,
        collection: str,
        payload: dict[str, Any],
        ids: list[str],
    ) -> None:
        """Set *payload* fields on points identified by *ids*."""
        ...

    async def retrieve(
        self,
        collection: str,
        ids: list[str],
        with_payload: bool = False,
    ) -> list[Any]:
        """Retrieve points by *ids*; return empty list if none found."""
        ...

    async def ensure_collection(
        self,
        collection: str,
        vector_size: int,
    ) -> None:
        """Create *collection* if it does not exist."""
        ...


# ---------------------------------------------------------------------------
# Graph / relational-store Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class GraphBackend(Protocol):
    """Thin protocol over any relational or graph store.

    Both ``SQLiteGraphBackend`` (the refactored ``SQLitePool``) and
    ``AsyncpgGraphBackend`` satisfy this protocol structurally — they expose
    ``execute``, ``executemany``, and ``fetchall`` as convenience methods in
    addition to their concrete ``write()`` / ``read()`` context managers.

    The ``write()`` / ``read()`` context managers remain implementation details
    on the concrete classes, not part of this protocol.  This follows the
    "thin Protocol" philosophy: keep the contract minimal now; expand only
    when a third backend reveals the true common interface.

    Note: this protocol is deliberately *not* a context manager — callers use
    ``MemoryTransaction`` to scope transactions, not the backend directly.
    """

    async def execute(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> Any:
        """Execute *sql* with *params* and return the cursor."""
        ...

    async def executemany(
        self,
        sql: str,
        params: list[tuple[Any, ...]],
    ) -> None:
        """Execute *sql* for each row in *params*."""
        ...

    async def fetchall(
        self,
        sql: str,
        params: tuple[Any, ...] = (),
    ) -> list[tuple[Any, ...]]:
        """Execute *sql* with *params* and return all rows."""
        ...


# ---------------------------------------------------------------------------
# Qdrant adapter
# ---------------------------------------------------------------------------


class QdrantVectorBackend:
    """Concrete ``VectorBackend`` wrapping the shared ``QdrantClient`` singleton.

    All synchronous ``QdrantClient`` calls are dispatched via
    ``asyncio.to_thread`` so the event loop is never blocked.  The adapter
    holds a reference to the client that was passed at construction time —
    callers should pass ``qdrant_client()`` from ``archivist.storage.qdrant``.

    Args:
        client: A ``QdrantClient`` instance.  Typically the singleton returned
            by ``archivist.storage.qdrant.qdrant_client()``.
    """

    def __init__(self, client: Any) -> None:
        self._client = client

    # ------------------------------------------------------------------
    # VectorBackend implementation
    # ------------------------------------------------------------------

    async def upsert(
        self,
        collection: str,
        points: list[PointStruct],
    ) -> None:
        """Upsert *points* into *collection*.

        Args:
            collection: Target collection name.
            points: Non-empty list of ``PointStruct`` objects to upsert.
        """
        client = self._client

        def _do() -> UpdateResult:
            return client.upsert(collection_name=collection, points=points)

        await asyncio.to_thread(_do)

    async def delete(
        self,
        collection: str,
        ids: list[str],
    ) -> None:
        """Delete points by *ids* from *collection*.

        A 404 (point not found) is silently ignored so retry loops are safe.

        Args:
            collection: Target collection name.
            ids: List of point IDs to delete.
        """
        if not ids:
            return
        client = self._client

        def _do() -> None:
            try:
                client.delete(collection_name=collection, points_selector=ids)
            except Exception as exc:
                # 404-equivalent: already deleted — treat as success.
                _msg = str(exc).lower()
                if "not found" in _msg or "404" in _msg:
                    logger.debug(
                        "QdrantVectorBackend.delete: %d ids already absent in %s",
                        len(ids),
                        collection,
                    )
                    return
                raise

        await asyncio.to_thread(_do)

    async def delete_by_filter(
        self,
        collection: str,
        filt: Filter,
    ) -> None:
        """Delete all points matching *filt* from *collection*.

        Args:
            collection: Target collection name.
            filt: Qdrant ``Filter`` expression.
        """
        client = self._client

        def _do() -> None:
            client.delete(collection_name=collection, points_selector=filt)

        await asyncio.to_thread(_do)

    async def set_payload(
        self,
        collection: str,
        payload: dict[str, Any],
        ids: list[str],
    ) -> None:
        """Set *payload* fields on points identified by *ids*.

        Args:
            collection: Target collection name.
            payload: Key-value pairs to set on the matched points.
            ids: List of point IDs to update.
        """
        if not ids:
            return
        client = self._client

        def _do() -> None:
            client.set_payload(collection_name=collection, payload=payload, points=ids)

        await asyncio.to_thread(_do)

    async def retrieve(
        self,
        collection: str,
        ids: list[str],
        with_payload: bool = False,
    ) -> list[Any]:
        """Retrieve points by *ids*.

        Args:
            collection: Source collection name.
            ids: List of point IDs to fetch.
            with_payload: Whether to include payload in the response.

        Returns:
            List of ``ScoredPoint`` or ``Record`` objects (may be empty).
        """
        if not ids:
            return []
        client = self._client

        def _do() -> list[Any]:
            return client.retrieve(
                collection_name=collection,
                ids=ids,
                with_payload=with_payload,
            )

        return await asyncio.to_thread(_do)

    async def ensure_collection(
        self,
        collection: str,
        vector_size: int,
    ) -> None:
        """Create *collection* if it does not exist.

        Args:
            collection: Collection name to create.
            vector_size: Dimensionality of vectors stored in the collection.
        """
        from qdrant_client.models import Distance, VectorParams

        client = self._client

        def _do() -> None:
            existing = {c.name for c in client.get_collections().collections}
            if collection not in existing:
                client.create_collection(
                    collection_name=collection,
                    vectors_config=VectorParams(
                        size=vector_size,
                        distance=Distance.COSINE,
                    ),
                )
                logger.info(
                    "QdrantVectorBackend: created collection %s (dim=%d)",
                    collection,
                    vector_size,
                )

        await asyncio.to_thread(_do)
