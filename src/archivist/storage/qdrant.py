"""Shared Qdrant client singleton.

All modules that need a QdrantClient should import and call ``qdrant_client()``
rather than constructing one inline.  The client is created once and reused for
the lifetime of the process, avoiding per-call TCP/gRPC setup overhead.

Callers that need a non-default timeout can pass ``timeout=`` to individual
Qdrant operations (e.g. ``client.scroll(..., timeout=60)``).
"""

from qdrant_client import QdrantClient

from config import QDRANT_URL

_instance: QdrantClient | None = None


def qdrant_client(timeout: int = 30) -> QdrantClient:
    """Return the shared QdrantClient instance (created on first call)."""
    global _instance
    if _instance is None:
        _instance = QdrantClient(url=QDRANT_URL, timeout=timeout)
    return _instance
