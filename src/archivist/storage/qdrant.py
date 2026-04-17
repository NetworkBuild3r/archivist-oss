"""Shared Qdrant client singleton.

All modules that need a QdrantClient should import and call ``qdrant_client()``
rather than constructing one inline.  The client is created once and reused for
the lifetime of the process, avoiding per-call TCP/gRPC setup overhead.

Callers that need a non-default timeout can pass ``timeout=`` to individual
Qdrant operations (e.g. ``client.scroll(..., timeout=60)``).
"""

import threading

from qdrant_client import QdrantClient

from archivist.core.config import QDRANT_URL

_instance: QdrantClient | None = None
_instance_lock = threading.Lock()


def qdrant_client(timeout: int = 30) -> QdrantClient:
    """Return the shared QdrantClient instance (created on first call).

    Uses double-checked locking so the common read-path (instance already
    created) is lock-free.  The ``threading.Lock`` protects the one-time
    initialisation from concurrent callers racing at startup.
    """
    global _instance
    if _instance is not None:
        return _instance
    with _instance_lock:
        if _instance is None:
            _instance = QdrantClient(url=QDRANT_URL, timeout=timeout)
    return _instance
