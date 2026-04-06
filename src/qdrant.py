"""Shared Qdrant client factory.

All modules that need a QdrantClient should import and call ``qdrant_client()``
rather than constructing one inline, so that URL and timeout settings are
configured in one place.
"""

from qdrant_client import QdrantClient

from config import QDRANT_URL


def qdrant_client(timeout: int = 30) -> QdrantClient:
    """Return a configured QdrantClient instance."""
    return QdrantClient(url=QDRANT_URL, timeout=timeout)
