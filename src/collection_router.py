"""Namespace-sharded Qdrant collection routing (v1.10 enterprise scaling).

At mega-scale (5M+ points) a single Qdrant collection degrades.  This
module provides transparent namespace-to-collection mapping so each
namespace gets its own smaller, faster collection while existing code
continues to call a single routing API.

When ``NAMESPACE_SHARDING_ENABLED`` is False (default), everything maps
to the single ``QDRANT_COLLECTION`` — zero behaviour change from before.

When enabled:
  - ``collection_for(namespace)`` returns ``QDRANT_COLLECTION_{namespace}``
  - ``collections_for_query(namespace)`` returns either [single] or [all]
    for cross-namespace searches
  - ``ensure_collection(namespace)`` lazily creates a shard collection
    with the same HNSW config and payload indexes as the primary
"""

import logging
import threading

from config import (
    QDRANT_COLLECTION,
    VECTOR_DIM,
    NAMESPACE_SHARDING_ENABLED,
    SINGLE_COLLECTION_MODE,
    QDRANT_HNSW_M,
    QDRANT_HNSW_EF_CONSTRUCT,
)
from qdrant import qdrant_client

logger = logging.getLogger("archivist.collection_router")

_MAX_KNOWN_COLLECTIONS = 1024
_known_collections: set[str] = set()
_known_lock = threading.Lock()


def collection_for(namespace: str) -> str:
    """Return the Qdrant collection name for a given namespace."""
    if not NAMESPACE_SHARDING_ENABLED or SINGLE_COLLECTION_MODE or not namespace:
        return QDRANT_COLLECTION
    safe_ns = namespace.replace("/", "_").replace(" ", "_").lower()
    return f"{QDRANT_COLLECTION}_{safe_ns}"


def collections_for_query(namespace: str = "") -> list[str]:
    """Return the list of collections to search for a given query.

    - If a specific namespace is given, returns just that shard.
    - If no namespace is given (fleet-wide search), returns all known shards.
    - If sharding is disabled, always returns the single collection.
    """
    if not NAMESPACE_SHARDING_ENABLED or SINGLE_COLLECTION_MODE:
        return [QDRANT_COLLECTION]
    if namespace:
        return [collection_for(namespace)]
    with _known_lock:
        if _known_collections:
            return sorted(_known_collections)
    return [QDRANT_COLLECTION]


def ensure_collection(namespace: str) -> str:
    """Ensure the shard collection for ``namespace`` exists; create if needed.

    Returns the collection name.  This is idempotent and safe to call on
    every write — it only creates a collection on first encounter.
    """
    name = collection_for(namespace)
    if name == QDRANT_COLLECTION:
        return name

    with _known_lock:
        if name in _known_collections:
            return name

    client = qdrant_client()
    try:
        existing = [c.name for c in client.get_collections().collections]
    except Exception as e:
        logger.warning("Cannot list collections: %s — falling back to primary", e)
        return QDRANT_COLLECTION

    if name not in existing:
        from qdrant_client.models import VectorParams, Distance, HnswConfigDiff, PayloadSchemaType, OptimizersConfigDiff

        try:
            client.create_collection(
                collection_name=name,
                vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
                hnsw_config=HnswConfigDiff(m=QDRANT_HNSW_M, ef_construct=QDRANT_HNSW_EF_CONSTRUCT),
                optimizers_config=OptimizersConfigDiff(indexing_threshold=0),
            )
            for field, schema in [
                ("agent_id", PayloadSchemaType.KEYWORD),
                ("file_path", PayloadSchemaType.KEYWORD),
                ("file_type", PayloadSchemaType.KEYWORD),
                ("team", PayloadSchemaType.KEYWORD),
                ("date", PayloadSchemaType.KEYWORD),
                ("namespace", PayloadSchemaType.KEYWORD),
                ("version", PayloadSchemaType.INTEGER),
                ("ttl_expires_at", PayloadSchemaType.INTEGER),
                ("consistency_level", PayloadSchemaType.KEYWORD),
                ("checksum", PayloadSchemaType.KEYWORD),
                ("parent_id", PayloadSchemaType.KEYWORD),
                ("is_parent", PayloadSchemaType.BOOL),
                ("importance_score", PayloadSchemaType.FLOAT),
                ("memory_type", PayloadSchemaType.KEYWORD),
                ("retention_class", PayloadSchemaType.KEYWORD),
                ("topic", PayloadSchemaType.KEYWORD),
                ("thought_type", PayloadSchemaType.KEYWORD),
                ("text", PayloadSchemaType.TEXT),
            ]:
                client.create_payload_index(
                    collection_name=name, field_name=field, field_schema=schema,
                )
            logger.info("Created shard collection '%s' for namespace '%s'", name, namespace)
        except Exception as e:
            logger.warning("Failed to create shard collection '%s': %s", name, e)
            return QDRANT_COLLECTION

    with _known_lock:
        if len(_known_collections) >= _MAX_KNOWN_COLLECTIONS:
            logger.warning("Known collections set at cap (%d) — not adding '%s'", _MAX_KNOWN_COLLECTIONS, name)
        else:
            _known_collections.add(name)
            _known_collections.add(QDRANT_COLLECTION)

    return name


def drop_collection(namespace: str) -> bool:
    """Drop the shard collection for a namespace (instant deletion)."""
    name = collection_for(namespace)
    if name == QDRANT_COLLECTION:
        return False
    client = qdrant_client()
    try:
        client.delete_collection(name)
        with _known_lock:
            _known_collections.discard(name)
        logger.info("Dropped shard collection '%s' for namespace '%s'", name, namespace)
        return True
    except Exception as e:
        logger.warning("Failed to drop shard collection '%s': %s", name, e)
        return False


def refresh_known_collections() -> int:
    """Scan Qdrant for existing shard collections and populate the cache."""
    if not NAMESPACE_SHARDING_ENABLED:
        return 0
    client = qdrant_client()
    try:
        prefix = f"{QDRANT_COLLECTION}_"
        existing = [c.name for c in client.get_collections().collections]
        shards = [c for c in existing if c.startswith(prefix) or c == QDRANT_COLLECTION]
        with _known_lock:
            _known_collections.clear()
            _known_collections.update(shards)
        return len(shards)
    except Exception as e:
        logger.warning("Failed to refresh shard list: %s", e)
        return 0
