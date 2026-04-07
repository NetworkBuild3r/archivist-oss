"""Archivist main entrypoint — starts MCP server, file watcher, curator loop, and initial index."""

import asyncio
from contextlib import asynccontextmanager
import logging
import os
import sys
import time

from watchfiles import awatch, Change

from config import MEMORY_ROOT, MCP_PORT, QDRANT_URL, QDRANT_COLLECTION, VECTOR_DIM, ARCHIVIST_API_KEY, CURATOR_QUEUE_DRAIN_INTERVAL
from qdrant import qdrant_client
from graph import init_schema
from indexer import full_index, index_file, delete_file_points
from curator import curator_loop
from curator_queue import drain as drain_curator_queue
from mcp_server import server
from rbac import load_config as load_rbac_config

from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.routing import Mount, Route
from starlette.responses import JSONResponse, PlainTextResponse
from mcp.server.sse import SseServerTransport

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("archivist")


def ensure_qdrant_collection():
    """Create or migrate the Qdrant collection to the target vector dimension.

    Retries until Qdrant accepts connections (Docker Compose may start archivist
    before qdrant is listening; avoids brittle image-specific healthchecks).
    """
    from qdrant_client import QdrantClient
    from qdrant_client.models import VectorParams, Distance, PayloadSchemaType

    deadline = time.monotonic() + 120
    last_err: Exception | None = None
    client: QdrantClient | None = None
    collections: list[str] = []
    while time.monotonic() < deadline:
        try:
            client = qdrant_client()
            collections = [c.name for c in client.get_collections().collections]
            break
        except Exception as e:
            last_err = e
            logger.warning("Waiting for Qdrant at %s: %s — retrying in 2s", QDRANT_URL, e)
            time.sleep(2)
    else:
        raise RuntimeError(f"Qdrant not reachable at {QDRANT_URL} after 120s") from last_err

    assert client is not None

    needs_create = QDRANT_COLLECTION not in collections

    if not needs_create:
        info = client.get_collection(QDRANT_COLLECTION)
        current_dim = info.config.params.vectors.size
        if current_dim != VECTOR_DIM:
            logger.warning(
                "Collection '%s' has %d-dim vectors but target is %d-dim — recreating",
                QDRANT_COLLECTION, current_dim, VECTOR_DIM,
            )
            client.delete_collection(QDRANT_COLLECTION)
            needs_create = True
        else:
            logger.info(
                "Qdrant collection '%s' exists: %d points (%d-dim)",
                QDRANT_COLLECTION, info.points_count, current_dim,
            )

    if needs_create:
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE),
        )
        logger.info("Created Qdrant collection '%s' (%d-dim)", QDRANT_COLLECTION, VECTOR_DIM)

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
        ]:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=schema,
            )
        logger.info("Created payload indexes for %s", QDRANT_COLLECTION)


async def file_watcher():
    """Watch MEMORY_ROOT for .md file changes and re-index them."""
    if not os.path.isdir(MEMORY_ROOT):
        logger.warning("MEMORY_ROOT %s does not exist — file watcher disabled", MEMORY_ROOT)
        return
    logger.info("File watcher started on %s", MEMORY_ROOT)
    async for changes in awatch(MEMORY_ROOT):
        for change_type, path in changes:
            if not path.endswith(".md"):
                continue
            if change_type in (Change.added, Change.modified):
                try:
                    await index_file(path)
                except Exception as e:
                    logger.error("Watcher index failed for %s: %s", path, e)
            elif change_type == Change.deleted:
                try:
                    await delete_file_points(path)
                except Exception as e:
                    logger.error("Watcher delete failed for %s: %s", path, e)


async def health(_request):
    return JSONResponse({"status": "ok", "service": "archivist", "version": "1.0.0"})


async def handle_invalidate(_request):
    """Endpoint to delete expired memories (TTL-based)."""
    from qdrant_client.models import Filter, FieldCondition, Range

    now_ts = int(time.time())
    client = qdrant_client(timeout=60)

    try:
        expired = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="ttl_expires_at", range=Range(lte=now_ts, gt=0))]
            ),
            limit=1000,
            with_payload=True,
        )

        point_ids = [p.id for p in expired[0]]
        if point_ids:
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=point_ids,
            )
            logger.info("Invalidation: deleted %d expired memories", len(point_ids))

            from audit import log_memory_event
            for pid in point_ids:
                await log_memory_event(
                    agent_id="system",
                    action="delete",
                    memory_id=str(pid),
                    namespace="",
                    text_hash="",
                    version=0,
                    metadata={"trigger": "invalidation", "reason": "ttl_expired"},
                )
        else:
            logger.info("Invalidation: no expired memories found")

        return JSONResponse({"invalidated": len(point_ids)})
    except Exception as e:
        logger.error("Invalidation failed: %s", e)
        return JSONResponse({"error": str(e)}, status_code=500)


_background_tasks: list[asyncio.Task] = []


def _log_task_exception(task: asyncio.Task):
    """Callback for background tasks — logs unhandled exceptions instead of silently dropping them."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.exception("Background task %r crashed", task.get_name(), exc_info=exc)


async def _startup():
    """Run on app startup: init DB, load RBAC, ensure Qdrant collection, start background tasks."""
    logger.info("Archivist v1.0.0 starting up...")

    init_schema()
    logger.info("Graph schema initialized")

    load_rbac_config()
    logger.info("RBAC config loaded")

    ensure_qdrant_collection()

    for coro, name in [
        (run_initial_index(), "initial_index"),
        (file_watcher(), "file_watcher"),
        (curator_loop(), "curator_loop"),
        (curator_queue_drain_loop(), "curator_queue_drain"),
    ]:
        t = asyncio.create_task(coro, name=name)
        t.add_done_callback(_log_task_exception)
        _background_tasks.append(t)
    logger.info("Background tasks started: %s", [t.get_name() for t in _background_tasks])


@asynccontextmanager
async def lifespan(_app: Starlette):
    """Starlette ≥0.37 lifespan (replaces deprecated on_startup / on_shutdown)."""
    await _startup()
    try:
        yield
    finally:
        for t in _background_tasks:
            if not t.done():
                t.cancel()
        if _background_tasks:
            await asyncio.gather(*_background_tasks, return_exceptions=True)
        _background_tasks.clear()


async def curator_queue_drain_loop():
    """Periodically drain the curator write-ahead queue."""
    logger.info("Curator queue drain loop started (interval: %ds)", CURATOR_QUEUE_DRAIN_INTERVAL)
    while True:
        try:
            applied = drain_curator_queue(limit=50)
            if applied:
                logger.info("Curator queue: drained %d operations", len(applied))
        except Exception as e:
            logger.error("Curator queue drain error: %s", e)
        await asyncio.sleep(CURATOR_QUEUE_DRAIN_INTERVAL)


async def run_initial_index():
    """Run initial index in background so health endpoint is available immediately."""
    try:
        total = await full_index()
        logger.info("Initial index complete: %d chunks", total)
    except Exception as e:
        logger.error("Initial index failed: %s", e)


sse_transport = SseServerTransport("/mcp/messages/")


class ArchivistAuthMiddleware(BaseHTTPMiddleware):
    """Optional API key on all routes except GET /health (for probes)."""

    async def dispatch(self, request, call_next):
        if request.url.path == "/health":
            return await call_next(request)
        if not ARCHIVIST_API_KEY:
            return await call_next(request)
        auth = request.headers.get("authorization", "")
        xkey = request.headers.get("x-api-key", "")
        ok = auth == f"Bearer {ARCHIVIST_API_KEY}" or xkey == ARCHIVIST_API_KEY
        if not ok:
            return JSONResponse({"error": "unauthorized"}, status_code=401)
        return await call_next(request)


async def handle_sse(request):
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await server.run(
            streams[0], streams[1], server.create_initialization_options()
        )


async def handle_retrieval_export(request):
    """REST endpoint for retrieval log export (dashboard/debugging)."""
    from retrieval_log import get_retrieval_logs, get_retrieval_stats
    params = request.query_params
    if params.get("stats") == "true":
        stats = get_retrieval_stats(
            agent_id=params.get("agent_id", ""),
            window_days=int(params.get("window_days", "7")),
        )
        return JSONResponse(stats)
    logs = get_retrieval_logs(
        agent_id=params.get("agent_id", ""),
        limit=int(params.get("limit", "50")),
        since=params.get("since", ""),
    )
    return JSONResponse({"logs": logs, "count": len(logs)})


async def handle_metrics(_request):
    """Prometheus text exposition format."""
    from metrics import render
    return PlainTextResponse(render(), media_type="text/plain; version=0.0.4; charset=utf-8")


async def handle_dashboard(request):
    """Health dashboard JSON."""
    from dashboard import build_dashboard, batch_heuristic
    params = request.query_params
    window = int(params.get("window_days", "7"))
    if params.get("batch") == "true":
        return JSONResponse(batch_heuristic(window))
    return JSONResponse(build_dashboard(window))


async def handle_namespace_index(request):
    """Plain-text per-agent memory index (same output as archivist_index MCP tool).

    Used by CronJobs to write MEMORY_INDEX.md on agent NFS workspaces.
    Query: agent_id (required).
    """
    agent_id = request.query_params.get("agent_id", "").strip()
    if not agent_id:
        return PlainTextResponse("missing agent_id query parameter", status_code=400)
    from compressed_index import build_namespace_index
    from rbac import get_namespace_for_agent

    namespace = get_namespace_for_agent(agent_id)
    text = build_namespace_index(namespace, agent_ids=[agent_id])
    return PlainTextResponse(text, media_type="text/markdown; charset=utf-8")


app = Starlette(
    routes=[
        Route("/health", health),
        Route("/metrics", handle_metrics),
        Route("/admin/invalidate", handle_invalidate, methods=["POST", "GET"]),
        Route("/admin/retrieval-logs", handle_retrieval_export),
        Route("/admin/dashboard", handle_dashboard),
        Route("/admin/namespace-index", handle_namespace_index, methods=["GET"]),
        Route("/mcp/sse", endpoint=handle_sse),
        Mount("/mcp/messages/", app=sse_transport.handle_post_message),
    ],
    middleware=[Middleware(ArchivistAuthMiddleware)],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=MCP_PORT)
