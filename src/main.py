"""Archivist main entrypoint — starts MCP server, file watcher, curator loop, and initial index."""

import asyncio
from contextlib import asynccontextmanager
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone

from watchfiles import awatch, Change

from config import (
    MEMORY_ROOT, MCP_PORT, QDRANT_URL, QDRANT_COLLECTION, VECTOR_DIM, ARCHIVIST_API_KEY,
    CURATOR_QUEUE_DRAIN_INTERVAL, ARCHIVIST_INVALIDATION_EXPORT_PATH,
)
from qdrant import qdrant_client
import health
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
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager

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
        health.register("qdrant", healthy=False, detail=f"Qdrant not reachable after 120s: {last_err}")
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

    health.register("qdrant", healthy=True)


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


async def handle_health(_request):
    return JSONResponse({"status": "ok", "service": "archivist", "version": "1.0.0"})


async def handle_invalidate(_request):
    """Endpoint to delete expired memories (TTL-based)."""
    import metrics as met

    from qdrant_client.models import Filter, FieldCondition, Range

    t0 = time.monotonic()
    now_ts = int(time.time())
    client = qdrant_client()

    try:
        expired = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[FieldCondition(key="ttl_expires_at", range=Range(lte=now_ts, gt=0))]
            ),
            limit=1000,
            with_payload=True,
        )

        points = expired[0]
        point_ids = [p.id for p in points]
        sample = [str(x) for x in point_ids[:5]]
        sample_ns = []
        for p in points[:5]:
            pl = getattr(p, "payload", None) or {}
            sample_ns.append(str(pl.get("namespace", "") or ""))

        if point_ids:
            client.delete(
                collection_name=QDRANT_COLLECTION,
                points_selector=point_ids,
            )

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
        dur_ms = round((time.monotonic() - t0) * 1000, 1)
        n = len(point_ids)
        met.observe(met.INVALIDATE_DURATION, dur_ms)
        met.inc(met.INVALIDATE_COUNT, value=float(n))

        logger.info(
            "invalidation.complete count=%d duration_ms=%.1f sample_ids=%s sample_namespaces=%s",
            n,
            dur_ms,
            sample,
            sample_ns,
        )

        if ARCHIVIST_INVALIDATION_EXPORT_PATH:
            try:
                line = {
                    "ts": datetime.now(timezone.utc).isoformat(),
                    "count": n,
                    "duration_ms": dur_ms,
                    "sample_ids": sample,
                    "sample_namespaces": sample_ns,
                    "reason": "ttl_expired",
                }
                with open(ARCHIVIST_INVALIDATION_EXPORT_PATH, "a", encoding="utf-8") as f:
                    f.write(json.dumps(line, ensure_ascii=False) + "\n")
                    f.flush()
            except OSError as ex:
                logger.warning("invalidation export append failed: %s", ex)

        return JSONResponse({"invalidated": n})
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
    try:
        global streamable_http_session_manager
        streamable_http_session_manager = _create_streamable_http_session_manager()
        async with streamable_http_session_manager.run():
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
    finally:
        streamable_http_session_manager = None


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
streamable_http_session_manager: StreamableHTTPSessionManager | None = None


def _create_streamable_http_session_manager() -> StreamableHTTPSessionManager:
    """Create a fresh Streamable HTTP session manager for each app lifespan."""
    return StreamableHTTPSessionManager(server, json_response=True)


class SseASGIApp:
    async def __call__(self, scope, receive, send):
        from observability import reset_request_id, set_request_id_from_scope

        token = set_request_id_from_scope(scope)
        try:
            async with sse_transport.connect_sse(scope, receive, send) as streams:
                await server.run(
                    streams[0], streams[1], server.create_initialization_options()
                )
        finally:
            reset_request_id(token)


class StreamableHTTPASGIApp:
    async def __call__(self, scope, receive, send):
        from observability import reset_request_id, set_request_id_from_scope

        token = set_request_id_from_scope(scope)
        try:
            manager = streamable_http_session_manager
            if manager is None:
                raise RuntimeError("Streamable HTTP session manager is not initialized")
            await manager.handle_request(scope, receive, send)
        finally:
            reset_request_id(token)


sse_app = SseASGIApp()
streamable_http_app = StreamableHTTPASGIApp()


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
        Route("/health", handle_health),
        Route("/metrics", handle_metrics),
        Route("/admin/invalidate", handle_invalidate, methods=["POST", "GET"]),
        Route("/admin/retrieval-logs", handle_retrieval_export),
        Route("/admin/dashboard", handle_dashboard),
        Route("/admin/namespace-index", handle_namespace_index, methods=["GET"]),
        Route("/mcp", endpoint=streamable_http_app, methods=["GET", "POST", "DELETE"]),
        Route("/mcp/sse", endpoint=sse_app, methods=["GET"]),
        Mount("/mcp/messages/", app=sse_transport.handle_post_message),
    ],
    middleware=[Middleware(ArchivistAuthMiddleware)],
    lifespan=lifespan,
)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=MCP_PORT)
