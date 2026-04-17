"""Webhook dispatcher — fires HTTP POSTs on configurable Archivist events.

Events: memory_store, memory_conflict, skill_event, trajectory_logged,
        annotation_added, cache_invalidated.

Configure via WEBHOOK_URL (single endpoint) or WEBHOOK_CONFIG_PATH (YAML
with per-event routing). Fires asynchronously — never blocks the caller.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone

import httpx

from archivist.core.config import WEBHOOK_URL, WEBHOOK_TIMEOUT, WEBHOOK_EVENTS

logger = logging.getLogger("archivist.webhooks")


async def fire(event: str, payload: dict) -> bool:
    """Fire a webhook for the given event. Returns True if accepted, False on error."""
    if not WEBHOOK_URL:
        return False
    if WEBHOOK_EVENTS and event not in WEBHOOK_EVENTS:
        return False

    body = {
        "event": event,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "data": payload,
    }

    from archivist.core.metrics import inc, WEBHOOK_FIRE, WEBHOOK_FAIL
    inc(WEBHOOK_FIRE, {"event": event})

    try:
        async with httpx.AsyncClient(timeout=WEBHOOK_TIMEOUT) as client:
            resp = await client.post(WEBHOOK_URL, json=body)
            if resp.status_code >= 400:
                logger.warning("Webhook %s returned %d", event, resp.status_code)
                inc(WEBHOOK_FAIL, {"event": event})
                return False
            return True
    except Exception as e:
        logger.warning("Webhook %s failed: %s", event, e)
        inc(WEBHOOK_FAIL, {"event": event})
        return False


_pending_fires: set[asyncio.Task] = set()


def _log_fire_exception(task: asyncio.Task):
    """Log unhandled exceptions from fire-and-forget webhook tasks."""
    if task.cancelled():
        return
    exc = task.exception()
    if exc is not None:
        logger.error("Webhook task %r failed: %s", task.get_name(), exc)


def fire_background(event: str, payload: dict):
    """Schedule a webhook fire without awaiting it (fire-and-forget).

    Task references are held in _pending_fires so they are not garbage-collected
    before completion and any exceptions are logged rather than silently dropped.
    """
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(fire(event, payload), name=f"webhook_{event}")
        _pending_fires.add(task)
        task.add_done_callback(_log_fire_exception)
        task.add_done_callback(_pending_fires.discard)
    except RuntimeError:
        pass
