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

from config import WEBHOOK_URL, WEBHOOK_TIMEOUT, WEBHOOK_EVENTS

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

    from metrics import inc, WEBHOOK_FIRE, WEBHOOK_FAIL
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


def fire_background(event: str, payload: dict):
    """Schedule a webhook fire without awaiting it (fire-and-forget)."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(fire(event, payload))
    except RuntimeError:
        pass
