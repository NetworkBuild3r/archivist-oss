"""Subsystem health registry -- tracks operational status of external dependencies.

Any module can call ``register("name", healthy=True/False)`` to report its
status.  Other modules call ``is_healthy("name")`` to decide whether to
attempt an operation or degrade gracefully.  ``all_status()`` returns the
full snapshot for the health dashboard.
"""

import logging
import threading
from datetime import datetime, timezone

logger = logging.getLogger("archivist.health")

_lock = threading.Lock()
_status: dict[str, dict] = {}


def register(name: str, healthy: bool = True, detail: str = ""):
    """Record (or update) the health of a named subsystem."""
    with _lock:
        _status[name] = {
            "healthy": healthy,
            "detail": detail,
            "since": datetime.now(timezone.utc).isoformat(),
        }
    level = logging.INFO if healthy else logging.ERROR
    logger.log(
        level, "Subsystem %s: %s%s",
        name, "UP" if healthy else "DOWN",
        f" ({detail})" if detail else "",
    )


def is_healthy(name: str) -> bool:
    """Return False only if the subsystem has been explicitly marked unhealthy.

    Unknown subsystems (never registered) are assumed healthy so that the
    registry is opt-in and cannot block callers that pre-date it.
    """
    with _lock:
        entry = _status.get(name)
    return entry["healthy"] if entry else True


def all_status() -> dict[str, dict]:
    """Return a snapshot of every registered subsystem's status."""
    with _lock:
        return dict(_status)
