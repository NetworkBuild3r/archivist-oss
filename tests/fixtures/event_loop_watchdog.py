"""Asyncio event-loop stall watchdog (INIT-025/SPEC-012).

Detects synchronous blocking I/O or CPU work running directly on the asyncio
event loop for longer than a configurable threshold (default ``50ms``).

Uses Python's built-in asyncio debug slow-callback instrumentation
(``loop.set_debug(True)`` + ``loop.slow_callback_duration``) paired with a
scoped handler on the ``asyncio`` logger so any single callback exceeding the
threshold fails the test deterministically with actionable diagnostics.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass

DEFAULT_STALL_THRESHOLD_MS: float = 50.0

_SLOW_CALLBACK_RE = re.compile(
    r"Executing\s+(?P<callback>.+?)\s+took\s+(?P<seconds>[0-9]+(?:\.[0-9]+)?)\s+seconds"
)


@dataclass(frozen=True)
class StallRecord:
    """Single observed event-loop callback stall."""

    callback: str
    duration_ms: float
    message: str


class EventLoopStallError(AssertionError):
    """Raised when one or more asyncio callbacks block the loop beyond threshold."""

    def __init__(self, stalls: list[StallRecord], threshold_ms: float) -> None:
        self.stalls = list(stalls)
        self.threshold_ms = threshold_ms
        details = "\n".join(
            f"  - {s.duration_ms:.1f}ms (> {threshold_ms:.1f}ms): {s.callback}" for s in stalls
        )
        super().__init__(
            f"Detected {len(stalls)} asyncio event-loop stall(s) exceeding "
            f"{threshold_ms:.1f}ms (offload blocking I/O with asyncio.to_thread):\n{details}"
        )


class _SlowCallbackLogHandler(logging.Handler):
    """Capture asyncio slow-callback warnings exceeding threshold_s."""

    def __init__(self, threshold_s: float) -> None:
        super().__init__(level=logging.WARNING)
        self._threshold_s = threshold_s
        self.stalls: list[StallRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        msg = record.getMessage()
        match = _SLOW_CALLBACK_RE.search(msg)
        if not match:
            return
        seconds = float(match.group("seconds"))
        if seconds >= self._threshold_s:
            self.stalls.append(
                StallRecord(
                    callback=match.group("callback"),
                    duration_ms=seconds * 1000.0,
                    message=msg,
                )
            )


class EventLoopStallWatchdog:
    """Scoped detector for blocking callbacks on the active asyncio event loop.

    Usage::

        async def test_handler():
            with EventLoopStallWatchdog(threshold_ms=50.0) as watchdog:
                await my_async_handler()
            # Raises EventLoopStallError if any callback exceeded 50ms.
    """

    def __init__(
        self,
        threshold_ms: float = DEFAULT_STALL_THRESHOLD_MS,
        *,
        loop: asyncio.AbstractEventLoop | None = None,
    ) -> None:
        if threshold_ms <= 0:
            raise ValueError("threshold_ms must be positive")
        self.threshold_ms = float(threshold_ms)
        self._threshold_s = self.threshold_ms / 1000.0
        self._loop = loop
        self._active_loop: asyncio.AbstractEventLoop | None = None
        self._prev_debug: bool = False
        self._prev_slow_duration: float = 0.1
        self._prev_logger_level: int = logging.NOTSET
        self._prev_logger_disabled: bool = False
        self._handler: _SlowCallbackLogHandler | None = None

    @property
    def stalls(self) -> list[StallRecord]:
        return list(self._handler.stalls) if self._handler is not None else []

    def start(self) -> EventLoopStallWatchdog:
        loop = self._loop or asyncio.get_running_loop()
        self._active_loop = loop
        self._prev_debug = loop.get_debug()
        self._prev_slow_duration = getattr(loop, "slow_callback_duration", 0.1)

        asyncio_logger = logging.getLogger("asyncio")
        self._prev_logger_level = asyncio_logger.level
        self._prev_logger_disabled = asyncio_logger.disabled
        if asyncio_logger.disabled:
            asyncio_logger.disabled = False
        if asyncio_logger.level > logging.WARNING:
            asyncio_logger.setLevel(logging.WARNING)

        self._handler = _SlowCallbackLogHandler(self._threshold_s)
        asyncio_logger.addHandler(self._handler)

        loop.set_debug(True)
        loop.slow_callback_duration = self._threshold_s
        return self

    def stop(self) -> list[StallRecord]:
        recorded = self.stalls
        asyncio_logger = logging.getLogger("asyncio")
        if self._handler is not None:
            asyncio_logger.removeHandler(self._handler)
        asyncio_logger.setLevel(self._prev_logger_level)
        asyncio_logger.disabled = self._prev_logger_disabled

        if self._active_loop is not None:
            self._active_loop.slow_callback_duration = self._prev_slow_duration
            self._active_loop.set_debug(self._prev_debug)
            self._active_loop = None
        return recorded

    def assert_no_stalls(self) -> None:
        if self.stalls:
            raise EventLoopStallError(self.stalls, self.threshold_ms)

    def __enter__(self) -> EventLoopStallWatchdog:
        return self.start()

    def __exit__(self, exc_type, _exc, _tb) -> None:
        recorded = self.stop()
        if exc_type is None and recorded:
            raise EventLoopStallError(recorded, self.threshold_ms)
