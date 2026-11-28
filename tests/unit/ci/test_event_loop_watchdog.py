"""Tests for the asyncio >50ms event-loop stall watchdog (INIT-025/SPEC-012)."""

from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest
from tests.fixtures.event_loop_watchdog import (
    EventLoopStallError,
    EventLoopStallWatchdog,
)

pytestmark = [pytest.mark.unit]


def _subprocess_env(repo_root: Path, extra: dict[str, str] | None = None) -> dict[str, str]:
    env = dict(os.environ)
    prefix = f"{repo_root}{os.pathsep}{repo_root / 'src'}"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = f"{prefix}{os.pathsep}{existing}" if existing else prefix
    if extra:
        env.update(extra)
    return env


async def test_watchdog_allows_yielding_async_work(event_loop_watchdog) -> None:
    """Properly yielding awaits and offloaded thread work never trip the watchdog."""
    await asyncio.sleep(0.07)
    await asyncio.to_thread(time.sleep, 0.07)
    assert event_loop_watchdog.stalls == []


@pytest.mark.allow_event_loop_stall
async def test_watchdog_detects_blocking_sleep_over_50ms_async_with() -> None:
    """`async with EventLoopStallWatchdog()` detects synchronous blocking without manual awaits."""
    with pytest.raises(EventLoopStallError) as exc_info:
        async with EventLoopStallWatchdog(threshold_ms=50.0):
            time.sleep(0.07)

    err = exc_info.value
    assert len(err.stalls) >= 1
    assert any(s.duration_ms >= 50.0 for s in err.stalls)


async def test_watchdog_async_with_allows_yielding_sleep() -> None:
    """`async with EventLoopStallWatchdog()` does not false-positive on `await asyncio.sleep`."""
    async with EventLoopStallWatchdog(threshold_ms=50.0) as wd:
        await asyncio.sleep(0.07)
    assert wd.stalls == []


@pytest.mark.allow_event_loop_stall
async def test_watchdog_sync_with_detects_bare_blocking_sleep_without_await() -> None:
    """Sync `with EventLoopStallWatchdog(): time.sleep(0.07)` detects stall with no trailing await."""
    with pytest.raises(EventLoopStallError) as exc_info:
        with EventLoopStallWatchdog(threshold_ms=50.0):
            time.sleep(0.07)

    err = exc_info.value
    assert len(err.stalls) >= 1
    assert any(s.duration_ms >= 50.0 for s in err.stalls)


async def test_watchdog_sync_with_allows_yielding_await() -> None:
    """Sync `with EventLoopStallWatchdog():` does not false-positive on yielding awaits."""
    with EventLoopStallWatchdog(threshold_ms=50.0) as wd:
        await asyncio.sleep(0.07)
    assert wd.stalls == []


@pytest.mark.allow_event_loop_stall
async def test_watchdog_detects_stall_when_root_logger_is_error() -> None:
    """Effective level > WARNING (via root logger=ERROR/CRITICAL) must not mask stalls."""
    root_logger = logging.getLogger()
    asyncio_logger = logging.getLogger("asyncio")
    orig_root_level = root_logger.level
    orig_asyncio_level = asyncio_logger.level
    try:
        root_logger.setLevel(logging.CRITICAL)
        asyncio_logger.setLevel(logging.NOTSET)
        assert asyncio_logger.getEffectiveLevel() == logging.CRITICAL

        with pytest.raises(EventLoopStallError) as exc_info:
            async with EventLoopStallWatchdog(threshold_ms=50.0):
                time.sleep(0.07)
        assert any(s.duration_ms >= 50.0 for s in exc_info.value.stalls)
    finally:
        asyncio_logger.setLevel(orig_asyncio_level)
        root_logger.setLevel(orig_root_level)


@pytest.mark.allow_event_loop_stall
async def test_watchdog_restores_loop_debug_and_threshold() -> None:
    """Event-loop debug flag and slow_callback_duration are restored cleanly on exit."""
    loop = asyncio.get_running_loop()
    orig_debug = loop.get_debug()
    orig_threshold = loop.slow_callback_duration

    with pytest.raises(EventLoopStallError):
        async with EventLoopStallWatchdog(threshold_ms=50.0):
            assert loop.get_debug() is True
            assert loop.slow_callback_duration == pytest.approx(0.05)
            time.sleep(0.065)

    assert loop.get_debug() is orig_debug
    assert loop.slow_callback_duration == pytest.approx(orig_threshold)


async def test_watchdog_stop_is_idempotent_before_start_and_twice() -> None:
    """Calling stop() before start() or twice must not mutate asyncio logger settings."""
    asyncio_logger = logging.getLogger("asyncio")
    orig_level = asyncio_logger.level
    orig_disabled = asyncio_logger.disabled
    try:
        asyncio_logger.setLevel(logging.INFO)
        asyncio_logger.disabled = False

        wd = EventLoopStallWatchdog(threshold_ms=50.0)
        # stop() before start() is a no-op
        assert wd.stop() == []
        assert asyncio_logger.level == logging.INFO
        assert asyncio_logger.disabled is False

        wd.start()
        first = wd.stop()
        second = wd.stop()
        assert first == second == []
        assert asyncio_logger.level == logging.INFO
        assert asyncio_logger.disabled is False
    finally:
        asyncio_logger.setLevel(orig_level)
        asyncio_logger.disabled = orig_disabled


def test_watchdog_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValueError, match="positive"):
        EventLoopStallWatchdog(threshold_ms=0)


def test_cli_stall_watchdog_plugin_fails_bare_blocking_async_test(tmp_path) -> None:
    """Running pytest with --stall-watchdog fails an async test that calls time.sleep(0.07)."""
    repo_root = Path(__file__).resolve().parents[3]
    test_file = tmp_path / "test_synthetic_stall.py"
    test_file.write_text(
        "import time\n"
        "import pytest\n\n"
        "@pytest.mark.unit\n"
        "async def test_blocking_without_trailing_await():\n"
        "    time.sleep(0.07)\n",
        encoding="utf-8",
    )
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-c",
            str(repo_root / "pyproject.toml"),
            "-p",
            "conftest",
            "--stall-watchdog",
            str(test_file),
            "-q",
        ],
        capture_output=True,
        text=True,
        env=_subprocess_env(repo_root, {"ARCHIVIST_STALL_THRESHOLD_MS": ""}),
        cwd=str(repo_root),
        timeout=30,
    )
    assert proc.returncode != 0, (
        f"Expected stall failure, got exit 0:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "EventLoopStallError" in proc.stdout or "EventLoopStallError" in proc.stderr
