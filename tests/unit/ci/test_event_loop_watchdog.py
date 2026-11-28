"""Tests for the asyncio >50ms event-loop stall watchdog (INIT-025/SPEC-012)."""

from __future__ import annotations

import asyncio
import time

import pytest
from tests.fixtures.event_loop_watchdog import (
    EventLoopStallError,
    EventLoopStallWatchdog,
)

pytestmark = [pytest.mark.unit]


async def test_watchdog_allows_yielding_async_work(event_loop_watchdog) -> None:
    """Properly yielding awaits and offloaded thread work never trip the watchdog."""
    await asyncio.sleep(0.07)
    await asyncio.to_thread(time.sleep, 0.07)
    assert event_loop_watchdog.stalls == []


@pytest.mark.allow_event_loop_stall
async def test_watchdog_detects_blocking_sleep_over_50ms() -> None:
    """Deliberate synchronous time.sleep(0.07) on the event loop trips the >50ms watchdog."""
    with pytest.raises(EventLoopStallError) as exc_info:
        with EventLoopStallWatchdog(threshold_ms=50.0):
            # Yield once so the blocking step runs inside a monitored Task callback.
            await asyncio.sleep(0)
            time.sleep(0.07)
            await asyncio.sleep(0)

    err = exc_info.value
    assert len(err.stalls) >= 1
    assert any(s.duration_ms >= 50.0 for s in err.stalls)


@pytest.mark.allow_event_loop_stall
async def test_watchdog_restores_loop_debug_and_threshold() -> None:
    """Event-loop debug flag and slow_callback_duration are restored cleanly on exit."""
    loop = asyncio.get_running_loop()
    orig_debug = loop.get_debug()
    orig_threshold = loop.slow_callback_duration

    with pytest.raises(EventLoopStallError):
        with EventLoopStallWatchdog(threshold_ms=50.0):
            assert loop.get_debug() is True
            assert loop.slow_callback_duration == pytest.approx(0.05)
            await asyncio.sleep(0)
            time.sleep(0.065)
            await asyncio.sleep(0)

    assert loop.get_debug() is orig_debug
    assert loop.slow_callback_duration == pytest.approx(orig_threshold)


def test_watchdog_rejects_non_positive_threshold() -> None:
    with pytest.raises(ValueError, match="positive"):
        EventLoopStallWatchdog(threshold_ms=0)


def test_cli_stall_watchdog_plugin_fails_bare_blocking_async_test(tmp_path) -> None:
    """Running pytest with --stall-watchdog fails an async test that calls time.sleep(0.07)."""
    import os
    import subprocess
    import sys
    from pathlib import Path

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
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{repo_root}{os.pathsep}{repo_root / 'src'}"
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
        env=env,
        cwd=str(repo_root),
        timeout=30,
    )
    assert proc.returncode != 0, (
        f"Expected stall failure, got exit 0:\n{proc.stdout}\n{proc.stderr}"
    )
    assert "EventLoopStallError" in proc.stdout or "EventLoopStallError" in proc.stderr
