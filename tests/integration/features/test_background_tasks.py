"""Tests for background task tracking and exception logging in main.py and webhooks.py."""

import asyncio
import logging

import pytest

pytestmark = [pytest.mark.integration]

class TestLogTaskException:
    """Verify _log_task_exception callback in main.py."""

    def test_logs_on_exception(self, caplog):
        from main import _log_task_exception

        async def _boom():
            raise RuntimeError("simulated crash")

        loop = asyncio.new_event_loop()
        task = loop.create_task(_boom(), name="test_boom")
        try:
            loop.run_until_complete(task)
        except RuntimeError:
            pass

        with caplog.at_level(logging.ERROR, logger="archivist"):
            _log_task_exception(task)

        assert any("crashed" in r.message for r in caplog.records)
        assert any("test_boom" in r.message for r in caplog.records)
        loop.close()

    def test_silent_on_cancelled(self, caplog):
        from main import _log_task_exception

        async def _forever():
            await asyncio.sleep(3600)

        loop = asyncio.new_event_loop()
        task = loop.create_task(_forever(), name="test_cancel")
        task.cancel()
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass

        with caplog.at_level(logging.ERROR, logger="archivist"):
            _log_task_exception(task)

        assert not any("crashed" in r.message for r in caplog.records)
        loop.close()

    def test_silent_on_success(self, caplog):
        from main import _log_task_exception

        async def _ok():
            return 42

        loop = asyncio.new_event_loop()
        task = loop.create_task(_ok(), name="test_ok")
        loop.run_until_complete(task)

        with caplog.at_level(logging.ERROR, logger="archivist"):
            _log_task_exception(task)

        assert not any("crashed" in r.message for r in caplog.records)
        loop.close()

    def test_background_tasks_list_populated(self):
        from main import _background_tasks

        assert isinstance(_background_tasks, list)

class TestWebhookPendingFires:
    """Verify _pending_fires tracking and _log_fire_exception in webhooks.py."""

    def test_log_fire_exception_on_error(self, caplog):
        from webhooks import _log_fire_exception

        async def _boom():
            raise ValueError("webhook kaboom")

        loop = asyncio.new_event_loop()
        task = loop.create_task(_boom(), name="webhook_test")
        try:
            loop.run_until_complete(task)
        except ValueError:
            pass

        with caplog.at_level(logging.ERROR, logger="archivist.webhooks"):
            _log_fire_exception(task)

        assert any("failed" in r.message for r in caplog.records)
        assert any("webhook_test" in r.message for r in caplog.records)
        loop.close()

    def test_log_fire_exception_silent_on_cancel(self, caplog):
        from webhooks import _log_fire_exception

        async def _forever():
            await asyncio.sleep(3600)

        loop = asyncio.new_event_loop()
        task = loop.create_task(_forever(), name="webhook_cancel")
        task.cancel()
        try:
            loop.run_until_complete(task)
        except asyncio.CancelledError:
            pass

        with caplog.at_level(logging.ERROR, logger="archivist.webhooks"):
            _log_fire_exception(task)

        assert not any("failed" in r.message for r in caplog.records)
        loop.close()

    def test_pending_fires_is_set(self):
        from webhooks import _pending_fires

        assert isinstance(_pending_fires, set)

    async def test_fire_background_tracks_task(self, monkeypatch):
        from webhooks import _pending_fires, fire_background

        monkeypatch.setenv("WEBHOOK_URL", "http://localhost:9999/hook")
        import webhooks

        monkeypatch.setattr(webhooks, "WEBHOOK_URL", "http://localhost:9999/hook")
        monkeypatch.setattr(webhooks, "WEBHOOK_EVENTS", set())

        before = len(_pending_fires)
        fire_background("test_event", {"key": "value"})

        assert len(_pending_fires) >= before + 1

        for task in list(_pending_fires):
            task.cancel()
        await asyncio.sleep(0.1)

    async def test_fire_background_cleans_up_after_completion(self, monkeypatch):
        """Task should be removed from _pending_fires once it completes."""
        import webhooks
        from webhooks import _pending_fires


        async def _fake_fire(event, payload):
            return True

        monkeypatch.setattr(webhooks, "WEBHOOK_URL", "http://localhost:9999/hook")
        monkeypatch.setattr(webhooks, "WEBHOOK_EVENTS", set())
        monkeypatch.setattr(webhooks, "fire", _fake_fire)

        _pending_fires.clear()

        webhooks.fire_background("cleanup_test", {"ok": True})
        assert len(_pending_fires) == 1

        await asyncio.sleep(0.1)

        assert len(_pending_fires) == 0
