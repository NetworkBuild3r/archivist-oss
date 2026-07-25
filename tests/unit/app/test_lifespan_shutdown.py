"""Regression test for INIT-022/SPEC-007's M6 fix.

``lifespan``'s shutdown block used to close the DB pool but never the
module-level embeddings/LLM ``httpx.AsyncClient`` singletons, leaking pooled
sockets on graceful shutdown. It now calls ``aclose_embed_client()`` and
``aclose_llm_client()`` alongside the existing pool close.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest

pytestmark = [pytest.mark.unit]


async def test_lifespan_shutdown_closes_embeddings_and_llm_clients(monkeypatch):
    import archivist.app.main as main

    class FakeManager:
        @asynccontextmanager
        async def run(self):
            yield

    async def fake_startup():
        return None

    fake_pool = AsyncMock()
    mock_aclose_embed = AsyncMock()
    mock_aclose_llm = AsyncMock()

    monkeypatch.setattr(main, "_create_streamable_http_session_manager", lambda: FakeManager())
    monkeypatch.setattr(main, "_startup", fake_startup)
    monkeypatch.setattr(main, "_background_tasks", [])
    monkeypatch.setattr("archivist.storage.sqlite_pool.pool", fake_pool)
    monkeypatch.setattr("archivist.features.embeddings.aclose_embed_client", mock_aclose_embed)
    monkeypatch.setattr("archivist.features.llm.aclose_llm_client", mock_aclose_llm)

    async with main.lifespan(main.app):
        pass

    fake_pool.close.assert_awaited_once()
    mock_aclose_embed.assert_awaited_once()
    mock_aclose_llm.assert_awaited_once()


async def test_aclose_embed_client_is_idempotent_when_never_created(monkeypatch):
    """M6: closing when the client was never lazily created must not raise."""
    import archivist.features.embeddings as embeddings_module

    monkeypatch.setattr(embeddings_module, "_embed_client", None)

    await embeddings_module.aclose_embed_client()  # must not raise

    assert embeddings_module._embed_client is None


async def test_aclose_llm_client_is_idempotent_when_never_created(monkeypatch):
    """M6: closing when the client was never lazily created must not raise."""
    import archivist.features.llm as llm_module

    monkeypatch.setattr(llm_module, "_llm_client", None)

    await llm_module.aclose_llm_client()  # must not raise

    assert llm_module._llm_client is None
