"""llm_query() call-site coverage for _openai_base normalization (ac-8, INIT-021/SPEC-001).

test_llm_base_url.py covers _openai_base() in isolation (ac-1..ac-7). This file covers the
wiring: both request-construction call sites in llm_query() (the primary POST and the
json_mode-unsupported retry-without-response_format POST) must post to the *normalized*
base URL, not the raw operator-configured value.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from archivist.features import llm

pytestmark = [pytest.mark.unit]


class _FakeResponse:
    """Minimal httpx.Response stand-in — only what llm_query() reads."""

    def __init__(self, payload: dict, status_code: int = 200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            request = httpx.Request("POST", "http://test")
            response = httpx.Response(self.status_code, request=request)
            raise httpx.HTTPStatusError("error", request=request, response=response)

    def json(self) -> dict:
        return self._payload


def _fake_client(post_side_effect) -> MagicMock:
    client = MagicMock()
    client.post = AsyncMock(side_effect=post_side_effect)
    return client


@pytest.mark.parametrize(
    "configured_url",
    [
        "http://host:11434/",
        "http://host:11435/v1",
        "http://host:11435/v1/",
    ],
)
async def test_primary_call_site_uses_normalized_url(monkeypatch, configured_url):
    """The primary POST (llm.py's first call site, ~line 112) uses the normalized base URL."""
    posted_urls: list[str] = []

    async def _post(url, json=None, headers=None):
        posted_urls.append(url)
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(llm, "_get_llm_client", lambda: _fake_client(_post))

    result = await llm.llm_query("hello", url=configured_url)

    assert result == "ok"
    expected_base = llm._openai_base(configured_url)
    assert posted_urls == [f"{expected_base}/v1/chat/completions"]


async def test_json_mode_retry_call_site_uses_normalized_url(monkeypatch):
    """The json_mode-unsupported retry POST (llm.py's second call site, ~line 129) also
    uses the normalized base URL, not the raw configured value."""
    posted_urls: list[str] = []
    call_count = 0

    async def _post(url, json=None, headers=None):
        nonlocal call_count
        posted_urls.append(url)
        call_count += 1
        if call_count == 1:
            # First call includes response_format; simulate a provider that rejects it.
            return _FakeResponse({"error": "response_format unsupported"}, status_code=400)
        return _FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr(llm, "_get_llm_client", lambda: _fake_client(_post))

    result = await llm.llm_query("hello", url="http://host:11435/v1/", json_mode=True)

    assert result == "ok"
    assert posted_urls == [
        "http://host:11435/v1/chat/completions",
        "http://host:11435/v1/chat/completions",
    ]
