"""Unit tests for model-agnostic LLMClient, SpendTracker, and httpx adapters (SPEC-003)."""

from __future__ import annotations

import json

import httpx
import pytest

from archivist.crsi.config import BudgetConfig, CrsiConfig, ModelRoleConfig, ModelsConfig
from archivist.crsi.llm.base import LLMError, LLMRequest, Message, RoutedLLMClient
from archivist.crsi.llm.budget import BudgetExhaustedError, PaidApiNotEnabledError
from archivist.crsi.llm.fake import FakeScriptedAdapter
from archivist.crsi.llm.openai_compat import normalize_openai_base_url

pytestmark = [pytest.mark.crsi, pytest.mark.unit]


def _make_request(
    role: str = "policy", *, text: str = "hello", json_mode: bool = False
) -> LLMRequest:
    return LLMRequest(
        role=role,  # type: ignore[arg-type]
        messages=[
            Message(role="system", content="You are a test assistant."),
            Message(role="user", content=text),
        ],
        json_mode=json_mode,
    )


def test_normalize_openai_base_url():
    assert normalize_openai_base_url("https://api.openai.com") == "https://api.openai.com"
    assert normalize_openai_base_url("https://api.openai.com/") == "https://api.openai.com"
    assert normalize_openai_base_url("https://api.x.ai/v1") == "https://api.x.ai"
    assert normalize_openai_base_url("http://127.0.0.1:11434/v1/") == "http://127.0.0.1:11434"


async def test_fake_scripted_adapter_handles_all_three_roles_at_zero_cost():
    fake = FakeScriptedAdapter(
        responses={
            "policy": ["policy-step-1", "policy-step-2"],
            "selector": ["selector-pick"],
        },
        handlers={
            "proposer": lambda req, cfg: f"proposed-by-{cfg.model}:{req.messages[-1].content}"
        },
    )
    cfg = CrsiConfig()
    client = RoutedLLMClient(cfg, adapters={"fake": fake})

    r1 = await client.complete(_make_request("policy", text="p1"))
    r2 = await client.complete(_make_request("policy", text="p2"))
    r3 = await client.complete(_make_request("proposer", text="diff"))
    r4 = await client.complete(_make_request("selector", text="pick"))

    assert r1.content == "policy-step-1"
    assert r2.content == "policy-step-2"
    assert r3.content == "proposed-by-fake-proposer-v1:diff"
    assert r4.content == "selector-pick"
    assert client.spend_tracker.calls_used == 4
    assert client.spend_tracker.spent_usd == 0.0
    assert len(fake.call_log) == 4


@pytest.mark.parametrize(
    ("provider", "base_url", "env_name"),
    [
        ("openai_compat", "https://api.openai.com/v1", "OPENAI_API_KEY"),
        ("anthropic", "https://api.anthropic.com", "ANTHROPIC_API_KEY"),
        ("gemini", "https://generativelanguage.googleapis.com", "GEMINI_API_KEY"),
    ],
)
async def test_paid_provider_blocked_before_http_when_not_opted_in(
    provider: str,
    base_url: str,
    env_name: str,
):
    http_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal http_calls
        http_calls += 1
        return httpx.Response(200, json={})

    cfg = CrsiConfig(
        models=ModelsConfig(
            policy=ModelRoleConfig(
                provider=provider,  # type: ignore[arg-type]
                model="test-model",
                base_url=base_url,
                api_key_env=env_name,
            )
        ),
        budget=BudgetConfig(allow_paid_apis=False, max_spend_usd=0.0),
    )
    client = RoutedLLMClient(
        cfg,
        environ={env_name: "sk-test-never-sent-1234567890"},
        transport=httpx.MockTransport(handler),
    )

    with pytest.raises(PaidApiNotEnabledError):
        await client.complete(_make_request("policy"))

    assert http_calls == 0
    assert client.spend_tracker.calls_used == 0


async def test_openai_compatible_adapter_translation_and_budget_accounting():
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "model": "grok-3-mini",
                "choices": [{"message": {"role": "assistant", "content": '{"ok": true}'}}],
                "usage": {"prompt_tokens": 1000, "completion_tokens": 500},
            },
        )

    cfg = CrsiConfig(
        models=ModelsConfig(
            proposer=ModelRoleConfig(
                provider="openai_compat",
                model="grok-3-mini",
                base_url="https://api.x.ai/v1",
                api_key_env="XAI_API_KEY",
                max_tokens=1000,
                input_usd_per_1m=1.0,
                output_usd_per_1m=2.0,
            )
        ),
        budget=BudgetConfig(allow_paid_apis=True, max_spend_usd=1.0, max_llm_calls=5),
    )
    client = RoutedLLMClient(
        cfg,
        environ={"XAI_API_KEY": "xai-mock-secret-value-abcdef123456"},
        transport=httpx.MockTransport(handler),
    )

    resp = await client.complete(_make_request("proposer", json_mode=True))
    assert resp.content == '{"ok": true}'
    assert resp.provider == "openai_compat"
    assert resp.prompt_tokens == 1000
    assert resp.completion_tokens == 500
    # 1000 * $1/1M + 500 * $2/1M = $0.002
    assert resp.estimated_cost_usd == pytest.approx(0.002)
    assert client.spend_tracker.spent_usd == pytest.approx(0.002)

    assert len(captured) == 1
    req = captured[0]
    assert str(req.url) == "https://api.x.ai/v1/chat/completions"
    assert req.headers["authorization"] == "Bearer xai-mock-secret-value-abcdef123456"
    body = json.loads(req.content)
    assert body["response_format"] == {"type": "json_object"}


async def test_local_loopback_openai_compat_works_without_paid_opt_in():
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://127.0.0.1:11434/v1/chat/completions"
        assert "authorization" not in request.headers
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "local-ollama-reply"}}]},
        )

    cfg = CrsiConfig(
        models=ModelsConfig(
            policy=ModelRoleConfig(
                provider="openai_compat",
                model="qwen2.5-coder",
                base_url="http://127.0.0.1:11434/v1",
                api_key_env=None,
            )
        ),
        budget=BudgetConfig(allow_paid_apis=False, max_spend_usd=0.0),
    )
    client = RoutedLLMClient(cfg, transport=httpx.MockTransport(handler))
    resp = await client.complete(_make_request("policy"))
    assert resp.content == "local-ollama-reply"
    assert resp.estimated_cost_usd == 0.0


async def test_anthropic_adapter_translation_and_error_secret_redaction():
    secret = "sk-ant-mock-secret-9999888877776666"
    captured: list[httpx.Request] = []

    def ok_handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "model": "claude-sonnet-test",
                "content": [{"type": "text", "text": "anthropic-reply"}],
                "usage": {"input_tokens": 200, "output_tokens": 100},
            },
        )

    cfg = CrsiConfig(
        models=ModelsConfig(
            selector=ModelRoleConfig(
                provider="anthropic",
                model="claude-sonnet-test",
                api_key_env="ANTHROPIC_API_KEY",
                max_tokens=512,
                input_usd_per_1m=3.0,
                output_usd_per_1m=15.0,
            )
        ),
        budget=BudgetConfig(allow_paid_apis=True, max_spend_usd=1.0),
    )
    client = RoutedLLMClient(
        cfg,
        environ={"ANTHROPIC_API_KEY": secret},
        transport=httpx.MockTransport(ok_handler),
    )
    resp = await client.complete(_make_request("selector"))
    assert resp.content == "anthropic-reply"
    assert resp.provider == "anthropic"
    assert captured[0].headers["x-api-key"] == secret
    body = json.loads(captured[0].content)
    assert body["system"] == "You are a test assistant."
    assert body["messages"] == [{"role": "user", "content": "hello"}]

    # Error response echoing the secret must be scrubbed before raising LLMError.
    def err_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, text=f"invalid key: {secret}")

    err_client = RoutedLLMClient(
        cfg,
        environ={"ANTHROPIC_API_KEY": secret},
        transport=httpx.MockTransport(err_handler),
    )
    with pytest.raises(LLMError) as exc_info:
        await err_client.complete(_make_request("selector"))
    assert secret not in str(exc_info.value)
    assert "[REDACTED]" in str(exc_info.value)


async def test_gemini_adapter_translation_keeps_key_out_of_url():
    secret = "AIzaSyMockGeminiSecretKeyValue1234567890"
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(
            200,
            json={
                "candidates": [{"content": {"parts": [{"text": '{"picked": "c1"}'}]}}],
                "usageMetadata": {"promptTokenCount": 300, "candidatesTokenCount": 50},
            },
        )

    cfg = CrsiConfig(
        models=ModelsConfig(
            policy=ModelRoleConfig(
                provider="gemini",
                model="gemini-2.5-pro",
                api_key_env="GEMINI_API_KEY",
                max_tokens=512,
                input_usd_per_1m=1.25,
                output_usd_per_1m=5.0,
            )
        ),
        budget=BudgetConfig(allow_paid_apis=True, max_spend_usd=1.0),
    )
    client = RoutedLLMClient(
        cfg,
        environ={"GEMINI_API_KEY": secret},
        transport=httpx.MockTransport(handler),
    )
    resp = await client.complete(_make_request("policy", json_mode=True))
    assert resp.content == '{"picked": "c1"}'
    assert resp.provider == "gemini"
    assert secret not in str(captured[0].url)
    assert captured[0].headers["x-goog-api-key"] == secret
    body = json.loads(captured[0].content)
    assert body["generationConfig"]["responseMimeType"] == "application/json"
    assert body["systemInstruction"]["parts"][0]["text"] == "You are a test assistant."


async def test_spend_tracker_enforces_call_cap_spend_cap_and_snapshot_restore():
    cfg = CrsiConfig(
        models=ModelsConfig(
            policy=ModelRoleConfig(
                provider="openai_compat",
                model="expensive",
                base_url="https://api.openai.com",
                api_key_env="OPENAI_API_KEY",
                max_tokens=100_000,
                input_usd_per_1m=10.0,
                output_usd_per_1m=10.0,
            ),
            selector=ModelRoleConfig(provider="fake", model="fake-selector-v1"),
        ),
        budget=BudgetConfig(allow_paid_apis=True, max_spend_usd=0.05, max_llm_calls=2),
    )
    client = RoutedLLMClient(
        cfg,
        environ={"OPENAI_API_KEY": "sk-proj-1234567890abcdefghijklmnop"},
        transport=httpx.MockTransport(lambda req: httpx.Response(200, json={})),
    )

    # Projected cost (100k output tokens @ $10/1M = $1.00) exceeds $0.05 cap BEFORE HTTP.
    with pytest.raises(BudgetExhaustedError, match="max_spend_usd"):
        await client.complete(_make_request("policy"))

    # Fake calls still obey max_llm_calls = 2.
    await client.complete(_make_request("selector"))
    await client.complete(_make_request("selector"))
    with pytest.raises(BudgetExhaustedError, match="max_llm_calls"):
        await client.complete(_make_request("selector"))

    snap = client.spend_tracker.snapshot()
    client2 = RoutedLLMClient(cfg)
    client2.spend_tracker.restore_snapshot(snap)
    assert client2.spend_tracker.calls_used == 2
    with pytest.raises(BudgetExhaustedError):
        await client2.complete(_make_request("selector"))


async def test_maj3_concurrent_calls_cannot_overshoot_call_or_spend_budget():
    import asyncio

    adapter_entries = 0

    class _SlowPaidAdapter:
        async def complete(self, request: LLMRequest, role_config: ModelRoleConfig):
            nonlocal adapter_entries
            adapter_entries += 1
            await asyncio.sleep(0.05)
            from archivist.crsi.llm.base import LLMResponse

            return LLMResponse(
                content="ok",
                prompt_tokens=100,
                completion_tokens=100,
                estimated_cost_usd=0.04,
                model=role_config.model,
                provider=role_config.provider,
                latency_ms=50.0,
            )

    cfg = CrsiConfig(
        models=ModelsConfig(
            policy=ModelRoleConfig(
                provider="openai_compat",
                model="paid-test",
                base_url="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
                max_tokens=1000,
                input_usd_per_1m=10.0,
                output_usd_per_1m=40.0,
            )
        ),
        budget=BudgetConfig(
            allow_paid_apis=True,
            max_spend_usd=0.05,
            max_llm_calls=1,
            parallelism=4,
        ),
    )
    client = RoutedLLMClient(
        cfg,
        adapters={"openai_compat": _SlowPaidAdapter()},
        environ={"OPENAI_API_KEY": "sk-proj-1234567890abcdefghijklmnop"},
    )

    results = await asyncio.gather(
        *(client.complete(_make_request("policy")) for _ in range(4)),
        return_exceptions=True,
    )
    successes = [r for r in results if not isinstance(r, BaseException)]
    exhausted = [r for r in results if isinstance(r, BudgetExhaustedError)]

    assert len(successes) == 1
    assert len(exhausted) == 3
    assert adapter_entries == 1
    assert client.spend_tracker.calls_used == 1
    assert client.spend_tracker.spent_usd <= 0.05


async def test_maj3_failed_call_releases_budget_reservation():
    def fail_once_handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="server error")

    cfg = CrsiConfig(
        models=ModelsConfig(
            policy=ModelRoleConfig(
                provider="openai_compat",
                model="test-model",
                base_url="https://api.openai.com/v1",
                api_key_env="OPENAI_API_KEY",
                max_tokens=100,
                input_usd_per_1m=1.0,
                output_usd_per_1m=1.0,
            )
        ),
        budget=BudgetConfig(allow_paid_apis=True, max_spend_usd=0.05, max_llm_calls=1),
    )
    client = RoutedLLMClient(
        cfg,
        environ={"OPENAI_API_KEY": "sk-proj-1234567890abcdefghijklmnop"},
        transport=httpx.MockTransport(fail_once_handler),
    )
    with pytest.raises(LLMError):
        await client.complete(_make_request("policy"))

    # Reservation was released on error, so calls_used remains 0 and slot is reusable.
    assert client.spend_tracker.calls_used == 0
    assert client.spend_tracker.spent_usd == 0.0
