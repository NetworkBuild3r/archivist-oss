"""Unit tests for CRSI config schema, secret redaction, and import isolation (SPEC-001)."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from pydantic import ValidationError

from archivist.crsi.config import CrsiConfig, ModelRoleConfig, load_config
from archivist.crsi.guards.secrets import (
    REDACTED_TOKEN,
    SecretLeakError,
    SecretRedactor,
    sanitize_env,
    validate_env_var_name,
)

pytestmark = [pytest.mark.crsi, pytest.mark.unit]

FORBIDDEN_IMPORT_PREFIXES = (
    "archivist.app",
    "archivist.core",
    "archivist.storage",
    "archivist.retrieval",
    "archivist.lifecycle",
    "archivist.write",
    "archivist.features",
    "archivist.utils",
    "qdrant_client",
)


def test_default_config_disables_paid_apis_and_uses_fake_models():
    cfg = CrsiConfig()
    assert cfg.budget.allow_paid_apis is False
    assert cfg.budget.max_spend_usd == 0.0
    assert cfg.models.policy.provider == "fake"
    assert cfg.models.proposer.provider == "fake"
    assert cfg.models.selector.provider == "fake"
    assert cfg.memory_mode == "full"
    assert cfg.models.policy.is_paid_provider() is False


def test_allow_paid_apis_requires_positive_spend_cap():
    with pytest.raises(ValidationError, match="max_spend_usd"):
        CrsiConfig.model_validate(
            {
                "budget": {
                    "allow_paid_apis": True,
                    "max_spend_usd": 0.0,
                }
            }
        )

    cfg = CrsiConfig.model_validate(
        {
            "budget": {
                "allow_paid_apis": True,
                "max_spend_usd": 5.0,
            }
        }
    )
    assert cfg.budget.allow_paid_apis is True
    assert cfg.budget.max_spend_usd == 5.0


def test_config_rejects_inline_api_key_field():
    with pytest.raises(ValidationError):
        ModelRoleConfig.model_validate(
            {
                "provider": "openai_compat",
                "model": "gpt-4o-mini",
                "api_key": "sk-live-secret-12345678901234567890",
            }
        )


@pytest.mark.parametrize(
    "bad_env",
    [
        "sk-1234567890abcdef1234567890",
        "sk-ant-api03-1234567890abcdef",
        "AIzaSyDummyKeyValue12345678901234",
        "xai-1234567890abcdef1234567890",
        "ghp_1234567890abcdef1234567890",
        "lowercase_var",
        "123_INVALID",
        "HAS SPACE",
    ],
)
def test_api_key_env_rejects_literal_secrets_and_invalid_names(bad_env: str):
    with pytest.raises((ValidationError, ValueError, SecretLeakError)):
        ModelRoleConfig(provider="openai_compat", model="test", api_key_env=bad_env)


def test_api_key_env_accepts_valid_env_name_and_resolves_at_runtime():
    assert validate_env_var_name("OPENAI_API_KEY") == "OPENAI_API_KEY"
    role = ModelRoleConfig(
        provider="openai_compat",
        model="gpt-4o-mini",
        api_key_env="OPENAI_API_KEY",
    )
    assert role.api_key_env == "OPENAI_API_KEY"
    assert role.is_paid_provider() is True
    assert role.resolve_api_key({"OPENAI_API_KEY": "runtime-only-val"}) == "runtime-only-val"
    assert role.resolve_api_key({}) is None


def test_local_ollama_loopback_without_key_is_free():
    role = ModelRoleConfig(
        provider="openai_compat",
        model="llama3.2",
        base_url="http://127.0.0.1:11434/v1",
        api_key_env=None,
    )
    assert role.is_local_free_endpoint() is True
    assert role.is_paid_provider() is False

    remote = ModelRoleConfig(
        provider="openai_compat",
        model="grok-3",
        base_url="https://api.x.ai/v1",
        api_key_env="XAI_API_KEY",
    )
    assert remote.is_local_free_endpoint() is False
    assert remote.is_paid_provider() is True


def test_load_config_rejects_toml_containing_inline_secret(tmp_path: Path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
project_name = "demo"
# accidental secret in comment: sk-proj-1234567890abcdefghij1234
[budget]
allow_paid_apis = false
max_spend_usd = 0.0
""",
        encoding="utf-8",
    )
    with pytest.raises(SecretLeakError):
        load_config(cfg_file)


def test_load_config_parses_clean_toml(tmp_path: Path):
    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        """
project_name = "toy-crsi"
domain = "toy_base"
memory_mode = "full"

[budget]
rounds = 3
candidates = 2
parallelism = 1
allow_paid_apis = false
max_spend_usd = 0.0
regression_tolerance = 0.05

[models.policy]
provider = "fake"
model = "fake-policy-v1"
""",
        encoding="utf-8",
    )
    cfg = load_config(cfg_file)
    assert cfg.project_name == "toy-crsi"
    assert cfg.budget.rounds == 3
    assert cfg.budget.regression_tolerance == 0.05


def test_secret_redactor_masks_registered_values_and_patterns():
    redactor = SecretRedactor.from_env(
        ["CUSTOM_PROVIDER_KEY"],
        environ={
            "CUSTOM_PROVIDER_KEY": "super-secret-custom-token-999",
            "OPENAI_API_KEY": "env-openai-secret-value-xyz",
            "PATH": "/usr/bin",
        },
    )
    raw = (
        "key=super-secret-custom-token-999 "
        "openai=env-openai-secret-value-xyz "
        "inline=sk-ant-1234567890abcdef1234 "
        "google=AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234 "
        "auth=Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig"
    )
    scrubbed = redactor.redact_text(raw)
    assert "super-secret-custom-token-999" not in scrubbed
    assert "env-openai-secret-value-xyz" not in scrubbed
    assert "sk-ant-1234567890abcdef1234" not in scrubbed
    assert "AIzaSyAbCdEfGhIjKlMnOpQrStUvWxYz1234" not in scrubbed
    assert "eyJhbGciOiJIUzI1NiJ9.payload.sig" not in scrubbed
    assert REDACTED_TOKEN in scrubbed

    nested = redactor.redact_obj(
        {"a": ["super-secret-custom-token-999", ("ok", "sk-1234567890abcdefghij")]}
    )
    assert nested == {"a": [REDACTED_TOKEN, ("ok", REDACTED_TOKEN)]}


def test_sanitize_env_strips_secret_vars():
    cleaned = sanitize_env(
        {
            "PATH": "/usr/bin",
            "HOME": "/tmp/sandbox",
            "OPENAI_API_KEY": "nope",
            "ANTHROPIC_API_KEY": "nope",
            "VAULT_TOKEN": "nope",
            "CUSTOM_SECRET": "nope",
            "MY_DB_PASSWORD": "nope",
            "SAFE_FLAG": "1",
        }
    )
    assert cleaned == {"PATH": "/usr/bin", "HOME": "/tmp/sandbox", "SAFE_FLAG": "1"}


def test_zero_imports_from_core_archivist_or_qdrant():
    """Enforce ADR-006 / GR-LAYER-001 decoupling across src/archivist/crsi/."""
    crsi_root = Path(__file__).resolve().parents[3] / "src" / "archivist" / "crsi"
    assert crsi_root.is_dir()
    violations: list[str] = []
    for py_file in sorted(crsi_root.rglob("*.py")):
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(FORBIDDEN_IMPORT_PREFIXES):
                        violations.append(f"{py_file}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith(FORBIDDEN_IMPORT_PREFIXES):
                    violations.append(f"{py_file}: from {node.module} import ...")
    assert violations == []
