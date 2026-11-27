"""Pydantic v2 configuration and budget schema for CRSI + SIM.

Safe-by-default:
- ``allow_paid_apis = False`` and ``max_spend_usd = 0.0``
- Default provider is ``"fake"`` (zero network calls, deterministic CI)
- API keys may ONLY be configured via environment variable names (``api_key_env``),
  never literal secrets.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from archivist.crsi.guards.secrets import SecretRedactor, validate_env_var_name

ProviderName = Literal["fake", "openai_compat", "anthropic", "gemini"]
MemoryMode = Literal["full", "problem_only", "none"]

DEFAULT_EDITABLE_PATHS: tuple[str, ...] = (
    "src/archivist/crsi/harness/interlock.py",
    "src/archivist/crsi/harness/tools.py",
    "src/archivist/crsi/harness/context.py",
    "src/archivist/crsi/harness/runner.py",
    "src/archivist/crsi/harness/prompts/**",
)

_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


class ModelRoleConfig(BaseModel):
    """Per-role model configuration (policy, proposer, or selector)."""

    model_config = ConfigDict(extra="forbid")

    provider: ProviderName = "fake"
    model: str = "fake-policy-v1"
    base_url: str | None = None
    api_key_env: str | None = None
    max_tokens: int = Field(default=2048, ge=1, le=200_000)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    input_usd_per_1m: float = Field(default=0.0, ge=0.0)
    output_usd_per_1m: float = Field(default=0.0, ge=0.0)
    timeout_s: float = Field(default=60.0, gt=0.0)

    @field_validator("model")
    @classmethod
    def _validate_model_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model must be a non-empty string")
        return cleaned

    @field_validator("api_key_env")
    @classmethod
    def _validate_api_key_env(cls, value: str | None) -> str | None:
        if value is None:
            return None
        if not value.strip():
            return None
        return validate_env_var_name(value)

    def is_local_free_endpoint(self) -> bool:
        """Return True when calls do not reach a paid remote API."""
        if self.provider == "fake":
            return True
        if self.provider == "openai_compat" and self.base_url and not self.api_key_env:
            parsed = urlparse(self.base_url)
            host = (parsed.hostname or "").lower()
            return host in _LOOPBACK_HOSTS
        return False

    def is_paid_provider(self) -> bool:
        """Return True if this role requires paid-API opt-in before network I/O."""
        return not self.is_local_free_endpoint()

    def resolve_api_key(self, environ: Mapping[str, str] | None = None) -> str | None:
        """Resolve the runtime API key from ``environ`` by env-var name only."""
        if not self.api_key_env:
            return None
        source = os.environ if environ is None else environ
        return source.get(self.api_key_env)


class ModelsConfig(BaseModel):
    """Independent model assignments for Policy (M), Proposer (P), and Selector (S)."""

    model_config = ConfigDict(extra="forbid")

    policy: ModelRoleConfig = Field(
        default_factory=lambda: ModelRoleConfig(provider="fake", model="fake-policy-v1")
    )
    proposer: ModelRoleConfig = Field(
        default_factory=lambda: ModelRoleConfig(provider="fake", model="fake-proposer-v1")
    )
    selector: ModelRoleConfig = Field(
        default_factory=lambda: ModelRoleConfig(provider="fake", model="fake-selector-v1")
    )

    def for_role(self, role: Literal["policy", "proposer", "selector"]) -> ModelRoleConfig:
        """Return the ``ModelRoleConfig`` for ``role``."""
        if role == "policy":
            return self.policy
        if role == "proposer":
            return self.proposer
        if role == "selector":
            return self.selector
        raise ValueError(f"Unknown model role: {role!r}")

    def api_key_env_names(self) -> list[str]:
        """Return all configured env-var names across roles."""
        names: list[str] = []
        for cfg in (self.policy, self.proposer, self.selector):
            if cfg.api_key_env and cfg.api_key_env not in names:
                names.append(cfg.api_key_env)
        return names


class BudgetConfig(BaseModel):
    """Hard resource, wall-time, LLM-call, and USD spend caps."""

    model_config = ConfigDict(extra="forbid")

    rounds: int = Field(default=5, ge=1)
    candidates: int = Field(default=3, ge=1)
    parallelism: int = Field(default=2, ge=1)
    rollouts: int = Field(default=1, ge=1)
    task_timeout_s: float = Field(default=60.0, gt=0.0)
    agent_timeout_s: float = Field(default=120.0, gt=0.0)
    total_wall_timeout_s: float = Field(default=1800.0, gt=0.0)
    max_llm_calls: int = Field(default=200, ge=1)
    max_spend_usd: float = Field(default=0.0, ge=0.0)
    allow_paid_apis: bool = False
    regression_tolerance: float = Field(default=0.0, ge=0.0, le=1.0)


class HarnessPathsConfig(BaseModel):
    """Editable harness allowlist and optional project-specific frozen patterns."""

    model_config = ConfigDict(extra="forbid")

    editable: list[str] = Field(default_factory=lambda: list(DEFAULT_EDITABLE_PATHS))
    frozen: list[str] = Field(default_factory=list)


class CommandsConfig(BaseModel):
    """Optional frozen external commands for scoring and candidate checks."""

    model_config = ConfigDict(extra="forbid")

    run_task: list[str] = Field(default_factory=list)
    check: list[str] = Field(default_factory=list)


class PathsConfig(BaseModel):
    """Repository and private-directory path settings."""

    model_config = ConfigDict(extra="forbid")

    sim_path: str = ".crsi/sim.json"
    state_dir: str = ".crsi/runs"
    private_dir: str | None = None
    base_branch: str = "main"
    branch_prefix: str = "crsi/"

    @field_validator("branch_prefix")
    @classmethod
    def _validate_branch_prefix(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned or cleaned in {"main", "master"}:
            raise ValueError("branch_prefix must be a non-empty prefix distinct from main/master")
        return cleaned


class CrsiConfig(BaseModel):
    """Top-level configuration for a CRSI + SIM project."""

    model_config = ConfigDict(extra="forbid")

    project_name: str = "crsi-project"
    domain: str = "default"
    memory_mode: MemoryMode = "full"
    models: ModelsConfig = Field(default_factory=ModelsConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    harness: HarnessPathsConfig = Field(default_factory=HarnessPathsConfig)
    commands: CommandsConfig = Field(default_factory=CommandsConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @model_validator(mode="after")
    def _check_paid_budget_consistency(self) -> CrsiConfig:
        if self.budget.allow_paid_apis and self.budget.max_spend_usd <= 0.0:
            raise ValueError(
                "budget.max_spend_usd must be > 0.0 when budget.allow_paid_apis is True"
            )
        return self

    def build_redactor(self, environ: Mapping[str, str] | None = None) -> SecretRedactor:
        """Build a ``SecretRedactor`` aware of configured API key env vars."""
        return SecretRedactor.from_env(
            self.models.api_key_env_names(),
            environ=environ,
            include_common_secret_vars=True,
        )


def load_config(path: Path, *, redactor: SecretRedactor | None = None) -> CrsiConfig:
    """Load and validate a ``CrsiConfig`` from a TOML file.

    Rejects files containing inline secret values before parsing.
    """
    raw_text = path.read_text(encoding="utf-8")
    checker = redactor if redactor is not None else SecretRedactor()
    checker.assert_no_secrets(raw_text, context=str(path))
    data = tomllib.loads(raw_text)
    return CrsiConfig.model_validate(data)
