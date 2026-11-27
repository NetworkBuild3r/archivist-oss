"""Secret redaction, env-var-name validation, and child-process env sanitization.

Enforces the hard security rule: no API keys, tokens, or private keys may appear
in code, git, configs, SIM records, logs, traces, or LLM prompts.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterable, Mapping
from typing import Any

REDACTED_TOKEN = "[REDACTED]"  # noqa: S105

ENV_VAR_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]*$")

# Env var names whose values must be stripped before invoking sandboxed task commands.
SECRET_ENV_NAME_RE = re.compile(
    r"(?:^|_)(?:API_KEY|TOKEN|SECRET|PASSWORD|PASSWD|PRIVATE_KEY|CREDENTIAL|CREDENTIALS|AUTH)$"
    r"|^VAULT_|^AWS_SECRET|^AWS_SESSION_TOKEN|^OPENAI_API_KEY$|^ANTHROPIC_API_KEY$"
    r"|^GEMINI_API_KEY$|^GOOGLE_API_KEY$|^XAI_API_KEY$|^GH_TOKEN$|^GITHUB_TOKEN$",
    re.IGNORECASE,
)

# High-signal secret value patterns (prefix / structural detectors).
SECRET_VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("anthropic_key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{12,}\b")),
    ("openai_key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_\-]{16,}\b")),
    ("xai_key", re.compile(r"\bxai-[A-Za-z0-9_\-]{16,}\b")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b")),
    ("github_pat", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}\b")),
    ("github_fine_pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    (
        "private_key_pem",
        re.compile(
            r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"
        ),
    ),
    ("private_key_header", re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----")),
    ("bearer_token", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._\-]{16,}")),
    (
        "inline_secret_assignment",
        re.compile(
            r"(?i)(?:api_key|api-key|x-api-key|secret_key|access_token)\s*[:=]\s*['\"]?"
            r"(?!\[REDACTED\])([A-Za-z0-9_\-\.]{16,})['\"]?"
        ),
    ),
)

_LITERAL_KEY_PREFIXES: tuple[str, ...] = (
    "sk-",
    "sk-ant-",
    "xai-",
    "AIza",
    "ghp_",
    "gho_",
    "github_pat_",
    "AKIA",
    "Bearer ",
    "bearer ",
    "-----BEGIN",
)


class SecretLeakError(ValueError):
    """Raised when a literal secret is detected in a forbidden surface."""


def validate_env_var_name(name: str) -> str:
    """Validate that a configured secret reference is strictly an env-var name.

    Rejects inline API keys, tokens, whitespace, lowercase literals, or
    non-identifier values.
    """
    stripped = name.strip()
    if not stripped:
        raise ValueError("api_key_env must be a non-empty environment variable name or null")
    for prefix in _LITERAL_KEY_PREFIXES:
        if stripped.startswith(prefix):
            raise SecretLeakError(
                "Literal API key or token detected in api_key_env; provide only an environment "
                "variable name (e.g. 'OPENAI_API_KEY')."
            )
    if not ENV_VAR_NAME_RE.match(stripped):
        raise ValueError(
            f"Invalid environment variable name {stripped!r}: must match ^[A-Z][A-Z0-9_]*$"
        )
    return stripped


def sanitize_env(
    environ: Mapping[str, str] | None = None,
    *,
    extra_strip_names: Iterable[str] = (),
    extra_keep_names: Iterable[str] = (),
) -> dict[str, str]:
    """Return a copy of ``environ`` with secret-like variables removed."""
    source = os.environ if environ is None else environ
    strip_exact = {name.upper() for name in extra_strip_names}
    keep_exact = {name.upper() for name in extra_keep_names}
    cleaned: dict[str, str] = {}
    for key, value in source.items():
        upper_key = key.upper()
        if upper_key in strip_exact:
            continue
        if upper_key not in keep_exact and SECRET_ENV_NAME_RE.search(key):
            continue
        cleaned[key] = value
    return cleaned


class SecretRedactor:
    """Scrubs known secret values and common credential patterns from text and objects."""

    def __init__(
        self,
        secrets: Iterable[str] = (),
        *,
        min_literal_length: int = 4,
    ) -> None:
        self._min_literal_length = min_literal_length
        self._literals: set[str] = set()
        for secret in secrets:
            self.register_secret(secret)

    @classmethod
    def from_env(
        cls,
        env_var_names: Iterable[str] = (),
        *,
        environ: Mapping[str, str] | None = None,
        include_common_secret_vars: bool = True,
    ) -> SecretRedactor:
        """Build a redactor populated from current environment secret values."""
        source = os.environ if environ is None else environ
        values: list[str] = []
        for var_name in env_var_names:
            val = source.get(var_name)
            if val:
                values.append(val)
        if include_common_secret_vars:
            for key, val in source.items():
                if val and SECRET_ENV_NAME_RE.search(key):
                    values.append(val)
        return cls(secrets=values)

    def register_secret(self, value: str | None) -> None:
        """Register an additional secret literal to scrub on sight."""
        if not value:
            return
        if len(value) >= self._min_literal_length:
            self._literals.add(value)

    def find_secret_matches(self, text: str) -> list[str]:
        """Return labels of secret patterns or registered literals found in ``text``.

        Never returns raw secret values.
        """
        findings: list[str] = []
        for literal in sorted(self._literals, key=len, reverse=True):
            if literal in text:
                findings.append("registered_secret")
                break
        for label, pattern in SECRET_VALUE_PATTERNS:
            if pattern.search(text):
                findings.append(label)
        return findings

    def contains_secret(self, text: str) -> bool:
        """Return True if ``text`` appears to contain a registered secret or key pattern."""
        return bool(self.find_secret_matches(text))

    def assert_no_secrets(self, text: str, *, context: str = "input") -> None:
        """Raise ``SecretLeakError`` if ``text`` contains any secret."""
        matches = self.find_secret_matches(text)
        if matches:
            joined = ", ".join(matches)
            raise SecretLeakError(
                f"Detected potential secret material in {context} (matched: {joined}). "
                "Use environment variable names instead of inline credentials."
            )

    def redact_text(self, text: str) -> str:
        """Replace all registered secret literals and matched secret patterns."""
        result = text
        for literal in sorted(self._literals, key=len, reverse=True):
            if literal in result:
                result = result.replace(literal, REDACTED_TOKEN)
        for label, pattern in SECRET_VALUE_PATTERNS:
            if label == "bearer_token":
                result = pattern.sub(f"Bearer {REDACTED_TOKEN}", result)
            elif label == "inline_secret_assignment":
                result = pattern.sub(
                    lambda m: m.group(0).replace(m.group(1), REDACTED_TOKEN),
                    result,
                )
            else:
                result = pattern.sub(REDACTED_TOKEN, result)
        return result

    def redact_obj(self, obj: Any) -> Any:
        """Recursively redact strings inside dicts, lists, and tuples."""
        if isinstance(obj, str):
            return self.redact_text(obj)
        if isinstance(obj, dict):
            return {self.redact_obj(k): self.redact_obj(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self.redact_obj(item) for item in obj]
        if isinstance(obj, tuple):
            return tuple(self.redact_obj(item) for item in obj)
        return obj
