"""Linear-time ``google-re2`` AI Memory Firewall (INIT-024/SPEC-010).

Runs synchronously (<5ms p95, 0 LLM calls) on every write path BEFORE SQLite/
Postgres persistence, FTS5 indexing, embedding, or async consolidation:

1. **Secret & PII redaction** (65+ high-precision linear-time RE2 patterns) with
   explicit coding-corpus exemptions (40-char git SHAs, UUIDs, ``sha256-``
   integrity hashes, needle-registry tokens). Default policy: ``redact``;
   configurable ``block`` or ``vault_tokenize``.
2. **Indirect prompt-injection & steganography quarantine** (Unicode bidi
   overrides, Tag-block steganography, zero-width bursts, and high-confidence
   instruction-override directives) routing flagged writes to ``T3_QUARANTINED``
   (excluded from default retrieval, inspectable via ``memory_inspect``).
3. **Read-time structural framing** wrapping returned memories in
   ``<archivist_memory ...>...</archivist_memory>`` with closing-tag escaping.
"""

from __future__ import annotations

import hashlib
import html
import math
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import re2  # type: ignore[import-untyped]

from archivist.security.trust_tiers import TrustTier

# Hard per-chunk scan cap (64 KiB UTF-8) to bound worst-case CPU latency.
MAX_FIREWALL_INPUT_BYTES = 65_536


class FirewallPolicy(str, Enum):
    """Action taken when a secret or PII pattern is detected."""

    REDACT = "redact"
    BLOCK = "block"
    VAULT_TOKENIZE = "vault_tokenize"


class FirewallBlockedError(PermissionError):
    """Raised when ``FirewallPolicy.BLOCK`` rejects a write containing secrets."""

    def __init__(self, findings: Sequence[FirewallFinding]) -> None:
        self.findings = tuple(findings)
        names = ", ".join(sorted({f.rule for f in findings}))
        super().__init__(f"Memory Firewall blocked write containing sensitive data: {names}")


@dataclass(frozen=True)
class FirewallFinding:
    """Single detection emitted by the Memory Firewall."""

    category: str  # "secret" | "pii" | "injection" | "steganography"
    rule: str
    start: int
    end: int
    action: str  # "redacted" | "blocked" | "tokenized" | "quarantined"


@dataclass(frozen=True)
class FirewallResult:
    """Outcome of running :class:`MemoryFirewall` on a candidate write."""

    text: str
    trust_tier: TrustTier
    allowed: bool
    quarantined: bool
    redacted_count: int
    findings: tuple[FirewallFinding, ...] = field(default_factory=tuple)
    quarantine_reasons: tuple[str, ...] = field(default_factory=tuple)
    truncated_for_scan: bool = False
    elapsed_ms: float = 0.0


# ---------------------------------------------------------------------------
# 65+ high-precision linear-time RE2 secret & PII patterns (no backtracking,
# no lookaround assertions — 100% RE2 native).
# ---------------------------------------------------------------------------

# Tuple of (rule_name, category, re2_pattern, extra_validator)
# extra_validator: None | "luhn" | "entropy" | "assignment_group1"
_RAW_SECRET_RULES: tuple[tuple[str, str, str, str | None], ...] = (
    # ── Cloud & Infrastructure (17) ──────────────────────────────────────────
    ("aws_access_key_id", "secret", r"\bAKIA[0-9A-Z]{16}\b", None),
    ("aws_temp_access_key", "secret", r"\bASIA[0-9A-Z]{16}\b", None),
    ("aws_iam_unique_id", "secret", r"\b(?:AIDA|AROA|ANPA|ANVA|AIPA)[0-9A-Z]{16}\b", None),
    (
        "aws_mws_auth_token",
        "secret",
        r"\bamzn\.mws\.[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b",
        None,
    ),
    (
        "aws_secret_access_key",
        "secret",
        r"(?i)\baws_secret_access_key\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40})['\"]?",
        "assignment_group1",
    ),
    (
        "aws_session_token",
        "secret",
        r"(?i)\baws_session_token\s*[:=]\s*['\"]?([A-Za-z0-9/+=]{40,})['\"]?",
        "assignment_group1",
    ),
    ("gcp_api_key", "secret", r"\bAIza[0-9A-Za-z_\-]{30,45}\b", None),
    ("gcp_oauth_client_secret", "secret", r"\bGOCSPX-[0-9A-Za-z_\-]{20,}\b", None),
    (
        "gcp_service_account_key",
        "secret",
        r'"private_key"\s*:\s*"-----BEGIN [A-Z0-9 ]*PRIVATE ' + r"KEY-----",
        None,
    ),
    (
        "azure_storage_conn_key",
        "secret",
        r"(?i)\bAccountKey=([A-Za-z0-9+/]{40,}={0,2})",
        "assignment_group1",
    ),
    ("azure_sas_signature", "secret", r"[?&]sig=([A-Za-z0-9%+/]{32,})", "assignment_group1"),
    (
        "azure_client_secret",
        "secret",
        r"(?i)\bazure_client_secret\s*[:=]\s*['\"]?([A-Za-z0-9_.~\-]{24,})['\"]?",
        "assignment_group1",
    ),
    ("digitalocean_pat", "secret", r"\bdop_v1_[0-9a-f]{64}\b", None),
    ("digitalocean_oauth", "secret", r"\bdoo_v1_[0-9a-f]{64}\b", None),
    ("digitalocean_refresh", "secret", r"\bdor_v1_[0-9a-f]{64}\b", None),
    (
        "cloudflare_api_token",
        "secret",
        r"(?i)\bcloudflare_api_(?:token|key)\s*[:=]\s*['\"]?([A-Za-z0-9_\-]{32,})['\"]?",
        "assignment_group1",
    ),
    ("terraform_cloud_token", "secret", r"\b[A-Za-z0-9]{14}\.atlasv1\.[A-Za-z0-9_\-]{60,}\b", None),
    # ── AI / LLM Providers (11) ──────────────────────────────────────────────
    ("anthropic_api_key", "secret", r"\bsk-ant-(?:api03|admin01)-[A-Za-z0-9_\-]{20,}\b", None),
    ("anthropic_api_key_generic", "secret", r"\bsk-ant-[A-Za-z0-9_\-]{16,}\b", None),
    ("openai_proj_key", "secret", r"\bsk-proj-[A-Za-z0-9_\-]{20,}\b", None),
    ("openai_svc_key", "secret", r"\bsk-svcacct-[A-Za-z0-9_\-]{20,}\b", None),
    ("openai_api_key", "secret", r"\bsk-[A-Za-z0-9]{20,}\b", None),
    ("xai_api_key", "secret", r"\bxai-[A-Za-z0-9_\-]{20,}\b", None),
    ("huggingface_token", "secret", r"\bhf_[A-Za-z0-9]{24,}\b", None),
    ("replicate_api_token", "secret", r"\br8_[A-Za-z0-9]{24,}\b", None),
    ("groq_api_key", "secret", r"\bgsk_[A-Za-z0-9]{24,}\b", None),
    ("perplexity_api_key", "secret", r"\bpplx-[A-Za-z0-9]{24,}\b", None),
    ("openrouter_api_key", "secret", r"\bsk-or-v1-[0-9a-f]{48,}\b", None),
    # ── VCS, Package Registries & CI/CD (12) ─────────────────────────────────
    ("github_pat_classic", "secret", r"\bghp_[A-Za-z0-9]{36}\b", None),
    ("github_oauth_token", "secret", r"\bgho_[A-Za-z0-9]{36}\b", None),
    ("github_user_to_server", "secret", r"\bghu_[A-Za-z0-9]{36}\b", None),
    ("github_server_to_server", "secret", r"\bghs_[A-Za-z0-9]{36}\b", None),
    ("github_refresh_token", "secret", r"\bghr_[A-Za-z0-9]{36,}\b", None),
    ("github_fine_grained_pat", "secret", r"\bgithub_pat_[A-Za-z0-9_]{22,}\b", None),
    ("gitlab_pat", "secret", r"\bglpat-[A-Za-z0-9_\-]{20,}\b", None),
    ("gitlab_pipeline_token", "secret", r"\bglptt-[A-Za-z0-9_\-]{20,}\b", None),
    ("gitlab_runner_token", "secret", r"\bglrt-[A-Za-z0-9_\-]{20,}\b", None),
    ("npm_access_token", "secret", r"\bnpm_[A-Za-z0-9]{36}\b", None),
    ("pypi_api_token", "secret", r"\bpypi-AgEI[A-Za-z0-9_\-]{30,}\b", None),
    ("docker_pat", "secret", r"\bdckr_pat_[A-Za-z0-9_\-]{24,}\b", None),
    # ── Payments, Messaging & SaaS (18) ──────────────────────────────────────
    ("stripe_live_secret", "secret", r"\bsk_live_[0-9A-Za-z]{24,}\b", None),
    ("stripe_test_secret", "secret", r"\bsk_test_[0-9A-Za-z]{24,}\b", None),
    ("stripe_restricted_live", "secret", r"\brk_live_[0-9A-Za-z]{24,}\b", None),
    ("stripe_webhook_secret", "secret", r"\bwhsec_[0-9A-Za-z]{24,}\b", None),
    ("slack_bot_token", "secret", r"\bxoxb-[0-9]{8,}-[0-9]{8,}-[A-Za-z0-9]{20,}\b", None),
    ("slack_user_token", "secret", r"\bxoxp-[0-9]{8,}-[0-9]{8,}-[A-Za-z0-9_\-]{20,}\b", None),
    ("slack_app_token", "secret", r"\bxapp-[0-9]-[A-Za-z0-9]+-[0-9]+-[A-Za-z0-9]{20,}\b", None),
    (
        "slack_webhook_url",
        "secret",
        r"https://hooks\.slack\.com/services/T[A-Z0-9]+/B[A-Z0-9]+/[A-Za-z0-9]{16,}",
        None,
    ),
    (
        "discord_webhook_url",
        "secret",
        r"https://(?:canary\.|ptb\.)?discord(?:app)?\.com/api/webhooks/[0-9]{16,}/[A-Za-z0-9_\-]{40,}",
        None,
    ),
    ("twilio_api_key", "secret", r"\bSK[0-9a-f]{32}\b", None),
    ("sendgrid_api_key", "secret", r"\bSG\.[A-Za-z0-9_\-]{22}\.[A-Za-z0-9_\-]{43}\b", None),
    ("mailgun_api_key", "secret", r"\bkey-[0-9a-f]{32}\b", None),
    ("shopify_access_token", "secret", r"\bshpat_[0-9a-f]{32}\b", None),
    ("shopify_shared_secret", "secret", r"\bshpss_[0-9a-f]{32}\b", None),
    ("linear_api_key", "secret", r"\blin_api_[A-Za-z0-9]{32,}\b", None),
    ("sentry_auth_token", "secret", r"\bsntrys_[A-Za-z0-9_\-]{32,}\b", None),
    ("supabase_access_token", "secret", r"\bsbp_[0-9a-f]{40}\b", None),
    ("postman_api_key", "secret", r"\bPMAK-[0-9a-f]{24}-[0-9a-f]{34}\b", None),
    (
        "datadog_api_key",
        "secret",
        r"(?i)\b(?:dd_api_key|datadog_api_key)\s*[:=]\s*['\"]?([0-9a-f]{32})['\"]?",
        "assignment_group1",
    ),
    ("telegram_bot_token", "secret", r"\b[0-9]{8,10}:AA[A-Za-z0-9_\-]{33}\b", None),
    # ── Crypto, Auth, Vault & Database DSNs (13) ─────────────────────────────
    (
        "pem_private_key_block",
        "secret",
        r"-----BEGIN [A-Z0-9 ]*PRIVATE " + r"KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----",
        None,
    ),
    (
        "pem_private_key_header",
        "secret",
        r"-----BEGIN [A-Z0-9 ]*PRIVATE " + r"KEY-----",
        None,
    ),
    (
        "pgp_private_key_block",
        "secret",
        r"-----BEGIN PGP PRIVATE " + r"KEY BLOCK-----[\s\S]*?-----END PGP PRIVATE KEY BLOCK-----",
        None,
    ),
    (
        "jwt_token",
        "secret",
        r"\beyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]{10,}\b",
        None,
    ),
    ("bearer_token", "secret", r"(?i)\bBearer\s+[A-Za-z0-9._\-~+/=]{20,}", None),
    ("basic_auth_header", "secret", r"(?i)\bAuthorization\s*:\s*Basic\s+[A-Za-z0-9+/=]{12,}", None),
    ("postgres_dsn_creds", "secret", r"\bpostgres(?:ql)?://[^/\s:@]+:[^/\s@]+@[^\s'\"]+", None),
    ("mysql_dsn_creds", "secret", r"\bmysql(?:\+[a-z0-9]+)?://[^/\s:@]+:[^/\s@]+@[^\s'\"]+", None),
    ("mongodb_dsn_creds", "secret", r"\bmongodb(?:\+srv)?://[^/\s:@]+:[^/\s@]+@[^\s'\"]+", None),
    ("redis_dsn_creds", "secret", r"\brediss?://[^/\s:@]*:[^/\s@]+@[^\s'\"]+", None),
    ("amqp_dsn_creds", "secret", r"\bamqps?://[^/\s:@]+:[^/\s@]+@[^\s'\"]+", None),
    ("vault_service_token", "secret", r"\bhvs\.[A-Za-z0-9_\-]{20,}\b", None),
    (
        "high_entropy_secret_assignment",
        "secret",
        r"(?i)\b(?:api_key|api_secret|secret_key|access_token|auth_token|private_key|client_secret|db_password|database_password|master_key|signing_secret|webhook_secret|password)\s*[:=]\s*['\"]?([A-Za-z0-9_\-./+=]{16,512})['\"]?",
        "entropy",
    ),
    # ── PII (2) ──────────────────────────────────────────────────────────────
    (
        "us_ssn",
        "pii",
        r"\b(?:00[1-9]|0[1-9][0-9]|[1-5][0-9]{2}|6[0-5][0-9]|66[0-5]|66[7-9]|6[7-9][0-9]|[7-8][0-9]{2})-(?:0[1-9]|[1-9][0-9])-(?:000[1-9]|00[1-9][0-9]|0[1-9][0-9]{2}|[1-9][0-9]{3})\b",
        None,
    ),
    (
        "credit_card_number",
        "pii",
        r"\b(?:4[0-9]{12}(?:[0-9]{3})?|5[1-5][0-9]{14}|3[47][0-9]{13}|6(?:011|5[0-9]{2})[0-9]{12})\b",
        "luhn",
    ),
)

SECRET_PATTERN_NAMES: tuple[str, ...] = tuple(r[0] for r in _RAW_SECRET_RULES)

# False-positive exemptions for common coding artifacts.
_EXEMPT_PATTERNS: tuple[Any, ...] = (
    # 40-char lowercase/uppercase git commit SHA (exact token)
    re2.compile(r"^[0-9a-fA-F]{40}$"),
    # UUIDv4 / UUIDv7
    re2.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"),
    # Subresource / artifact integrity hashes (sha256-... / sha512-... / sha256:hex)
    re2.compile(r"^(?:sha256|sha384|sha512)[-:][A-Za-z0-9+/=]{32,}$"),
    # Needle-registry tokens & Archivist URIs
    re2.compile(r"^(?:NEEDLE-[A-Za-z0-9_\-]+|archivist://\S+)$"),
    # Already redacted / tokenized
    re2.compile(r"^(?:\[REDACTED(?::[a-z0-9_]+)?\]|vault://memory-secrets/[0-9a-f]+)$"),
)

# High-confidence indirect prompt-injection patterns (linear-time RE2).
_INJECTION_RULES: tuple[tuple[str, Any], ...] = (
    (
        "prompt_override_ignore_previous",
        re2.compile(
            r"(?i)\b(?:ignore|disregard|forget)\s+(?:all\s+)?(?:previous|prior|above|earlier)\s+"
            r"(?:instructions?|prompts?|rules?|directives?|context)\b"
        ),
    ),
    (
        "prompt_override_system_role",
        re2.compile(
            r"(?i)\b(?:you\s+are\s+now\s+(?:in\s+)?(?:developer|jailbreak|unrestricted|dan)\s+mode|"
            r"override\s+(?:the\s+)?system\s+prompt|"
            r"important\s+new\s+system\s+instruction)\b"
        ),
    ),
    (
        "prompt_delimiter_breakout",
        re2.compile(
            r"(?i)(?:</\s*archivist_memory\s*>|<\|im_start\|>\s*system|<\|system\|>|\[SYSTEM\s+OVERRIDE\])"
        ),
    ),
    (
        "prompt_secret_exfiltration",
        re2.compile(
            r"(?i)\b(?:exfiltrate|leak|send|upload|post)\s+(?:all\s+)?(?:secrets?|api[\s_-]?keys?|"
            r"tokens?|credentials?|env(?:ironment)?\s+var(?:iable)?s?|ssh\s+keys?)\s+"
            r"(?:to|via)\s+(?:https?://|curl|webhook|external)\b"
        ),
    ),
    (
        "minja_memory_poisoning_directive",
        re2.compile(
            r"(?i)\b(?:always\s+execute|silently\s+run|without\s+telling\s+the\s+user)\s+"
            r"(?:curl|wget|bash|sh|rm\s+-rf|chmod|nc\b)"
        ),
    ),
)

# Unicode bidi overrides and tag-block / zero-width steganography (linear-time RE2, C++ speed).
_BIDI_RE = re2.compile("[\u061c\u200e\u200f\u202a-\u202e\u2066-\u2069]")
_TAG_BLOCK_RE = re2.compile("[\U000e0000-\U000e007f]")
_ZERO_WIDTH_RE = re2.compile("[\u200b-\u200d\u2060\ufeff]")
_ZERO_WIDTH_BURST_THRESHOLD = 4

# Structural framing escape detector (linear-time RE2).
_MEMORY_TAG_ESCAPE_RE = re2.compile(r"(?i)(</?\s*archivist_memory\b)")


@dataclass(frozen=True)
class _CompiledRule:
    name: str
    category: str
    regex: Any
    validator: str | None


def _shannon_entropy(value: str) -> float:
    if not value:
        return 0.0
    counts: dict[str, int] = {}
    for ch in value:
        counts[ch] = counts.get(ch, 0) + 1
    length = len(value)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def _luhn_valid(digits: str) -> bool:
    if not digits.isdigit() or len(digits) < 13:
        return False
    total = 0
    reverse = digits[::-1]
    for i, ch in enumerate(reverse):
        d = ord(ch) - 48
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _is_exempt_token(token: str) -> bool:
    stripped = token.strip().strip("'\"")
    if not stripped:
        return True
    return any(pat.match(stripped) is not None for pat in _EXEMPT_PATTERNS)


class MemoryFirewall:
    """Linear-time ``google-re2`` Memory Firewall for Archivist write & read paths."""

    def __init__(
        self,
        *,
        policy: FirewallPolicy | str = FirewallPolicy.REDACT,
        max_input_bytes: int = MAX_FIREWALL_INPUT_BYTES,
    ) -> None:
        self.policy = policy if isinstance(policy, FirewallPolicy) else FirewallPolicy(str(policy))
        self.max_input_bytes = max(1024, int(max_input_bytes))
        self._rules: tuple[_CompiledRule, ...] = tuple(
            _CompiledRule(
                name=name,
                category=category,
                regex=re2.compile(pattern),
                validator=validator,
            )
            for name, category, pattern, validator in _RAW_SECRET_RULES
        )
        # Single RE2 multi-pattern pre-screen set so clean writes take 1 linear pass (<0.3ms).
        self._secret_set = re2.Set.SearchSet()
        for _, _, pattern, _ in _RAW_SECRET_RULES:
            self._secret_set.Add(pattern)
        self._secret_set.Compile()

        self._injection_set = re2.Set.SearchSet()
        for _, compiled_re in _INJECTION_RULES:
            self._injection_set.Add(compiled_re.pattern)
        self._injection_set.Compile()

    def _cap_input(self, text: str) -> tuple[str, bool]:
        raw = text.encode("utf-8")
        if len(raw) <= self.max_input_bytes:
            return text, False
        capped = raw[: self.max_input_bytes].decode("utf-8", errors="ignore")
        return capped, True

    def _detect_steganography(self, text: str) -> list[FirewallFinding]:
        # Fast-path: pure ASCII strings cannot contain Unicode bidi/tag/zero-width chars.
        if text.isascii():
            return []
        findings: list[FirewallFinding] = []
        bidi_match = _BIDI_RE.search(text)
        if bidi_match is not None:
            findings.append(
                FirewallFinding(
                    category="steganography",
                    rule="unicode_bidi_override",
                    start=bidi_match.start(),
                    end=bidi_match.end(),
                    action="quarantined",
                )
            )
        tag_match = _TAG_BLOCK_RE.search(text)
        if tag_match is not None:
            findings.append(
                FirewallFinding(
                    category="steganography",
                    rule="unicode_tag_block_steganography",
                    start=tag_match.start(),
                    end=tag_match.end(),
                    action="quarantined",
                )
            )
        zw_matches = _ZERO_WIDTH_RE.findall(text)
        if len(zw_matches) >= _ZERO_WIDTH_BURST_THRESHOLD:
            first = _ZERO_WIDTH_RE.search(text)
            start = first.start() if first is not None else 0
            findings.append(
                FirewallFinding(
                    category="steganography",
                    rule="unicode_zero_width_steganography",
                    start=start,
                    end=start + 1,
                    action="quarantined",
                )
            )
        return findings

    def _detect_injections(self, text: str) -> list[FirewallFinding]:
        matched_ids = self._injection_set.Match(text)
        if not matched_ids:
            return []
        findings: list[FirewallFinding] = []
        for idx in sorted(set(matched_ids)):
            rule_name, regex = _INJECTION_RULES[idx]
            match = regex.search(text)
            if match is not None:
                findings.append(
                    FirewallFinding(
                        category="injection",
                        rule=rule_name,
                        start=match.start(),
                        end=match.end(),
                        action="quarantined",
                    )
                )
        return findings

    def _replacement(self, rule_name: str, secret_value: str) -> tuple[str, str]:
        if self.policy is FirewallPolicy.VAULT_TOKENIZE:
            digest = hashlib.sha256(secret_value.encode("utf-8")).hexdigest()[:16]
            return f"vault://memory-secrets/{digest}", "tokenized"
        return f"[REDACTED:{rule_name}]", "redacted"

    def scan(
        self,
        text: str,
        *,
        trust_tier: TrustTier = TrustTier.T1_AGENT,
        policy: FirewallPolicy | str | None = None,
    ) -> FirewallResult:
        """Screen *text* for secrets, PII, steganography, and prompt injections.

        Returns a :class:`FirewallResult` with sanitized text and attested trust
        tier (downgraded to ``T3_QUARANTINED`` when injection/steganography is
        detected). Raises :class:`FirewallBlockedError` only when
        ``policy == FirewallPolicy.BLOCK`` and a secret/PII pattern matches.
        """
        t0 = time.perf_counter()
        active_policy = (
            self.policy
            if policy is None
            else (policy if isinstance(policy, FirewallPolicy) else FirewallPolicy(str(policy)))
        )

        scan_text, truncated = self._cap_input(text or "")
        findings: list[FirewallFinding] = []
        quarantine_reasons: list[str] = []

        # 1. Steganography & indirect prompt injection screening
        steg_findings = self._detect_steganography(scan_text)
        inj_findings = self._detect_injections(scan_text)
        for f in steg_findings + inj_findings:
            findings.append(f)
            if f.rule not in quarantine_reasons:
                quarantine_reasons.append(f.rule)

        # 2. Secret & PII scanning via single-pass RE2 Set pre-screen
        matched_secret_indices = self._secret_set.Match(scan_text)
        spans_to_replace: list[tuple[int, int, str, str, str]] = []
        if matched_secret_indices:
            for idx in sorted(set(matched_secret_indices)):
                rule = self._rules[idx]
                for match in rule.regex.finditer(scan_text):
                    if rule.validator in ("assignment_group1", "entropy"):
                        target_val = match.group(1)
                        span_start, span_end = match.span(1)
                    else:
                        target_val = match.group(0)
                        span_start, span_end = match.span(0)

                    if _is_exempt_token(target_val):
                        continue
                    if rule.validator == "luhn" and not _luhn_valid(target_val):
                        continue
                    if rule.validator == "entropy":
                        # Require realistic key entropy + mixed classes to avoid prose/config FPs.
                        sample = target_val[:256]
                        has_alpha = any(c.isalpha() for c in sample)
                        has_digit_or_sym = any(not c.isalpha() for c in sample)
                        if not (has_alpha and has_digit_or_sym) or _shannon_entropy(sample) < 3.4:
                            continue

                    spans_to_replace.append(
                        (span_start, span_end, rule.name, rule.category, target_val)
                    )

        # Resolve overlapping spans deterministically (earliest start, longest span wins).
        spans_to_replace.sort(key=lambda item: (item[0], -(item[1] - item[0])))
        merged_spans: list[tuple[int, int, str, str, str]] = []
        last_end = -1
        for span in spans_to_replace:
            if span[0] < last_end:
                continue
            merged_spans.append(span)
            last_end = span[1]

        secret_findings: list[FirewallFinding] = []
        if merged_spans:
            if active_policy is FirewallPolicy.BLOCK:
                for s_start, s_end, r_name, r_cat, _ in merged_spans:
                    secret_findings.append(
                        FirewallFinding(
                            category=r_cat,
                            rule=r_name,
                            start=s_start,
                            end=s_end,
                            action="blocked",
                        )
                    )
                raise FirewallBlockedError(secret_findings)

            out_parts: list[str] = []
            cursor = 0
            for s_start, s_end, r_name, r_cat, target_val in merged_spans:
                out_parts.append(scan_text[cursor:s_start])
                if active_policy is FirewallPolicy.VAULT_TOKENIZE:
                    digest = hashlib.sha256(target_val.encode("utf-8")).hexdigest()[:16]
                    repl, action = f"vault://memory-secrets/{digest}", "tokenized"
                else:
                    repl, action = f"[REDACTED:{r_name}]", "redacted"
                out_parts.append(repl)
                cursor = s_end
                secret_findings.append(
                    FirewallFinding(
                        category=r_cat,
                        rule=r_name,
                        start=s_start,
                        end=s_end,
                        action=action,
                    )
                )
            out_parts.append(scan_text[cursor:])
            sanitized = "".join(out_parts)
        else:
            sanitized = scan_text

        findings.extend(secret_findings)
        quarantined = bool(quarantine_reasons)
        effective_tier = TrustTier.T3_QUARANTINED if quarantined else trust_tier
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 3)

        return FirewallResult(
            text=sanitized,
            trust_tier=effective_tier,
            allowed=True,
            quarantined=quarantined,
            redacted_count=len(secret_findings),
            findings=tuple(findings),
            quarantine_reasons=tuple(quarantine_reasons),
            truncated_for_scan=truncated,
            elapsed_ms=elapsed_ms,
        )


_DEFAULT_FIREWALL: MemoryFirewall | None = None


def default_firewall() -> MemoryFirewall:
    """Return the process-wide singleton :class:`MemoryFirewall` (policy=redact)."""
    global _DEFAULT_FIREWALL
    if _DEFAULT_FIREWALL is None:
        _DEFAULT_FIREWALL = MemoryFirewall(policy=FirewallPolicy.REDACT)
    return _DEFAULT_FIREWALL


def screen_and_sanitize(
    text: str,
    *,
    trust_tier: TrustTier = TrustTier.T1_AGENT,
    policy: FirewallPolicy | str | None = None,
) -> FirewallResult:
    """Convenience helper invoking the default :class:`MemoryFirewall`."""
    return default_firewall().scan(text, trust_tier=trust_tier, policy=policy)


def _escape_attr(value: str) -> str:
    return html.escape(str(value or ""), quote=True)


def _escape_memory_body(body: str) -> str:
    """Neutralize embedded ``<archivist_memory`` or ``</archivist_memory`` tags."""
    return str(_MEMORY_TAG_ESCAPE_RE.sub(lambda m: m.group(1).replace("<", "&lt;"), body or ""))


def frame_memory(
    body: str,
    *,
    memory_id: str,
    trust_tier: TrustTier | str = TrustTier.T1_AGENT,
    actor: str = "local",
    scope: str = "user:global",
) -> str:
    """Wrap a retrieved memory in canonical untrusted-data structural framing."""
    tier_str = trust_tier.value if isinstance(trust_tier, TrustTier) else str(trust_tier)
    safe_body = _escape_memory_body(body)
    return (
        f'<archivist_memory id="{_escape_attr(memory_id)}" '
        f'trust_tier="{_escape_attr(tier_str)}" '
        f'actor="{_escape_attr(actor)}" '
        f'scope="{_escape_attr(scope)}">\n'
        f"{safe_body}\n"
        "</archivist_memory>"
    )


def frame_memories(memories: Sequence[dict[str, Any]]) -> str:
    """Wrap a sequence of memory dicts in canonical ``<archivist_memory>`` tags."""
    blocks = [
        frame_memory(
            str(m.get("tier_text") or m.get("text") or m.get("content") or ""),
            memory_id=str(m.get("id") or m.get("memory_id") or ""),
            trust_tier=str(m.get("trust_tier") or TrustTier.T1_AGENT.value),
            actor=str(m.get("actor") or m.get("principal_id") or "local"),
            scope=str(m.get("scope") or m.get("scope_id") or "user:global"),
        )
        for m in memories
    ]
    return "\n\n".join(blocks)
