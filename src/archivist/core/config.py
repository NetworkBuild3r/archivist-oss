"""Archivist configuration — Pydantic Settings v2.

All values are read from environment variables (case-insensitive) and from
an optional ``.env`` file in the working directory.  Existing ``.env`` files
and deployment scripts need no changes — variable names are unchanged.

Architecture
------------
* ``ArchivistSettings`` — frozen Pydantic Settings v2 model that owns the
  authoritative parsed + validated config.
* ``_settings`` — module-level singleton constructed once at import time.
* Module-level UPPER_CASE re-exports — backward-compat layer so that all 51
  existing ``from archivist.core.config import FOO`` call-sites continue to
  work without modification (Phase A).  These will be removed in Phase B once
  consumers migrate to ``settings.foo``.

Testing
-------
* ``monkeypatch.setattr(config, "FOO", value)`` patches the module-level
  re-export — works for 99 % of tests, no changes required.
* Tests that need to validate parsing / validation logic call
  ``_build_settings()`` after ``monkeypatch.setenv()``.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any

import yaml
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger("archivist.config")


# ---------------------------------------------------------------------------
# Settings model
# ---------------------------------------------------------------------------


class ArchivistSettings(BaseSettings):
    """Validated, immutable Archivist configuration.

    All fields are loaded from environment variables (case-insensitive) or a
    ``.env`` file.  Pydantic Settings v2 lowercases env-var keys during lookup,
    so ``QDRANT_URL`` in the environment matches the ``qdrant_url`` field.

    Attributes are documented inline.  Refer to the section comments for
    logical groupings.
    """

    model_config = SettingsConfigDict(
        env_prefix="",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        frozen=True,
        extra="ignore",
    )

    # ── Vector store ──────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "archivist_memories"
    vector_dim: int = Field(default=1024, ge=1)

    # ── Embedding model (OpenAI-compatible API) ───────────────────────────────
    embed_model: str = "text-embedding-v3"
    embed_url: str = ""
    embed_api_key: str = ""
    # When True, skip OpenAI batch embedding (array `input`) and use parallel single-text
    # requests only. Some vLLM embed servers return 422 for array input.
    embed_disable_batch: bool = False
    # Max strings per /v1/embeddings POST. Indexing can send 100+ chunks; vLLM often caps
    # batch size or total tokens — huge arrays return 422. Sub-batch below this limit.
    embed_max_batch_inputs: int = Field(default=32, ge=1)

    # ── LLM (OpenAI-compatible chat/completions API) ─────────────────────────
    llm_url: str = "http://localhost:4000"
    llm_model: str = "gpt-4o-mini"
    llm_api_key: str = ""
    llm_refine_model: str = ""
    llm_synth_model: str = ""
    llm_refine_concurrency: int = Field(default=5, ge=1)

    # ── Curator LLM override (v1.11) ─────────────────────────────────────────
    curator_llm_url: str = ""
    curator_llm_model: str = ""
    curator_llm_api_key: str = ""

    # ── Benchmark judge LLM ───────────────────────────────────────────────────
    benchmark_judge_llm_url: str = ""
    benchmark_judge_llm_model: str = ""
    benchmark_judge_llm_api_key: str | None = None
    refine_skip_threshold: float = 0.0

    # ── Storage paths ─────────────────────────────────────────────────────────
    memory_root: str = "/data/memories"
    sqlite_path: str = "/data/archivist/graph.db"
    namespaces_config_path: str = ""

    # ── Chunking ──────────────────────────────────────────────────────────────
    chunk_size: int = Field(default=800, ge=1)
    chunk_overlap: int = Field(default=100, ge=0)
    parent_chunk_size: int = Field(default=2000, ge=1)
    parent_chunk_overlap: int = Field(default=200, ge=0)
    child_chunk_size: int = Field(default=500, ge=1)
    child_chunk_overlap: int = Field(default=100, ge=0)
    chunking_strategy: str = "semantic"

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    rerank_enabled: bool = False
    rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rerank_top_k: int = Field(default=10, ge=1)
    vector_search_limit: int = Field(default=64, ge=1)

    # ── Tiered context (v0.5) ─────────────────────────────────────────────────
    tiered_context_enabled: bool = True
    l0_max_tokens: int = Field(default=100, ge=1)
    l1_max_tokens: int = Field(default=500, ge=1)

    # ── Graph-augmented retrieval (v0.5) ─────────────────────────────────────
    graph_retrieval_enabled: bool = True
    graph_retrieval_weight: float = 0.3
    multi_hop_depth: int = Field(default=2, ge=1)
    temporal_decay_halflife_days: int = Field(default=30, ge=0)  # 0 = decay disabled

    # ── Trajectory & feedback (v0.6) ─────────────────────────────────────────
    outcome_retrieval_boost: float = 0.15
    outcome_retrieval_penalty: float = 0.1

    # ── Hot cache (v0.8) ──────────────────────────────────────────────────────
    hot_cache_enabled: bool = True
    hot_cache_max_per_agent: int = Field(default=128, ge=1)
    hot_cache_ttl_seconds: int = Field(default=600, ge=1)

    # ── Retrieval trajectory export (v0.8) ────────────────────────────────────
    trajectory_export_enabled: bool = True
    trajectory_export_max: int = Field(default=200, ge=1)

    # ── Observability (v0.9) ──────────────────────────────────────────────────
    metrics_enabled: bool = True
    metrics_auth_exempt: bool = False
    metrics_collect_interval_seconds: int = Field(default=60, ge=5)
    default_consistency: str = "eventual"
    slow_embed_ms: float = 0.0
    slow_qdrant_ms: float = 0.0
    slow_llm_ms: float = 0.0
    archivist_invalidation_export_path: str = ""

    # ── Webhooks (v0.9) ──────────────────────────────────────────────────────
    webhook_url: str = ""
    webhook_timeout: float = 5.0
    # Stored as str so pydantic-settings passes env-var value to our validator
    # without attempting JSON-decode first (set[str] would require JSON format).
    webhook_events: str = ""

    # ── Curator intelligence (v1.0) ──────────────────────────────────────────
    dedup_llm_enabled: bool = False
    dedup_llm_threshold: float = 0.80
    curator_tip_budget: int = Field(default=20, ge=0)
    curator_max_parallel: int = Field(default=4, ge=1)
    curator_queue_drain_interval: int = Field(default=30, ge=1)
    hotness_weight: float = 0.15
    hotness_halflife_days: int = Field(default=7, ge=1)
    importance_weight: float = 0.10
    importance_floor: float = 0.3
    importance_grace_days: int = Field(default=7, ge=0)

    # ── Entity injection tuning (v1.8) ────────────────────────────────────────
    max_entity_fact_injections: int = Field(default=15, ge=0)
    entity_specificity_max_mentions: int = Field(default=20, ge=1)

    # ── Temporal intent & adaptive retrieval (v1.9) ───────────────────────────
    temporal_intent_enabled: bool = True
    temporal_historical_halflife_multiplier: float = 10.0
    bm25_rescue_enabled: bool = True
    bm25_rescue_min_score_ratio: float = 0.6
    bm25_rescue_max_slots: int = Field(default=3, ge=0)
    adaptive_vector_limit_enabled: bool = True
    adaptive_vector_min_results: int = Field(default=3, ge=1)
    adaptive_vector_limit_multiplier: float = 3.0
    cross_agent_max_share: float = 0.6

    # ── Topic routing (v1.10) ────────────────────────────────────────────────
    topic_routing_enabled: bool = True
    topic_map_path: str = ""

    # ── Curator ──────────────────────────────────────────────────────────────
    curator_interval_minutes: int = Field(default=30, ge=1)
    orphan_sweep_enabled: bool = True
    orphan_sweep_every_n_cycles: int = Field(default=12, ge=1)
    # Stored as str so pydantic-settings passes env-var value to our validator
    # without attempting JSON-decode first (list[str] would require JSON format).
    curator_extract_prefixes: str = "agents/,memories/"
    curator_extract_skip_segments: str = "skills,.cursor,.git"

    # ── Memory awareness — Stage 0 query classification (v1.6) ───────────────
    query_classification_enabled: bool = True

    # ── BM25 / FTS5 hybrid search (v1.2) ─────────────────────────────────────
    bm25_enabled: bool = False
    bm25_weight: float = 0.3
    vector_weight: float = 0.7

    # ── Needle-finding: query expansion + dynamic threshold (v1.10) ──────────
    query_expansion_enabled: bool = False
    query_expansion_count: int = Field(default=3, ge=1)
    query_expansion_model: str = ""
    dynamic_threshold_enabled: bool = True

    # ── Contextual chunk augmentation (v1.10) ────────────────────────────────
    contextual_augmentation_enabled: bool = True

    # ── Reverse HyDE: write-time question generation (v2.0) ──────────────────
    reverse_hyde_enabled: bool = True
    reverse_hyde_questions_per_chunk: int = Field(default=3, ge=1)

    # ── Synthetic question generation: index-time multi-representation (v2.1) ─
    synthetic_questions_enabled: bool = False
    synthetic_questions_count: int = Field(default=4, ge=1)

    # ── Provenance & actor-aware memory (v2.3 — Phase 6) ─────────────────────
    provenance_enabled: bool = True
    default_confidence_human: float = Field(default=1.0, ge=0.0, le=1.0)
    default_confidence_agent: float = Field(default=0.8, ge=0.0, le=1.0)
    default_confidence_system: float = Field(default=0.7, ge=0.0, le=1.0)
    default_confidence_tool: float = Field(default=0.7, ge=0.0, le=1.0)
    default_confidence_extracted: float = Field(default=0.5, ge=0.0, le=1.0)
    min_fact_confidence: float = Field(default=0.3, ge=0.0, le=1.0)

    # ── Cross-encoder reranker (v2.2) ─────────────────────────────────────────
    reranker_enabled: bool = False
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    reranker_top_k: int = Field(default=20, ge=1)

    # ── Micro-chunk limits (v1.10) ────────────────────────────────────────────
    max_micro_chunks_per_memory: int = Field(default=5, ge=1)

    # ── HNSW tuning (v1.10) ──────────────────────────────────────────────────
    qdrant_hnsw_m: int = Field(default=32, ge=4)
    qdrant_hnsw_ef_construct: int = Field(default=256, ge=4)
    qdrant_search_ef: int = Field(default=256, ge=4)

    # ── Enterprise scaling (v1.10) ────────────────────────────────────────────
    namespace_sharding_enabled: bool = False
    single_collection_mode: bool = True
    cache_backend: str = "memory"
    redis_url: str = "redis://localhost:6379/0"
    redis_key_prefix: str = "archivist:"
    latency_budget_ms: int = Field(default=500, ge=1)

    # ── Backup & restore (v1.10) ──────────────────────────────────────────────
    backup_dir: str = "/data/archivist/backups"
    backup_retention_count: int = Field(default=5, ge=1)
    backup_include_files: bool = False
    backup_pre_prune: bool = False

    # ── SQLite async pool (v1.12) ──────────────────────────────────────────────
    sqlite_wal_autocheckpoint: int = Field(default=1000, ge=0)
    sqlite_busy_timeout_ms: int = Field(default=5000, ge=0)

    # ── Graph backend selection (v2.4 — Phase 4) ─────────────────────────────
    # Set GRAPH_BACKEND=postgres to switch from SQLite to PostgreSQL.
    # When GRAPH_BACKEND=sqlite (default) DATABASE_URL is ignored entirely.
    graph_backend: str = "sqlite"
    # PostgreSQL DSN in asyncpg format: postgresql://user:pw@host:5432/dbname
    database_url: str = ""
    pg_pool_min: int = Field(default=5, ge=1)
    pg_pool_max: int = Field(default=20, ge=1)

    # ── Transactional outbox (v2.1 — Phase 3) ────────────────────────────────
    outbox_enabled: bool = False
    outbox_drain_interval: int = Field(default=2, ge=1)
    outbox_batch_size: int = Field(default=50, ge=1)
    outbox_max_retries: int = Field(default=5, ge=1)
    outbox_orphan_timeout_seconds: int = Field(default=60, ge=1)
    outbox_orphan_sweep_every_n: int = Field(default=30, ge=1)
    outbox_retention_days: int = Field(default=7, ge=1)

    # ── Context window management (v1.1) ─────────────────────────────────────
    default_context_budget: int = Field(default=128000, ge=1)

    # ── Journal exports (v1.5) ────────────────────────────────────────────────
    journal_enabled: bool = False
    journal_dir: str = "/data/archivist/journal"

    # ── Server ────────────────────────────────────────────────────────────────
    mcp_port: int = Field(default=3100, ge=1, le=65535)
    mcp_sse_enabled: bool = True
    archivist_api_key: str = ""

    # ── Conflict detection ────────────────────────────────────────────────────
    conflict_check_on_store: bool = False
    conflict_block_on_store: bool = False

    # ── Agent → team mapping ──────────────────────────────────────────────────
    team_map_path: str = ""
    team_map_json: str = ""

    # ── Field validators ──────────────────────────────────────────────────────

    @field_validator("embed_url", mode="before")
    @classmethod
    def _resolve_embed_url(cls, v: Any, info: Any) -> str:
        """Fall back to LLM_URL when EMBED_URL is not set."""
        if v:
            return str(v).strip()
        return os.getenv("LLM_URL", "http://localhost:4000")

    @field_validator("embed_api_key", mode="before")
    @classmethod
    def _resolve_embed_api_key(cls, v: Any, info: Any) -> str:
        """Fall back to LLM_API_KEY when EMBED_API_KEY is not set."""
        if v:
            return str(v).strip()
        return os.getenv("LLM_API_KEY", "")

    @field_validator("curator_llm_api_key", mode="before")
    @classmethod
    def _resolve_curator_llm_api_key(cls, v: Any, info: Any) -> str:
        """Fall back to LLM_API_KEY when CURATOR_LLM_API_KEY is not set."""
        if v:
            return str(v).strip()
        return os.getenv("LLM_API_KEY", "")

    @field_validator(
        "llm_refine_model",
        "llm_synth_model",
        "curator_llm_url",
        "curator_llm_model",
        "benchmark_judge_llm_url",
        "benchmark_judge_llm_model",
        "query_expansion_model",
        "archivist_invalidation_export_path",
        "webhook_url",
        "namespaces_config_path",
        "topic_map_path",
        "team_map_path",
        "archivist_api_key",
        mode="before",
    )
    @classmethod
    def _strip_str(cls, v: Any) -> str:
        """Strip whitespace from optional string fields."""
        return str(v).strip() if v else ""

    @field_validator("webhook_events", mode="before")
    @classmethod
    def _parse_webhook_events(cls, v: Any) -> str:
        """Normalise: accept str, set, or list and convert to a canonical comma-separated str."""
        if isinstance(v, set | list):
            return ",".join(str(e).strip() for e in v if str(e).strip())
        return str(v).strip() if v else ""

    @field_validator("curator_extract_prefixes", "curator_extract_skip_segments", mode="before")
    @classmethod
    def _parse_csv_list(cls, v: Any) -> str:
        """Normalise: accept str or list and convert to a canonical comma-separated str."""
        if isinstance(v, list):
            return ",".join(str(x).strip() for x in v if str(x).strip())
        return str(v).strip() if v else ""

    @field_validator("cache_backend", mode="before")
    @classmethod
    def _lowercase_cache_backend(cls, v: Any) -> str:
        """Normalize cache backend to lowercase."""
        return str(v).lower().strip() if v else "memory"

    @field_validator("metrics_collect_interval_seconds", mode="before")
    @classmethod
    def _min_metrics_interval(cls, v: Any) -> int:
        """Enforce minimum metrics collection interval of 5 seconds."""
        return max(5, int(v))

    @field_validator("llm_refine_concurrency", mode="before")
    @classmethod
    def _min_refine_concurrency(cls, v: Any) -> int:
        """Enforce minimum LLM refinement concurrency of 1."""
        return max(1, int(v))

    # ── Cross-field validators ────────────────────────────────────────────────

    @model_validator(mode="after")
    def _validate_cross_field(self) -> ArchivistSettings:
        """Enforce cross-field constraints that require multiple field values."""
        errors: list[str] = []

        if self.chunk_overlap >= self.chunk_size:
            errors.append(
                f"CHUNK_OVERLAP ({self.chunk_overlap}) must be < CHUNK_SIZE ({self.chunk_size})"
            )
        if not (0.0 <= self.retrieval_threshold <= 1.0):
            errors.append(f"RETRIEVAL_THRESHOLD must be in [0, 1], got {self.retrieval_threshold}")
        if self.sqlite_wal_autocheckpoint < 0:
            errors.append(
                f"SQLITE_WAL_AUTOCHECKPOINT must be >= 0, got {self.sqlite_wal_autocheckpoint}"
            )
        if self.sqlite_busy_timeout_ms < 0:
            errors.append(f"SQLITE_BUSY_TIMEOUT_MS must be >= 0, got {self.sqlite_busy_timeout_ms}")

        if errors:
            raise ValueError(
                "Archivist config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
            )
        return self

    # ── Computed properties ───────────────────────────────────────────────────

    @property
    def default_confidence_by_actor_type(self) -> dict[str, float]:
        """Confidence defaults keyed by actor type (matches original dict shape)."""
        return {
            "human": self.default_confidence_human,
            "agent": self.default_confidence_agent,
            "system": self.default_confidence_system,
            "tool": self.default_confidence_tool,
            "extracted": self.default_confidence_extracted,
        }

    @property
    def webhook_events_set(self) -> set[str]:
        """Parsed WEBHOOK_EVENTS as a set of strings."""
        return set(e.strip() for e in self.webhook_events.split(",") if e.strip())

    @property
    def curator_extract_prefixes_list(self) -> list[str]:
        """Parsed CURATOR_EXTRACT_PREFIXES as a list."""
        return [x.strip() for x in self.curator_extract_prefixes.split(",") if x.strip()]

    @property
    def curator_extract_skip_segments_list(self) -> list[str]:
        """Parsed CURATOR_EXTRACT_SKIP_SEGMENTS as a list."""
        return [x.strip() for x in self.curator_extract_skip_segments.split(",") if x.strip()]

    @property
    def feature_flags(self) -> dict[str, bool]:
        """All boolean feature flags for operational visibility at startup."""
        return {
            "BM25_ENABLED": self.bm25_enabled,
            "QUERY_EXPANSION_ENABLED": self.query_expansion_enabled,
            "DYNAMIC_THRESHOLD_ENABLED": self.dynamic_threshold_enabled,
            "CONTEXTUAL_AUGMENTATION_ENABLED": self.contextual_augmentation_enabled,
            "REVERSE_HYDE_ENABLED": self.reverse_hyde_enabled,
            "GRAPH_RETRIEVAL_ENABLED": self.graph_retrieval_enabled,
            "TIERED_CONTEXT_ENABLED": self.tiered_context_enabled,
            "HOT_CACHE_ENABLED": self.hot_cache_enabled,
            "RERANK_ENABLED": self.rerank_enabled,
            "TOPIC_ROUTING_ENABLED": self.topic_routing_enabled,
            "TEMPORAL_INTENT_ENABLED": self.temporal_intent_enabled,
            "BM25_RESCUE_ENABLED": self.bm25_rescue_enabled,
            "ADAPTIVE_VECTOR_LIMIT_ENABLED": self.adaptive_vector_limit_enabled,
            "SYNTHETIC_QUESTIONS_ENABLED": self.synthetic_questions_enabled,
            "RERANKER_ENABLED": self.reranker_enabled,
            "NAMESPACE_SHARDING_ENABLED": self.namespace_sharding_enabled,
            "SINGLE_COLLECTION_MODE": self.single_collection_mode,
            "QUERY_CLASSIFICATION_ENABLED": self.query_classification_enabled,
            "CONFLICT_CHECK_ON_STORE": self.conflict_check_on_store,
            "PROVENANCE_ENABLED": self.provenance_enabled,
            "JOURNAL_ENABLED": self.journal_enabled,
            "METRICS_ENABLED": self.metrics_enabled,
            "METRICS_AUTH_EXEMPT": self.metrics_auth_exempt,
            "MCP_SSE_ENABLED": self.mcp_sse_enabled,
            "OUTBOX_ENABLED": self.outbox_enabled,
        }

    def _load_team_map(self) -> dict[str, str]:
        """Load agent-to-team mapping from YAML file or JSON env var.

        Returns:
            Dict mapping agent IDs to team names.  Empty dict when neither
            source is configured or both fail to parse.
        """
        path = self.team_map_path
        if path and os.path.isfile(path):
            try:
                with open(path) as f:
                    data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except Exception:
                pass
        raw = self.team_map_json
        if raw:
            try:
                return json.loads(raw)
            except Exception:
                pass
        return {}

    def _log_startup(self) -> None:
        """Emit structured startup log lines for feature flags and outbox config."""
        flags = self.feature_flags
        enabled = [k for k, v in flags.items() if v]
        disabled = [k for k, v in flags.items() if not v]
        logger.info(
            "config.feature_flags",
            extra={
                "enabled": enabled,
                "disabled": disabled,
                "enabled_count": len(enabled),
                "disabled_count": len(disabled),
            },
        )
        logger.info(
            "config.outbox",
            extra={
                "OUTBOX_ENABLED": self.outbox_enabled,
                "OUTBOX_DRAIN_INTERVAL": self.outbox_drain_interval,
                "OUTBOX_BATCH_SIZE": self.outbox_batch_size,
                "OUTBOX_MAX_RETRIES": self.outbox_max_retries,
                "OUTBOX_ORPHAN_TIMEOUT_SECONDS": self.outbox_orphan_timeout_seconds,
                "OUTBOX_ORPHAN_SWEEP_EVERY_N": self.outbox_orphan_sweep_every_n,
                "OUTBOX_RETENTION_DAYS": self.outbox_retention_days,
            },
        )
        if self.curator_llm_model:
            logger.info(
                "config.curator_llm_override",
                extra={
                    "curator_llm_model": self.curator_llm_model,
                    "curator_llm_url": self.curator_llm_url or self.llm_url,
                },
            )


# ---------------------------------------------------------------------------
# Singleton + test helper
# ---------------------------------------------------------------------------


def _build_settings(**overrides: Any) -> ArchivistSettings:
    """Construct a fresh ArchivistSettings instance from the current environment.

    Intended for use in tests that need to exercise config parsing or
    validation.  Call after ``monkeypatch.setenv()`` to pick up changes::

        def test_invalid_chunk(monkeypatch):
            monkeypatch.setenv("CHUNK_SIZE", "100")
            monkeypatch.setenv("CHUNK_OVERLAP", "100")
            with pytest.raises(ValidationError):
                _build_settings()

    Args:
        **overrides: Field values to pass directly to the constructor,
            bypassing env-var lookup for those fields.

    Returns:
        A new frozen ``ArchivistSettings`` instance.
    """
    return ArchivistSettings(**overrides)


_settings: ArchivistSettings = ArchivistSettings()
_settings._log_startup()

# ---------------------------------------------------------------------------
# Static constants (not env-configurable)
# ---------------------------------------------------------------------------

VALID_RETENTION_CLASSES = ("ephemeral", "standard", "durable", "permanent")
DURABLE_ENTITY_TYPES = frozenset(
    {
        "person",
        "host",
        "server",
        "service",
        "credential",
        "organization",
        "cluster",
        "database",
        "network",
        "user",
    }
)

# ---------------------------------------------------------------------------
# Module-level UPPER_CASE re-exports (Phase A backward-compat layer)
#
# These allow all existing ``from archivist.core.config import FOO`` imports
# to continue working without modification.  In Phase B these will be removed
# and consumers will use ``from archivist.core.config import settings``.
# ---------------------------------------------------------------------------

QDRANT_URL = _settings.qdrant_url
QDRANT_COLLECTION = _settings.qdrant_collection
VECTOR_DIM = _settings.vector_dim

EMBED_MODEL = _settings.embed_model
EMBED_URL = _settings.embed_url
EMBED_API_KEY = _settings.embed_api_key
EMBED_DISABLE_BATCH = _settings.embed_disable_batch
EMBED_MAX_BATCH_INPUTS = _settings.embed_max_batch_inputs

LLM_URL = _settings.llm_url
LLM_MODEL = _settings.llm_model
LLM_API_KEY = _settings.llm_api_key
LLM_REFINE_MODEL = _settings.llm_refine_model
LLM_SYNTH_MODEL = _settings.llm_synth_model
LLM_REFINE_CONCURRENCY = _settings.llm_refine_concurrency

CURATOR_LLM_URL = _settings.curator_llm_url
CURATOR_LLM_MODEL = _settings.curator_llm_model
CURATOR_LLM_API_KEY = _settings.curator_llm_api_key

BENCHMARK_JUDGE_LLM_URL = _settings.benchmark_judge_llm_url
BENCHMARK_JUDGE_LLM_MODEL = _settings.benchmark_judge_llm_model
BENCHMARK_JUDGE_LLM_API_KEY = _settings.benchmark_judge_llm_api_key
REFINE_SKIP_THRESHOLD = _settings.refine_skip_threshold

MEMORY_ROOT = _settings.memory_root
SQLITE_PATH = _settings.sqlite_path
NAMESPACES_CONFIG_PATH = _settings.namespaces_config_path

CHUNK_SIZE = _settings.chunk_size
CHUNK_OVERLAP = _settings.chunk_overlap
PARENT_CHUNK_SIZE = _settings.parent_chunk_size
PARENT_CHUNK_OVERLAP = _settings.parent_chunk_overlap
CHILD_CHUNK_SIZE = _settings.child_chunk_size
CHILD_CHUNK_OVERLAP = _settings.child_chunk_overlap
CHUNKING_STRATEGY = _settings.chunking_strategy

RETRIEVAL_THRESHOLD = _settings.retrieval_threshold
RERANK_ENABLED = _settings.rerank_enabled
RERANK_MODEL = _settings.rerank_model
RERANK_TOP_K = _settings.rerank_top_k
VECTOR_SEARCH_LIMIT = _settings.vector_search_limit

TIERED_CONTEXT_ENABLED = _settings.tiered_context_enabled
L0_MAX_TOKENS = _settings.l0_max_tokens
L1_MAX_TOKENS = _settings.l1_max_tokens

GRAPH_RETRIEVAL_ENABLED = _settings.graph_retrieval_enabled
GRAPH_RETRIEVAL_WEIGHT = _settings.graph_retrieval_weight
MULTI_HOP_DEPTH = _settings.multi_hop_depth
TEMPORAL_DECAY_HALFLIFE_DAYS = _settings.temporal_decay_halflife_days

OUTCOME_RETRIEVAL_BOOST = _settings.outcome_retrieval_boost
OUTCOME_RETRIEVAL_PENALTY = _settings.outcome_retrieval_penalty

HOT_CACHE_ENABLED = _settings.hot_cache_enabled
HOT_CACHE_MAX_PER_AGENT = _settings.hot_cache_max_per_agent
HOT_CACHE_TTL_SECONDS = _settings.hot_cache_ttl_seconds

TRAJECTORY_EXPORT_ENABLED = _settings.trajectory_export_enabled
TRAJECTORY_EXPORT_MAX = _settings.trajectory_export_max

METRICS_ENABLED = _settings.metrics_enabled
METRICS_AUTH_EXEMPT = _settings.metrics_auth_exempt
METRICS_COLLECT_INTERVAL_SECONDS = _settings.metrics_collect_interval_seconds
DEFAULT_CONSISTENCY = _settings.default_consistency
SLOW_EMBED_MS = _settings.slow_embed_ms
SLOW_QDRANT_MS = _settings.slow_qdrant_ms
SLOW_LLM_MS = _settings.slow_llm_ms
ARCHIVIST_INVALIDATION_EXPORT_PATH = _settings.archivist_invalidation_export_path

WEBHOOK_URL = _settings.webhook_url
WEBHOOK_TIMEOUT = _settings.webhook_timeout
WEBHOOK_EVENTS: set[str] = _settings.webhook_events_set

DEDUP_LLM_ENABLED = _settings.dedup_llm_enabled
DEDUP_LLM_THRESHOLD = _settings.dedup_llm_threshold
CURATOR_TIP_BUDGET = _settings.curator_tip_budget
CURATOR_MAX_PARALLEL = _settings.curator_max_parallel
CURATOR_QUEUE_DRAIN_INTERVAL = _settings.curator_queue_drain_interval
HOTNESS_WEIGHT = _settings.hotness_weight
HOTNESS_HALFLIFE_DAYS = _settings.hotness_halflife_days
IMPORTANCE_WEIGHT = _settings.importance_weight
IMPORTANCE_FLOOR = _settings.importance_floor
IMPORTANCE_GRACE_DAYS = _settings.importance_grace_days

MAX_ENTITY_FACT_INJECTIONS = _settings.max_entity_fact_injections
ENTITY_SPECIFICITY_MAX_MENTIONS = _settings.entity_specificity_max_mentions

TEMPORAL_INTENT_ENABLED = _settings.temporal_intent_enabled
TEMPORAL_HISTORICAL_HALFLIFE_MULTIPLIER = _settings.temporal_historical_halflife_multiplier
BM25_RESCUE_ENABLED = _settings.bm25_rescue_enabled
BM25_RESCUE_MIN_SCORE_RATIO = _settings.bm25_rescue_min_score_ratio
BM25_RESCUE_MAX_SLOTS = _settings.bm25_rescue_max_slots
ADAPTIVE_VECTOR_LIMIT_ENABLED = _settings.adaptive_vector_limit_enabled
ADAPTIVE_VECTOR_MIN_RESULTS = _settings.adaptive_vector_min_results
ADAPTIVE_VECTOR_LIMIT_MULTIPLIER = _settings.adaptive_vector_limit_multiplier
CROSS_AGENT_MAX_SHARE = _settings.cross_agent_max_share

TOPIC_ROUTING_ENABLED = _settings.topic_routing_enabled
TOPIC_MAP_PATH = _settings.topic_map_path

CURATOR_INTERVAL_MINUTES = _settings.curator_interval_minutes
ORPHAN_SWEEP_ENABLED = _settings.orphan_sweep_enabled
ORPHAN_SWEEP_EVERY_N_CYCLES = _settings.orphan_sweep_every_n_cycles
CURATOR_EXTRACT_PREFIXES: list[str] = _settings.curator_extract_prefixes_list
CURATOR_EXTRACT_SKIP_SEGMENTS: list[str] = _settings.curator_extract_skip_segments_list

QUERY_CLASSIFICATION_ENABLED = _settings.query_classification_enabled

BM25_ENABLED = _settings.bm25_enabled
BM25_WEIGHT = _settings.bm25_weight
VECTOR_WEIGHT = _settings.vector_weight

QUERY_EXPANSION_ENABLED = _settings.query_expansion_enabled
QUERY_EXPANSION_COUNT = _settings.query_expansion_count
QUERY_EXPANSION_MODEL = _settings.query_expansion_model
DYNAMIC_THRESHOLD_ENABLED = _settings.dynamic_threshold_enabled

CONTEXTUAL_AUGMENTATION_ENABLED = _settings.contextual_augmentation_enabled

REVERSE_HYDE_ENABLED = _settings.reverse_hyde_enabled
REVERSE_HYDE_QUESTIONS_PER_CHUNK = _settings.reverse_hyde_questions_per_chunk

SYNTHETIC_QUESTIONS_ENABLED = _settings.synthetic_questions_enabled
SYNTHETIC_QUESTIONS_COUNT = _settings.synthetic_questions_count

PROVENANCE_ENABLED = _settings.provenance_enabled
DEFAULT_CONFIDENCE_BY_ACTOR_TYPE = _settings.default_confidence_by_actor_type
MIN_FACT_CONFIDENCE = _settings.min_fact_confidence

RERANKER_ENABLED = _settings.reranker_enabled
RERANKER_MODEL = _settings.reranker_model
RERANKER_TOP_K = _settings.reranker_top_k

MAX_MICRO_CHUNKS_PER_MEMORY = _settings.max_micro_chunks_per_memory

QDRANT_HNSW_M = _settings.qdrant_hnsw_m
QDRANT_HNSW_EF_CONSTRUCT = _settings.qdrant_hnsw_ef_construct
QDRANT_SEARCH_EF = _settings.qdrant_search_ef

NAMESPACE_SHARDING_ENABLED = _settings.namespace_sharding_enabled
SINGLE_COLLECTION_MODE = _settings.single_collection_mode
CACHE_BACKEND = _settings.cache_backend
REDIS_URL = _settings.redis_url
REDIS_KEY_PREFIX = _settings.redis_key_prefix
LATENCY_BUDGET_MS = _settings.latency_budget_ms

BACKUP_DIR = _settings.backup_dir
BACKUP_RETENTION_COUNT = _settings.backup_retention_count
BACKUP_INCLUDE_FILES = _settings.backup_include_files
BACKUP_PRE_PRUNE = _settings.backup_pre_prune

SQLITE_WAL_AUTOCHECKPOINT = _settings.sqlite_wal_autocheckpoint
SQLITE_BUSY_TIMEOUT_MS = _settings.sqlite_busy_timeout_ms

GRAPH_BACKEND = _settings.graph_backend
DATABASE_URL = _settings.database_url
PG_POOL_MIN = _settings.pg_pool_min
PG_POOL_MAX = _settings.pg_pool_max

OUTBOX_ENABLED = _settings.outbox_enabled
OUTBOX_DRAIN_INTERVAL = _settings.outbox_drain_interval
OUTBOX_BATCH_SIZE = _settings.outbox_batch_size
OUTBOX_MAX_RETRIES = _settings.outbox_max_retries
OUTBOX_ORPHAN_TIMEOUT_SECONDS = _settings.outbox_orphan_timeout_seconds
OUTBOX_ORPHAN_SWEEP_EVERY_N = _settings.outbox_orphan_sweep_every_n
OUTBOX_RETENTION_DAYS = _settings.outbox_retention_days

DEFAULT_CONTEXT_BUDGET = _settings.default_context_budget

JOURNAL_ENABLED = _settings.journal_enabled
JOURNAL_DIR = _settings.journal_dir

MCP_PORT = _settings.mcp_port
MCP_SSE_ENABLED = _settings.mcp_sse_enabled
ARCHIVIST_API_KEY = _settings.archivist_api_key

CONFLICT_CHECK_ON_STORE = _settings.conflict_check_on_store
CONFLICT_BLOCK_ON_STORE = _settings.conflict_block_on_store

TEAM_MAP_PATH = _settings.team_map_path
TEAM_MAP: dict[str, str] = _settings._load_team_map()


# ---------------------------------------------------------------------------
# Backward-compat shim for callers that invoke config._log_feature_flags()
# directly (e.g. startup code, tests).  The logic now lives on the settings
# instance; this function delegates to it.
# ---------------------------------------------------------------------------


def _log_feature_flags() -> None:
    """Emit structured feature-flag log lines.  Delegates to _settings._log_startup()."""
    _settings._log_startup()
