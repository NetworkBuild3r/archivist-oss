"""Archivist configuration loaded from environment variables.

All values have sensible defaults for local / docker-compose development.
Override via .env or environment variables in production.
"""

import logging
import os
import yaml

logger = logging.getLogger("archivist.config")


def _env_bool(key: str, default: str = "true") -> bool:
    """Parse a boolean environment variable (true/1/yes → True)."""
    return os.getenv(key, default).lower() in ("true", "1", "yes")

# ── Vector store ──────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "archivist_memories")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))

# ── Embedding model (OpenAI-compatible API) ───────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v3")
EMBED_URL = os.getenv("EMBED_URL", os.getenv("LLM_URL", "http://localhost:4000"))
EMBED_API_KEY = os.getenv("EMBED_API_KEY", os.getenv("LLM_API_KEY", ""))

# ── LLM (OpenAI-compatible chat/completions API) ─────────────────────────────
LLM_URL = os.getenv("LLM_URL", "http://localhost:4000")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")
# Optional: separate models for per-chunk refinement vs final synthesis (empty = LLM_MODEL for both).
LLM_REFINE_MODEL = os.getenv("LLM_REFINE_MODEL", "").strip()
LLM_SYNTH_MODEL = os.getenv("LLM_SYNTH_MODEL", "").strip()
# Parallel refinement: max concurrent LLM calls for Stage 5 (minimum 1).
LLM_REFINE_CONCURRENCY = max(1, int(os.getenv("LLM_REFINE_CONCURRENCY", "5")))
# If top hit score is >= this, skip per-chunk LLM refinement and use tier text (0 = disabled, always refine).
REFINE_SKIP_THRESHOLD = float(os.getenv("REFINE_SKIP_THRESHOLD", "0.0"))

# ── Storage paths ─────────────────────────────────────────────────────────────
MEMORY_ROOT = os.getenv("MEMORY_ROOT", "/data/memories")
SQLITE_PATH = os.getenv("SQLITE_PATH", "/data/archivist/graph.db")
NAMESPACES_CONFIG_PATH = os.getenv("NAMESPACES_CONFIG_PATH", "/data/archivist/config/namespaces.yaml")

# ── Chunking ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
PARENT_CHUNK_SIZE = int(os.getenv("PARENT_CHUNK_SIZE", "2000"))
PARENT_CHUNK_OVERLAP = int(os.getenv("PARENT_CHUNK_OVERLAP", "200"))
CHILD_CHUNK_SIZE = int(os.getenv("CHILD_CHUNK_SIZE", "500"))
CHILD_CHUNK_OVERLAP = int(os.getenv("CHILD_CHUNK_OVERLAP", "100"))

# ── Retrieval ─────────────────────────────────────────────────────────────────
RETRIEVAL_THRESHOLD = float(os.getenv("RETRIEVAL_THRESHOLD", "0.65"))
RERANK_ENABLED = _env_bool("RERANK_ENABLED", "false")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
# Coarse vector search pulls this many points before threshold + rerank (higher = better recall).
VECTOR_SEARCH_LIMIT = int(os.getenv("VECTOR_SEARCH_LIMIT", "64"))

# ── Tiered context (v0.5 — OpenViking / Memex-inspired) ──────────────────────
TIERED_CONTEXT_ENABLED = _env_bool("TIERED_CONTEXT_ENABLED")
L0_MAX_TOKENS = int(os.getenv("L0_MAX_TOKENS", "100"))
L1_MAX_TOKENS = int(os.getenv("L1_MAX_TOKENS", "500"))

# ── Graph-augmented retrieval (v0.5) ─────────────────────────────────────────
GRAPH_RETRIEVAL_ENABLED = _env_bool("GRAPH_RETRIEVAL_ENABLED")
GRAPH_RETRIEVAL_WEIGHT = float(os.getenv("GRAPH_RETRIEVAL_WEIGHT", "0.3"))
MULTI_HOP_DEPTH = int(os.getenv("MULTI_HOP_DEPTH", "2"))
TEMPORAL_DECAY_HALFLIFE_DAYS = int(os.getenv("TEMPORAL_DECAY_HALFLIFE_DAYS", "30"))

# ── Trajectory & feedback (v0.6) ─────────────────────────────────────────────
OUTCOME_RETRIEVAL_BOOST = float(os.getenv("OUTCOME_RETRIEVAL_BOOST", "0.15"))
OUTCOME_RETRIEVAL_PENALTY = float(os.getenv("OUTCOME_RETRIEVAL_PENALTY", "0.1"))

# ── Hot cache (v0.8 — three-layer memory hierarchy) ─────────────────────────
HOT_CACHE_ENABLED = _env_bool("HOT_CACHE_ENABLED")
HOT_CACHE_MAX_PER_AGENT = int(os.getenv("HOT_CACHE_MAX_PER_AGENT", "128"))
HOT_CACHE_TTL_SECONDS = int(os.getenv("HOT_CACHE_TTL_SECONDS", "600"))

# ── Retrieval trajectory export (v0.8) ───────────────────────────────────────
TRAJECTORY_EXPORT_ENABLED = _env_bool("TRAJECTORY_EXPORT_ENABLED")
TRAJECTORY_EXPORT_MAX = int(os.getenv("TRAJECTORY_EXPORT_MAX", "200"))

# ── Observability (v0.9) ──────────────────────────────────────────────────────
METRICS_ENABLED = _env_bool("METRICS_ENABLED", "true")
DEFAULT_CONSISTENCY = os.getenv("DEFAULT_CONSISTENCY", "eventual")
# Slow-path warnings (0 = disabled). Logs one line when a step exceeds the threshold (ms).
SLOW_EMBED_MS = float(os.getenv("SLOW_EMBED_MS", "0"))
SLOW_QDRANT_MS = float(os.getenv("SLOW_QDRANT_MS", "0"))
SLOW_LLM_MS = float(os.getenv("SLOW_LLM_MS", "0"))
# Optional: append one JSON object per TTL invalidation run (see REFERENCE.md).
ARCHIVIST_INVALIDATION_EXPORT_PATH = os.getenv("ARCHIVIST_INVALIDATION_EXPORT_PATH", "").strip()

# ── Webhooks (v0.9) ─────────────────────────────────────────────────────────
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
WEBHOOK_TIMEOUT = float(os.getenv("WEBHOOK_TIMEOUT", "5"))
_raw_events = os.getenv("WEBHOOK_EVENTS", "").strip()
WEBHOOK_EVENTS: set[str] = set(e.strip() for e in _raw_events.split(",") if e.strip()) if _raw_events else set()

# ── Curator intelligence (v1.0) ─────────────────────────────────────────────
DEDUP_LLM_ENABLED = _env_bool("DEDUP_LLM_ENABLED")
DEDUP_LLM_THRESHOLD = float(os.getenv("DEDUP_LLM_THRESHOLD", "0.80"))
CURATOR_TIP_BUDGET = int(os.getenv("CURATOR_TIP_BUDGET", "20"))
CURATOR_MAX_PARALLEL = int(os.getenv("CURATOR_MAX_PARALLEL", "4"))
CURATOR_QUEUE_DRAIN_INTERVAL = int(os.getenv("CURATOR_QUEUE_DRAIN_INTERVAL", "30"))
HOTNESS_WEIGHT = float(os.getenv("HOTNESS_WEIGHT", "0.15"))
HOTNESS_HALFLIFE_DAYS = int(os.getenv("HOTNESS_HALFLIFE_DAYS", "7"))
IMPORTANCE_WEIGHT = float(os.getenv("IMPORTANCE_WEIGHT", "0.10"))
IMPORTANCE_FLOOR = float(os.getenv("IMPORTANCE_FLOOR", "0.3"))
IMPORTANCE_GRACE_DAYS = int(os.getenv("IMPORTANCE_GRACE_DAYS", "7"))

# ── Entity injection tuning (v1.8 — needle regression fix) ──────────────────
MAX_ENTITY_FACT_INJECTIONS = int(os.getenv("MAX_ENTITY_FACT_INJECTIONS", "15"))
ENTITY_SPECIFICITY_MAX_MENTIONS = int(os.getenv("ENTITY_SPECIFICITY_MAX_MENTIONS", "20"))

# ── Temporal intent & adaptive retrieval (v1.9 — recall improvements) ────────
TEMPORAL_INTENT_ENABLED = _env_bool("TEMPORAL_INTENT_ENABLED")
TEMPORAL_HISTORICAL_HALFLIFE_MULTIPLIER = float(os.getenv("TEMPORAL_HISTORICAL_HALFLIFE_MULTIPLIER", "10"))
BM25_RESCUE_ENABLED = _env_bool("BM25_RESCUE_ENABLED")
BM25_RESCUE_MIN_SCORE_RATIO = float(os.getenv("BM25_RESCUE_MIN_SCORE_RATIO", "0.6"))
BM25_RESCUE_MAX_SLOTS = int(os.getenv("BM25_RESCUE_MAX_SLOTS", "3"))
ADAPTIVE_VECTOR_LIMIT_ENABLED = _env_bool("ADAPTIVE_VECTOR_LIMIT_ENABLED")
ADAPTIVE_VECTOR_MIN_RESULTS = int(os.getenv("ADAPTIVE_VECTOR_MIN_RESULTS", "3"))
ADAPTIVE_VECTOR_LIMIT_MULTIPLIER = float(os.getenv("ADAPTIVE_VECTOR_LIMIT_MULTIPLIER", "3"))
CROSS_AGENT_MAX_SHARE = float(os.getenv("CROSS_AGENT_MAX_SHARE", "0.6"))

# ── Topic routing (v1.10 — keyword-based pre-vector filter) ──────────────
TOPIC_ROUTING_ENABLED = _env_bool("TOPIC_ROUTING_ENABLED")
TOPIC_MAP_PATH = os.getenv("TOPIC_MAP_PATH", "")

# ── Retention classes (v1.7 — "never forget" pinning) ────────────────────────
VALID_RETENTION_CLASSES = ("ephemeral", "standard", "durable", "permanent")
DURABLE_ENTITY_TYPES = frozenset({
    "person", "host", "server", "service", "credential",
    "organization", "cluster", "database", "network", "user",
})

# ── Curator ───────────────────────────────────────────────────────────────────
CURATOR_INTERVAL_MINUTES = int(os.getenv("CURATOR_INTERVAL_MINUTES", "30"))
ORPHAN_SWEEP_ENABLED = _env_bool("ORPHAN_SWEEP_ENABLED", "true")
ORPHAN_SWEEP_EVERY_N_CYCLES = int(os.getenv("ORPHAN_SWEEP_EVERY_N_CYCLES", "12"))

CURATOR_EXTRACT_PREFIXES: list[str] = [
    p.strip() for p in
    os.getenv("CURATOR_EXTRACT_PREFIXES", "agents/,memories/").split(",")
    if p.strip()
]
CURATOR_EXTRACT_SKIP_SEGMENTS: list[str] = [
    p.strip() for p in
    os.getenv("CURATOR_EXTRACT_SKIP_SEGMENTS", "skills,.cursor,.git").split(",")
    if p.strip()
]

# ── Memory awareness — Stage 0 query classification (v1.6) ───────────────────
QUERY_CLASSIFICATION_ENABLED = _env_bool("QUERY_CLASSIFICATION_ENABLED")

# ── BM25 / FTS5 hybrid search (v1.2) ─────────────────────────────────────────
BM25_ENABLED = _env_bool("BM25_ENABLED")
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.3"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.7"))

# ── Needle-finding: query expansion + dynamic threshold (v1.10) ──────────────
QUERY_EXPANSION_ENABLED = _env_bool("QUERY_EXPANSION_ENABLED", "true")
QUERY_EXPANSION_COUNT = int(os.getenv("QUERY_EXPANSION_COUNT", "3"))
QUERY_EXPANSION_MODEL = os.getenv("QUERY_EXPANSION_MODEL", "").strip()
DYNAMIC_THRESHOLD_ENABLED = _env_bool("DYNAMIC_THRESHOLD_ENABLED", "true")

# ── Contextual chunk augmentation (v1.10 — index-time enrichment) ────────────
CONTEXTUAL_AUGMENTATION_ENABLED = _env_bool("CONTEXTUAL_AUGMENTATION_ENABLED", "true")

# ── Reverse HyDE: write-time question generation (v2.0) ──────────────────────
REVERSE_HYDE_ENABLED = _env_bool("REVERSE_HYDE_ENABLED", "true")
REVERSE_HYDE_QUESTIONS_PER_CHUNK = int(os.getenv("REVERSE_HYDE_QUESTIONS_PER_CHUNK", "3"))

# ── Micro-chunk limits (v1.10 — prevent unbounded store latency) ─────────────
MAX_MICRO_CHUNKS_PER_MEMORY = int(os.getenv("MAX_MICRO_CHUNKS_PER_MEMORY", "5"))

# ── HNSW tuning (v1.10 — recall over speed) ─────────────────────────────────
QDRANT_HNSW_M = int(os.getenv("QDRANT_HNSW_M", "32"))
QDRANT_HNSW_EF_CONSTRUCT = int(os.getenv("QDRANT_HNSW_EF_CONSTRUCT", "256"))
QDRANT_SEARCH_EF = int(os.getenv("QDRANT_SEARCH_EF", "256"))

# ── Enterprise scaling (v1.10) ───────────────────────────────────────────────
NAMESPACE_SHARDING_ENABLED = _env_bool("NAMESPACE_SHARDING_ENABLED", "false")
SINGLE_COLLECTION_MODE = _env_bool("SINGLE_COLLECTION_MODE", "true")
CACHE_BACKEND = os.getenv("CACHE_BACKEND", "memory").lower()  # "memory" or "redis"
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
REDIS_KEY_PREFIX = os.getenv("REDIS_KEY_PREFIX", "archivist:")
LATENCY_BUDGET_MS = int(os.getenv("LATENCY_BUDGET_MS", "500"))

# ── Backup & restore (v1.10) ──────────────────────────────────────────────────
BACKUP_DIR = os.getenv("BACKUP_DIR", "/data/archivist/backups")
BACKUP_RETENTION_COUNT = int(os.getenv("BACKUP_RETENTION_COUNT", "5"))
BACKUP_INCLUDE_FILES = _env_bool("BACKUP_INCLUDE_FILES", "false")
BACKUP_PRE_PRUNE = _env_bool("BACKUP_PRE_PRUNE", "false")

# ── Context window management (v1.1) ─────────────────────────────────────────
DEFAULT_CONTEXT_BUDGET = int(os.getenv("DEFAULT_CONTEXT_BUDGET", "128000"))

# ── Journal exports (v1.5 — human-readable markdown alongside Qdrant) ────────
JOURNAL_ENABLED = _env_bool("JOURNAL_ENABLED")
JOURNAL_DIR = os.getenv("JOURNAL_DIR", "/data/archivist/journal")

# ── Server ────────────────────────────────────────────────────────────────────
MCP_PORT = int(os.getenv("MCP_PORT", "3100"))

# Optional: require `Authorization: Bearer <key>` or `X-API-Key` on all routes except /health
ARCHIVIST_API_KEY = os.getenv("ARCHIVIST_API_KEY", "").strip()

# Pre-store conflict check (vector similarity vs other agents' memories in same namespace)
CONFLICT_CHECK_ON_STORE = _env_bool("CONFLICT_CHECK_ON_STORE")
# When true, block the write when check_for_conflicts reports a conflict (unless force_skip_conflict_check)
CONFLICT_BLOCK_ON_STORE = _env_bool("CONFLICT_BLOCK_ON_STORE")

# ── Agent → team mapping (override via TEAM_MAP_PATH YAML file) ──────────────
TEAM_MAP_PATH = os.getenv("TEAM_MAP_PATH", "")


def _load_team_map() -> dict[str, str]:
    """Load agent→team mapping from YAML file or fall back to env/empty."""
    path = TEAM_MAP_PATH
    if path and os.path.isfile(path):
        try:
            with open(path) as f:
                data = yaml.safe_load(f)
            if isinstance(data, dict):
                return {str(k): str(v) for k, v in data.items()}
        except Exception:
            pass
    # Fall back to a simple env-var override (JSON string)
    raw = os.getenv("TEAM_MAP_JSON", "")
    if raw:
        import json
        try:
            return json.loads(raw)
        except Exception:
            pass
    return {}


TEAM_MAP: dict[str, str] = _load_team_map()


# ── Feature flag summary (logged at startup) ─────────────────────────────────
def _log_feature_flags() -> None:
    """Log all feature flags once at import time for operational visibility."""
    flags = {
        "BM25_ENABLED": BM25_ENABLED,
        "QUERY_EXPANSION_ENABLED": QUERY_EXPANSION_ENABLED,
        "DYNAMIC_THRESHOLD_ENABLED": DYNAMIC_THRESHOLD_ENABLED,
        "CONTEXTUAL_AUGMENTATION_ENABLED": CONTEXTUAL_AUGMENTATION_ENABLED,
        "REVERSE_HYDE_ENABLED": REVERSE_HYDE_ENABLED,
        "GRAPH_RETRIEVAL_ENABLED": GRAPH_RETRIEVAL_ENABLED,
        "TIERED_CONTEXT_ENABLED": TIERED_CONTEXT_ENABLED,
        "HOT_CACHE_ENABLED": HOT_CACHE_ENABLED,
        "RERANK_ENABLED": RERANK_ENABLED,
        "TOPIC_ROUTING_ENABLED": TOPIC_ROUTING_ENABLED,
        "TEMPORAL_INTENT_ENABLED": TEMPORAL_INTENT_ENABLED,
        "BM25_RESCUE_ENABLED": BM25_RESCUE_ENABLED,
        "ADAPTIVE_VECTOR_LIMIT_ENABLED": ADAPTIVE_VECTOR_LIMIT_ENABLED,
        "NAMESPACE_SHARDING_ENABLED": NAMESPACE_SHARDING_ENABLED,
        "SINGLE_COLLECTION_MODE": SINGLE_COLLECTION_MODE,
        "QUERY_CLASSIFICATION_ENABLED": QUERY_CLASSIFICATION_ENABLED,
        "CONFLICT_CHECK_ON_STORE": CONFLICT_CHECK_ON_STORE,
        "JOURNAL_ENABLED": JOURNAL_ENABLED,
        "METRICS_ENABLED": METRICS_ENABLED,
    }
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


_log_feature_flags()
