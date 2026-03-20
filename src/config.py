"""Archivist configuration loaded from environment variables.

All values have sensible defaults for local / docker-compose development.
Override via .env or environment variables in production.
"""

import os
import yaml

# ── Vector store ──────────────────────────────────────────────────────────────
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "archivist_memories")
VECTOR_DIM = int(os.getenv("VECTOR_DIM", "1024"))

# ── Embedding model (OpenAI-compatible API) ───────────────────────────────────
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-v3")
EMBED_URL = os.getenv("EMBED_URL", os.getenv("LLM_URL", "http://localhost:4000"))

# ── LLM (OpenAI-compatible chat/completions API) ─────────────────────────────
LLM_URL = os.getenv("LLM_URL", "http://localhost:4000")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")

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
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "false").lower() in ("true", "1", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "10"))
# Coarse vector search pulls this many points before threshold + rerank (higher = better recall).
VECTOR_SEARCH_LIMIT = int(os.getenv("VECTOR_SEARCH_LIMIT", "64"))

# ── Tiered context (v0.5 — OpenViking / Memex-inspired) ──────────────────────
TIERED_CONTEXT_ENABLED = os.getenv("TIERED_CONTEXT_ENABLED", "true").lower() in ("true", "1", "yes")
L0_MAX_TOKENS = int(os.getenv("L0_MAX_TOKENS", "100"))
L1_MAX_TOKENS = int(os.getenv("L1_MAX_TOKENS", "500"))

# ── Graph-augmented retrieval (v0.5) ─────────────────────────────────────────
GRAPH_RETRIEVAL_ENABLED = os.getenv("GRAPH_RETRIEVAL_ENABLED", "true").lower() in ("true", "1", "yes")
GRAPH_RETRIEVAL_WEIGHT = float(os.getenv("GRAPH_RETRIEVAL_WEIGHT", "0.3"))
MULTI_HOP_DEPTH = int(os.getenv("MULTI_HOP_DEPTH", "2"))
TEMPORAL_DECAY_HALFLIFE_DAYS = int(os.getenv("TEMPORAL_DECAY_HALFLIFE_DAYS", "30"))

# ── Trajectory & feedback (v0.6) ─────────────────────────────────────────────
OUTCOME_RETRIEVAL_BOOST = float(os.getenv("OUTCOME_RETRIEVAL_BOOST", "0.15"))
OUTCOME_RETRIEVAL_PENALTY = float(os.getenv("OUTCOME_RETRIEVAL_PENALTY", "0.1"))

# ── Hot cache (v0.8 — three-layer memory hierarchy) ─────────────────────────
HOT_CACHE_ENABLED = os.getenv("HOT_CACHE_ENABLED", "true").lower() in ("true", "1", "yes")
HOT_CACHE_MAX_PER_AGENT = int(os.getenv("HOT_CACHE_MAX_PER_AGENT", "128"))
HOT_CACHE_TTL_SECONDS = int(os.getenv("HOT_CACHE_TTL_SECONDS", "600"))

# ── Consistency (v0.8) ───────────────────────────────────────────────────────
DEFAULT_CONSISTENCY = os.getenv("DEFAULT_CONSISTENCY", "eventual")

# ── Retrieval trajectory export (v0.8) ───────────────────────────────────────
TRAJECTORY_EXPORT_ENABLED = os.getenv("TRAJECTORY_EXPORT_ENABLED", "true").lower() in ("true", "1", "yes")
TRAJECTORY_EXPORT_MAX = int(os.getenv("TRAJECTORY_EXPORT_MAX", "200"))

# ── Webhooks (v0.9) ─────────────────────────────────────────────────────────
WEBHOOK_URL = os.getenv("WEBHOOK_URL", "").strip()
WEBHOOK_TIMEOUT = float(os.getenv("WEBHOOK_TIMEOUT", "5"))
_raw_events = os.getenv("WEBHOOK_EVENTS", "").strip()
WEBHOOK_EVENTS: set[str] = set(e.strip() for e in _raw_events.split(",") if e.strip()) if _raw_events else set()

# ── Metrics (v0.9) ──────────────────────────────────────────────────────────
METRICS_ENABLED = os.getenv("METRICS_ENABLED", "true").lower() in ("true", "1", "yes")

# ── Curator intelligence (v1.0) ─────────────────────────────────────────────
DEDUP_LLM_ENABLED = os.getenv("DEDUP_LLM_ENABLED", "true").lower() in ("true", "1", "yes")
DEDUP_LLM_THRESHOLD = float(os.getenv("DEDUP_LLM_THRESHOLD", "0.80"))
CURATOR_TIP_BUDGET = int(os.getenv("CURATOR_TIP_BUDGET", "20"))
CURATOR_QUEUE_DRAIN_INTERVAL = int(os.getenv("CURATOR_QUEUE_DRAIN_INTERVAL", "30"))
HOTNESS_WEIGHT = float(os.getenv("HOTNESS_WEIGHT", "0.15"))
HOTNESS_HALFLIFE_DAYS = int(os.getenv("HOTNESS_HALFLIFE_DAYS", "7"))

# ── Curator ───────────────────────────────────────────────────────────────────
CURATOR_INTERVAL_MINUTES = int(os.getenv("CURATOR_INTERVAL_MINUTES", "30"))

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

# ── Server ────────────────────────────────────────────────────────────────────
MCP_PORT = int(os.getenv("MCP_PORT", "3100"))

# Optional: require `Authorization: Bearer <key>` or `X-API-Key` on all routes except /health
ARCHIVIST_API_KEY = os.getenv("ARCHIVIST_API_KEY", "").strip()

# Pre-store conflict check (vector similarity vs other agents' memories in same namespace)
CONFLICT_CHECK_ON_STORE = os.getenv("CONFLICT_CHECK_ON_STORE", "true").lower() in ("true", "1", "yes")
# When true, block the write when check_for_conflicts reports a conflict (unless force_skip_conflict_check)
CONFLICT_BLOCK_ON_STORE = os.getenv("CONFLICT_BLOCK_ON_STORE", "true").lower() in ("true", "1", "yes")

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
