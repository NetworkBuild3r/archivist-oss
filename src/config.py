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
