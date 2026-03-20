# Archivist

**Recursive memory service with knowledge graph and vector retrieval for AI agents.**

Archivist provides long-term memory for AI agent fleets. It combines vector search (Qdrant), a temporal knowledge graph (SQLite), and LLM-powered retrieval refinement into a single MCP-compatible service.

## Features

- **Multi-agent fleet search** — Query one agent, a list of agents (`agent_ids`), or the whole fleet; RBAC enforces who can read whose namespaces; results are deduplicated and merged before rerank and synthesis
- **RLM Recursive Retrieval** — Wide vector recall (`VECTOR_SEARCH_LIMIT`) → dedupe → threshold → optional cross-encoder rerank → parent enrichment → LLM refinement → synthesis with attribution when multiple agents contributed
- **Temporal Knowledge Graph** — SQLite-backed entity/relationship tracking with automatic extraction from agent notes
- **Hierarchical Chunking** — Parent-child chunk relationships for richer retrieval context
- **Namespace RBAC** — File-based access control for multi-agent/multi-team deployments
- **Autonomous Curation** — Background curator extracts entities, relationships, and facts from new files
- **MCP Protocol** — Exposes tools via HTTP SSE (Model Context Protocol) for agent integration
- **Audit Trail** — Immutable logging of all memory operations
- **Memory Merging** — CRDT-style merge strategies (latest, concat, semantic, manual review)
- **TTL-based Expiry** — Configurable retention per namespace with importance-based override

## Quick Start

### Prerequisites

- Docker & Docker Compose
- An OpenAI-compatible LLM API (OpenAI, LiteLLM, Ollama, vLLM, etc.)
- An OpenAI-compatible embeddings API

### 1. Clone and configure

```bash
git clone https://github.com/AHEAD-Labs/ai-archivist-oss.git
cd archivist
cp .env.example .env
# Edit .env with your LLM/embedding API details
```

### 2. Start services

```bash
docker compose up -d
```

This starts:
- **Archivist** on port `3100`
- **Qdrant** on port `6333`

### 3. Verify

```bash
curl http://localhost:3100/health
# {"status": "ok", "service": "archivist", "version": "0.3.0"}
```

### 4. Connect an MCP client

Point your MCP client at: `http://localhost:3100/mcp/sse`

## Architecture

```
┌─────────────────────────────────────────────────┐
│                   MCP Clients                    │
│              (AI agents, tools)                  │
└──────────────────────┬──────────────────────────┘
                       │ HTTP SSE
┌──────────────────────▼──────────────────────────┐
│              Archivist MCP Server                │
│                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────┐ │
│  │ RLM         │  │ Knowledge   │  │ RBAC    │ │
│  │ Retriever   │  │ Graph       │  │ Middleware│ │
│  │             │  │ (SQLite)    │  │         │ │
│  │ • Threshold │  │             │  │         │ │
│  │ • Rerank    │  │ • Entities  │  │         │ │
│  │ • Refine    │  │ • Relations │  │         │ │
│  │ • Synthesize│  │ • Facts     │  │         │ │
│  └──────┬──────┘  └─────────────┘  └─────────┘ │
│         │                                        │
│  ┌──────▼──────┐  ┌─────────────┐               │
│  │ Embeddings  │  │ Curator     │               │
│  │ (OpenAI API)│  │ (Background)│               │
│  └──────┬──────┘  └─────────────┘               │
│         │                                        │
└─────────┼────────────────────────────────────────┘
          │
┌─────────▼──────────┐  ┌────────────────────────┐
│   Qdrant           │  │   File System          │
│   (Vector Store)   │  │   (MEMORY_ROOT)        │
└────────────────────┘  └────────────────────────┘
```

## MCP Tools

| Tool | Description |
|------|-------------|
| `archivist_search` | Semantic search with RLM pipeline; optional `min_score` per call overrides `RETRIEVAL_THRESHOLD` |
| `archivist_recall` | Graph-based entity/relationship lookup |
| `archivist_store` | Explicitly store a memory with entity extraction |
| `archivist_timeline` | Chronological timeline of memories about a topic |
| `archivist_insights` | Cross-agent insights for a topic |
| `archivist_namespaces` | List accessible memory namespaces |
| `archivist_audit_trail` | View audit log for memory operations |
| `archivist_merge` | Merge conflicting memories |

## Configuration

All configuration is via environment variables. See [`.env.example`](.env.example) for the full list.

### Key Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_URL` | `http://localhost:4000` | OpenAI-compatible chat API |
| `LLM_MODEL` | `gpt-4o-mini` | Model for refinement/synthesis |
| `LLM_API_KEY` | *(empty)* | API key for LLM |
| `EMBED_URL` | `$LLM_URL` | OpenAI-compatible embeddings API |
| `EMBED_MODEL` | `text-embedding-v3` | Embedding model name |
| `VECTOR_DIM` | `1024` | Embedding vector dimension |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `MEMORY_ROOT` | `/data/memories` | Directory to watch for .md files |
| `RETRIEVAL_THRESHOLD` | `0.65` | Minimum vector score for retrieval |
| `VECTOR_SEARCH_LIMIT` | `64` | Coarse vector hits to pull before threshold/rerank (higher recall) |
| `RERANK_ENABLED` | `false` | Enable cross-encoder reranking |
| `RERANK_MODEL` | `BAAI/bge-reranker-v2-m3` | Reranker model |

Cross-encoder reranking is **off** in the default Docker image. To enable it, install optional dependencies (uncomment `sentence-transformers` / `torch` in `requirements.txt` or extend the Dockerfile), set `RERANK_ENABLED=true`, and rebuild.

### RBAC / Namespaces

Create a `namespaces.yaml` (see [`namespaces.yaml.example`](namespaces.yaml.example)) to define per-namespace read/write ACLs. Without it, Archivist runs in **permissive mode** (all agents can read/write everything) — suitable for single-user setups.

### Agent Team Mapping

For multi-agent setups, create a `team_map.yaml` (see [`team_map.yaml.example`](team_map.yaml.example)) and set `TEAM_MAP_PATH` to its location. This maps agent IDs to teams for metadata tagging.

## Phase 1 Improvements (v0.2.0+) / v0.3.0 fleet search

### Retrieval Threshold
Results below `RETRIEVAL_THRESHOLD` (default 0.65) are filtered out before LLM refinement, reducing noise and saving LLM tokens.

### Cross-Encoder Reranking
When `RERANK_ENABLED=true`, a cross-encoder model re-scores vector search results for higher precision. Requires `sentence-transformers` (uncomment in `requirements.txt`).

### Parent-Child Chunking
Documents are split into large parent chunks (2000 chars) containing smaller child chunks (500 chars). Search matches on specific child chunks; retrieval enriches them with full parent context for better LLM refinement.

### Multi-agent search (`archivist_search`)

- **Fleet-wide** — Omit both `agent_id` and `agent_ids` to search all indexed memories (subject to namespace filter if set).
- **One agent** — Set `agent_id` (caller must have read access to that agent’s default namespace unless running in permissive RBAC mode).
- **Several agents** — Set `agent_ids` to `["alice","bob","carol"]`. Use `caller_agent_id` for the invoking agent so RBAC can allow or deny each target. Partial allow lists are supported: only permitted agents are searched.

### Upgrading indexes (flat → hierarchical)

v0.2.0 uses hierarchical point IDs and payloads (`parent_id`, `is_parent`) that differ from earlier flat-only indexes. **Do not mix** old and new point shapes in one collection.

1. Point `QDRANT_COLLECTION` at a **new** collection name (e.g. `archivist_memories_v2`), or delete the existing collection.
2. Restart Archivist (or call your usual full reindex path) so all `.md` files under `MEMORY_ROOT` are re-ingested.

## Development

```bash
pip install -r requirements.txt pytest
python -m pytest tests/ -v
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for conventions.

## Sharing this repo

- Copy [`.env.example`](.env.example) to `.env` and set LLM/embed endpoints for your team.
- Run `docker compose up --build` for a local demo; use `python -m pytest tests/` to verify a checkout.
- CI runs on push/PR via [`.github/workflows/ci.yml`](.github/workflows/ci.yml).

### First-time Git publish

From the `archivist-oss` directory (or after copying it to a standalone clone):

```bash
git init
git add -A
git commit -m "Archivist OSS v0.3.0"
git branch -M main
git remote add origin git@github.com:AHEAD-Labs/ai-archivist-oss.git
git tag -a v0.3.0 -m "Fleet multi-agent search, dedupe, wide vector recall"
git push -u origin main --tags
```

Replace `YOUR_ORG/archivist` with your organization and repository name.

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/mcp/sse` | GET | MCP SSE connection |
| `/mcp/messages/` | POST | MCP message handler |
| `/admin/invalidate` | GET/POST | Trigger TTL-based memory expiry |

## License

Apache License 2.0 — see [LICENSE](LICENSE).
