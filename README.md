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
git clone https://github.com/NetworkBuild3r/archivist-oss.git
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
# {"status": "ok", "service": "archivist", "version": "1.0.0"}
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
| `archivist_compress` | Archive memory blocks and return compact summaries |
| `archivist_skill_relate` | Record relations between skills (similar, depends, composes, replaces) |
| `archivist_skill_dependencies` | Get skill dependency/relation graph |

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

## Phase 1–7 Improvements (v0.2.0 → v1.0.0)

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

### Git on a UNC path (`\\server\share\...`)

If Git reports **fatal: detected dubious ownership** (common for clones on SMB/NAS), register this folder **once**. Pick one:

**A — Batch helper** (works even when PowerShell blocks unsigned `.ps1`):

```bat
scripts\trust-unc-repo.cmd
```

**B — PowerShell with bypass** (if you prefer the `.ps1`):

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\scripts\trust-unc-repo.ps1
```

**C — Git only** (no scripts; replace the path with yours if different):

```powershell
git config --global --add safe.directory '%(prefix)///192.168.11.102/k8s-argocd/openclaw/agents/nova/archivist-oss'
```

After that, `git add`, `git commit`, and pushes work normally.

To push to GitHub without running any `.ps1`, use:

```bat
scripts\push-networkbuild3r.cmd
```

Or from any cwd (UNC-safe; uses `git -c safe.directory=*`):

```bash
python scripts/sync-public.py
```

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
git remote add origin git@github.com:NetworkBuild3r/archivist-oss.git
git tag -a v0.3.0 -m "Fleet multi-agent search, dedupe, wide vector recall"
git push -u origin main --tags
```

Replace `YOUR_ORG/archivist` with your organization and repository name.

### Private team repo + public GitHub

If you maintain an **internal** clone for the team and mirror **public** OSS to GitHub, use two remotes and see [docs/REMOTES.md](docs/REMOTES.md). Quick push to both:

```powershell
.\scripts\publish-remotes.ps1 `
  -InternalUrl "https://your-gitlab.example.com/org/archivist-oss.git" `
  -PublicUrl "git@github.com:NetworkBuild3r/archivist-oss.git" `
  -PublicRemoteName origin `
  -WithTags
```

(`-PublicRemoteName origin` if your GitHub remote is still named `origin`.)

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics (text exposition) |
| `/mcp/sse` | GET | MCP SSE connection |
| `/mcp/messages/` | POST | MCP message handler |
| `/admin/invalidate` | GET/POST | Trigger TTL-based memory expiry |
| `/admin/retrieval-logs` | GET | Export retrieval pipeline logs and stats |
| `/admin/dashboard` | GET | Health dashboard (add `?batch=true` for batch heuristic) |

### Skill Registry (v0.7.0)

Archivist tracks the operational health of skills (MCP tools) used by agents:

```text
archivist_register_skill → catalog a new tool with provider, version, endpoint
archivist_skill_event    → log invocation outcome (success/partial/failure)
archivist_skill_lesson   → record failure modes, workarounds, best practices
archivist_skill_health   → get success rate, recent failures, health grade
```

Tag stored memories with `memory_type` (experience / skill / general) and filter searches by type.

### Memory Hierarchy & URIs (v0.8.0)

Three-layer architecture: session/ephemeral → per-agent hot cache → long-term (Qdrant + SQLite).

```text
archivist_resolve_uri       → resolve archivist://{ns}/{type}/{id} to its resource
archivist_retrieval_logs    → export/analyze pipeline execution traces
archivist_cache_stats       → hot cache health per agent
archivist_cache_invalidate  → manual eviction by namespace, agent, or all
```

URIs follow the format `archivist://namespace/memory|entity|namespace|skill/id` and are included in `archivist_store` and `archivist_deref` responses.

### Observability (v0.9.0)

```text
GET /metrics                       → Prometheus scrape endpoint
archivist_health_dashboard         → memory, retrieval, skill, cache health in one view
archivist_batch_heuristic          → recommended batch size from health signals (1-10)
GET /admin/dashboard               → same data as REST (add ?batch=true for heuristic)
```

Webhooks fire on `memory_store`, `memory_conflict`, and `skill_event` — configure `WEBHOOK_URL` in your environment.

### Memory Intelligence Layer (v1.0.0)

Active curation pipeline that maintains memory quality automatically:

```text
archivist_compress      → archive old memories, get compact summaries
archivist_skill_relate  → record skill relationships (substitutes, dependencies)
archivist_skill_dependencies → get skill relation graph
```

**Curator queue:** Background write-ahead queue stages dedup merges, archival, and tip consolidation. No lock contention on the hot write path.

**LLM-adjudicated dedup:** Stores above similarity threshold trigger LLM decision (skip/create/merge/delete) instead of just blocking.

**Context-status signaling:** Every search response includes token estimates and compression hints so agents can manage their context windows.

**Curator agent persona:** System prompt at `prompts/curator.md` defines a memory librarian persona for NemoClaw/OpenClaw deployments that performs scheduled health checks, contradiction resolution, and stale memory compression.

## License

Apache License 2.0 — see [LICENSE](LICENSE).
