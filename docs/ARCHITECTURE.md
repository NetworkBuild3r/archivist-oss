# Archivist Architecture

## Overview

Archivist is a memory service for AI agent fleets. It combines three storage backends:

1. **Qdrant** — Vector store for semantic search over chunked documents
2. **SQLite** — Temporal knowledge graph for entities, relationships, and facts
3. **File System** — Source of truth (markdown files watched for changes)

## Data Flow

### Ingestion

```
Markdown Files (MEMORY_ROOT)
    │
    ├──→ File Watcher (watchfiles)
    │        │
    │        ▼
    │    Chunker (indexer.py)
    │        │
    │        ├──→ Flat chunks (legacy mode)
    │        │
    │        └──→ Hierarchical chunks (Phase 1)
    │                 │
    │                 ├── Parent chunks (2000 chars)
    │                 │       │
    │                 │       └── Child chunks (500 chars)
    │                 │             with parent_id reference
    │                 │
    │                 ▼
    │            Embedding API
    │                 │
    │                 ▼
    │           Qdrant Upsert
    │
    └──→ Curator (background loop)
             │
             ▼
         LLM Extraction
             │
             ├──→ Entities → SQLite entities table
             ├──→ Facts → SQLite facts table
             └──→ Relationships → SQLite relationships table
```

### Retrieval (RLM Pipeline)

```
Query
  │
  ▼
Stage 1: Coarse Vector Search (Qdrant, wide limit VECTOR_SEARCH_LIMIT; optional MatchAny on agent_ids)
  │
  ▼
Stage 1b: Dedupe (memory_fusion — same file/chunk/text)
  │
  ▼
Stage 2: Threshold Filter (score >= RETRIEVAL_THRESHOLD or per-call min_score)
  │
  ▼
Stage 3: Cross-Encoder Rerank (optional)
  │
  ▼
Stage 4: Parent-Child Enrichment
  │        (fetch parent context for child matches)
  │
  ▼
Stage 5: LLM Refinement
  │        (per-chunk relevance extraction)
  │
  ▼
Stage 6: LLM Synthesis
  │        (combine extractions into answer)
  │
  ▼
Response with sources
```

## Module Map

| Module | Responsibility |
|--------|---------------|
| `main.py` | Entry point, Starlette app, startup tasks |
| `mcp_server.py` | MCP tool definitions and handlers |
| `rlm_retriever.py` | RLM recursive retrieval pipeline |
| `reranker.py` | Cross-encoder reranking (optional) |
| `indexer.py` | File chunking, embedding, Qdrant indexing |
| `graph.py` | SQLite schema, entity/relationship CRUD |
| `embeddings.py` | Embedding API client |
| `llm.py` | LLM API client |
| `config.py` | Environment variable configuration |
| `rbac.py` | Namespace access control |
| `curator.py` | Background knowledge extraction |
| `audit.py` | Immutable audit logging |
| `merge.py` | Memory merge strategies |
| `versioning.py` | Memory version tracking |
| `conflict_detection.py` | Pre-write conflict detection |

## Storage Schema

### Qdrant Payload Fields

| Field | Type | Description |
|-------|------|-------------|
| `agent_id` | keyword | Source agent |
| `file_path` | keyword | Relative path from MEMORY_ROOT |
| `file_type` | keyword | daily, durable, system, explicit, merged |
| `team` | keyword | Agent's team |
| `date` | keyword | ISO date |
| `namespace` | keyword | RBAC namespace |
| `text` | text | Chunk content |
| `chunk_index` | integer | Position in source file |
| `parent_id` | keyword | Parent chunk ID (hierarchical) |
| `is_parent` | bool | Whether this is a parent chunk |
| `version` | integer | Monotonic version |
| `importance_score` | float | 0.0–1.0 retention score |
| `ttl_expires_at` | integer | Unix timestamp for expiry |
| `checksum` | keyword | Content hash for dedup |

### SQLite Tables

- **entities** — Named entities with type, mention count, first/last seen
- **relationships** — Typed edges between entities with evidence and confidence
- **facts** — Text facts linked to entities, with active/superseded status
- **curator_state** — Key-value store for curator bookkeeping
- **audit_log** — Immutable log of all memory operations
- **memory_versions** — Version history per memory ID
