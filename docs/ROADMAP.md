# Archivist Roadmap

## Current: v0.3.0 (Phase 1 + fleet search)

### ✅ Multi-agent fleet search
- `agent_ids` + `caller_agent_id` on `archivist_search` with namespace RBAC
- Qdrant `MatchAny` filter, `memory_fusion` dedupe before threshold/rerank
- Wide coarse recall via `VECTOR_SEARCH_LIMIT` (default 64)
- Multi-agent synthesis prompts attribute facts to sources

### ✅ Retrieval Threshold
- Configurable `RETRIEVAL_THRESHOLD` (default 0.65)
- Results below threshold filtered before LLM refinement
- Saves tokens and reduces noise

### ✅ Cross-Encoder Reranking
- Optional `RERANK_ENABLED` flag
- Uses `BAAI/bge-reranker-v2-m3` by default
- Graceful degradation when model unavailable
- Configurable top-K after rerank

### ✅ Parent-Child Chunking
- Hierarchical chunk creation (parent 2000 / child 500 chars)
- Child chunks reference parent via `parent_id`
- Retrieval enriches child matches with parent context
- Configurable sizes via environment variables

---

## Phase 2: Graph-Augmented Retrieval

### Hybrid Search
- Combine vector similarity with knowledge graph traversal
- Entity mentions in query trigger graph lookups alongside vector search
- Weight and merge results from both sources

### Temporal Awareness
- Date range filters in vector search
- "What changed since X" queries using graph timestamps
- Automatic decay weighting (older = lower relevance)

### Contradiction Detection
- Automatic detection of conflicting facts in the knowledge graph
- LLM-based resolution suggestions
- Alert mechanism for human review

### Multi-Hop Retrieval
- Follow relationship chains: A → B → C
- Configurable hop depth (default 2)
- Evidence aggregation across hops

---

## Phase 3: Advanced Features

### Streaming MCP
- Migrate from SSE to Streamable HTTP transport
- Lower latency for long synthesis operations
- Progress updates during multi-stage retrieval

### Embedding Model Flexibility
- Support for local embedding models (sentence-transformers)
- Automatic dimension detection
- Mixed-model collections (migration tooling)

### Memory Compaction
- Periodic LLM-based summarization of old daily notes
- Replace N daily chunks with 1 summary chunk
- Preserve original in cold storage

### Multi-Collection Support
- Per-namespace Qdrant collections
- Cross-collection search
- Collection-level configuration (dimensions, distance metrics)

### Webhooks / Event Bus
- Emit events on memory operations (create, update, delete, merge)
- Webhook delivery for external integrations
- Event sourcing pattern for full replay

### UI Dashboard
- Web UI for browsing memories, entities, relationships
- Search interface with filter controls
- Audit trail viewer
- Merge conflict resolution UI

---

## Non-Goals (Out of Scope)

- **Real-time chat** — Archivist is memory, not conversation
- **File storage** — Archivist indexes files, doesn't host them
- **Agent orchestration** — Memory service only; agents managed externally
