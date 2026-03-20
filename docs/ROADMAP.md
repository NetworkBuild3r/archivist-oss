# Archivist Roadmap

## Current: v1.0.0 (Phase 7 — Memory Intelligence Layer)

### ✅ v1.0.0 — Memory Intelligence Layer (Curator)
- **Write-ahead curator queue** — Background `curator_queue` SQLite table stages dedup merges, archival, tip consolidation, and hotness updates. Periodic drain loop applies ops during idle, avoiding lock contention on the hot path.
- **LLM-adjudicated memory dedup** — On-write similarity check triggers LLM adjudication above configurable threshold (`DEDUP_LLM_THRESHOLD`). LLM decides skip/create/merge/delete for each existing match. Below threshold, fast cosine-only path preserved.
- **Tip consolidation pipeline** — Batch clustering of similar tips by embedding similarity, LLM merge into canonical tips with negative examples, budget cap (`CURATOR_TIP_BUDGET`), originals archived for provenance.
- **`archivist_compress` MCP tool** — Mid-session archival: agents call compress to archive memory blocks and get compact summaries. Originals kept but excluded from default search.
- **Context-status signaling** — Every retrieval response includes `context_status` dict with token estimates, budget usage, and compression hints for agents to manage context windows.
- **Hotness scoring** — Frequency × recency signal: `sigmoid(log1p(retrieval_count)) * temporal_decay`. Batch aggregation from `retrieval_logs`, blended into RLM pipeline after temporal decay.
- **Skill relation graph** — `skill_relations` table tracks similar_to, depend_on, compose_with, replaced_by. `archivist_skill_relate` and `archivist_skill_dependencies` MCP tools. `archivist_skill_health` extended with substitutes.
- **Curator agent persona** — System prompt for NemoClaw Curator agent (`prompts/curator.md`): automated health checks, contradiction resolution, compression, and reporting.
- **Curator metrics** — 5 new Prometheus metrics: queue depth, dedup decisions (labeled), tip consolidations, LLM calls, drain duration.

### ✅ v0.9.0 — Observability & operational excellence
- **Prometheus `/metrics`** — Pure-Python text exposition endpoint with counters (search, store, conflict, cache hit/miss, webhook fire/fail, skill events, LLM calls), histograms (search duration, LLM duration), and gauges. Scrape-ready for any Prometheus-compatible system.
- **Webhooks** — Fire-and-forget HTTP POSTs on key events: `memory_store`, `memory_conflict`, `skill_event`. Configure via `WEBHOOK_URL` and optional `WEBHOOK_EVENTS` filter. Failures are counted in metrics, never block the caller.
- **Health dashboard** — New `archivist_health_dashboard` MCP tool and `GET /admin/dashboard` endpoint. Single-pane aggregation: Qdrant point count, stale %, conflict rate, retrieval stats (cache hit rate, avg latency), skill health overview (degraded count, success rate), cache status.
- **Batch heuristic** — New `archivist_batch_heuristic` MCP tool and `GET /admin/dashboard?batch=true`. Recommends batch size (1–10 scale) from memory health signals: conflict rate, stale %, cache hit rate, degraded skills. Inspired by Batch Size Gravity.
- **Metrics instrumentation** — RLM pipeline, store, conflict, skill events all emit metrics automatically. Search duration tracked as histogram.

### ✅ v0.8.0 — Memory architecture formalization
- **Three-layer memory hierarchy** — session/ephemeral → per-agent hot cache → long-term (Qdrant + SQLite). Hot cache is LRU with configurable TTL and max entries per agent.
- **Per-agent hot cache** — Repeated queries within a session skip the full RLM pipeline. Cache is automatically invalidated when `archivist_store` writes to the same namespace.
- **Manual cache control** — New `archivist_cache_invalidate` tool for bulk invalidation (by namespace, agent, or all). `archivist_cache_stats` exposes per-agent entry counts and TTL status.
- **`archivist://` URI scheme** — Addressable references to memories, entities, namespaces, and skills. `archivist_store` and `archivist_deref` responses include `uri` fields. New `archivist_resolve_uri` tool resolves any archivist:// URI to its resource.
- **Retrieval trajectory logging** — Every search pipeline execution is recorded in a `retrieval_logs` SQLite table with full trace (stages, durations, cache hits). Available via `archivist_retrieval_logs` MCP tool and `GET /admin/retrieval-logs` REST endpoint.
- **Retrieval analytics** — Aggregate stats: cache hit rate, avg/min/max duration, top agents — via `archivist_retrieval_logs` with `stats_only=true`.
- **Consistency config** — `DEFAULT_CONSISTENCY` env var (eventual / session / strong) documents and sets the baseline consistency semantics for the deployment.

### ✅ v0.7.0 — Skill registry & dual-stream knowledge
- **Skill registry** — `skills`, `skill_versions`, `skill_lessons`, `skill_events` SQLite tables for MCP tool tracking
- **Four new skill tools** — `archivist_register_skill`, `archivist_skill_event`, `archivist_skill_lesson`, `archivist_skill_health`
- **Version & lessons tracking** — Per-skill version history, failure mode docs, workarounds, best practices
- **Experience vs skill routing** — `memory_type` (experience/skill/general) on store + search filter

### ✅ v0.6.0 — Trajectory learning & feedback loops
- **Trajectory logging** — New `archivist_log_trajectory` tool records task + actions + outcome; auto-extracts strategy/recovery/optimization tips via LLM post-mortem
- **Decision attribution** — LLM analysis links outcomes to the specific memories that informed decisions; stored in `memory_outcomes` table
- **Outcome-aware retrieval** — Memories linked to successful trajectories get score boosts; failure-linked memories get penalties; configurable via `OUTCOME_RETRIEVAL_BOOST` / `OUTCOME_RETRIEVAL_PENALTY`
- **Annotations** — New `archivist_annotate` tool for quality notes, corrections, staleness markers, and verified status on memory points
- **Ratings** — New `archivist_rate` tool for +1/-1 voting with aggregate summaries; feeds audit log
- **Tips retrieval** — New `archivist_tips` tool surfaces strategy/recovery/optimization learnings from past trajectories
- **Session-end summarization** — New `archivist_session_end` tool aggregates a session's trajectories into a durable summary memory

### ✅ v0.5.0 — Tiered context & graph-augmented retrieval
- **L0/L1/L2 tiered context** — Auto-generated summaries on ingest: L0 (~1 sentence), L1 (~2–4 sentences), L2 (full text). `tier` param on search controls detail level.
- **Context budget** — `max_tokens` on `archivist_search` caps approximate token usage; combine with `tier=l0` for compact pre-message injection.
- **Progressive dereference** — New `archivist_deref` tool fetches full L2 text by point ID for drill-down after compact search.
- **Compressed index** — New `archivist_index` tool returns a ~500-token navigational TOC of entities/topics per namespace.
- **Hybrid vector+graph retrieval** — Entity mentions in query trigger multi-hop KG traversal; graph context merged into vector results with configurable weight.
- **Temporal decay** — Exponential score decay based on document age (`TEMPORAL_DECAY_HALFLIFE_DAYS`).
- **Date range filters** — `date_from` / `date_to` on `archivist_search` for Qdrant payload range filtering.
- **Multi-hop graph walks** — Follows relationship chains up to `MULTI_HOP_DEPTH` hops; facts and relationships from each hop included in retrieval context.
- **Contradiction surfacing** — New `archivist_contradictions` tool detects opposing facts from different agents about the same entity.

### ✅ v0.4.0 — Critical foundations
- **Conflict check on store** — Optional vector similarity vs other agents (`CONFLICT_CHECK_ON_STORE` / `CONFLICT_BLOCK_ON_STORE`); `force_skip_conflict_check` on `archivist_store`
- **Explicit memory IDs** — `archivist_store` uses UUID point IDs (no MD5 collisions)
- **Transport auth** — Optional `ARCHIVIST_API_KEY` (`Authorization: Bearer` or `X-API-Key`); `/health` stays open for probes
- **Graph/timeline/insights RBAC** — `caller_agent_id` + fact/rel filtering by agent; timeline/insights filtered like search
- **SQLite write serialization** — `GRAPH_WRITE_LOCK` on graph writes, audit, versioning, curator decay
- **LLM retries** — Backoff on transient failures (same pattern as embeddings)
- **Retrieval trace** — `retrieval_trace` object on `archivist_search` responses (pipeline counts and flags)

### ✅ Multi-agent fleet search (v0.3.x)
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

## Future considerations
- Migrate SSE → Streamable HTTP transport for lower latency
- PostgreSQL option for graph/metadata at scale
- Streaming partial search results
- Local embedding models (sentence-transformers)
- Memory compaction (periodic LLM summarization)
- Multi-collection Qdrant support
- Web UI dashboard

---

## Non-Goals (Out of Scope)

- **Real-time chat** — Archivist is memory, not conversation
- **File storage** — Archivist indexes files, doesn't host them
- **Agent orchestration** — Memory service only; agents managed externally
