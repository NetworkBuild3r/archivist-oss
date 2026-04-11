# Archivist Roadmap

## Current: v1.11.0 (Needle retrieval v2 + audit hardening)

### ✅ v1.11.0 — Needle retrieval v2, delete cascade hardening, observability

**Breaking change:** `delete_memory_complete()` now enumerates child Qdrant points via `scroll()` before deletion. Callers that mock `qdrant_client` in tests must add `client.scroll.return_value = ([], None)` to their fixtures.

#### Needle retrieval v2 (v1.10 features, production-hardened in v1.11)
- **Deterministic needle registry** — O(1) lookup for IPs, UUIDs, cron expressions, ticket IDs, and port numbers via SQLite `needle_registry` table with composite `(token, namespace)` index.
- **Reverse HyDE (write-time questions)** — At store/index time, LLM generates hypothetical questions the content would answer. These question embeddings live in vector space alongside user queries for dramatically better needle recall.
- **API micro-chunking** — High-specificity tokens (IPs, UUIDs, etc.) get 200-char focused micro-chunks with their own embeddings, capped at `MAX_MICRO_CHUNKS_PER_MEMORY` (default 5).
- **Entity auto-extraction** — Deterministic regex + backtick/bold entity extraction feeds the knowledge graph at write time, not just curator-time.
- **Contextual augmentation** — Metadata headers prepended to chunks before embedding (Anthropic "Contextual Retrieval" pattern). Raw text preserved for FTS5 to prevent IDF dilution.
- **Uniform result types** — `ResultCandidate` dataclass + `RetrievalSource` enum ensure all retrieval sources (vector, BM25, literal, registry, graph) flow through the same RRF → threshold → reranker pipeline. No hardcoded score bypasses.
- **Multi-query expansion** — Queries expanded into semantic variants, each embedded and searched independently, results fused via RRF.
- **Embedding cache** — In-process LRU+TTL cache with immutable tuple storage (no defensive copies).

#### v1.11.0 audit fixes (post-implementation hardening)
- **Delete cascade: child artifact cleanup** — `delete_memory_complete()` now enumerates micro-chunk and reverse HyDE point IDs via Qdrant `scroll()` BEFORE deleting them, then cleans up FTS5 entries and needle registry rows for each child. Previously, only the primary memory's FTS5/registry rows were deleted, leaving orphaned micro-chunk and reverse HyDE entries in SQLite.
- **Indexer gather resilience** — `asyncio.gather()` for parallel reverse HyDE generation now uses `return_exceptions=True`. Previously, an unhandled exception from one concurrent LLM call could cancel all others despite the internal try/except guard.
- **Structured observability** — `store_pipeline.complete` and `retrieval_pipeline.complete` structured log events with per-operation metrics. Feature flag summary logged at startup via `config.feature_flags`.
- **DRY needle patterns** — `NEEDLE_PATTERNS` consolidated to single source in `chunking.py`. Imported by `graph.py`, `pre_extractor.py`, and all consumers.
- **FTS5 IDF quality** — All `upsert_fts_chunk` calls now pass raw text (not augmented text with metadata headers), preventing BM25 IDF dilution.

### ✅ v1.5.0 — Journal exports & post-deploy review fixes
- **Markdown journal exports** — New `journal.py` module writes daily `YYYY-MM-DD.md` files to `JOURNAL_DIR` on every `archivist_store`. Human-readable, greppable, editable. Secondary to Qdrant — losing the journal has no effect on retrieval. Config: `JOURNAL_ENABLED` (default true), `JOURNAL_DIR` (default `/data/archivist/journal`).
- **Webhook crash logging** — `fire_background()` now logs exceptions via `_log_fire_exception` callback (previously only stored task refs without logging failures).
- **Remove dead `src/mcp/` directory** — Empty directory left over from `handlers/` rename.
- **Integration test markers** — `test_dispatch.py` marked with `@pytest.mark.integration` so the marker defined in `pyproject.toml` is actually exercised.

### ✅ v1.4.1 — Code review & bug fixes
- **CRITICAL: Rename `src/mcp/` → `src/handlers/`** — The `src/mcp/` package shadowed the pip-installed `mcp` package, preventing `from mcp.types import Tool` and `from mcp.server import Server` from resolving. Renamed to `handlers/` to eliminate the conflict. `mcp_server.py` updated to import from `handlers._registry`.
- **Fix `graph.py` missing `import logging`** — FTS5 error handlers used `logging.getLogger()` but `logging` was never imported. Any FTS failure would raise `NameError` instead of being gracefully logged.
- **Fix `tokenizer.py` empty message list** — `count_message_tokens([])` returned 2 (reply priming) instead of 0. Added early return for empty input.
- **Clean dead code** — Removed unused backward-scan loop in `context_manager.py`, unused imports in `tools_admin.py` (`search_entities`, `is_permissive_mode`, `_ct`), redundant `json.JSONDecodeError` in `compaction.py` exception handler.
- **Fix 3 test assertions** — Relationship confidence test expected >1.0 but SQL uses `min(confidence+0.1, 1.0)` (caps at 1.0). Compaction fallback test missing import. Token counter empty list test updated.

### ✅ v1.4.0 — Documentation & test coverage
- **Full MCP tool documentation** — `CURSOR_SKILL.md` now documents all 30 tools with parameters, types, defaults, and examples. `REFERENCE.md` updated with condensed tool table and fleet agent mappings.
- **pytest infrastructure** — `pyproject.toml` with `pythonpath`, `asyncio_mode=auto`, markers. Root `conftest.py` with shared fixtures: env isolation (`_isolate_env`), in-memory `graph_db`, `rbac_config` from test YAML, `mock_llm`.
- **RBAC tests** (`test_rbac.py`, 13 tests) — `check_access` for own/other/unknown namespaces, `all` wildcard, namespace resolution, agent filtering, `list_accessible_namespaces` flags, permissive fallback mode.
- **Graph tests** (`test_graph.py`, 14 tests) — Entity CRUD (create, upsert, case-insensitive, search, limit), facts, relationships with confidence, curator_state, FTS5 (upsert, namespace filter, delete by file).
- **MCP dispatch tests** (`test_dispatch.py`, 8 tests) — Registry completeness (30+ tools, unique names, schemas, descriptions, expected set), unknown tool error, `archivist_context_check` and `archivist_namespaces` integration.
- **New module tests** (`test_new_modules.py`, 20 tests) — Tokenizer (count_tokens, message tokens, fallback), context_manager (under/over budget, empty, memories), compaction (flat, structured, fallback, format), fts_search (query sanitization, disabled, fusion), N-gram entity extraction, curator checksum.

### ✅ v1.3.0 — Curator improvements
- **Content checksum guard** — `curate_cycle()` now computes `sha256(content)` and compares against `curator_state` before processing. Files with unchanged content are skipped even if mtime changed. Checksums written after successful processing so interrupted cycles retry.
- **N-gram entity extraction** — `extract_entity_mentions()` upgraded from single-word matching to 3/2/1-word N-gram windows. Matches multi-word entities ("Argo CD", "hot cache"), short names (min 2 chars instead of 3), and compound phrases. Longer phrases tried first for priority.
- **Curator loop backoff** — Exponential backoff on failure (2x, capped at 1 hour), reset to base interval on success. Prevents hammering failing dependencies.

### ✅ v1.2.0 — BM25/FTS5 hybrid search (ReMe-inspired)
- **SQLite FTS5 schema** — New `memory_chunks` table + `memory_fts` FTS5 virtual table (porter stemming, unicode61 tokenizer) in `graph.py`. Created on `init_schema()`.
- **Dual-write indexing** — `indexer.index_file()` and `tools_storage._handle_store()` sync chunks to FTS5 after Qdrant upsert when `BM25_ENABLED=true`. `delete_file_points()` cleans both stores.
- **FTS search module** — New `fts_search.py`: `search_bm25()` with FTS5 query sanitization, `merge_vector_and_bm25()` with normalized weighted fusion (`VECTOR_WEIGHT * norm_v + BM25_WEIGHT * norm_b`).
- **Retriever integration** — BM25 results fused into `recursive_retrieve()` as "Stage 1-bm25" between vector search and graph augmentation. Full pipeline (dedup, decay, threshold, rerank) runs on fused results.
- **Retrieval trace** — `bm25_enabled` and `bm25_hits` fields added to `retrieval_trace` for observability.
- **Config** — `BM25_ENABLED` (default true), `BM25_WEIGHT` (default 0.3), `VECTOR_WEIGHT` (default 0.7).

### ✅ v1.1.0 — Context window management & structured compaction (ReMe-inspired)
- **Token counting module** — New `tokenizer.py` with tiktoken cl100k_base (lazy-loaded) + `chars//4` fallback. `count_tokens()` for strings, `count_message_tokens()` for chat message lists with per-message overhead.
- **Context manager** — New `context_manager.py`: `check_context()` analyzes messages against token budget, returns usage %, hint (ok/compress/critical), split recommendations. `check_memories_budget()` for raw text lists.
- **Structured compaction** — New `compaction.py`: `compact_structured()` produces Goal/Progress/Decisions/Next Steps/Critical Context JSON via LLM. Supports incremental compaction with `previous_summary`. Falls back to flat summary on parse failure.
- **`archivist_context_check` MCP tool** — Pre-reasoning context check: agents send messages or memory texts, get token count, budget usage, and compaction hint. Added to `tools_admin.py`.
- **Upgraded `archivist_compress`** — New `format` parameter (`flat`/`structured`) and `previous_summary` for incremental compaction. Delegates to `compaction` module. Structured responses include full JSON breakdown.
- **Retriever token counting** — `rlm_retriever.py` budget cap and context-status signaling now use `tokenizer.count_tokens()` instead of `len(text) // 4`.
- **Config** — `DEFAULT_CONTEXT_BUDGET` env var (default 128000 tokens).

### ✅ v1.0.1 — Bug fixes, refactor & hardening (ReMe comparison review)
- **Split monolithic `mcp_server.py`** — Refactored 1,694-line single file into `src/mcp/` package with 8 modules by domain (search, storage, trajectory, skills, admin, cache) plus shared helpers and central registry. `mcp_server.py` is now ~30 lines. Handler dispatch uses `dict.get()` instead of 58-branch if/elif chain.
- **Fix `decay_old_entries()` row count** — Replaced `conn.total_changes` (cumulative for connection) with `cursor.rowcount` (per-statement) in `curator.py`.
- **Track background tasks** — `startup()` now stores task references in `_background_tasks` with named tasks and `done_callback` for crash logging. Same pattern applied to `webhooks.py` `fire_background()`.
- **Non-root Docker** — Container runs as `archivist` user (UID/GID 1000 by default, configurable via `--build-arg`). Data directories pre-created and chowned.
- **Inspiration doc** — New `docs/INSPIRATION.md` tracking design influences from ReMe and other projects.

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

### Near-term
- End-to-end integration tests requiring Qdrant + LLM (marked `@pytest.mark.integration`)

### Longer-term
- Migrate SSE → Streamable HTTP transport for lower latency
- PostgreSQL option for graph/metadata at scale
- Streaming partial search results
- Local embedding models (sentence-transformers)
- Multi-collection Qdrant support
- Web UI dashboard
- Formal benchmarks (LoCoMo, HaluMem)

---

## Non-Goals (Out of Scope)

- **Real-time chat** — Archivist is memory, not conversation
- **File storage** — Archivist indexes files, doesn't host them
- **Agent orchestration** — Memory service only; agents managed externally
