# Archivist Fleet Optimization — Research Findings

## Date: 2026-04-11
## Scope: Multi-agent fleet memory optimization — needle-in-haystack, context engineering, waste reduction

---

## Current Architecture Summary

Archivist v1.11.0 is a Memory-as-a-Service MCP server with:
- **10-stage RLM retrieval pipeline** (vector + BM25 + graph + temporal + hotness + threshold + rerank + refine + synthesize)
- **14-stage write pipeline** (RBAC + conflict + dedup + entities + augment + embed + FTS5 + registry + micro-chunks + reverse HyDE + audit + cache + metrics + webhook)
- **35 MCP tools** across search, storage, trajectory, skills, admin, cache
- **Knowledge graph** with entities, facts, relationships, needle registry
- **Background curator** for knowledge extraction, decay, consolidation
- SQLite (FTS5 + knowledge graph) + Qdrant (vector store)

---

## Critical Findings

### F1: Needle Retrieval — 0% Recall in Benchmarks (ROOT CAUSE IDENTIFIED)
The pipeline benchmark shows **0% recall** for needle queries across ALL pipeline variants.

**Root cause confirmed:** The benchmark uses `indexer.py:full_index(hierarchical=True)` to create the corpus (see `benchmarks/pipeline/evaluate.py:250-295`). The indexer creates FTS5 entries, entity extraction, needle registry entries, and reverse HyDE -- but **does NOT call `_extract_needle_micro_chunks()`**. Grep for `micro.chunk|_extract_needle_micro` in `src/indexer.py` returns zero hits. Micro-chunk generation only exists in the production MCP store path (`tools_storage.py:342-393`).

Without micro-chunks, needle tokens (IPs, crons, UUIDs, key=value) are embedded only as part of 2000-char parent chunks diluted by filler text. The vector search can't discriminate the needle from the haystack.

The needle registry IS populated by the indexer (line 288-293), and the retrieval pipeline consults it unconditionally (no `is_needle_query()` guard -- see `rlm_retriever.py:711`). However, registry hits are validated against Qdrant (lines 728-752), and since the registry stores the *parent chunk* point ID, they should survive validation. The issue is that even when found, these parent chunks rank poorly in RRF because their vector similarity to a needle query is low (the embedding is dominated by filler text, not the needle fact).

### F2: Entity Namespace Isolation Gap
The `entities` table has **no namespace column**. Entity names are globally unique (`UNIQUE COLLATE NOCASE`). In a fleet: Agent-A's "redis" and Agent-B's "redis" collide. Facts reference entities by integer FK, so they're implicitly global too. `extract_entity_mentions()` in graph_retrieval.py searches the global entity table with no namespace filter. This is the **biggest namespace isolation gap**.

### F3: Pipeline Stages Beyond Vector Don't Improve Recall
In benchmarks, BM25/graph/temporal/hotness/rerank all show **flat or reduced** recall vs vector-only. The knowledge graph is likely empty during benchmarking. This means the graph, BM25, and scoring stages are adding latency without measurable benefit when the graph is cold.

### F4: Up to 6 LLM Calls Per Retrieval
Classification + expansion + HyDE + retry + N×refinement + synthesis. Even budget-gated, refinement (N parallel calls) and synthesis (1 call) are ungated. For a fleet of 20 agents each doing 10 queries/minute, that's potentially 1200+ LLM calls/minute.

### F5: embed_batch Is Not Truly Batched
`embed_batch()` fires N individual `/v1/embeddings` HTTP calls via `asyncio.gather`. For a memory with 8 micro-chunks + 4 reverse HyDE questions, that's 12 concurrent HTTP requests vs 1 batch call. Most embedding APIs accept array inputs.

### F6: Conflict + Dedup Double Query
`check_for_conflicts()` and `llm_adjudicated_dedup()` both independently embed text and query Qdrant. Two vector searches per store when both enabled. Both also hardcode `QDRANT_COLLECTION` instead of using `collection_for(namespace)`.

### F7: Reranker Blocks the Event Loop
`model.predict(pairs)` in reranker.py is synchronous CPU inference (~560M params) NOT wrapped in `asyncio.to_thread`. Blocks the entire event loop during inference.

### F8: GRAPH_WRITE_LOCK is Process-Level
56+ call sites across the codebase using a single `threading.Lock()`. Fine for single-process but a hard bottleneck for scaling.

### F9: Hot Cache Invalidation is O(N)
`invalidate_namespace()` scans all entries for all agents under a lock. At scale, this becomes a bottleneck on every write (since writes invalidate the cache).

### F10: Wake-Up Context Cap at 50 Agent/Namespace Pairs
`_refresh_wake_up_caches()` is LIMIT 50 — a 100-agent fleet leaves half without cached wake-up context.

### F11: Hotness Scoring Population Bug
`batch_update_hotness()` only updates *existing* rows in `memory_hotness`. New memories that have been retrieved but don't have a row never get scored.

### F12: Academic Benchmarks All Failed (0%)
LongMemEval, LoCoMo, HaluMem all return zeros. May be harness issues but means no validated academic comparison exists.

### F13: QA Checklist 100% Unverified
~250+ items defined, none checked off.

### F14: Missing Enrichment in Augmentation
`_handle_store()` computes `thought_type` and `topic` but doesn't pass them to `augment_chunk()`. The `topic` param in `augment_chunk()` is dead code.

### F15: No Per-Namespace Configuration
Every namespace gets the same chunking size, retrieval threshold, budget, and feature flags. A high-frequency monitoring agent and a low-frequency architecture agent share identical parameters.

### F16: Adaptive Vector Widen Doesn't Re-Rerank
When the adaptive widen triggers a second Qdrant search with wider limits, the new results go through threshold filtering but NOT through reranking. So widened results use unranked scores.

### F17: Budget Reserve Doesn't Deduct
`LatencyBudget.reserve()` stores the estimate but doesn't subtract from remaining budget. Multiple operations can both pass `can_afford()` and execute, blowing the budget.

### F18: Multi-Hop Recall at 67-78%
Graph-augmented multi-hop retrieval is where the knowledge graph should shine, but it's only moderate. The empty graph during benchmarks may explain this.

### F19: Cross-Agent Cap Only in Vector Path
`CROSS_AGENT_MAX_SHARE=0.6` is enforced in vector retrieval but NOT in graph retrieval, needle lookup, or entity fact injection. Agent-B's graph entities can flood Agent-A's results unchecked.

### F20: No Retrieval-Aware Storage
Agents have no visibility into what queries will be asked. Memories are stored with static enrichment (reverse HyDE, micro-chunks) but there's no feedback loop from retrieval patterns to storage strategy.

---

## What's Working Well

- **7-way fusion** (vector + BM25 porter + BM25 exact + needle registry + graph + temporal + hotness) is architecturally sound
- **Single-hop recall 99-100%** — the core vector search is excellent
- **Broad synthesis recall 91-97%** — good aggregation
- **Tiered context (L0/L1/L2)** — agents can choose detail level
- **Contrastive tip consolidation** — best-designed cross-agent sharing mechanism
- **Task fingerprinting** — SHA256 grouping enables cross-agent trajectory analysis
- **Pre-extract → thread everywhere** pattern avoids duplicate extraction
- **Fire-and-forget reverse HyDE** — doesn't block the store hot path
- **Wake-up context** — compact session bootstrap is a great pattern
- **Context budget checking** — `archivist_context_check` enables proactive context management
- **Dual FTS5** (porter + exact) — good for mixed workloads
- **Topic routing** — zero-latency classification without LLM
- **Outcome-aware scoring** — trajectory feedback influences retrieval

---

## Optimization Opportunities (Ranked by Impact)

### Tier 1: High Impact, Critical for Fleet

1. **Fix needle retrieval** — 0% → target 95%+ deterministic recall for structured tokens
2. **Namespace-scope the knowledge graph** — add namespace column to entities/facts
3. **True batch embedding API** — 12 HTTP calls → 1 per store operation
4. **Consolidate conflict+dedup Qdrant queries** — 2 vector searches → 1
5. **Wrap reranker in asyncio.to_thread** — unblock event loop during inference
6. **Wire thought_type + topic into augment_chunk** — free precision improvement
7. **Fix hotness scoring population bug** — new memories never get scored

### Tier 2: Important for Scale

8. **Budget system: make reserve() actually deduct** — prevent budget overruns
9. **Per-namespace config profiles** (lite/standard/heavy) — fleet agents have different needs
10. **Retrieval-feedback storage** — use retrieval logs to identify and reinforce high-value memories
11. **Adaptive latency budget** — learn per-query-type optimal budgets from p95 telemetry
12. **Cross-agent cap enforcement in graph retrieval** — prevent entity flooding
13. **Hot cache: use indexed namespace lookup** — O(1) invalidation instead of O(N) scan

### Tier 3: Quality of Life for Fleet

14. **Fleet-level tip discovery** — agents auto-discover consolidated fleet tips
15. **Incremental tip consolidation** — don't re-embed entire corpus each cycle
16. **Parallel curator extraction** — parallelize file processing and LLM calls
17. **Re-rerank after adaptive widen** — widened results need scoring
18. **Raise wake-up cache cap** — 50 → 500 agent/namespace pairs
19. **Dynamic threshold consistency** — fix the <4 results branch below-floor bug
