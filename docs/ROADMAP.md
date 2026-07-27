# Archivist multi-agent memory — roadmap (April 2026)

**Status** — Retrieval foundation is solid (v2 pipeline complete). Semantic chunking (Phase 5) is done. **Phase 3 + 3.5 (transactional outbox + `MemoryTransaction` + conn-passing shims)** is **complete**: see [`docs/rearchitect_storage_phase3.md`](rearchitect_storage_phase3.md) and [`CHANGELOG.md`](../CHANGELOG.md) **v2.1.0**. **PostgreSQL first-class backend** is **complete** (v2.2.0): all hot paths, schema init, FTS, backups, and Docker wiring work on both SQLite and Postgres — see [`CHANGELOG.md`](../CHANGELOG.md) **v2.2.0** and [`docs/DOCKER.md`](DOCKER.md). **Answer Finder (v2.3.0)** is **complete**: hierarchical tiered memory, token-budgeted context packing, auto-compression, ephemeral `SessionStore`, high-level `get_relevant_context()` API, multi-agent handoff protocol, and token savings observability — see [`CHANGELOG.md`](../CHANGELOG.md) **v2.3.0**.

**Goal** — Be the most **trustworthy and production-ready** open multi-agent memory layer in 2026: observable, RBAC-aware, safe under fleet load, and the best answer finder in the industry.

**Next milestones (engineering)** — **INIT-009** / [ADR-009](adr/ADR-009-native-multi-agent-coordination.md)
(Diff #5 productize: `share_*` on **ops**, conflict→resolver, tip/lesson share via
handoff + `tip_ids`) is **complete** (architecture:
[`INIT-009-…-architecture.md`](../sdd/initiatives/INIT-009-native-multi-agent-coordination/design/INIT-009-native-multi-agent-coordination-architecture.md)).
Suggested next differentiator wedge: Unique Differentiator **#6** (Intelligent
Self-Curation / hierarchical reconsolidation) — choose explicitly; do not
auto-start. INIT-008 / [ADR-008](adr/ADR-008-retire-skills-tip-lessons.md)
(PR #52), INIT-007 / [ADR-007](adr/ADR-007-procedural-memory-wedge.md) (PR #49), INIT-006 /
[ADR-006](adr/ADR-006-agentic-memory-eval-gym.md) (PR #48), INIT-005 /
[ADR-005](adr/ADR-005-coach-path-performance.md), and INIT-004 /
[ADR-004](adr/ADR-004-llm-native-coach-memory-surfaces.md) are **historical**.
(INIT-001 / [ADR-001](adr/ADR-001-platform-coherence-sequencing.md)
and INIT-003 / [ADR-003](adr/ADR-003-coach-core-reliability.md) remain foundational.)
<!-- INIT-009/SPEC-006 -->

---

## Why this roadmap matters

Current top systems win on:
- Hybrid storage (vector + graph + relational)
- Provenance & actor-awareness
- Multi-tier memory + intelligent lifecycle
- Checkpointing / time-travel
- Strong observability & auditability

Archivist already has strong retrieval (synthetic questions, reranker, semantic chunking, needle registry, graph).
We now shift from “great retrieval” to “great memory system for collaborating agents.”

---

## Unique Differentiators (What Will Make Us Stand Out)

| # | Differentiator | Why It Wins | Current Status |
|---|----------------|-------------|----------------|
| 1 | **Full Provenance & Actor-Aware Memory** | Every fact knows *who* said it, when, and with what confidence | Done (Phase 6) |
| 2 | **Answer Finder — Token-Efficient Context** | Hierarchical tiers + token-budgeted packing + auto-compress; ≥60% token reduction vs naive retrieval | Done (v2.3.0) |
| 3 | **Multi-Agent Handoff Protocol** | Typed `HandoffPacket` transfers session summary, goals, tips, and knowledge snapshot between agents | Done (v2.3.0) |
| 4 | **Memory as a Product** | Versioned, exportable, forkable, auditable memory graphs (Git for agent knowledge) | In progress (service core — INIT-001/SPEC-009) |
| 5 | **Native Multi-Agent Coordination** | Built-in shared/institutional *use* of memory with selective sharing, conflict resolution, and negotiation | **Productized (INIT-009 / ADR-009)** — `share_*` on ops; tip_ids + handoff tip share; conflict apply→resolver |
| 6 | **Intelligent Self-Curation** | Automatic summarization, relevance-based forgetting, contradiction detection | Partial |
| 7 | **Full Checkpointing + Time-Travel** | LangGraph-style resume, replay, branch, human-in-the-loop | Not started |
| 8 | **Observability Dashboard** | Memory lineage, audit trails, cost tracking, visualization | Partial (token savings heatmap done) |

These six features combined will make Archivist the **most trustworthy and production-ready** memory layer.

---

## Phased Roadmap (2026)

| Phase | Name | Focus | Effort | Target Completion | Success Metric |
|-------|------|-------|--------|-------------------|----------------|
| **5** | Semantic Chunking | Production-grade markdown-aware chunking (headings, code blocks, lists) | 1–2 days | Done | Zero regression on short docs + measurable gains on long docs |
| **6** | Provenance & Actor-Aware Memory | Every memory entry carries `actor_id`, `actor_type`, `confidence`, `source_trace` — reranker is sole ranking authority | 1–2 weeks | Done | Actor-aware retrieval + provenance queries work |
| **7** | Multi-Tier Memory + Checkpointing | Explicit tiers (working/episodic/semantic/procedural/institutional) + LangGraph-style checkpointing | 2–3 weeks | — | Full tier support + time-travel/resume |
| **8** | Intelligent Lifecycle Management | Auto-summarization, relevance-based forgetting, contradiction resolution, reflection loops | 3–4 weeks | — | Self-curation works without manual tuning |
| **9** | Observability & Control Plane | Memory explorer dashboard, audit logs, cost tracking, lineage visualization | 2–3 weeks | — | Full visibility into memory state |
| **10** | Multi-Agent Coordination Primitives | Selective share + accept/reject consensus (`archivist_share_*`; INIT-001/SPEC-010) | 2–3 weeks | Plumbing done | Diff #5 product maturity = INIT-009 (not a second invent) |

---

## Immediate Next Steps (Recommended)

<!-- INIT-009/SPEC-006 -->

**Authoritative next work (suggested):** Unique Differentiator **#6 Intelligent
Self-Curation** (hierarchical reconsolidation / contradiction depth) — park until
chosen as a formal INIT. Do **not** invent a skill OS; tip-only lessons remain
([ADR-007](adr/ADR-007-procedural-memory-wedge.md) / [ADR-008](adr/ADR-008-retire-skills-tip-lessons.md)).

1. **Suggested next (not started)** — Diff **#6** Self-Curation / reconsolidation
   (choose INIT explicitly after INIT-009 Mode E / merge).
2. **Complete — INIT-009** Diff #5 productize /
   [ADR-009](adr/ADR-009-native-multi-agent-coordination.md) (`share_*` on ops,
   conflict→resolver, tip/lesson share; architecture diagrams under
   `sdd/initiatives/INIT-009-…/design/`). Historical: **INIT-008** (PR #52),
   **INIT-007** (PR #49), **INIT-006**…**INIT-004**. Foundational: INIT-001 /
   [ADR-001](adr/ADR-001-platform-coherence-sequencing.md), INIT-003 /
   [ADR-003](adr/ADR-003-coach-core-reliability.md).
3. **Parked for later** — Memory-as-Product finish (Diff #4), Phase 7 checkpoint
   UX / institutional **tier** (Diff #7 — **future only**; not part of Diff #5),
   MaP / recipe demos, Phase 9 observability billboard (Diff #8). Skill↔tips
   bridge cancelled (ADR-008). Phase 10 `share_*` plumbing + Diff #5 product
   maturity = INIT-009.
4. Optional: add a **domain-specific long-document fixture** (your own docs +
   questions) locally to tune retrieval beyond the public toy corpus — keep
   private data out of the public repo.

---

## Phase 6.5 — OpenClaw Compatibility Fix (April 2026, revised by PR #39)

**Status**: Done — **security-revised**. The original fix (below the line)
accepted a known-broken auth value; PR #39 (Security Hardening) reverted that
acceptance. This section reflects the **current, correct** behavior.
<!-- INIT-001/SPEC-001: rewritten per PR #39; do not reintroduce the
     bypass this section used to document. -->

**Motivation**: OpenClaw v2026.4.8 uses the deprecated SSE MCP transport and
has a client-side env-var interpolation bug in the `mcp.servers` headers
config — it sends the literal string `"Bearer ${ARCHIVIST_API_KEY}"` rather
than the resolved key.

**Current behavior (post-PR #39)**:

| Area | Behavior | Effect |
|------|----------|--------|
| `MCP_SSE_ENABLED` default | `true` | Both transports mount on startup; no config change needed for legacy clients |
| Auth middleware | Literal `Bearer ${ARCHIVIST_API_KEY}` is **rejected with `401`** | Closes the auth-bypass surface a client-side templating bug could otherwise create; no key value is ever treated as valid by pattern-matching its literal placeholder form |
| Startup warning | Emitted when no API key **and** no namespaces config are set | Operators see missing-auth misconfiguration at boot, independent of the OpenClaw case |

**Why the original fix was rejected:** Accepting the literal placeholder
string as "close enough" to a real key means any client that fails to
interpolate its config (not just OpenClaw) authenticates with a fixed,
publicly-known string. That is a bypass, not a compatibility shim. Fix the
client config instead — see below.

**Supported fix for OpenClaw (and any client with the same templating bug):**
Use `X-API-Key` instead of the `Authorization: Bearer` header — it is not
subject to the Bearer-interpolation bug and was always supported:

```json
"headers": { "X-API-Key": "${ARCHIVIST_API_KEY}" }
```

If the client only supports `Authorization: Bearer`, confirm the client
actually resolves the `${...}` template to the real key value before startup
(fixed in later OpenClaw releases) — a **resolved** `Bearer <real-key>` header
continues to work exactly as it always has. Only the literal, unresolved
placeholder string is rejected.

**Transport summary (unchanged)**:

| Endpoint | Transport | Client |
|----------|-----------|--------|
| `POST /mcp` | Streamable HTTP (MCP spec ≥2025-03) | Modern clients (Cursor, Claude Desktop ≥2025-06) |
| `GET /mcp/sse` | Legacy SSE | OpenClaw ≤v2026.4.8 and any other SSE-only client |
| `POST /mcp/messages/` | Legacy SSE message channel | Same (paired with `GET /mcp/sse`) |

Set `MCP_SSE_ENABLED=false` once all clients are on the modern transport to
reclaim the two extra routes.

---

## Tracking checklist

- [x] Phase 5 — Semantic chunking
- [x] Phase 6 — Provenance & actor-aware memory
- [x] Phase 6.5 — OpenClaw compatibility fix
- [x] Phase 3 + 3.5 — Transactional outbox + atomic SQLite writes (see `docs/rearchitect_storage_phase3.md`)
- [x] PostgreSQL first-class backend — v2.2.0 (see `CHANGELOG.md`)
- [x] Answer Finder — v2.3.0: tiered memory, token packing, handoff protocol, savings observability
- [ ] Phase 7 — Multi-tier memory + checkpointing
- [ ] Phase 8 — Intelligent Lifecycle Management
- [ ] Phase 9 — Observability & Control Plane
- [x] Phase 10 — Multi-Agent Coordination Primitives plumbing (`archivist_share_*`; INIT-001/SPEC-010) + Diff #5 **product** (INIT-009 / ADR-009)

---

## BEIR thin (NFCorpus) — regression log

These numbers come from `benchmarks/scripts/run_thin_reference.sh` → `benchmarks/academic/beir_thin.py` (dense bi-encoder **only**, not the full RLM pipeline). Use them to **track embedding defaults and harness drift over time**, not to claim “Archivist vs BEIR SOTA.” See the main [README](../README.md#benchmarks) for why BEIR is secondary here.

Expected console flow: `Encoding Batch …` → tqdm batch bar → nDCG / MAP / Recall / P@k blocks → boxed **BEIR thin** summary (repeats NDCG@k) → path to `.benchmarks/beir_nfcorpus_thin.json`. Some library versions also print a one-line note about `ignore_identical_ids` (default evaluator behavior); it is not a failure.

| Date | Git / notes | Dataset | Queries | Embedding model | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | MAP@10 | Recall@10 | P@1 |
|------|-------------|---------|---------|-----------------|--------|--------|--------|---------|--------|-----------|-----|
| 2026-04-14 | `feature/v1.12-cascade-tech-debt`, local thin run | NFCorpus | 50 | `sentence-transformers/all-MiniLM-L6-v2` | 0.4300 | 0.3907 | 0.3560 | 0.3456 | 0.1239 | 0.1660 | 0.5000 |

_Add new rows when you change default embed models, BEIR limits, or the thin harness._

**Last Updated**: INIT-009 Diff #5 complete (SPEC-006 architecture); Immediate Next → Diff #6 suggested (2026-07-26)
**Goal**: Become the most trustworthy, observable, and production-ready multi-agent memory system in 2026.
