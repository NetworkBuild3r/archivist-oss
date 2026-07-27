# Archivist multi-agent memory — roadmap (April 2026)

**Status** — Retrieval foundation is solid (v2 pipeline complete). Semantic chunking (Phase 5) is done. **Phase 3 + 3.5 (transactional outbox + `MemoryTransaction` + conn-passing shims)** is **complete**: see [`docs/rearchitect_storage_phase3.md`](rearchitect_storage_phase3.md) and [`CHANGELOG.md`](../CHANGELOG.md) **v2.1.0**. **PostgreSQL first-class backend** is **complete** (v2.2.0): all hot paths, schema init, FTS, backups, and Docker wiring work on both SQLite and Postgres — see [`CHANGELOG.md`](../CHANGELOG.md) **v2.2.0** and [`docs/DOCKER.md`](DOCKER.md). **Answer Finder (v2.3.0)** is **complete**: hierarchical tiered memory, token-budgeted context packing, auto-compression, ephemeral `SessionStore`, high-level `get_relevant_context()` API, multi-agent handoff protocol, and token savings observability — see [`CHANGELOG.md`](../CHANGELOG.md) **v2.3.0**.

**Goal** — Be the most **trustworthy and production-ready** open multi-agent memory layer in 2026: observable, RBAC-aware, safe under fleet load, and the best answer finder in the industry.

**Next milestones (engineering)** — Unique Differentiators completion program
([BRAIN-005](../sdd/brainstorms/BRAIN-005-complete-unique-differentiators/decision-document.md)):
**Immediate next — maintenance / optional demos.** Diff **#8** Observability billboard
**Done** ([INIT-013](../sdd/initiatives/INIT-013-observability-billboard/) /
[ADR-013](adr/ADR-013-observability-billboard.md)) — served `/admin/ui/` + HTTP lineage/audit.
Diff **#7** Checkpoint / time-travel **Done (scoped)** ([INIT-012](../sdd/initiatives/INIT-012-checkpoint-time-travel/) /
[ADR-012](adr/ADR-012-checkpoint-time-travel.md)) — ops promotion, branch, thin HITL;
institutional tier DDL **deferred**. Diff **#4** Memory as a Product **Done**
([INIT-011](../sdd/initiatives/INIT-011-memory-as-product-mcp/) /
[ADR-011](adr/ADR-011-memory-as-product-mcp.md)). Diff **#6** Self-Curation **Done**
([INIT-010](../sdd/initiatives/INIT-010-intelligent-self-curation/) /
[ADR-010](adr/ADR-010-intelligent-self-curation.md)). Diff **#5** (INIT-009 /
[ADR-009](adr/ADR-009-native-multi-agent-coordination.md)) **merged** (PR #53).
Tip-only lessons remain ([ADR-007](adr/ADR-007-procedural-memory-wedge.md) /
[ADR-008](adr/ADR-008-retire-skills-tip-lessons.md)).
<!-- BRAIN-005 / INIT-013/SPEC-005 -->

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
| 4 | **Memory as a Product** | Versioned, exportable, forkable, auditable memory graphs (Git for agent knowledge) | **Done (INIT-011 / [ADR-011](adr/ADR-011-memory-as-product-mcp.md))** — `archivist_map_*` on **ops**/**full** (snapshot/fork/export/**import**); service from INIT-001/SPEC-009 |
| 5 | **Native Multi-Agent Coordination** | Built-in shared/institutional *use* of memory with selective sharing, conflict resolution, and negotiation | **Done (INIT-009 / ADR-009 / PR #53)** — `share_*` on ops; tip_ids + handoff; conflict→resolver |
| 6 | **Intelligent Self-Curation** | Automatic summarization, relevance-based forgetting, contradiction detection | **Done (INIT-010 / [ADR-010](adr/ADR-010-intelligent-self-curation.md))** — curator product loop: reconsolidation + relevance forget + contradiction resolve (safe defaults / staged enablement) |
| 7 | **Full Checkpointing + Time-Travel** | LangGraph-style resume, replay, branch, human-in-the-loop | **Done (scoped, INIT-012 / [ADR-012](adr/ADR-012-checkpoint-time-travel.md))** — `archivist_checkpoint_*` on **ops**/**full** (save/list/get/resume/replay/**branch**/interrupt/**approve**); institutional tier DDL deferred |
| 8 | **Observability Dashboard** | Memory lineage, audit trails, cost tracking, visualization | **Done (INIT-013 / [ADR-013](adr/ADR-013-observability-billboard.md))** — served `/admin/ui/` billboard + `GET /admin/lineage` / `/admin/audit`; reuses dashboard/retrieval-logs JSON + MCP admin tools |

These eight features combined will make Archivist the **most trustworthy and production-ready** memory layer.

---

## Phased Roadmap (2026)

| Phase | Name | Focus | Effort | Target Completion | Success Metric |
|-------|------|-------|--------|-------------------|----------------|
| **5** | Semantic Chunking | Production-grade markdown-aware chunking (headings, code blocks, lists) | 1–2 days | Done | Zero regression on short docs + measurable gains on long docs |
| **6** | Provenance & Actor-Aware Memory | Every memory entry carries `actor_id`, `actor_type`, `confidence`, `source_trace` — reranker is sole ranking authority | 1–2 weeks | Done | Actor-aware retrieval + provenance queries work |
| **7** | Multi-Tier Memory + Checkpointing | Explicit tiers (working/episodic/semantic/procedural/institutional) + LangGraph-style checkpointing | 2–3 weeks | — | Full tier support + time-travel/resume |
| **8** | Intelligent Lifecycle Management | Auto-summarization, relevance-based forgetting, contradiction resolution, reflection loops | 3–4 weeks | — | Self-curation works without manual tuning |
| **9** | Observability & Control Plane | Memory explorer dashboard, audit logs, cost tracking, lineage visualization | 2–3 weeks | Done (INIT-013) | Full visibility into memory state |
| **10** | Multi-Agent Coordination Primitives | Selective share + accept/reject consensus (`archivist_share_*`; INIT-001/SPEC-010) | 2–3 weeks | Plumbing done | Diff #5 product maturity = INIT-009 (not a second invent) |

---

## Immediate Next Steps (Recommended)

<!-- BRAIN-005 -->

**Authoritative next work:** [BRAIN-005](../sdd/brainstorms/BRAIN-005-complete-unique-differentiators/decision-document.md)
program **complete** for Unique Differentiators **#1–#8** (Diff #8 closed by INIT-013 /
[ADR-013](adr/ADR-013-observability-billboard.md)).

1. **Immediate next** — maintenance / optional demos / domain MCP outside Archivist.
2. **Done** — Diff **#1–#8** including Diff **#8** (INIT-013 /
   [ADR-013](adr/ADR-013-observability-billboard.md) — `/admin/ui/` + HTTP lineage/audit),
   Diff **#7** (INIT-012 / [ADR-012](adr/ADR-012-checkpoint-time-travel.md) — scoped),
   Diff **#4** (INIT-011 / [ADR-011](adr/ADR-011-memory-as-product-mcp.md)), Diff **#5**
   (INIT-009 / PR #53), Diff **#6** (INIT-010 / [ADR-010](adr/ADR-010-intelligent-self-curation.md)).
   Historical: INIT-008…004. Foundational: INIT-001 /
   [ADR-001](adr/ADR-001-platform-coherence-sequencing.md),
   INIT-003 / [ADR-003](adr/ADR-003-coach-core-reliability.md).
3. **Still open (non-diff)** — Phase 7 tracking for **institutional / multi-tier DDL**
   (ADR-012 GR-TIER-001 deferred). Skill OS remains cancelled (ADR-008).
4. Optional: domain-specific long-document fixture locally (keep private data out of repo).
   Short recipes: [`demos/map-roundtrip.md`](demos/map-roundtrip.md),
   [`demos/checkpoint-branch-hitl.md`](demos/checkpoint-branch-hitl.md),
   [`demos/observability-billboard.md`](demos/observability-billboard.md).

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
- [ ] Phase 7 — Multi-tier memory + checkpointing *(Diff #7 checkpoint product Done scoped — ADR-012; institutional / multi-tier DDL still open)*
- [ ] Phase 8 — Intelligent Lifecycle Management *(Diff #6 product Done — ADR-010; tracking may remain for further lifecycle maturity)*
- [x] Phase 9 — Observability & Control Plane *(INIT-013 / ADR-013 — `/admin/ui/` + HTTP lineage/audit)*
- [x] Phase 10 — Multi-Agent Coordination Primitives plumbing (`archivist_share_*`; INIT-001/SPEC-010) + Diff #5 **product** (INIT-009 / ADR-009)

---

## BEIR thin (NFCorpus) — regression log

These numbers come from `benchmarks/scripts/run_thin_reference.sh` → `benchmarks/academic/beir_thin.py` (dense bi-encoder **only**, not the full RLM pipeline). Use them to **track embedding defaults and harness drift over time**, not to claim “Archivist vs BEIR SOTA.” See the main [README](../README.md#benchmarks) for why BEIR is secondary here.

Expected console flow: `Encoding Batch …` → tqdm batch bar → nDCG / MAP / Recall / P@k blocks → boxed **BEIR thin** summary (repeats NDCG@k) → path to `.benchmarks/beir_nfcorpus_thin.json`. Some library versions also print a one-line note about `ignore_identical_ids` (default evaluator behavior); it is not a failure.

| Date | Git / notes | Dataset | Queries | Embedding model | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | MAP@10 | Recall@10 | P@1 |
|------|-------------|---------|---------|-----------------|--------|--------|--------|---------|--------|-----------|-----|
| 2026-04-14 | `feature/v1.12-cascade-tech-debt`, local thin run | NFCorpus | 50 | `sentence-transformers/all-MiniLM-L6-v2` | 0.4300 | 0.3907 | 0.3560 | 0.3456 | 0.1239 | 0.1660 | 0.5000 |

_Add new rows when you change default embed models, BEIR limits, or the thin harness._

**Last Updated**: INIT-013/SPEC-005 — Diff #8 Observability billboard Done; Phase 9 checked; Immediate Next → maintenance (2026-07-27)
**Goal**: Become the most trustworthy, observable, and production-ready multi-agent memory system in 2026.
