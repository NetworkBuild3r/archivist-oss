# Archivist Multi-Agent Memory Tool — Roadmap (April 2026)

**Status**: Retrieval foundation is solid (v2 pipeline complete). Semantic chunking (Phase 5) is done.  
**Goal**: Become the **clear #1 production-ready multi-agent memory system** in 2026 — more trustworthy, observable, and collaborative than Mem0, Hindsight, Letta, Zep/Graphiti, or LangGraph+LangMem.

---

## Why This Roadmap Matters

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
| 2 | **Memory as a Product** | Versioned, exportable, forkable, auditable memory graphs (Git for agent knowledge) | Not started |
| 3 | **Native Multi-Agent Coordination** | Built-in shared/institutional memory with selective sharing, conflict resolution, and negotiation | Not started |
| 4 | **Intelligent Self-Curation** | Automatic summarization, relevance-based forgetting, contradiction detection | Partial |
| 5 | **Full Checkpointing + Time-Travel** | LangGraph-style resume, replay, branch, human-in-the-loop | Not started |
| 6 | **Observability Dashboard** | Memory lineage, audit trails, cost tracking, visualization | Not started |

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
| **10** | Multi-Agent Coordination Primitives | Built-in planner/researcher/executor/verifier patterns with shared memory & consensus | 2–3 weeks | — | Native support for complex agent teams |

---

## Immediate Next Steps (Recommended)

1. **Phase 7** (Multi-Tier Memory + Checkpointing) — next highest differentiation.
2. Optional: add a **domain-specific long-document fixture** (your own docs + questions) locally to tune retrieval beyond the public toy corpus — keep private data out of the public repo.

---

## Tracking Checklist

- [x] Phase 5 — Semantic Chunking
- [x] Phase 6 — Provenance & Actor-Aware Memory
- [ ] Phase 7 — Multi-Tier Memory + Checkpointing
- [ ] Phase 8 — Intelligent Lifecycle Management
- [ ] Phase 9 — Observability & Control Plane
- [ ] Phase 10 — Multi-Agent Coordination Primitives

---

## BEIR thin (NFCorpus) — regression log

These numbers come from `benchmarks/scripts/run_thin_reference.sh` → `benchmarks/academic/beir_thin.py` (dense bi-encoder **only**, not the full RLM pipeline). Use them to **track embedding defaults and harness drift over time**, not to claim “Archivist vs BEIR SOTA.” See the main [README](../README.md#benchmarks) for why BEIR is secondary here.

Expected console flow: `Encoding Batch …` → tqdm batch bar → nDCG / MAP / Recall / P@k blocks → boxed **BEIR thin** summary (repeats NDCG@k) → path to `.benchmarks/beir_nfcorpus_thin.json`. Some library versions also print a one-line note about `ignore_identical_ids` (default evaluator behavior); it is not a failure.

| Date | Git / notes | Dataset | Queries | Embedding model | NDCG@1 | NDCG@3 | NDCG@5 | NDCG@10 | MAP@10 | Recall@10 | P@1 |
|------|-------------|---------|---------|-----------------|--------|--------|--------|---------|--------|-----------|-----|
| 2026-04-14 | `feature/v1.12-cascade-tech-debt`, local thin run | NFCorpus | 50 | `sentence-transformers/all-MiniLM-L6-v2` | 0.4300 | 0.3907 | 0.3560 | 0.3456 | 0.1239 | 0.1660 | 0.5000 |

_Add new rows when you change default embed models, BEIR limits, or the thin harness._

**Last Updated**: April 14, 2026  
**Goal**: Become the most trustworthy, observable, and production-ready multi-agent memory system in 2026.
