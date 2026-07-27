# ADR-010: Intelligent Self-Curation (Diff #6 productize)

<!-- INIT-010/SPEC-001 -->

**Status:** Accepted
**Date:** 2026-07-27
**Deciders:** Archivist maintainers
**Source:** [BRAIN-005 — Complete Unique Differentiators](../../sdd/brainstorms/BRAIN-005-complete-unique-differentiators/decision-document.md);
→ curator / resolve / reflection plumbing in tree;
→ [INIT-010](../../sdd/initiatives/INIT-010-intelligent-self-curation/INIT-010-intelligent-self-curation-initiative.md);
→ prior: [ADR-009](ADR-009-native-multi-agent-coordination.md) (share↔resolve gates),
[ADR-007](ADR-007-procedural-memory-wedge.md) / [ADR-008](ADR-008-retire-skills-tip-lessons.md).

## Decision

**Productize** Unique Differentiator **#6** (Intelligent Self-Curation) on the
**existing** curator cycle, contradiction resolver, reflection, decay/hotness,
and compaction helpers — do **not** invent a new lifecycle engine, do **not**
build a skill OS, and do **not** pull Diff #4 / #7 / #8 or Phase 7 institutional
tier DDL into this INIT.

### Plumbing vs Diff #6 product

| Layer | Meaning | Status entering INIT-010 |
|---|---|---|
| **Lifecycle plumbing** | `lifecycle/curator.py` scheduled cycle; `contradiction_resolve.py`; `reflection.py`; decay/hotness; write-time L0/L1 tiering; manual `archivist_compress` | **Shipped** (flags mostly off / dry-run) |
| **Differentiator #6 product** | Documented self-curation loop: reconsolidation + relevance forget + contradiction resolve with **safe defaults**, audit, and tests — honest ROADMAP “Done” | **This INIT** |

“Flags exist” is not Done. Diff #6 Done means an operator can enable and measure a
**first-class product loop**, not only call `archivist_compress` by hand.

### Product contract (INIT-010)

1. **Reconsolidation** — Curator cycle (or ADR-named entrypoint) performs
   hierarchical reconsolidation / auto-summarization using existing tiering /
   compaction helpers where practical (SPEC-002).
2. **Relevance forget** — Product path for relevance-based forgetting / decay
   beyond raw TTL delete or coach `archivist_delete` alone (SPEC-003).
3. **Contradiction resolve** — Curator cycle invokes resolve under the flag
   matrix below; share `attach_conflict` mutating apply remains write-gated
   (INIT-009 / SEC-009) (SPEC-003).
4. **Reflection** — Optional deepen when `REFLECTION_ENABLED`; **not** required
   on by default for Diff #6 Done.
5. **No net-new core MCP tools** unless a later ADR amendment proves a coach-core
   gap — prefer lifecycle flags + ops/admin surfaces (GR-PROD-002).

### Safe defaults (GR-SAFE-001)

OSS / default settings **must not** silently mutate fleet graphs:

| Setting | Default (shipped / keep) | Product meaning |
|---|---|---|
| `CONTRADICTION_RESOLVE_ENABLED` | `false` | Master switch for resolve in curator cycle |
| `CONTRADICTION_RESOLVE_DRY_RUN` | `true` | When enabled, prefer propose/audit before mutate |
| `CONTRADICTION_RESOLVE_LLM_ENABLED` | `false` | Optional LLM assist for proposals |
| `CONTRADICTION_RESOLVE_MAX_PER_CYCLE` | `20` | Cap blast radius per cycle |
| `REFLECTION_ENABLED` | `false` | Optional reflection tip write |
| `REFLECTION_DRY_RUN` | `true` | Prefer dry-run when reflection on |
| `REFLECTION_MAX_PER_CYCLE` | `20` | Cap |
| `CURATOR_INTERVAL_MINUTES` | `30` | Scheduled curator cadence |
| `RECONSOLIDATION_ENABLED` | `false` | Master switch for hierarchical reconsolidation in curator |
| `RECONSOLIDATION_DRY_RUN` | `true` | When enabled, prefer propose/audit before writing L1 summary |
| `RELEVANCE_FORGET_ENABLED` | `false` | Master switch for hotness/TTL relevance forget in curator |
| `RELEVANCE_FORGET_DRY_RUN` | `true` | When enabled, prefer propose/audit before suppress |
| Temporal / hotness decay | existing defaults | Age-based fact decay + hotness; forget uses `RELEVANCE_FORGET_*` |

**Staged enablement (operators):**

1. Run curator with resolve **disabled** (default) — reconsolidation/forget paths
   as implemented may still run if ADR/SPEC define them as non-resolve actions.
2. Enable `CONTRADICTION_RESOLVE_ENABLED=true` with **`DRY_RUN=true`** — audit
   proposals only.
3. Flip `CONTRADICTION_RESOLVE_DRY_RUN=false` only after reviewing audit volume.
4. Share attach `apply=true` + `dry_run=false` additionally requires namespace
   **write** + resolve enabled (INIT-009).

INIT-010 **must not** flip global mutate-on-by-default for OSS. Productize =
documented loop + tested paths + honest docs, with optional recommended
“ops profile” env snippet — not forcing production mutation.

### Flag / module matrix

| Concern | Module(s) | Primary flags / knobs |
|---|---|---|
| Curator cycle | `lifecycle/curator.py`, `main.py` curator loop | `CURATOR_INTERVAL_MINUTES`, extract prefixes |
| Reconsolidation / summarize | `lifecycle/reconsolidation.py` + `write/tiering.py` | `RECONSOLIDATION_ENABLED` (default false), `RECONSOLIDATION_DRY_RUN` (default true), max groups/chunks knobs |
| Contradiction detect/resolve | `lifecycle/contradiction_resolve.py`, write-time `conflict_detection` | `CONTRADICTION_RESOLVE_*` |
| Reflection tips | `lifecycle/reflection.py` | `REFLECTION_*` |
| Relevance / decay | `lifecycle/relevance_forget.py` + hotness + curator fact decay | `RELEVANCE_FORGET_*`, `HOTNESS_*`, `TEMPORAL_DECAY_*`, TTL |
| Share → resolve | `tools_coordination.share_attach_conflict` | write RBAC + `CONTRADICTION_RESOLVE_ENABLED` when mutating |

### Frozen guardrails

| ID | Rule |
|---|---|
| **GR-DIFF6-001** | Diff #6 = **productize** curator/resolve/forget/reconsolidation — not inventing a parallel lifecycle stack. |
| **GR-SAFE-001** | Defaults must not silently mutate fleet graphs; dry-run / staged enablement; every mutate audited. |
| **GR-PROD-002** | **No net-new core MCP tools** without ADR exception. |
| **GR-LAYER-001** | Memory layer only — no skill OS (ADR-007/008). |
| **GR-TIER-001** | **No** institutional tier / Phase 7 multi-tier DDL this INIT. |
| **GR-WEDGE-001** | Diff #6 only — no MaP MCP (#4), checkpoint ops (#7), or UI billboard (#8). |
| **GR-SHARE-001** | Do not regress INIT-009 share attach mutating-apply gates. |
| **GR-SCHEMA-001** | Prefer no DROP; new columns only if reconsolidation truly needs them (`db_migration` HITL). |
| **GR-CE-001** / **GR-COACH-001** (carry) | Cite-or-refuse; `-m coach_core` / `agentic_memory` green. |

### Diff #6 Done criteria (ROADMAP claim)

Diff #6 may be marked **Done** only when all hold:

1. ADR-010 Accepted (this doc).
2. Documented product loop: reconsolidation + relevance forget + contradiction
   resolve path with tests (INIT-010 SPEC-002…004).
3. Flag matrix + staged enablement published in REFERENCE / CHANGELOG (SPEC-005).
4. Security Review: 0 unresolved Critical/High (SPEC-006).
5. Architecture Mermaid for the loop (SPEC-007).
6. `-m coach_core` and `-m agentic_memory` green.

### Success spirit (INIT-010 SMs)

| ID | Spirit |
|---|---|
| **SM-001** | ADR-010 Accepted; Diff #6 Done criteria explicit. |
| **SM-002** | Product loop implemented + tested (reconsolidate / forget / resolve). |
| **SM-003** | Safe defaults + audit; no silent OSS mutate-on. |
| **SM-004** | Marker suites green. |
| **SM-005** | Security Review pass. |
| **SM-006** | ROADMAP #6 → Done; Immediate Next → INIT-011. |

## Context

After INIT-009 (PR #53), coordination is productized. Self-curation remains the
suggested next differentiator (BRAIN-005), but the codebase already has curator
and resolve plumbing with conservative flags. Without this ADR, SPEC-002…005
would re-open: whether to flip resolve on by default; whether Done requires a
new MCP tool on core; whether institutional tier belongs here.

## Alternatives considered

| Option | Description | Why not chosen |
|---|---|---|
| **Invent new lifecycle engine** | Greenfield reconsolidation service | Plumbing exists; wastes INIT budget |
| **Flip resolve mutate-on by default** | `ENABLED=true`, `DRY_RUN=false` in OSS | Violates GR-SAFE-001 |
| **Docs-only Diff #6 Done** | Mark Done without product loop | Dishonest claim |
| **Pull MaP / checkpoint / UI** | Multi-diff mega INIT | Violates GR-WEDGE-001 |
| **Institutional tier now** | Phase 7 DDL | GR-TIER-001; INIT-012 |
| **Require reflection on** | Always write reflection tips | Optional deepen only |

## Consequences

**Positive:**

- Honest Diff #6 / Phase 8 storytelling.
- Operators get a clear enablement ladder.
- SPEC-002/003 have a frozen contract.

**Negative / accepted trade-offs:**

- Defaults stay conservative — “automatic” curation still needs operator enablement for mutate.
- Reflection may remain off for many fleets.

## Non-goals (INIT-010)

- Net-new **core** MCP tools (unless amended)
- Skill registry / skill OS
- Institutional tier DDL / Phase 7 taxonomy
- Diff #4 MaP MCP, Diff #7 checkpoint ops/HITL, Diff #8 UI billboard
- Replacing Answer Finder packing or handoff

## References

- Modules: `src/archivist/lifecycle/curator.py`, `contradiction_resolve.py`, `reflection.py`
- Config: `src/archivist/core/config.py` (`contradiction_resolve_*`, `reflection_*`, `curator_interval_minutes`)
- ROADMAP Unique Differentiator #6; Phase 8 Intelligent Lifecycle Management
- BRAIN-005 completion program → next INIT-011 after Mode E
