# ADR-009: Native Multi-Agent Coordination (Diff #5 productize)

<!-- INIT-009/SPEC-001 -->

**Status:** Accepted
**Date:** 2026-07-26
**Deciders:** Archivist maintainers
**Source:** [BRAIN-004 — INIT-008 choice (amended)](../../sdd/brainstorms/BRAIN-004-init-008-choice/decision-document.md)
→ tip path: [ADR-007](ADR-007-procedural-memory-wedge.md) / INIT-007 / PR #49;
→ skills retirement: [ADR-008](ADR-008-retire-skills-tip-lessons.md) / INIT-008 / PR #52;
→ share plumbing: INIT-001/SPEC-010 (`archivist_share_*`);
→ [INIT-009](../../sdd/initiatives/INIT-009-native-multi-agent-coordination/INIT-009-native-multi-agent-coordination-initiative.md).

## Decision

**Productize** Unique Differentiator **#5** (Native Multi-Agent Coordination)
on the **existing** selective-share and handoff surfaces — do **not** invent
`share_*` from zero, do **not** build a skill OS, and do **not** dump Phase
7–10 breadth into one INIT.

### Phase 10 plumbing vs Diff #5 product

| Layer | Meaning | Status entering INIT-009 |
|---|---|---|
| **Phase 10 plumbing** | `archivist_share_propose/accept/reject/attach_conflict/get` + `memory_share_grants` (INIT-001/SPEC-010) | **Shipped** |
| **Differentiator #5 product** | Ops-usable selective share; conflict attach aligned to contradiction resolver; tip/lesson share honesty via handoff + share; docs/ROADMAP consistent | **This INIT** |

ROADMAP may mark Phase 10 primitives complete while Diff #5 was still “Not
started” / REFERENCE called share_* an “unfinished wedge.” That is not a
contradiction once read as **plumbing vs product maturity**. INIT-009 closes
the product gap.

### Product contract (INIT-009)

1. **Promote** `archivist_share_*` into the **ops** profile (and keep **full**).
   Keep them **off core** (no net-new core MCP tools).
2. **Wire** `archivist_share_attach_conflict` to the real contradiction
   vocabulary / optional `contradiction_resolve` path — outcomes remain
   `{supersede, merge, keep_both}`, not free-form labels.
3. **Share lessons** via **handoff tips** (already in `HandoffPacket`) and/or
   selective share of tip-related memory ids — **not** via a skill registry
   (ADR-008).
4. **Institutional** in Diff #5 means **institutional use** of shared
   memory/tips across agents (shared namespace / grants) — **not** a new
   `institutional` tier table (Phase 7).

### Frozen guardrails

| ID | Rule |
|---|---|
| **GR-DIFF5-001** | Diff #5 = **productize** selective share + conflict + institutional *use* of shared memory/tips — not inventing `share_*` from zero. |
| **GR-PROD-002** | **No net-new core MCP tools.** `share_*` may appear on **ops** (+ full); **core** stays free of `share_*`. |
| **GR-LAYER-001** | Memory layer only — no Letta-style skill OS / agent runtime. |
| **GR-TIER-001** | **No** new `institutional` tier schema / Phase 7 multi-tier DDL this INIT. |
| **GR-CONSENSUS-001** | **No** multi-party quorum / Paxos / counter-propose negotiation loop this INIT. Consensus v1 remains accept/reject + audited conflict attach. |
| **GR-HANDOFF-001** | Do **not** replace `archivist_handoff` / `archivist_receive_handoff`. |
| **GR-SKILL-001** / tip-only (carry) | No skill-registry resurrection; tip-only lessons (ADR-007/008). |
| **GR-SCHEMA-001** | Prefer no DROP; new columns only if attach/resolver wiring truly requires them (`db_migration` HITL). |
| **GR-CE-001** / **GR-DUR-001** / **GR-COACH-001** (carry) | Cite-or-refuse, durable store-ack, `-m coach_core` green. |

### Success spirit (INIT-009 SMs)

| ID | Spirit |
|---|---|
| **SM-001** | ADR-009 Accepted; ROADMAP Diff #5 + Phase 10 language consistent. |
| **SM-002** | `share_*` visible on ops+full; absent from core. |
| **SM-003** | Conflict attach validates `{supersede,merge,keep_both}` with resolver-aligned path tested. |
| **SM-004** | Tip/lesson share path documented (handoff and/or share); no skill_* lesson API. |
| **SM-005** | `coach_core` + `agentic_memory` green; coordination tests green. |
| **SM-006** | Security Review: 0 unresolved Critical/High. |

## Context

After INIT-008 (PR #52), procedural lessons are tip-only. Agents still need a
**mature** way to selectively share memory and record conflict outcomes across
agent ids. The MCP tools already exist but sit behind full-only / “unfinished”
docs, and `attach_conflict` records caller-supplied outcomes without tying to
the lifecycle resolver.

Without this ADR, SPEC-002…004 would re-open:

- Whether to invent new coordination MCP tools on core (no — GR-PROD-002)
- Whether “institutional memory” means a new tier table (no — GR-TIER-001)
- Whether Diff #5 requires multi-party negotiation (no — GR-CONSENSUS-001)
- Whether Phase 10 “done” means Diff #5 is done (no — plumbing ≠ product)

## Alternatives considered

| Option | Description | Why not chosen |
|---|---|---|
| **Invent new share stack** | Greenfield coordination API | Plumbing already shipped; wastes INIT budget |
| **Promote share_* to core** | Coach default sees share tools | Violates GR-PROD-002 / coach surface budget |
| **Phase 7 institutional tier now** | New tier enum + DDL | Diff #7/#Phase-7 breadth; GR-TIER-001 |
| **Counter-propose / quorum** | Multi-round negotiation | Out of sizing; GR-CONSENSUS-001 |
| **Skill-based lesson share** | Resurrect skill_lessons for cross-agent | Rejected by ADR-008 |
| **Docs-only Diff #5** | Mark Diff #5 Done without ops promotion / conflict wire | Dishonest product claim |

## Consequences

**Positive:**

- Honest Diff #5 / Phase 10 storytelling.
- Ops agents can use selective share without full profile.
- Conflict attach becomes real coordination, not a label dump.
- Tip-only lesson story stays compatible with multi-agent transfer.

**Negative / accepted trade-offs:**

- Ops surface grows (share_* tools) — mitigated by existing RBAC + audit.
- Consensus remains two-party accept/reject (no quorum).
- Institutional *tier* remains future Phase 7 work.

## Non-goals (INIT-009)

- Net-new **core** MCP tools
- Skill registry / skill↔tips bridge
- Phase 7 multi-tier / institutional **table** DDL
- Multi-party quorum, Paxos, or counter-propose loops
- Replacing handoff
- Diff #4 MaP finish, Diff #6 self-curation depth, Diff #7 checkpoint UX, Diff #8 observability billboard
- Edits inside `myaifitness-android` / Domain MCP

## Implementing specs

| Spec | What it delivers |
|---|---|
| [SPEC-001](../../sdd/initiatives/INIT-009-native-multi-agent-coordination/specs/SPEC-001-docs-adr-diff5-productize.md) | This ADR; ROADMAP Immediate Next / Diff #5 honesty |
| [SPEC-002](../../sdd/initiatives/INIT-009-native-multi-agent-coordination/specs/SPEC-002-service-conflict-ops-tip-share.md) | Ops profile + conflict wire-up + tip-share path |
| [SPEC-003](../../sdd/initiatives/INIT-009-native-multi-agent-coordination/specs/SPEC-003-infrastructure-coordination-tests.md) | Coordination + agentic tests |
| [SPEC-004](../../sdd/initiatives/INIT-009-native-multi-agent-coordination/specs/SPEC-004-docs-product-diff5.md) | Product docs |
| [SPEC-005](../../sdd/initiatives/INIT-009-native-multi-agent-coordination/specs/SPEC-005-docs-security-review.md) / [SPEC-006](../../sdd/initiatives/INIT-009-native-multi-agent-coordination/specs/SPEC-006-docs-architecture-diagrams.md) | Security review and architecture diagrams |

See [`docs/ROADMAP.md`](../ROADMAP.md) for the product-facing pointer. Follow-on
candidates after INIT-009: Differentiator **#6** (Intelligent Self-Curation) or
other parked Diffs — choose explicitly; do not auto-start.
