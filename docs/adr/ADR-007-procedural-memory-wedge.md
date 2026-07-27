# ADR-007: Procedural memory wedge (tips-first)

<!-- INIT-007/SPEC-001 -->

**Status:** Accepted
**Date:** 2026-07-26
**Deciders:** Archivist maintainers
**Source:** [BRAIN-003 — What to do next (post coach track + memory research)](../../sdd/brainstorms/BRAIN-003-post-coach-next-from-memory-research/decision-document.md)
→ [RSCH-001 — Latest AI/agent memory research & Archivist improvement bets](../../sdd/research/RSCH-001-latest-ai-agent-memory/RSCH-001-latest-ai-agent-memory-report.md)
→ unlock: [ADR-006](ADR-006-agentic-memory-eval-gym.md) / INIT-006 / PR #48 **merged**
(Memory→Action SM-001/SM-002 exist);
→ [INIT-007](../../sdd/initiatives/INIT-007-procedural-memory-wedge/INIT-007-procedural-memory-wedge-initiative.md);
follows ADR-003…006.

> **Superseded for skills (2026-07-26):** [ADR-008](ADR-008-retire-skills-tip-lessons.md)
> **retires** the skill registry. ADR-007 tip-path guardrails
> (GR-PROC-001 / GR-SURF-001 / GR-RANK-001 / GR-SHAPE-001) remain in force;
> **GR-SKILL-001** (“park registry / don’t merge into core”) is replaced by
> retirement — do **not** bridge `skill_lessons` into tips.

## Decision

Freeze Archivist’s **procedural memory product wedge** so INIT-007 specs do not
re-litigate whether to build a Letta-style skill OS, merge the ops skill-registry
into core, add net-new core MCP tools, or invent a second procedure store beside
trajectory tips.

**Contract:** productize the existing **trajectory → `tips` table →
`archivist_get_context` (`include_tips`)** path — make tips **actually surface**,
**task/query-conditioned**, and **falsifiable** via `-m agentic_memory`
procedure→action scenarios. Archivist remains a **memory layer** (ADR-006
GR-LAYER-001).

### Frozen procedural guardrails

| ID | Rule |
|---|---|
| **GR-PROC-001** | **Tips / trajectory first.** Reuse `tips` + trajectory extraction hooks. Do **not** invent a parallel procedure store or Letta-style skill OS. |
| **GR-SKILL-001** | **Superseded by [ADR-008](ADR-008-retire-skills-tip-lessons.md).** Was: skill-registry out of INIT-007 scope. Now: registry **retired**; tip-only lessons; no skill↔tips bridge. |
| **GR-SURF-001** | **Tips must surface.** `search_tips` row fields (`tip_text`) map correctly into `get_context` / handoff tip strings when `include_tips=true`. Silent empty tips are a defect. |
| **GR-RANK-001** | **Task/query-conditioned recall.** Tip selection is not pure `ORDER BY created_at DESC`; prefer relevance (keyword / fingerprint / category). Usage counters may update on retrieve. Prefer CI-deterministic ranking over a new embed dependency. |
| **GR-SHAPE-001** | **Compat shape.** Keep `tips: list[str]` on get_context responses unless a later ADR explicitly adds additive fields. |
| **GR-PROD-002** (carry) | **No net-new core MCP tools.** Default profile remains **core**. Do not promote `log_trajectory` / `archivist_tips` into core in this INIT. |
| **GR-LAYER-001** (carry) | **Memory layer only.** Procedure→action oracles stay **test-only** under `tests/` (ADR-006). |
| **GR-EVAL-001** (carry) | Extend MemoryArena-*inspired* `agentic_memory` scenarios — **no** full gym port. |
| **GR-CE-001** / **GR-DUR-001** / **GR-COACH-001** (carry) | ADR-004 CE, ADR-003 durable ack, and `-m coach_core` green remain mandatory. |

### Success spirit (initiative SMs)

| ID | Spirit |
|---|---|
| **SM-001** | Real `tip_text` rows appear in `get_context` tips when `include_tips=true` (mapping fixed + tested). |
| **SM-002** | Tip retrieval is query/task-conditioned (not pure recency) with test evidence. |
| **SM-003** | ≥1 green `agentic_memory` scenario: omit tips → action refuse/fail; with tips → correct action (**procedure helped**). |
| **SM-004** | `pytest -m coach_core` and existing `agentic_memory` suite stay green. |
| **SM-005** | Security Review: 0 unresolved Critical/High. |

Exact fixture prose and QA text are owned by later INIT-007 specs; this ADR
freezes the **contract**.

### Eval marker

| Item | Decision |
|---|---|
| **Marker** | Extend **`agentic_memory`** (no new marker) |
| **Rationale** | Procedure→action is the same Memory→Action family; avoids marker sprawl |

## Context

INIT-006 (PR #48 **merged** 2026-07-26) delivered the agentic eval gym and
unlocked BRAIN-003’s parked **INIT-007 procedural memory wedge**: mechanism work
only after evals can score “procedure helped.”

Today’s tips path already exists (`log_trajectory` → extract → `tips` table;
`get_context(include_tips=…)`), but:

- Normal get_context maps `content`/`tip` while rows use **`tip_text`** (likely
  silent empty tips).
- `search_tips` is **recency-only**; `usage_count` / `last_used_at` are unused.
- Skill registry is a **separate** ops subsystem — not a unified skill-library
  product.

Without this ADR, implementing specs would re-open:

- Whether to ship a Letta-like skill OS (no — GR-PROC-001 / GR-LAYER-001)
- Whether to fold skill-registry into core get_context (no — GR-SKILL-001)
- Whether to add core MCP tools for procedures (no — GR-PROD-002)
- Whether pure recency tip lists count as “productized” (no — GR-RANK-001)

`docs/ROADMAP.md` Immediate Next still pointed at completed INIT-006 and listed
INIT-007 as parked — stale after PR #48.

## Alternatives considered

| Option | Description | Why not chosen |
|---|---|---|
| **Skill-registry merge into core** | Unify tips + skills in get_context | GR-SKILL-001; ops blast radius; tool/profile sprawl |
| **Letta-style skill OS** | First-class procedure runtime in MCP | Category error for a memory layer; GR-LAYER-001 |
| **New procedure store / schema** | Parallel to `tips` | Duplicates trajectory tips; GR-PROC-001 |
| **Promote trajectory tools to core** | Writers in default profile | GR-PROD-002; core stays lean five-tool coach path |
| **New pytest marker** | e.g. `procedural_memory` | Unnecessary; extend `agentic_memory` |
| **Docs-only / skip tip mapping fix** | Rank without surfacing | Surfacing bug makes productization theater (GR-SURF-001) |

## Consequences

**Positive:**

- SPEC-002…007 inherit tips-first scope, shape compat, marker choice, and
  non-goals — no mid-flight skill-OS debates.
- ROADMAP Immediate Next can point at a single authoritative next INIT.
- Operators get a clear story: Memory→Action evals exist; next make procedures
  help actions for real.

**Negative / accepted trade-offs:**

- Skill-registry lessons remain invisible on the core coach path until a later
  INIT.
- Ops/full still required to *write* tips via trajectory tools; core *reads*
  via get_context (tests may seed `tips` directly).
- Keyword/fingerprint ranking will not match learned retrievers; accepted for
  CI determinism.

## Non-goals (INIT-007)

Explicitly **out of scope** for work governed by this ADR:

- Unifying / promoting skill-registry into core or get_context (GR-SKILL-001;
  **see ADR-008 — registry retired**)
- Letta-style skill OS / production agent runtime (GR-LAYER-001)
- Net-new core MCP tools or promoting `log_trajectory` into core (GR-PROD-002)
- Full MemoryArena / web-nav gym port (GR-EVAL-001)
- Hierarchical HiMem/HiGMem reconsolidation
- Multi-agent MESI coherence protocol beyond existing `share_*`
- Enabling `ARCHIVIST_EMBED_DEFER` by default (ADR-005)
- Phase 7 multi-tier memory product / checkpoint UX as this INIT’s deliverable
- Edits inside `myaifitness-android` or Domain MCP packet builders
- Replacing or weakening `-m coach_core` (GR-COACH-001)
- Re-opening ADR-003 durability, ADR-004 CE, or ADR-006 eval contracts

## Implementing specs

| Spec | What it delivers |
|---|---|
| [SPEC-001](../../sdd/initiatives/INIT-007-procedural-memory-wedge/specs/SPEC-001-docs-adr-procedural-memory.md) | This ADR; ROADMAP Immediate Next → INIT-007 |
| [SPEC-002](../../sdd/initiatives/INIT-007-procedural-memory-wedge/specs/SPEC-002-service-tip-surfacing.md) | `tip_text` → get_context / handoff mapping fix |
| [SPEC-003](../../sdd/initiatives/INIT-007-procedural-memory-wedge/specs/SPEC-003-service-conditioned-tip-recall.md) | Task/query-conditioned tip recall + usage |
| [SPEC-004](../../sdd/initiatives/INIT-007-procedural-memory-wedge/specs/SPEC-004-infrastructure-procedure-action-evals.md) | Procedure→action `agentic_memory` scenarios |
| [SPEC-005](../../sdd/initiatives/INIT-007-procedural-memory-wedge/specs/SPEC-005-docs-qa-procedural-tips.md) | QA playbook |
| [SPEC-006](../../sdd/initiatives/INIT-007-procedural-memory-wedge/specs/SPEC-006-docs-security-review.md) / [SPEC-007](../../sdd/initiatives/INIT-007-procedural-memory-wedge/specs/SPEC-007-docs-architecture-diagrams.md) | Security review and architecture diagrams |

See [`docs/ROADMAP.md`](../ROADMAP.md) for the product-facing pointer to this
ADR, and BRAIN-003 / RSCH-001 for eval-before-mechanism sequencing (now
satisfied by ADR-006).
