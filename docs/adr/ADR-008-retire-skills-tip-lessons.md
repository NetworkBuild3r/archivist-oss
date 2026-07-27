# ADR-008: Retire skill registry; tip-only lessons

<!-- INIT-008/SPEC-001 -->

**Status:** Accepted
**Date:** 2026-07-26
**Deciders:** Archivist maintainers
**Source:** [BRAIN-004 — INIT-008 choice (amended)](../../sdd/brainstorms/BRAIN-004-init-008-choice/decision-document.md)
→ tip path: [ADR-007](ADR-007-procedural-memory-wedge.md) / INIT-007 / PR #49;
→ eval gym: [ADR-006](ADR-006-agentic-memory-eval-gym.md) / INIT-006 / PR #48;
→ [INIT-008](../../sdd/initiatives/INIT-008-retire-skills-tip-lessons/INIT-008-retire-skills-tip-lessons-initiative.md).

## Decision

**Retire** Archivist’s dedicated MCP **skill registry** as a product surface.
Procedural **lessons** live only on the tip path already productized in
ADR-007:

**trajectory → `tips` → `archivist_get_context` (`include_tips`) / handoff.**

Agents share lessons via **handoff** (tips already travel in `HandoffPacket`)
and, later, Unique Differentiator **#5** (Native Multi-Agent Coordination /
`share_*` depth) as **INIT-009** — **not** via `skill_lessons` or a skill OS.

Do **not** bridge `skill_lessons` into tips. Delete the registry rather than
merge dual stores.

### Tools removed (breaking for ops/full callers)

| Tool | Former role |
|---|---|
| `archivist_register_skill` | Register/update skill (MCP tool) catalog entry |
| `archivist_skill_event` | Log invocation outcome for health scoring |
| `archivist_skill_lesson` | Record lessons against a skill id |
| `archivist_skill_health` | Health grade / substitutes |
| `archivist_skill_relate` | Skill dependency/similarity edges |
| `archivist_skill_dependencies` | Relation graph read |

These tools were **never** on the default **core** coach profile (ops/full
only). Removal is still a **breaking** change for any ops/full client that
called them.

### Frozen guardrails

| ID | Rule |
|---|---|
| **GR-RETIRE-001** | **Remove, don’t deprecate-in-place.** Skill MCP tools and `features/skills.py` leave the tree; no “deprecated but present” shims. |
| **GR-TIP-001** | **Tip-only lessons.** Trajectory tips + get_context / handoff are the lesson API (ADR-007 GR-PROC-001 / GR-SURF-001 / GR-RANK-001 remain). |
| **GR-BRIDGE-001** | **No skill↔tips bridge.** Do not project or dual-write `skill_lessons` into `tips`. |
| **GR-SKILL-001** | **Superseded.** ADR-007’s “park skill-registry / don’t merge into core” is replaced by **retirement** under this ADR. |
| **GR-MEMTYPE-001** | **Keep** procedural retrieval enum `memory_type=skill` — it is **not** the MCP skill registry. |
| **GR-SCHEMA-001** | **Dormant tables OK this INIT.** Stop greenfield DDL / schema_guard creation of `skills*` tables; do **not** require DROP migration in INIT-008. |
| **GR-DIFF5-001** | **Differentiator #5 is INIT-009**, not INIT-008. ROADMAP may point ahead; no coordination product work here. |
| **GR-PROD-002** / **GR-LAYER-001** / **GR-COACH-001** (carry) | No net-new core MCP tools; memory layer only; `-m coach_core` stays green. |
| **GR-CE-001** / **GR-DUR-001** / **GR-EVAL-001** (carry) | CE, durable store-ack, and agentic_memory contracts unchanged. |

### Success spirit (INIT-008 SMs)

| ID | Spirit |
|---|---|
| **SM-001** | Zero skill-registry MCP tools in the tool registry. |
| **SM-002** | `features/skills.py` and `tools_skills.py` deleted; no live imports. |
| **SM-003** | REFERENCE / CURSOR_SKILL document tip-only lessons; no skill-registry tool section. |
| **SM-004** | ROADMAP Immediate Next: INIT-008 then INIT-009 Diff #5. |
| **SM-005** | `coach_core` + `agentic_memory` green after removal. |
| **SM-006** | Security Review: 0 unresolved Critical/High. |

## Context

After INIT-007, tips surface and condition correctly on the core coach path.
The ops/full skill registry remained a **second** place for “lessons”
(`skill_lessons`) plus tool-inventory concerns (version/health/relations).
BRAIN-004 initially scored a skill↔tips **bridge**; operator correction
rejected that dual-store approach: **strip skills** and aim Unique
Differentiators (#5 next).

Without this ADR, SPEC-002…004 would re-open:

- Whether to bridge lessons into tips (no — GR-BRIDGE-001)
- Whether to keep deprecated skill tools (no — GR-RETIRE-001)
- Whether to DROP tables now (no — GR-SCHEMA-001)
- Whether to ship Diff #5 inside INIT-008 (no — GR-DIFF5-001)

## Alternatives considered

| Option | Description | Why not chosen |
|---|---|---|
| **Skill↔tips bridge** | Read-path project lessons into get_context | Dual store; skill OS creep; rejected in BRAIN-004 amendment |
| **Deprecate but keep tools** | Mark docs deprecated; leave handlers | Confuses operators; GR-RETIRE-001 |
| **DROP tables in INIT-008** | Migration / destructive DDL | Extra one_way gate; dormant data acceptable |
| **Diff #5 in INIT-008** | Start coordination wedge now | Blows sizing; retirement is the scoped INIT |
| **Freeze Archivist** | Only Domain MCP | Rejected — tip path + differentiator calendar still matter |

## Consequences

**Positive:**

- One procedural lesson story (tips).
- Smaller ops/full attack and doc surface.
- ROADMAP can point at Differentiator #5 without skill-registry noise.

**Negative / accepted trade-offs:**

- Breaking for any ops/full skill-tool callers (none on core).
- Existing `skills*` rows may linger dormant until a later DROP INIT.
- Tool-inventory / skill-health observability goes away (not a Unique Differentiator).

## Non-goals (INIT-008)

- Skill↔tips bridge or Letta-style skill OS
- Implementing Differentiator #5 / #6 / #7 / #8
- DROP of existing `skills*` tables (unless a later ADR)
- Changing `memory_type=skill` retrieval semantics
- Net-new core MCP tools
- Edits inside `myaifitness-android` / Domain MCP
- Re-opening ADR-003 / ADR-004 / ADR-006 / ADR-007 tip contracts (except GR-SKILL-001 supersession)

## Implementing specs

| Spec | What it delivers |
|---|---|
| [SPEC-001](../../sdd/initiatives/INIT-008-retire-skills-tip-lessons/specs/SPEC-001-docs-adr-retire-skills.md) | This ADR; ROADMAP Immediate Next |
| [SPEC-002](../../sdd/initiatives/INIT-008-retire-skills-tip-lessons/specs/SPEC-002-service-remove-skills.md) | Delete skills feature/handlers; unwind call sites |
| [SPEC-003](../../sdd/initiatives/INIT-008-retire-skills-tip-lessons/specs/SPEC-003-infrastructure-skills-tests.md) | Tests/fixtures without skill registry |
| [SPEC-004](../../sdd/initiatives/INIT-008-retire-skills-tip-lessons/specs/SPEC-004-docs-tip-only-lessons.md) | Tip-only product docs |
| [SPEC-005](../../sdd/initiatives/INIT-008-retire-skills-tip-lessons/specs/SPEC-005-docs-security-review.md) / [SPEC-006](../../sdd/initiatives/INIT-008-retire-skills-tip-lessons/specs/SPEC-006-docs-architecture-diagrams.md) | Security review and architecture diagrams |

See [`docs/ROADMAP.md`](../ROADMAP.md) for the product-facing pointer. Follow-on:
**INIT-009** — Unique Differentiator #5 (Native Multi-Agent Coordination).
