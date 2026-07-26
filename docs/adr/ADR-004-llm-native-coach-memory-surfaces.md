# ADR-004: LLM-native coach memory surfaces

<!-- INIT-004/SPEC-002 -->

**Status:** Accepted
**Date:** 2026-07-26
**Deciders:** Archivist maintainers
**Source:** [BRAIN-002 — LLM-native index & coach-path performance](../../sdd/brainstorms/BRAIN-002-llm-index-and-coach-performance/decision-document.md)
→ [INIT-004](../../sdd/initiatives/INIT-004-llm-native-coach-memory-surfaces/INIT-004-llm-native-coach-memory-surfaces-initiative.md)

## Decision

Freeze Archivist’s **LLM-native coach memory surface contract** so later INIT-004
specs do not re-litigate index-as-evidence vs map, bootstrap mechanism, or
performance bundling.

[ADR-003](ADR-003-coach-core-reliability.md) remains the authoritative
**coach-core five-tool contract**. This ADR **extends** that contract with
context-engineering (CE) rules for how `archivist_index` and
`archivist_get_context` must behave as **ingredients** for LLM turns — not as
a maximal chat dump or citable fact store.

### Frozen CE guardrails

| ID | Rule |
|---|---|
| **GR-CE-001** | **Index = progressive-disclosure map, not evidence.** `archivist_index` returns navigational pointers (entities, types, search hints). It must **not** contain citable key-fact prose that invites treating the TOC as evidence. |
| **GR-CE-002** | **No synchronous LLM on `archivist_index`.** “LLM-optimize the index” means shape, budget, and progressive disclosure — **not** model-generated TOC on the hot path. |
| **GR-CE-003** | **Empty recall is OK.** When filters/thresholds yield nothing, an empty `memories[]` (or equivalent) is a valid success — cite-or-refuse; do not invent facts. |

### Bootstrap mechanism

- **Session bootstrap on the core path** = `archivist_get_context(mode=bootstrap)`
  (or equivalent `mode=bootstrap` on the get_context surface).
- **`archivist_wake_up` stays on ops/full profiles** — do **not** promote it into
  the default **core** tool set in this initiative.
- Recommended session sequence for coach consumers: bootstrap via
  `get_context(mode=bootstrap)` → turn evidence via `search` /
  `get_context` (budgeted) → navigational refresh via `archivist_index` as a
  **map only**.

### Consumer placement guidance

Archivist supplies thresholded, provenance-bearing, budget-aware surfaces.
Domain MCP / harness packet builders own Need→Budget→Assemble→Verify. Placement
guidance for consumers of Archivist ingredients:

1. **Evidence high** — put provenance-bearing `memories[]` (or equivalent
   citable payloads) in decisive positions (start/end of the assembled pack);
   do not bury them mid-window.
2. **Cite-or-refuse** — agents must cite only from those provenance-bearing
   memories; refuse when empty rather than inventing from the index map.
3. **Empty OK** — empty packs under tight budgets/filters are correct behavior
   (GR-CE-003), not a failure to “fill the window.”

Optional dual index shape (compact structured block **plus** short markdown)
remains allowed under the **single** `archivist_index` tool — still one tool,
still map-only (GR-PROD-002 / GR-CE-001).

### Parked: INIT-005 (not this INIT)

**Embed-deferred store / write-path performance work** (hard-skip optional write
gates under ack budget; searchable-lag SLO; embed reuse) is explicitly parked
as **INIT-005**. Do not bundle it into INIT-004. GR-CE-002 still forbids LLM
on the index path as a “performance” fix.

## Context

INIT-003 / ADR-003 made the five-tool coach path reliable under personal
production. BRAIN-002 then asked what to do next for (1) coach-path performance
and (2) LLM-consumable index/recall surfaces. The panel chose **measure-first,
then split**: INIT-004 for LLM-native surfaces; INIT-005 for embed-defer and
related perf — not one mega-INIT.

Without this ADR, implementing specs would re-open:

- Whether the MEMORY INDEX may keep key-fact prose (it must not — GR-CE-001)
- Whether “LLM-optimize index” means calling a model on every index call (it
  must not — GR-CE-002)
- Whether empty filtered recall is an error (it is not — GR-CE-003)
- Bootstrap via `wake_up` in core vs `get_context(mode=bootstrap)` (bootstrap
  mode wins; wake_up stays ops/full)
- Whether embed-defer belongs in the same initiative (parked as INIT-005)

ADR-003’s non-goals (MaP MCP, `share_*`, Phase 7–10 breadth, OSS billboard
success) still apply. This ADR does not reopen that product track.

## Alternatives considered

| Option | Description | Why not chosen |
|---|---|---|
| **Index as evidence dump** — keep key-fact prose in TOC so agents “have facts without search” | Familiar today; fewer discovery round-trips | Violates cite-or-refuse; TOC invites inventing citations from non-provenance text. Progressive-disclosure **map** wins (GR-CE-001). |
| **LLM-rewrite index on every call** — generate a model-written TOC for “LLM-native” quality | Maximal LLM shaping of the surface | Hot-path cost and non-determinism; BRAIN-002 rejected Option E. Shape/budget instead (GR-CE-002). |
| **Promote `archivist_wake_up` into core** — session bootstrap as a sixth core tool | Clear named bootstrap tool | GR-PROD-002 / BRAIN-002: enrich `get_context` with `mode=bootstrap`; keep wake_up on ops/full unless a later AC proves insufficiency. |
| **Bundle embed-defer into INIT-004** — one INIT for surfaces + write-path perf | Fewer initiative IDs | BRAIN-002 Option C: split so CE surface work is not blocked by perf design; park INIT-005. |
| **Soak INIT-003 only** — no surface reshape | Avoids churn | Leaves index inviting TOC-as-evidence and no frozen bootstrap path for coach profiles. |

## Consequences

**Positive:**

- SPEC-003…006 inherit frozen map-only index, bootstrap mode, empty-OK, and
  consumer placement guidance — no re-deciding CE rules mid-implementation.
- Coach_core evals can assert token ceilings and “TOC must not contain citable
  fact prose” against a written contract.
- INIT-005 remains a clean follow-on for embed-defer without scope creep here.

**Negative / accepted trade-offs:**

- Slimmer index may reduce “facts at a glance”; mitigated by entity names +
  search hints and coach_core discovery evals.
- Bootstrap mode adds a documented `get_context` mode rather than a new tool —
  harness/docs must teach one recommended sequence.
- Write-path latency improvements wait for INIT-005 even if desirable now.

## Non-goals (INIT-004)

Explicitly **out of scope** for work governed by this ADR:

- Synchronous LLM generation of the MEMORY INDEX / TOC
- Embed-deferred store, write-path hard-skips, searchable-lag SLO (**INIT-005**)
- Promoting `archivist_wake_up` into the default **core** profile
- MaP MCP, `share_*`, checkpoint UX, Phase 7–10 product breadth
- Edits inside `myaifitness-android` or Domain MCP implementation (consumer /
  parallel track)
- Weakening cite-or-refuse or treating empty recall as failure

## Implementing specs

| Spec | What it delivers |
|---|---|
| [SPEC-001](../../sdd/initiatives/INIT-004-llm-native-coach-memory-surfaces/specs/SPEC-001-infrastructure-coach-path-timing-baselines.md) | Stage timing baselines + QA playbook (complete) |
| [SPEC-002](../../sdd/initiatives/INIT-004-llm-native-coach-memory-surfaces/specs/SPEC-002-docs-adr-llm-native-surfaces.md) | This ADR; ROADMAP “Immediate Next Steps” pointer (current) |
| SPEC-003 | Reshape `archivist_index` map contract |
| SPEC-004 | `get_context` budgets + `mode=bootstrap` |
| SPEC-005 | Map-only compressed index builder |
| SPEC-006 | coach_core CE evals |
| SPEC-007 / SPEC-008 | Security review and architecture/design diagrams (initiative close) |

See [`docs/ROADMAP.md`](../ROADMAP.md) for the product-facing pointer to this
ADR, and BRAIN-002 for the measure-first / split rationale.
