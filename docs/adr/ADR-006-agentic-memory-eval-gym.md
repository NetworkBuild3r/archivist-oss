# ADR-006: Agentic memory eval gym (MemoryArena-inspired)

<!-- INIT-006/SPEC-001 -->

**Status:** Accepted
**Date:** 2026-07-26
**Deciders:** Archivist maintainers
**Source:** [BRAIN-003 — What to do next (post coach track + memory research)](../../sdd/brainstorms/BRAIN-003-post-coach-next-from-memory-research/decision-document.md)
→ [RSCH-001 — Latest AI/agent memory research & Archivist improvement bets](../../sdd/research/RSCH-001-latest-ai-agent-memory/RSCH-001-latest-ai-agent-memory-report.md)
→ [INIT-006](../../sdd/initiatives/INIT-006-agentic-memory-eval-gym/INIT-006-agentic-memory-eval-gym-initiative.md);
follows completed INIT-003→005 ([ADR-003](ADR-003-coach-core-reliability.md),
[ADR-004](ADR-004-llm-native-coach-memory-surfaces.md),
[ADR-005](ADR-005-coach-path-performance.md); PR #47 merged).

## Decision

Freeze Archivist’s **agentic memory evaluation contract** so INIT-006 specs do
not re-litigate whether we port MemoryArena wholesale, whether action loops
belong in production MCP, or whether `coach_core` may be replaced.

Passive recall / CE / write-path durability are already locked by ADR-003…005.
2025–26 research (MemoryArena and surveys cited in RSCH-001) shows those locks
are necessary but **not sufficient**: systems that ace LoCoMo-style recall can
still fail when memory must change later *actions* across interdependent
sessions. This ADR freezes how Archivist proves **memory → action** in CI
without becoming an agent runtime.

### Frozen eval guardrails

| ID | Rule |
|---|---|
| **GR-EVAL-001** | **MemoryArena-*inspired* only.** Multi-session interdependent Memory→Action scenarios in CI. **No** full MemoryArena / web-nav gym port in INIT-006. |
| **GR-EVAL-002** | **CI-shaped stubs.** Scenarios use SQLite, fake/stub embed, and test outbox drain helpers. No live external agent harness or live Qdrant required for the gate. |
| **GR-LAYER-001** | **Archivist stays a memory layer.** Action selection lives in **tests** (harness oracle), not a production agent OS / Letta-shaped runtime. No net-new MCP tools for “run the agent.” |
| **GR-CE-001** (carry) | **Cite-or-refuse / empty-evidence-OK / map-only index** (ADR-004) must not regress. Index TOC is never sufficient evidence for action. |
| **GR-DUR-001** (carry) | **Store ack = durable graph + outbox** (ADR-003). Do not reintroduce sync Qdrant on ack for eval convenience. |
| **GR-COACH-001** | **`coach_core` remains required.** Existing `-m coach_core` suite must stay green. Agentic scenarios use a **separate** marker and must not replace or dilute coach_core. |

### Pytest marker

| Item | Decision |
|---|---|
| **Marker name** | **`agentic_memory`** |
| **Relation to `coach_core`** | **Sibling**, not a rename. `coach_core` stays lean (store/index/search/get_context, CE, perf/lag). `agentic_memory` owns Memory→Action multi-session asserts. |
| **Rationale** | Independent CI selection; avoids ballooning coach_core wall-clock; clear provenance for SM-001/SM-002. |

### What “action” means in INIT-006

| Item | Contract |
|---|---|
| **Action** | Discrete id chosen by a **test-only oracle** from retrieved provenance-bearing `memories[]` / get_context packs (e.g. `order_express`, `refuse`). |
| **Not action** | Live side-effects in production, Domain MCP packet assembly, or promoting an agent loop into the MCP server. |
| **Evidence** | Only provenance-bearing memories (and documented empty/refuse paths). Never index markdown alone. |

### Success spirit (initiative SMs)

| ID | Spirit |
|---|---|
| **SM-001** | ≥1 green scenario: omit Session-A store → Session-B action fails or refuses. |
| **SM-002** | ≥1 green scenario: stale/contradictory/ambiguous memory does not invent facts. |
| **SM-003** | `pytest -m coach_core` remains green. |

Exact scenario files and QA playbook text are owned by later INIT-006 specs;
this ADR freezes the **contract**, not the fixture prose.

## Context

INIT-001 shipped platform + differentiator wedges. INIT-003→005 (PR #47
**merged** 2026-07-26) closed coach reliability, LLM-native CE surfaces, and
write-path performance. `docs/ROADMAP.md` Immediate Next still pointed at
INIT-005 and was stale.

BRAIN-003 / RSCH-001 scored next bets: **agentic eval gym** ahead of procedural
memory, hierarchical reconsolidation, and multi-agent coherence protocols —
because without action-coupled AC, mechanism PRs are hard to falsify.

Without this ADR, implementing specs would re-open:

- Whether to vendor/port MemoryArena (no — GR-EVAL-001)
- Whether to ship an in-process agent runtime (no — GR-LAYER-001)
- Whether agentic asserts may fold into `coach_core` only (no — separate marker)
- Whether inventing facts from TOC / empty packs is allowed for “green” tests (no)

ADR-003…005 non-goals still apply. This ADR does not reopen CE map/bootstrap,
embed-defer defaults, or outbox durability.

## Alternatives considered

| Option | Description | Why not chosen |
|---|---|---|
| **Full MemoryArena port** — web-nav / multi-domain gym | Strongest external benchmark alignment | Out of sizing/CI budget; GR-EVAL-001; inspired scenarios suffice for Archivist’s memory-layer role |
| **Fold into `coach_core` only** — no new marker | Fewer markers | Inflates coach_core time and blurs recall/CE vs action coupling; BRAIN-003 prefers sibling marker |
| **Production action loop in MCP** — “agentic tools” | Closer to Letta | Category error for a memory layer (RSCH-001); GR-LAYER-001 / GR-PROD-002 |
| **Procedural memory product first (INIT-007)** | Visible differentiator | RSCH-001: eval before mechanism; park until SM-001/002 exist |
| **Soak / docs-only after INIT-005** — no eval INIT | Avoid test complexity | Leaves MemoryArena gap unmeasured; ROADMAP thrash continues |

## Consequences

**Positive:**

- SPEC-002…007 inherit frozen marker, stub/CI rules, layer boundary, and
  coach_core non-regression — no re-deciding mid-implementation.
- ROADMAP Immediate Next can point at a single authoritative next INIT.
- Operators get a clear story: recall/CE/perf locked; next prove memory changes
  action.

**Negative / accepted trade-offs:**

- Inspired scenarios will not match MemoryArena leaderboard numbers; accepted.
- Test-only oracle is a simplified stand-in for real Domain MCP assembly;
  consumer parallel track (BRAIN-002/003) still owns Need→Budget→Assemble→Verify.
- Separate marker means two CI invocations to claim full coach+agentic coverage.

## Non-goals (INIT-006)

Explicitly **out of scope** for work governed by this ADR:

- Full MemoryArena / web-nav / multi-domain gym port (GR-EVAL-001)
- Procedural skill-library product (parked INIT-007)
- Hierarchical HiMem/HiGMem reconsolidation
- Multi-agent MESI / coherence protocol beyond existing `share_*`
- Enabling `ARCHIVIST_EMBED_DEFER` by default (ADR-005)
- Net-new MCP tools / tool sprawl; default profile remains **core**
- Edits inside `myaifitness-android` or Domain MCP packet builders
- Replacing or weakening `-m coach_core` (GR-COACH-001)
- Re-opening ADR-003 durability or ADR-004 CE contracts

## Implementing specs

| Spec | What it delivers |
|---|---|
| [SPEC-001](../../sdd/initiatives/INIT-006-agentic-memory-eval-gym/specs/SPEC-001-docs-adr-agentic-memory-eval.md) | This ADR; ROADMAP Immediate Next → INIT-006 |
| [SPEC-002](../../sdd/initiatives/INIT-006-agentic-memory-eval-gym/specs/SPEC-002-infrastructure-agentic-memory-harness.md) | Harness + `agentic_memory` marker + test-only action oracle |
| [SPEC-003](../../sdd/initiatives/INIT-006-agentic-memory-eval-gym/specs/SPEC-003-infrastructure-positive-scenarios.md) | Positive multi-session memory→action (+ omit-store control) |
| [SPEC-004](../../sdd/initiatives/INIT-006-agentic-memory-eval-gym/specs/SPEC-004-infrastructure-negative-scenarios.md) | Stale / contradictory / ambiguous negatives |
| [SPEC-005](../../sdd/initiatives/INIT-006-agentic-memory-eval-gym/specs/SPEC-005-docs-qa-agentic-memory.md) | QA playbook + baselines |
| [SPEC-006](../../sdd/initiatives/INIT-006-agentic-memory-eval-gym/specs/SPEC-006-docs-security-review.md) / [SPEC-007](../../sdd/initiatives/INIT-006-agentic-memory-eval-gym/specs/SPEC-007-docs-architecture-diagrams.md) | Security review and architecture/design diagrams (initiative close) |

See [`docs/ROADMAP.md`](../ROADMAP.md) for the product-facing pointer to this
ADR, and BRAIN-003 / RSCH-001 for the eval-before-mechanism sequencing.
