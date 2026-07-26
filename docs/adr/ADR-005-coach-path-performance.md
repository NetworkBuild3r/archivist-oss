# ADR-005: Coach-path performance (embed-defer, ack budget, searchable lag)

<!-- INIT-005/SPEC-001 -->

**Status:** Accepted
**Date:** 2026-07-26
**Deciders:** Archivist maintainers
**Source:** [BRAIN-002 — LLM-native index & coach-path performance](../../sdd/brainstorms/BRAIN-002-llm-index-and-coach-performance/decision-document.md)
(§ INIT-005; parked skeleton) → [INIT-005](../../sdd/initiatives/INIT-005-coach-path-performance/INIT-005-coach-path-performance-initiative.md);
extends [ADR-003](ADR-003-coach-core-reliability.md) store-ack durability and the
INIT-005 park noted in [ADR-004](ADR-004-llm-native-coach-memory-surfaces.md).

## Decision

Freeze Archivist’s **coach write-path performance contract** so later INIT-005
specs do not re-litigate durability vs latency, silent vector lag, or which
pre-ack gates may hard-skip under budget.

[ADR-003](ADR-003-coach-core-reliability.md) remains authoritative for the
**five-tool coach-core contract** and **store ack = durable graph + outbox**.
[ADR-004](ADR-004-llm-native-coach-memory-surfaces.md) remains authoritative for
**LLM-native CE surfaces** (map-only index, bootstrap, empty-OK). This ADR
**does not reopen** either; it freezes how we cut store-path latency **without**
weakening those contracts.

### Frozen performance guardrails

| ID | Rule |
|---|---|
| **GR-DUR-001** | **Store ack = durable graph + outbox row** (INIT-003 / ADR-003). Qdrant upsert remains **async** via outbox drain. Ack must never wait on Qdrant health or completion. |
| **GR-LAG-001** | **Searchable-vector lag is explicit and measured.** After embed-defer (when enabled), FTS / needle / graph durability at ack must not be confused with “vector searchable now.” Empty or incomplete rank during lag is OK; silent pretend-complete is not. |
| **GR-CE-002** (carry) | **No synchronous LLM on `archivist_index`.** Perf work must not introduce model calls on the index hot path (ADR-004). |

### Feature flag: embed-deferred store

| Item | Decision |
|---|---|
| **Env name** | `ARCHIVIST_EMBED_DEFER` |
| **Default** | **`false` (opt-in)** |
| **Rationale** | Deferring primary embed shortens ack but widens the window where hybrid/vector search cannot see the new point. Keep the safe default (embed before outbox enqueue / current INIT-003 path) until coach_core proves the searchable-lag SLO and hard-skip metrics under load. Operators may set `true` once SPEC-005…007 land and evals pass. |

When `ARCHIVIST_EMBED_DEFER=true`:

- Store **acks** after durable graph + outbox (and FTS/needle durability as today), **without** blocking the client on the primary embed call.
- Outbox drain **embeds before** Qdrant upsert (fill deferred payload, then upsert).
- Response / observability surfaces MUST expose deferred / searchable-lag signals (SPEC-006) so clients and evals never assume vector readiness at ack.

### Ack-budget hard-skips

Under `STORE_ACK_BUDGET_MS` (ADR-003 soft budget; INIT-005 makes skips **hard** for optional gates):

| Gate / stage | Under expired budget |
|---|---|
| Conflict detection | **May hard-skip** (fail-open; metric/log when skipped) |
| Dedup | **May hard-skip** |
| Extract / other optional pre-ack enrichment | **May hard-skip** |
| **Outbox row write** | **Never skip** — durability ack still requires the outbox row (GR-DUR-001) |
| Graph commit of the memory write | **Never skip** |

Hard-skip is a **quality / latency tradeoff**, not an authz bypass. Skips must be
observable (log and/or metric) for coach_core / ops.

### Embed reuse

When conflict-detection already embedded text that is **byte-identical** to the
primary store `embed_input` (no augmentation changed the string), reuse the
conflict `_shared_vec` (or equivalent) for the primary store embed — do not
re-call the embed provider. If augmentation changes text, embed the new input
normally. Measure reuse hit/miss in stage timings (SPEC-002 / SPEC-004).

### Searchable-lag SLO (spirit + concrete target)

| Item | Contract |
|---|---|
| **Spirit** | After a successful store with deferred embed, the memory becomes **vector-searchable** within a documented drain window; lag is **measured**, not silent. |
| **CI / local target (fake embed)** | **p95 drain-to-searchable ≤ 5s** on coach_core (or equivalent integration) lanes that use a **fake/stub embed provider** and an in-process or test outbox drain — wall-clock against real embed APIs is too flaky for a hard gate. |
| **Production / real embed** | Documented SLO + **metric hook** (stage timings / lag histogram); assert “eventually searchable” in coach_core rather than a brittle wall-clock against live providers. |
| **At ack (always)** | Graph + outbox durable; FTS/needle usable per existing INIT-003 semantics. Vector rank may lag until drain completes. |

Exact metric names and response fields are owned by SPEC-002 / SPEC-006; this
ADR freezes the **target spirit** and the **5s p95 fake-embed CI bar**.

## Context

INIT-003 / ADR-003 made store ack durable under outbox and budgeted conflict
queries fail-open. INIT-004 / ADR-004 froze CE surfaces and **parked** write-path
perf as INIT-005. BRAIN-002’s measure-first split kept CE work from blocking on
embed-defer design.

Without this ADR, implementing specs would re-open:

- Whether ack may wait on Qdrant or sync embed “for correctness” (it must not
  for Qdrant; embed-defer is opt-in and still never sync-Qdrant-ack)
- Whether vector lag may stay silent (it must not — GR-LAG-001)
- Whether conflict/dedup/extract may drop the outbox under budget (never)
- Whether conflict embed can be reused when text is unchanged (yes)
- Default for `ARCHIVIST_EMBED_DEFER` (false until evals prove SLO)

ADR-003 non-goals and ADR-004 CE non-goals still apply. This ADR does not
reopen INIT-004 CE map/bootstrap contracts.

## Alternatives considered

| Option | Description | Why not chosen |
|---|---|---|
| **Bundle with INIT-004** — CE surfaces + embed-defer in one INIT | Fewer initiative IDs; one “coach path” ship | Rejected in BRAIN-002 / ADR-004: CE surface work must not wait on perf design; park INIT-005. |
| **Soak INIT-003 / INIT-004 only** — no write-path perf initiative | Avoids defer/lag complexity | Leaves store latency dominated by optional gates + embed on the ack path; personal-prod coach turns stay slow. Measure-then-cut latency is the BRAIN-002 intent for INIT-005. |
| **Default `ARCHIVIST_EMBED_DEFER=true`** — opt-out | Faster ack out of the box | Widens searchable lag before coach_core proves SLO; opt-in is safer. |
| **Sync Qdrant before ack** — “searchable at ack” | Strongest read-your-writes for vectors | Rejects ADR-003 durability model; dead Qdrant would stall coach clients again. |
| **Skip outbox under budget** — fastest possible ack | Extreme latency cut | Violates GR-DUR-001; durability regression vs INIT-003. |

## Consequences

**Positive:**

- SPEC-002…009 inherit frozen durability, lag honesty, hard-skip rules, embed
  reuse, and flag default — no re-deciding mid-implementation.
- Coach_core can assert ack-under-dead-Qdrant, hard-skip observability, and
  drain-to-searchable (fake embed) against a written contract.
- Operators get an explicit opt-in for embed-defer once evals pass.

**Negative / accepted trade-offs:**

- With defer on, hybrid/vector search may miss fresh writes until drain;
  mitigated by GR-LAG-001, FTS at ack, and cite-or-refuse / empty-OK (ADR-004).
- Hard-skip conflict/dedup can weaken quality under load; accepted with metrics
  (not an authz bypass).
- Default defer off means ack latency gains require explicit enablement after
  SPEC-007 confidence.

## Non-goals (INIT-005)

Explicitly **out of scope** for work governed by this ADR:

- Synchronous LLM generation on `archivist_index` / index hot path (GR-CE-002)
- Requiring **sync Qdrant** upsert before store ack
- Net-new MCP tools / tool sprawl (GR-PROD-002); default profile remains **core**
- Phase 7–10 product breadth (checkpoints, MaP MCP, `share_*`, fleet demos)
- Edits inside `myaifitness-android` or Domain MCP packet builders
- Re-opening INIT-004 CE map / bootstrap / empty-OK contracts
- Weakening outbox durability or raising mypy / CI gate ceilings for “perf”

## Implementing specs

| Spec | What it delivers |
|---|---|
| [SPEC-001](../../sdd/initiatives/INIT-005-coach-path-performance/specs/SPEC-001-docs-adr-coach-path-performance.md) | This ADR; ROADMAP Immediate Next → INIT-005 (current) |
| [SPEC-002](../../sdd/initiatives/INIT-005-coach-path-performance/specs/SPEC-002-infrastructure-store-stage-timings.md) | Store-side stage timings + lag instrumentation |
| [SPEC-003](../../sdd/initiatives/INIT-005-coach-path-performance/specs/SPEC-003-service-ack-budget-hard-skips.md) | Hard-skip optional write gates under ack budget |
| [SPEC-004](../../sdd/initiatives/INIT-005-coach-path-performance/specs/SPEC-004-service-embed-reuse.md) | Conflict embed reuse into primary store |
| [SPEC-005](../../sdd/initiatives/INIT-005-coach-path-performance/specs/SPEC-005-service-embed-deferred-store.md) | Embed-deferred store + drain fill + lag SLO |
| [SPEC-006](../../sdd/initiatives/INIT-005-coach-path-performance/specs/SPEC-006-api-store-lag-contract.md) | Store response deferred/lag contract + REFERENCE |
| [SPEC-007](../../sdd/initiatives/INIT-005-coach-path-performance/specs/SPEC-007-infrastructure-coach-core-perf-evals.md) | coach_core perf + lag + durability evals |
| [SPEC-008](../../sdd/initiatives/INIT-005-coach-path-performance/specs/SPEC-008-docs-security-review.md) / [SPEC-009](../../sdd/initiatives/INIT-005-coach-path-performance/specs/SPEC-009-docs-architecture-diagrams.md) | Security review and architecture/design diagrams (initiative close) |

See [`docs/ROADMAP.md`](../ROADMAP.md) for the product-facing pointer to this
ADR, and BRAIN-002 / ADR-004 for the measure-first split that parked this work
until after INIT-004.
