# CRSI + SIM Design Note

- **Reference:** Yu et al., *"Self-Improvement Memory Enables Continual Recursive Self-Improvement"* ([arXiv:2611.18856](https://arxiv.org/abs/2611.18856))
- **Package:** `src/archivist/crsi/` (decoupled offline subpackage carved out under the 2026-11-27 [`ADR-006 / GR-LAYER-001`](adr/ADR-006-agentic-memory-eval-gym.md#amendment--2026-11-27-gr-layer-001-carve-out-for-srcarchivistcrsi) amendment)

---

## 1. Paper-to-Code Component Map

| Yu et al. Concept | Paper Ref | Module(s) | Key Contract |
|---|---|---|---|
| Frozen policy model \(M\) | §3.1 | `crsi/llm/` | Model weights frozen; role-separated config (`policy`, `proposer`, `selector`) over `httpx` + deterministic `FakeScriptedAdapter`. Pre-call `SpendTracker` enforces `allow_paid_apis`, `max_spend_usd`, and `max_llm_calls`. |
| Seed harness \(H_0\) | §3.1, App. C.2 | `crsi/harness/` | Frozen `TaskInput -> HarnessResult` contract, sandboxed `WorkspaceManager`, `shell=False` `ToolRuntime` (sanitized env), and fail-closed `PermissionGate` around editable `tools.py`, `context.py`, `runner.py`, `prompts/`. |
| `ExecutionGatingInterlock` + additive quality hook | App. F.2.1–F.2.2 | `crsi/harness/interlock.py` | Cross-turn completion gate blocking Cases 0 (active error/signature), A (prior error), B (discarded turn), and C (zero executions), bounded by `max_rejections`, with additive `CompletionAdvisor` hooks (`QualityFeasibilityCheck`) running beside it. |
| Frozen evaluator & disjoint splits | §3.1, App. B, §5 | `crsi/evaluator/`, `crsi/guards/heldout_firewall.py` | Strictly disjoint `evolve` vs. `heldout` splits in a private dir outside git; harness receives `TaskInput` only; `HeldOutIsolationGuard` scans every Proposer/Selector prompt and feedback dir before every LLM call. |
| CRSI outer loop \((H_t, F_t, P, \mathcal{C}_t, S, \mathcal{D}_t)\) | §3.1 Eqs. 1 & 3 | `crsi/loop/` | Per-candidate `git worktree` (`git reset --hard && git clean -fdx`), required pre-eval `ChangeManifest`, deterministic `LegalEditFilter` (frozen paths, secrets, `commands.check`, interlock anti-erosion), Selector screen + pick + hard `regression_tolerance` floor, `D_t` retaining rejected candidates, atomic `run_state.json` resume, dedicated `crsi/<run_id>` branch only. |
| Self-Improvement Memory \(\mathcal{M}_t\) & carry-over | §3.2–3.3, App. D.1–D.3 | `crsi/sim/` | Atomic JSON (`.crsi/sim.json`) grouped by root-cause problem (`next_id`, `carried_from`, `aliases`, `problems[]`, `attempts[]` with rationale + predicted vs. observed + `(model, domain)`). Deterministic merge preserving all attempts; untruncated `full` render (default) plus `problem_only` / `none` ablations. |
| Transfer verification | §3.3, App. C.3 | `crsi/toy/`, `tests/integration/crsi/test_toy_scenarios.py` | Zero-cost deterministic CI scenarios proving interlock gating, Full SIM vs. no-SIM on a harder split, `problem_only` ablation gap, and cross-policy carry-over without a blank start. |

---

## 2. Boundary & Governance Invariants

1. **Editable vs. Frozen (`PathPolicyGuard`):** Proposer edits are restricted to `harness.editable`. Core `archivist.*`, CRSI control-plane/guards/evaluator/loop/SIM, `.git/`, `.crsi/`, tests, and scorer/check command targets are non-overridably frozen.
2. **Secret Hygiene (`SecretRedactor`):** Config accepts env-var names only (`^[A-Z][A-Z0-9_]*$`), rejecting inline keys; child task environments strip credential variables; diffs, logs, history `D_t`, and SIM writes are scrubbed and checked.
3. **Human Governance:** Destructive tool actions fail closed in `PermissionGate`; CRSI never commits to or auto-merges into `main`/`master`; paid providers fail fast unless `allow_paid_apis = true` and `max_spend_usd > 0.0`.

---

## 3. Known Limits

1. **Held-out stays strictly outside the loop:** Held-out tasks, answers, and scores are evaluated out-of-loop for human review only and never enter `F_t`, `D_t`, or SIM. Overfitting to the evolve set is only caught by periodic human inspection of held-out reports.
2. **A self-modifying harness can game the evolve scorer:** Even with frozen scorers and path boundaries, an optimizer can exploit evolve-set quirks (e.g., overfitting prompts or heuristics to evolve tasks rather than generalizable verification). Review the `crsi/<run_id>` diff and held-out report before merging.
3. **Cost is uncapped unless budgets are enforced:** Real LLM runs scale as `O(rounds * candidates * evolve_tasks * rollouts * max_steps)`. Operators must configure `max_spend_usd`, `max_llm_calls`, and wall/task timeouts, and inspect `dry-run` counts before opting into paid APIs.
4. **Re-validation is mandatory after a major model or distribution change:** Inherited harness guards and SIM rationales reflect prior policy failure modes; swapping to a significantly stronger or different model requires fresh baseline evaluation (`--carry-from` enforces re-baselining) so stale workarounds are reassessed with new evidence rather than blindly kept or deleted.
5. **Toy scenarios validate mechanism invariants, not paper benchmark gains:** The deterministic toy suite verifies interlock enforcement, SIM carry-over, ablation contrast, and firewall/worktree mechanics at `$0` cost. It does **not** reproduce Yu et al.'s Terminal-Bench, Harvey LAB, GDPval, or APEX-Agents results and makes no benchmark claims.
