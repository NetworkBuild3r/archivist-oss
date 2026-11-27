# CRSI + SIM Operator Guide

`archivist.crsi` is a decoupled subpackage implementing a **Continual Recursive Self-Improvement (CRSI)** agent-harness loop with **Self-Improvement Memory (SIM)**, inspired by Yu et al. (*"Self-Improvement Memory Enables Continual Recursive Self-Improvement"*, [arXiv:2611.18856](https://arxiv.org/abs/2611.18856)).

- **Frozen policy weights:** Only files matching `harness.editable` may change; scorers, guards, SIM, loop code, and core `archivist.*` packages are frozen.
- **Zero paid calls by default:** `allow_paid_apis = false` and `max_spend_usd = 0.0` out of the box.
- **No auto-merge to `main`:** Every run commits accepted harness edits + `.crsi/sim.json` onto a dedicated `crsi/<run_id>` branch for human review.
- **Zero coupling to Archivist MCP:** `src/archivist/crsi/` has zero imports from core `archivist.{app,core,storage,retrieval,lifecycle,write,features,utils}` or `qdrant_client` (honors `ADR-006 / GR-LAYER-001`).

---

## 1. Run the Deterministic Toy End-to-End ($0 Spend)

Run from the repository root with `PYTHONPATH=src` (or in your Python environment where `src/` is on the path):

```bash
# 1) Initialize a scratch toy git repo + private dataset outside the repo
PYTHONPATH=src python3 -m archivist.crsi init \
  --repo-root /tmp/crsi-toy \
  --private-dir /tmp/crsi-toy-private \
  --toy --toy-split base

# 2) Validate config, secrets policy, editable/frozen path boundaries, git repo, and disjoint splits
PYTHONPATH=src python3 -m archivist.crsi validate --repo-root /tmp/crsi-toy

# 3) Inspect exact task-run count and upper-bound budget (executes 0 tasks, 0 LLM calls)
PYTHONPATH=src python3 -m archivist.crsi dry-run \
  --repo-root /tmp/crsi-toy --rounds 2 --candidates 3 --include-heldout

# 4) Execute (or resume if interrupted) on dedicated branch crsi/toy-01
PYTHONPATH=src python3 -m archivist.crsi run \
  --repo-root /tmp/crsi-toy \
  --run-id toy-01 --rounds 1 --candidates 2 --parallel 2 --new

# 5) Check run progress, scores, branch, and budget usage
PYTHONPATH=src python3 -m archivist.crsi status --repo-root /tmp/crsi-toy --run-id toy-01

# 6) Inspect Self-Improvement Memory (SIM) — human-readable or raw JSON
PYTHONPATH=src python3 -m archivist.crsi show-sim --repo-root /tmp/crsi-toy --run-id toy-01
PYTHONPATH=src python3 -m archivist.crsi show-sim --repo-root /tmp/crsi-toy --run-id toy-01 --json

# Optional: deterministically merge two SIM root-cause entries without losing attempt history
PYTHONPATH=src python3 -m archivist.crsi sim merge P2 P1 --sim /tmp/crsi-toy/.crsi/runs/toy-01/sim.json
```

Inspect the reviewable diff on the run branch (never auto-merged onto `main`):

```bash
git -C /tmp/crsi-toy diff main..crsi/toy-01
```

---

## 2. Point at a Real Model (Env Vars / Vault + Explicit Budget)

1. **Never put literal API keys in `.crsi/config.toml`, git, prompts, or SIM.** Reference only environment variable names (`api_key_env`), injected from your shell or Vault/Kubernetes secret mounts:

```toml
# .crsi/config.toml
[budget]
rounds = 3
candidates = 2
parallelism = 2
task_timeout_s = 60.0
total_wall_timeout_s = 1800.0
max_llm_calls = 150
max_spend_usd = 5.0        # Hard USD cap (must be > 0.0 when allow_paid_apis = true)
allow_paid_apis = true     # Explicit operator opt-in required for remote paid APIs
regression_tolerance = 0.0 # Reject candidates scoring below baseline - tolerance

[models.policy]
provider = "openai_compat"         # "openai_compat" | "anthropic" | "gemini" | "fake"
model = "gpt-4.1-mini"
base_url = "https://api.openai.com/v1"
api_key_env = "OPENAI_API_KEY"     # Env-var NAME only
input_usd_per_1m = 0.40
output_usd_per_1m = 1.60

[models.proposer]
provider = "anthropic"
model = "claude-sonnet-4-20250514"
api_key_env = "ANTHROPIC_API_KEY"
input_usd_per_1m = 3.00
output_usd_per_1m = 15.00

[models.selector]
provider = "gemini"
model = "gemini-2.5-pro"
api_key_env = "GEMINI_API_KEY"
input_usd_per_1m = 1.25
output_usd_per_1m = 5.00
```

- **xAI:** use `provider = "openai_compat"`, `base_url = "https://api.x.ai/v1"`, `api_key_env = "XAI_API_KEY"`.
- **Local Ollama / vLLM (loopback, $0):** use `provider = "openai_compat"`, `base_url = "http://127.0.0.1:11434/v1"` with no `api_key_env` (runs without `--allow-paid`).
- **External frozen scorer:** set `commands.run_task = ["python3", "/frozen/path/scorer.py", "{task_file}", "{task_id}", "{checkout}", "{out}", "{rollout}"]` and `commands.check = ["ruff", "check", "harness/"]`. Both must stay outside `harness.editable` (`validate` enforces this).

2. **Always dry-run before enabling paid spend**, then opt in explicitly:

```bash
export OPENAI_API_KEY="..."      # or inject via Vault
export ANTHROPIC_API_KEY="..."
export GEMINI_API_KEY="..."

PYTHONPATH=src python3 -m archivist.crsi validate --repo-root /path/to/repo
PYTHONPATH=src python3 -m archivist.crsi dry-run --repo-root /path/to/repo

PYTHONPATH=src python3 -m archivist.crsi run \
  --repo-root /path/to/repo \
  --run-id real-01 \
  --allow-paid --max-spend 5.00 --new
```

If interrupted, re-run the same command **without** `--new` to resume from `.crsi/runs/<run_id>/run_state.json` without duplicating completed evaluations or resetting spend counters.

---

## 3. Continue After a Model Swap or Task-Set Change

Transferring only evolved harness code to a new policy model or harder task distribution often stalls or regresses because the optimizer cannot see *why* inherited guards exist. Carry **Full SIM** alongside the evolved harness using `--carry-from`:

```bash
PYTHONPATH=src python3 -m archivist.crsi run \
  --repo-root /path/to/evolved-repo \
  --private-dir /path/to/new-private-tasks \
  --run-id hard-transfer-01 \
  --carry-from /path/to/prior-run/sim.json \
  --carry-domain toy_base \
  --carry-model prior-policy-v1 \
  --memory-mode full \
  --new
```

What `--carry-from` guarantees:
1. Preserves 100% of inherited problems, attempts, rationales, predicted-vs-observed outcomes, aliases, and original per-attempt `(model, domain)` provenance.
2. Stamps `carried_from` (`arm`, `domain`, `model`, `into`, `carried_at`) on the new SIM.
3. Forces **mandatory baseline re-validation** on the new policy model and task split before any edits are proposed.

> **Ablation warning:** `--memory-mode problem-only` (root-cause labels without rationales or predicted/observed history) and `--memory-mode none` are provided strictly for controlled ablations. Yu et al. showed problem-only memory can underperform no memory; keep `--memory-mode full` for normal runs.

---

## 4. Running the Test Suite

```bash
# Deterministic CRSI unit + integration suite (including all 4 toy scenarios, $0 spend)
python3 -m pytest tests/unit/crsi/ tests/integration/crsi/ -m crsi -v
```

For architectural mapping to Yu et al. (`arXiv:2611.18856`) and known limits, see [`docs/CRSI_DESIGN_NOTE.md`](CRSI_DESIGN_NOTE.md).
