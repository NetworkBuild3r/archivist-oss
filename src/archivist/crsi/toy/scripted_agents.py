"""FROZEN scripted policy, proposer, and selector agents for deterministic CI toy scenarios."""

from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

from archivist.crsi.config import ModelRoleConfig
from archivist.crsi.evaluator.runner import FrozenEvaluator
from archivist.crsi.evaluator.splits import TaskDataset
from archivist.crsi.evaluator.types import EvaluationReport
from archivist.crsi.harness.context import SessionContext
from archivist.crsi.harness.contract import HarnessProtocol
from archivist.crsi.harness.interlock import ExecutionGatingInterlock
from archivist.crsi.harness.runner import SeedHarness
from archivist.crsi.llm.base import LLMClient, LLMRequest
from archivist.crsi.llm.fake import ScriptHandler
from archivist.crsi.loop.engine import WorktreeEvaluatorFn
from archivist.crsi.toy.scorer import ToyScorer

_SEED_TEMPLATE_DIR = Path(__file__).resolve().parent / "seed_harness"

_MATH_EXPR_RE = re.compile(r"Compute\s+(.+?)\s+in Python", re.IGNORECASE)
_V2_NORM_RE = re.compile(r"compute\s+(\d+\s*\*\s*\d+)", re.IGNORECASE)
_FAILED_TASK_RE = re.compile(r"-\s+\[FAIL\]\s+([A-Za-z0-9_\-]+)")
_CAND_SCORE_RE = re.compile(r"### Candidate\s+([A-Za-z0-9_\-]+)\s+\(score=([0-9.]+)")
_BASELINE_SCORE_RE = re.compile(r"baseline_score=([0-9.]+)")


def _run_git(cwd: Path, *args: str) -> str:
    git_exe = shutil.which("git")
    if git_exe is None:
        raise RuntimeError("git executable not found on PATH")
    proc = subprocess.run(
        [git_exe, *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
        shell=False,
    )
    return proc.stdout.strip()


def render_toy_interlock_source(
    *,
    interlock_enabled: bool,
    quality_check_enabled: bool,
    v2_recovery_enabled: bool = False,
) -> str:
    """Render clean Python source for ``harness/interlock.py``."""
    return f'''"""Editable toy interlock configuration wired into the seed harness."""

from __future__ import annotations

from archivist.crsi.harness.interlock import (
    CompletionAdvisor,
    ExecutionGatingInterlock,
    QualityFeasibilityCheck,
)

INTERLOCK_ENABLED = {interlock_enabled!r}
QUALITY_CHECK_ENABLED = {quality_check_enabled!r}
V2_RECOVERY_GUIDANCE_ENABLED = {v2_recovery_enabled!r}


def build_interlock() -> ExecutionGatingInterlock:
    """Build the ExecutionGatingInterlock plus any additive CompletionAdvisors."""
    advisors: list[CompletionAdvisor] = []
    if QUALITY_CHECK_ENABLED:
        advisors.append(QualityFeasibilityCheck(block_once=True))
    return ExecutionGatingInterlock(
        enabled=INTERLOCK_ENABLED,
        min_required_executions=1,
        max_rejections=3,
        advisors=advisors,
    )
'''


def seed_toy_repo(
    repo_dir: Path,
    *,
    interlock_enabled: bool = False,
    quality_check_enabled: bool = False,
    v2_recovery_enabled: bool = False,
) -> Path:
    """Initialize a self-contained git repository with editable toy harness files."""
    repo_dir.mkdir(parents=True, exist_ok=True)
    _run_git(repo_dir, "init", "-b", "main")
    _run_git(repo_dir, "config", "user.name", "CRSI Toy")
    _run_git(repo_dir, "config", "user.email", "toy@local.invalid")

    harness_dir = repo_dir / "harness"
    prompts_dir = harness_dir / "prompts"
    prompts_dir.mkdir(parents=True, exist_ok=True)

    (harness_dir / "harness.py").write_text(
        (_SEED_TEMPLATE_DIR / "harness.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (harness_dir / "interlock.py").write_text(
        render_toy_interlock_source(
            interlock_enabled=interlock_enabled,
            quality_check_enabled=quality_check_enabled,
            v2_recovery_enabled=v2_recovery_enabled,
        ),
        encoding="utf-8",
    )
    (prompts_dir / "system.md").write_text(
        (_SEED_TEMPLATE_DIR / "prompts" / "system.md").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo_dir / "frozen_scorer_stub.py").write_text(
        "# Frozen marker file (never editable by CRSI proposals).\n",
        encoding="utf-8",
    )

    _run_git(repo_dir, "add", "-A")
    _run_git(repo_dir, "commit", "-m", "chore: seed toy CRSI harness")
    return repo_dir


def load_toy_harness_from_worktree(
    wt_path: Path,
    llm_client: LLMClient,
) -> HarnessProtocol:
    """Dynamically load ``harness/interlock.py`` and ``prompts/system.md`` from ``wt_path``."""
    interlock_path = wt_path / "harness" / "interlock.py"
    prompt_path = wt_path / "harness" / "prompts" / "system.md"

    mod_name = f"_crsi_toy_interlock_{uuid.uuid4().hex}"
    spec = importlib.util.spec_from_file_location(mod_name, interlock_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load toy interlock module from {interlock_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    build_fn = getattr(module, "build_interlock", None)
    if not callable(build_fn):
        raise RuntimeError(f"Toy interlock module {interlock_path} missing build_interlock()")

    def _interlock_factory() -> ExecutionGatingInterlock:
        inst = build_fn()
        if not isinstance(inst, ExecutionGatingInterlock):
            raise TypeError("build_interlock() must return ExecutionGatingInterlock")
        return inst

    system_prompt = (
        prompt_path.read_text(encoding="utf-8").strip()
        if prompt_path.is_file()
        else "Toy coding agent."
    )
    if getattr(module, "V2_RECOVERY_GUIDANCE_ENABLED", False):
        system_prompt = f"{system_prompt}\nMANDATORY_POLICY_V2_RULE: NORMALIZE_OUTPUT"

    return SeedHarness(
        llm_client=llm_client,
        interlock_factory=_interlock_factory,
        context=SessionContext(system_prompt=system_prompt),
    )


def make_toy_worktree_evaluator(
    llm_client: LLMClient,
    *,
    parallelism: int = 2,
    task_timeout_s: float = 30.0,
) -> WorktreeEvaluatorFn:
    """Build a ``WorktreeEvaluatorFn`` running ``FrozenEvaluator`` + ``ToyScorer``."""
    scorer = ToyScorer()

    async def _evaluate(wt_path: Path, dataset: TaskDataset, rollouts: int) -> EvaluationReport:
        evaluator = FrozenEvaluator(
            harness_factory=lambda: load_toy_harness_from_worktree(wt_path, llm_client),
            scorer=scorer,
            parallelism=parallelism,
            task_timeout_s=task_timeout_s,
        )
        return await evaluator.evaluate_evolve(dataset, rollouts=rollouts, checkout_dir=wt_path)

    return _evaluate


def _extract_last_observation(request: LLMRequest) -> str:
    for msg in reversed(request.messages):
        if msg.role == "user" and msg.content.startswith("Observation (step"):
            return msg.content
    return ""


def make_scripted_policy_v1() -> ScriptHandler:
    """Deterministic policy ``fake-policy-v1`` with premature-completion and suboptimal habits."""

    def _handle(request: LLMRequest, _cfg: ModelRoleConfig) -> str:
        step = int(request.metadata.get("step", 1))
        user_task_msg = request.messages[1].content if len(request.messages) > 1 else ""
        reminder = request.messages[-1].content
        last_obs = _extract_last_observation(request)

        # Habit 1: On step 1, always attempt premature/hallucinated completion without execution!
        if step == 1:
            return json.dumps({"action": "finish", "answer": "HALLUCINATED_GUESS"})

        is_opt = "optimization task" in user_task_msg.lower()

        if is_opt:
            # If the additive QualityFeasibilityCheck advisor rejected the 0.55 draft,
            # run an optimization pass achieving score=0.96.
            if "Quality/feasibility check" in reminder:
                return json.dumps(
                    {
                        "action": "run_command",
                        "argv": [
                            sys.executable,
                            "-c",
                            "print('CONSTRAINTS PASSED score=0.96')",
                        ],
                    }
                )
            if "score=0.96" in last_obs:
                return json.dumps({"action": "finish", "answer": "CONSTRAINTS PASSED score=0.96"})
            if "score=0.55" in last_obs:
                return json.dumps({"action": "finish", "answer": "CONSTRAINTS PASSED score=0.55"})
            # First real execution produces a feasible-but-suboptimal plan (0.55).
            return json.dumps(
                {
                    "action": "run_command",
                    "argv": [
                        sys.executable,
                        "-c",
                        "print('CONSTRAINTS PASSED score=0.55')",
                    ],
                }
            )

        # Verification task: execute Python expression and return verified stdout.
        expr_match = _MATH_EXPR_RE.search(user_task_msg)
        expr = expr_match.group(1).strip() if expr_match else "0"
        if "VERIFIED_RESULT=" in last_obs:
            ans = last_obs.split("VERIFIED_RESULT=", 1)[1].strip().splitlines()[0].strip()
            return json.dumps({"action": "finish", "answer": ans})
        return json.dumps(
            {
                "action": "run_command",
                "argv": [
                    sys.executable,
                    "-c",
                    f"print('VERIFIED_RESULT=' + str({expr}))",
                ],
            }
        )

    return _handle


def make_scripted_policy_v2() -> ScriptHandler:
    """Second scripted policy ``fake-policy-v2`` with premature completion + v2 normalization habit."""
    base_v1 = make_scripted_policy_v1()

    def _handle(request: LLMRequest, cfg: ModelRoleConfig) -> str:
        step = int(request.metadata.get("step", 1))
        system_msg = request.messages[0].content if request.messages else ""
        user_task_msg = request.messages[1].content if len(request.messages) > 1 else ""
        last_obs = _extract_last_observation(request)

        if "v2 normalization task" not in user_task_msg.lower():
            return base_v1(request, cfg)

        # Habit 1 (shared): still tries premature finish on step 1 if interlock is absent!
        if step == 1:
            return json.dumps({"action": "finish", "answer": "V2_HALLUCINATED"})

        # Habit 2 (distinct to v2): only emits VERIFIED_NORMALIZED:<val> when the harness
        # supplies explicit normalization guidance in system prompt / interlock config.
        has_norm_guidance = "NORMALIZE_OUTPUT" in system_msg
        match = _V2_NORM_RE.search(user_task_msg)
        expr = match.group(1) if match else "0"

        if "VERIFIED_NORMALIZED:" in last_obs:
            line = [ln for ln in last_obs.splitlines() if "VERIFIED_NORMALIZED:" in ln][-1]
            return json.dumps({"action": "finish", "answer": line.strip()})
        if "RAW_DRAFT:" in last_obs:
            val = last_obs.split("RAW_DRAFT:", 1)[1].strip().splitlines()[0].strip()
            return json.dumps({"action": "finish", "answer": f"draft:{val}"})

        if has_norm_guidance:
            return json.dumps(
                {
                    "action": "run_command",
                    "argv": [
                        sys.executable,
                        "-c",
                        f"print('VERIFIED_NORMALIZED:' + str({expr}))",
                    ],
                }
            )
        return json.dumps(
            {
                "action": "run_command",
                "argv": [
                    sys.executable,
                    "-c",
                    f"print('RAW_DRAFT:' + str({expr}))",
                ],
            }
        )

    return _handle


def make_scripted_proposer() -> ScriptHandler:
    """Deterministic SIM-aware Proposer reflecting Yu et al. transfer dynamics."""

    def _handle(request: LLMRequest, _cfg: ModelRoleConfig) -> str:
        prompt = request.messages[-1].content
        round_num = int(request.metadata.get("round", 0))
        failed_ids = _FAILED_TASK_RE.findall(prompt)
        if not failed_ids:
            return json.dumps({"candidates": []})

        has_interlock_enabled = "INTERLOCK_ENABLED = True" in prompt
        has_quality_enabled = "QUALITY_CHECK_ENABLED = True" in prompt
        has_v2_enabled = "V2_RECOVERY_GUIDANCE_ENABLED = True" in prompt

        # Full SIM provides rationale + predicted/observed proof for the interlock.
        has_full_sim_interlock_proof = (
            "Memory mode: full" in prompt
            and "rationale:" in prompt
            and "observed:" in prompt
            and "ExecutionGatingInterlock" in prompt
        )

        # Case 1: Base split — premature completion failures while interlock is disabled.
        if not has_interlock_enabled and "premature_completion" in prompt:
            return json.dumps(
                {
                    "candidates": [
                        {
                            "candidate_id": f"r{round_num}-enable-interlock",
                            "manifest": {
                                "problem_id": None,
                                "new_problem_name": "premature completion without execution verification",
                                "what": "Enable ExecutionGatingInterlock in harness/interlock.py",
                                "files": ["harness/interlock.py"],
                                "rationale": (
                                    "ExecutionGatingInterlock blocks premature completion when "
                                    "zero verified commands ran, forcing real execution."
                                ),
                                "predicted": failed_ids,
                                "interlock_removal_sim_evidence": None,
                            },
                            "file_edits": {
                                "harness/interlock.py": render_toy_interlock_source(
                                    interlock_enabled=True,
                                    quality_check_enabled=has_quality_enabled,
                                    v2_recovery_enabled=has_v2_enabled,
                                )
                            },
                        }
                    ]
                }
            )

        # Case 2: Harder split — feasible_suboptimal failures while interlock is already enabled.
        if has_interlock_enabled and not has_quality_enabled and "feasible_suboptimal" in prompt:
            if has_full_sim_interlock_proof:
                # Full SIM sees P1's rationale + observed proof: KEEP interlock, ADD quality check!
                return json.dumps(
                    {
                        "candidates": [
                            {
                                "candidate_id": f"r{round_num}-add-quality-check",
                                "manifest": {
                                    "problem_id": None,
                                    "new_problem_name": "feasible but suboptimal optimization solutions",
                                    "what": (
                                        "Retain ExecutionGatingInterlock and add "
                                        "QualityFeasibilityCheck advisor beside it"
                                    ),
                                    "files": ["harness/interlock.py"],
                                    "rationale": (
                                        "Full SIM shows P1 ExecutionGatingInterlock prevents "
                                        "premature completion; keep it intact and layer "
                                        "QualityFeasibilityCheck beside it for optimization tasks."
                                    ),
                                    "predicted": failed_ids,
                                    "interlock_removal_sim_evidence": None,
                                },
                                "file_edits": {
                                    "harness/interlock.py": render_toy_interlock_source(
                                        interlock_enabled=True,
                                        quality_check_enabled=True,
                                        v2_recovery_enabled=has_v2_enabled,
                                    )
                                },
                            }
                        ]
                    }
                )
            # Without full SIM rationale/evidence (none OR problem_only ablation),
            # proposer misdiagnoses the interlock as overly restrictive and tries to disable it.
            return json.dumps(
                {
                    "candidates": [
                        {
                            "candidate_id": f"r{round_num}-weaken-interlock",
                            "manifest": {
                                "problem_id": None,
                                "new_problem_name": "relax completion interlock on optimization split",
                                "what": "Disable ExecutionGatingInterlock to let policy finish freely",
                                "files": ["harness/interlock.py"],
                                "rationale": (
                                    "No full SIM rationale available; assuming completion gating "
                                    "interferes with optimization tasks."
                                ),
                                "predicted": failed_ids,
                                "interlock_removal_sim_evidence": None,
                            },
                            "file_edits": {
                                "harness/interlock.py": render_toy_interlock_source(
                                    interlock_enabled=False,
                                    quality_check_enabled=False,
                                    v2_recovery_enabled=has_v2_enabled,
                                )
                            },
                        }
                    ]
                }
            )

        # Case 3: Cross-policy transfer (fake-policy-v2) — v2_unnormalized failures.
        if has_interlock_enabled and not has_v2_enabled and "v2_unnormalized" in prompt:
            return json.dumps(
                {
                    "candidates": [
                        {
                            "candidate_id": f"r{round_num}-add-v2-normalization",
                            "manifest": {
                                "problem_id": None,
                                "new_problem_name": "policy-v2 unnormalized output without verification rule",
                                "what": (
                                    "Preserve ExecutionGatingInterlock and QualityFeasibilityCheck; "
                                    "add V2_RECOVERY_GUIDANCE_ENABLED normalization directive"
                                ),
                                "files": [
                                    "harness/interlock.py",
                                    "harness/prompts/system.md",
                                ],
                                "rationale": (
                                    "Carried SIM rationales prove ExecutionGatingInterlock and "
                                    "QualityFeasibilityCheck remain valuable; add explicit "
                                    "NORMALIZE_OUTPUT guidance for policy-v2."
                                ),
                                "predicted": failed_ids,
                                "interlock_removal_sim_evidence": None,
                            },
                            "file_edits": {
                                "harness/interlock.py": render_toy_interlock_source(
                                    interlock_enabled=True,
                                    quality_check_enabled=has_quality_enabled,
                                    v2_recovery_enabled=True,
                                ),
                                "harness/prompts/system.md": (
                                    "You are a toy coding agent inside a sandboxed workspace.\n"
                                    "Use run_command to verify every computation before calling finish.\n"
                                    "NORMALIZE_OUTPUT: always emit VERIFIED_NORMALIZED:<value> when requested.\n"
                                ),
                            },
                        }
                    ]
                }
            )

        return json.dumps({"candidates": []})

    return _handle


def make_scripted_selector() -> ScriptHandler:
    """Deterministic Selector keeping legal candidates on screen and picking best improvement."""

    def _handle(request: LLMRequest, _cfg: ModelRoleConfig) -> str:
        stage = request.metadata.get("stage")
        prompt = request.messages[-1].content
        if stage == "screen":
            cids = re.findall(r"### Candidate\s+([A-Za-z0-9_\-]+)", prompt)
            return json.dumps({"keep": cids, "drop": {}})

        base_match = _BASELINE_SCORE_RE.search(prompt)
        baseline = float(base_match.group(1)) if base_match else 0.0
        scored = [(cid, float(score_s)) for cid, score_s in _CAND_SCORE_RE.findall(prompt)]
        if not scored:
            return json.dumps({"selected": "NONE", "reason": "no candidates"})

        best_id, best_score = max(scored, key=lambda pair: pair[1])
        if best_score > baseline + 1e-9:
            return json.dumps(
                {
                    "selected": best_id,
                    "reason": f"Selected {best_id} improving evolve score {baseline:.3f} -> {best_score:.3f}",
                }
            )
        return json.dumps(
            {
                "selected": "NONE",
                "reason": f"No candidate beat baseline {baseline:.3f} (best={best_score:.3f})",
            }
        )

    return _handle
