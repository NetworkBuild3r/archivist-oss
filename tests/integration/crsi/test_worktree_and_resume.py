"""Integration tests for git worktree isolation, reset/clean, and resumable CRSI loop (SPEC-006)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from archivist.crsi.config import BudgetConfig, CrsiConfig, HarnessPathsConfig
from archivist.crsi.evaluator.splits import SplitManifest, TaskDataset
from archivist.crsi.evaluator.types import EvaluationReport, TaskScore, TaskSpec
from archivist.crsi.llm.base import LLMRequest, RoutedLLMClient
from archivist.crsi.llm.fake import FakeScriptedAdapter
from archivist.crsi.loop.engine import CrsiEngine
from archivist.crsi.loop.history import RunHistory
from archivist.crsi.loop.worktree import GitWorktreeError, GitWorktreeManager
from archivist.crsi.sim.store import SimStore

pytestmark = [pytest.mark.crsi, pytest.mark.integration]


def _git(cwd: Path, *args: str) -> str:
    git_exe = shutil.which("git")
    assert git_exe is not None
    proc = subprocess.run(
        [git_exe, *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        check=True,
        shell=False,
    )
    return proc.stdout.strip()


def _init_temp_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@local.invalid")

    (repo / "harness").mkdir()
    (repo / "harness" / "interlock.py").write_text(
        "VERSION = 0\nINTERLOCK_ENABLED = True\nclass ExecutionGatingInterlock: pass\n",
        encoding="utf-8",
    )
    (repo / "frozen_scorer.py").write_text("# frozen\n", encoding="utf-8")
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "initial seed")
    return repo


def test_worktree_reset_clean_strips_untracked_files_and_blocks_main_branch(tmp_path: Path):
    repo = _init_temp_repo(tmp_path)
    wt_mgr = GitWorktreeManager(
        repo,
        worktree_root=tmp_path / "worktrees",
        base_branch="main",
        branch_prefix="crsi/",
    )

    with pytest.raises(GitWorktreeError):
        wt_mgr.ensure_run_branch("main")
    with pytest.raises(GitWorktreeError):
        wt_mgr.advance_run_branch("main", wt_mgr.rev_parse("HEAD"))

    base_sha = wt_mgr.rev_parse("HEAD")
    wt = wt_mgr.checkout_detached_worktree("dirty_test", base_sha)
    try:
        (wt / "stray_untracked.txt").write_text("junk", encoding="utf-8")
        (wt / "harness" / "interlock.py").write_text("CORRUPTED\n", encoding="utf-8")
        wt_mgr.reset_and_clean(wt, base_sha)
        assert not (wt / "stray_untracked.txt").exists()
        assert "VERSION = 0" in (wt / "harness" / "interlock.py").read_text(encoding="utf-8")
    finally:
        wt_mgr.cleanup_all()


async def test_crsi_engine_isolates_branch_records_rejected_and_resumes_without_duplicate_evals(
    tmp_path: Path,
):
    repo = _init_temp_repo(tmp_path)
    main_sha_before = _git(repo, "rev-parse", "main")

    dataset = TaskDataset(
        tasks={
            "evo-1": TaskSpec(id="evo-1", prompt="Solve evo-1", expected="1"),
            "held-1": TaskSpec(id="held-1", prompt="Secret heldout", expected="SECRET_VAL"),
        },
        split=SplitManifest(evolve=["evo-1"], heldout=["held-1"]),
    )

    eval_calls: list[str] = []

    async def evaluator_fn(wt_path: Path, ds: TaskDataset, rollouts: int) -> EvaluationReport:
        _ = (ds, rollouts)
        text = (wt_path / "harness" / "interlock.py").read_text(encoding="utf-8")
        # Confirm reset_and_clean ensured no stray untracked file exists during scoring.
        assert not (wt_path / "stray.tmp").exists()
        sha = _git(wt_path, "rev-parse", "HEAD")
        eval_calls.append(sha)
        score = 1.0 if "VERSION = 1" in text else 0.25
        return EvaluationReport(
            split="evolve",
            mean_score=score,
            passed=1 if score >= 1.0 else 0,
            total=1,
            task_scores=[
                TaskScore(task_id="evo-1", score=score, passed=score >= 1.0, summary=f"s={score}")
            ],
            by_task_passed={"evo-1": score >= 1.0},
            by_task_score={"evo-1": score},
        )

    def proposer_handler(req: LLMRequest, _cfg: object) -> str:
        _ = req
        return json.dumps(
            {
                "candidates": [
                    # Illegal edit touching frozen_scorer.py -> dropped by LegalEditFilter into D_t
                    {
                        "candidate_id": "c_illegal",
                        "manifest": {
                            "new_problem_name": "try illegal frozen edit",
                            "what": "Touch frozen scorer",
                            "files": ["frozen_scorer.py"],
                            "rationale": "Should be dropped",
                            "predicted": ["evo-1"],
                        },
                        "file_edits": {"frozen_scorer.py": "# hacked\n"},
                    },
                    # Legal improving edit
                    {
                        "candidate_id": "c_good",
                        "manifest": {
                            "new_problem_name": "improve interlock version",
                            "what": "Upgrade harness interlock to VERSION = 1",
                            "files": ["harness/interlock.py"],
                            "rationale": "Fixes evo-1 cleanly while preserving interlock",
                            "predicted": ["evo-1"],
                        },
                        "file_edits": {
                            "harness/interlock.py": (
                                "VERSION = 1\nINTERLOCK_ENABLED = True\n"
                                "class ExecutionGatingInterlock: pass\n"
                            )
                        },
                    },
                ]
            }
        )

    def selector_handler(req: LLMRequest, _cfg: object) -> str:
        stage = req.metadata.get("stage")
        if stage == "screen":
            return json.dumps({"keep": ["c_good"], "drop": {}})
        return json.dumps({"selected": "c_good", "reason": "Score improved 0.25 -> 1.0"})

    fake = FakeScriptedAdapter(
        handlers={"proposer": proposer_handler, "selector": selector_handler}
    )
    cfg = CrsiConfig(
        project_name="resume-test",
        domain="toy_base",
        budget=BudgetConfig(rounds=1, candidates=2, parallelism=1, regression_tolerance=0.0),
        harness=HarnessPathsConfig(editable=["harness/**"], frozen=["frozen_scorer.py"]),
    )
    client = RoutedLLMClient(cfg, adapters={"fake": fake})

    engine1 = CrsiEngine(
        repo_root=repo,
        config=cfg,
        dataset=dataset,
        llm_client=client,
        evaluator_fn=evaluator_fn,
    )

    # Simulate interruption right after candidate evaluation (before select & record).
    partial = await engine1.run(
        run_id="run-resume-01",
        resume=False,
        stop_after_phase=(0, "evaluate"),
    )
    assert partial.status == "running"
    assert len(eval_calls) == 2  # baseline + c_good

    # Resume on a fresh engine instance; completed evaluations must NOT be re-run.
    client2 = RoutedLLMClient(cfg, adapters={"fake": fake})
    engine2 = CrsiEngine(
        repo_root=repo,
        config=cfg,
        dataset=dataset,
        llm_client=client2,
        evaluator_fn=evaluator_fn,
    )
    final = await engine2.run(run_id="run-resume-01", resume=True)

    assert final.status == "completed"
    assert final.rounds_completed == 1
    assert final.baseline_score == pytest.approx(0.25)
    assert final.final_score == pytest.approx(1.0)
    assert final.accepted_count == 1
    assert final.rejected_count == 1
    assert len(eval_calls) == 2  # ZERO duplicate rollouts on resume!

    # Main branch was NEVER mutated; changes live only on crsi/run-resume-01.
    assert _git(repo, "rev-parse", "main") == main_sha_before
    assert _git(repo, "rev-parse", "crsi/run-resume-01") == final.final_sha
    assert final.final_sha != main_sha_before

    # SIM was updated and committed on crsi/run-resume-01.
    sim_loaded = SimStore.load(Path(final.sim_path))
    assert len(sim_loaded.doc.problems) == 1
    assert sim_loaded.doc.problems[0].attempts[0].observed.fixed == ["evo-1"]

    # Rejected illegal candidate is preserved in D_t history.jsonl.
    hist = RunHistory.load_or_create(Path(final.history_path))
    assert any(
        r.candidate_id == "c_illegal" and r.drop_reason == "illegal_path" for r in hist.records
    )
    assert any(r.candidate_id == "c_good" and r.status == "accepted" for r in hist.records)
