"""Deterministic toy integration tests proving all 4 required CRSI + SIM scenarios (SPEC-007).

All 4 scenarios run end-to-end through the real CRSI loop, real git worktrees,
SeedHarness, ExecutionGatingInterlock, QualityFeasibilityCheck, FrozenEvaluator,
HeldOutIsolationGuard, and SimStore with $0 spend and zero paid API calls.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pytest

from archivist.crsi.config import (
    BudgetConfig,
    CrsiConfig,
    HarnessPathsConfig,
    MemoryMode,
    ModelRoleConfig,
    ModelsConfig,
)
from archivist.crsi.evaluator.runner import FrozenEvaluator
from archivist.crsi.evaluator.splits import TaskDataset
from archivist.crsi.llm.base import RoutedLLMClient
from archivist.crsi.llm.fake import FakeScriptedAdapter, ScriptHandler
from archivist.crsi.loop.engine import CrsiEngine, CrsiRunResult
from archivist.crsi.loop.history import RunHistory
from archivist.crsi.loop.worktree import GitWorktreeManager
from archivist.crsi.sim.store import SimStore
from archivist.crsi.toy.scorer import ToyScorer
from archivist.crsi.toy.scripted_agents import (
    load_toy_harness_from_worktree,
    make_scripted_policy_v1,
    make_scripted_policy_v2,
    make_scripted_proposer,
    make_scripted_selector,
    make_toy_worktree_evaluator,
    seed_toy_repo,
)
from archivist.crsi.toy.tasks import materialize_private_dataset

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


def _build_client(
    cfg: CrsiConfig,
    *,
    policy_handler: ScriptHandler,
) -> RoutedLLMClient:
    fake = FakeScriptedAdapter(
        handlers={
            "policy": policy_handler,
            "proposer": make_scripted_proposer(),
            "selector": make_scripted_selector(),
        }
    )
    return RoutedLLMClient(cfg, adapters={"fake": fake})


async def _evaluate_heldout_commit(
    repo: Path,
    commit_sha: str,
    dataset: TaskDataset,
    client: RoutedLLMClient,
    report_dir: Path,
    run_id: str,
) -> float:
    wt_mgr = GitWorktreeManager(repo, worktree_root=report_dir / "_heldout_wt")
    wt = wt_mgr.checkout_detached_worktree(f"heldout_{run_id}", commit_sha)
    try:
        wt_mgr.reset_and_clean(wt, commit_sha)
        evaluator = FrozenEvaluator(
            harness_factory=lambda: load_toy_harness_from_worktree(wt, client),
            scorer=ToyScorer(),
            parallelism=2,
        )
        rep, report_path = await evaluator.evaluate_heldout(
            dataset,
            report_dir=report_dir,
            run_id=run_id,
            checkout_dir=wt,
        )
        assert report_path.is_file()
        return rep.mean_score
    finally:
        wt_mgr.cleanup_all()


async def _run_source_base_evolution(tmp_path: Path) -> tuple[Path, CrsiRunResult, SimStore]:
    """Helper producing the source-evolved harness + Full SIM on toy_base."""
    repo = seed_toy_repo(tmp_path / "source_repo", interlock_enabled=False)
    private_dir = tmp_path / "private_base"
    dataset = materialize_private_dataset(private_dir, "base")

    cfg = CrsiConfig(
        project_name="toy-base",
        domain="toy_base",
        memory_mode="full",
        models=ModelsConfig(
            policy=ModelRoleConfig(provider="fake", model="fake-policy-v1"),
            proposer=ModelRoleConfig(provider="fake", model="fake-proposer-v1"),
            selector=ModelRoleConfig(provider="fake", model="fake-selector-v1"),
        ),
        budget=BudgetConfig(rounds=1, candidates=2, parallelism=2, allow_paid_apis=False),
        harness=HarnessPathsConfig(editable=["harness/**"], frozen=["frozen_scorer_stub.py"]),
    )
    client = _build_client(cfg, policy_handler=make_scripted_policy_v1())
    engine = CrsiEngine(
        repo_root=repo,
        config=cfg,
        dataset=dataset,
        llm_client=client,
        evaluator_fn=make_toy_worktree_evaluator(client, parallelism=2),
        private_dir=private_dir,
    )
    result = await engine.run(run_id="base-source-01", resume=False)
    sim = SimStore.load(Path(result.sim_path))
    return repo, result, sim


async def test_scenario_1_interlock_prevents_hallucinated_completion(tmp_path: Path):
    """Scenario 1: without interlock policy hallucinates finish and fails; with interlock it passes."""
    repo = seed_toy_repo(tmp_path / "repo_s1", interlock_enabled=False)
    main_sha = _git(repo, "rev-parse", "main")
    private_dir = tmp_path / "private_s1"
    dataset = materialize_private_dataset(private_dir, "base")

    cfg = CrsiConfig(
        project_name="toy-s1",
        domain="toy_base",
        memory_mode="full",
        budget=BudgetConfig(rounds=1, candidates=2, parallelism=2),
        harness=HarnessPathsConfig(editable=["harness/**"], frozen=["frozen_scorer_stub.py"]),
    )
    client = _build_client(cfg, policy_handler=make_scripted_policy_v1())

    # Without interlock, policy finishes on turn 1 with zero execution -> 0.0 score.
    pre_heldout = await _evaluate_heldout_commit(
        repo, main_sha, dataset, client, private_dir / "heldout_reports", "s1-before"
    )
    assert pre_heldout == pytest.approx(0.0)

    # CRSI round enables ExecutionGatingInterlock -> premature finish rejected -> passes 1.0.
    engine = CrsiEngine(
        repo_root=repo,
        config=cfg,
        dataset=dataset,
        llm_client=client,
        evaluator_fn=make_toy_worktree_evaluator(client, parallelism=2),
        private_dir=private_dir,
    )
    res = await engine.run(run_id="s1-interlock", resume=False)
    assert res.status == "completed"
    assert res.baseline_score == pytest.approx(0.0)
    assert res.final_score == pytest.approx(1.0)
    assert res.accepted_count == 1
    assert client.spend_tracker.spent_usd == 0.0

    # Verify held-out split also jumps 0.0 -> 1.0 and main branch stays untouched.
    post_heldout = await _evaluate_heldout_commit(
        repo, res.final_sha, dataset, client, private_dir / "heldout_reports", "s1-after"
    )
    assert post_heldout == pytest.approx(1.0)
    assert _git(repo, "rev-parse", "main") == main_sha

    sim = SimStore.load(Path(res.sim_path))
    assert len(sim.doc.problems) == 1
    assert "premature completion" in sim.doc.problems[0].name.lower()
    assert sim.doc.problems[0].attempts[0].observed.passed == 4


async def test_scenario_2_harder_split_full_sim_vs_no_sim(tmp_path: Path):
    """Scenario 2: on harder split, no-SIM tries to disable interlock & stalls; Full SIM adds quality check."""
    _, _, source_sim = await _run_source_base_evolution(tmp_path / "src")
    private_hard = tmp_path / "private_hard"
    hard_ds = materialize_private_dataset(private_hard, "hard")

    async def _run_arm(
        arm_name: str,
        memory_mode: MemoryMode,
        sim_store: SimStore | None,
    ) -> tuple[Path, CrsiRunResult, float]:
        repo = seed_toy_repo(
            tmp_path / f"repo_{arm_name}",
            interlock_enabled=True,
            quality_check_enabled=False,
        )
        cfg = CrsiConfig(
            project_name=f"toy-{arm_name}",
            domain="toy_hard",
            memory_mode=memory_mode,
            budget=BudgetConfig(rounds=1, candidates=2, parallelism=2),
            harness=HarnessPathsConfig(editable=["harness/**"], frozen=["frozen_scorer_stub.py"]),
        )
        client = _build_client(cfg, policy_handler=make_scripted_policy_v1())
        engine = CrsiEngine(
            repo_root=repo,
            config=cfg,
            dataset=hard_ds,
            llm_client=client,
            evaluator_fn=make_toy_worktree_evaluator(client, parallelism=2),
            sim_store=sim_store,
            private_dir=private_hard,
        )
        res = await engine.run(run_id=arm_name, resume=False)
        heldout_score = await _evaluate_heldout_commit(
            repo, res.final_sha, hard_ds, client, private_hard / "heldout_reports", arm_name
        )
        return repo, res, heldout_score

    # Arm A: Harness-only / no-SIM continuation.
    repo_none, res_none, held_none = await _run_arm("arm-no-sim", "none", SimStore())
    # Arm B: Carried Full SIM continuation.
    carried_full = source_sim.carry_over(
        into_arm="arm-full-sim",
        source_arm="base-source-01",
        source_domain="toy_base",
        source_model="fake-policy-v1",
    )
    repo_full, res_full, held_full = await _run_arm("arm-full-sim", "full", carried_full)

    # No-SIM tries to disable the interlock (blocked by interlock_guard -> D_t) and stalls at 0.75.
    hist_none = RunHistory.load_or_create(Path(res_none.history_path))
    assert any(r.drop_reason == "interlock_guard" for r in hist_none.records)
    assert res_none.accepted_count == 0
    assert res_none.final_score == pytest.approx(0.75)
    assert held_none == pytest.approx(0.75)
    _ = repo_none

    # Full SIM retains ExecutionGatingInterlock AND adds QualityFeasibilityCheck -> 1.00 evolve & held-out!
    assert res_full.accepted_count == 1
    assert res_full.final_score == pytest.approx(1.0)
    assert held_full == pytest.approx(1.0)
    assert res_full.final_score > res_none.final_score
    assert held_full > held_none

    evolved_interlock = _git(repo_full, "show", f"{res_full.final_sha}:harness/interlock.py")
    assert "INTERLOCK_ENABLED = True" in evolved_interlock
    assert "QUALITY_CHECK_ENABLED = True" in evolved_interlock


async def test_scenario_3_problem_only_ablation_lacks_full_sim_protection(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
):
    """Scenario 3: problem_only memory strips rationales/outcomes and fails to match Full SIM."""
    _, _, source_sim = await _run_source_base_evolution(tmp_path / "src")
    private_hard = tmp_path / "private_hard"
    hard_ds = materialize_private_dataset(private_hard, "hard")

    # Problem-only arm
    repo_po = seed_toy_repo(
        tmp_path / "repo_problem_only",
        interlock_enabled=True,
        quality_check_enabled=False,
    )
    carried_po = source_sim.carry_over(
        into_arm="arm-problem-only",
        source_arm="base-source-01",
        source_domain="toy_base",
        source_model="fake-policy-v1",
    )
    cfg_po = CrsiConfig(
        project_name="toy-problem-only",
        domain="toy_hard",
        memory_mode="problem_only",
        budget=BudgetConfig(rounds=1, candidates=2, parallelism=2),
        harness=HarnessPathsConfig(editable=["harness/**"], frozen=["frozen_scorer_stub.py"]),
    )
    client_po = _build_client(cfg_po, policy_handler=make_scripted_policy_v1())
    engine_po = CrsiEngine(
        repo_root=repo_po,
        config=cfg_po,
        dataset=hard_ds,
        llm_client=client_po,
        evaluator_fn=make_toy_worktree_evaluator(client_po, parallelism=2),
        sim_store=carried_po,
        private_dir=private_hard,
    )
    with caplog.at_level(logging.WARNING):
        res_po = await engine_po.run(run_id="arm-problem-only", resume=False)
    held_po = await _evaluate_heldout_commit(
        repo_po, res_po.final_sha, hard_ds, client_po, private_hard / "heldout_reports", "po"
    )

    # Full SIM arm
    repo_full = seed_toy_repo(
        tmp_path / "repo_full",
        interlock_enabled=True,
        quality_check_enabled=False,
    )
    carried_full = source_sim.carry_over(
        into_arm="arm-full",
        source_arm="base-source-01",
        source_domain="toy_base",
        source_model="fake-policy-v1",
    )
    cfg_full = CrsiConfig(
        project_name="toy-full",
        domain="toy_hard",
        memory_mode="full",
        budget=BudgetConfig(rounds=1, candidates=2, parallelism=2),
        harness=HarnessPathsConfig(editable=["harness/**"], frozen=["frozen_scorer_stub.py"]),
    )
    client_full = _build_client(cfg_full, policy_handler=make_scripted_policy_v1())
    engine_full = CrsiEngine(
        repo_root=repo_full,
        config=cfg_full,
        dataset=hard_ds,
        llm_client=client_full,
        evaluator_fn=make_toy_worktree_evaluator(client_full, parallelism=2),
        sim_store=carried_full,
        private_dir=private_hard,
    )
    res_full = await engine_full.run(run_id="arm-full", resume=False)
    held_full = await _evaluate_heldout_commit(
        repo_full,
        res_full.final_sha,
        hard_ds,
        client_full,
        private_hard / "heldout_reports",
        "full",
    )

    assert any("problem_only" in rec.message for rec in caplog.records)
    assert res_po.final_score < res_full.final_score
    assert held_po < held_full
    assert res_full.final_score == pytest.approx(1.0)
    assert held_full == pytest.approx(1.0)


async def test_scenario_4_cross_policy_carry_over_preserves_rationales_without_blank_start(
    tmp_path: Path,
):
    """Scenario 4: carrying evolved harness + Full SIM into fake-policy-v2 preserves rationales & adapts."""
    _, _, source_sim = await _run_source_base_evolution(tmp_path / "src")
    private_v2 = tmp_path / "private_v2"
    v2_ds = materialize_private_dataset(private_v2, "v2")

    # Carry evolved harness (interlock + quality check) + Full SIM into fake-policy-v2.
    repo_v2 = seed_toy_repo(
        tmp_path / "repo_v2",
        interlock_enabled=True,
        quality_check_enabled=True,
        v2_recovery_enabled=False,
    )
    carried_sim = source_sim.carry_over(
        into_arm="policy-v2-transfer",
        source_arm="base-source-01",
        source_domain="toy_base",
        source_model="fake-policy-v1",
    )

    cfg_v2 = CrsiConfig(
        project_name="toy-policy-v2",
        domain="toy_v2",
        memory_mode="full",
        models=ModelsConfig(
            policy=ModelRoleConfig(provider="fake", model="fake-policy-v2"),
            proposer=ModelRoleConfig(provider="fake", model="fake-proposer-v1"),
            selector=ModelRoleConfig(provider="fake", model="fake-selector-v1"),
        ),
        budget=BudgetConfig(rounds=1, candidates=2, parallelism=2),
        harness=HarnessPathsConfig(editable=["harness/**"], frozen=["frozen_scorer_stub.py"]),
    )
    client_v2 = _build_client(cfg_v2, policy_handler=make_scripted_policy_v2())
    engine_v2 = CrsiEngine(
        repo_root=repo_v2,
        config=cfg_v2,
        dataset=v2_ds,
        llm_client=client_v2,
        evaluator_fn=make_toy_worktree_evaluator(client_v2, parallelism=2),
        sim_store=carried_sim,
        private_dir=private_v2,
    )
    res_v2 = await engine_v2.run(run_id="policy-v2-transfer", resume=False)

    # Not a blank start: inherited guards immediately pass verification & optimization tasks (0.50).
    assert res_v2.baseline_score == pytest.approx(0.5)
    assert res_v2.final_score == pytest.approx(1.0)
    assert res_v2.accepted_count == 1

    held_v2 = await _evaluate_heldout_commit(
        repo_v2, res_v2.final_sha, v2_ds, client_v2, private_v2 / "heldout_reports", "v2"
    )
    assert held_v2 == pytest.approx(1.0)

    # Verify inherited rationales, carried_from metadata, and original fake-policy-v1 provenance remain intact.
    final_sim = SimStore.load(Path(res_v2.sim_path))
    assert final_sim.doc.carried_from is not None
    assert final_sim.doc.carried_from.arm == "base-source-01"
    assert final_sim.doc.carried_from.model == "fake-policy-v1"
    assert final_sim.doc.carried_from.into == "policy-v2-transfer"
    assert len(final_sim.doc.problems) == 2
    assert final_sim.doc.problems[0].attempts[0].model == "fake-policy-v1"
    assert final_sim.doc.problems[0].attempts[0].domain == "toy_base"
    assert final_sim.doc.problems[1].attempts[0].model == "fake-policy-v2"
    assert final_sim.doc.problems[1].attempts[0].domain == "toy_v2"

    evolved_interlock = _git(repo_v2, "show", f"{res_v2.final_sha}:harness/interlock.py")
    assert "INTERLOCK_ENABLED = True" in evolved_interlock
    assert "QUALITY_CHECK_ENABLED = True" in evolved_interlock
    assert "V2_RECOVERY_GUIDANCE_ENABLED = True" in evolved_interlock
