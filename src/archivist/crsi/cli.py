"""Operator CLI for CRSI + SIM (``python -m archivist.crsi <command>``).

Commands:
- ``init``: initialize ``.crsi/config.toml`` (safe defaults) and empty ``.crsi/sim.json``
- ``validate``: verify config, secrets policy, path boundaries, git state, and private split
- ``dry-run``: compute exact task-run counts and upper-bound budget without executing anything
- ``run``: start or resume a CRSI loop on ``crsi/<run_id>`` (never auto-merges to main)
- ``status``: report run progress, branch, scores, accepted/rejected counts, and budget usage
- ``show-sim``: display ``.crsi/sim.json`` (human-readable or ``--json``)
- ``sim merge``: deterministically merge two SIM problem entries preserving all attempts
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import ValidationError

from archivist.crsi.config import CrsiConfig, MemoryMode, load_config
from archivist.crsi.evaluator.runner import FrozenEvaluator
from archivist.crsi.evaluator.splits import SplitViolationError, TaskDataset, load_private_dataset
from archivist.crsi.evaluator.types import EvaluationReport
from archivist.crsi.guards.path_policy import PathPolicyGuard
from archivist.crsi.guards.secrets import SecretLeakError
from archivist.crsi.llm.base import RoutedLLMClient
from archivist.crsi.llm.fake import FakeScriptedAdapter
from archivist.crsi.loop.engine import CrsiEngine, WorktreeEvaluatorFn
from archivist.crsi.loop.state import RunState, load_run_state
from archivist.crsi.sim.models import SimDocument
from archivist.crsi.sim.render import format_sim_summary
from archivist.crsi.sim.store import SimStore
from archivist.crsi.toy.scripted_agents import (
    make_scripted_policy_v1,
    make_scripted_policy_v2,
    make_scripted_proposer,
    make_scripted_selector,
    make_toy_worktree_evaluator,
    seed_toy_repo,
)
from archivist.crsi.toy.tasks import ToySplitName, materialize_private_dataset

_TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "config.example.toml"


def _default_private_dir(project_name: str) -> Path:
    return (Path.home() / ".crsi" / project_name / "private").resolve()


def _resolve_private_dir(cfg: CrsiConfig, override: str | None) -> Path:
    if override:
        return Path(override).expanduser().resolve()
    if cfg.paths.private_dir:
        return Path(cfg.paths.private_dir).expanduser().resolve()
    return _default_private_dir(cfg.project_name)


def _normalize_memory_mode(raw: str) -> MemoryMode:
    cleaned = raw.strip().lower().replace("-", "_")
    if cleaned not in {"full", "problem_only", "none"}:
        raise ValueError(f"Unsupported --memory-mode {raw!r}; expected full, problem-only, or none")
    return cast(MemoryMode, cleaned)


def _load_template_text() -> str:
    return _TEMPLATE_PATH.read_text(encoding="utf-8")


def _find_latest_run_state(state_dir: Path) -> tuple[str, Path] | None:
    if not state_dir.is_dir():
        return None
    candidates: list[tuple[float, str, Path]] = []
    for child in state_dir.iterdir():
        state_file = child / "run_state.json"
        if child.is_dir() and state_file.is_file():
            candidates.append((state_file.stat().st_mtime, child.name, state_file))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    _, run_id, state_path = candidates[0]
    return run_id, state_path


def _is_git_repo(repo_root: Path) -> tuple[bool, str]:
    git_exe = shutil.which("git")
    if git_exe is None:
        return False, "git executable not found on PATH."
    proc = subprocess.run(
        [git_exe, "rev-parse", "--is-inside-work-tree"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    if proc.returncode != 0 or proc.stdout.strip() != "true":
        return False, f"{repo_root} is not a git repository."
    head = subprocess.run(
        [git_exe, "rev-parse", "--verify", "HEAD"],
        cwd=str(repo_root),
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    if head.returncode != 0:
        return False, f"Git repository {repo_root} has no HEAD commit yet."
    return True, head.stdout.strip()


def compute_dry_run_plan(
    cfg: CrsiConfig,
    dataset: TaskDataset,
    *,
    include_heldout: bool = False,
) -> dict[str, Any]:
    """Calculate exact task-run counts and upper-bound LLM/spend budget without executing."""
    evolve_count = len(dataset.evolve_ids())
    heldout_count = len(dataset.heldout_ids())
    rounds = cfg.budget.rounds
    candidates = cfg.budget.candidates
    rollouts = cfg.budget.rollouts

    baseline_runs = evolve_count * rollouts
    candidate_runs = rounds * candidates * evolve_count * rollouts
    total_evolve_runs = baseline_runs + candidate_runs
    heldout_runs = (heldout_count * rollouts) if include_heldout else 0

    # Upper-bound LLM calls: 1 proposer + 2 selector calls per round + max_steps policy calls/task.
    max_policy_steps = max((t.max_steps for t in dataset.evolve_tasks()), default=12)
    upper_llm_calls = min(
        cfg.budget.max_llm_calls,
        (total_evolve_runs * max_policy_steps) + (rounds * 3),
    )

    paid_roles = [
        role
        for role in ("policy", "proposer", "selector")
        if cfg.models.for_role(
            cast(Literal["policy", "proposer", "selector"], role)
        ).is_paid_provider()
    ]
    return {
        "project_name": cfg.project_name,
        "domain": cfg.domain,
        "memory_mode": cfg.memory_mode,
        "evolve_tasks": evolve_count,
        "heldout_tasks": heldout_count,
        "rounds": rounds,
        "candidates_per_round": candidates,
        "rollouts": rollouts,
        "baseline_task_runs": baseline_runs,
        "candidate_task_runs": candidate_runs,
        "total_evolve_task_runs": total_evolve_runs,
        "heldout_task_runs": heldout_runs,
        "upper_bound_llm_calls": upper_llm_calls,
        "max_llm_calls_budget": cfg.budget.max_llm_calls,
        "allow_paid_apis": cfg.budget.allow_paid_apis,
        "max_spend_usd_budget": cfg.budget.max_spend_usd,
        "paid_roles": paid_roles,
    }


def _cmd_init(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).expanduser().resolve()
    repo_root.mkdir(parents=True, exist_ok=True)

    private_dir: Path | None
    if args.toy:
        if not (repo_root / ".git").exists():
            seed_toy_repo(repo_root, interlock_enabled=False)
        private_dir = (
            Path(args.private_dir).expanduser().resolve()
            if args.private_dir
            else (repo_root.parent / f"{repo_root.name}_private").resolve()
        )
        toy_split = cast(ToySplitName, args.toy_split)
        materialize_private_dataset(private_dir, split_name=toy_split)
    else:
        private_dir = Path(args.private_dir).expanduser().resolve() if args.private_dir else None

    crsi_dir = repo_root / ".crsi"
    crsi_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = crsi_dir / "config.toml"
    sim_path = crsi_dir / "sim.json"

    template = _load_template_text()
    if private_dir is not None:
        template = template.replace(
            '# private_dir = "/path/outside/repo/private"',
            f'private_dir = "{private_dir.as_posix()}"',
        )
    if args.toy:
        domain_map = {"base": "toy_base", "hard": "toy_hard", "v2": "toy_v2"}
        template = template.replace(
            'domain = "toy_base"',
            f'domain = "{domain_map[args.toy_split]}"',
        )

    if cfg_path.exists() and not args.force:
        print(f"Config already exists: {cfg_path} (use --force to overwrite)")
    else:
        cfg_path.write_text(template, encoding="utf-8")
        print(f"Initialized config: {cfg_path}")

    if sim_path.exists() and not args.force:
        print(f"SIM already exists: {sim_path} (use --force to overwrite)")
    else:
        SimStore(document=SimDocument()).save(sim_path)
        print(f"Initialized empty SIM: {sim_path}")

    if private_dir is not None and args.toy:
        print(f"Materialized private toy dataset ({args.toy_split}) at: {private_dir}")
    return 0


def _validate_all(
    *,
    repo_root: Path,
    config_path: Path,
    private_dir_override: str | None = None,
) -> tuple[CrsiConfig | None, TaskDataset | None, list[str]]:
    errors: list[str] = []
    if not config_path.is_file():
        return None, None, [f"Config file not found: {config_path}"]

    try:
        cfg = load_config(config_path)
    except (SecretLeakError, ValidationError, ValueError) as exc:
        return None, None, [f"Config validation failed: {exc}"]

    private_dir = _resolve_private_dir(cfg, private_dir_override)

    # Private dir must live outside the git repository to guarantee held-out isolation.
    try:
        if private_dir.is_relative_to(repo_root.resolve()):
            errors.append(
                f"private_dir ({private_dir}) must live outside repository root ({repo_root})."
            )
    except OSError:
        pass

    guard = PathPolicyGuard(
        repo_root=repo_root,
        editable_patterns=cfg.harness.editable,
        extra_frozen_patterns=cfg.harness.frozen,
        private_dir=private_dir,
    )
    cmd_argvs = [argv for argv in (cfg.commands.run_task, cfg.commands.check) if argv]
    errors.extend(guard.validate_configuration(command_argvs=cmd_argvs))

    git_ok, git_detail = _is_git_repo(repo_root)
    if not git_ok:
        errors.append(git_detail)

    dataset: TaskDataset | None = None
    if not private_dir.is_dir():
        errors.append(f"Private task directory not found: {private_dir}")
    else:
        try:
            dataset = load_private_dataset(private_dir)
        except (SplitViolationError, FileNotFoundError, ValidationError, ValueError) as exc:
            errors.append(f"Private dataset validation failed: {exc}")

    sim_file = repo_root / cfg.paths.sim_path
    if sim_file.is_file():
        try:
            SimStore.load(sim_file)
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            errors.append(f"SIM file {sim_file} is invalid: {exc}")

    return cfg, dataset, errors


def _cmd_validate(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).expanduser().resolve()
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else (repo_root / ".crsi" / "config.toml")
    )
    cfg, dataset, errors = _validate_all(
        repo_root=repo_root,
        config_path=config_path,
        private_dir_override=args.private_dir,
    )
    if errors:
        print("CRSI validation FAILED:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    assert cfg is not None and dataset is not None
    print(
        f"CRSI validation OK: project={cfg.project_name} domain={cfg.domain} "
        f"evolve={len(dataset.evolve_ids())} heldout={len(dataset.heldout_ids())} "
        f"allow_paid_apis={cfg.budget.allow_paid_apis} max_spend_usd={cfg.budget.max_spend_usd:.2f}"
    )
    return 0


def _apply_run_overrides(cfg: CrsiConfig, args: argparse.Namespace) -> CrsiConfig:
    budget_updates: dict[str, Any] = {}
    if getattr(args, "rounds", None) is not None:
        budget_updates["rounds"] = int(args.rounds)
    if getattr(args, "candidates", None) is not None:
        budget_updates["candidates"] = int(args.candidates)
    if getattr(args, "parallel", None) is not None:
        budget_updates["parallelism"] = int(args.parallel)
    if getattr(args, "rollouts", None) is not None:
        budget_updates["rollouts"] = int(args.rollouts)
    if getattr(args, "timeout", None) is not None:
        budget_updates["task_timeout_s"] = float(args.timeout)
    if getattr(args, "max_spend", None) is not None:
        budget_updates["max_spend_usd"] = float(args.max_spend)
    if getattr(args, "allow_paid", False):
        budget_updates["allow_paid_apis"] = True

    top_updates: dict[str, Any] = {}
    if getattr(args, "memory_mode", None) is not None:
        top_updates["memory_mode"] = _normalize_memory_mode(args.memory_mode)
    if budget_updates:
        merged_budget = cfg.budget.model_dump()
        merged_budget.update(budget_updates)
        top_updates["budget"] = merged_budget

    merged = cfg.model_dump()
    merged.update(top_updates)
    return CrsiConfig.model_validate(merged)


def _print_dry_run(plan: dict[str, Any], *, as_json: bool = False) -> None:
    if as_json:
        print(json.dumps(plan, indent=2, sort_keys=True))
        return
    print(f"CRSI Dry-Run Plan ({plan['project_name']} / {plan['domain']}):")
    print(f"  Memory mode:          {plan['memory_mode']}")
    print(f"  Evolve tasks:         {plan['evolve_tasks']}")
    print(f"  Held-out tasks:       {plan['heldout_tasks']} (strictly out-of-loop)")
    print(
        f"  Schedule:             {plan['rounds']} round(s) x {plan['candidates_per_round']} "
        f"candidate(s) x {plan['rollouts']} rollout(s)"
    )
    print(f"  Baseline task runs:   {plan['baseline_task_runs']}")
    print(f"  Candidate task runs:  {plan['candidate_task_runs']}")
    print(f"  Total evolve runs:    {plan['total_evolve_task_runs']}")
    if plan["heldout_task_runs"]:
        print(f"  Held-out runs:        {plan['heldout_task_runs']} (optional post-run)")
    print(
        f"  Upper-bound LLM calls:{plan['upper_bound_llm_calls']} "
        f"(cap: {plan['max_llm_calls_budget']})"
    )
    print(
        f"  Paid APIs enabled:    {plan['allow_paid_apis']} "
        f"(max spend: ${plan['max_spend_usd_budget']:.2f})"
    )
    print("  Executed tasks:       0 (dry-run)")


def _cmd_dry_run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).expanduser().resolve()
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else (repo_root / ".crsi" / "config.toml")
    )
    cfg, dataset, errors = _validate_all(
        repo_root=repo_root,
        config_path=config_path,
        private_dir_override=args.private_dir,
    )
    if errors or cfg is None or dataset is None:
        print("CRSI dry-run blocked by validation errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    try:
        effective_cfg = _apply_run_overrides(cfg, args)
    except (ValidationError, ValueError) as exc:
        print(f"Invalid CLI override: {exc}", file=sys.stderr)
        return 1

    plan = compute_dry_run_plan(
        effective_cfg,
        dataset,
        include_heldout=bool(getattr(args, "include_heldout", False)),
    )
    _print_dry_run(plan, as_json=bool(getattr(args, "json", False)))
    return 0


def _build_evaluator_for_cli(
    cfg: CrsiConfig,
    client: RoutedLLMClient,
) -> WorktreeEvaluatorFn:
    if cfg.commands.run_task:

        async def _external_eval(
            wt_path: Path,
            dataset: TaskDataset,
            rollouts: int,
        ) -> EvaluationReport:
            ev = FrozenEvaluator(
                run_task_argv=cfg.commands.run_task,
                parallelism=cfg.budget.parallelism,
                task_timeout_s=cfg.budget.task_timeout_s,
                redactor=client.redactor,
            )
            return await ev.evaluate_evolve(dataset, rollouts=rollouts, checkout_dir=wt_path)

        return _external_eval

    return make_toy_worktree_evaluator(
        client,
        parallelism=cfg.budget.parallelism,
        task_timeout_s=cfg.budget.task_timeout_s,
    )


def _cmd_run(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).expanduser().resolve()
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else (repo_root / ".crsi" / "config.toml")
    )
    cfg, dataset, errors = _validate_all(
        repo_root=repo_root,
        config_path=config_path,
        private_dir_override=args.private_dir,
    )
    if errors or cfg is None or dataset is None:
        print("CRSI run blocked by validation errors:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1

    try:
        effective_cfg = _apply_run_overrides(cfg, args)
    except (ValidationError, ValueError) as exc:
        print(f"Invalid CLI override: {exc}", file=sys.stderr)
        return 2

    if args.dry_run:
        plan = compute_dry_run_plan(effective_cfg, dataset)
        _print_dry_run(plan, as_json=False)
        return 0

    # Fail fast BEFORE any worktree/network I/O if paid providers are configured without opt-in.
    for role_name in ("policy", "proposer", "selector"):
        role_cfg = effective_cfg.models.for_role(
            cast(Literal["policy", "proposer", "selector"], role_name)
        )
        if role_cfg.is_paid_provider() and (
            not effective_cfg.budget.allow_paid_apis or effective_cfg.budget.max_spend_usd <= 0.0
        ):
            print(
                f"Refusing to run paid provider {role_cfg.provider!r} for role {role_name!r}: "
                "pass --allow-paid AND --max-spend <USD> (or set budget.allow_paid_apis=true "
                "and budget.max_spend_usd > 0 in config.toml).",
                file=sys.stderr,
            )
            return 2

    state_dir = repo_root / effective_cfg.paths.state_dir
    if args.run_id:
        run_id = args.run_id
    elif not args.new:
        latest = _find_latest_run_state(state_dir)
        run_id = (
            latest[0] if latest is not None else datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
        )
    else:
        run_id = datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")

    sim_store: SimStore | None = None
    force_rebaseline = False
    if args.carry_from:
        carry_path = Path(args.carry_from).expanduser().resolve()
        if not carry_path.is_file():
            print(f"--carry-from SIM file not found: {carry_path}", file=sys.stderr)
            return 1
        source_sim = SimStore.load(carry_path)
        sim_store = source_sim.carry_over(
            into_arm=run_id,
            source_arm=carry_path.parent.name or "carried-source",
            source_domain=args.carry_domain or effective_cfg.domain,
            source_model=args.carry_model or effective_cfg.models.policy.model,
        )
        force_rebaseline = True

    policy_handler = (
        make_scripted_policy_v2() if args.toy_policy == "v2" else make_scripted_policy_v1()
    )
    fake_adapter = FakeScriptedAdapter(
        handlers={
            "policy": policy_handler,
            "proposer": make_scripted_proposer(),
            "selector": make_scripted_selector(),
        }
    )
    client = RoutedLLMClient(effective_cfg, adapters={"fake": fake_adapter})
    evaluator_fn = _build_evaluator_for_cli(effective_cfg, client)
    private_dir = _resolve_private_dir(effective_cfg, args.private_dir)

    engine = CrsiEngine(
        repo_root=repo_root,
        config=effective_cfg,
        dataset=dataset,
        llm_client=client,
        evaluator_fn=evaluator_fn,
        sim_store=sim_store,
        private_dir=private_dir,
    )

    result = asyncio.run(
        engine.run(
            run_id=run_id,
            resume=not args.new,
            force_rebaseline=force_rebaseline,
        )
    )
    # Mirror latest SIM to configured repo sim_path only if it's inside .crsi/ (never touches main git commits).
    print(f"CRSI Run {result.run_id} [{result.status}]")
    print(
        f"  Branch:     {result.run_branch} (never auto-merged to {effective_cfg.paths.base_branch})"
    )
    print(f"  Commit SHA: {result.final_sha}")
    print(f"  Score:      {result.baseline_score:.4f} -> {result.final_score:.4f}")
    print(
        f"  Rounds:     {result.rounds_completed}/{effective_cfg.budget.rounds} "
        f"(accepted={result.accepted_count}, rejected={result.rejected_count})"
    )
    print(
        f"  Budget:     calls={client.spend_tracker.calls_used}/{effective_cfg.budget.max_llm_calls} "
        f"spend=${client.spend_tracker.spent_usd:.6f}/${effective_cfg.budget.max_spend_usd:.2f}"
    )
    print(f"  SIM:        {result.sim_path}")
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    repo_root = Path(args.repo_root).expanduser().resolve()
    config_path = (
        Path(args.config).expanduser().resolve()
        if args.config
        else (repo_root / ".crsi" / "config.toml")
    )
    cfg = load_config(config_path) if config_path.is_file() else CrsiConfig()
    state_dir = repo_root / cfg.paths.state_dir

    if args.run_id:
        state_path = state_dir / args.run_id / "run_state.json"
        if not state_path.is_file():
            print(f"Run state not found for run_id={args.run_id!r}: {state_path}", file=sys.stderr)
            return 1
    else:
        latest = _find_latest_run_state(state_dir)
        if latest is None:
            print(f"No CRSI runs found under {state_dir}", file=sys.stderr)
            return 1
        _, state_path = latest

    state: RunState = load_run_state(state_path)
    spend = state.spend_snapshot or {}
    payload: dict[str, Any] = {
        "run_id": state.run_id,
        "status": state.status,
        "run_branch": state.run_branch,
        "current_sha": state.current_sha,
        "current_round": state.current_round,
        "phase": state.phase,
        "memory_mode": state.memory_mode,
        "baseline_score": state.baseline_score,
        "current_score": state.current_score,
        "accepted_count": state.accepted_count,
        "rejected_count": state.rejected_count,
        "elapsed_wall_s": state.elapsed_wall_s,
        "llm_calls": int(spend.get("calls_used", 0)),
        "max_llm_calls": cfg.budget.max_llm_calls,
        "spent_usd": float(spend.get("spent_usd", 0.0)),
        "max_spend_usd": cfg.budget.max_spend_usd,
    }
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    print(f"Run ID:    {payload['run_id']} ({payload['status']})")
    print(f"Branch:    {payload['run_branch']} @ {payload['current_sha'][:12]}")
    print(
        f"Progress:  round={payload['current_round']} phase={payload['phase']} mode={payload['memory_mode']}"
    )
    print(f"Scores:    baseline={payload['baseline_score']} current={payload['current_score']}")
    print(f"Outcomes:  accepted={payload['accepted_count']} rejected={payload['rejected_count']}")
    print(
        f"Budget:    wall={payload['elapsed_wall_s']:.2f}s | "
        f"calls={payload['llm_calls']}/{payload['max_llm_calls']} | "
        f"spend=${payload['spent_usd']:.6f}/${payload['max_spend_usd']:.2f}"
    )
    return 0


def _resolve_sim_path(args: argparse.Namespace) -> Path:
    if getattr(args, "sim", None):
        return Path(args.sim).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else (repo_root / ".crsi" / "config.toml")
    )
    cfg = load_config(config_path) if config_path.is_file() else CrsiConfig()
    if getattr(args, "run_id", None):
        return repo_root / cfg.paths.state_dir / args.run_id / "sim.json"
    return repo_root / cfg.paths.sim_path


def _cmd_show_sim(args: argparse.Namespace) -> int:
    sim_path = _resolve_sim_path(args)
    if not sim_path.is_file():
        print(f"SIM file not found: {sim_path}", file=sys.stderr)
        return 1
    store = SimStore.load(sim_path)
    if args.json:
        print(json.dumps(store.doc.model_dump(mode="json"), indent=2, sort_keys=False))
        return 0
    print(format_sim_summary(store.doc, problem_id=args.problem))
    return 0


def _cmd_sim_merge(args: argparse.Namespace) -> int:
    sim_path = _resolve_sim_path(args)
    if not sim_path.is_file():
        print(f"SIM file not found: {sim_path}", file=sys.stderr)
        return 1
    store = SimStore.load(sim_path)
    try:
        merged = store.merge_entries(args.src_id, args.dst_id)
    except (KeyError, ValueError) as exc:
        print(f"SIM merge failed: {exc}", file=sys.stderr)
        return 1
    store.save(sim_path)
    print(
        f"Merged {args.src_id} -> {merged.id} ({len(merged.attempts)} attempt(s) preserved); "
        f"aliases={store.doc.aliases}"
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Build the ``python -m archivist.crsi`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m archivist.crsi",
        description="Continual Recursive Self-Improvement (CRSI) harness loop & SIM CLI.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # init
    p_init = sub.add_parser("init", help="Initialize .crsi/config.toml and empty .crsi/sim.json.")
    p_init.add_argument("--repo-root", default=".", help="Target repository directory.")
    p_init.add_argument("--private-dir", default=None, help="Private task dir outside git.")
    p_init.add_argument(
        "--toy",
        action="store_true",
        help="Scaffold deterministic toy harness & private toy dataset.",
    )
    p_init.add_argument(
        "--toy-split",
        choices=("base", "hard", "v2"),
        default="base",
        help="Toy split to materialize when --toy is set.",
    )
    p_init.add_argument("--force", action="store_true", help="Overwrite existing config/SIM.")
    p_init.set_defaults(func=_cmd_init)

    # validate
    p_val = sub.add_parser("validate", help="Validate config, secrets, paths, git, and splits.")
    p_val.add_argument("--repo-root", default=".")
    p_val.add_argument("--config", default=None)
    p_val.add_argument("--private-dir", default=None)
    p_val.set_defaults(func=_cmd_validate)

    # dry-run
    p_dry = sub.add_parser(
        "dry-run", help="Print exact task-run counts and budget without running."
    )
    p_dry.add_argument("--repo-root", default=".")
    p_dry.add_argument("--config", default=None)
    p_dry.add_argument("--private-dir", default=None)
    p_dry.add_argument("--rounds", type=int, default=None)
    p_dry.add_argument("--candidates", type=int, default=None)
    p_dry.add_argument("--rollouts", type=int, default=None)
    p_dry.add_argument("--memory-mode", default=None)
    p_dry.add_argument("--include-heldout", action="store_true")
    p_dry.add_argument("--json", action="store_true")
    p_dry.set_defaults(func=_cmd_dry_run)

    # run
    p_run = sub.add_parser("run", help="Start or resume a CRSI run on a dedicated crsi/* branch.")
    p_run.add_argument("--repo-root", default=".")
    p_run.add_argument("--config", default=None)
    p_run.add_argument("--private-dir", default=None)
    p_run.add_argument("--run-id", default=None)
    p_run.add_argument("--rounds", type=int, default=None)
    p_run.add_argument("--candidates", type=int, default=None)
    p_run.add_argument("--parallel", type=int, default=None)
    p_run.add_argument("--rollouts", type=int, default=None)
    p_run.add_argument("--timeout", type=float, default=None)
    p_run.add_argument("--max-spend", type=float, default=None)
    p_run.add_argument("--allow-paid", action="store_true")
    p_run.add_argument("--memory-mode", default=None)
    p_run.add_argument("--carry-from", default=None, help="Path to source sim.json to carry over.")
    p_run.add_argument("--carry-domain", default=None, help="Source domain label for carried_from.")
    p_run.add_argument("--carry-model", default=None, help="Source model label for carried_from.")
    p_run.add_argument("--toy-policy", choices=("v1", "v2"), default="v1")
    p_run.add_argument("--new", action="store_true", help="Start a fresh run instead of resuming.")
    p_run.add_argument("--dry-run", action="store_true", help="Print plan without executing.")
    p_run.set_defaults(func=_cmd_run)

    # status
    p_status = sub.add_parser("status", help="Show latest or specified CRSI run status.")
    p_status.add_argument("--repo-root", default=".")
    p_status.add_argument("--config", default=None)
    p_status.add_argument("--run-id", default=None)
    p_status.add_argument("--json", action="store_true")
    p_status.set_defaults(func=_cmd_status)

    # show-sim
    p_show = sub.add_parser("show-sim", help="Display SIM contents (full summary or raw JSON).")
    p_show.add_argument("--repo-root", default=".")
    p_show.add_argument("--config", default=None)
    p_show.add_argument("--sim", default=None)
    p_show.add_argument("--run-id", default=None)
    p_show.add_argument("--problem", default=None, help="Filter to one problem ID (e.g. P1).")
    p_show.add_argument("--json", action="store_true")
    p_show.set_defaults(func=_cmd_show_sim)

    # sim merge
    p_sim = sub.add_parser("sim", help="SIM maintenance subcommands (e.g. merge).")
    sim_sub = p_sim.add_subparsers(dest="sim_action", required=True)
    p_merge = sim_sub.add_parser("merge", help="Merge src problem ID into dst problem ID.")
    p_merge.add_argument("src_id", help="Source problem ID (e.g. P6).")
    p_merge.add_argument("dst_id", help="Destination canonical problem ID (e.g. P5).")
    p_merge.add_argument("--repo-root", default=".")
    p_merge.add_argument("--config", default=None)
    p_merge.add_argument("--sim", default=None)
    p_merge.add_argument("--run-id", default=None)
    p_merge.set_defaults(func=_cmd_sim_merge)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint returning process exit code."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    handler = args.func
    return int(handler(args))
