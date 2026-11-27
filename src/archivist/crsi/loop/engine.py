"""Resumable CRSI outer loop engine wiring worktrees, P, legal filter, S, D_t, and SIM."""

from __future__ import annotations

import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from archivist.crsi.config import CrsiConfig
from archivist.crsi.evaluator.splits import TaskDataset
from archivist.crsi.evaluator.types import EvaluationReport
from archivist.crsi.guards.heldout_firewall import HeldOutIsolationGuard
from archivist.crsi.guards.path_policy import PathPolicyGuard
from archivist.crsi.guards.secrets import SecretRedactor
from archivist.crsi.llm.base import RoutedLLMClient
from archivist.crsi.llm.budget import BudgetExhaustedError
from archivist.crsi.loop.history import DropReason, HistoryRecord, RunHistory
from archivist.crsi.loop.legal_filter import LegalEditFilter
from archivist.crsi.loop.manifest import CandidateProposal
from archivist.crsi.loop.proposer import ProposerAgent
from archivist.crsi.loop.selector import SelectorAgent
from archivist.crsi.loop.state import (
    RoundPhase,
    RunState,
    RunStatus,
    load_run_state,
    save_run_state,
    validate_run_id,
)
from archivist.crsi.loop.worktree import GitWorktreeManager
from archivist.crsi.sim.models import AttemptRecord, ObservedOutcome
from archivist.crsi.sim.store import SimStore

WorktreeEvaluatorFn = Callable[[Path, TaskDataset, int], Awaitable[EvaluationReport]]


class CrsiRunResult(BaseModel):
    """Final summary of a completed, resumed, or budget-stopped CRSI run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    run_branch: str
    base_sha: str
    final_sha: str
    baseline_score: float
    final_score: float
    rounds_completed: int
    accepted_count: int
    rejected_count: int
    status: RunStatus
    state_path: str
    sim_path: str
    history_path: str


def _collect_editable_sources(wt_path: Path, path_guard: PathPolicyGuard) -> dict[str, str]:
    """Read all existing editable UTF-8 files from ``wt_path``."""
    wt_guard = PathPolicyGuard(
        repo_root=wt_path,
        editable_patterns=path_guard.editable_patterns,
        extra_frozen_patterns=path_guard.frozen_patterns,
        private_dir=path_guard.private_dir,
    )
    sources: dict[str, str] = {}
    for file_path in sorted(wt_path.rglob("*")):
        if not file_path.is_file() or file_path.is_symlink():
            continue
        rel = file_path.relative_to(wt_path).as_posix()
        if wt_guard.is_editable(rel):
            sources[rel] = file_path.read_text(encoding="utf-8", errors="replace")
    return sources


class CrsiEngine:
    """Orchestrates the full resumable CRSI loop across rounds ``0 .. rounds-1``."""

    def __init__(
        self,
        *,
        repo_root: Path,
        config: CrsiConfig,
        dataset: TaskDataset,
        llm_client: RoutedLLMClient,
        evaluator_fn: WorktreeEvaluatorFn,
        sim_store: SimStore | None = None,
        private_dir: Path | None = None,
        redactor: SecretRedactor | None = None,
    ) -> None:
        self.repo_root = repo_root.resolve()
        self.config = config
        self.dataset = dataset
        self.llm_client = llm_client
        self.evaluator_fn = evaluator_fn
        self.redactor = redactor if redactor is not None else config.build_redactor()
        self.private_dir = private_dir.resolve() if private_dir is not None else None

        sim_disk_path = self.repo_root / config.paths.sim_path
        self.sim_store = (
            sim_store
            if sim_store is not None
            else SimStore.load_or_create(sim_disk_path, redactor=self.redactor)
        )

        self.path_guard = PathPolicyGuard(
            repo_root=self.repo_root,
            editable_patterns=config.harness.editable,
            extra_frozen_patterns=config.harness.frozen,
            private_dir=self.private_dir,
        )
        self.heldout_guard = HeldOutIsolationGuard.from_dataset(
            dataset,
            private_dir=self.private_dir,
        )
        self.proposer = ProposerAgent(
            llm_client=llm_client,
            heldout_guard=self.heldout_guard,
            memory_mode=config.memory_mode,
            redactor=self.redactor,
        )
        self.selector = SelectorAgent(
            llm_client=llm_client,
            heldout_guard=self.heldout_guard,
            regression_tolerance=config.budget.regression_tolerance,
            memory_mode=config.memory_mode,
            redactor=self.redactor,
        )

    def _run_dir(self, run_id: str) -> Path:
        safe_id = validate_run_id(run_id)
        base = (self.repo_root / self.config.paths.state_dir).resolve()
        target = (base / safe_id).resolve()
        if not target.is_relative_to(base):
            raise ValueError(f"Run directory escapes state_dir for run_id={run_id!r}")
        return target

    def _checkpoint(
        self, state_path: Path, state: RunState, wall_start: float, prior_wall: float
    ) -> None:
        state.elapsed_wall_s = round(prior_wall + (time.perf_counter() - wall_start), 3)
        state.spend_snapshot = self.llm_client.spend_tracker.snapshot()
        save_run_state(state_path, state, redactor=self.redactor)

    def _proposal_contains_secret(self, proposal: CandidateProposal) -> str | None:
        fields: list[tuple[str, str]] = [
            ("candidate_id", proposal.candidate_id),
            ("manifest.what", proposal.manifest.what),
            ("manifest.rationale", proposal.manifest.rationale),
        ]
        if proposal.manifest.new_problem_name:
            fields.append(("manifest.new_problem_name", proposal.manifest.new_problem_name))
        for path_str in proposal.manifest.files:
            fields.append(("manifest.files", path_str))
        for path_str in proposal.delete_files:
            fields.append(("delete_files", path_str))
        for path_str, content in proposal.file_edits.items():
            fields.append(("file_edits.path", path_str))
            fields.append(("file_edits.content", content))
        for label, text in fields:
            if self.redactor.contains_secret(text):
                return label
        return None

    async def run(
        self,
        *,
        run_id: str,
        resume: bool = True,
        force_rebaseline: bool = False,
        stop_after_phase: tuple[int, RoundPhase] | None = None,
    ) -> CrsiRunResult:
        """Execute or resume a CRSI run on dedicated branch ``<branch_prefix><run_id>``."""
        run_id = validate_run_id(run_id)
        run_dir = self._run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)
        state_path = run_dir / "run_state.json"
        history_path = run_dir / "history.jsonl"
        run_sim_path = run_dir / "sim.json"
        worktree_root = run_dir / "worktrees"

        wt_mgr = GitWorktreeManager(
            self.repo_root,
            worktree_root=worktree_root,
            base_branch=self.config.paths.base_branch,
            branch_prefix=self.config.paths.branch_prefix,
        )
        run_branch = wt_mgr.assert_safe_run_branch(f"{self.config.paths.branch_prefix}{run_id}")
        history = RunHistory.load_or_create(history_path, redactor=self.redactor)

        if resume and state_path.is_file():
            state = load_run_state(state_path)
            if state.spend_snapshot:
                self.llm_client.spend_tracker.restore_snapshot(state.spend_snapshot)
            if run_sim_path.is_file():
                self.sim_store = SimStore.load(run_sim_path, redactor=self.redactor)
            wt_mgr.ensure_run_branch(state.run_branch, base_ref=state.current_sha)
        else:
            base_sha = wt_mgr.ensure_run_branch(run_branch, base_ref="HEAD")
            state = RunState(
                run_id=run_id,
                run_branch=run_branch,
                base_sha=base_sha,
                current_sha=base_sha,
                current_round=0,
                phase="baseline",
                memory_mode=self.config.memory_mode,
                domain=self.config.domain,
                policy_model=self.config.models.policy.model,
            )
            self.sim_store.save(run_sim_path)
            save_run_state(state_path, state, redactor=self.redactor)

        prior_wall = state.elapsed_wall_s
        wall_start = time.perf_counter()
        pending_rebaseline = force_rebaseline

        try:
            while state.current_round < self.config.budget.rounds:
                elapsed = prior_wall + (time.perf_counter() - wall_start)
                if elapsed > self.config.budget.total_wall_timeout_s:
                    state.status = "stopped_budget"
                    self._checkpoint(state_path, state, wall_start, prior_wall)
                    break

                round_idx = state.current_round

                # ── Phase 1: Baseline (H_t -> F_t) ─────────────────────────────────
                if state.phase == "baseline":
                    need_eval = (
                        state.baseline_report is None
                        or pending_rebaseline
                        or (round_idx == 0 and state.baseline_score is None)
                    )
                    if need_eval:
                        base_wt = wt_mgr.checkout_detached_worktree(
                            f"r{round_idx}_baseline", state.current_sha
                        )
                        try:
                            wt_mgr.reset_and_clean(base_wt, state.current_sha)
                            base_rep = await self.evaluator_fn(
                                base_wt, self.dataset, self.config.budget.rollouts
                            )
                            self.heldout_guard.assert_evolve_only_report(base_rep)
                        finally:
                            wt_mgr.remove_worktree(base_wt)
                        state.baseline_report = base_rep
                        if state.baseline_score is None:
                            state.baseline_score = base_rep.mean_score
                        state.current_score = base_rep.mean_score
                        pending_rebaseline = False

                    state.phase = "propose"
                    self._checkpoint(state_path, state, wall_start, prior_wall)
                    if stop_after_phase == (round_idx, "baseline"):
                        break

                assert state.baseline_report is not None

                # ── Phase 2: Propose (P) ───────────────────────────────────────────
                if state.phase == "propose":
                    prop_wt = wt_mgr.checkout_detached_worktree(
                        f"r{round_idx}_propose", state.current_sha
                    )
                    try:
                        wt_mgr.reset_and_clean(prop_wt, state.current_sha)
                        editable_sources = _collect_editable_sources(prop_wt, self.path_guard)
                    finally:
                        wt_mgr.remove_worktree(prop_wt)

                    batch = await self.proposer.propose(
                        round_num=round_idx,
                        k_candidates=self.config.budget.candidates,
                        editable_sources=editable_sources,
                        baseline_report=state.baseline_report,
                        sim_doc=self.sim_store.doc,
                        history=history,
                        domain=self.config.domain,
                        policy_model=self.config.models.policy.model,
                    )
                    for inv in batch.invalid:
                        history.append(
                            HistoryRecord(
                                round=round_idx,
                                candidate_id=inv.candidate_id,
                                status="rejected",
                                drop_reason="invalid_manifest",
                                detail=inv.error,
                                baseline_score=state.baseline_report.mean_score,
                            )
                        )
                        state.rejected_count += 1

                    # Scan proposals for secrets BEFORE checkpointing into run_state.json.
                    clean_proposals: list[CandidateProposal] = []
                    for prop in batch.proposals:
                        leaked_field = self._proposal_contains_secret(prop)
                        if leaked_field is not None:
                            history.append(
                                HistoryRecord(
                                    round=round_idx,
                                    candidate_id=prop.candidate_id,
                                    status="rejected",
                                    drop_reason="secret_leak",
                                    detail=f"Secret material detected in candidate {leaked_field}.",
                                    problem_ref=prop.manifest.problem_ref_label(),
                                    what=prop.manifest.what,
                                    files=prop.manifest.files,
                                    predicted=prop.manifest.predicted,
                                    baseline_score=state.baseline_report.mean_score,
                                )
                            )
                            state.rejected_count += 1
                            continue
                        clean_proposals.append(prop)

                    state.round_proposals = [p.model_dump(mode="json") for p in clean_proposals]
                    state.round_legal_ids = []
                    state.round_diffs = {}
                    state.round_candidate_shas = {}
                    state.round_screened_ids = []
                    state.round_eval_reports = {}
                    state.round_selected_id = None
                    state.round_select_reason = ""
                    state.round_rejected_by_tolerance = False
                    state.round_tolerance_rejected_id = None
                    state.round_record_committed_sha = None
                    state.phase = "filter"
                    self._checkpoint(state_path, state, wall_start, prior_wall)
                    if stop_after_phase == (round_idx, "propose"):
                        break

                proposals = [
                    CandidateProposal.model_validate(item) for item in state.round_proposals
                ]
                proposals_by_id = {p.candidate_id: p for p in proposals}

                # ── Phase 3: Legal-Edit Filter (Deterministic) ─────────────────────
                if state.phase == "filter":
                    legal_filter = LegalEditFilter(
                        path_guard=self.path_guard,
                        evolve_ids=self.dataset.evolve_ids(),
                        sim_store=self.sim_store,
                        domain=self.config.domain,
                        model=self.config.models.policy.model,
                        check_argv=self.config.commands.check,
                        check_timeout_s=self.config.budget.task_timeout_s,
                        redactor=self.redactor,
                    )
                    legal_ids: list[str] = []
                    diffs: dict[str, str] = {}
                    cand_shas: dict[str, str] = {}

                    for prop in proposals:
                        cand_wt = wt_mgr.checkout_detached_worktree(
                            f"r{round_idx}_flt_{prop.candidate_id}", state.current_sha
                        )
                        try:
                            wt_mgr.reset_and_clean(cand_wt, state.current_sha)
                            before_sources = _collect_editable_sources(cand_wt, self.path_guard)
                            try:
                                touched = wt_mgr.apply_file_edits(
                                    cand_wt, prop.file_edits, prop.delete_files
                                )
                            except Exception as exc:
                                history.append(
                                    HistoryRecord(
                                        round=round_idx,
                                        candidate_id=prop.candidate_id,
                                        status="rejected",
                                        drop_reason="apply_failed",
                                        detail=str(exc),
                                        problem_ref=prop.manifest.problem_ref_label(),
                                        what=prop.manifest.what,
                                        files=prop.manifest.files,
                                        predicted=prop.manifest.predicted,
                                        baseline_score=state.baseline_report.mean_score,
                                    )
                                )
                                state.rejected_count += 1
                                continue

                            after_sources = _collect_editable_sources(cand_wt, self.path_guard)
                            diff_text = wt_mgr.diff_against_head(cand_wt)
                            flt = legal_filter.evaluate_candidate(
                                proposal=prop,
                                worktree_path=cand_wt,
                                touched_files=touched,
                                diff_text=diff_text,
                                before_contents=before_sources,
                                after_contents=after_sources,
                            )
                            if not flt.legal:
                                history.append(
                                    HistoryRecord(
                                        round=round_idx,
                                        candidate_id=prop.candidate_id,
                                        status="rejected",
                                        drop_reason=flt.drop_reason,
                                        detail=flt.detail,
                                        problem_ref=prop.manifest.problem_ref_label(),
                                        what=prop.manifest.what,
                                        files=flt.touched_files or prop.manifest.files,
                                        predicted=prop.manifest.predicted,
                                        baseline_score=state.baseline_report.mean_score,
                                    )
                                )
                                state.rejected_count += 1
                                continue

                            cand_sha = wt_mgr.commit_worktree_changes(
                                cand_wt,
                                message=(
                                    f"crsi({run_id}): r{round_idx} candidate {prop.candidate_id} "
                                    f"- {prop.manifest.what[:60]}"
                                ),
                            )
                            legal_ids.append(prop.candidate_id)
                            diffs[prop.candidate_id] = diff_text
                            cand_shas[prop.candidate_id] = cand_sha
                        finally:
                            wt_mgr.remove_worktree(cand_wt)

                    state.round_legal_ids = legal_ids
                    state.round_diffs = diffs
                    state.round_candidate_shas = cand_shas
                    state.phase = "screen"
                    self._checkpoint(state_path, state, wall_start, prior_wall)
                    if stop_after_phase == (round_idx, "filter"):
                        break

                # ── Phase 4: Selector Pre-Execution Screen (S_screen) ──────────────
                if state.phase == "screen":
                    legal_props = [
                        proposals_by_id[cid]
                        for cid in state.round_legal_ids
                        if cid in proposals_by_id
                    ]
                    if legal_props:
                        screen_dec = await self.selector.screen(
                            round_num=round_idx,
                            baseline_report=state.baseline_report,
                            candidates=legal_props,
                            diffs=state.round_diffs,
                            sim_doc=self.sim_store.doc,
                            history=history,
                        )
                        for dropped_id, drop_msg in screen_dec.dropped.items():
                            prop = proposals_by_id[dropped_id]
                            history.append(
                                HistoryRecord(
                                    round=round_idx,
                                    candidate_id=dropped_id,
                                    status="rejected",
                                    drop_reason="screen_dropped",
                                    detail=drop_msg,
                                    problem_ref=prop.manifest.problem_ref_label(),
                                    what=prop.manifest.what,
                                    files=prop.manifest.files,
                                    predicted=prop.manifest.predicted,
                                    baseline_score=state.baseline_report.mean_score,
                                    commit_sha=state.round_candidate_shas.get(dropped_id),
                                )
                            )
                            state.rejected_count += 1
                        state.round_screened_ids = screen_dec.kept_ids
                    else:
                        state.round_screened_ids = []

                    state.phase = "evaluate"
                    self._checkpoint(state_path, state, wall_start, prior_wall)
                    if stop_after_phase == (round_idx, "screen"):
                        break

                # ── Phase 5: Candidate Evaluation (F_t^C, reset+clean per candidate) ─
                if state.phase == "evaluate":
                    for cid in state.round_screened_ids:
                        if cid in state.round_eval_reports:
                            continue
                        cand_sha = state.round_candidate_shas[cid]
                        eval_wt = wt_mgr.checkout_detached_worktree(
                            f"r{round_idx}_eval_{cid}", cand_sha
                        )
                        try:
                            wt_mgr.reset_and_clean(eval_wt, cand_sha)
                            cand_rep = await self.evaluator_fn(
                                eval_wt, self.dataset, self.config.budget.rollouts
                            )
                            self.heldout_guard.assert_evolve_only_report(cand_rep)
                        finally:
                            wt_mgr.remove_worktree(eval_wt)
                        state.round_eval_reports[cid] = cand_rep
                        # Checkpoint after each candidate rollout for fine-grained resumability.
                        self._checkpoint(state_path, state, wall_start, prior_wall)

                    state.phase = "select"
                    self._checkpoint(state_path, state, wall_start, prior_wall)
                    if stop_after_phase == (round_idx, "evaluate"):
                        break

                # ── Phase 6: Selector Pick (S_pick) + Tolerance Floor ──────────────
                if state.phase == "select":
                    eval_props = [
                        proposals_by_id[cid]
                        for cid in state.round_screened_ids
                        if cid in proposals_by_id and cid in state.round_eval_reports
                    ]
                    if eval_props:
                        pick_dec = await self.selector.pick(
                            round_num=round_idx,
                            baseline_report=state.baseline_report,
                            candidates=eval_props,
                            candidate_reports=state.round_eval_reports,
                            diffs=state.round_diffs,
                            sim_doc=self.sim_store.doc,
                            history=history,
                        )
                        state.round_selected_id = pick_dec.selected_id
                        state.round_select_reason = self.redactor.redact_text(pick_dec.reason)
                        state.round_rejected_by_tolerance = pick_dec.rejected_by_tolerance
                        state.round_tolerance_rejected_id = pick_dec.rejected_candidate_id
                    else:
                        state.round_selected_id = None
                        state.round_select_reason = "No surviving candidates to select."
                        state.round_rejected_by_tolerance = False
                        state.round_tolerance_rejected_id = None

                    state.phase = "record"
                    self._checkpoint(state_path, state, wall_start, prior_wall)
                    if stop_after_phase == (round_idx, "select"):
                        break

                # ── Phase 7: SIM Update, Commit on crsi/<run_id>, Record D_t ───────
                if state.phase == "record":
                    selected_id = state.round_selected_id
                    round_baseline_score = state.baseline_report.mean_score
                    already_recorded_cids = {
                        (r.round, r.candidate_id) for r in history.records if r.round == round_idx
                    }

                    # Record non-selected evaluated candidates in D_t (idempotent on resume).
                    for cid in state.round_screened_ids:
                        if cid == selected_id or (round_idx, cid) in already_recorded_cids:
                            continue
                        prop = proposals_by_id[cid]
                        rej_rep = state.round_eval_reports.get(cid)
                        is_tol_rejected = state.round_rejected_by_tolerance and (
                            state.round_tolerance_rejected_id is None
                            or cid == state.round_tolerance_rejected_id
                        )
                        drop_reason: DropReason = (
                            "below_tolerance" if is_tol_rejected else "not_selected"
                        )
                        history.append(
                            HistoryRecord(
                                round=round_idx,
                                candidate_id=cid,
                                status="rejected",
                                drop_reason=drop_reason,
                                detail=state.round_select_reason,
                                problem_ref=prop.manifest.problem_ref_label(),
                                what=prop.manifest.what,
                                files=prop.manifest.files,
                                predicted=prop.manifest.predicted,
                                score=rej_rep.mean_score if rej_rep else None,
                                baseline_score=round_baseline_score,
                                commit_sha=state.round_candidate_shas.get(cid),
                            )
                        )
                        state.rejected_count += 1

                    if selected_id is not None and selected_id in state.round_eval_reports:
                        win_prop = proposals_by_id[selected_id]
                        win_rep = state.round_eval_reports[selected_id]
                        win_sha = state.round_candidate_shas[selected_id]

                        if state.round_record_committed_sha is None:
                            # Guard against duplicate SIM attempt if crashed after sim.save().
                            attempt_already_logged = any(
                                att.round == round_idx + 1 and att.sample == selected_id
                                for prob in self.sim_store.doc.problems
                                for att in prob.attempts
                            )
                            if not attempt_already_logged:
                                observed = ObservedOutcome.from_task_results(
                                    predicted=win_prop.manifest.predicted,
                                    before_passed=state.baseline_report.by_task_passed,
                                    after_passed=win_rep.by_task_passed,
                                    after_scores=win_rep.by_task_score,
                                )
                                attempt = AttemptRecord(
                                    round=round_idx + 1,
                                    sample=selected_id,
                                    accepted=True,
                                    backfilled=False,
                                    what=win_prop.manifest.what,
                                    files=win_prop.manifest.files,
                                    rationale=win_prop.manifest.rationale,
                                    predicted=win_prop.manifest.predicted,
                                    observed=observed,
                                    model=self.config.models.policy.model,
                                    domain=self.config.domain,
                                )
                                self.sim_store.record_attempt(
                                    attempt=attempt,
                                    problem_id=win_prop.manifest.problem_id,
                                    new_problem_name=win_prop.manifest.new_problem_name,
                                )
                                self.sim_store.save(run_sim_path)

                            # Commit updated .crsi/sim.json on top of the winning candidate SHA
                            # and advance ONLY the dedicated crsi/<run_id> branch.
                            commit_wt = wt_mgr.checkout_detached_worktree(
                                f"r{round_idx}_commit", win_sha
                            )
                            try:
                                self.sim_store.save(commit_wt / self.config.paths.sim_path)
                                final_round_sha = wt_mgr.commit_worktree_changes(
                                    commit_wt,
                                    message=(
                                        f"crsi({run_id}): accept {selected_id} "
                                        f"(score={win_rep.mean_score:.4f})"
                                    ),
                                )
                            finally:
                                wt_mgr.remove_worktree(commit_wt)

                            wt_mgr.advance_run_branch(state.run_branch, final_round_sha)
                            state.round_record_committed_sha = final_round_sha
                            state.current_sha = final_round_sha
                            state.baseline_report = win_rep
                            state.current_score = win_rep.mean_score
                            state.accepted_count += 1
                            self._checkpoint(state_path, state, wall_start, prior_wall)

                        if (round_idx, selected_id) not in already_recorded_cids:
                            history.append(
                                HistoryRecord(
                                    round=round_idx,
                                    candidate_id=selected_id,
                                    status="accepted",
                                    drop_reason=None,
                                    detail=state.round_select_reason,
                                    problem_ref=win_prop.manifest.problem_ref_label(),
                                    what=win_prop.manifest.what,
                                    files=win_prop.manifest.files,
                                    predicted=win_prop.manifest.predicted,
                                    score=win_rep.mean_score,
                                    baseline_score=round_baseline_score,
                                    commit_sha=state.round_record_committed_sha,
                                )
                            )

                    # Advance to next round cleanly.
                    state.current_round += 1
                    state.phase = (
                        "complete"
                        if state.current_round >= self.config.budget.rounds
                        else "baseline"
                    )
                    state.round_proposals = []
                    state.round_legal_ids = []
                    state.round_diffs = {}
                    state.round_candidate_shas = {}
                    state.round_screened_ids = []
                    state.round_eval_reports = {}
                    state.round_selected_id = None
                    state.round_select_reason = ""
                    state.round_rejected_by_tolerance = False
                    state.round_tolerance_rejected_id = None
                    state.round_record_committed_sha = None
                    self._checkpoint(state_path, state, wall_start, prior_wall)
                    if stop_after_phase == (round_idx, "record"):
                        break

            if state.current_round >= self.config.budget.rounds and state.status == "running":
                state.status = "completed"
                state.phase = "complete"
                self._checkpoint(state_path, state, wall_start, prior_wall)

        except BudgetExhaustedError:
            state.status = "stopped_budget"
            self._checkpoint(state_path, state, wall_start, prior_wall)
        finally:
            wt_mgr.cleanup_all()

        return CrsiRunResult(
            run_id=state.run_id,
            run_branch=state.run_branch,
            base_sha=state.base_sha,
            final_sha=state.current_sha,
            baseline_score=state.baseline_score if state.baseline_score is not None else 0.0,
            final_score=state.current_score if state.current_score is not None else 0.0,
            rounds_completed=state.current_round,
            accepted_count=state.accepted_count,
            rejected_count=state.rejected_count,
            status=state.status,
            state_path=str(state_path),
            sim_path=str(run_sim_path),
            history_path=str(history_path),
        )
