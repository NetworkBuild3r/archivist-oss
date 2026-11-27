"""FROZEN CRSI outer loop: worktrees, manifests, legal-edit filter, P/S, D_t, and resume."""

from __future__ import annotations

from archivist.crsi.loop.engine import CrsiEngine, CrsiRunResult, WorktreeEvaluatorFn
from archivist.crsi.loop.history import DropReason, HistoryRecord, RunHistory
from archivist.crsi.loop.legal_filter import (
    LegalEditFilter,
    LegalFilterResult,
    interlock_is_weakened_or_removed,
)
from archivist.crsi.loop.manifest import CandidateProposal, ChangeManifest
from archivist.crsi.loop.proposer import InvalidProposal, ProposalBatch, ProposerAgent
from archivist.crsi.loop.selector import PickDecision, ScreenDecision, SelectorAgent
from archivist.crsi.loop.state import (
    RoundPhase,
    RunState,
    load_run_state,
    save_run_state,
    validate_run_id,
)
from archivist.crsi.loop.worktree import GitWorktreeError, GitWorktreeManager

__all__ = [
    "CandidateProposal",
    "ChangeManifest",
    "CrsiEngine",
    "CrsiRunResult",
    "DropReason",
    "GitWorktreeError",
    "GitWorktreeManager",
    "HistoryRecord",
    "InvalidProposal",
    "LegalEditFilter",
    "LegalFilterResult",
    "PickDecision",
    "ProposalBatch",
    "ProposerAgent",
    "RoundPhase",
    "RunHistory",
    "RunState",
    "ScreenDecision",
    "SelectorAgent",
    "WorktreeEvaluatorFn",
    "interlock_is_weakened_or_removed",
    "load_run_state",
    "save_run_state",
    "validate_run_id",
]
