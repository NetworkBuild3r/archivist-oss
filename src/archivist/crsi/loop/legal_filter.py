"""Deterministic pre-evaluation LegalEditFilter and interlock anti-erosion guard."""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from archivist.crsi.guards.path_policy import PathPolicyGuard
from archivist.crsi.guards.secrets import SecretRedactor, sanitize_env
from archivist.crsi.loop.history import DropReason
from archivist.crsi.loop.manifest import CandidateProposal
from archivist.crsi.sim.store import SimStore

_INTERLOCK_DISABLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("interlock_disabled_flag", re.compile(r"\bINTERLOCK_ENABLED\s*=\s*False\b")),
    (
        "interlock_ctor_disabled",
        re.compile(r"ExecutionGatingInterlock\s*\([^)]*enabled\s*=\s*False"),
    ),
    ("interlock_return_true_bypass", re.compile(r"\bBYPASS_INTERLOCK\s*=\s*True\b")),
    (
        "interlock_unconditional_allow",
        re.compile(
            r"def\s+check_completion_attempt[\s\S]{0,240}?return\s+InterlockDecision\(\s*allowed\s*=\s*True",
        ),
    ),
)

_INTERLOCK_CORE_TOKENS: tuple[str, ...] = (
    "ExecutionGatingInterlock",
    "check_completion_attempt",
)


class LegalFilterResult(BaseModel):
    """Outcome of deterministically filtering one candidate before evaluation."""

    model_config = ConfigDict(extra="forbid")

    legal: bool
    drop_reason: DropReason | None = None
    detail: str = ""
    touched_files: list[str] = Field(default_factory=list)
    diff_text: str = ""


def interlock_is_weakened_or_removed(
    before_contents: Mapping[str, str],
    after_contents: Mapping[str, str],
) -> tuple[bool, str]:
    """Detect whether candidate edits remove, disable, or weaken ExecutionGatingInterlock."""
    for rel_path, before_src in before_contents.items():
        after_src = after_contents.get(rel_path)

        # If an interlock module itself was deleted, treat as removal.
        if after_src is None:
            if "interlock" in rel_path.lower() or "ExecutionGatingInterlock" in before_src:
                return True, f"Deleted interlock-bearing file {rel_path!r}"
            continue

        for token in _INTERLOCK_CORE_TOKENS:
            if token in before_src and token not in after_src:
                return True, f"Removed core interlock symbol {token!r} from {rel_path!r}"

        # Check explicit disable/bypass patterns newly introduced in after_src.
        for label, pattern in _INTERLOCK_DISABLE_PATTERNS:
            if pattern.search(after_src) and not pattern.search(before_src):
                return True, f"Introduced interlock disable/bypass ({label}) in {rel_path!r}"

        # Detect weakening of minimum execution proof (e.g. min_required_executions=0).
        if re.search(r"min_required_executions\s*=\s*0\b", after_src) and not re.search(
            r"min_required_executions\s*=\s*0\b", before_src
        ):
            return True, f"Weakened min_required_executions to 0 in {rel_path!r}"

    return False, ""


class LegalEditFilter:
    """Deterministic gate dropping illegal candidates BEFORE spending evaluation budget."""

    def __init__(
        self,
        *,
        path_guard: PathPolicyGuard,
        evolve_ids: frozenset[str],
        sim_store: SimStore,
        domain: str,
        model: str,
        check_argv: Sequence[str] = (),
        check_timeout_s: float = 60.0,
        redactor: SecretRedactor | None = None,
    ) -> None:
        self.path_guard = path_guard
        self.evolve_ids = evolve_ids
        self.sim_store = sim_store
        self.domain = domain
        self.model = model
        self.check_argv = [str(x) for x in check_argv]
        self.check_timeout_s = max(1.0, check_timeout_s)
        self.redactor = redactor if redactor is not None else SecretRedactor()

    def _run_project_check(self, worktree_path: Path) -> tuple[bool, str]:
        if not self.check_argv:
            return True, ""
        exe = self.check_argv[0]
        resolved = exe if Path(exe).is_absolute() and Path(exe).is_file() else shutil.which(exe)
        if resolved is None:
            return False, f"Project check executable not found: {exe!r}"
        cmd = [resolved, *self.check_argv[1:]]
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(worktree_path.resolve()),
                env=sanitize_env(),
                capture_output=True,
                text=True,
                timeout=self.check_timeout_s,
                check=False,
                shell=False,
            )
        except subprocess.TimeoutExpired:
            return False, f"Project check timed out after {self.check_timeout_s:.1f}s"

        if proc.returncode != 0:
            snippet = self.redactor.redact_text((proc.stderr or proc.stdout or "").strip()[:400])
            return False, f"Project check exited {proc.returncode}: {snippet}"
        return True, ""

    def evaluate_candidate(
        self,
        *,
        proposal: CandidateProposal,
        worktree_path: Path,
        touched_files: Sequence[str],
        diff_text: str,
        before_contents: Mapping[str, str],
        after_contents: Mapping[str, str],
    ) -> LegalFilterResult:
        """Apply all deterministic legality checks in strict fail-fast order."""
        manifest = proposal.manifest
        touched = sorted(set(touched_files))

        # 1. Verify manifest predictions target evolve tasks ONLY (never held-out or unknown).
        invalid_preds = [tid for tid in manifest.predicted if tid not in self.evolve_ids]
        if invalid_preds:
            return LegalFilterResult(
                legal=False,
                drop_reason="invalid_manifest",
                detail=f"Predicted task IDs not in evolve split: {invalid_preds}",
                touched_files=touched,
                diff_text=diff_text,
            )

        if (
            manifest.problem_id is not None
            and self.sim_store.get_problem(manifest.problem_id) is None
        ):
            return LegalFilterResult(
                legal=False,
                drop_reason="invalid_manifest",
                detail=f"Manifest cites unknown SIM problem_id {manifest.problem_id!r}",
                touched_files=touched,
                diff_text=diff_text,
            )

        # 2. Enforce editable/frozen path policy on every touched and declared file.
        if not touched:
            return LegalFilterResult(
                legal=False,
                drop_reason="invalid_manifest",
                detail="Candidate produced zero file changes.",
                touched_files=touched,
                diff_text=diff_text,
            )

        wt_guard = PathPolicyGuard(
            repo_root=worktree_path,
            editable_patterns=self.path_guard.editable_patterns,
            extra_frozen_patterns=self.path_guard.frozen_patterns,
            private_dir=self.path_guard.private_dir,
        )
        all_paths = sorted(
            set(touched)
            | set(manifest.files)
            | set(proposal.file_edits.keys())
            | set(proposal.delete_files)
        )
        for rel in all_paths:
            res = wt_guard.check_path(rel)
            if not res.allowed:
                return LegalFilterResult(
                    legal=False,
                    drop_reason="illegal_path",
                    detail=f"Path {rel!r} rejected: {res.reason}",
                    touched_files=touched,
                    diff_text=diff_text,
                )

        # 3. Secret scan across diff and manifest prose.
        for label, text in (
            ("diff", diff_text),
            ("manifest.what", manifest.what),
            ("manifest.rationale", manifest.rationale),
        ):
            if self.redactor.contains_secret(text):
                return LegalFilterResult(
                    legal=False,
                    drop_reason="secret_leak",
                    detail=f"Secret material detected in candidate {label}.",
                    touched_files=touched,
                    diff_text=self.redactor.redact_text(diff_text),
                )

        # 4. Anti-erosion interlock guard: deleting/weakening interlock requires SIM negative evidence.
        weakened, reason = interlock_is_weakened_or_removed(before_contents, after_contents)
        if weakened:
            ev_id = manifest.interlock_removal_sim_evidence
            has_evidence = ev_id is not None and self.sim_store.has_negative_evidence_for(
                ev_id,
                domain=self.domain,
                model=self.model,
            )
            if not has_evidence:
                return LegalFilterResult(
                    legal=False,
                    drop_reason="interlock_guard",
                    detail=(
                        f"{reason}. Weakening or deleting ExecutionGatingInterlock requires "
                        "interlock_removal_sim_evidence citing a SIM entry with negative "
                        "observed evidence in the current domain/model."
                    ),
                    touched_files=touched,
                    diff_text=diff_text,
                )

        # 5. Project check command (lint/syntax/unit tests inside worktree).
        check_ok, check_detail = self._run_project_check(worktree_path)
        if not check_ok:
            return LegalFilterResult(
                legal=False,
                drop_reason="check_failed",
                detail=check_detail,
                touched_files=touched,
                diff_text=diff_text,
            )

        return LegalFilterResult(
            legal=True,
            drop_reason=None,
            detail="passed all legal-edit checks",
            touched_files=touched,
            diff_text=diff_text,
        )
