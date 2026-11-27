"""Unit tests for ChangeManifest and LegalEditFilter (SPEC-006)."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from archivist.crsi.guards.path_policy import PathPolicyGuard
from archivist.crsi.loop.legal_filter import LegalEditFilter
from archivist.crsi.loop.manifest import CandidateProposal, ChangeManifest
from archivist.crsi.sim.models import AttemptRecord, ObservedOutcome
from archivist.crsi.sim.store import SimStore

pytestmark = [pytest.mark.crsi, pytest.mark.unit]


def _make_manifest(**overrides: object) -> ChangeManifest:
    base: dict[str, object] = {
        "new_problem_name": "premature completion",
        "what": "Tighten completion checklist",
        "files": ["harness/interlock.py"],
        "rationale": "Addresses unverified finish claims on evo-1.",
        "predicted": ["evo-1"],
    }
    base.update(overrides)
    return ChangeManifest.model_validate(base)


def test_change_manifest_requires_exactly_one_problem_reference():
    with pytest.raises(ValidationError):
        ChangeManifest(
            problem_id=None,
            new_problem_name=None,
            what="x",
            files=["harness/a.py"],
            rationale="r",
            predicted=["evo-1"],
        )
    with pytest.raises(ValidationError):
        ChangeManifest(
            problem_id="P1",
            new_problem_name="duplicate",
            what="x",
            files=["harness/a.py"],
            rationale="r",
            predicted=["evo-1"],
        )
    m = _make_manifest()
    assert m.problem_ref_label() == "premature completion"


def test_legal_filter_drops_frozen_paths_invalid_predictions_secrets_and_failed_checks(
    tmp_path: Path,
):
    guard = PathPolicyGuard(repo_root=tmp_path, editable_patterns=("harness/**",))
    sim = SimStore()
    flt = LegalEditFilter(
        path_guard=guard,
        evolve_ids=frozenset({"evo-1", "evo-2"}),
        sim_store=sim,
        domain="toy_base",
        model="fake-policy-v1",
        check_argv=[sys.executable, "-c", "import sys; sys.exit(0)"],
    )

    # 1. Held-out / non-evolve prediction rejected.
    prop_bad_pred = CandidateProposal(
        candidate_id="c1",
        manifest=_make_manifest(predicted=["heldout-99"]),
        file_edits={"harness/interlock.py": "# ok\n"},
    )
    res = flt.evaluate_candidate(
        proposal=prop_bad_pred,
        worktree_path=tmp_path,
        touched_files=["harness/interlock.py"],
        diff_text="+ok",
        before_contents={"harness/interlock.py": "# before\n"},
        after_contents={"harness/interlock.py": "# ok\n"},
    )
    assert res.legal is False
    assert res.drop_reason == "invalid_manifest"

    # 2. Touching frozen path rejected.
    prop_frozen = CandidateProposal(
        candidate_id="c2",
        manifest=_make_manifest(files=["pyproject.toml"]),
        file_edits={"pyproject.toml": "# hack\n"},
    )
    res_frozen = flt.evaluate_candidate(
        proposal=prop_frozen,
        worktree_path=tmp_path,
        touched_files=["pyproject.toml"],
        diff_text="+hack",
        before_contents={},
        after_contents={},
    )
    assert res_frozen.legal is False
    assert res_frozen.drop_reason == "illegal_path"

    # 3. Embedded secret in diff rejected.
    prop_secret = CandidateProposal(
        candidate_id="c3",
        manifest=_make_manifest(),
        file_edits={"harness/interlock.py": "TOKEN = 'sk-proj-1234567890abcdefghij'\n"},
    )
    res_secret = flt.evaluate_candidate(
        proposal=prop_secret,
        worktree_path=tmp_path,
        touched_files=["harness/interlock.py"],
        diff_text="+TOKEN = 'sk-proj-1234567890abcdefghij'",
        before_contents={"harness/interlock.py": "# before\n"},
        after_contents={"harness/interlock.py": "TOKEN = 'sk-proj-1234567890abcdefghij'\n"},
    )
    assert res_secret.legal is False
    assert res_secret.drop_reason == "secret_leak"

    # 4. Failing project check command drops candidate.
    flt_fail_check = LegalEditFilter(
        path_guard=guard,
        evolve_ids=frozenset({"evo-1"}),
        sim_store=sim,
        domain="toy_base",
        model="fake-policy-v1",
        check_argv=[sys.executable, "-c", "import sys; sys.exit(1)"],
    )
    prop_ok = CandidateProposal(
        candidate_id="c4",
        manifest=_make_manifest(),
        file_edits={"harness/interlock.py": "# clean edit\n"},
    )
    res_check = flt_fail_check.evaluate_candidate(
        proposal=prop_ok,
        worktree_path=tmp_path,
        touched_files=["harness/interlock.py"],
        diff_text="+# clean edit",
        before_contents={"harness/interlock.py": "# before\n"},
        after_contents={"harness/interlock.py": "# clean edit\n"},
    )
    assert res_check.legal is False
    assert res_check.drop_reason == "check_failed"


def test_legal_filter_blocks_interlock_removal_without_sim_negative_evidence(tmp_path: Path):
    guard = PathPolicyGuard(repo_root=tmp_path, editable_patterns=("harness/**",))
    sim = SimStore()
    p1 = sim.record_attempt(
        new_problem_name="interlock positive fix",
        attempt=AttemptRecord(
            round=1,
            sample="s1",
            accepted=True,
            what="Enabled ExecutionGatingInterlock",
            files=["harness/interlock.py"],
            rationale="Blocks unverified finish",
            predicted=["evo-1"],
            observed=ObservedOutcome(
                fixed=["evo-1"], missed=[], regressed=[], passed=2, total=2, score=1.0
            ),
            model="fake-policy-v1",
            domain="toy_base",
        ),
    )

    before = {
        "harness/interlock.py": "INTERLOCK_ENABLED = True\nclass ExecutionGatingInterlock: pass\n"
    }
    after_disabled = {
        "harness/interlock.py": "INTERLOCK_ENABLED = False\nclass ExecutionGatingInterlock: pass\n"
    }

    flt = LegalEditFilter(
        path_guard=guard,
        evolve_ids=frozenset({"evo-1"}),
        sim_store=sim,
        domain="toy_base",
        model="fake-policy-v1",
    )

    # Without SIM negative evidence, disabling the interlock is blocked.
    prop_disable = CandidateProposal(
        candidate_id="c_disable",
        manifest=_make_manifest(problem_id=p1.id, new_problem_name=None),
        file_edits=after_disabled,
    )
    blocked = flt.evaluate_candidate(
        proposal=prop_disable,
        worktree_path=tmp_path,
        touched_files=["harness/interlock.py"],
        diff_text="-INTERLOCK_ENABLED = True\n+INTERLOCK_ENABLED = False",
        before_contents=before,
        after_contents=after_disabled,
    )
    assert blocked.legal is False
    assert blocked.drop_reason == "interlock_guard"

    # Once SIM records observed regression in the current domain/model and manifest cites it, allowed.
    p2 = sim.record_attempt(
        new_problem_name="interlock regression in current domain",
        attempt=AttemptRecord(
            round=2,
            sample="s2",
            accepted=False,
            what="Strict interlock regressed evo-1",
            files=["harness/interlock.py"],
            rationale="Observed regression",
            predicted=["evo-1"],
            observed=ObservedOutcome(
                fixed=[], missed=["evo-1"], regressed=["evo-1"], passed=0, total=2, score=0.0
            ),
            model="fake-policy-v1",
            domain="toy_base",
        ),
    )
    prop_with_evidence = CandidateProposal(
        candidate_id="c_disable_evidenced",
        manifest=_make_manifest(
            problem_id=p2.id,
            new_problem_name=None,
            interlock_removal_sim_evidence=p2.id,
        ),
        file_edits=after_disabled,
    )
    allowed = flt.evaluate_candidate(
        proposal=prop_with_evidence,
        worktree_path=tmp_path,
        touched_files=["harness/interlock.py"],
        diff_text="-INTERLOCK_ENABLED = True\n+INTERLOCK_ENABLED = False",
        before_contents=before,
        after_contents=after_disabled,
    )
    assert allowed.legal is True
