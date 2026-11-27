"""Unit tests for deterministic SIM root-cause merge and alias resolution (SPEC-002)."""

from __future__ import annotations

import pytest

from archivist.crsi.sim.models import AttemptRecord, ObservedOutcome, ProblemEntry, SimDocument
from archivist.crsi.sim.store import SimStore, normalize_root_cause

pytestmark = [pytest.mark.crsi, pytest.mark.unit]


def _make_attempt(round_num: int, sample: str, what: str) -> AttemptRecord:
    passed = min(round_num, 5)
    return AttemptRecord(
        round=round_num,
        sample=sample,
        accepted=True,
        what=what,
        files=["harness/interlock.py"],
        rationale=f"rationale for {what}",
        predicted=[f"t-{round_num}"],
        observed=ObservedOutcome(
            fixed=[f"t-{round_num}"],
            missed=[],
            regressed=[],
            passed=passed,
            total=5,
            score=passed / 5.0,
        ),
        model="fake-policy-v1",
        domain="toy_base",
    )


def test_normalize_root_cause_collapses_case_and_whitespace():
    assert (
        normalize_root_cause("  Premature   Completion\tWithout\nVerification ")
        == "premature completion without verification"
    )


def test_record_attempt_auto_merges_matching_normalized_root_cause():
    store = SimStore()
    p1 = store.record_attempt(
        new_problem_name="Premature completion without execution",
        attempt=_make_attempt(1, "s1", "initial interlock"),
    )
    assert p1.id == "P1"
    assert store.doc.next_id == 2

    # Same root cause with different casing and extra whitespace reuses P1.
    p_again = store.record_attempt(
        new_problem_name="  premature   COMPLETION without execution  ",
        attempt=_make_attempt(2, "s2", "tightened interlock"),
    )
    assert p_again.id == "P1"
    assert store.doc.next_id == 2
    assert len(store.doc.problems) == 1
    assert [a.what for a in store.doc.problems[0].attempts] == [
        "initial interlock",
        "tightened interlock",
    ]


def test_manual_merge_preserves_all_attempts_in_chronological_order_and_flattens_aliases():
    doc = SimDocument(
        next_id=8,
        aliases={"P7": "P6"},
        problems=[
            ProblemEntry(
                id="P5",
                name="premature completion",
                opened_round=2,
                attempts=[
                    _make_attempt(2, "s1", "r2-p5"),
                    _make_attempt(5, "s2", "r5-p5"),
                ],
            ),
            ProblemEntry(
                id="P6",
                name="premature task completion",
                opened_round=1,
                attempts=[
                    _make_attempt(1, "s1", "r1-p6"),
                    _make_attempt(3, "s1", "r3-p6"),
                ],
            ),
        ],
    )
    store = SimStore(document=doc)
    merged = store.merge_entries("P6", "P5")

    assert merged.id == "P5"
    assert merged.opened_round == 1
    assert [a.what for a in merged.attempts] == ["r1-p6", "r2-p5", "r3-p6", "r5-p5"]
    assert [p.id for p in store.doc.problems] == ["P5"]
    assert store.doc.aliases == {"P6": "P5", "P7": "P5"}
    assert store.resolve_id("P7") == "P5"
    assert store.resolve_id("P6") == "P5"
    assert store.get_problem("P7") == merged
    # next_id stays monotonically non-decreasing.
    assert store.doc.next_id == 8

    # Citing an aliased ID appends onto canonical P5.
    store.record_attempt(problem_id="P7", attempt=_make_attempt(6, "s1", "r6-via-alias"))
    assert len(merged.attempts) == 5
    assert merged.attempts[-1].what == "r6-via-alias"


def test_deduplicate_by_root_cause_merges_existing_duplicates_preserving_history():
    doc = SimDocument(
        next_id=4,
        problems=[
            ProblemEntry(
                id="P1",
                name="Hallucinated   Success",
                opened_round=1,
                attempts=[_make_attempt(1, "s1", "first")],
            ),
            ProblemEntry(
                id="P2",
                name="Suboptimal plan",
                opened_round=2,
                attempts=[_make_attempt(2, "s1", "quality")],
            ),
            ProblemEntry(
                id="P3",
                name="hallucinated success",
                opened_round=3,
                attempts=[_make_attempt(3, "s1", "second")],
            ),
        ],
    )
    store = SimStore(document=doc)
    merges = store.deduplicate_by_root_cause()
    assert merges == [("P3", "P1")]
    assert [p.id for p in store.doc.problems] == ["P1", "P2"]
    assert store.doc.aliases == {"P3": "P1"}
    p1 = store.get_problem("P1")
    assert p1 is not None
    assert [a.what for a in p1.attempts] == ["first", "second"]
