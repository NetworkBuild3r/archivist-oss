"""Unit tests for predicted-vs-observed logging, carry-over, and render modes (SPEC-002)."""

from __future__ import annotations

import logging

import pytest

from archivist.crsi.sim.models import AttemptRecord, ObservedOutcome
from archivist.crsi.sim.render import SIM_USAGE_INSTRUCTIONS, format_sim_summary, render_sim
from archivist.crsi.sim.store import SimStore

pytestmark = [pytest.mark.crsi, pytest.mark.unit]


def test_observed_outcome_tracks_fixed_missed_and_regressed_predictions():
    before = {"t1": False, "t2": False, "t3": True, "t4": True}
    after = {"t1": True, "t2": False, "t3": False, "t4": True}
    scores = {"t1": 1.0, "t2": 0.25, "t3": 0.0, "t4": 1.0}

    observed = ObservedOutcome.from_task_results(
        predicted=["t1", "t2"],
        before_passed=before,
        after_passed=after,
        after_scores=scores,
    )
    assert observed.fixed == ["t1"]
    assert observed.missed == ["t2"]
    assert observed.regressed == ["t3"]
    assert observed.passed == 2
    assert observed.total == 4
    assert observed.score == pytest.approx(0.5625)


def test_full_render_never_truncates_and_includes_predicted_vs_observed():
    store = SimStore()
    for i in range(1, 26):
        store.record_attempt(
            new_problem_name=f"root cause {i}",
            attempt=AttemptRecord(
                round=i,
                sample=f"cand-{i}",
                accepted=True,
                what=f"change-marker-{i}",
                files=[f"harness/file_{i}.py"],
                rationale=f"detailed-rationale-{i} " + ("x" * 120),
                predicted=[f"pred-{i}-a", f"pred-{i}-b"],
                observed=ObservedOutcome(
                    fixed=[f"pred-{i}-a"],
                    missed=[f"pred-{i}-b"],
                    regressed=[],
                    passed=3,
                    total=4,
                    score=0.75,
                ),
                model="fake-policy-v1",
                domain="toy_base",
            ),
        )

    rendered = render_sim(store.doc, mode="full")
    assert SIM_USAGE_INSTRUCTIONS in rendered
    for i in range(1, 26):
        assert f"P{i}: root cause {i}" in rendered
        assert f"change-marker-{i}" in rendered
        assert f"detailed-rationale-{i}" in rendered
        assert f"pred-{i}-a" in rendered
        assert f"pred-{i}-b" in rendered
        assert "missed=[pred-" in rendered


def test_problem_only_ablation_strips_attempts_rationales_and_logs_warning(caplog):
    store = SimStore()
    store.record_attempt(
        new_problem_name="premature completion without execution",
        attempt=AttemptRecord(
            round=1,
            sample="s1",
            accepted=True,
            what="Added ExecutionGatingInterlock",
            files=["harness/interlock.py"],
            rationale="UNIQUE_RATIONALE_TOKEN",
            predicted=["task-a"],
            observed=ObservedOutcome(
                fixed=["task-a"], missed=[], regressed=[], passed=1, total=1, score=1.0
            ),
            model="fake-policy-v1",
            domain="toy_base",
        ),
    )

    with caplog.at_level(logging.WARNING):
        rendered = render_sim(store.doc, mode="problem_only")

    assert "P1: premature completion without execution (opened_round=1)" in rendered
    assert "UNIQUE_RATIONALE_TOKEN" not in rendered
    assert "Added ExecutionGatingInterlock" not in rendered
    assert "task-a" not in rendered
    assert "fake-policy-v1" not in rendered
    assert any("problem_only" in rec.message for rec in caplog.records)


def test_none_mode_returns_empty_string():
    store = SimStore()
    store.open_or_get_problem("anything", opened_round=1)
    assert render_sim(store.doc, mode="none") == ""


def test_carry_over_preserves_source_attempts_model_domain_and_stamps_carried_from():
    source = SimStore()
    source.record_attempt(
        new_problem_name="premature completion",
        attempt=AttemptRecord(
            round=1,
            sample="s1",
            accepted=True,
            what="Wired interlock",
            files=["harness/interlock.py"],
            rationale="Prevents hallucinated completion",
            predicted=["t1"],
            observed=ObservedOutcome(
                fixed=["t1"], missed=[], regressed=[], passed=4, total=4, score=1.0
            ),
            model="policy-alpha",
            domain="toy_base",
        ),
    )
    source.doc.aliases["P9"] = "P1"
    source.doc.next_id = 10

    carried = source.carry_over(
        into_arm="hard-split-run",
        source_arm="base-run",
        source_domain="toy_base",
        source_model="policy-alpha",
        carried_at="2026-11-27T18:00:00Z",
    )

    assert carried.doc.carried_from is not None
    assert carried.doc.carried_from.arm == "base-run"
    assert carried.doc.carried_from.domain == "toy_base"
    assert carried.doc.carried_from.model == "policy-alpha"
    assert carried.doc.carried_from.into == "hard-split-run"
    assert carried.doc.next_id == 10
    assert carried.doc.aliases == {"P9": "P1"}

    # Append a new attempt on a new model & domain; source attempt provenance stays intact.
    carried.record_attempt(
        problem_id="P1",
        attempt=AttemptRecord(
            round=1,
            sample="s2",
            accepted=True,
            what="Added quality check beside interlock",
            files=["harness/interlock.py"],
            rationale="Harder split adds suboptimal feasible answers",
            predicted=["hard-1"],
            observed=ObservedOutcome(
                fixed=["hard-1"], missed=[], regressed=[], passed=5, total=5, score=1.0
            ),
            model="policy-beta",
            domain="toy_hard",
        ),
    )

    p1 = carried.get_problem("P1")
    assert p1 is not None
    assert [(a.model, a.domain) for a in p1.attempts] == [
        ("policy-alpha", "toy_base"),
        ("policy-beta", "toy_hard"),
    ]
    # Source store was not mutated by carry_over deep copy.
    assert source.doc.carried_from is None
    assert len(source.doc.problems[0].attempts) == 1

    # Summary helper resolves aliases cleanly.
    summary = format_sim_summary(carried.doc, problem_id="P9")
    assert "policy-alpha" in summary
    assert "policy-beta" in summary
