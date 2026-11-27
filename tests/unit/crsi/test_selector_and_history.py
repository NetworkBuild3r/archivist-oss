"""Unit tests for SelectorAgent regression floor and RunHistory D_t retention (SPEC-006)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from archivist.crsi.config import CrsiConfig
from archivist.crsi.evaluator.splits import SplitManifest, TaskDataset
from archivist.crsi.evaluator.types import EvaluationReport, TaskScore, TaskSpec
from archivist.crsi.guards.heldout_firewall import HeldOutIsolationGuard
from archivist.crsi.llm.base import RoutedLLMClient
from archivist.crsi.llm.fake import FakeScriptedAdapter
from archivist.crsi.loop.history import HistoryRecord, RunHistory
from archivist.crsi.loop.manifest import CandidateProposal, ChangeManifest
from archivist.crsi.loop.selector import SelectorAgent
from archivist.crsi.sim.models import SimDocument

pytestmark = [pytest.mark.crsi, pytest.mark.unit]


def _make_dataset() -> TaskDataset:
    t1 = TaskSpec(id="evo-1", prompt="Solve task 1", expected="ok")
    t2 = TaskSpec(id="held-1", prompt="Secret heldout prompt", expected="SECRET_TOKEN")
    return TaskDataset(
        tasks={"evo-1": t1, "held-1": t2},
        split=SplitManifest(evolve=["evo-1"], heldout=["held-1"]),
    )


def _make_report(score: float) -> EvaluationReport:
    return EvaluationReport(
        split="evolve",
        mean_score=score,
        passed=1 if score >= 0.8 else 0,
        total=1,
        task_scores=[TaskScore(task_id="evo-1", score=score, passed=score >= 0.8, summary="s")],
        by_task_passed={"evo-1": score >= 0.8},
        by_task_score={"evo-1": score},
    )


def _make_candidate(cid: str) -> CandidateProposal:
    return CandidateProposal(
        candidate_id=cid,
        manifest=ChangeManifest(
            new_problem_name="root cause",
            what=f"change {cid}",
            files=["harness/interlock.py"],
            rationale="test rationale",
            predicted=["evo-1"],
        ),
        file_edits={"harness/interlock.py": f"# {cid}\n"},
    )


async def test_selector_enforces_hard_regression_tolerance_even_if_llm_picks_regressed():
    ds = _make_dataset()
    guard = HeldOutIsolationGuard.from_dataset(ds)

    # Selector LLM explicitly tries to pick c_regressed (score 0.50 vs baseline 0.80).
    fake = FakeScriptedAdapter(
        responses={
            "selector": [
                json.dumps({"selected": "c_regressed", "reason": "looks neat"}),
            ]
        }
    )
    client = RoutedLLMClient(CrsiConfig(), adapters={"fake": fake})
    selector = SelectorAgent(
        llm_client=client,
        heldout_guard=guard,
        regression_tolerance=0.05,
    )

    cand = _make_candidate("c_regressed")
    decision = await selector.pick(
        round_num=0,
        baseline_report=_make_report(0.80),
        candidates=[cand],
        candidate_reports={"c_regressed": _make_report(0.50)},
        diffs={"c_regressed": "+change"},
        sim_doc=SimDocument(),
        history=RunHistory(),
    )
    assert decision.selected_id is None
    assert decision.rejected_by_tolerance is True
    assert "regression tolerance" in decision.reason


async def test_selector_screen_drops_duplicate_candidates_and_history_retains_drop_reasons(
    tmp_path: Path,
):
    ds = _make_dataset()
    guard = HeldOutIsolationGuard.from_dataset(ds)
    fake = FakeScriptedAdapter(
        responses={
            "selector": [
                json.dumps(
                    {
                        "keep": ["c1"],
                        "drop": {"c2": "duplicate of c1"},
                    }
                )
            ]
        }
    )
    client = RoutedLLMClient(CrsiConfig(), adapters={"fake": fake})
    selector = SelectorAgent(llm_client=client, heldout_guard=guard)

    c1, c2 = _make_candidate("c1"), _make_candidate("c2")
    screen = await selector.screen(
        round_num=0,
        baseline_report=_make_report(0.5),
        candidates=[c1, c2],
        diffs={"c1": "+a", "c2": "+a"},
        sim_doc=SimDocument(),
        history=RunHistory(),
    )
    assert screen.kept_ids == ["c1"]
    assert screen.dropped == {"c2": "duplicate of c1"}

    # Verify RunHistory persists both accepted & rejected candidates across reload.
    hist_path = tmp_path / "history.jsonl"
    hist = RunHistory.load_or_create(hist_path)
    hist.append(
        HistoryRecord(
            round=0,
            candidate_id="c2",
            status="rejected",
            drop_reason="screen_dropped",
            detail="duplicate of c1",
            problem_ref="root cause",
            what="change c2",
            files=["harness/interlock.py"],
        )
    )
    hist.append(
        HistoryRecord(
            round=0,
            candidate_id="c1",
            status="accepted",
            detail="improved score",
            problem_ref="root cause",
            what="change c1",
            files=["harness/interlock.py"],
            score=1.0,
            baseline_score=0.5,
        )
    )

    reloaded = RunHistory.load_or_create(hist_path)
    assert len(reloaded.records) == 2
    formatted = reloaded.format_for_prompt()
    assert "screen_dropped" in formatted
    assert "duplicate of c1" in formatted
    assert "status=accepted" in formatted
