"""Unit tests for frozen evaluator, disjoint splits, and held-out firewall (SPEC-005)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from archivist.crsi.evaluator.runner import FrozenEvaluator
from archivist.crsi.evaluator.splits import (
    SplitManifest,
    SplitViolationError,
    TaskDataset,
    build_deterministic_split,
    load_private_dataset,
    save_tasks_jsonl,
)
from archivist.crsi.evaluator.types import TaskScore, TaskSpec
from archivist.crsi.guards.heldout_firewall import HeldOutIsolationGuard, HeldOutLeakError
from archivist.crsi.harness.contract import HarnessResult, TaskInput, TurnRecord
from archivist.crsi.llm.base import LLMRequest, Message

pytestmark = [pytest.mark.crsi, pytest.mark.unit]


def _make_dataset(tmp_path: Path) -> tuple[TaskDataset, Path]:
    private_dir = tmp_path / "private"
    evolve_tasks = [
        TaskSpec(
            id="evo-1",
            prompt="Compute the sum of 2 + 2 in Python.",
            expected="4",
            verifier_params={"exact": True},
            tags=["math"],
        ),
        TaskSpec(
            id="evo-2",
            prompt="Reverse the string 'crsi' in Python.",
            expected="isrc",
            verifier_params={"exact": True},
            tags=["str"],
        ),
    ]
    heldout_tasks = [
        TaskSpec(
            id="held-secret-99",
            prompt="Secret held-out prompt about orbital eccentricity calculation.",
            expected="HELD_OUT_ANSWER_TOKEN_OMEGA_777",
            verifier_params={"secret_formula": "eccentricity"},
            tags=["heldout"],
        ),
    ]
    save_tasks_jsonl(private_dir / "tasks_evolve.jsonl", evolve_tasks)
    save_tasks_jsonl(private_dir / "tasks_heldout.jsonl", heldout_tasks)
    dataset = load_private_dataset(private_dir)
    return dataset, private_dir


class _SpyHarness:
    """Captures every TaskInput received by the harness for isolation assertions."""

    def __init__(self, answers: dict[str, str], received: list[TaskInput]) -> None:
        self._answers = answers
        self._received = received

    async def run(self, task: TaskInput) -> HarnessResult:
        self._received.append(task)
        ans = self._answers.get(task.task_id, "")
        return HarnessResult(
            task_id=task.task_id,
            completed=bool(ans),
            final_answer=ans,
            trace=[TurnRecord(step=1, action="finish", output=f"answered {ans}")],
        )


class _ExactMatchScorer:
    def score(self, task: TaskSpec, result: HarnessResult) -> TaskScore:
        ok = result.completed and result.final_answer.strip() == str(task.expected).strip()
        return TaskScore(
            task_id=task.id,
            score=1.0 if ok else 0.0,
            passed=ok,
            summary="exact match" if ok else "mismatch",
        )


def test_deterministic_split_is_disjoint_and_overlapping_splits_fail(tmp_path: Path):
    tasks = [TaskSpec(id=f"t-{i}", prompt=f"Task prompt number {i}") for i in range(10)]
    s1 = build_deterministic_split(tasks, heldout_fraction=0.3, seed=7)
    s2 = build_deterministic_split(tasks, heldout_fraction=0.3, seed=7)
    assert s1 == s2
    assert len(s1.heldout) == 3
    assert len(s1.evolve) == 7
    assert set(s1.evolve) & set(s1.heldout) == set()

    # Overlapping SplitManifest raises ValidationError.
    with pytest.raises(ValidationError, match="disjoint"):
        SplitManifest(evolve=["t-1", "t-2"], heldout=["t-2", "t-3"])

    # Overlapping private JSONL files raise SplitViolationError.
    bad_dir = tmp_path / "bad_private"
    save_tasks_jsonl(bad_dir / "tasks_evolve.jsonl", [tasks[0], tasks[1]])
    save_tasks_jsonl(bad_dir / "tasks_heldout.jsonl", [tasks[1], tasks[2]])
    with pytest.raises(SplitViolationError, match="Overlapping"):
        load_private_dataset(bad_dir)


async def test_evaluator_hands_harness_only_task_input_and_isolates_heldout_reports(
    tmp_path: Path,
):
    dataset, private_dir = _make_dataset(tmp_path)
    received: list[TaskInput] = []
    answers = {
        "evo-1": "4",
        "evo-2": "wrong",
        "held-secret-99": "HELD_OUT_ANSWER_TOKEN_OMEGA_777",
    }
    evaluator = FrozenEvaluator(
        harness_factory=lambda: _SpyHarness(answers, received),
        scorer=_ExactMatchScorer(),
        parallelism=2,
    )

    evolve_report = await evaluator.evaluate_evolve(dataset)
    assert evolve_report.split == "evolve"
    assert evolve_report.total == 2
    assert evolve_report.passed == 1
    assert evolve_report.mean_score == pytest.approx(0.5)
    assert evolve_report.by_task_passed == {"evo-1": True, "evo-2": False}

    # Verify harness saw ONLY evolve TaskInput objects with zero expected/verifier fields.
    assert {t.task_id for t in received} == {"evo-1", "evo-2"}
    for task_in in received:
        dumped = task_in.model_dump()
        assert "expected" not in dumped
        assert "verifier_params" not in dumped
        assert "HELD_OUT_ANSWER_TOKEN_OMEGA_777" not in json.dumps(dumped)

    # Held-out evaluation writes strictly to private human report path and refuses feedback format.
    report_dir = private_dir / "heldout_reports"
    heldout_report, report_path = await evaluator.evaluate_heldout(
        dataset,
        report_dir=report_dir,
        run_id="run-001",
    )
    assert heldout_report.split == "heldout"
    assert heldout_report.passed == 1
    assert report_path.is_file()
    assert report_path.parent == report_dir
    with pytest.raises(ValueError, match="Held-out"):
        heldout_report.format_evolve_feedback()


async def test_heldout_firewall_allows_clean_evolve_prompts_and_blocks_all_leaks(
    tmp_path: Path,
):
    dataset, private_dir = _make_dataset(tmp_path)
    received: list[TaskInput] = []
    evaluator = FrozenEvaluator(
        harness_factory=lambda: _SpyHarness({"evo-1": "4", "evo-2": "isrc"}, received),
        scorer=_ExactMatchScorer(),
    )
    evolve_report = await evaluator.evaluate_evolve(dataset)
    feedback_text = evolve_report.format_evolve_feedback()

    guard = HeldOutIsolationGuard.from_dataset(
        dataset,
        private_dir=private_dir,
        extra_forbidden_tokens=["HELDOUT_SCORE=0.9123"],
    )
    guard.assert_evolve_only_report(evolve_report)

    # Clean proposer & selector requests pass the firewall.
    clean_proposer_req = LLMRequest(
        role="proposer",
        messages=[
            Message(role="system", content="Propose harness edits based on evolve feedback."),
            Message(role="user", content=feedback_text),
        ],
    )
    clean_selector_req = LLMRequest(
        role="selector",
        messages=[
            Message(role="system", content="Select best candidate."),
            Message(role="user", content=feedback_text),
        ],
    )
    guard.assert_request_clean(clean_proposer_req)
    guard.assert_request_clean(clean_selector_req)

    # Every form of held-out leak into proposer or selector prompts MUST raise HeldOutLeakError.
    leaked_payloads = [
        ("task_id", "Please also fix task held-secret-99."),
        ("prompt", "Context: Secret held-out prompt about orbital eccentricity calculation."),
        ("answer", "Hint: output HELD_OUT_ANSWER_TOKEN_OMEGA_777."),
        ("score", "Previous run achieved HELDOUT_SCORE=0.9123."),
        ("path", f"Read {private_dir / 'tasks_heldout.jsonl'} for extra examples."),
    ]
    for label, bad_text in leaked_payloads:
        bad_req = LLMRequest(
            role="proposer",
            messages=[Message(role="user", content=f"{feedback_text}\n{bad_text}")],
        )
        with pytest.raises(HeldOutLeakError) as exc_info:
            guard.assert_request_clean(bad_req)
        # Firewall error must never echo back the raw held-out answer.
        assert "HELD_OUT_ANSWER_TOKEN_OMEGA_777" not in str(exc_info.value), label

    # Held-out file inside feedback_dir is also blocked.
    fb_dir = tmp_path / "feedback"
    fb_dir.mkdir()
    (fb_dir / "note.txt").write_text("leaked held-secret-99", encoding="utf-8")
    with pytest.raises(HeldOutLeakError):
        guard.assert_request_clean(clean_selector_req, feedback_dir=fb_dir)


async def test_external_command_evaluator_scores_valid_and_failing_runs(tmp_path: Path):
    dataset, _ = _make_dataset(tmp_path)

    good_scorer = tmp_path / "scorer_ok.py"
    good_scorer.write_text(
        """
import json, pathlib, sys
task_file, out_dir = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
task = json.loads(task_file.read_text(encoding="utf-8"))
result = {"score": 1.0, "passed": True, "summary": f"scored {task['id']}"}
(out_dir / "result.json").write_text(json.dumps(result), encoding="utf-8")
""",
        encoding="utf-8",
    )

    ev_ok = FrozenEvaluator(
        run_task_argv=[sys.executable, str(good_scorer), "{task_file}", "{out}"],
        parallelism=2,
        task_timeout_s=10.0,
    )
    report_ok = await ev_ok.evaluate_evolve(dataset, checkout_dir=tmp_path)
    assert report_ok.passed == 2
    assert report_ok.mean_score == pytest.approx(1.0)

    # Non-zero exit from external scorer deterministically scores 0.0.
    bad_scorer = tmp_path / "scorer_fail.py"
    bad_scorer.write_text("import sys; sys.exit(2)\n", encoding="utf-8")
    ev_fail = FrozenEvaluator(
        run_task_argv=[sys.executable, str(bad_scorer), "{task_file}", "{out}"],
        parallelism=1,
        task_timeout_s=10.0,
    )
    report_fail = await ev_fail.evaluate_evolve(dataset, checkout_dir=tmp_path)
    assert report_fail.passed == 0
    assert report_fail.mean_score == 0.0
    assert all("nonzero_exit:2" in ts.errors for ts in report_fail.task_scores)


@pytest.mark.parametrize(
    "exc_cls",
    ["BudgetExhaustedError", "PaidApiNotEnabledError", "HeldOutLeakError"],
)
async def test_maj4_evaluator_propagates_control_plane_and_security_errors(
    tmp_path: Path,
    exc_cls: str,
):
    from archivist.crsi.llm.budget import BudgetExhaustedError, PaidApiNotEnabledError

    exc_map = {
        "BudgetExhaustedError": BudgetExhaustedError("cap hit"),
        "PaidApiNotEnabledError": PaidApiNotEnabledError("paid off"),
        "HeldOutLeakError": HeldOutLeakError("leak"),
    }
    to_raise = exc_map[exc_cls]

    class _RaisingHarness:
        async def run(self, task: TaskInput) -> HarnessResult:
            _ = task
            raise to_raise

    dataset, _ = _make_dataset(tmp_path)
    evaluator = FrozenEvaluator(
        harness_factory=_RaisingHarness,
        scorer=_ExactMatchScorer(),
        parallelism=1,
    )
    with pytest.raises(type(to_raise)):
        await evaluator.evaluate_evolve(dataset)


async def test_min2_min3_min6_firewall_combined_layout_dir_symlink_run_id_and_trace_redaction(
    tmp_path: Path,
):
    dataset, private_dir = _make_dataset(tmp_path)
    guard = HeldOutIsolationGuard.from_dataset(dataset, private_dir=private_dir)

    # MIN-2: tasks.jsonl, split.json, private_dir text markers and dir symlinks in feedback_dir blocked.
    for leak_text in (
        "Read tasks.jsonl for answers.",
        "Inspect split.json to see splits.",
        f"Data lives under {private_dir.resolve()}",
    ):
        with pytest.raises(HeldOutLeakError):
            guard.assert_no_leak(prompt_text=leak_text)

    fb_dir = tmp_path / "fb_symlink_dir"
    fb_dir.mkdir()
    (fb_dir / "linked_private").symlink_to(private_dir)
    with pytest.raises(HeldOutLeakError):
        guard.assert_no_leak(prompt_text="clean prompt", feedback_dir=fb_dir)

    # MIN-3: traversal run_id rejected on evaluate_heldout.
    class _SecretTraceScorer:
        def score(self, task: TaskSpec, result: HarnessResult) -> TaskScore:
            return TaskScore(
                task_id=task.id,
                score=1.0,
                passed=True,
                summary="ok",
                trace_excerpt=["leaked sk-proj-1234567890abcdefghijklmnop"],
            )

    received: list[TaskInput] = []
    ev = FrozenEvaluator(
        harness_factory=lambda: _SpyHarness({"evo-1": "4", "evo-2": "isrc"}, received),
        scorer=_SecretTraceScorer(),
    )
    with pytest.raises(ValueError, match="run_id"):
        await ev.evaluate_heldout(dataset, report_dir=private_dir / "reports", run_id="../escape")

    # MIN-6: custom scorer trace_excerpt is redacted.
    rep = await ev.evaluate_evolve(dataset)
    for ts in rep.task_scores:
        assert all("sk-proj-1234567890abcdefghijklmnop" not in line for line in ts.trace_excerpt)
        assert all("[REDACTED]" in line for line in ts.trace_excerpt)
