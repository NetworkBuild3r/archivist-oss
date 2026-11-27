"""Unit tests for SIM schema validation and atomic JSON persistence (SPEC-002)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from archivist.crsi.guards.secrets import SecretLeakError
from archivist.crsi.sim.models import (
    AttemptRecord,
    CarriedFrom,
    ObservedOutcome,
    ProblemEntry,
    SimDocument,
)
from archivist.crsi.sim.store import SimStore

pytestmark = [pytest.mark.crsi, pytest.mark.unit]


def _sample_attempt(round_num: int = 1, sample: str = "s1") -> AttemptRecord:
    return AttemptRecord(
        round=round_num,
        sample=sample,
        accepted=True,
        backfilled=False,
        what="Wired ExecutionGatingInterlock to block unverified completion",
        files=["src/archivist/crsi/harness/interlock.py", "src/archivist/crsi/harness/runner.py"],
        rationale="Policy declared task_complete without executing any command",
        predicted=["task-1", "task-2"],
        observed=ObservedOutcome(
            fixed=["task-1", "task-2"],
            missed=[],
            regressed=[],
            passed=6,
            total=6,
            score=1.0,
        ),
        model="fake-policy-v1",
        domain="toy_base",
    )


def test_sim_schema_validates_full_canonical_document_and_round_trips(tmp_path: Path):
    doc = SimDocument(
        schema_version="1.0",
        next_id=3,
        carried_from=CarriedFrom(
            arm="source-run-01",
            domain="toy_base",
            model="fake-policy-v1",
            into="target-run-02",
            carried_at="2026-11-27T17:00:00Z",
        ),
        aliases={"P2": "P1"},
        problems=[
            ProblemEntry(
                id="P1",
                name="premature completion without execution verification",
                opened_round=1,
                attempts=[_sample_attempt()],
            )
        ],
    )
    sim_file = tmp_path / ".crsi" / "sim.json"
    store = SimStore(document=doc)
    written_path = store.save(sim_file)
    assert written_path == sim_file
    assert sim_file.exists()

    # Ensure atomic write left no stray temporary files behind.
    assert [p.name for p in sim_file.parent.iterdir()] == ["sim.json"]

    raw_json = json.loads(sim_file.read_text(encoding="utf-8"))
    assert raw_json["next_id"] == 3
    assert raw_json["aliases"] == {"P2": "P1"}
    assert raw_json["carried_from"]["arm"] == "source-run-01"
    assert len(raw_json["problems"]) == 1
    assert raw_json["problems"][0]["attempts"][0]["observed"]["fixed"] == ["task-1", "task-2"]

    reloaded = SimStore.load(sim_file)
    assert reloaded.doc == doc


def test_sim_schema_rejects_invalid_invariants():
    # Invalid problem ID format
    with pytest.raises(ValidationError):
        ProblemEntry(id="problem-1", name="bad id", opened_round=1)

    # passed > total
    with pytest.raises(ValidationError):
        ObservedOutcome(fixed=[], missed=[], regressed=[], passed=5, total=3, score=1.0)

    # next_id too low
    with pytest.raises(ValidationError):
        SimDocument(
            next_id=1,
            problems=[ProblemEntry(id="P1", name="x", opened_round=1)],
        )

    # self-alias
    with pytest.raises(ValidationError):
        SimDocument(next_id=3, aliases={"P1": "P1"})

    # alias cycle
    with pytest.raises(ValidationError):
        SimDocument(next_id=4, aliases={"P1": "P2", "P2": "P1"})

    # duplicate problem IDs
    with pytest.raises(ValidationError):
        SimDocument(
            next_id=3,
            problems=[
                ProblemEntry(id="P1", name="a", opened_round=1),
                ProblemEntry(id="P1", name="b", opened_round=2),
            ],
        )


def test_sim_save_refuses_embedded_secret_material(tmp_path: Path):
    doc = SimDocument(
        next_id=2,
        problems=[
            ProblemEntry(
                id="P1",
                name="leaked key test",
                opened_round=1,
                attempts=[
                    _sample_attempt().model_copy(
                        update={"rationale": "oops sk-proj-1234567890abcdefghijklmnop"}
                    )
                ],
            )
        ],
    )
    store = SimStore(document=doc)
    with pytest.raises(SecretLeakError):
        store.save(tmp_path / "sim.json")
    assert not (tmp_path / "sim.json").exists()
