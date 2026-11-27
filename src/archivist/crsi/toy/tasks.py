"""FROZEN deterministic toy datasets (base split, harder split, and policy-v2 transfer split)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from archivist.crsi.evaluator.splits import SplitManifest, TaskDataset, save_tasks_jsonl
from archivist.crsi.evaluator.types import TaskSpec

ToySplitName = Literal["base", "hard", "v2"]


def build_base_dataset() -> TaskDataset:
    """Base toy split: execution-verified coding/math tasks (evolve + disjoint held-out)."""
    evolve = [
        TaskSpec(
            id="base-evo-1",
            prompt="Compute 6 * 7 in Python, verify by executing it, and return the result.",
            expected="42",
            verifier_params={"kind": "verification", "expr": "6 * 7"},
            tags=["verification", "base"],
        ),
        TaskSpec(
            id="base-evo-2",
            prompt="Compute sum(range(1, 11)) in Python, verify by executing it, and return the result.",
            expected="55",
            verifier_params={"kind": "verification", "expr": "sum(range(1, 11))"},
            tags=["verification", "base"],
        ),
        TaskSpec(
            id="base-evo-3",
            prompt="Compute len('archivist-crsi') in Python, verify by executing it, and return the result.",
            expected="14",
            verifier_params={"kind": "verification", "expr": "len('archivist-crsi')"},
            tags=["verification", "base"],
        ),
        TaskSpec(
            id="base-evo-4",
            prompt="Compute 2 ** 8 in Python, verify by executing it, and return the result.",
            expected="256",
            verifier_params={"kind": "verification", "expr": "2 ** 8"},
            tags=["verification", "base"],
        ),
    ]
    heldout = [
        TaskSpec(
            id="base-held-1",
            prompt="Held-out verification task: compute 13 * 17 in Python and verify.",
            expected="221",
            verifier_params={"kind": "verification", "expr": "13 * 17"},
            tags=["verification", "heldout"],
        ),
        TaskSpec(
            id="base-held-2",
            prompt="Held-out verification task: compute 99 + 101 in Python and verify.",
            expected="200",
            verifier_params={"kind": "verification", "expr": "99 + 101"},
            tags=["verification", "heldout"],
        ),
    ]
    all_tasks = {t.id: t for t in (*evolve, *heldout)}
    return TaskDataset(
        tasks=all_tasks,
        split=SplitManifest(
            evolve=[t.id for t in evolve],
            heldout=[t.id for t in heldout],
            seed=101,
        ),
    )


def build_hard_dataset() -> TaskDataset:
    """Harder split: combines execution-verification tasks with feasible-but-suboptimal tasks."""
    evolve = [
        TaskSpec(
            id="hard-evo-1",
            prompt="Compute 7 * 9 in Python, verify by executing it, and return the result.",
            expected="63",
            verifier_params={"kind": "verification", "expr": "7 * 9"},
            tags=["verification", "hard"],
        ),
        TaskSpec(
            id="hard-evo-2",
            prompt="Compute 15 * 15 in Python, verify by executing it, and return the result.",
            expected="225",
            verifier_params={"kind": "verification", "expr": "15 * 15"},
            tags=["verification", "hard"],
        ),
        TaskSpec(
            id="hard-opt-evo-1",
            prompt=(
                "Optimization task (opt-1): produce a feasible plan and optimize objective score "
                "to at least 0.90. Verify via Python execution."
            ),
            expected="0.90",
            verifier_params={"kind": "optimization", "min_score": 0.90},
            tags=["optimization", "hard"],
        ),
        TaskSpec(
            id="hard-opt-evo-2",
            prompt=(
                "Optimization task (opt-2): satisfy all constraints and refine objective score "
                "to at least 0.90. Verify via Python execution."
            ),
            expected="0.90",
            verifier_params={"kind": "optimization", "min_score": 0.90},
            tags=["optimization", "hard"],
        ),
    ]
    heldout = [
        TaskSpec(
            id="hard-held-1",
            prompt="Held-out verification task: compute 19 * 11 in Python and verify.",
            expected="209",
            verifier_params={"kind": "verification", "expr": "19 * 11"},
            tags=["verification", "heldout"],
        ),
        TaskSpec(
            id="hard-opt-held-1",
            prompt=(
                "Held-out optimization task: produce a feasible schedule and maximize objective "
                "score to at least 0.90. Verify via Python execution."
            ),
            expected="0.90",
            verifier_params={"kind": "optimization", "min_score": 0.90},
            tags=["optimization", "heldout"],
        ),
    ]
    all_tasks = {t.id: t for t in (*evolve, *heldout)}
    return TaskDataset(
        tasks=all_tasks,
        split=SplitManifest(
            evolve=[t.id for t in evolve],
            heldout=[t.id for t in heldout],
            seed=202,
        ),
    )


def build_policy_v2_dataset() -> TaskDataset:
    """Transfer split for fake-policy-v2 with execution + normalization-recovery tasks."""
    evolve = [
        TaskSpec(
            id="v2-evo-1",
            prompt="Compute 8 * 8 in Python, verify by executing it, and return the result.",
            expected="64",
            verifier_params={"kind": "verification", "expr": "8 * 8"},
            tags=["verification", "v2"],
        ),
        TaskSpec(
            id="v2-opt-evo-1",
            prompt=(
                "Optimization task (v2-opt-1): produce a feasible plan and refine objective score "
                "to at least 0.90. Verify via Python execution."
            ),
            expected="0.90",
            verifier_params={"kind": "optimization", "min_score": 0.90},
            tags=["optimization", "v2"],
        ),
        TaskSpec(
            id="v2-norm-evo-1",
            prompt=(
                "V2 normalization task (norm-1): compute 12 * 12, run normalization verification "
                "producing VERIFIED_NORMALIZED:144, and return it."
            ),
            expected="VERIFIED_NORMALIZED:144",
            verifier_params={"kind": "v2_normalized", "value": "144"},
            tags=["v2_normalized", "v2"],
        ),
        TaskSpec(
            id="v2-norm-evo-2",
            prompt=(
                "V2 normalization task (norm-2): compute 25 * 4, run normalization verification "
                "producing VERIFIED_NORMALIZED:100, and return it."
            ),
            expected="VERIFIED_NORMALIZED:100",
            verifier_params={"kind": "v2_normalized", "value": "100"},
            tags=["v2_normalized", "v2"],
        ),
    ]
    heldout = [
        TaskSpec(
            id="v2-held-1",
            prompt="Held-out verification task: compute 14 * 5 in Python and verify.",
            expected="70",
            verifier_params={"kind": "verification", "expr": "14 * 5"},
            tags=["verification", "heldout"],
        ),
        TaskSpec(
            id="v2-norm-held-1",
            prompt=(
                "Held-out V2 normalization task: compute 9 * 9, run normalization verification "
                "producing VERIFIED_NORMALIZED:81, and return it."
            ),
            expected="VERIFIED_NORMALIZED:81",
            verifier_params={"kind": "v2_normalized", "value": "81"},
            tags=["v2_normalized", "heldout"],
        ),
    ]
    all_tasks = {t.id: t for t in (*evolve, *heldout)}
    return TaskDataset(
        tasks=all_tasks,
        split=SplitManifest(
            evolve=[t.id for t in evolve],
            heldout=[t.id for t in heldout],
            seed=303,
        ),
    )


def materialize_private_dataset(
    private_dir: Path,
    split_name: ToySplitName = "base",
) -> TaskDataset:
    """Write a deterministic toy dataset into ``private_dir`` (outside git repo)."""
    if split_name == "base":
        dataset = build_base_dataset()
    elif split_name == "hard":
        dataset = build_hard_dataset()
    elif split_name == "v2":
        dataset = build_policy_v2_dataset()
    else:
        raise ValueError(f"Unknown toy split name: {split_name!r}")

    private_dir.mkdir(parents=True, exist_ok=True)
    save_tasks_jsonl(private_dir / "tasks_evolve.jsonl", dataset.evolve_tasks())
    save_tasks_jsonl(private_dir / "tasks_heldout.jsonl", dataset.heldout_tasks())
    (private_dir / "split.json").write_text(
        json.dumps(dataset.split.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return dataset
