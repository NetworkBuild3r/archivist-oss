"""Deterministic zero-cost toy domain, frozen scorer, seed harness template, and scripted agents."""

from __future__ import annotations

from archivist.crsi.toy.scorer import ToyScorer
from archivist.crsi.toy.scripted_agents import (
    load_toy_harness_from_worktree,
    make_scripted_policy_v1,
    make_scripted_policy_v2,
    make_scripted_proposer,
    make_scripted_selector,
    make_toy_worktree_evaluator,
    seed_toy_repo,
)
from archivist.crsi.toy.tasks import (
    build_base_dataset,
    build_hard_dataset,
    build_policy_v2_dataset,
    materialize_private_dataset,
)

__all__ = [
    "ToyScorer",
    "build_base_dataset",
    "build_hard_dataset",
    "build_policy_v2_dataset",
    "load_toy_harness_from_worktree",
    "make_scripted_policy_v1",
    "make_scripted_policy_v2",
    "make_scripted_proposer",
    "make_scripted_selector",
    "make_toy_worktree_evaluator",
    "materialize_private_dataset",
    "seed_toy_repo",
]
