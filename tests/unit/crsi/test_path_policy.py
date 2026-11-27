"""Unit tests for PathPolicyGuard frozen-path rejection and containment (SPEC-001)."""

from __future__ import annotations

from pathlib import Path

import pytest

from archivist.crsi.config import DEFAULT_EDITABLE_PATHS
from archivist.crsi.guards.path_policy import PathPolicyGuard, PathPolicyViolation

pytestmark = [pytest.mark.crsi, pytest.mark.unit]


def _make_guard(tmp_path: Path, *, private_dir: Path | None = None) -> PathPolicyGuard:
    return PathPolicyGuard(
        repo_root=tmp_path,
        editable_patterns=DEFAULT_EDITABLE_PATHS,
        extra_frozen_patterns=("custom_frozen/**",),
        dynamic_frozen_paths=("harness_extra/scorer.py",),
        private_dir=private_dir,
    )


def test_allows_explicit_editable_paths(tmp_path: Path):
    guard = _make_guard(tmp_path)
    allowed_paths = [
        "src/archivist/crsi/harness/interlock.py",
        "src/archivist/crsi/harness/tools.py",
        "src/archivist/crsi/harness/context.py",
        "src/archivist/crsi/harness/runner.py",
        "src/archivist/crsi/harness/prompts/system.md",
        "src/archivist/crsi/harness/prompts/nested/checklist.md",
    ]
    normalized = guard.assert_all_editable(allowed_paths)
    assert normalized == allowed_paths


@pytest.mark.parametrize(
    "frozen_path",
    [
        "src/archivist/core/config.py",
        "src/archivist/storage/qdrant.py",
        "src/archivist/app/main.py",
        "src/archivist/crsi/config.py",
        "src/archivist/crsi/cli.py",
        "src/archivist/crsi/guards/path_policy.py",
        "src/archivist/crsi/sim/store.py",
        "src/archivist/crsi/llm/budget.py",
        "src/archivist/crsi/evaluator/runner.py",
        "src/archivist/crsi/loop/engine.py",
        "src/archivist/crsi/harness/contract.py",
        "src/archivist/crsi/harness/permissions.py",
        "src/archivist/crsi/harness/workspace.py",
        "src/archivist/crsi/toy/scorer.py",
        ".crsi/sim.json",
        ".git/HEAD",
        ".github/workflows/ci.yml",
        "pyproject.toml",
        "requirements.txt",
        "Dockerfile",
        "tests/unit/crsi/test_path_policy.py",
        "custom_frozen/secret.py",
        "harness_extra/scorer.py",
    ],
)
def test_rejects_frozen_paths(tmp_path: Path, frozen_path: str):
    guard = _make_guard(tmp_path)
    result = guard.check_path(frozen_path)
    assert result.allowed is False
    assert guard.is_frozen(frozen_path) is True
    assert guard.is_editable(frozen_path) is False


def test_frozen_deny_list_cannot_be_overridden_by_editable_glob(tmp_path: Path):
    guard = PathPolicyGuard(
        repo_root=tmp_path,
        editable_patterns=("src/archivist/crsi/**", "pyproject.toml"),
    )
    assert guard.is_editable("src/archivist/crsi/guards/path_policy.py") is False
    assert guard.is_editable("src/archivist/crsi/harness/permissions.py") is False
    assert guard.is_editable("pyproject.toml") is False
    assert guard.is_editable("src/archivist/crsi/harness/interlock.py") is True


def test_rejects_paths_outside_editable_allowlist(tmp_path: Path):
    guard = _make_guard(tmp_path)
    result = guard.check_path("docs/CRSI_README.md")
    assert result.allowed is False
    assert result.reason is not None
    assert "editable allowlist" in result.reason


def test_rejects_traversal_and_absolute_escape(tmp_path: Path):
    guard = _make_guard(tmp_path)
    for bad in (
        "../outside.py",
        "src/archivist/crsi/harness/../../../etc/passwd",
        "/etc/passwd",
        "",
        "src/archivist/crsi/harness/interlock.py\x00.txt",
    ):
        res = guard.check_path(bad)
        assert res.allowed is False


def test_rejects_symlink_escape_outside_repo_and_into_frozen_file(tmp_path: Path):
    outside_dir = tmp_path / "outside"
    outside_dir.mkdir()
    outside_file = outside_dir / "target.py"
    outside_file.write_text("# external", encoding="utf-8")

    repo = tmp_path / "repo"
    harness_dir = repo / "src" / "archivist" / "crsi" / "harness"
    harness_dir.mkdir(parents=True)
    frozen_file = harness_dir / "permissions.py"
    frozen_file.write_text("# frozen", encoding="utf-8")

    # Symlink escaping repo root
    escape_link = harness_dir / "tools.py"
    escape_link.symlink_to(outside_file)

    # Internal symlink from editable path pointing to frozen file
    frozen_link = harness_dir / "context.py"
    frozen_link.symlink_to(frozen_file)

    guard = PathPolicyGuard(repo_root=repo, editable_patterns=DEFAULT_EDITABLE_PATHS)
    escape_res = guard.check_path("src/archivist/crsi/harness/tools.py")
    assert escape_res.allowed is False
    assert escape_res.reason is not None
    assert "symlink" in escape_res.reason

    frozen_res = guard.check_path("src/archivist/crsi/harness/context.py")
    assert frozen_res.allowed is False
    assert frozen_res.reason is not None
    assert "frozen" in frozen_res.reason


def test_rejects_private_dir_paths(tmp_path: Path):
    repo = tmp_path / "repo"
    repo.mkdir()
    private_dir = tmp_path / "private"
    private_dir.mkdir()
    guard = PathPolicyGuard(
        repo_root=repo,
        editable_patterns=("harness/**",),
        private_dir=private_dir,
    )
    res = guard.check_path(private_dir / "tasks_heldout.jsonl")
    assert res.allowed is False


def test_assert_all_editable_raises_with_all_violations(tmp_path: Path):
    guard = _make_guard(tmp_path)
    with pytest.raises(PathPolicyViolation) as exc_info:
        guard.assert_all_editable(
            [
                "src/archivist/crsi/harness/interlock.py",
                "src/archivist/crsi/harness/permissions.py",
                "pyproject.toml",
            ]
        )
    assert len(exc_info.value.violations) == 2


def test_validate_configuration_detects_editable_scorer_and_frozen_overlap(tmp_path: Path):
    (tmp_path / "harness").mkdir()
    (tmp_path / "harness" / "scorer.py").write_text("# bad placement", encoding="utf-8")

    guard = PathPolicyGuard(
        repo_root=tmp_path,
        editable_patterns=("harness/**", "pyproject.toml"),
    )
    errors = guard.validate_configuration(
        command_argvs=[["python3", "harness/scorer.py", "{task_file}"]]
    )
    assert any("pyproject.toml" in e for e in errors)
    assert any("harness/scorer.py" in e for e in errors)
