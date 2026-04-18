"""Code-quality gate tests — run the same checks CI's pre-commit step runs.

Gap that prompted this file
---------------------------
The QA suite previously only ran pytest but never validated that all source
files were properly formatted.  CI uses ``pre-commit run --all-files`` which
runs ``ruff-format`` and ``ruff check`` before any tests execute.  Six QA
files were committed without being formatted; CI caught the divergence but
local ``pytest tests/qa/ -q`` passed because pytest doesn't check formatting.

These tests close that gap by running the same ruff invocations that CI runs
as synchronous subprocess checks.  Any file that needs reformatting or has a
lint violation will cause a hard test failure with the exact diff / error list
printed, matching what a developer would see from ``ruff format --check``.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Repo root is two levels up from tests/qa/
_REPO_ROOT = Path(__file__).parent.parent.parent


def _run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(_REPO_ROOT),
    )


# ---------------------------------------------------------------------------
# ruff-format gate
# ---------------------------------------------------------------------------


def test_ruff_format_check_src():
    """src/ contains no files that need reformatting (mirrors CI ruff-format hook)."""
    result = _run([sys.executable, "-m", "ruff", "format", "--check", "src/"])
    assert result.returncode == 0, (
        "ruff-format would reformat files in src/.\n"
        "Run:  ruff format src/\n\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_ruff_format_check_tests():
    """tests/ contains no files that need reformatting (mirrors CI ruff-format hook)."""
    result = _run([sys.executable, "-m", "ruff", "format", "--check", "tests/"])
    assert result.returncode == 0, (
        "ruff-format would reformat files in tests/.\n"
        "Run:  ruff format tests/\n\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# ruff-lint gate
# ---------------------------------------------------------------------------


def test_ruff_lint_src():
    """src/ has no ruff lint violations (mirrors CI ruff check step)."""
    result = _run([sys.executable, "-m", "ruff", "check", "src/"])
    assert result.returncode == 0, (
        "ruff check found violations in src/.\n"
        "Run:  ruff check --fix src/\n\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_ruff_lint_tests():
    """tests/ has no ruff lint violations (mirrors CI ruff check step)."""
    result = _run([sys.executable, "-m", "ruff", "check", "tests/"])
    assert result.returncode == 0, (
        "ruff check found violations in tests/.\n"
        "Run:  ruff check --fix tests/\n\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


# ---------------------------------------------------------------------------
# trailing-whitespace gate
# ---------------------------------------------------------------------------


def test_no_trailing_whitespace_in_docs():
    """docs/*.md files contain no trailing whitespace (mirrors CI trailing-whitespace hook).

    CI's ``trailing-whitespace`` pre-commit hook runs on all tracked files.
    docs/BENCHMARKS.md was committed with trailing spaces; this test ensures
    that class of error is caught locally before push.
    """
    docs_dir = _REPO_ROOT / "docs"
    offenders: list[str] = []
    for md_file in sorted(docs_dir.glob("**/*.md")):
        for lineno, line in enumerate(md_file.read_text(errors="replace").splitlines(), 1):
            if line != line.rstrip():
                offenders.append(f"  {md_file.relative_to(_REPO_ROOT)}:{lineno}: {line!r}")
    assert not offenders, (
        "Trailing whitespace found in docs/ — run:\n"
        "  sed -i 's/[[:space:]]*$//' <file>\n\n"
        "Offending lines:\n" + "\n".join(offenders)
    )


def test_no_trailing_whitespace_in_tests():
    """tests/**/*.py files contain no trailing whitespace."""
    tests_dir = _REPO_ROOT / "tests"
    offenders: list[str] = []
    for py_file in sorted(tests_dir.rglob("*.py")):
        for lineno, line in enumerate(py_file.read_text(errors="replace").splitlines(), 1):
            if line != line.rstrip():
                offenders.append(f"  {py_file.relative_to(_REPO_ROOT)}:{lineno}: {line!r}")
    assert not offenders, (
        "Trailing whitespace found in tests/ — run:\n"
        "  find tests/ -name '*.py' | xargs sed -i 's/[[:space:]]*$//'\n\n"
        "Offending lines:\n" + "\n".join(offenders)
    )
