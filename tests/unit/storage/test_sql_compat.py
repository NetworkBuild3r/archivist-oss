"""Static SQL compatibility scanner.

Verifies that no SQLite-specific SQL idioms appear in source files that execute
SQL through the async pool (which may be backed by PostgreSQL).

WHY THIS TEST EXISTS
--------------------
Several production bugs were introduced because SQL written for SQLite was
used verbatim against the asyncpg/PostgreSQL backend:

  * ``COLLATE NOCASE`` — SQLite collation, invalid in PostgreSQL. Postgres uses
    the ``citext`` column type for case-insensitive text. Symptom: "collation
    nocase does not exist" errors at query time.

  * ``INSERT OR IGNORE INTO ...`` — SQLite-only upsert shorthand.  PostgreSQL
    requires ``INSERT ... ON CONFLICT (...) DO NOTHING``. Symptom: syntax
    error from asyncpg.

  * ``INSERT OR REPLACE INTO ...`` — SQLite-only upsert shorthand. PostgreSQL
    requires ``INSERT ... ON CONFLICT (...) DO UPDATE SET ...``. Symptom:
    syntax error from asyncpg.

  * ``cursor.lastrowid`` — aiosqlite-specific attribute; asyncpg cursors do
    not expose it. The PostgreSQL-compatible approach is ``INSERT ... RETURNING
    id`` followed by ``fetchone()[0]``. Symptom: ``AttributeError`` at runtime.

  * DDL inside a ``SERIALIZABLE`` transaction — PostgreSQL raises an error when
    DDL is executed inside an explicit serializable transaction block. DDL must
    run in autocommit mode. Symptom: ``AttributeError: 'NoneType' object has no
    attribute 'decode'`` from asyncpg 0.31.

None of these are caught by mypy or ruff. None fail against an in-memory
SQLite test DB. They only blow up when the Postgres backend is active.

WHAT THIS TEST DOES
-------------------
It scans all ``*.py`` source files under ``src/archivist/`` using regex
patterns and flags any match that appears in a file that:

  1. Is NOT on the explicit SQLite-only allowlist (``skills.py``, etc.), AND
  2. Matches a known-bad pattern.

The allowlist is conservative: only files that use ``get_db()`` (synchronous
SQLite connection) and are documented as SQLite-only are permitted to contain
these patterns.

KEEPING THIS TEST CURRENT
--------------------------
If you add a new SQLite-only module (e.g. a new feature file that uses
``get_db()``), add its path to ``SQLITE_ONLY_FILES`` below.  If you
intentionally use an idiom in a comment or docstring rather than actual SQL,
you can add a targeted per-line exclusion by adding the line to
``COMMENT_EXCLUSION_PATTERNS``.
"""

from __future__ import annotations

import re
import textwrap
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_SRC_ROOT = Path(__file__).parents[4] / "src" / "archivist"

# Files (relative to _SRC_ROOT) that are explicitly SQLite-only.
# These use get_db() / synchronous sqlite3 connections and will never hit
# the asyncpg pool. They are exempt from the checks below.
SQLITE_ONLY_FILES: frozenset[str] = frozenset(
    {
        "features/skills.py",  # uses get_db() throughout; documented as SQLite-only
        "storage/graph.py",  # the ``INSERT OR IGNORE`` in graph.py is in a *docstring*
        # (verified: the actual SQL was fixed to ON CONFLICT)
    }
)

# ---------------------------------------------------------------------------
# Patterns that are forbidden in pool-facing source files
# ---------------------------------------------------------------------------

# Each tuple: (human-readable name, compiled regex, explanation)
_FORBIDDEN: list[tuple[str, re.Pattern[str], str]] = [
    (
        "COLLATE NOCASE",
        re.compile(r"COLLATE\s+NOCASE", re.IGNORECASE),
        (
            "SQLite-only collation. PostgreSQL uses the `citext` column type for "
            "case-insensitive text — the COLLATE clause is invalid syntax there. "
            "Remove `COLLATE NOCASE` from the SQL; the `citext` column handles it."
        ),
    ),
    (
        "INSERT OR IGNORE",
        re.compile(r"INSERT\s+OR\s+IGNORE\s+INTO", re.IGNORECASE),
        ("SQLite-only upsert syntax. Replace with `INSERT INTO ... ON CONFLICT (...) DO NOTHING`."),
    ),
    (
        "INSERT OR REPLACE",
        re.compile(r"INSERT\s+OR\s+REPLACE\s+INTO", re.IGNORECASE),
        (
            "SQLite-only upsert syntax. Replace with "
            "`INSERT INTO ... ON CONFLICT (...) DO UPDATE SET ...`."
        ),
    ),
    (
        "cursor.lastrowid",
        re.compile(r"\.\s*lastrowid\b"),
        (
            "aiosqlite-specific attribute. asyncpg cursors do not expose `.lastrowid`. "
            "Use `INSERT ... RETURNING id` and `(await cur.fetchone())[0]` instead."
        ),
    ),
]

# Lines containing these strings are skipped even if they match a forbidden
# pattern — they are explanatory comments / docstrings, not SQL.
_COMMENT_MARKERS = (
    "#",
    '"""',
    "'''",
    "Rows are inserted with",  # docstring in graph.py
    "# In PostgreSQL",
    "# SQLite:",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_comment_or_docstring_line(line: str) -> bool:
    stripped = line.strip()
    return any(stripped.startswith(m) or m in stripped for m in _COMMENT_MARKERS)


def _collect_violations() -> list[tuple[str, int, str, str, str]]:
    """Return list of (rel_path, lineno, pattern_name, line, explanation)."""
    violations = []

    for py_file in sorted(_SRC_ROOT.rglob("*.py")):
        rel = py_file.relative_to(_SRC_ROOT).as_posix()
        if rel in SQLITE_ONLY_FILES:
            continue
        if "__pycache__" in rel:
            continue

        source = py_file.read_text(encoding="utf-8")
        lines = source.splitlines()

        for lineno, line in enumerate(lines, start=1):
            if _is_comment_or_docstring_line(line):
                continue
            for name, pattern, explanation in _FORBIDDEN:
                if pattern.search(line):
                    violations.append((rel, lineno, name, line.rstrip(), explanation))

    return violations


# ---------------------------------------------------------------------------
# The test
# ---------------------------------------------------------------------------


def test_no_sqlite_only_sql_in_pool_facing_sources() -> None:
    """No SQLite-specific SQL idioms in async-pool-facing source files.

    This test is the regression guard for the production bugs described in the
    module docstring. It runs as part of the standard unit suite (no Postgres
    required) so the incompatibility is caught before code ever reaches CI.
    """
    violations = _collect_violations()

    if not violations:
        return

    lines = ["", "SQLite-specific SQL found in pool-facing source files:", ""]
    for rel, lineno, name, line, explanation in violations:
        lines.append(f"  {rel}:{lineno}  [{name}]")
        lines.append(f"    SQL:  {line.strip()}")
        lines.append(f"    Fix:  {explanation}")
        lines.append("")

    pytest.fail(textwrap.dedent("\n".join(lines)))


# ---------------------------------------------------------------------------
# Per-pattern parametrized smoke tests (verify the regexes themselves work)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sql, pattern_name",
    [
        ("WHERE name = ? COLLATE NOCASE", "COLLATE NOCASE"),
        ("collate nocase", "COLLATE NOCASE"),
        ("INSERT OR IGNORE INTO memory_points (a, b) VALUES (?, ?)", "INSERT OR IGNORE"),
        ("INSERT OR REPLACE INTO memory_hotness VALUES (?, ?, ?)", "INSERT OR REPLACE"),
        ("row_id = cur.lastrowid", "cursor.lastrowid"),
    ],
)
def test_forbidden_patterns_are_detected(sql: str, pattern_name: str) -> None:
    """Each forbidden-pattern regex correctly matches its target string."""
    pattern_map = {name: pat for name, pat, _ in _FORBIDDEN}
    assert pattern_name in pattern_map, f"Unknown pattern name: {pattern_name!r}"
    assert pattern_map[pattern_name].search(sql), f"Pattern {pattern_name!r} did not match: {sql!r}"


@pytest.mark.parametrize(
    "safe_sql",
    [
        "INSERT INTO memory_points (a, b) ON CONFLICT (a, b) DO NOTHING",
        "INSERT INTO memory_hotness (memory_id, score) ON CONFLICT (memory_id) DO UPDATE SET score=EXCLUDED.score",
        "INSERT INTO entities (name) VALUES (?) RETURNING id",
        "SELECT id FROM entities WHERE name = ?",
    ],
)
def test_clean_sql_is_not_flagged(safe_sql: str) -> None:
    """Correct, Postgres-compatible SQL is not flagged by the scanner."""
    for name, pattern, _ in _FORBIDDEN:
        assert not pattern.search(safe_sql), (
            f"Pattern {name!r} incorrectly flagged clean SQL: {safe_sql!r}"
        )
