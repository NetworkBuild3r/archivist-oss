"""AST layer-isolation guard for CRSI (GR-LAYER-001 / INIT-025/SPEC-012).

Enforces that ``src/archivist/crsi/`` remains strictly decoupled from the
Archivist core runtime and vector client:
  1. Zero imports from ``archivist.*`` outside ``archivist.crsi`` (including
     bare ``import archivist``, relative escapes like ``from ..storage import ...``,
     and dynamic ``importlib.import_module`` / ``__import__`` string literals).
  2. Zero imports of ``qdrant_client`` or ``qdrant_client.*``.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
_CRSI_ROOT = _SRC_ROOT / "archivist" / "crsi"


def _is_forbidden_target(target: str) -> bool:
    """Return True if *target* violates GR-LAYER-001 for a module in archivist.crsi."""
    if not target:
        return False
    if target == "qdrant_client" or target.startswith("qdrant_client."):
        return True
    if target == "archivist":
        return True
    return target.startswith("archivist.") and not (
        target == "archivist.crsi" or target.startswith("archivist.crsi.")
    )


def _package_parts_for_module(module_rel_path: Path) -> list[str]:
    """Return dotted package parts containing *module_rel_path* (relative to ``src/``)."""
    parts = list(module_rel_path.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        return parts[:-1]
    return parts[:-1]


def _resolve_relative_import(package_parts: list[str], level: int, module: str | None) -> str:
    """Resolve an ``ast.ImportFrom`` with ``level > 0`` to its dotted module path."""
    climb = level - 1
    if climb > len(package_parts):
        # Escapes above src/ root — cannot resolve inside archivist.crsi.
        return module or ""
    base = package_parts[: len(package_parts) - climb] if climb > 0 else list(package_parts)
    if module:
        return ".".join([*base, module])
    return ".".join(base)


def find_crsi_layer_violations(source: str, module_rel_path: Path) -> list[str]:
    """Parse *source* and return human-readable GR-LAYER-001 violations."""
    tree = ast.parse(source, filename=str(module_rel_path))
    pkg_parts = _package_parts_for_module(module_rel_path)
    violations: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_forbidden_target(alias.name):
                    violations.append(
                        f"{module_rel_path}:{node.lineno}: forbidden import '{alias.name}'"
                    )

        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                resolved_mod = _resolve_relative_import(pkg_parts, node.level, node.module)
            else:
                resolved_mod = node.module or ""

            if _is_forbidden_target(resolved_mod):
                # Allow `from archivist import crsi` only if every imported alias is `crsi`.
                if resolved_mod == "archivist" and all(a.name == "crsi" for a in node.names):
                    continue
                violations.append(
                    f"{module_rel_path}:{node.lineno}: forbidden import-from '{resolved_mod}'"
                )
            else:
                for alias in node.names:
                    candidate = f"{resolved_mod}.{alias.name}" if resolved_mod else alias.name
                    if _is_forbidden_target(candidate):
                        violations.append(
                            f"{module_rel_path}:{node.lineno}: forbidden import '{candidate}'"
                        )

        elif isinstance(node, ast.Call):
            func = node.func
            is_dynamic_import = (isinstance(func, ast.Name) and func.id == "__import__") or (
                isinstance(func, ast.Attribute)
                and func.attr == "import_module"
                and isinstance(func.value, ast.Name)
                and func.value.id == "importlib"
            )
            if is_dynamic_import and node.args:
                arg0 = node.args[0]
                if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                    if _is_forbidden_target(arg0.value):
                        violations.append(
                            f"{module_rel_path}:{node.lineno}: forbidden dynamic import '{arg0.value}'"
                        )

    return violations


def scan_crsi_tree(crsi_root: Path = _CRSI_ROOT, src_root: Path = _SRC_ROOT) -> list[str]:
    """Scan every Python file under *crsi_root* for GR-LAYER-001 violations."""
    violations: list[str] = []
    for py_file in sorted(crsi_root.rglob("*.py")):
        rel = py_file.relative_to(src_root)
        source = py_file.read_text(encoding="utf-8")
        violations.extend(find_crsi_layer_violations(source, rel))
    return violations


def test_crsi_has_zero_core_or_qdrant_imports() -> None:
    """GR-LAYER-001: src/archivist/crsi/ has zero core archivist.* or qdrant_client imports."""
    assert _CRSI_ROOT.is_dir(), f"Expected CRSI directory at {_CRSI_ROOT}"
    py_files = list(_CRSI_ROOT.rglob("*.py"))
    assert len(py_files) > 0, "Expected Python modules under src/archivist/crsi/"

    violations = scan_crsi_tree(_CRSI_ROOT, _SRC_ROOT)
    assert violations == [], (
        "GR-LAYER-001 violation(s) in src/archivist/crsi/ "
        "(CRSI must not import core archivist.* or qdrant_client):\n"
        + "\n".join(f"  - {v}" for v in violations)
    )


def test_guard_catches_direct_and_nested_core_imports() -> None:
    """Verify the AST guard flags direct, from-, bare, and function-scoped core imports."""
    sample = """
import archivist
import archivist.core.config
from archivist.storage import graph
from archivist import retrieval

def nested():
    from archivist.app.main import app
    return app
"""
    rel = Path("archivist/crsi/loop/sample.py")
    violations = find_crsi_layer_violations(sample, rel)
    assert len(violations) == 5
    assert any("archivist.core.config" in v for v in violations)
    assert any("archivist.storage" in v for v in violations)
    assert any("archivist.app.main" in v for v in violations)


def test_guard_catches_relative_escape_and_allows_internal_crsi_imports() -> None:
    """Relative imports escaping archivist.crsi fail; intra-crsi relative imports pass."""
    bad_sample = """
from ...storage import graph
from ... import core
"""
    rel = Path("archivist/crsi/loop/engine.py")
    violations = find_crsi_layer_violations(bad_sample, rel)
    assert len(violations) == 2

    good_sample = """
from ..guards import secrets
from .state import LoopState
from archivist.crsi.config import CrsiConfig
import archivist.crsi.sim.models
from archivist import crsi
"""
    assert find_crsi_layer_violations(good_sample, rel) == []


def test_guard_catches_qdrant_client_and_dynamic_imports() -> None:
    """Verify qdrant_client and importlib/__import__ core bypasses are caught."""
    sample = """
import importlib
import qdrant_client
from qdrant_client.models import PointStruct

def load():
    importlib.import_module("archivist.storage.graph")
    __import__("qdrant_client")
    importlib.import_module("archivist.crsi.sim")
"""
    rel = Path("archivist/crsi/cli.py")
    violations = find_crsi_layer_violations(sample, rel)
    assert len(violations) == 4
    assert all("archivist.crsi.sim" not in v for v in violations)
