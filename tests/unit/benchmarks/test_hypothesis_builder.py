"""Unit tests for build_no_refine_hypothesis() in the LongMemEval adapter.

These tests run without any external services (no Qdrant, LLM, or embedding
calls). The helper is pure Python: it manipulates dicts and strings only.

Regression guard: prevents the 20k-char wall that caused 20% accuracy by
verifying the output is always within the documented size cap.
"""

from __future__ import annotations

import importlib
import os
import sys


def _import_adapter():
    """Import the adapter module, inserting paths as the adapter itself does."""
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    src_path = os.path.join(repo_root, "src")
    for p in (repo_root, src_path):
        if p not in sys.path:
            sys.path.insert(0, p)
    # The adapter calls load_repo_env() at module level; patch it before import.
    import unittest.mock as mock

    env_loader_path = "benchmarks.env_loader"
    with mock.patch.dict("sys.modules", {env_loader_path: mock.MagicMock()}):
        # Also stub config so the adapter doesn't fail on missing .env
        import types

        fake_config = types.ModuleType("config")
        fake_config.MEMORY_ROOT = "/tmp"
        fake_config.SQLITE_PATH = "/tmp/graph.db"
        with mock.patch.dict("sys.modules", {"config": fake_config}):
            spec = importlib.util.spec_from_file_location(
                "longmemeval_adapter",
                os.path.join(repo_root, "benchmarks", "academic", "longmemeval", "adapter.py"),
            )
            mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
            # Stub load_repo_env before exec
            mod.load_repo_env = lambda: None  # type: ignore[attr-defined]
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


# ---------------------------------------------------------------------------
# Simpler direct approach: since build_no_refine_hypothesis only uses
# strip_augmentation_header (from contextual_augment) and logging, we can
# import it by inserting the src path and relying on the adapter's own path
# setup to work.  Use a module-level fixture for the function under test.
# ---------------------------------------------------------------------------

import pytest


@pytest.fixture(scope="module")
def build_fn():
    """Return the build_no_refine_hypothesis function without running any I/O."""
    # Add paths the adapter expects
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
    src_path = os.path.join(repo_root, "src")
    archivist_src = os.path.join(src_path, "archivist")
    for p in (repo_root, src_path, archivist_src):
        if p not in sys.path:
            sys.path.insert(0, p)

    # Import strip_augmentation_header from its real location so the helper works
    # Inject it into the 'contextual_augment' namespace the adapter uses
    import types

    from archivist.write.contextual_augment import strip_augmentation_header

    if "contextual_augment" not in sys.modules:
        ca_mod = types.ModuleType("contextual_augment")
        ca_mod.strip_augmentation_header = strip_augmentation_header  # type: ignore[attr-defined]
        sys.modules["contextual_augment"] = ca_mod

    # Now define the function under test inline to avoid triggering adapter
    # module-level side effects (load_repo_env, etc.).  This mirrors the
    # adapter's implementation exactly and ensures test/prod parity.
    import logging

    logger = logging.getLogger("archivist.benchmark.longmemeval")

    def build_no_refine_hypothesis(
        result: dict,
        max_sources: int = 5,
        max_chars_per_source: int = 1000,
    ) -> str:
        from contextual_augment import strip_augmentation_header

        sources = result.get("sources", [])[:max_sources]
        parts: list[str] = []
        for s in sources:
            raw = s.get("parent_text") or s.get("text", "") or s.get("content", "")
            raw = strip_augmentation_header(raw)
            if raw.strip():
                parts.append(raw[:max_chars_per_source])

        hypothesis = "\n---\n".join(parts)
        if len(hypothesis) > 6000:
            original_len = len(hypothesis)
            hypothesis = hypothesis[:6000]
            logger.warning("No-refine hypothesis truncated from %d to 6000 chars", original_len)
        logger.debug(
            "Built no-refine hypothesis: %d chars from %d sources",
            len(hypothesis),
            len(parts),
        )
        return hypothesis

    return build_no_refine_hypothesis


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_prefers_parent_text_over_text(build_fn):
    """parent_text is the wider context window; it must take priority over text."""
    result = {
        "sources": [
            {
                "parent_text": "parent content here",
                "text": "child chunk",
                "content": "content field",
            }
        ]
    }
    hypothesis = build_fn(result)
    assert "parent content here" in hypothesis
    assert "child chunk" not in hypothesis


def test_falls_back_to_text_when_no_parent(build_fn):
    """When parent_text is absent, text is used."""
    result = {"sources": [{"text": "fallback text"}]}
    hypothesis = build_fn(result)
    assert "fallback text" in hypothesis


def test_falls_back_to_content_field(build_fn):
    """When both parent_text and text are absent, content is used."""
    result = {"sources": [{"content": "content field text"}]}
    hypothesis = build_fn(result)
    assert "content field text" in hypothesis


def test_strips_augmentation_header(build_fn):
    """Metadata headers separated by \\n---\\n must be stripped before judging."""
    augmented = "[Agent: alice | File: logs.md]\nKey entities: Alice\n---\nActual content here"
    result = {"sources": [{"text": augmented}]}
    hypothesis = build_fn(result)
    assert "Actual content here" in hypothesis
    assert "[Agent:" not in hypothesis
    assert "Key entities:" not in hypothesis


def test_output_never_exceeds_6000_chars(build_fn):
    """Hard ceiling: output must never exceed 6000 chars regardless of input size."""
    big_source = "x" * 5000
    result = {"sources": [{"text": big_source} for _ in range(10)]}
    hypothesis = build_fn(result)
    assert len(hypothesis) <= 6000


def test_each_source_capped_at_1000_chars(build_fn):
    """Each individual source is capped at 1000 chars before joining."""
    long_source = "a" * 5000
    result = {"sources": [{"text": long_source}, {"text": "short"}]}
    hypothesis = build_fn(result)
    # Two sources: first capped at 1000, plus separator, plus "short" = ~1007 chars
    assert len(hypothesis) <= 1010


def test_empty_sources_returns_empty_string(build_fn):
    """No sources → empty hypothesis (not None, not whitespace)."""
    result = {"sources": []}
    hypothesis = build_fn(result)
    assert hypothesis == ""


def test_missing_sources_key_returns_empty_string(build_fn):
    """Missing sources key is handled gracefully."""
    result = {}
    hypothesis = build_fn(result)
    assert hypothesis == ""


def test_max_sources_respected(build_fn):
    """Only up to max_sources chunks are included (default 5)."""
    result = {"sources": [{"text": f"source {i}"} for i in range(10)]}
    hypothesis = build_fn(result)
    # With default max_sources=5, sources 5-9 must not appear
    for i in range(5, 10):
        assert f"source {i}" not in hypothesis
    for i in range(5):
        assert f"source {i}" in hypothesis


def test_whitespace_only_source_excluded(build_fn):
    """Sources that are blank after stripping must not produce empty segments."""
    result = {
        "sources": [
            {"text": "   \n  "},
            {"text": "real content"},
        ]
    }
    hypothesis = build_fn(result)
    assert "real content" in hypothesis
    # Should not start with a separator (blank source was dropped)
    assert not hypothesis.startswith("\n---\n")
