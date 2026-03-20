"""Tests for hierarchical parent-child chunking."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunking import chunk_text, chunk_text_hierarchical


def test_chunk_text_basic():
    """Basic flat chunking produces non-empty results."""
    text = "Hello world.\n\nThis is a test.\n\nAnother paragraph here."
    chunks = chunk_text(text, size=100, overlap=20)
    assert len(chunks) >= 1
    assert all(len(c) > 0 for c in chunks)


def test_chunk_text_empty():
    """Empty text returns empty list."""
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_single_paragraph():
    """Text smaller than chunk size returns single chunk."""
    text = "Short text."
    chunks = chunk_text(text, size=1000, overlap=100)
    assert len(chunks) == 1
    assert chunks[0] == "Short text."


def test_hierarchical_produces_parents_and_children():
    """Hierarchical chunking creates both parent and child chunks."""
    paragraphs = [f"Paragraph number {i} with some content to fill space." for i in range(20)]
    text = "\n\n".join(paragraphs)

    result = chunk_text_hierarchical(
        text,
        "agents/alice/notes.md",
        parent_size=300,
        child_size=100,
        parent_overlap=50,
        child_overlap=20,
    )

    assert len(result) > 0

    parents = [c for c in result if c["is_parent"]]
    children = [c for c in result if not c["is_parent"]]

    assert len(parents) >= 1, "Should have at least one parent chunk"
    assert len(children) >= 1, "Should have at least one child chunk"


def test_hierarchical_parent_ids():
    """Every child references an existing parent."""
    paragraphs = [f"Content block {i} with enough text to split." for i in range(15)]
    text = "\n\n".join(paragraphs)

    result = chunk_text_hierarchical(
        text,
        "memories/bob/daily.md",
        parent_size=200,
        child_size=80,
        parent_overlap=40,
        child_overlap=10,
    )

    parent_ids = {c["id"] for c in result if c["is_parent"]}
    for child in result:
        if not child["is_parent"]:
            assert child["parent_id"] is not None, "Child must have a parent_id"
            assert child["parent_id"] in parent_ids, f"Child parent_id '{child['parent_id']}' not in parent set"


def test_hierarchical_parent_has_no_parent():
    """Parent chunks have parent_id=None."""
    text = "A decent chunk of text.\n\n" * 10
    result = chunk_text_hierarchical(text, "x.md", parent_size=200, child_size=80)
    for chunk in result:
        if chunk["is_parent"]:
            assert chunk["parent_id"] is None


def test_hierarchical_ids_are_unique():
    """All chunk IDs are unique."""
    text = "Test content.\n\n" * 20
    result = chunk_text_hierarchical(text, "unique.md", parent_size=200, child_size=80)
    ids = [c["id"] for c in result]
    assert len(ids) == len(set(ids)), "Duplicate IDs found"


def test_hierarchical_small_text():
    """Small text produces at least one parent."""
    text = "Just a small note."
    result = chunk_text_hierarchical(text, "small.md", parent_size=2000, child_size=500)
    assert len(result) >= 1
    parents = [c for c in result if c["is_parent"]]
    assert len(parents) == 1


def test_parent_ids_differ_by_filepath():
    """Same body text under different paths must not share parent IDs."""
    text = "Paragraph one.\n\nParagraph two.\n\n" * 5
    a = chunk_text_hierarchical(text, "/a/x.md", parent_size=200, child_size=80)
    b = chunk_text_hierarchical(text, "/b/y.md", parent_size=200, child_size=80)
    ids_a = {c["id"] for c in a}
    ids_b = {c["id"] for c in b}
    assert ids_a.isdisjoint(ids_b), "Parent/child IDs should be unique per file path"


if __name__ == "__main__":
    test_chunk_text_basic()
    test_chunk_text_empty()
    test_chunk_text_single_paragraph()
    test_hierarchical_produces_parents_and_children()
    test_hierarchical_parent_ids()
    test_hierarchical_parent_has_no_parent()
    test_hierarchical_ids_are_unique()
    test_hierarchical_small_text()
    test_parent_ids_differ_by_filepath()
    print("All parent-child chunking tests passed ✓")
