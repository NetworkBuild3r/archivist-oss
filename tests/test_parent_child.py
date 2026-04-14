"""Tests for hierarchical parent-child chunking."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from chunking import chunk_text, chunk_text_hierarchical, chunk_text_semantic


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


# ── Semantic chunking tests ───────────────────────────────────────────────────

def test_semantic_short_document_fast_path():
    """Documents shorter than size return a single chunk, unchanged."""
    text = "## Summary\n\nThis is a short document with one section."
    chunks = chunk_text_semantic(text, size=2000)
    assert len(chunks) == 1
    assert chunks[0] == text.strip()


def test_semantic_empty_returns_empty():
    """Empty and whitespace-only text returns an empty list."""
    assert chunk_text_semantic("") == []
    assert chunk_text_semantic("   \n\n  ") == []


def test_semantic_heading_based_split():
    """Each top-level heading becomes its own chunk when document exceeds size."""
    section_body = "Content " * 60  # ~480 chars per section
    text = (
        f"## Introduction\n\n{section_body}\n\n"
        f"## Architecture\n\n{section_body}\n\n"
        f"## Deployment\n\n{section_body}"
    )
    chunks = chunk_text_semantic(text, size=600)
    # Each section ~520 chars — should stay together; no merging across headings
    assert len(chunks) >= 3, f"Expected ≥3 chunks, got {len(chunks)}: {[c[:60] for c in chunks]}"
    # Every chunk must contain its heading
    headings_found = sum(1 for c in chunks if c.startswith("## "))
    assert headings_found >= 3, "Each section chunk should start with its heading"


def test_semantic_never_merges_across_headings():
    """Content from different sections must not appear in the same chunk."""
    intro_body = "Alpha content. " * 10
    arch_body = "Beta content. " * 10
    text = f"## Alpha\n\n{intro_body}\n\n## Beta\n\n{arch_body}"
    chunks = chunk_text_semantic(text, size=300)
    for chunk in chunks:
        has_alpha = "Alpha content" in chunk
        has_beta = "Beta content" in chunk
        assert not (has_alpha and has_beta), (
            "A single chunk must not contain content from both ## Alpha and ## Beta sections"
        )


def test_semantic_code_block_not_split():
    """A fenced code block is never broken across chunks."""
    preamble = "Setup instructions follow.\n\n" * 5  # push over size limit
    code = "```\nimport os\nos.environ['FOO'] = 'bar'\nprint('done')\n```"
    text = f"## Setup\n\n{preamble}{code}"
    chunks = chunk_text_semantic(text, size=300)
    # Find the chunk that contains the code block
    code_chunks = [c for c in chunks if "```" in c]
    for c in code_chunks:
        # Opening and closing fences must both appear
        assert c.count("```") >= 2, f"Code block split across chunk boundary: {c}"


def test_semantic_heading_prepended_to_sub_chunks():
    """When a section is split, the heading is prepended to each sub-chunk."""
    # Build a section that's definitely over 400 chars
    body = "Detail paragraph. " * 15  # ~270 chars × 2 paragraphs
    text = f"## Big Section\n\n{body}\n\n{body}"
    chunks = chunk_text_semantic(text, size=400)
    if len(chunks) > 1:
        # All sub-chunks from the oversized section carry the heading
        for c in chunks:
            assert "Big Section" in c, (
                f"Sub-chunk missing heading context: {c[:80]!r}"
            )


def test_semantic_no_headings_falls_back_to_paragraph_split():
    """Documents without markdown headings fall back to paragraph splitting."""
    paragraphs = ["Paragraph number {}. ".format(i) * 10 for i in range(8)]
    text = "\n\n".join(paragraphs)
    # With size=300 this should produce multiple chunks
    chunks = chunk_text_semantic(text, size=300)
    assert len(chunks) >= 2, "Headingless long text should still be split"
    assert all(len(c) > 0 for c in chunks)


def test_semantic_all_content_preserved():
    """No content is silently dropped — all unique words survive."""
    sections = {
        "Alpha": "unique_alpha_word",
        "Beta": "unique_beta_word",
        "Gamma": "unique_gamma_word",
    }
    body = "filler text " * 60
    parts = [f"## {name}\n\n{body} {word}" for name, word in sections.items()]
    text = "\n\n".join(parts)
    chunks = chunk_text_semantic(text, size=800)
    combined = " ".join(chunks)
    for word in sections.values():
        assert word in combined, f"Word '{word}' was dropped during semantic chunking"


def test_semantic_hierarchical_uses_semantic_strategy():
    """chunk_text_hierarchical with strategy='semantic' calls chunk_text_semantic."""
    section_body = "Detail line. " * 40  # ~520 chars
    text = (
        f"## Section One\n\n{section_body}\n\n"
        f"## Section Two\n\n{section_body}\n\n"
        f"## Section Three\n\n{section_body}"
    )
    result_semantic = chunk_text_hierarchical(
        text, "test.md", parent_size=700, child_size=300, strategy="semantic"
    )
    result_fixed = chunk_text_hierarchical(
        text, "test.md", parent_size=700, child_size=300, strategy="fixed"
    )
    parents_semantic = [c for c in result_semantic if c["is_parent"]]
    parents_fixed = [c for c in result_fixed if c["is_parent"]]
    # Semantic should produce heading-aligned parents; fixed may split differently
    semantic_headings = sum(1 for p in parents_semantic if p["content"].startswith("## "))
    assert semantic_headings >= 2, (
        f"Semantic strategy should produce heading-aligned parents, got: "
        f"{[p['content'][:60] for p in parents_semantic]}"
    )


def test_semantic_short_document_same_as_fixed():
    """Short documents produce identical output regardless of strategy."""
    text = "## Note\n\nThis is a short note that fits in one chunk."
    sem = chunk_text_hierarchical(text, "note.md", parent_size=2000, strategy="semantic")
    fix = chunk_text_hierarchical(text, "note.md", parent_size=2000, strategy="fixed")
    sem_contents = [c["content"] for c in sem if c["is_parent"]]
    fix_contents = [c["content"] for c in fix if c["is_parent"]]
    assert sem_contents == fix_contents, (
        "Short documents must produce identical parent content regardless of strategy"
    )


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
    test_semantic_short_document_fast_path()
    test_semantic_empty_returns_empty()
    test_semantic_heading_based_split()
    test_semantic_never_merges_across_headings()
    test_semantic_code_block_not_split()
    test_semantic_heading_prepended_to_sub_chunks()
    test_semantic_no_headings_falls_back_to_paragraph_split()
    test_semantic_all_content_preserved()
    test_semantic_hierarchical_uses_semantic_strategy()
    test_semantic_short_document_same_as_fixed()
    print("All parent-child chunking tests passed ✓")
