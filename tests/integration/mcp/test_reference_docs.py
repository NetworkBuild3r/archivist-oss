"""Integration tests for the archivist_get_reference_docs handler."""

import json

import pytest

pytestmark = [pytest.mark.integration, pytest.mark.mcp]


class TestGetReferenceDocs:
    """Tests for archivist_get_reference_docs handler."""

    async def test_full_return(self, tmp_path, monkeypatch):
        """Handler returns full file content when no section is given."""
        import handlers.tools_docs as tools_docs

        doc = tmp_path / "CURSOR_SKILL.md"
        doc.write_text("## Search\n\ncontent A\n\n## Storage\n\ncontent B\n")
        monkeypatch.setattr(tools_docs, "_SKILL_DOC", doc)

        result = await tools_docs._handle_get_reference_docs({})
        assert len(result) == 1
        assert "content A" in result[0].text
        assert "content B" in result[0].text

    async def test_section_filter_match(self, tmp_path, monkeypatch):
        """Handler returns only the matching section block."""
        import handlers.tools_docs as tools_docs

        doc = tmp_path / "CURSOR_SKILL.md"
        doc.write_text("## Search\n\ncontent A\n\n## Storage\n\ncontent B\n")
        monkeypatch.setattr(tools_docs, "_SKILL_DOC", doc)

        result = await tools_docs._handle_get_reference_docs({"section": "storage"})
        assert len(result) == 1
        assert "content B" in result[0].text
        assert "content A" not in result[0].text

    async def test_section_filter_no_match(self, tmp_path, monkeypatch):
        """Handler returns error JSON listing available sections when no match."""
        import handlers.tools_docs as tools_docs

        doc = tmp_path / "CURSOR_SKILL.md"
        doc.write_text("## Search\n\ncontent A\n\n## Storage\n\ncontent B\n")
        monkeypatch.setattr(tools_docs, "_SKILL_DOC", doc)

        result = await tools_docs._handle_get_reference_docs({"section": "nonexistent"})
        assert len(result) == 1
        payload = json.loads(result[0].text)
        assert payload["error"] == "section_not_found"
        assert "Search" in payload["available_sections"]
        assert "Storage" in payload["available_sections"]

    async def test_missing_doc_returns_error(self, tmp_path, monkeypatch):
        """Handler returns error JSON when the doc file does not exist."""
        import handlers.tools_docs as tools_docs

        monkeypatch.setattr(tools_docs, "_SKILL_DOC", tmp_path / "missing.md")
        monkeypatch.setattr(tools_docs, "_FALLBACK_DOC", tmp_path / "also_missing.md")

        result = await tools_docs._handle_get_reference_docs({})
        assert len(result) == 1
        payload = json.loads(result[0].text)
        assert payload["error"] == "reference_docs_not_found"
