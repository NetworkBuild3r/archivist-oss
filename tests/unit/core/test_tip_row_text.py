"""INIT-007/SPEC-002 — tip_row_text / tip_rows_to_strings mapping."""

from __future__ import annotations

from archivist.core.trajectory import tip_row_text, tip_rows_to_strings


class TestTipRowText:
    def test_prefers_tip_text(self):
        assert tip_row_text({"tip_text": "from db", "content": "legacy"}) == "from db"

    def test_falls_back_to_content_then_tip(self):
        assert tip_row_text({"content": "c"}) == "c"
        assert tip_row_text({"tip": "t"}) == "t"

    def test_empty_and_none(self):
        assert tip_row_text(None) == ""
        assert tip_row_text({}) == ""
        assert tip_row_text({"tip_text": "  "}) == ""


class TestTipRowsToStrings:
    def test_filters_blanks_preserves_order(self):
        rows = [
            {"tip_text": "first"},
            {"tip_text": ""},
            {"content": "second"},
            {},
        ]
        assert tip_rows_to_strings(rows) == ["first", "second"]

    def test_none_rows(self):
        assert tip_rows_to_strings(None) == []
