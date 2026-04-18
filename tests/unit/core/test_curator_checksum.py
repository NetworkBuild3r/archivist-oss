"""Unit tests for curator checksum logic."""

import pytest

pytestmark = [pytest.mark.unit]


class TestCuratorChecksum:
    def test_file_checksum_deterministic(self):
        from curator import _file_checksum

        h1 = _file_checksum("hello world")
        h2 = _file_checksum("hello world")
        assert h1 == h2

    def test_file_checksum_differs(self):
        from curator import _file_checksum

        h1 = _file_checksum("hello world")
        h2 = _file_checksum("hello world!")
        assert h1 != h2
