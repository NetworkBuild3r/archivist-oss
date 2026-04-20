"""Unit tests for _openai_base — ensures trailing /v1 (and double-slash) variants
are normalised before the client appends /v1/chat/completions."""

from archivist.features.llm import _openai_base


def test_plain_root_unchanged():
    assert _openai_base("http://host:11434") == "http://host:11434"


def test_trailing_slash_stripped():
    assert _openai_base("http://host:11434/") == "http://host:11434"


def test_v1_suffix_stripped():
    assert _openai_base("http://host:11435/v1") == "http://host:11435"


def test_v1_with_trailing_slash_stripped():
    assert _openai_base("http://host:11435/v1/") == "http://host:11435"


def test_deeper_path_untouched():
    """Custom base paths (e.g. a proxy with /openai/v1) are left as-is."""
    assert _openai_base("http://proxy/openai/v1") == "http://proxy/openai"


def test_localhost_no_v1():
    assert _openai_base("http://localhost:4000") == "http://localhost:4000"
