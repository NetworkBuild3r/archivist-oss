"""Benchmark: tokenizer — tiktoken vs chars//4 fallback speed and accuracy."""

import sys
import os

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))


def _sample_text(length: int = 4000) -> str:
    base = (
        "The Archivist memory service provides hybrid search combining "
        "vector similarity with BM25 keyword matching. It maintains a "
        "temporal knowledge graph for entity-relationship tracking across "
        "multiple AI agents operating in fleet configuration. "
    )
    return (base * ((length // len(base)) + 1))[:length]


def test_count_tokens_short(benchmark):
    """Measure count_tokens on a short string (~100 chars)."""
    from tokenizer import count_tokens

    text = _sample_text(100)
    result = benchmark(count_tokens, text)
    assert result > 0


def test_count_tokens_medium(benchmark):
    """Measure count_tokens on a medium string (~4KB)."""
    from tokenizer import count_tokens

    text = _sample_text(4000)
    result = benchmark(count_tokens, text)
    assert result > 0


def test_count_tokens_large(benchmark):
    """Measure count_tokens on a large string (~50KB)."""
    from tokenizer import count_tokens

    text = _sample_text(50000)
    result = benchmark(count_tokens, text)
    assert result > 0


def test_count_message_tokens(benchmark):
    """Measure count_message_tokens over a realistic message list."""
    from tokenizer import count_message_tokens

    messages = [
        {"role": "system", "content": _sample_text(500)},
        {"role": "user", "content": _sample_text(2000)},
        {"role": "assistant", "content": _sample_text(3000)},
        {"role": "user", "content": _sample_text(1000)},
    ]
    result = benchmark(count_message_tokens, messages)
    assert result > 0


def test_fallback_vs_tiktoken_accuracy():
    """Verify chars//4 fallback is within 30% of tiktoken for English text."""
    text = _sample_text(4000)
    fallback_count = max(1, len(text) // 4)

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        real_count = len(enc.encode(text))
        ratio = fallback_count / real_count
        assert 0.7 < ratio < 1.3, (
            f"Fallback {fallback_count} vs tiktoken {real_count} "
            f"(ratio {ratio:.2f}) exceeds 30% tolerance"
        )
    except ImportError:
        pytest.skip("tiktoken not installed, accuracy comparison skipped")
