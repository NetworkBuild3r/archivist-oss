"""Unit tests for src/tokenizer.py."""

import pytest

pytestmark = [pytest.mark.unit]


class TestTokenizer:
    def test_count_tokens_nonempty(self):
        from tokenizer import count_tokens

        n = count_tokens("Hello, world!")
        assert n > 0

    def test_count_tokens_empty(self):
        from tokenizer import count_tokens

        n = count_tokens("")
        assert n >= 0

    def test_count_message_tokens(self):
        from tokenizer import count_message_tokens

        msgs = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "What is 2+2?"},
        ]
        n = count_message_tokens(msgs)
        assert n > 10

    def test_count_message_tokens_empty(self):
        from tokenizer import count_message_tokens

        n = count_message_tokens([])
        assert n == 0

    def test_fallback_approximation(self):
        from tokenizer import count_tokens

        text = "a" * 400
        n = count_tokens(text)
        assert n >= 50
