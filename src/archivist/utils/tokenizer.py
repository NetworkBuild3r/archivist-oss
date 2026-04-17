"""Token counting with tiktoken (OpenAI models) and a chars//4 fallback.

Used by context_manager and rlm_retriever for token budget enforcement.
"""

import logging

logger = logging.getLogger("archivist.tokenizer")

_encoder = None
_encoder_loaded = False


def _load_encoder():
    """Lazy-load tiktoken encoder. Falls back gracefully if unavailable."""
    global _encoder, _encoder_loaded
    if _encoder_loaded:
        return _encoder
    _encoder_loaded = True
    try:
        import tiktoken

        _encoder = tiktoken.get_encoding("cl100k_base")
        logger.info("Tokenizer: using tiktoken cl100k_base")
    except Exception:
        logger.info("Tokenizer: tiktoken not available, using chars//4 fallback")
        _encoder = None
    return _encoder


def count_tokens(text: str) -> int:
    """Count tokens in text. Uses tiktoken if available, otherwise chars // 4."""
    enc = _load_encoder()
    if enc is not None:
        return len(enc.encode(text))
    return max(1, len(text) // 4)


def count_message_tokens(messages: list[dict]) -> int:
    """Count total tokens across a list of chat messages.

    Each message contributes its content tokens plus ~4 tokens of overhead
    for role/formatting (matching OpenAI's documented formula).
    """
    if not messages:
        return 0
    total = 0
    for msg in messages:
        total += 4  # role + formatting overhead
        content = msg.get("content", "")
        if isinstance(content, str):
            total += count_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    total += count_tokens(part.get("text", ""))
    total += 2  # reply priming
    return total
