"""Shared text and path utility helpers.

Centralises small, frequently-duplicated functions so they are maintained in
one place and imported everywhere they are needed.
"""

import hashlib
import re
from pathlib import Path


def strip_fences(raw: str) -> str:
    """Strip markdown code fences from an LLM response string."""
    raw = raw.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
    return raw.strip()


def extract_agent_id_from_path(rel_path: str) -> str:
    """Return the agent_id encoded in a relative memory file path.

    Looks for ``agents/<agent_id>/`` or ``memories/<agent_id>/`` path
    segments (whichever comes first).  Returns an empty string when no
    recognised segment is found.
    """
    parts = Path(rel_path).parts
    for sentinel in ("agents", "memories"):
        if sentinel in parts:
            idx = list(parts).index(sentinel)
            if idx + 1 < len(parts):
                return parts[idx + 1]
    return ""


def compute_memory_checksum(text: str, agent_id: str, namespace: str) -> str:
    """Return a SHA-256 hex digest that uniquely identifies a memory's content."""
    return hashlib.sha256(f"{text}:{agent_id}:{namespace}".encode()).hexdigest()
