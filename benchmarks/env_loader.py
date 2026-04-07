"""Load repo-root `.env` into `os.environ` before importing `src.config`.

Matches `benchmarks/pipeline/evaluate.py` so academic adapters see LLM_URL, EMBED_URL,
QDRANT_URL, etc. when run via `python -m benchmarks.academic.*.adapter` or shell scripts.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_repo_env() -> None:
    root = Path(__file__).resolve().parent.parent
    env_path = root / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip())
