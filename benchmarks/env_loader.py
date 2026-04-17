"""Load repo root `.env` into os.environ (for benchmarks)."""

import os

_REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def load_repo_env() -> None:
    _env = os.path.join(_REPO, ".env")
    if not os.path.isfile(_env):
        return
    with open(_env, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v


load_repo_env()
