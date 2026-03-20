"""Commit all + push to origin (https://github.com/NetworkBuild3r/archivist-oss). UNC-safe via git -c safe.directory=*.

Run from anywhere:
  python scripts/sync-public.py

Uses repo root = parent of scripts/.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", "-c", "safe.directory=*", "-C", str(REPO), *args],
        capture_output=True,
        text=True,
    )


def main() -> int:
    r = git("remote", "set-url", "origin", "https://github.com/NetworkBuild3r/archivist-oss.git")
    if r.returncode != 0:
        print(r.stderr, file=sys.stderr)
        return r.returncode
    git("add", "-A")
    st = git("status", "--porcelain")
    if st.stdout.strip():
        r = git("commit", "-m", "sync: archivist-oss")
        if r.returncode != 0:
            print(r.stderr, file=sys.stderr)
            return r.returncode
    r = git("push", "-u", "origin", "main", "--tags")
    sys.stdout.write(r.stdout)
    sys.stderr.write(r.stderr)
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
