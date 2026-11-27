"""Package entrypoint for ``python -m archivist.crsi``."""

from __future__ import annotations

import sys

from archivist.crsi.cli import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
