"""Human-readable markdown journal exports — daily files alongside Qdrant.

Each day's memories are appended to ``JOURNAL_DIR/YYYY-MM-DD.md``.  Files are
plain markdown that can be browsed, grepped, or edited by humans without any
tooling.  This is a *secondary* view of data already stored in Qdrant — losing
the journal directory has no effect on retrieval.
"""

import logging
import os
import threading
from datetime import UTC, datetime

from archivist.core.config import JOURNAL_DIR, JOURNAL_ENABLED

logger = logging.getLogger("archivist.journal")

_dir_ensured = False
_write_lock = threading.Lock()


def _ensure_dir():
    global _dir_ensured
    if _dir_ensured:
        return
    try:
        os.makedirs(JOURNAL_DIR, exist_ok=True)
        _dir_ensured = True
    except OSError as e:
        logger.warning("Journal directory creation failed (%s): %s", JOURNAL_DIR, e)


def append_entry(
    *,
    memory_id: str,
    agent_id: str,
    namespace: str,
    text: str,
    memory_type: str = "general",
    importance: float = 0.5,
    extra_label: str = "",
) -> bool:
    """Append a memory entry to today's journal file. Returns True on success."""
    if not JOURNAL_ENABLED:
        return False

    _ensure_dir()

    now = datetime.now(UTC)
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S UTC")
    path = os.path.join(JOURNAL_DIR, f"{date_str}.md")

    label = extra_label or memory_type
    header = f"### [{time_str}] {agent_id} ({namespace}) — {label}"
    body = text.rstrip()
    entry = f"\n{header}\n\n{body}\n\n`id:{memory_id}` `importance:{importance}`\n\n---\n"

    is_new = not os.path.exists(path)

    try:
        with _write_lock, open(path, "a", encoding="utf-8") as f:
            if is_new:
                f.write(f"# Archivist Journal — {date_str}\n\n---\n")
            f.write(entry)
        return True
    except OSError as e:
        logger.warning("Journal write failed (%s): %s", path, e)
        return False
