"""Milestone progress logging and atomic JSON checkpoints for long benchmark runs."""
from __future__ import annotations

import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("archivist.benchmark.progress")


def default_checkpoint_path(output_json: str) -> str:
    """``full_medium.json`` → ``full_medium.run_state.json`` in the same directory."""
    p = Path(output_json)
    return str(p.parent / f"{p.stem}.run_state.json")


class ProgressTracker:
    """Tracks benchmark query progress, logs milestones, and writes checkpoints."""

    def __init__(
        self,
        total: int,
        phase: str = "",
        memory_scale: str | None = None,
        pct_step: int = 10,
        checkpoint_path: str | None = None,
        use_progress_bar: bool = True,
    ) -> None:
        self.total = total
        self.phase = phase
        self.memory_scale = memory_scale
        self.pct_step = max(1, pct_step)
        self.checkpoint_path = checkpoint_path
        self._last_pct_logged = 0
        self._bar = None

        if use_progress_bar:
            try:
                from tqdm import tqdm
                self._bar = tqdm(total=total, desc=phase or "queries", unit="q", dynamic_ncols=True)
            except ImportError:
                pass

    def step(
        self,
        completed: int,
        *,
        results: list[dict] | None = None,
        rolling_recall: float = 0.0,
        rolling_mrr: float = 0.0,
    ) -> None:
        if self._bar is not None:
            self._bar.update(1)
            self._bar.set_postfix(recall=f"{rolling_recall:.3f}", mrr=f"{rolling_mrr:.3f}")

        pct = int(completed / self.total * 100) if self.total else 100
        if pct >= self._last_pct_logged + self.pct_step:
            self._last_pct_logged = pct
            logger.info(
                "[%s] %d/%d (%.0f%%) recall=%.3f mrr=%.3f",
                self.phase, completed, self.total, pct, rolling_recall, rolling_mrr,
            )

        if self.checkpoint_path and results:
            write_checkpoint(self.checkpoint_path, {
                "phase": self.phase,
                "memory_scale": self.memory_scale,
                "completed": completed,
                "total": self.total,
                "rolling_recall_mean": round(rolling_recall, 4),
                "results": results,
            })

    def close(self) -> None:
        if self._bar is not None:
            self._bar.close()


def write_checkpoint(path: str, data: dict[str, Any]) -> None:
    """Atomically write JSON (temp file + replace) so readers never see partial files."""
    d = dict(data)
    d["updated_at"] = datetime.now(timezone.utc).isoformat()
    parent = os.path.dirname(path) or "."
    os.makedirs(parent, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".checkpoint_", suffix=".tmp", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(d, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
