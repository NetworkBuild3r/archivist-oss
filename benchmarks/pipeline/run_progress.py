"""Milestone progress logging and atomic JSON checkpoints for long benchmark runs."""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("archivist.benchmark.progress")


def format_eta(seconds: float) -> str:
    """Human-readable ETA (e.g. ``8m12s``, ``2h15m``)."""
    if seconds < 0 or seconds != seconds:  # nan
        return "?"
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    if s < 3600:
        m, sec = s // 60, s % 60
        return f"{m}m{sec}s"
    h, rem = s // 3600, s % 3600
    m = rem // 60
    return f"{h}h{m}m"


def default_checkpoint_path(output_json: str) -> str:
    """``full_medium.json`` → ``full_medium.run_state.json`` in the same directory."""
    p = Path(output_json)
    return str(p.parent / f"{p.stem}.run_state.json")


def write_checkpoint(path: str, payload: dict[str, Any]) -> None:
    """Atomically write JSON (temp file + replace) so readers never see partial files."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    payload = dict(payload)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _should_log_milestone(i: int, total: int, pct_step: int) -> bool:
    """Log when crossing 10%, 20%, ... or on the final query (1-based ``i``)."""
    if total <= 0:
        return False
    if i == total:
        return True
    pct_step = max(1, pct_step)
    curr = (100 * i) // total
    prev = (100 * (i - 1)) // total if i > 1 else -1
    for boundary in range(pct_step, 100, pct_step):
        if prev < boundary <= curr:
            return True
    return False


class _FallbackStderrBar:
    """Single-line ``\\r`` bar when tqdm is unavailable."""

    def __init__(self, total: int, desc: str) -> None:
        self.total = total
        self.desc = desc
        self._start = time.monotonic()

    def update(
        self,
        i: int,
        *,
        rolling_recall: float,
        rolling_mrr: float,
    ) -> None:
        width = 28
        if self.total > 0:
            filled = min(width, int(width * i / self.total))
            bar = "[" + "=" * max(0, filled - 1) + ">" + " " * (width - filled) + "]"
            pct = int(100.0 * i / self.total)
        else:
            bar = "[" + "?" * width + "]"
            pct = 0
        elapsed = time.monotonic() - self._start
        eta_sec = (elapsed / i) * (self.total - i) if i > 0 and i < self.total else 0.0
        line = (
            f"\r{self.desc} {bar} {i}/{self.total} {pct}% "
            f"R={rolling_recall:.3f} MRR={rolling_mrr:.3f} ETA~{format_eta(eta_sec)}"
        )
        sys.stderr.write(line[:200].ljust(min(120, len(line) + 20)))
        sys.stderr.flush()

    def close(self) -> None:
        sys.stderr.write("\n")
        sys.stderr.flush()


class ProgressTracker:
    """Progress bar (default) or milestone logs + optional per-query checkpoint."""

    def __init__(
        self,
        *,
        total: int,
        phase: str,
        memory_scale: str | None,
        pct_step: int,
        checkpoint_path: str | None,
        use_progress_bar: bool = True,
    ) -> None:
        self.total = total
        self.phase = phase
        self.memory_scale = memory_scale
        self.pct_step = max(1, pct_step)
        self.checkpoint_path = checkpoint_path
        self._start = time.monotonic()
        self._use_progress_bar = use_progress_bar and total > 0
        self._pbar: Any = None
        self._fallback: _FallbackStderrBar | None = None

        if self._use_progress_bar:
            desc = phase
            if memory_scale:
                desc = f"{phase} [{memory_scale}]"
            try:
                from tqdm import tqdm

                self._pbar = tqdm(
                    total=total,
                    desc=desc[:48],
                    unit="q",
                    file=sys.stderr,
                    dynamic_ncols=True,
                    mininterval=0.15,
                    maxinterval=2.0,
                    smoothing=0.05,
                )
            except ImportError:
                self._pbar = None
                self._fallback = _FallbackStderrBar(total, desc[:48])

    def step(
        self,
        i: int,
        *,
        results: list[dict[str, Any]],
        rolling_recall: float,
        rolling_mrr: float,
    ) -> None:
        """1-based query index ``i`` after appending the i-th result."""
        if self._pbar is not None:
            self._pbar.set_postfix_str(
                f"R={rolling_recall:.3f} MRR={rolling_mrr:.3f}",
                refresh=False,
            )
            self._pbar.update(1)
        elif self._fallback is not None:
            self._fallback.update(
                i,
                rolling_recall=rolling_recall,
                rolling_mrr=rolling_mrr,
            )
        elif not self._use_progress_bar:
            should_log = _should_log_milestone(i, self.total, self.pct_step)
            if should_log:
                elapsed = time.monotonic() - self._start
                eta_sec = (elapsed / i) * (self.total - i) if i > 0 and i < self.total else 0.0
                logger.info(
                    "PROGRESS %s %d/%d (%d%%) recall_so_far=%.4f mrr_so_far=%.4f ETA~%s",
                    self.phase,
                    i,
                    self.total,
                    int(100.0 * i / self.total) if self.total else 0,
                    rolling_recall,
                    rolling_mrr,
                    format_eta(eta_sec),
                )

        if self.checkpoint_path:
            payload: dict[str, Any] = {
                "phase": self.phase,
                "memory_scale": self.memory_scale,
                "completed": i,
                "total": self.total,
                "rolling_recall_mean": round(rolling_recall, 4),
                "results": results,
                "partial_summary": {
                    "recall_so_far": round(rolling_recall, 4),
                    "mrr_so_far": round(rolling_mrr, 4),
                },
            }
            write_checkpoint(self.checkpoint_path, payload)

    def close(self) -> None:
        """Finish the bar and release the terminal line."""
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None
        if self._fallback is not None:
            self._fallback.close()
            self._fallback = None
