"""Selector agent S: pre-execution screening, post-evaluation pick, and tolerance floor."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from archivist.crsi.config import MemoryMode
from archivist.crsi.evaluator.types import EvaluationReport
from archivist.crsi.guards.heldout_firewall import HeldOutIsolationGuard
from archivist.crsi.guards.secrets import SecretRedactor
from archivist.crsi.llm.base import LLMClient, LLMRequest, Message
from archivist.crsi.loop.history import RunHistory
from archivist.crsi.loop.manifest import CandidateProposal
from archivist.crsi.sim.models import SimDocument
from archivist.crsi.sim.render import render_sim


class ScreenDecision(BaseModel):
    """Pre-execution screening result from Selector ``S_screen``."""

    model_config = ConfigDict(extra="forbid")

    kept_ids: list[str] = Field(default_factory=list)
    dropped: dict[str, str] = Field(default_factory=dict)


class PickDecision(BaseModel):
    """Post-evaluation selection result with deterministic tolerance enforcement."""

    model_config = ConfigDict(extra="forbid")

    selected_id: str | None = None
    reason: str = ""
    rejected_by_tolerance: bool = False


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


class SelectorAgent:
    """Selector ``S`` performing pre-eval screening and regression-guarded final pick."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        heldout_guard: HeldOutIsolationGuard,
        regression_tolerance: float = 0.0,
        memory_mode: MemoryMode = "full",
        redactor: SecretRedactor | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.heldout_guard = heldout_guard
        self.regression_tolerance = max(0.0, regression_tolerance)
        self.memory_mode: MemoryMode = memory_mode
        self.redactor = redactor if redactor is not None else SecretRedactor()

    async def screen(
        self,
        *,
        round_num: int,
        baseline_report: EvaluationReport,
        candidates: Sequence[CandidateProposal],
        diffs: Mapping[str, str],
        sim_doc: SimDocument,
        history: RunHistory,
        context_paths: list[Path] | None = None,
        feedback_dir: Path | None = None,
    ) -> ScreenDecision:
        """Pre-evaluation screen dropping duplicates or unsafe/overfit proposals."""
        self.heldout_guard.assert_evolve_only_report(baseline_report)
        if not candidates:
            return ScreenDecision()

        cand_blocks: list[str] = []
        for cand in candidates:
            diff_text = diffs.get(cand.candidate_id, "")
            cand_blocks.append(
                f"### Candidate {cand.candidate_id}\n"
                f"Manifest: {json.dumps(cand.manifest.model_dump(mode='json'), sort_keys=True)}\n"
                f"Diff:\n```diff\n{diff_text}\n```"
            )

        sim_block = render_sim(sim_doc, mode=self.memory_mode, include_usage_instructions=True)
        parts: list[str] = [
            f"Round {round_num} — Selector Pre-Execution Screen",
            baseline_report.format_evolve_feedback(),
            history.format_for_prompt(),
        ]
        if sim_block:
            parts.append(sim_block)
        parts.append("\n\n".join(cand_blocks))

        req = LLMRequest(
            role="selector",
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are the CRSI Selector (screen stage). Return strict JSON: "
                        '{"keep": ["<candidate_id>"], "drop": {"<candidate_id>": "<reason>"}}.'
                    ),
                ),
                Message(role="user", content=self.redactor.redact_text("\n\n".join(parts))),
            ],
            json_mode=True,
            metadata={"stage": "screen", "round": round_num},
        )
        self.heldout_guard.assert_request_clean(
            req,
            context_paths=context_paths or [],
            feedback_dir=feedback_dir,
        )
        resp = await self.llm_client.complete(req)
        return self._parse_screen(resp.content, [c.candidate_id for c in candidates])

    @staticmethod
    def _parse_screen(raw: str, valid_ids: Sequence[str]) -> ScreenDecision:
        valid_set = set(valid_ids)
        try:
            payload: Any = json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError:
            return ScreenDecision(kept_ids=list(valid_ids), dropped={})

        if not isinstance(payload, dict):
            return ScreenDecision(kept_ids=list(valid_ids), dropped={})

        raw_keep = payload.get("keep")
        raw_drop = payload.get("drop")

        dropped: dict[str, str] = {}
        if isinstance(raw_drop, dict):
            for cid, reason in raw_drop.items():
                if str(cid) in valid_set:
                    dropped[str(cid)] = str(reason)

        if isinstance(raw_keep, list):
            kept = [
                str(cid) for cid in raw_keep if str(cid) in valid_set and str(cid) not in dropped
            ]
            for cid in valid_ids:
                if cid not in kept and cid not in dropped:
                    dropped[cid] = "not retained by selector screen"
            return ScreenDecision(kept_ids=kept, dropped=dropped)

        kept = [cid for cid in valid_ids if cid not in dropped]
        return ScreenDecision(kept_ids=kept, dropped=dropped)

    async def pick(
        self,
        *,
        round_num: int,
        baseline_report: EvaluationReport,
        candidates: Sequence[CandidateProposal],
        candidate_reports: Mapping[str, EvaluationReport],
        diffs: Mapping[str, str],
        sim_doc: SimDocument,
        history: RunHistory,
        context_paths: list[Path] | None = None,
        feedback_dir: Path | None = None,
    ) -> PickDecision:
        """Select one evaluated candidate or NONE, enforcing hard regression tolerance floor."""
        self.heldout_guard.assert_evolve_only_report(baseline_report)
        for rep in candidate_reports.values():
            self.heldout_guard.assert_evolve_only_report(rep)

        if not candidates:
            return PickDecision(selected_id=None, reason="No candidates evaluated.")

        eval_blocks: list[str] = []
        for cand in candidates:
            cand_rep = candidate_reports.get(cand.candidate_id)
            if cand_rep is None:
                continue
            eval_blocks.append(
                f"### Candidate {cand.candidate_id} "
                f"(score={cand_rep.mean_score:.4f}, passed={cand_rep.passed}/{cand_rep.total})\n"
                f"Manifest: {json.dumps(cand.manifest.model_dump(mode='json'), sort_keys=True)}\n"
                f"{cand_rep.format_evolve_feedback()}\n"
                f"Diff:\n```diff\n{diffs.get(cand.candidate_id, '')}\n```"
            )

        sim_block = render_sim(sim_doc, mode=self.memory_mode, include_usage_instructions=True)
        parts: list[str] = [
            f"Round {round_num} — Selector Pick (baseline_score={baseline_report.mean_score:.4f}, "
            f"regression_tolerance={self.regression_tolerance:.4f})",
            baseline_report.format_evolve_feedback(),
            history.format_for_prompt(),
        ]
        if sim_block:
            parts.append(sim_block)
        parts.append("\n\n".join(eval_blocks))

        req = LLMRequest(
            role="selector",
            messages=[
                Message(
                    role="system",
                    content=(
                        "You are the CRSI Selector (pick stage). Choose ONE candidate ID or 'NONE'. "
                        'Return strict JSON: {"selected": "<candidate_id>|NONE", "reason": "..."}.'
                    ),
                ),
                Message(role="user", content=self.redactor.redact_text("\n\n".join(parts))),
            ],
            json_mode=True,
            metadata={"stage": "pick", "round": round_num},
        )
        self.heldout_guard.assert_request_clean(
            req,
            context_paths=context_paths or [],
            feedback_dir=feedback_dir,
        )
        resp = await self.llm_client.complete(req)
        raw_pick = self._parse_pick(resp.content, set(candidate_reports.keys()))

        if raw_pick.selected_id is None:
            return raw_pick

        # Code-enforced hard regression tolerance floor (cannot be overridden by selector LLM).
        chosen_report = candidate_reports[raw_pick.selected_id]
        floor = baseline_report.mean_score - self.regression_tolerance
        if chosen_report.mean_score < floor - 1e-9:
            return PickDecision(
                selected_id=None,
                reason=(
                    f"Candidate {raw_pick.selected_id} rejected by hard regression tolerance: "
                    f"score {chosen_report.mean_score:.4f} < floor {floor:.4f} "
                    f"(baseline={baseline_report.mean_score:.4f}, "
                    f"tolerance={self.regression_tolerance:.4f})."
                ),
                rejected_by_tolerance=True,
            )
        return raw_pick

    @staticmethod
    def _parse_pick(raw: str, valid_ids: set[str]) -> PickDecision:
        try:
            payload: Any = json.loads(_strip_json_fences(raw))
        except json.JSONDecodeError as exc:
            return PickDecision(selected_id=None, reason=f"Invalid selector pick JSON: {exc}")

        if not isinstance(payload, dict):
            return PickDecision(selected_id=None, reason="Selector pick JSON must be an object.")

        selected = payload.get("selected")
        reason = str(payload.get("reason", ""))
        if selected is None or str(selected).strip().upper() == "NONE":
            return PickDecision(selected_id=None, reason=reason or "Selector chose NONE.")

        sel_id = str(selected).strip()
        if sel_id not in valid_ids:
            return PickDecision(
                selected_id=None,
                reason=f"Selector returned unknown candidate id {sel_id!r}.",
            )
        return PickDecision(selected_id=sel_id, reason=reason)
