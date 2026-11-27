"""Proposer agent P: builds full-SIM prompts under HeldOutIsolationGuard and parses proposals."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from archivist.crsi.config import MemoryMode
from archivist.crsi.evaluator.types import EvaluationReport
from archivist.crsi.guards.heldout_firewall import HeldOutIsolationGuard
from archivist.crsi.guards.secrets import SecretRedactor
from archivist.crsi.llm.base import LLMClient, LLMRequest, Message
from archivist.crsi.loop.history import RunHistory
from archivist.crsi.loop.manifest import CandidateProposal
from archivist.crsi.sim.models import SimDocument
from archivist.crsi.sim.render import render_sim


class InvalidProposal(BaseModel):
    """Record of a malformed proposal rejected before worktree application."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    error: str


class ProposalBatch(BaseModel):
    """Parsed output of one Proposer turn."""

    model_config = ConfigDict(extra="forbid")

    proposals: list[CandidateProposal] = Field(default_factory=list)
    invalid: list[InvalidProposal] = Field(default_factory=list)


def _strip_json_fences(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if len(lines) >= 2 and lines[-1].strip() == "```":
            return "\n".join(lines[1:-1]).strip()
    return text


class ProposerAgent:
    """Generates up to ``k`` candidate harness edits using full SIM and evolve feedback."""

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        heldout_guard: HeldOutIsolationGuard,
        memory_mode: MemoryMode = "full",
        redactor: SecretRedactor | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.heldout_guard = heldout_guard
        self.memory_mode: MemoryMode = memory_mode
        self.redactor = redactor if redactor is not None else SecretRedactor()

    def build_request(
        self,
        *,
        round_num: int,
        k_candidates: int,
        editable_sources: Mapping[str, str],
        baseline_report: EvaluationReport,
        sim_doc: SimDocument,
        history: RunHistory,
        domain: str,
        policy_model: str,
    ) -> LLMRequest:
        """Assemble the Proposer prompt with untruncated SIM, ``F_t``, ``D_t``, and ``H_t``."""
        self.heldout_guard.assert_evolve_only_report(baseline_report)

        sim_block = render_sim(sim_doc, mode=self.memory_mode, include_usage_instructions=True)
        feedback_block = baseline_report.format_evolve_feedback()
        history_block = history.format_for_prompt()

        harness_sections: list[str] = ["## Current Editable Harness (H_t)"]
        for rel_path in sorted(editable_sources.keys()):
            harness_sections.append(f"### File: {rel_path}\n```\n{editable_sources[rel_path]}\n```")
        harness_block = "\n\n".join(harness_sections)

        system_prompt = (
            "You are the CRSI Proposer (P). Keep policy weights frozen; edit ONLY files in the "
            "editable harness allowlist. Return strict JSON: "
            '{"candidates": [{"candidate_id": "...", "manifest": {"problem_id": "P1"|null, '
            '"new_problem_name": "..."|null, "what": "...", "files": [...], "rationale": "...", '
            '"predicted": ["<evolve_task_id>"], "interlock_removal_sim_evidence": null}, '
            '"file_edits": {"<path>": "<full_file_content>"}, "delete_files": []}]}. '
            f"Produce up to {k_candidates} candidate(s)."
        )

        user_parts: list[str] = [
            f"Round: {round_num} | Domain: {domain} | Policy model: {policy_model}",
            f"Memory mode: {self.memory_mode}",
            feedback_block,
            history_block,
        ]
        if sim_block:
            user_parts.append(sim_block)
        user_parts.append(harness_block)

        user_prompt = self.redactor.redact_text("\n\n".join(user_parts))
        return LLMRequest(
            role="proposer",
            messages=[
                Message(role="system", content=system_prompt),
                Message(role="user", content=user_prompt),
            ],
            json_mode=True,
            metadata={
                "round": round_num,
                "k_candidates": k_candidates,
                "memory_mode": self.memory_mode,
                "domain": domain,
                "policy_model": policy_model,
            },
        )

    async def propose(
        self,
        *,
        round_num: int,
        k_candidates: int,
        editable_sources: Mapping[str, str],
        baseline_report: EvaluationReport,
        sim_doc: SimDocument,
        history: RunHistory,
        domain: str,
        policy_model: str,
        context_paths: list[Path] | None = None,
        feedback_dir: Path | None = None,
    ) -> ProposalBatch:
        """Invoke Proposer LLM through ``HeldOutIsolationGuard`` and parse candidate batch."""
        req = self.build_request(
            round_num=round_num,
            k_candidates=k_candidates,
            editable_sources=editable_sources,
            baseline_report=baseline_report,
            sim_doc=sim_doc,
            history=history,
            domain=domain,
            policy_model=policy_model,
        )
        self.heldout_guard.assert_request_clean(
            req,
            context_paths=context_paths or [],
            feedback_dir=feedback_dir,
        )
        resp = await self.llm_client.complete(req)
        return self.parse_response(resp.content, round_num=round_num, k_candidates=k_candidates)

    @staticmethod
    def parse_response(
        raw_content: str,
        *,
        round_num: int,
        k_candidates: int,
    ) -> ProposalBatch:
        """Parse Proposer JSON payload into validated proposals + invalid-manifest diagnostics."""
        cleaned = _strip_json_fences(raw_content)
        try:
            payload: Any = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            return ProposalBatch(
                invalid=[
                    InvalidProposal(
                        candidate_id=f"r{round_num}-parse-error",
                        error=f"Invalid proposer JSON: {exc}",
                    )
                ]
            )

        raw_list = payload.get("candidates") if isinstance(payload, dict) else None
        if not isinstance(raw_list, list):
            return ProposalBatch(
                invalid=[
                    InvalidProposal(
                        candidate_id=f"r{round_num}-schema-error",
                        error="Proposer JSON missing 'candidates' list.",
                    )
                ]
            )

        proposals: list[CandidateProposal] = []
        invalid: list[InvalidProposal] = []
        for idx, item in enumerate(raw_list[:k_candidates], start=1):
            fallback_id = f"r{round_num}-c{idx}"
            if not isinstance(item, dict):
                invalid.append(
                    InvalidProposal(
                        candidate_id=fallback_id,
                        error="Candidate entry must be a JSON object.",
                    )
                )
                continue
            cand_id = str(item.get("candidate_id") or fallback_id)
            normalized = dict(item)
            normalized["candidate_id"] = cand_id
            try:
                proposals.append(CandidateProposal.model_validate(normalized))
            except ValidationError as exc:
                invalid.append(
                    InvalidProposal(
                        candidate_id=cand_id,
                        error=f"Invalid ChangeManifest/CandidateProposal: {exc}",
                    )
                )
        return ProposalBatch(proposals=proposals, invalid=invalid)
