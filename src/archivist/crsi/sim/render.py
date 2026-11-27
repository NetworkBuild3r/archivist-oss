"""Untruncated SIM prompt rendering (full / problem_only / none) and CLI summary."""

from __future__ import annotations

import logging
from typing import Literal

from archivist.crsi.sim.models import ProblemEntry, SimDocument

logger = logging.getLogger(__name__)

MemoryMode = Literal["full", "problem_only", "none"]

# Yu et al. (arXiv:2611.18856), Appendix D.2 8-point SIM usage protocol.
SIM_USAGE_INSTRUCTIONS = """\
Self-Improvement Memory (SIM) Usage Instructions:
1. Treat SIM entries as prior optimization experience, not unquestionable ground truth.
2. Check each attempt's recorded model and domain provenance before relying on it.
3. Reuse rationales and verified guards (such as ExecutionGatingInterlock) that solved real failures.
4. Reassess inherited guards carefully after a model swap or task-distribution shift using fresh baseline evidence.
5. Prefer extending or revising working guards beside existing mechanisms rather than blindly deleting them.
6. Every proposal MUST cite an existing SIM problem ID or open a new root-cause diagnosis with a pre-eval prediction.
7. Do not repeat previously failed or regressed directions unless new diagnostic evidence explains why the change will behave differently.
8. Compare predicted outcomes against observed outcomes (fixed, missed, regressed) to separate real fixes from noise.\
"""


def _render_problem_full(entry: ProblemEntry) -> list[str]:
    lines: list[str] = [
        f"### {entry.id}: {entry.name} (opened_round={entry.opened_round})",
    ]
    if not entry.attempts:
        lines.append("  (no attempts recorded yet)")
        return lines

    for idx, att in enumerate(entry.attempts, start=1):
        obs = att.observed
        files_str = ", ".join(att.files) if att.files else "(none)"
        predicted_str = ", ".join(att.predicted) if att.predicted else "(none)"
        fixed_str = ", ".join(obs.fixed) if obs.fixed else "(none)"
        missed_str = ", ".join(obs.missed) if obs.missed else "(none)"
        regressed_str = ", ".join(obs.regressed) if obs.regressed else "(none)"
        lines.extend(
            [
                (
                    f"  - Attempt {idx} [round={att.round}, sample={att.sample}, "
                    f"accepted={att.accepted}, backfilled={att.backfilled}, "
                    f"model={att.model}, domain={att.domain}]"
                ),
                f"    what: {att.what}",
                f"    files: {files_str}",
                f"    rationale: {att.rationale}",
                f"    predicted: {predicted_str}",
                (
                    f"    observed: score={obs.score:.4f} ({obs.passed}/{obs.total}) | "
                    f"fixed=[{fixed_str}] | missed=[{missed_str}] | regressed=[{regressed_str}]"
                ),
            ]
        )
    return lines


def render_sim(
    doc: SimDocument,
    mode: MemoryMode = "full",
    *,
    include_usage_instructions: bool = True,
) -> str:
    """Render SIM for Proposer/Selector prompts without truncation.

    Modes:
    - ``full`` (default): complete SIM including rationales, predicted-vs-observed
      outcomes, modified files, provenance, ``carried_from``, and ``aliases``.
      Strictly never truncates entries or attempts.
    - ``problem_only`` (ablation only): renders ONLY ``{id, name, opened_round}``,
      stripping attempts, rationales, predicted/observed outcomes, and provenance.
      Logs a warning noting Yu et al.'s finding that problem-only memory can
      underperform no memory.
    - ``none``: returns an empty string (Cold Start / Harness Transfer).
    """
    if mode == "none":
        return ""

    if mode == "problem_only":
        logger.warning(
            "Rendering SIM in 'problem_only' ablation mode (rationales and predicted/observed "
            "history omitted). Yu et al. (arXiv:2611.18856) found problem-only memory can "
            "underperform no memory; use mode='full' for normal CRSI runs."
        )
        lines: list[str] = ["## Self-Improvement Memory (problem_only ablation)"]
        if not doc.problems:
            lines.append("(empty)")
        else:
            for entry in doc.problems:
                lines.append(f"- {entry.id}: {entry.name} (opened_round={entry.opened_round})")
        return "\n".join(lines)

    if mode != "full":
        raise ValueError(f"Unsupported SIM render mode: {mode!r}")

    sections: list[str] = []
    if include_usage_instructions:
        sections.append(SIM_USAGE_INSTRUCTIONS)

    header_lines: list[str] = [
        f"## Self-Improvement Memory (schema={doc.schema_version}, next_id={doc.next_id})"
    ]
    if doc.carried_from is not None:
        cf = doc.carried_from
        header_lines.append(
            f"Carried from: arm={cf.arm}, domain={cf.domain}, model={cf.model or 'unspecified'}, "
            f"into={cf.into}, carried_at={cf.carried_at}"
        )
    if doc.aliases:
        alias_pairs = ", ".join(f"{src}->{dst}" for src, dst in sorted(doc.aliases.items()))
        header_lines.append(f"Aliases: {alias_pairs}")

    if not doc.problems:
        header_lines.append("(no root-cause problems recorded yet)")
    else:
        for entry in doc.problems:
            header_lines.append("")
            header_lines.extend(_render_problem_full(entry))

    sections.append("\n".join(header_lines))
    return "\n\n".join(sections)


def format_sim_summary(doc: SimDocument, *, problem_id: str | None = None) -> str:
    """Format a human-readable operator summary for ``crsi show-sim``."""
    if problem_id is not None:
        # Resolve alias directly against doc.aliases for read-only display.
        cur = problem_id.strip()
        seen: set[str] = set()
        while cur in doc.aliases and cur not in seen:
            seen.add(cur)
            cur = doc.aliases[cur]
        matched = [p for p in doc.problems if p.id == cur]
        if not matched:
            return f"SIM problem {problem_id!r} not found."
        return "\n".join(_render_problem_full(matched[0]))

    return render_sim(doc, mode="full", include_usage_instructions=False)
