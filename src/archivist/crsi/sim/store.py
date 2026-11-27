"""Atomic JSON persistence, root-cause merge, and cross-model/domain carry-over for SIM."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from archivist.crsi.guards.secrets import SecretRedactor
from archivist.crsi.sim.models import (
    AttemptRecord,
    CarriedFrom,
    ProblemEntry,
    SimDocument,
    parse_problem_number,
)


def normalize_root_cause(name: str) -> str:
    """Canonical normalization for deterministic root-cause matching."""
    return " ".join(name.strip().lower().split())


class SimStore:
    """In-memory and atomic on-disk manager for ``.crsi/sim.json``."""

    def __init__(
        self,
        document: SimDocument | None = None,
        *,
        path: Path | None = None,
        redactor: SecretRedactor | None = None,
    ) -> None:
        self.doc: SimDocument = document if document is not None else SimDocument()
        self.path: Path | None = path
        self.redactor: SecretRedactor = redactor if redactor is not None else SecretRedactor()

    @classmethod
    def load(
        cls,
        path: Path,
        *,
        redactor: SecretRedactor | None = None,
    ) -> SimStore:
        """Load and validate an existing SIM JSON file."""
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
        doc = SimDocument.model_validate(data)
        return cls(document=doc, path=path, redactor=redactor)

    @classmethod
    def load_or_create(
        cls,
        path: Path,
        *,
        redactor: SecretRedactor | None = None,
    ) -> SimStore:
        """Load ``path`` if it exists; otherwise initialize an empty ``SimStore`` bound to it."""
        if path.exists():
            return cls.load(path, redactor=redactor)
        return cls(document=SimDocument(), path=path, redactor=redactor)

    def save(self, path: Path | None = None) -> Path:
        """Atomically persist the validated SIM document as canonical formatted JSON."""
        target = path if path is not None else self.path
        if target is None:
            raise ValueError("No target path provided for SimStore.save()")

        # Re-validate invariants before writing.
        validated = SimDocument.model_validate(self.doc.model_dump())
        payload = json.dumps(validated.model_dump(mode="json"), indent=2, sort_keys=False) + "\n"

        # Hard security invariant: never persist secrets into SIM.
        self.redactor.assert_no_secrets(payload, context=str(target))

        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(
            dir=str(target.parent),
            prefix=f".{target.name}.",
            suffix=".tmp",
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp_path, target)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        self.doc = validated
        self.path = target
        return target

    def resolve_id(self, problem_id: str) -> str:
        """Resolve ``problem_id`` transitively through ``doc.aliases``."""
        cur = problem_id.strip()
        visited: set[str] = set()
        while cur in self.doc.aliases:
            if cur in visited:
                raise ValueError(f"Cycle detected while resolving SIM alias {problem_id!r}")
            visited.add(cur)
            cur = self.doc.aliases[cur]
        return cur

    def get_problem(self, problem_id: str) -> ProblemEntry | None:
        """Return the canonical ``ProblemEntry`` for ``problem_id`` (resolving aliases)."""
        canonical = self.resolve_id(problem_id)
        for entry in self.doc.problems:
            if entry.id == canonical:
                return entry
        return None

    def find_by_root_cause(self, name: str) -> ProblemEntry | None:
        """Return existing ``ProblemEntry`` matching normalized root-cause text, if any."""
        target = normalize_root_cause(name)
        if not target:
            return None
        for entry in self.doc.problems:
            if normalize_root_cause(entry.name) == target:
                return entry
        return None

    def open_or_get_problem(self, name: str, *, opened_round: int) -> ProblemEntry:
        """Return existing entry with identical normalized root cause, or open a new entry."""
        scrubbed_name = self.redactor.redact_text(name).strip()
        if not scrubbed_name:
            raise ValueError("Root-cause problem name must be non-empty")

        existing = self.find_by_root_cause(scrubbed_name)
        if existing is not None:
            return existing

        new_id = f"P{self.doc.next_id}"
        self.doc.next_id += 1
        entry = ProblemEntry(
            id=new_id,
            name=scrubbed_name,
            opened_round=opened_round,
            attempts=[],
        )
        self.doc.problems.append(entry)
        return entry

    def record_attempt(
        self,
        *,
        attempt: AttemptRecord,
        problem_id: str | None = None,
        new_problem_name: str | None = None,
    ) -> ProblemEntry:
        """Append ``attempt`` to a cited problem ID or open/merge by root-cause name."""
        has_pid = bool(problem_id and problem_id.strip())
        has_name = bool(new_problem_name and new_problem_name.strip())
        if has_pid == has_name:
            raise ValueError(
                "Specify exactly one of 'problem_id' or 'new_problem_name' when recording an attempt"
            )

        scrubbed_attempt = attempt.model_copy(
            update={
                "what": self.redactor.redact_text(attempt.what),
                "rationale": self.redactor.redact_text(attempt.rationale),
            }
        )

        if has_pid and problem_id is not None:
            entry = self.get_problem(problem_id)
            if entry is None:
                raise KeyError(f"Unknown SIM problem id: {problem_id!r}")
        else:
            assert new_problem_name is not None
            entry = self.open_or_get_problem(new_problem_name, opened_round=attempt.round)

        entry.attempts.append(scrubbed_attempt)
        return entry

    def merge_entries(self, src_id: str, dst_id: str) -> ProblemEntry:
        """Deterministically merge ``src_id`` into ``dst_id`` preserving 100% of attempts.

        Chronological ordering by ``(attempt.round, source_precedence, original_index)``
        is preserved, ``opened_round`` takes the minimum, ``aliases[src] = dst`` is
        recorded, and transitive alias chains are canonicalized.
        """
        canon_src = self.resolve_id(src_id)
        canon_dst = self.resolve_id(dst_id)
        if canon_src == canon_dst:
            raise ValueError(f"Cannot merge SIM problem {canon_src!r} into itself")

        src_entry = self.get_problem(canon_src)
        dst_entry = self.get_problem(canon_dst)
        if src_entry is None:
            raise KeyError(f"Source SIM problem not found: {src_id!r}")
        if dst_entry is None:
            raise KeyError(f"Destination SIM problem not found: {dst_id!r}")

        tagged: list[tuple[int, int, int, AttemptRecord]] = []
        dst_rank = 0 if parse_problem_number(canon_dst) < parse_problem_number(canon_src) else 1
        src_rank = 1 - dst_rank
        for idx, att in enumerate(dst_entry.attempts):
            tagged.append((att.round, dst_rank, idx, att))
        for idx, att in enumerate(src_entry.attempts):
            tagged.append((att.round, src_rank, idx, att))
        tagged.sort(key=lambda item: (item[0], item[1], item[2]))

        dst_entry.attempts = [item[3] for item in tagged]
        dst_entry.opened_round = min(dst_entry.opened_round, src_entry.opened_round)

        self.doc.problems = [p for p in self.doc.problems if p.id != canon_src]

        # Update direct alias and flatten any transitive aliases pointing to canon_src.
        for alias_key, alias_target in list(self.doc.aliases.items()):
            if alias_target == canon_src:
                self.doc.aliases[alias_key] = canon_dst
        self.doc.aliases[canon_src] = canon_dst

        return dst_entry

    def deduplicate_by_root_cause(self) -> list[tuple[str, str]]:
        """Merge any problem entries sharing identical normalized root-cause text.

        Keeps the lowest problem ID as canonical destination. Returns ``(src_id, dst_id)``
        pairs for all merges performed.
        """
        by_norm: dict[str, str] = {}
        merges: list[tuple[str, str]] = []
        ordered = sorted(self.doc.problems, key=lambda p: parse_problem_number(p.id))
        for entry in ordered:
            key = normalize_root_cause(entry.name)
            if key in by_norm:
                merges.append((entry.id, by_norm[key]))
            else:
                by_norm[key] = entry.id

        for src_id, dst_id in merges:
            self.merge_entries(src_id, dst_id)
        return merges

    def carry_over(
        self,
        *,
        into_arm: str,
        source_arm: str,
        source_domain: str,
        source_model: str = "",
        carried_at: str | None = None,
        target_path: Path | None = None,
    ) -> SimStore:
        """Create a carried-over copy of SIM for a new model or domain.

        Preserves all problems, attempts (with each historical attempt's original
        ``model`` and ``domain`` intact), aliases, and ``next_id``.
        """
        cloned = self.doc.model_copy(deep=True)
        carried_kwargs: dict[str, str] = {
            "arm": source_arm,
            "domain": source_domain,
            "model": source_model,
            "into": into_arm,
        }
        if carried_at is not None:
            carried_kwargs["carried_at"] = carried_at
        cloned.carried_from = CarriedFrom.model_validate(carried_kwargs)
        return SimStore(document=cloned, path=target_path, redactor=self.redactor)

    def has_negative_evidence_for(
        self,
        problem_id: str,
        *,
        domain: str | None = None,
        model: str | None = None,
    ) -> bool:
        """Return True if ``problem_id`` records observed regressions or failed outcome.

        Used by the legal-edit filter (SPEC-006) when verifying whether an interlock
        modification cites concrete SIM evidence in the current setting.
        """
        entry = self.get_problem(problem_id)
        if entry is None:
            return False
        for att in entry.attempts:
            if domain is not None and att.domain != domain:
                continue
            if model is not None and att.model != model:
                continue
            if (
                att.observed.regressed
                or (not att.accepted)
                or (bool(att.observed.missed) and not att.observed.fixed)
            ):
                return True
        return False
