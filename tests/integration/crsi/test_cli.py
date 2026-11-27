"""End-to-end integration tests for the CRSI operator CLI (`python -m archivist.crsi`) (SPEC-008)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from archivist.crsi.cli import main
from archivist.crsi.sim.models import AttemptRecord, ObservedOutcome, ProblemEntry, SimDocument
from archivist.crsi.sim.store import SimStore

pytestmark = [pytest.mark.crsi, pytest.mark.integration]


def test_cli_init_validate_dry_run_run_status_and_show_sim(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    repo = tmp_path / "toy_project"
    private_dir = tmp_path / "toy_private"

    # 1. init --toy
    rc_init = main(
        [
            "init",
            "--repo-root",
            str(repo),
            "--private-dir",
            str(private_dir),
            "--toy",
            "--toy-split",
            "base",
        ]
    )
    assert rc_init == 0
    cfg_file = repo / ".crsi" / "config.toml"
    sim_file = repo / ".crsi" / "sim.json"
    assert cfg_file.is_file()
    assert sim_file.is_file()
    cfg_text = cfg_file.read_text(encoding="utf-8")
    assert "allow_paid_apis = false" in cfg_text
    assert "max_spend_usd = 0.0" in cfg_text
    capsys.readouterr()

    # 2. validate
    rc_val = main(["validate", "--repo-root", str(repo)])
    assert rc_val == 0
    val_out = capsys.readouterr().out
    assert "CRSI validation OK" in val_out
    assert "evolve=4 heldout=2" in val_out

    # 3. dry-run: exact math (baseline=4 + 2 rounds * 3 candidates * 4 tasks * 1 rollout = 28)
    rc_dry = main(
        [
            "dry-run",
            "--repo-root",
            str(repo),
            "--rounds",
            "2",
            "--candidates",
            "3",
            "--include-heldout",
            "--json",
        ]
    )
    assert rc_dry == 0
    dry_plan = json.loads(capsys.readouterr().out)
    assert dry_plan["evolve_tasks"] == 4
    assert dry_plan["heldout_tasks"] == 2
    assert dry_plan["baseline_task_runs"] == 4
    assert dry_plan["candidate_task_runs"] == 24
    assert dry_plan["total_evolve_task_runs"] == 28
    assert dry_plan["heldout_task_runs"] == 2
    # Confirm zero runs were created during dry-run.
    assert not (repo / ".crsi" / "runs").exists()

    # 4. run (1 round deterministic toy)
    rc_run = main(
        [
            "run",
            "--repo-root",
            str(repo),
            "--run-id",
            "cli-toy-01",
            "--rounds",
            "1",
            "--candidates",
            "2",
            "--parallel",
            "2",
            "--new",
        ]
    )
    assert rc_run == 0
    run_out = capsys.readouterr().out
    assert "CRSI Run cli-toy-01 [completed]" in run_out
    assert "crsi/cli-toy-01" in run_out
    assert "0.0000 -> 1.0000" in run_out

    # 5. status --json
    rc_status = main(["status", "--repo-root", str(repo), "--run-id", "cli-toy-01", "--json"])
    assert rc_status == 0
    status_data = json.loads(capsys.readouterr().out)
    assert status_data["run_id"] == "cli-toy-01"
    assert status_data["status"] == "completed"
    assert status_data["run_branch"] == "crsi/cli-toy-01"
    assert status_data["baseline_score"] == pytest.approx(0.0)
    assert status_data["current_score"] == pytest.approx(1.0)
    assert status_data["accepted_count"] == 1
    assert status_data["spent_usd"] == 0.0

    # 6. show-sim (text + --json)
    rc_show = main(["show-sim", "--repo-root", str(repo), "--run-id", "cli-toy-01"])
    assert rc_show == 0
    show_out = capsys.readouterr().out
    assert "P1: premature completion without execution verification" in show_out
    assert "ExecutionGatingInterlock" in show_out

    rc_show_json = main(["show-sim", "--repo-root", str(repo), "--run-id", "cli-toy-01", "--json"])
    assert rc_show_json == 0
    sim_json = json.loads(capsys.readouterr().out)
    assert sim_json["problems"][0]["id"] == "P1"


def test_cli_validate_catches_inline_secrets_and_editable_frozen_overlap(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    repo = tmp_path / "bad_proj"
    private_dir = tmp_path / "bad_priv"
    assert (
        main(
            [
                "init",
                "--repo-root",
                str(repo),
                "--private-dir",
                str(private_dir),
                "--toy",
            ]
        )
        == 0
    )
    capsys.readouterr()

    # Case A: Inline secret in config.toml must fail validate.
    cfg_file = repo / ".crsi" / "config.toml"
    orig = cfg_file.read_text(encoding="utf-8")
    cfg_file.write_text(orig + "\n# sk-proj-1234567890abcdefghijklmnop\n", encoding="utf-8")
    assert main(["validate", "--repo-root", str(repo)]) == 1
    assert "secret" in capsys.readouterr().err.lower()

    # Case B: Editable allowlist overlapping frozen pyproject.toml must fail validate.
    cfg_file.write_text(
        orig.replace('editable = ["harness/**"]', 'editable = ["harness/**", "pyproject.toml"]'),
        encoding="utf-8",
    )
    assert main(["validate", "--repo-root", str(repo)]) == 1
    assert "pyproject.toml" in capsys.readouterr().err


def test_cli_refuses_paid_provider_without_opt_in_and_positive_budget(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    repo = tmp_path / "paid_gate_proj"
    private_dir = tmp_path / "paid_gate_priv"
    assert (
        main(
            [
                "init",
                "--repo-root",
                str(repo),
                "--private-dir",
                str(private_dir),
                "--toy",
            ]
        )
        == 0
    )
    capsys.readouterr()

    cfg_file = repo / ".crsi" / "config.toml"
    cfg_text = cfg_file.read_text(encoding="utf-8").replace(
        'provider = "fake"          # "fake" | "openai_compat" | "anthropic" | "gemini"\nmodel = "fake-policy-v1"',
        'provider = "anthropic"\nmodel = "claude-sonnet-4-20250514"\napi_key_env = "ANTHROPIC_API_KEY"',
    )
    cfg_file.write_text(cfg_text, encoding="utf-8")

    # Without --allow-paid / --max-spend > 0, CLI exits non-zero (2) before any network call.
    rc = main(["run", "--repo-root", str(repo), "--run-id", "blocked-paid"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Refusing to run paid provider" in err


def test_cli_sim_merge_preserves_attempts_and_updates_aliases(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
):
    sim_path = tmp_path / "sim.json"
    att1 = AttemptRecord(
        round=1,
        sample="s1",
        accepted=True,
        what="first fix",
        files=["harness/interlock.py"],
        rationale="r1",
        predicted=["t1"],
        observed=ObservedOutcome(
            fixed=["t1"], missed=[], regressed=[], passed=1, total=2, score=0.5
        ),
        model="fake-policy-v1",
        domain="toy_base",
    )
    att2 = att1.model_copy(update={"round": 2, "sample": "s2", "what": "second fix"})
    doc = SimDocument(
        next_id=3,
        problems=[
            ProblemEntry(id="P1", name="premature completion", opened_round=1, attempts=[att1]),
            ProblemEntry(id="P2", name="premature finish", opened_round=2, attempts=[att2]),
        ],
    )
    SimStore(document=doc).save(sim_path)

    rc = main(["sim", "merge", "P2", "P1", "--sim", str(sim_path)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Merged P2 -> P1" in out

    reloaded = SimStore.load(sim_path)
    assert [p.id for p in reloaded.doc.problems] == ["P1"]
    assert len(reloaded.doc.problems[0].attempts) == 2
    assert reloaded.doc.aliases == {"P2": "P1"}


def test_python_m_archivist_crsi_module_entrypoint_help():
    """Verify `python -m archivist.crsi --help` runs cleanly via subprocess."""
    env = dict(os.environ)
    src_path = str(Path(__file__).resolve().parents[3] / "src")
    env["PYTHONPATH"] = f"{src_path}{os.pathsep}{env.get('PYTHONPATH', '')}".rstrip(os.pathsep)
    proc = subprocess.run(
        [sys.executable, "-m", "archivist.crsi", "--help"],
        env=env,
        capture_output=True,
        text=True,
        check=False,
        shell=False,
    )
    assert proc.returncode == 0
    for cmd in ("init", "validate", "dry-run", "run", "status", "show-sim", "sim"):
        assert cmd in proc.stdout
