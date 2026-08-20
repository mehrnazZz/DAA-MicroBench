from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

from microbench.rl import (
    LEARNED_CLOSED_LOOP_STUDY_SCHEMA_VERSION,
    run_learned_closed_loop_study,
    train_behavior_cloned_policy,
)


ROOT = Path(__file__).resolve().parents[1]


def _base_policy_spec(tmp_path: Path) -> Path:
    report = train_behavior_cloned_policy(
        out_dir=tmp_path / "base_bc",
        lanes=["head_on"],
        max_steps=2,
        hidden_dim=8,
        rollout_noise_std=0.0,
        eval_lanes=["head_on"],
        eval_max_steps=2,
    )
    return Path(report["policy_spec"])


def test_learned_closed_loop_study_writes_bundle_and_review_artifacts(tmp_path: Path) -> None:
    base_spec = _base_policy_spec(tmp_path)

    report = run_learned_closed_loop_study(
        out_dir=tmp_path / "study",
        base_policy_spec=base_spec,
        lanes=["head_on"],
        train_max_steps=2,
        generations=0,
        population_size=1,
        trainable_parameters="all_layers",
        sigma=0.01,
        eval_lanes=["head_on"],
        eval_max_steps=2,
        holdout_scenarios=["sphere_swap_3d_medium"],
        holdout_seeds=[0],
        holdout_comm_profiles=["ideal_50hz"],
        holdout_n_agents=3,
        holdout_max_runs=1,
        bundle_n_agents=3,
        bundle_max_steps=2,
        bundle_max_runs=1,
    )

    assert report["schema_version"] == LEARNED_CLOSED_LOOP_STUDY_SCHEMA_VERSION
    assert report["ok"] is True
    assert report["promotion_candidate"] is True
    assert report["recommendation"] == "promote"
    assert report["training"]["holdout"]["promotion_candidate"] is True
    assert report["bundle"]["ok"] is True
    assert report["leaderboard"]["ok"] is True
    assert report["diagnostics"]["ok"] is True

    artifacts = report["artifacts"]
    for key in (
        "study_manifest",
        "study_report",
        "training_report",
        "policy_spec",
        "model_artifact",
        "bundle_report",
        "leaderboard_json",
        "leaderboard_csv",
        "diagnostics_json",
        "diagnostics_csv",
        "diagnostics_markdown",
    ):
        assert Path(artifacts[key]).exists(), key

    manifest = json.loads(Path(artifacts["study_manifest"]).read_text(encoding="utf-8"))
    assert manifest["workflow"] == "learned-closed-loop-study"
    assert manifest["recommendation"] == "promote"

    leaderboard = json.loads(Path(artifacts["leaderboard_json"]).read_text(encoding="utf-8"))
    assert leaderboard["rows"][0]["lineage_label"] == "closed_loop_holdout_passed"
    assert leaderboard["rows"][0]["promotion_stage"] == "holdout_passed"


def test_learned_closed_loop_study_cli_smoke(tmp_path: Path) -> None:
    base_spec = _base_policy_spec(tmp_path)
    out_dir = tmp_path / "cli_study"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "learned-closed-loop-study",
            "--out-dir",
            str(out_dir),
            "--base-policy-spec",
            str(base_spec),
            "--lanes",
            "head_on",
            "--train-max-steps",
            "2",
            "--generations",
            "0",
            "--population-size",
            "1",
            "--trainable-parameters",
            "all_layers",
            "--sigma",
            "0.01",
            "--eval-lanes",
            "head_on",
            "--eval-max-steps",
            "2",
            "--holdout-scenarios",
            "sphere_swap_3d_medium",
            "--holdout-seeds",
            "0",
            "--holdout-comm",
            "ideal_50hz",
            "--holdout-n",
            "3",
            "--holdout-max-runs",
            "1",
            "--bundle-n",
            "3",
            "--bundle-max-steps",
            "2",
            "--bundle-max-runs",
            "1",
            "--require-pass",
            "--require-promotion",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["promotion_candidate"] is True
    assert report["recommendation"] == "promote"
    assert (out_dir / "study_manifest.json").exists()
    assert (out_dir / "learned_closed_loop_study_report.json").exists()
    assert (out_dir / "training" / "closed_loop_training_report.json").exists()
    assert (out_dir / "bundle" / "learned_submission_bundle.json").exists()
    assert (out_dir / "learned_leaderboard.json").exists()
    assert (out_dir / "learned_diagnostics.md").exists()
