from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from microbench.learned import MLP_LEARNED_MODEL_ID
from microbench.rl import build_behavior_cloned_policy_evidence, load_policy_from_spec, train_behavior_cloned_policy
from microbench.rl.schema import OBS_A_MAX_INDEX, OBS_GOAL_DIR_SLICE, OBS_RADIUS_INDEX, OBS_V_MAX_INDEX
from microbench.runner import run_episode
from microbench.scenarios import materialize_official_suite
from microbench.types import RunSpec


ROOT = Path(__file__).resolve().parents[1]


def _observation() -> np.ndarray:
    obs = np.zeros(89, dtype=np.float32)
    obs[OBS_GOAL_DIR_SLICE] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    obs[OBS_RADIUS_INDEX] = 0.5
    obs[OBS_V_MAX_INDEX] = 3.0
    obs[OBS_A_MAX_INDEX] = 2.0
    return obs


def test_behavior_cloned_training_writes_portable_policy_spec(tmp_path: Path) -> None:
    report = train_behavior_cloned_policy(
        out_dir=tmp_path / "bc_train",
        lanes=["head_on"],
        max_steps=2,
        hidden_dim=8,
        rollout_noise_std=0.0,
        eval_lanes=["head_on"],
        eval_max_steps=2,
    )

    assert report["ok"] is True
    assert report["sample_count"] == 8
    assert report["teacher_policy"] == "local_lateral_avoidance_teacher_v0"
    assert report["public_observations_only"] is True
    assert report["validation_matrix"]["ok"] is True

    model_payload = json.loads(Path(report["model_artifact"]).read_text(encoding="utf-8"))
    assert model_payload["model_id"] == MLP_LEARNED_MODEL_ID
    assert model_payload["postprocess"]["goal_forward_floor"] is True
    assert model_payload["training"]["source"] == "DAA Microbench RL validation-matrix rollouts"
    assert model_payload["training"]["privileged_global_state"] is False

    loaded = load_policy_from_spec(report["policy_spec"], seed=3)
    action = loaded.policy.action("agent_0", _observation(), None, {})
    assert loaded.summary["adapter"] == "mlp_json"
    assert loaded.policy_name == "bc_mlp_learned"
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))

    generated = materialize_official_suite("official_smoke_generated", tmp_path / "suite", overwrite=True)
    scenario = next(path for path in generated["scenario_paths"] if path.stem == "head_on_2d_easy")
    row = run_episode(
        RunSpec(
            scenario_path=str(scenario),
            method="learned_policy_spec",
            policy_spec=str(report["policy_spec"]),
            n_agents=4,
            seed=0,
            comm_profile="ideal_50hz",
            out_dir=str(tmp_path / "planner_run"),
            save_trace=False,
        )
    )
    assert row["method"] == "learned_policy_spec"
    assert int(row["planner_error_count"]) == 0
    assert int(row["planner_timeout_count"]) == 0


def test_train_learned_bc_cli_smoke(tmp_path: Path) -> None:
    out_dir = tmp_path / "bc_cli"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "train-learned-bc",
            "--out-dir",
            str(out_dir),
            "--lanes",
            "head_on",
            "--max-steps",
            "2",
            "--eval-lanes",
            "head_on",
            "--eval-max-steps",
            "2",
            "--hidden-dim",
            "8",
            "--rollout-noise-std",
            "0.0",
            "--require-pass",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["sample_count"] == 8
    assert Path(report["policy_spec"]).exists()
    assert Path(report["model_artifact"]).exists()
    assert (out_dir / "bc_training_report.json").exists()


def test_behavior_cloned_evidence_builds_bundles_and_leaderboard(tmp_path: Path) -> None:
    out_dir = tmp_path / "bc_evidence"
    report = build_behavior_cloned_policy_evidence(
        out_dir=out_dir,
        lanes=["head_on"],
        max_steps=2,
        hidden_dim=8,
        rollout_noise_std=0.0,
        eval_lanes=["head_on"],
        eval_max_steps=2,
        bundle_max_steps=2,
        bundle_max_runs=1,
    )

    assert report["ok"] is True
    assert set(report["bundle_paths"]) == {"bc", "tiny", "mlp"}
    assert report["bundles"]["bc"]["policy"] == "bc_mlp_learned"
    assert report["leaderboard"]["ok"] is True
    assert report["leaderboard"]["bundle_count"] == 3
    assert {row["policy"] for row in report["leaderboard"]["rows"]} == {
        "bc_mlp_learned",
        "tiny_learned",
        "mlp_learned",
    }

    bc_bundle = out_dir / "bc_bundle"
    assert (bc_bundle / "policy_spec.json").exists()
    assert (bc_bundle / "policy_artifacts" / "bc_mlp_policy.json").exists()
    manifest = json.loads((bc_bundle / "learned_submission_manifest.json").read_text(encoding="utf-8"))
    assert manifest["training_disclosure"]["teacher_policy"] == "local_lateral_avoidance_teacher_v0"
    assert manifest["training_disclosure"]["external_data"] == "none"
    assert manifest["review_notes"]["privileged_information"] == "none"


def test_learned_bc_evidence_cli_smoke_without_fixtures(tmp_path: Path) -> None:
    out_dir = tmp_path / "bc_evidence_cli"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "learned-bc-evidence",
            "--out-dir",
            str(out_dir),
            "--lanes",
            "head_on",
            "--max-steps",
            "2",
            "--eval-lanes",
            "head_on",
            "--eval-max-steps",
            "2",
            "--hidden-dim",
            "8",
            "--rollout-noise-std",
            "0.0",
            "--bundle-max-steps",
            "2",
            "--max-runs",
            "1",
            "--skip-fixtures",
            "--require-pass",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert set(report["bundle_paths"]) == {"bc"}
    assert report["leaderboard"]["bundle_count"] == 1
    assert (out_dir / "learned_bc_evidence.json").exists()
    assert (out_dir / "learned_policy_leaderboard.csv").exists()
