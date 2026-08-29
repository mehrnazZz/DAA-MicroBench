from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from microbench.learned import MLP_LEARNED_MODEL_ID
from microbench.rl import (
    CLOSED_LOOP_BROAD_3D_TRAINING_LANE_IDS,
    CLOSED_LOOP_TRAINING_SCHEMA_VERSION,
    fine_tune_closed_loop_policy,
    load_policy_from_spec,
    selected_closed_loop_training_lanes,
    train_behavior_cloned_policy,
)
from microbench.rl.schema import OBS_A_MAX_INDEX, OBS_GOAL_DIR_SLICE, OBS_RADIUS_INDEX, OBS_V_MAX_INDEX


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


def _observation() -> np.ndarray:
    obs = np.zeros(89, dtype=np.float32)
    obs[OBS_GOAL_DIR_SLICE] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    obs[OBS_RADIUS_INDEX] = 0.5
    obs[OBS_V_MAX_INDEX] = 3.0
    obs[OBS_A_MAX_INDEX] = 2.0
    return obs


def test_closed_loop_finetune_writes_guarded_policy_spec(tmp_path: Path) -> None:
    base_spec = _base_policy_spec(tmp_path)

    report = fine_tune_closed_loop_policy(
        out_dir=tmp_path / "closed_loop",
        base_policy_spec=base_spec,
        lanes=["head_on"],
        train_max_steps=2,
        generations=1,
        population_size=2,
        trainable_parameters="all_layers",
        sigma=0.01,
        eval_lanes=["head_on"],
        eval_max_steps=2,
    )

    assert report["schema_version"] == CLOSED_LOOP_TRAINING_SCHEMA_VERSION
    assert report["ok"] is True
    assert report["candidate_count"] == 3
    assert report["base_metrics"]["collision_ticks"] >= report["best_metrics"]["collision_ticks"]
    assert Path(report["policy_spec"]).exists()
    assert Path(report["model_artifact"]).exists()
    assert Path(report["candidate_summary_csv"]).exists()
    assert Path(report["candidate_episodes_csv"]).exists()
    assert Path(report["candidate_lane_summary_csv"]).exists()

    model = json.loads(Path(report["model_artifact"]).read_text(encoding="utf-8"))
    assert model["model_id"] == MLP_LEARNED_MODEL_ID
    assert model["training"]["recipe"] == "python -m microbench.cli learned-closed-loop-finetune"
    assert model["training"]["lane_profile"] == "validation"
    assert model["training"]["trainable_parameters"] == "all_layers"
    assert model["training"]["search_strategy"] == "single_stage"
    assert model["training"]["antithetic_sampling"] is False
    assert model["training"]["require_per_lane_safety"] is False
    assert model["training"]["per_lane_clearance_tolerance_m"] == 0.001
    assert model["training"]["public_observations_only"] is True
    assert model["training"]["privileged_global_state"] is False
    assert model["training"]["base_policy_spec"] == str(base_spec)

    loaded = load_policy_from_spec(report["policy_spec"], seed=5)
    action = loaded.policy.action("agent_0", _observation(), None, {})
    assert loaded.summary["adapter"] == "mlp_json"
    assert loaded.policy_name == "closed_loop_mlp_learned"
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))


def test_closed_loop_finetune_reports_lanes_and_two_stage_antithetic_search(tmp_path: Path) -> None:
    base_spec = _base_policy_spec(tmp_path)

    report = fine_tune_closed_loop_policy(
        out_dir=tmp_path / "closed_loop_two_stage",
        base_policy_spec=base_spec,
        lanes=["head_on"],
        train_max_steps=2,
        generations=2,
        population_size=1,
        trainable_parameters="all_layers",
        search_strategy="two_stage",
        stage1_generations=1,
        stage2_generations=1,
        antithetic_sampling=True,
        require_per_lane_safety=True,
        per_lane_clearance_tolerance_m=0.01,
        sigma=0.01,
        eval_lanes=["head_on"],
        eval_max_steps=2,
        run_validation=False,
    )

    assert report["ok"] is True
    assert report["candidate_count"] == 5
    assert report["search_strategy"] == "two_stage"
    assert report["antithetic_sampling"] is True
    assert report["require_per_lane_safety"] is True
    assert report["per_lane_clearance_tolerance_m"] == 0.01
    assert [stage["stage"] for stage in report["search_plan"]] == ["output_head_warmup", "all_layers_refine"]

    with Path(report["candidate_summary_csv"]).open("r", newline="", encoding="utf-8") as f:
        candidate_rows = list(csv.DictReader(f))
    assert {row["stage"] for row in candidate_rows} >= {"base", "output_head_warmup", "all_layers_refine"}
    assert {row["trainable_parameters"] for row in candidate_rows} >= {"output_head", "all_layers"}
    assert any(row["candidate_id"].endswith("_anti") for row in candidate_rows)

    with Path(report["candidate_lane_summary_csv"]).open("r", newline="", encoding="utf-8") as f:
        lane_rows = list(csv.DictReader(f))
    assert len(lane_rows) == report["candidate_count"]
    assert {row["lane_id"] for row in lane_rows} == {"head_on"}
    assert all(row["score"] for row in lane_rows)
    assert all(row["per_lane_clearance_tolerance_m"] == "0.01" for row in lane_rows)


def test_closed_loop_training_lane_profile_adds_broad_3d_lanes() -> None:
    validation = selected_closed_loop_training_lanes(lane_profile="validation")
    broad = selected_closed_loop_training_lanes(lane_profile="broad_3d_stress")
    combined = selected_closed_loop_training_lanes(lane_profile="validation_plus_broad_3d")

    assert [lane.lane_id for lane in broad] == list(CLOSED_LOOP_BROAD_3D_TRAINING_LANE_IDS)
    assert len(combined) == len(validation) + len(broad)
    assert [lane.lane_id for lane in combined[: len(validation)]] == [lane.lane_id for lane in validation]
    assert [lane.lane_id for lane in combined[len(validation) :]] == list(CLOSED_LOOP_BROAD_3D_TRAINING_LANE_IDS)


def test_closed_loop_finetune_supports_broad_3d_training_profile(tmp_path: Path) -> None:
    base_spec = _base_policy_spec(tmp_path)

    report = fine_tune_closed_loop_policy(
        out_dir=tmp_path / "closed_loop_broad_3d_training",
        base_policy_spec=base_spec,
        lane_profile="broad_3d_stress",
        train_max_steps=2,
        generations=0,
        population_size=1,
        trainable_parameters="all_layers",
        eval_lanes=["head_on"],
        eval_max_steps=2,
        holdout_profile="none",
    )

    assert report["ok"] is True
    assert report["lane_profile"] == "broad_3d_stress"
    assert [lane["lane_id"] for lane in report["training_lanes"]] == list(CLOSED_LOOP_BROAD_3D_TRAINING_LANE_IDS)
    model = json.loads(Path(report["model_artifact"]).read_text(encoding="utf-8"))
    assert model["training"]["lane_profile"] == "broad_3d_stress"
    assert model["training"]["training_lanes"] == list(CLOSED_LOOP_BROAD_3D_TRAINING_LANE_IDS)


def test_closed_loop_finetune_broad_3d_holdout_promotion_gate(tmp_path: Path) -> None:
    base_spec = _base_policy_spec(tmp_path)

    report = fine_tune_closed_loop_policy(
        out_dir=tmp_path / "closed_loop_holdout",
        base_policy_spec=base_spec,
        lanes=["head_on"],
        train_max_steps=2,
        generations=0,
        population_size=1,
        trainable_parameters="all_layers",
        eval_lanes=["head_on"],
        eval_max_steps=2,
        holdout_profile="broad_3d_stress",
        holdout_scenarios=["sphere_swap_3d_medium"],
        holdout_seeds=[0],
        holdout_comm_profiles=["ideal_50hz"],
        holdout_n_agents=3,
        holdout_max_runs=1,
        run_validation=False,
    )

    assert report["ok"] is True
    assert report["behavior_pass"] is True
    assert report["promotion_candidate"] is True
    assert report["promotion_status"] == "candidate"
    assert report["holdout"]["profile"] == "broad_3d_stress"
    assert report["holdout"]["expected_runs_per_policy"] == 1
    assert report["holdout"]["base"]["collision_episodes"] == 0
    assert report["holdout"]["tuned"]["collision_episodes"] == 0
    assert Path(report["holdout"]["comparison_csv"]).exists()
    assert Path(report["holdout"]["base"]["results_csv"]).exists()
    assert Path(report["holdout"]["tuned"]["summary_csv"]).exists()
    assert {check["name"] for check in report["checks"]} >= {
        "holdout_runs_completed",
        "holdout_no_collision_regression",
        "holdout_score_not_worse",
    }

    model = json.loads(Path(report["model_artifact"]).read_text(encoding="utf-8"))
    assert model["training"]["holdout"]["profile"] == "broad_3d_stress"
    assert model["training"]["holdout_result"]["promotion_candidate"] is True


def test_closed_loop_finetune_cli_smoke(tmp_path: Path) -> None:
    base_spec = _base_policy_spec(tmp_path)
    out_dir = tmp_path / "closed_loop_cli"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "learned-closed-loop-finetune",
            "--out-dir",
            str(out_dir),
            "--base-policy-spec",
            str(base_spec),
            "--lanes",
            "head_on",
            "--train-max-steps",
            "2",
            "--generations",
            "1",
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
    assert report["candidate_count"] == 2
    assert (out_dir / "closed_loop_training_report.json").exists()
    assert (out_dir / "policy_spec.json").exists()
    assert (out_dir / "candidate_summary.csv").exists()
