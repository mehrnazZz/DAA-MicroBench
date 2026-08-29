from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from microbench.learned import (
    MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES,
    MLP_LEARNED_PUBLIC_OBS_FEATURE_SET,
    MLP_LEARNED_PUBLIC_OBS_MODEL_ID,
)
from microbench.rl import (
    LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID,
    LEARNED_DATASET_BC_TRAINING_SOURCE,
    LEARNED_DATASET_PLANNER_EXPERT_SOURCE,
    LEARNED_HARD_LANE_LOOP_SCHEMA_VERSION,
    run_learned_hard_lane_loop,
    select_hard_lanes_from_diagnostics,
    train_behavior_cloned_policy_from_dataset,
)
from microbench.rl.hard_lane_training import (
    _sample_mask_from_diagnostics,
    _sample_selection_config,
    _sample_weighting_config,
    _sample_weights_from_diagnostics,
)
from microbench.rl.learned_dataset import export_learned_policy_dataset


ROOT = Path(__file__).resolve().parents[1]


def _diagnostics(rows: list[dict[str, object]]) -> dict[str, object]:
    return {
        "schema_version": "0.1",
        "ok": True,
        "bundle_count": len(rows),
        "rows": rows,
    }


def test_hard_lane_selector_maps_diagnostics_to_canonical_lanes() -> None:
    report = select_hard_lanes_from_diagnostics(
        _diagnostics(
            [
                {
                    "diagnostic_rank": 2,
                    "policy": "slow_policy",
                    "diagnostic_label": "safe_but_slow",
                    "primary_failure": "incomplete_missions",
                    "worst_scenario": "crossing_2d_medium",
                },
                {
                    "diagnostic_rank": 1,
                    "policy": "close_policy",
                    "diagnostic_label": "fast_but_close",
                    "primary_failure": "low_clearance",
                    "worst_rl_lane": "urban_obstacle",
                },
            ]
        ),
        max_lanes=2,
    )

    assert report["schema_version"] == LEARNED_HARD_LANE_LOOP_SCHEMA_VERSION
    assert report["selected_lanes"] == ["urban_obstacle", "crossing"]
    assert report["fallback_used"] is False
    assert report["reasons"][0]["source_field"] == "worst_rl_lane"


def test_hard_lane_selector_can_target_policy_and_fill_3d_fallbacks() -> None:
    report = select_hard_lanes_from_diagnostics(
        _diagnostics(
            [
                {
                    "diagnostic_rank": 1,
                    "policy": "tiny_learned",
                    "method": "learned_tiny",
                    "diagnostic_label": "fast_but_close",
                    "primary_failure": "low_clearance",
                    "worst_scenario": "crossing_2d_medium",
                },
                {
                    "diagnostic_rank": 3,
                    "policy": "bc_mlp_learned",
                    "method": "learned_policy_spec",
                    "diagnostic_label": "safe_but_slow",
                    "primary_failure": "incomplete_missions",
                    "worst_scenario": "sphere_swap_3d_medium",
                    "worst_rl_lane": "crossing",
                },
            ]
        ),
        fallback_lanes=["urban_obstacle", "communication_delay", "high_n_dense_merge"],
        max_lanes=3,
        target_policy="bc_mlp_learned",
    )

    assert report["selected_lanes"] == ["high_n_dense_merge", "urban_obstacle", "communication_delay"]
    assert report["diagnostic_rows_seen"] == 2
    assert report["diagnostic_rows_considered"] == 1
    assert report["reasons"][0]["source_field"] == "worst_scenario"
    assert report["reasons"][1]["source_field"] == "fallback_lanes"


def test_safety_sample_weighting_prioritizes_close_or_unsafe_samples() -> None:
    config = _sample_weighting_config(mode="safety", clearance_threshold_m=1.5)
    weights, summary = _sample_weights_from_diagnostics(
        {
            "collision": np.asarray([False, True, False, False], dtype=bool),
            "near_miss": np.asarray([False, False, True, False], dtype=bool),
            "min_sep_m": np.asarray([2.0, 0.1, 0.5, 1.2], dtype=np.float32),
        },
        config=config,
    )

    assert summary["mode"] == "safety"
    assert summary["collision_sample_count"] == 1
    assert summary["near_miss_sample_count"] == 1
    assert summary["low_clearance_sample_count"] == 3
    assert summary["weight_max"] > summary["weight_min"]
    assert np.isclose(float(np.mean(weights)), 1.0)
    assert float(weights[1]) > float(weights[0])


def test_hard_negative_sample_selection_keeps_event_windows_and_regular_lanes() -> None:
    config = _sample_selection_config(
        mode="hard_negative_windows",
        hard_lane_ids=[LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID],
        clearance_threshold_m=1.5,
        context_steps=1,
    )
    mask, summary = _sample_mask_from_diagnostics(
        {
            "collision": np.asarray([False, False, False, False, False], dtype=bool),
            "near_miss": np.asarray([False, True, False, False, False], dtype=bool),
            "min_sep_m": np.asarray([3.0, 0.8, 2.7, 2.9, 3.0], dtype=np.float32),
            "lane_id": np.asarray(
                [
                    LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID,
                    LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID,
                    LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID,
                    LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID,
                    "head_on",
                ]
            ),
            "episode_id": np.asarray([7, 7, 7, 7, 8], dtype=np.int32),
            "step": np.asarray([1, 2, 3, 8, 0], dtype=np.int32),
        },
        config=config,
    )

    assert mask.tolist() == [True, True, True, False, True]
    assert summary["mode"] == "hard_negative_windows"
    assert summary["hard_lane_input_count"] == 4
    assert summary["hard_lane_selected_count"] == 3
    assert summary["hard_event_count"] == 1
    assert summary["non_hard_selected_count"] == 1


def test_dataset_shard_training_writes_portable_policy(tmp_path: Path) -> None:
    dataset = export_learned_policy_dataset(
        out_dir=tmp_path / "dataset",
        lanes=["head_on"],
        max_steps=2,
        shard_size=4,
    )

    report = train_behavior_cloned_policy_from_dataset(
        out_dir=tmp_path / "training",
        dataset_manifest=dataset["manifest"],
        hidden_dim=8,
        eval_lanes=["head_on"],
        eval_max_steps=2,
    )

    assert report["ok"] is True
    assert report["training_source"] == LEARNED_DATASET_BC_TRAINING_SOURCE
    assert report["sample_count"] == 8
    assert report["feature_normalization"]["mode"] == "standard"
    assert report["sample_selection"]["mode"] == "all"
    assert report["sample_weighting"]["mode"] == "none"
    assert report["sample_weighting"]["weight_mean"] == 1.0
    assert Path(report["policy_spec"]).exists()
    model = json.loads(Path(report["model_artifact"]).read_text(encoding="utf-8"))
    assert model["feature_normalization"]["mode"] == "standard"
    assert model["training"]["source"] == LEARNED_DATASET_BC_TRAINING_SOURCE
    assert model["training"]["sample_selection"]["mode"] == "all"
    assert model["training"]["sample_weighting"]["mode"] == "none"
    assert model["training"]["source_dataset_manifest"] == dataset["manifest"]
    assert model["training"]["source_policy"] == "local_lateral_avoidance_teacher_v0"


def test_dataset_shard_training_records_planner_expert_source(tmp_path: Path) -> None:
    dataset = export_learned_policy_dataset(
        out_dir=tmp_path / "expert_dataset",
        planner_expert="orca_heuristic",
        lanes=["head_on"],
        max_steps=2,
        shard_size=16,
    )

    report = train_behavior_cloned_policy_from_dataset(
        out_dir=tmp_path / "expert_training",
        dataset_manifest=dataset["manifest"],
        hidden_dim=8,
        run_validation=False,
    )

    assert report["ok"] is True
    assert report["source_policy"] == "orca_heuristic"
    assert report["action_source"] == LEARNED_DATASET_PLANNER_EXPERT_SOURCE
    assert report["teacher_policy"] is None
    assert report["privileged_label_source"] is False
    model = json.loads(Path(report["model_artifact"]).read_text(encoding="utf-8"))
    assert model["training"]["source_policy"] == "orca_heuristic"
    assert model["training"]["action_source"] == LEARNED_DATASET_PLANNER_EXPERT_SOURCE
    assert model["training"]["privileged_label_source"] is False


def test_train_learned_bc_cli_can_train_from_dataset_manifest(tmp_path: Path) -> None:
    dataset = export_learned_policy_dataset(
        out_dir=tmp_path / "cli_expert_dataset",
        planner_expert="orca_heuristic",
        lanes=["head_on"],
        max_steps=2,
        shard_size=16,
    )
    out_dir = tmp_path / "cli_expert_training"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "train-learned-bc",
            "--out-dir",
            str(out_dir),
            "--dataset-manifest",
            dataset["manifest"],
            "--mlp-feature-set",
            MLP_LEARNED_PUBLIC_OBS_FEATURE_SET,
            "--hidden-dim",
            "8",
            "--skip-validation",
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
    assert report["source_policy"] == "orca_heuristic"
    assert report["action_source"] == LEARNED_DATASET_PLANNER_EXPERT_SOURCE
    assert report["feature_set"] == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET
    assert (out_dir / "policy_spec.json").exists()


def test_dataset_shard_training_can_use_public_obs_feature_set(tmp_path: Path) -> None:
    dataset = export_learned_policy_dataset(
        out_dir=tmp_path / "dataset",
        lanes=["head_on"],
        max_steps=2,
        shard_size=4,
    )

    report = train_behavior_cloned_policy_from_dataset(
        out_dir=tmp_path / "training_public_obs",
        dataset_manifest=dataset["manifest"],
        hidden_dim=8,
        feature_set=MLP_LEARNED_PUBLIC_OBS_FEATURE_SET,
        eval_lanes=["head_on"],
        eval_max_steps=2,
    )

    assert report["ok"] is True
    assert report["feature_set"] == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET
    assert report["feature_dim"] == len(MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES)
    model = json.loads(Path(report["model_artifact"]).read_text(encoding="utf-8"))
    assert model["model_id"] == MLP_LEARNED_PUBLIC_OBS_MODEL_ID
    assert model["feature_set"] == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET
    assert tuple(model["input_features"]) == MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES
    assert model["training"]["feature_set"] == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET


def test_dataset_shard_training_records_safety_sample_weighting(tmp_path: Path) -> None:
    dataset = export_learned_policy_dataset(
        out_dir=tmp_path / "dataset",
        lanes=["head_on"],
        max_steps=2,
        shard_size=4,
    )

    report = train_behavior_cloned_policy_from_dataset(
        out_dir=tmp_path / "training",
        dataset_manifest=dataset["manifest"],
        hidden_dim=8,
        eval_lanes=["head_on"],
        eval_max_steps=2,
        sample_weighting="safety",
    )

    assert report["ok"] is True
    assert report["sample_weighting"]["mode"] == "safety"
    assert report["sample_weighting"]["applied"] is True
    assert report["sample_weighting"]["sample_count"] == report["sample_count"]
    assert report["sample_weighting"]["weight_mean"] == 1.0
    assert next(check for check in report["checks"] if check["name"] == "sample_weights_finite")["ok"] is True
    model = json.loads(Path(report["model_artifact"]).read_text(encoding="utf-8"))
    assert model["training"]["sample_weighting"]["mode"] == "safety"


def test_dataset_shard_training_records_hard_negative_sample_selection(tmp_path: Path) -> None:
    dataset = export_learned_policy_dataset(
        out_dir=tmp_path / "dataset",
        lanes=["head_on", LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID],
        max_steps=3,
        shard_size=8,
    )

    report = train_behavior_cloned_policy_from_dataset(
        out_dir=tmp_path / "training",
        dataset_manifest=dataset["manifest"],
        hidden_dim=8,
        eval_lanes=["head_on"],
        eval_max_steps=2,
        sample_selection="hard_negative_windows",
        sample_selection_clearance_threshold_m=0.01,
        sample_selection_context_steps=0,
    )

    assert report["ok"] is True
    assert report["loaded_sample_count"] == dataset["sample_count"]
    assert report["sample_count"] < report["loaded_sample_count"]
    assert report["sample_selection"]["mode"] == "hard_negative_windows"
    assert report["sample_selection"]["closest_fallback_episode_count"] >= 1
    assert report["sample_selection"]["hard_lane_selected_count"] > 0
    assert next(check for check in report["checks"] if check["name"] == "samples_selected")["ok"] is True
    model = json.loads(Path(report["model_artifact"]).read_text(encoding="utf-8"))
    assert model["training"]["sample_selection"]["mode"] == "hard_negative_windows"


def test_learned_hard_lane_loop_smoke_without_fixtures(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics.json"
    diagnostics.write_text(
        json.dumps(
            _diagnostics(
                [
                    {
                        "diagnostic_rank": 1,
                        "policy": "candidate",
                        "diagnostic_label": "needs_training",
                        "primary_failure": "incomplete_missions",
                        "worst_rl_lane": "head_on",
                    }
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )

    report = run_learned_hard_lane_loop(
        out_dir=tmp_path / "loop",
        diagnostics=diagnostics,
        max_lanes=1,
        mix_lanes=["crossing", LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID],
        dataset_max_steps=2,
        dataset_shard_size=4,
        hidden_dim=8,
        sample_weighting="safety",
        eval_lanes=["head_on"],
        eval_max_steps=2,
        bundle_max_steps=2,
        bundle_max_runs=1,
        include_fixture_bundles=False,
    )

    assert report["ok"] is True
    assert report["selection"]["selected_lanes"] == ["head_on"]
    assert report["dataset_lanes"] == ["head_on", "crossing", LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID]
    assert report["mix_lanes"] == ["crossing", LEARNED_DENSE_SWARM_HARD_NEGATIVE_LANE_ID]
    assert report["dataset"]["sample_count"] == 44
    assert report["training"]["training_source"] == LEARNED_DATASET_BC_TRAINING_SOURCE
    assert report["training"]["feature_normalization"] == "standard"
    assert report["training"]["sample_selection"] == "all"
    assert report["training"]["sample_weighting"] == "safety"
    assert set(report["bundle_paths"]) == {"bc"}
    assert report["leaderboard"]["bundle_count"] == 1
    assert report["diagnostics"]["bundle_count"] == 1
    assert (tmp_path / "loop" / "learned_hard_lane_loop.json").exists()


def test_learned_hard_lane_loop_honors_dataset_seed_override(tmp_path: Path) -> None:
    report = run_learned_hard_lane_loop(
        out_dir=tmp_path / "seeded_loop",
        fallback_lanes=["head_on"],
        max_lanes=1,
        dataset_seeds=[0, 1],
        dataset_max_steps=1,
        dataset_shard_size=4,
        hidden_dim=8,
        eval_lanes=["head_on"],
        eval_max_steps=1,
        bundle_max_steps=1,
        bundle_max_runs=1,
        include_fixture_bundles=False,
    )

    assert report["ok"] is True
    assert report["selection"]["selected_lanes"] == ["head_on"]
    assert report["dataset_lanes"] == ["head_on"]
    assert report["dataset_seeds"] == [0, 1]
    assert report["dataset"]["planned_episode_count"] == 2
    assert report["dataset"]["episode_count"] == 2
    assert report["dataset"]["sample_count"] == 8


def test_learned_hard_lane_loop_cli_smoke(tmp_path: Path) -> None:
    diagnostics = tmp_path / "diagnostics_cli.json"
    diagnostics.write_text(
        json.dumps(
            _diagnostics(
                [
                    {
                        "diagnostic_rank": 1,
                        "policy": "candidate",
                        "diagnostic_label": "unsafe",
                        "primary_failure": "collision",
                        "worst_rl_lane": "head_on",
                    }
                ]
            )
        )
        + "\n",
        encoding="utf-8",
    )
    out_dir = tmp_path / "loop_cli"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "learned-hard-lane-loop",
            "--out-dir",
            str(out_dir),
            "--diagnostics",
            str(diagnostics),
            "--max-lanes",
            "1",
            "--mix-lanes",
            "crossing",
            "--dataset-max-steps",
            "2",
            "--dataset-seeds",
            "0:1",
            "--dataset-shard-size",
            "4",
            "--hidden-dim",
            "8",
            "--sample-weighting",
            "safety",
            "--sample-selection",
            "hard_negative_windows",
            "--eval-lanes",
            "head_on",
            "--eval-max-steps",
            "2",
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
    assert report["selection"]["selected_lanes"] == ["head_on"]
    assert report["dataset_lanes"] == ["head_on", "crossing"]
    assert report["dataset_seeds"] == [0, 1]
    assert report["dataset"]["planned_episode_count"] == 4
    assert report["dataset"]["sample_count"] == 40
    assert report["training"]["sample_selection"] == "hard_negative_windows"
    assert report["training"]["sample_weighting"] == "safety"
    assert (out_dir / "learned_hard_lane_loop.json").exists()
    assert (out_dir / "learned_policy_leaderboard.csv").exists()
