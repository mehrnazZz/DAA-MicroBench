from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from microbench.learned import (
    TEMPORAL_MLP_LEARNED_MODEL_ID,
    TEMPORAL_MLP_POLICY_ADAPTER,
    MLP_LEARNED_MODEL_ID,
    MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES,
    MLP_LEARNED_PUBLIC_OBS_FEATURE_SET,
    MLP_LEARNED_PUBLIC_OBS_MODEL_ID,
    stack_temporal_feature_rows,
    temporal_mlp_feature_names,
)
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


def test_temporal_feature_stacking_uses_agent_episode_history() -> None:
    features = np.asarray(
        [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
        ],
        dtype=np.float32,
    )
    stacked = stack_temporal_feature_rows(
        features,
        episode_ids=np.asarray([0, 0, 0, 0], dtype=np.int32),
        agent_ids=np.asarray([0, 1, 0, 0], dtype=np.int32),
        steps=np.asarray([0, 0, 1, 2], dtype=np.int32),
        history_len=3,
    )

    assert stacked.shape == (4, 6)
    assert stacked[0].tolist() == [1.0, 10.0, 1.0, 10.0, 1.0, 10.0]
    assert stacked[1].tolist() == [2.0, 20.0, 2.0, 20.0, 2.0, 20.0]
    assert stacked[2].tolist() == [3.0, 30.0, 1.0, 10.0, 3.0, 30.0]
    assert stacked[3].tolist() == [4.0, 40.0, 3.0, 30.0, 1.0, 10.0]


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
    assert report["feature_normalization"]["mode"] == "standard"
    assert report["validation_matrix"]["ok"] is True

    model_payload = json.loads(Path(report["model_artifact"]).read_text(encoding="utf-8"))
    assert model_payload["model_id"] == MLP_LEARNED_MODEL_ID
    assert model_payload["feature_normalization"]["mode"] == "standard"
    assert len(model_payload["feature_normalization"]["mean"]) == report["feature_dim"]
    assert model_payload["postprocess"]["goal_forward_floor"] is True
    assert model_payload["training"]["source"] == "DAA Microbench RL validation-matrix rollouts"
    assert model_payload["training"]["observation_normalization"].startswith("per-feature mean/std")
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
            "--feature-normalization",
            "none",
            "--mlp-feature-set",
            MLP_LEARNED_PUBLIC_OBS_FEATURE_SET,
            "--model-architecture",
            "temporal_mlp",
            "--history-len",
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
    assert report["sample_count"] == 8
    assert report["feature_set"] == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET
    assert report["model_architecture"] == "temporal_mlp"
    assert report["history_len"] == 2
    assert report["feature_dim"] == len(temporal_mlp_feature_names(history_len=2))
    assert report["feature_normalization"]["mode"] == "none"
    assert Path(report["policy_spec"]).exists()
    assert Path(report["model_artifact"]).exists()
    spec = json.loads((out_dir / "policy_spec.json").read_text(encoding="utf-8"))
    assert spec["adapter"] == TEMPORAL_MLP_POLICY_ADAPTER
    assert (out_dir / "bc_training_report.json").exists()


def test_behavior_cloned_training_can_use_public_obs_mlp_feature_set(tmp_path: Path) -> None:
    report = train_behavior_cloned_policy(
        out_dir=tmp_path / "bc_train_public_obs",
        lanes=["head_on"],
        max_steps=2,
        hidden_dim=8,
        rollout_noise_std=0.0,
        feature_set=MLP_LEARNED_PUBLIC_OBS_FEATURE_SET,
        eval_lanes=["head_on"],
        eval_max_steps=2,
    )

    assert report["ok"] is True
    assert report["feature_set"] == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET
    assert report["feature_dim"] == len(MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES)
    assert report["validation_matrix"]["ok"] is True

    model_payload = json.loads(Path(report["model_artifact"]).read_text(encoding="utf-8"))
    assert model_payload["model_id"] == MLP_LEARNED_PUBLIC_OBS_MODEL_ID
    assert model_payload["feature_set"] == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET
    assert model_payload["feature_top_k"] == 8
    assert tuple(model_payload["input_features"]) == MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES
    assert model_payload["training"]["feature_set"] == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET
    assert model_payload["training"]["feature_dim"] == len(MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES)

    loaded = load_policy_from_spec(report["policy_spec"], seed=3)
    action = loaded.policy.action("agent_0", _observation(), None, {})
    assert loaded.summary["adapter"] == "mlp_json"
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))


def test_behavior_cloned_training_can_write_temporal_public_obs_policy(tmp_path: Path) -> None:
    report = train_behavior_cloned_policy(
        out_dir=tmp_path / "bc_train_temporal",
        lanes=["head_on"],
        max_steps=2,
        hidden_dim=8,
        rollout_noise_std=0.0,
        feature_set=MLP_LEARNED_PUBLIC_OBS_FEATURE_SET,
        model_architecture="temporal_mlp",
        history_len=3,
        eval_lanes=["head_on"],
        eval_max_steps=2,
    )

    assert report["ok"] is True
    assert report["feature_set"] == MLP_LEARNED_PUBLIC_OBS_FEATURE_SET
    assert report["model_architecture"] == "temporal_mlp"
    assert report["history_len"] == 3
    assert report["feature_dim"] == len(temporal_mlp_feature_names(history_len=3))
    assert report["validation_matrix"]["ok"] is True

    spec = json.loads(Path(report["policy_spec"]).read_text(encoding="utf-8"))
    model_payload = json.loads(Path(report["model_artifact"]).read_text(encoding="utf-8"))
    assert spec["adapter"] == TEMPORAL_MLP_POLICY_ADAPTER
    assert model_payload["model_id"] == TEMPORAL_MLP_LEARNED_MODEL_ID
    assert model_payload["history_len"] == 3
    assert tuple(model_payload["input_features"]) == temporal_mlp_feature_names(history_len=3)

    loaded = load_policy_from_spec(report["policy_spec"], seed=3)
    action0 = loaded.policy.action("agent_0", _observation(), None, {})
    action1 = loaded.policy.action("agent_0", _observation(), None, {})
    assert loaded.summary["adapter"] == TEMPORAL_MLP_POLICY_ADAPTER
    assert action0.shape == (3,)
    assert action1.shape == (3,)
    assert np.all(np.isfinite(action0))
    assert np.all(np.isfinite(action1))


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
    assert report["training"]["feature_normalization"] == "standard"
    assert report["bundles"]["bc"]["policy"] == "bc_mlp_learned"
    assert report["leaderboard"]["ok"] is True
    assert report["leaderboard"]["bundle_count"] == 3
    assert report["diagnostics"]["ok"] is True
    assert report["diagnostics"]["bundle_count"] == 3
    assert {row["policy"] for row in report["diagnostics"]["rows"]} == {
        "bc_mlp_learned",
        "tiny_learned",
        "mlp_learned",
    }
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
    assert (out_dir / "learned_policy_diagnostics.json").exists()
    assert (out_dir / "learned_policy_diagnostics.csv").exists()
    assert (out_dir / "learned_policy_diagnostics.md").exists()
