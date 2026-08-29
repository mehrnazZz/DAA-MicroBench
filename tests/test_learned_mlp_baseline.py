from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from microbench.planners import make_planner, planner_metadata
from microbench.rl import (
    FrozenMlpPolicyModel,
    MLP_LEARNED_FEATURE_NAMES,
    MLP_LEARNED_MODEL_ID,
    MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES,
    MLP_LEARNED_PUBLIC_OBS_FEATURE_SET,
    MLP_LEARNED_PUBLIC_OBS_MODEL_ID,
    load_mlp_learned_spec,
    mlp_feature_names,
    observation_to_mlp_features,
    planner_input_to_mlp_features,
    run_rl_policy_smoke,
    run_rl_validation_matrix,
)
from microbench.runner import run_episode
from microbench.scenarios import materialize_official_suite
from microbench.tools.baseline_behavior import run_baseline_behavior_smoke
from microbench.types import AgentState, NeighborObs, PlannerInput, PlannerOutput, RunSpec


ROOT = Path(__file__).resolve().parents[1]


def _planner_input(*, neighbors: list[NeighborObs] | None = None, planar: bool = False) -> PlannerInput:
    ego = AgentState(
        idx=0,
        pos=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([0.0, 0.0, 0.0], dtype=np.float32),
        goal=np.asarray([10.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        v_max=3.0,
        a_max=2.0,
    )
    return PlannerInput(
        ego=ego,
        goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        neighbors=list(neighbors or []),
        dt=0.02,
        t=0.0,
        planar=planar,
    )


def _head_on_neighbor() -> NeighborObs:
    return NeighborObs(
        idx=1,
        pos=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        vel=np.asarray([-2.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        msg_age_sec=0.0,
        valid=True,
    )


def test_mlp_learned_model_artifact_loads_and_predicts() -> None:
    spec = load_mlp_learned_spec()
    assert spec["model_id"] == MLP_LEARNED_MODEL_ID
    assert spec["model_type"] == "mlp_tanh_policy"
    assert tuple(spec["input_features"]) == MLP_LEARNED_FEATURE_NAMES

    model = FrozenMlpPolicyModel.from_path()
    features = np.zeros(len(MLP_LEARNED_FEATURE_NAMES), dtype=np.float32)
    features[0] = 1.0
    action = model.action_from_features(features)

    assert model.hidden_dim == 24
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))
    assert float(np.linalg.norm(action)) <= np.sqrt(3.0) + 1e-6


def test_mlp_public_obs_feature_set_matches_rl_observation_contract() -> None:
    obs = np.zeros(len(MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES), dtype=np.float32)
    obs[0:3] = np.asarray([1.0, 2.0, 3.0], dtype=np.float32)
    obs[6:9] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    obs[9] = 10.0
    obs[12] = 0.25
    obs[13] = 2.0
    obs[14] = 0.5
    obs[15] = 3.0
    obs[16] = 2.0
    obs[17] = 1.0
    obs[18:21] = np.asarray([2.0, -1.0, 0.5], dtype=np.float32)
    obs[21:24] = np.asarray([-0.5, 0.0, 0.25], dtype=np.float32)
    obs[24] = 0.4
    obs[25] = 0.1

    features = observation_to_mlp_features(obs, feature_set=MLP_LEARNED_PUBLIC_OBS_FEATURE_SET)

    assert features.shape == (len(MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES),)
    assert tuple(MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES) == mlp_feature_names(MLP_LEARNED_PUBLIC_OBS_FEATURE_SET)
    assert MLP_LEARNED_PUBLIC_OBS_MODEL_ID != MLP_LEARNED_MODEL_ID
    assert np.allclose(features, obs)
    assert features[17] == 1.0
    assert np.allclose(features[18:21], [2.0, -1.0, 0.5])

    planner_features = planner_input_to_mlp_features(
        _planner_input(neighbors=[_head_on_neighbor()]),
        feature_set=MLP_LEARNED_PUBLIC_OBS_FEATURE_SET,
    )
    assert planner_features.shape == (len(MLP_LEARNED_PUBLIC_OBS_FEATURE_NAMES),)
    assert planner_features[17] == 1.0
    assert np.allclose(planner_features[18:21], [1.0, 0.0, 0.0])


def test_learned_mlp_planner_uses_public_neighbor_features() -> None:
    planner = make_planner("learned_mlp")
    planner.reset(123)

    free = planner.compute_cmd(_planner_input())
    threatened = planner.compute_cmd(_planner_input(neighbors=[_head_on_neighbor()]))
    assert isinstance(free, PlannerOutput)
    assert isinstance(threatened, PlannerOutput)

    assert np.asarray(free.v_cmd).shape == (3,)
    assert np.asarray(threatened.v_cmd).shape == (3,)
    assert np.all(np.isfinite(threatened.v_cmd))
    assert float(np.linalg.norm(threatened.v_cmd)) <= 3.0 + 1e-6
    assert not np.allclose(free.v_cmd, threatened.v_cmd)
    assert threatened.debug_info["learned_model"] is True
    assert threatened.debug_info["learned_model_id"] == MLP_LEARNED_MODEL_ID
    assert threatened.debug_info["learned_policy_architecture"] == "mlp_tanh"
    assert int(threatened.debug_info["learned_policy_hidden_dim"]) == 24
    assert float(threatened.debug_info["learned_policy_threat_scalar"]) > 0.0


def test_learned_mlp_registry_and_baseline_behavior_contract(tmp_path: Path) -> None:
    by_method = {entry["method"]: entry for entry in planner_metadata(include_aliases=False)}
    assert by_method["learned_mlp"]["learned"] is True
    assert by_method["learned_mlp"]["role"] == "experimental_baseline"
    assert by_method["learned_mlp"]["status"] == "experimental"
    assert by_method["learned_mlp"]["dimensions"] == ("2d", "3d")

    report = run_baseline_behavior_smoke(out_dir=tmp_path / "baseline_smoke", methods=("learned_mlp",))
    check = next(item for item in report["checks"] if item["name"] == "learned_mlp_model_contract")
    assert report["ok"] is True
    assert report["run_count"] == 2
    assert check["ok"] is True


def test_learned_mlp_runs_as_official_planner_and_rl_policy(tmp_path: Path) -> None:
    generated = materialize_official_suite("official_smoke_generated", tmp_path / "suite", overwrite=True)
    scenario = next(path for path in generated["scenario_paths"] if path.stem == "sphere_swap_3d_medium")
    row = run_episode(
        RunSpec(
            scenario_path=str(scenario),
            method="learned_mlp",
            n_agents=4,
            seed=0,
            comm_profile="ideal_50hz",
            out_dir=str(tmp_path / "planner_run"),
            save_trace=False,
        )
    )

    assert row["method"] == "learned_mlp"
    assert int(row["planner_error_count"]) == 0
    assert int(row["planner_timeout_count"]) == 0
    assert int(row["planner_fallback_count"]) == 0

    rl_report = run_rl_policy_smoke(
        out_dir=tmp_path / "rl_smoke",
        policy="mlp_learned",
        max_steps=3,
    )
    assert rl_report["ok"] is True
    assert rl_report["policy"] == "mlp_learned"

    matrix = run_rl_validation_matrix(
        out_dir=tmp_path / "rl_validation_matrix",
        policy="mlp_learned",
        lanes=["head_on"],
        duration_s=1.0,
        max_steps=3,
    )
    assert matrix["ok"] is True
    assert matrix["policy"] == "mlp_learned"


def test_mlp_training_script_writes_compatible_spec(tmp_path: Path) -> None:
    out = tmp_path / "mlp_policy.json"
    proc = subprocess.run(
        [
            sys.executable,
            "examples/rl_train_mlp_policy.py",
            "--out",
            str(out),
            "--samples",
            "128",
            "--hidden-dim",
            "8",
            "--seed",
            "7",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["feature_dim"] == len(MLP_LEARNED_FEATURE_NAMES)
    assert payload["hidden_dim"] == 8
    model = FrozenMlpPolicyModel.from_path(out)
    action = model.action_from_features(np.zeros(len(MLP_LEARNED_FEATURE_NAMES), dtype=np.float32))
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))
