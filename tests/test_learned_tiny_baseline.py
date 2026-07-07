from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from microbench.planners import make_planner, planner_metadata
from microbench.rl import (
    TINY_LEARNED_FEATURE_NAMES,
    TINY_LEARNED_MODEL_ID,
    TinyLinearPolicyModel,
    load_tiny_learned_spec,
    run_rl_policy_smoke,
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


def test_tiny_learned_model_artifact_loads_and_predicts() -> None:
    spec = load_tiny_learned_spec()
    assert spec["model_id"] == TINY_LEARNED_MODEL_ID
    assert tuple(spec["input_features"]) == TINY_LEARNED_FEATURE_NAMES

    model = TinyLinearPolicyModel.from_path()
    features = np.zeros(len(TINY_LEARNED_FEATURE_NAMES), dtype=np.float32)
    features[0] = 1.0
    action = model.action_from_features(features)

    assert action.shape == (3,)
    assert np.all(np.isfinite(action))
    assert 0.0 < float(action[0]) <= 1.0


def test_learned_tiny_planner_uses_public_neighbor_features() -> None:
    planner = make_planner("learned_tiny")
    planner.reset(123)

    free = planner.compute_cmd(_planner_input())
    threatened = planner.compute_cmd(_planner_input(neighbors=[_head_on_neighbor()]))
    assert isinstance(free, PlannerOutput)
    assert isinstance(threatened, PlannerOutput)

    assert np.asarray(free.v_cmd).shape == (3,)
    assert np.asarray(threatened.v_cmd).shape == (3,)
    assert np.all(np.isfinite(threatened.v_cmd))
    assert float(np.linalg.norm(threatened.v_cmd)) <= 3.0 + 1e-6
    assert threatened.debug_info["learned_model"] is True
    assert threatened.debug_info["learned_model_id"] == TINY_LEARNED_MODEL_ID
    assert float(threatened.debug_info["learned_policy_threat_scalar"]) > 0.0
    assert float(threatened.v_cmd[0]) < float(free.v_cmd[0])


def test_learned_tiny_registry_and_baseline_behavior_contract(tmp_path: Path) -> None:
    by_method = {entry["method"]: entry for entry in planner_metadata(include_aliases=False)}
    assert by_method["learned_tiny"]["learned"] is True
    assert by_method["learned_tiny"]["role"] == "experimental_baseline"
    assert by_method["learned_tiny"]["status"] == "experimental"
    assert by_method["learned_tiny"]["dimensions"] == ("2d", "3d")

    report = run_baseline_behavior_smoke(out_dir=tmp_path / "baseline_smoke", methods=("learned_tiny",))
    check = next(item for item in report["checks"] if item["name"] == "learned_tiny_model_contract")
    assert report["ok"] is True
    assert report["run_count"] == 2
    assert check["ok"] is True


def test_learned_tiny_runs_as_official_planner_and_rl_policy(tmp_path: Path) -> None:
    generated = materialize_official_suite("official_smoke_generated", tmp_path / "suite", overwrite=True)
    scenario = next(path for path in generated["scenario_paths"] if path.stem == "sphere_swap_3d_medium")
    row = run_episode(
        RunSpec(
            scenario_path=str(scenario),
            method="learned_tiny",
            n_agents=4,
            seed=0,
            comm_profile="ideal_50hz",
            out_dir=str(tmp_path / "planner_run"),
            save_trace=False,
        )
    )

    assert row["method"] == "learned_tiny"
    assert int(row["planner_error_count"]) == 0
    assert int(row["planner_timeout_count"]) == 0
    assert int(row["planner_fallback_count"]) == 0

    rl_report = run_rl_policy_smoke(
        out_dir=tmp_path / "rl_smoke",
        policy="tiny_learned",
        max_steps=3,
    )
    assert rl_report["ok"] is True
    assert rl_report["policy"] == "tiny_learned"


def test_tiny_learned_training_script_writes_compatible_spec(tmp_path: Path) -> None:
    out = tmp_path / "tiny_linear_policy.json"
    proc = subprocess.run(
        [
            sys.executable,
            "examples/rl_train_tiny_linear_policy.py",
            "--out",
            str(out),
            "--samples",
            "64",
            "--seed",
            "5",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    payload = json.loads(proc.stdout)
    assert payload["feature_dim"] == len(TINY_LEARNED_FEATURE_NAMES)
    model = TinyLinearPolicyModel.from_path(out)
    action = model.action_from_features(np.zeros(len(TINY_LEARNED_FEATURE_NAMES), dtype=np.float32))
    assert action.shape == (3,)
