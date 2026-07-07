from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from microbench.learned import tiny_learned_model_path
from microbench.learned.rl_bridge import planner_input_to_rl_observation
from microbench.rl.policy_spec import RL_POLICY_SPEC_SCHEMA_VERSION
from microbench.runner import run_episode
from microbench.scenarios import materialize_official_suite
from microbench.types import AgentContext, AgentState, NeighborObs, PlannerInput, RunSpec


ROOT = Path(__file__).resolve().parents[1]


def _tiny_policy_spec(tmp_path: Path) -> Path:
    path = tmp_path / "external_policy_spec.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": RL_POLICY_SPEC_SCHEMA_VERSION,
                "policy_name": "external_tiny_fixture",
                "adapter": "tiny_linear_json",
                "artifact_path": tiny_learned_model_path(),
                "deterministic": True,
                "clip": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return path


def _planner_input() -> PlannerInput:
    ego = AgentState(
        idx=2,
        pos=np.asarray([0.0, 1.0, 0.0], dtype=np.float32),
        vel=np.asarray([0.1, 0.0, 0.0], dtype=np.float32),
        goal=np.asarray([10.0, 1.0, 0.0], dtype=np.float32),
        radius=0.5,
        v_max=3.0,
        a_max=2.0,
    )
    neighbor = NeighborObs(
        idx=1,
        pos=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        vel=np.asarray([-1.0, 0.0, 0.0], dtype=np.float32),
        radius=0.5,
        msg_age_sec=0.04,
        valid=True,
    )
    return PlannerInput(
        ego=ego,
        goal_dir=np.asarray([1.0, 0.0, 0.0], dtype=np.float32),
        neighbors=[neighbor],
        dt=0.02,
        t=0.1,
        agent_context=AgentContext(agent_id=2, method="learned_policy_spec", seed=0, priority=7),
        planar=False,
    )


def test_policy_spec_planner_uses_rl_observation_contract(tmp_path: Path) -> None:
    from microbench.planners import make_planner

    spec_path = _tiny_policy_spec(tmp_path)
    planner = make_planner("learned_policy_spec", policy_spec=spec_path)
    planner.reset(seed=123, agent_id=2, config={"n_agents": 4, "neighbor_top_k": 3})

    planner_input = _planner_input()
    observation = planner_input_to_rl_observation(planner_input, top_k=3, n_agents=4)
    assert observation.shape == (44,)
    assert observation[12] == np.float32(2 / 3)
    assert observation[13] == np.float32(7)

    out = planner.compute_cmd(planner_input)
    assert out.v_cmd.shape == (3,)
    assert np.all(np.isfinite(out.v_cmd))
    assert float(np.linalg.norm(out.v_cmd)) <= planner_input.ego.v_max + 1e-6
    assert out.debug_info["learned_policy_spec"] is True
    assert out.debug_info["learned_policy_name"] == "external_tiny_fixture"
    assert out.debug_info["learned_policy_observation_dim"] == 44


def test_policy_spec_planner_runs_as_official_benchmark_method(tmp_path: Path) -> None:
    spec_path = _tiny_policy_spec(tmp_path)
    generated = materialize_official_suite("official_smoke_generated", tmp_path / "suite", overwrite=True)
    scenario = next(path for path in generated["scenario_paths"] if path.stem == "sphere_swap_3d_medium")

    row = run_episode(
        RunSpec(
            scenario_path=str(scenario),
            method="learned_policy_spec",
            n_agents=4,
            seed=0,
            comm_profile="ideal_50hz",
            out_dir=str(tmp_path / "planner_run"),
            save_trace=False,
            policy_spec=str(spec_path),
        )
    )

    assert row["method"] == "learned_policy_spec"
    assert int(row["planner_error_count"]) == 0
    assert int(row["planner_timeout_count"]) == 0
    assert int(row["planner_fallback_count"]) == 0
    assert np.isfinite(float(row["min_sep_min_m"]))


def test_policy_spec_cli_run_writes_standard_results_csv(tmp_path: Path) -> None:
    spec_path = _tiny_policy_spec(tmp_path)
    generated = materialize_official_suite("official_smoke_generated", tmp_path / "suite", overwrite=True)
    scenario = next(path for path in generated["scenario_paths"] if path.stem == "head_on_2d_easy")
    out_dir = tmp_path / "cli_policy_spec_run"

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "run",
            "--scenario",
            str(scenario),
            "--method",
            "learned_policy_spec",
            "--policy-spec",
            str(spec_path),
            "--n",
            "4",
            "--seed",
            "0",
            "--comm",
            "ideal_50hz",
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "method=learned_policy_spec" in proc.stdout
    assert (out_dir / "results.csv").exists()
    assert (out_dir / "summary.csv").exists()
