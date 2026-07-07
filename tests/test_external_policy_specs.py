from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from microbench.rl import load_policy_from_spec, run_rl_policy_smoke
from microbench.rl.schema import OBS_GOAL_DIR_SLICE
from microbench.runner import run_episode
from microbench.scenarios import materialize_official_suite
from microbench.types import RunSpec


ROOT = Path(__file__).resolve().parents[1]
MODEL_SPEC = ROOT / "examples" / "external_policy_model_predict_spec.json"
CALLABLE_SPEC = ROOT / "examples" / "external_policy_callable_spec.json"


def _observation() -> np.ndarray:
    obs = np.zeros(89, dtype=np.float32)
    obs[OBS_GOAL_DIR_SLICE] = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    return obs


def test_model_predict_and_callable_policy_specs_load_and_act() -> None:
    for spec_path, expected_name in (
        (MODEL_SPEC, "external_model_predict_fixture"),
        (CALLABLE_SPEC, "external_callable_fixture"),
    ):
        loaded = load_policy_from_spec(spec_path, seed=11)
        action = loaded.policy.action("agent_0", _observation(), None, {"reset": False})

        assert loaded.policy_name == expected_name
        assert action.shape == (3,)
        assert np.all(np.isfinite(action))
        assert 0.0 < float(action[0]) <= 1.0
        assert loaded.summary["adapter"] in {"model_predict", "callable"}


def test_model_predict_policy_spec_rl_smoke_and_cli(tmp_path: Path) -> None:
    report = run_rl_policy_smoke(
        out_dir=tmp_path / "model_predict_rl_smoke",
        policy_spec=MODEL_SPEC,
        max_steps=3,
    )

    assert report["ok"] is True
    assert report["policy"] == "external_model_predict_fixture"
    assert report["policy_spec"]["factory"] == "exported_policy:make_model"
    assert report["run_count"] == 2

    out_dir = tmp_path / "cli_model_predict_rl_smoke"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "rl-smoke",
            "--out-dir",
            str(out_dir),
            "--policy-spec",
            str(MODEL_SPEC),
            "--max-steps",
            "3",
            "--require-pass",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    cli_report = json.loads(proc.stdout)
    assert cli_report["ok"] is True
    assert cli_report["policy"] == "external_model_predict_fixture"
    assert (out_dir / "rl_smoke_episodes.csv").exists()


def test_model_predict_policy_spec_runs_through_planner_bridge(tmp_path: Path) -> None:
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
            policy_spec=str(MODEL_SPEC),
        )
    )

    assert row["method"] == "learned_policy_spec"
    assert int(row["planner_error_count"]) == 0
    assert int(row["planner_timeout_count"]) == 0
    assert int(row["planner_fallback_count"]) == 0
    assert np.isfinite(float(row["planner_ms_per_tick_per_agent_p95"]))
