from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from microbench.rl import (
    OBSERVATION_LAYOUT,
    RL_INTERFACE_VERSION,
    GoalDirectionPolicy,
    RandomPolicy,
    interface_contract,
    run_rl_policy_smoke,
)


ROOT = Path(__file__).resolve().parents[1]


def _check(report: dict, name: str) -> dict:
    return next(check for check in report["checks"] if check["name"] == name)


def test_rl_policy_smoke_runs_2d_and_3d(tmp_path: Path) -> None:
    report = run_rl_policy_smoke(
        out_dir=tmp_path / "rl_smoke",
        policy="goal_direction",
        max_steps=5,
    )

    assert report["schema_version"] == "0.1"
    assert report["interface_version"] == RL_INTERFACE_VERSION
    assert report["observation_schema_version"] == "0.1.0"
    assert report["ok"] is True
    assert report["run_count"] == 2
    assert report["scenario_ids"] == ["head_on_2d_easy", "sphere_swap_3d_medium"]
    assert set(report["dimensions"]) == {"2d", "3d"}
    assert _check(report, "finite_rollout_metrics")["ok"] is True
    assert _check(report, "two_d_and_three_d_coverage")["ok"] is True
    assert Path(report["episode_csv"]).exists()
    assert Path(report["suite_manifest"]).exists()
    assert report["interface_contract"]["observation"]["shape"] == [89]


def test_rl_smoke_cli_json_and_gate(tmp_path: Path) -> None:
    out_dir = tmp_path / "cli_rl_smoke"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "rl-smoke",
            "--out-dir",
            str(out_dir),
            "--policy",
            "zero",
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

    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["policy"] == "zero"
    assert report["run_count"] == 2
    assert (out_dir / "rl_smoke.json").exists()
    assert (out_dir / "rl_smoke_episodes.csv").exists()


def test_rl_contract_cli_json_and_schema_helper(tmp_path: Path) -> None:
    contract = interface_contract(top_k=3)
    assert contract["interface_version"] == RL_INTERFACE_VERSION
    assert contract["observation"]["shape"] == [44]
    assert contract["reward"]["weights"]["collision"] < 0

    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "rl-contract",
            "--top-k",
            "3",
            "--json",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    cli_contract = json.loads(proc.stdout)
    assert cli_contract["interface_version"] == RL_INTERFACE_VERSION
    assert cli_contract["observation"]["shape"] == [44]


def test_rl_policy_helpers_are_deterministic_and_layout_is_documented() -> None:
    assert OBSERVATION_LAYOUT["goal_dir"] == (6, 9)
    obs = np.zeros(32, dtype=np.float32)
    obs[6:9] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)

    goal_policy = GoalDirectionPolicy(speed_fraction=0.5)
    assert np.allclose(goal_policy.action("agent_0", obs, None, {}), np.asarray([0.0, 0.0, 0.5]))

    p1 = RandomPolicy()
    p2 = RandomPolicy()
    p1.reset(7)
    p2.reset(7)
    assert np.allclose(p1.action("agent_0", obs, None, {}), p2.action("agent_0", obs, None, {}))
