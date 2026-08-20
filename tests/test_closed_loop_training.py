from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np

from microbench.learned import MLP_LEARNED_MODEL_ID
from microbench.rl import CLOSED_LOOP_TRAINING_SCHEMA_VERSION, fine_tune_closed_loop_policy, load_policy_from_spec, train_behavior_cloned_policy
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

    model = json.loads(Path(report["model_artifact"]).read_text(encoding="utf-8"))
    assert model["model_id"] == MLP_LEARNED_MODEL_ID
    assert model["training"]["recipe"] == "python -m microbench.cli learned-closed-loop-finetune"
    assert model["training"]["public_observations_only"] is True
    assert model["training"]["privileged_global_state"] is False
    assert model["training"]["base_policy_spec"] == str(base_spec)

    loaded = load_policy_from_spec(report["policy_spec"], seed=5)
    action = loaded.policy.action("agent_0", _observation(), None, {})
    assert loaded.summary["adapter"] == "mlp_json"
    assert loaded.policy_name == "closed_loop_mlp_learned"
    assert action.shape == (3,)
    assert np.all(np.isfinite(action))


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
