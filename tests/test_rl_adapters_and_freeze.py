from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from microbench.rl import CallablePolicyAdapter, ModelPredictPolicyAdapter, normalize_action, run_rl_freeze_check
from microbench.rl.schema import OBS_GOAL_DIR_SLICE
from microbench.rl.spaces import box


ROOT = Path(__file__).resolve().parents[1]


def test_callable_policy_adapter_validates_and_clips_actions() -> None:
    action_space = box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
    obs = np.zeros(32, dtype=np.float32)
    adapter = CallablePolicyAdapter(lambda observation: [2.0, 0.25, -2.0], signature="observation")

    action = adapter.action("agent_0", obs, action_space, {})

    assert action.dtype == np.float32
    assert np.allclose(action, np.asarray([1.0, 0.25, -1.0], dtype=np.float32))
    assert action_space.contains(action)


def test_model_predict_policy_adapter_accepts_tuple_predict_output() -> None:
    class TinyModel:
        def __init__(self) -> None:
            self.seed = None

        def set_seed(self, seed: int) -> None:
            self.seed = seed

        def predict(self, observation: np.ndarray, deterministic: bool = True):
            assert deterministic is True
            return observation[OBS_GOAL_DIR_SLICE], None

    obs = np.zeros(32, dtype=np.float32)
    obs[OBS_GOAL_DIR_SLICE] = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
    model = TinyModel()
    adapter = ModelPredictPolicyAdapter(model)
    adapter.reset(42)

    action = adapter.action("agent_0", obs, box(low=-1.0, high=1.0, shape=(3,)), {})

    assert model.seed == 42
    assert np.allclose(action, np.asarray([0.0, 0.0, 1.0], dtype=np.float32))


def test_normalize_action_rejects_bad_external_outputs() -> None:
    with pytest.raises(ValueError, match="shape"):
        normalize_action([0.0, 1.0])
    with pytest.raises(ValueError, match="finite"):
        normalize_action([0.0, float("nan"), 0.0])


def test_rl_freeze_check_cli_and_schema(tmp_path: Path) -> None:
    report = run_rl_freeze_check(root=ROOT)
    assert report["schema_version"] == "0.1"
    assert report["ok"] is True
    assert {check["name"] for check in report["checks"]} >= {
        "versioned_contract",
        "action_contract_frozen",
        "observation_contract_frozen",
        "adapter_example_available",
    }

    out = tmp_path / "rl_freeze_check.json"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "microbench.cli",
            "rl-freeze-check",
            "--root",
            str(ROOT),
            "--out",
            str(out),
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
    assert out.exists()


def test_external_policy_adapter_example_runs(tmp_path: Path) -> None:
    proc = subprocess.run(
        [
            sys.executable,
            "examples/rl_external_policy_adapter.py",
            "--out-dir",
            str(tmp_path / "external_policy_example"),
            "--max-steps",
            "3",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    report = json.loads(proc.stdout)
    assert report["ok"] is True
    assert report["policy"] == "tiny_goal_model_adapter"
    assert report["dimension"] == "3d"
    assert report["steps"] == 3
