from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microbench.rl import DaaParallelEnv, ModelPredictPolicyAdapter, rollout_parallel_env
from microbench.rl.schema import OBS_GOAL_DIR_SLICE
from microbench.scenarios import materialize_official_suite


class TinyGoalModel:
    """Dependency-free stand-in for a trained policy object.

    Real adapters can swap this class for an object from Stable-Baselines,
    RLlib, CleanRL, TorchRL, or a custom inference wrapper. The only contract is
    that model inference returns a finite normalized action with shape `(3,)`.
    """

    def predict(self, observation: np.ndarray, deterministic: bool = True):
        _ = deterministic
        return np.asarray(observation[OBS_GOAL_DIR_SLICE], dtype=np.float32), None


def _scenario_path(out_dir: Path, scenario_id: str) -> Path:
    generated = materialize_official_suite("official_smoke_generated", out_dir / "_generated_scenarios", overwrite=True)
    by_id = {path.stem: path for path in generated["scenario_paths"]}
    return by_id[scenario_id]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a dependency-free learned-policy adapter example.")
    parser.add_argument("--out-dir", default="runs_rl_external_policy_example")
    parser.add_argument("--scenario", default="sphere_swap_3d_medium")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--comm", default="ideal_50hz")
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    policy = ModelPredictPolicyAdapter(TinyGoalModel())
    scenario_path = _scenario_path(out_dir, str(args.scenario))
    env = DaaParallelEnv(
        scenario_path=str(scenario_path),
        n_agents=int(args.n),
        seed=int(args.seed),
        comm_profile=str(args.comm),
    )
    try:
        row = rollout_parallel_env(
            env,
            policy,
            seed=int(args.seed),
            max_steps=int(args.max_steps),
            metadata={
                "suite": "official_smoke_generated",
                "scenario": str(args.scenario),
                "policy": "tiny_goal_model_adapter",
                "n_agents": int(args.n),
                "comm_profile": str(args.comm),
            },
        )
    finally:
        env.close()

    print(
        json.dumps(
            {
                "ok": row["api_error"] == "" and row["finite_observations"] and row["finite_rewards"],
                "policy": row["policy"],
                "scenario": row["scenario"],
                "dimension": row["dimension"],
                "steps": row["steps"],
                "final_min_sep_m": row["final_min_sep_m"],
                "api_error": row["api_error"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
