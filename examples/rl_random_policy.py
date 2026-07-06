from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from microbench.rl import run_rl_policy_smoke


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a compact DAA Microbench RL policy smoke example.")
    parser.add_argument("--out-dir", default="runs_rl_random_example")
    parser.add_argument("--policy", choices=["zero", "random", "goal_direction"], default="random")
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--max-steps", type=int, default=100)
    args = parser.parse_args()

    seeds = [int(part.strip()) for part in args.seeds.split(",") if part.strip()]
    report = run_rl_policy_smoke(
        out_dir=args.out_dir,
        policy=args.policy,
        n_agents=int(args.n),
        seeds=seeds,
        max_steps=args.max_steps,
    )
    print(json.dumps({k: report[k] for k in ("ok", "policy", "run_count", "dimensions", "episode_csv")}, indent=2))


if __name__ == "__main__":
    main()
