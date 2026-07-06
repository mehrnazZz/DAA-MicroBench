from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

from microbench.rl.envs import DaaParallelEnv
from microbench.rl.policies import make_policy
from microbench.scenarios import materialize_official_suite


RL_SMOKE_SCHEMA_VERSION = "0.1"
RL_SMOKE_SUITE = "official_smoke_generated"
RL_SMOKE_SCENARIOS = ("head_on_2d_easy", "sphere_swap_3d_medium")
RL_EPISODE_FIELDS = (
    "suite",
    "scenario",
    "dimension",
    "policy",
    "n_agents",
    "seed",
    "comm_profile",
    "steps",
    "controlled_agents",
    "completed_agents",
    "completion_rate",
    "terminated_agents",
    "truncated_agents",
    "total_reward",
    "mean_reward_per_agent",
    "final_min_sep_m",
    "collision_ticks",
    "near_miss_ticks",
    "finite_observations",
    "finite_rewards",
    "api_error",
)


def _as_list(values: tuple[str, ...] | list[str] | None, default: tuple[str, ...]) -> list[str]:
    return [str(v).strip() for v in (values if values is not None else default) if str(v).strip()]


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def _write_episode_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(RL_EPISODE_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RL_EPISODE_FIELDS})


def _episode_dimension(env: DaaParallelEnv) -> str:
    return "2d" if env.planar else "3d"


def run_rl_policy_smoke(
    *,
    out_dir: str | Path,
    policy: str = "goal_direction",
    scenario_ids: tuple[str, ...] | list[str] | None = None,
    n_agents: int = 4,
    seeds: tuple[int, ...] | list[int] | None = None,
    comm_profile: str = "ideal_50hz",
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run a compact RL wrapper smoke/evaluation matrix.

    The report intentionally checks API health and 2D/3D coverage, not safety
    leaderboard quality. Safety comparisons should use the benchmark metrics
    pipeline and official suite reports.
    """

    out = Path(out_dir)
    episode_csv = out / "rl_smoke_episodes.csv"
    if episode_csv.exists():
        raise RuntimeError(f"RL smoke output already exists: {episode_csv}")

    scenario_id_list = _as_list(scenario_ids, RL_SMOKE_SCENARIOS)
    seed_list = [int(s) for s in (seeds if seeds is not None else (0,))]

    generated_dir = out / "_generated_scenarios" / RL_SMOKE_SUITE
    generated = materialize_official_suite(RL_SMOKE_SUITE, generated_dir, overwrite=True)
    manifest = generated["manifest"]
    scenario_meta = {str(entry["id"]): entry for entry in manifest["scenarios"]}
    scenario_paths = {Path(path).stem: Path(path) for path in generated["scenario_paths"]}

    unknown = sorted(set(scenario_id_list) - set(scenario_paths))
    if unknown:
        raise ValueError(f"Unknown scenario(s) for {RL_SMOKE_SUITE}: {','.join(unknown)}")

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for scenario_id in scenario_id_list:
        for seed in seed_list:
            env = DaaParallelEnv(
                scenario_path=str(scenario_paths[scenario_id]),
                n_agents=int(n_agents),
                seed=int(seed),
                comm_profile=str(comm_profile),
            )
            api_error = ""
            observations: dict[str, np.ndarray] = {}
            infos: dict[str, dict[str, Any]] = {}
            try:
                observations, infos = env.reset(seed=int(seed))
                agent_names = list(env.agents)
                controlled_count = len(agent_names)
                policy_obj = make_policy(policy, seed=int(seed))
                total_rewards = {agent: 0.0 for agent in agent_names}
                completed: set[str] = set()
                terminated: set[str] = set()
                truncated: set[str] = set()
                collision_ticks = 0
                near_miss_ticks = 0
                final_min_sep = math.inf
                finite_observations = all(np.all(np.isfinite(obs)) for obs in observations.values())
                finite_rewards = True
                step_limit = env.episode_step_limit if env.episode_step_limit is not None else 0
                cap = int(max_steps) if max_steps is not None else int(step_limit)
                steps = 0

                while env.agents and steps < cap:
                    actions = {
                        agent: policy_obj.action(agent, observations[agent], env.action_space(agent), infos.get(agent, {}))
                        for agent in env.agents
                    }
                    observations, rewards, terminations, truncations, infos = env.step(actions)
                    steps += 1
                    finite_observations = finite_observations and all(
                        np.all(np.isfinite(obs)) for obs in observations.values()
                    )
                    finite_rewards = finite_rewards and all(_finite(value) for value in rewards.values())
                    for agent, reward in rewards.items():
                        total_rewards[agent] = total_rewards.get(agent, 0.0) + float(reward)
                    for agent, done in terminations.items():
                        if done:
                            terminated.add(agent)
                    for agent, done in truncations.items():
                        if done:
                            truncated.add(agent)
                    for agent, info in infos.items():
                        if bool(info.get("done", False)):
                            completed.add(agent)
                        final_min_sep = min(final_min_sep, float(info.get("min_sep_m", final_min_sep)))
                    if any(bool(info.get("collision", False)) for info in infos.values()):
                        collision_ticks += 1
                    if any(bool(info.get("near_miss", False)) for info in infos.values()):
                        near_miss_ticks += 1

                if env.agents and steps >= cap:
                    truncated.update(env.agents)
                if not math.isfinite(final_min_sep):
                    final_min_sep = float("nan")

                total_reward = float(sum(total_rewards.values()))
                row = {
                    "suite": RL_SMOKE_SUITE,
                    "scenario": scenario_id,
                    "dimension": _episode_dimension(env),
                    "policy": str(policy),
                    "n_agents": int(n_agents),
                    "seed": int(seed),
                    "comm_profile": str(comm_profile),
                    "steps": int(steps),
                    "controlled_agents": int(controlled_count),
                    "completed_agents": int(len(completed)),
                    "completion_rate": float(len(completed) / max(1, controlled_count)),
                    "terminated_agents": int(len(terminated)),
                    "truncated_agents": int(len(truncated)),
                    "total_reward": total_reward,
                    "mean_reward_per_agent": float(total_reward / max(1, controlled_count)),
                    "final_min_sep_m": float(final_min_sep),
                    "collision_ticks": int(collision_ticks),
                    "near_miss_ticks": int(near_miss_ticks),
                    "finite_observations": bool(finite_observations),
                    "finite_rewards": bool(finite_rewards),
                    "api_error": api_error,
                }
            except Exception as exc:  # pragma: no cover - exercised by failure reports, not happy-path tests.
                api_error = f"{type(exc).__name__}: {exc}"
                errors.append({"scenario": scenario_id, "seed": int(seed), "error": api_error})
                row = {
                    "suite": RL_SMOKE_SUITE,
                    "scenario": scenario_id,
                    "dimension": str(scenario_meta.get(scenario_id, {}).get("dimension", "unknown")),
                    "policy": str(policy),
                    "n_agents": int(n_agents),
                    "seed": int(seed),
                    "comm_profile": str(comm_profile),
                    "steps": 0,
                    "controlled_agents": 0,
                    "completed_agents": 0,
                    "completion_rate": 0.0,
                    "terminated_agents": 0,
                    "truncated_agents": 0,
                    "total_reward": 0.0,
                    "mean_reward_per_agent": 0.0,
                    "final_min_sep_m": float("nan"),
                    "collision_ticks": 0,
                    "near_miss_ticks": 0,
                    "finite_observations": False,
                    "finite_rewards": False,
                    "api_error": api_error,
                }
            finally:
                env.close()
            rows.append(row)

    _write_episode_csv(episode_csv, rows)

    dimensions = sorted({str(row["dimension"]) for row in rows})
    finite_metric_violations = [
        row
        for row in rows
        if not bool(row["finite_observations"])
        or not bool(row["finite_rewards"])
        or not _finite(row["total_reward"])
        or not _finite(row["completion_rate"])
    ]
    checks = [
        _check(
            "run_count",
            len(rows) == len(scenario_id_list) * len(seed_list),
            {"expected": len(scenario_id_list) * len(seed_list), "actual": len(rows)},
        ),
        _check("no_api_errors", not errors, {"errors": errors[:10]}),
        _check("finite_rollout_metrics", not finite_metric_violations, {"violations": finite_metric_violations[:10]}),
        _check("two_d_and_three_d_coverage", {"2d", "3d"}.issubset(set(dimensions)), {"dimensions": dimensions}),
        _check(
            "controlled_agents_present",
            all(int(row["controlled_agents"]) > 0 for row in rows),
            {"controlled_agents": [int(row["controlled_agents"]) for row in rows]},
        ),
        _check("episodes_progressed", all(int(row["steps"]) > 0 for row in rows), {"steps": [int(row["steps"]) for row in rows]}),
    ]

    ok = all(check["ok"] for check in checks)
    return {
        "schema_version": RL_SMOKE_SCHEMA_VERSION,
        "ok": bool(ok),
        "suite": RL_SMOKE_SUITE,
        "policy": str(policy),
        "scenario_ids": scenario_id_list,
        "n_agents": int(n_agents),
        "seeds": seed_list,
        "comm_profile": str(comm_profile),
        "max_steps": None if max_steps is None else int(max_steps),
        "run_count": len(rows),
        "dimensions": dimensions,
        "episode_csv": str(episode_csv),
        "suite_manifest": str(generated["manifest_path"]),
        "episodes": rows,
        "checks": checks,
    }
