from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from microbench.rl.envs import DaaParallelEnv
from microbench.rl.policy_spec import policy_factory_from_spec
from microbench.rl.rollout import RL_ROLLOUT_FIELDS, RL_ROLLOUT_SCHEMA_VERSION, rollout_parallel_env
from microbench.rl.schema import (
    RL_ACTION_SCHEMA_VERSION,
    RL_INTERFACE_VERSION,
    RL_OBSERVATION_SCHEMA_VERSION,
    RL_REWARD_SCHEMA_VERSION,
    interface_contract,
)
from microbench.scenarios import materialize_official_suite


RL_CALIBRATION_SCHEMA_VERSION = "0.1"
RL_CALIBRATION_SUITE = "official_promotion_calibration"
RL_CALIBRATION_LANES = (
    {
        "band": "rl_3d_stress",
        "scenario": "sphere_swap_3d_medium",
        "comm_profile": "ideal_50hz",
        "purpose": "compact 3D volumetric stress lane",
    },
    {
        "band": "rl_degraded_sensing_comm",
        "scenario": "sensor_volume_3d_hard",
        "comm_profile": "degraded_20hz",
        "purpose": "compact degraded V2V and fused-sensing lane",
    },
)
RL_CALIBRATION_FIELDS = ("band", *RL_ROLLOUT_FIELDS)


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
        writer = csv.DictWriter(f, fieldnames=list(RL_CALIBRATION_FIELDS))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RL_CALIBRATION_FIELDS})


def run_rl_policy_calibration(
    *,
    out_dir: str | Path,
    policy: str = "goal_direction",
    policy_spec: str | Path | None = None,
    n_agents: int = 4,
    seeds: tuple[int, ...] | list[int] | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Run compact 3D/degraded RL policy calibration lanes.

    This is a wrapper-health and stress-exposure gate for learned-policy
    submissions. It is not a leaderboard score and does not replace official
    metric CSVs for benchmark comparisons.
    """

    out = Path(out_dir)
    episode_csv = out / "rl_calibration_episodes.csv"
    if episode_csv.exists():
        raise RuntimeError(f"RL calibration output already exists: {episode_csv}")

    seed_list = [int(seed) for seed in (seeds if seeds is not None else (0,))]
    policy_for_rollout: Any = str(policy)
    policy_name = str(policy)
    policy_spec_summary = None
    if policy_spec is not None:
        policy_for_rollout, policy_spec_summary = policy_factory_from_spec(policy_spec)
        policy_name = str(policy_spec_summary["policy_name"])

    generated_dir = out / "_generated_scenarios" / RL_CALIBRATION_SUITE
    generated = materialize_official_suite(RL_CALIBRATION_SUITE, generated_dir, overwrite=True)
    scenario_paths = {Path(path).stem: Path(path) for path in generated["scenario_paths"]}

    rows: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for lane in RL_CALIBRATION_LANES:
        scenario_id = str(lane["scenario"])
        scenario_path = scenario_paths[scenario_id]
        for seed in seed_list:
            env = DaaParallelEnv(
                scenario_path=str(scenario_path),
                n_agents=int(n_agents),
                seed=int(seed),
                comm_profile=str(lane["comm_profile"]),
            )
            try:
                row = rollout_parallel_env(
                    env,
                    policy_for_rollout,
                    seed=int(seed),
                    max_steps=max_steps,
                    metadata={
                        "suite": RL_CALIBRATION_SUITE,
                        "scenario": scenario_id,
                        "band": str(lane["band"]),
                        "policy": policy_name,
                        "n_agents": int(n_agents),
                        "comm_profile": str(lane["comm_profile"]),
                    },
                )
            except Exception as exc:  # pragma: no cover - failure reporting path.
                row = {
                    "suite": RL_CALIBRATION_SUITE,
                    "scenario": scenario_id,
                    "band": str(lane["band"]),
                    "dimension": "unknown",
                    "policy": policy_name,
                    "n_agents": int(n_agents),
                    "seed": int(seed),
                    "comm_profile": str(lane["comm_profile"]),
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
                    "api_error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                env.close()
            if row.get("api_error"):
                errors.append({"scenario": scenario_id, "seed": int(seed), "error": row.get("api_error")})
            rows.append(row)

    _write_episode_csv(episode_csv, rows)

    finite_metric_violations = [
        row
        for row in rows
        if not bool(row["finite_observations"])
        or not bool(row["finite_rewards"])
        or not _finite(row["total_reward"])
        or not _finite(row["completion_rate"])
    ]
    bands = sorted({str(row["band"]) for row in rows})
    comm_profiles = sorted({str(row["comm_profile"]) for row in rows})
    checks = [
        _check(
            "run_count",
            len(rows) == len(RL_CALIBRATION_LANES) * len(seed_list),
            {"expected": len(RL_CALIBRATION_LANES) * len(seed_list), "actual": len(rows)},
        ),
        _check("no_api_errors", not errors, {"errors": errors[:10]}),
        _check("finite_rollout_metrics", not finite_metric_violations, {"violations": finite_metric_violations[:10]}),
        _check("three_d_only", all(str(row["dimension"]) == "3d" for row in rows), {"dimensions": sorted({str(row["dimension"]) for row in rows})}),
        _check("degraded_lane_present", "degraded_20hz" in comm_profiles, {"comm_profiles": comm_profiles}),
        _check(
            "expected_bands_present",
            {"rl_3d_stress", "rl_degraded_sensing_comm"}.issubset(set(bands)),
            {"bands": bands},
        ),
        _check(
            "controlled_agents_present",
            all(int(row["controlled_agents"]) > 0 for row in rows),
            {"controlled_agents": [int(row["controlled_agents"]) for row in rows]},
        ),
        _check("episodes_progressed", all(int(row["steps"]) > 0 for row in rows), {"steps": [int(row["steps"]) for row in rows]}),
        _check(
            "schema_versions_present",
            bool(RL_INTERFACE_VERSION)
            and bool(RL_ACTION_SCHEMA_VERSION)
            and bool(RL_OBSERVATION_SCHEMA_VERSION)
            and bool(RL_REWARD_SCHEMA_VERSION)
            and bool(RL_ROLLOUT_SCHEMA_VERSION),
        ),
    ]

    return {
        "schema_version": RL_CALIBRATION_SCHEMA_VERSION,
        "rollout_schema_version": RL_ROLLOUT_SCHEMA_VERSION,
        "interface_version": RL_INTERFACE_VERSION,
        "action_schema_version": RL_ACTION_SCHEMA_VERSION,
        "observation_schema_version": RL_OBSERVATION_SCHEMA_VERSION,
        "reward_schema_version": RL_REWARD_SCHEMA_VERSION,
        "ok": all(check["ok"] for check in checks),
        "suite": RL_CALIBRATION_SUITE,
        "policy": policy_name,
        "policy_spec": policy_spec_summary,
        "n_agents": int(n_agents),
        "seeds": seed_list,
        "max_steps": None if max_steps is None else int(max_steps),
        "run_count": len(rows),
        "bands": bands,
        "episode_csv": str(episode_csv),
        "suite_manifest": str(generated["manifest_path"]),
        "interface_contract": interface_contract(top_k=8),
        "lanes": [dict(lane) for lane in RL_CALIBRATION_LANES],
        "episodes": rows,
        "checks": checks,
    }
