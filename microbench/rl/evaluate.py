from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

from microbench.rl.policy_spec import policy_factory_from_spec
from microbench.rl.rollout import RL_ROLLOUT_FIELDS, RL_ROLLOUT_SCHEMA_VERSION, run_parallel_policy_rollouts
from microbench.rl.schema import (
    RL_ACTION_SCHEMA_VERSION,
    RL_INTERFACE_VERSION,
    RL_OBSERVATION_SCHEMA_VERSION,
    RL_REWARD_SCHEMA_VERSION,
    interface_contract,
)
from microbench.scenarios import materialize_official_suite


RL_SMOKE_SCHEMA_VERSION = "0.1"
RL_SMOKE_SUITE = "official_smoke_generated"
RL_SMOKE_SCENARIOS = ("head_on_2d_easy", "sphere_swap_3d_medium")
RL_EPISODE_FIELDS = RL_ROLLOUT_FIELDS


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


def run_rl_policy_smoke(
    *,
    out_dir: str | Path,
    policy: str = "goal_direction",
    policy_spec: str | Path | None = None,
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
    scenario_paths = {Path(path).stem: Path(path) for path in generated["scenario_paths"]}

    unknown = sorted(set(scenario_id_list) - set(scenario_paths))
    if unknown:
        raise ValueError(f"Unknown scenario(s) for {RL_SMOKE_SUITE}: {','.join(unknown)}")

    selected_paths = {scenario_id: scenario_paths[scenario_id] for scenario_id in scenario_id_list}
    policy_for_rollout: Any = str(policy)
    policy_name = str(policy)
    policy_spec_summary = None
    if policy_spec is not None:
        policy_for_rollout, policy_spec_summary = policy_factory_from_spec(policy_spec)
        policy_name = str(policy_spec_summary["policy_name"])
    rows = run_parallel_policy_rollouts(
        scenario_paths=selected_paths,
        policy=policy_for_rollout,
        n_agents=int(n_agents),
        seeds=seed_list,
        comm_profile=str(comm_profile),
        max_steps=max_steps,
        suite=RL_SMOKE_SUITE,
        policy_name=policy_name,
    )
    errors = [
        {"scenario": row.get("scenario"), "seed": int(row.get("seed", 0)), "error": row.get("api_error")}
        for row in rows
        if row.get("api_error")
    ]

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
        _check(
            "schema_versions_present",
            bool(RL_INTERFACE_VERSION)
            and bool(RL_ACTION_SCHEMA_VERSION)
            and bool(RL_OBSERVATION_SCHEMA_VERSION)
            and bool(RL_REWARD_SCHEMA_VERSION),
        ),
    ]

    ok = all(check["ok"] for check in checks)
    return {
        "schema_version": RL_SMOKE_SCHEMA_VERSION,
        "rollout_schema_version": RL_ROLLOUT_SCHEMA_VERSION,
        "interface_version": RL_INTERFACE_VERSION,
        "action_schema_version": RL_ACTION_SCHEMA_VERSION,
        "observation_schema_version": RL_OBSERVATION_SCHEMA_VERSION,
        "reward_schema_version": RL_REWARD_SCHEMA_VERSION,
        "ok": bool(ok),
        "suite": RL_SMOKE_SUITE,
        "policy": policy_name,
        "policy_spec": policy_spec_summary,
        "scenario_ids": scenario_id_list,
        "n_agents": int(n_agents),
        "seeds": seed_list,
        "comm_profile": str(comm_profile),
        "max_steps": None if max_steps is None else int(max_steps),
        "run_count": len(rows),
        "dimensions": dimensions,
        "episode_csv": str(episode_csv),
        "suite_manifest": str(generated["manifest_path"]),
        "interface_contract": interface_contract(top_k=8),
        "episodes": rows,
        "checks": checks,
    }
