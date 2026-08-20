from __future__ import annotations

from dataclasses import asdict
import copy
import csv
import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from microbench.learned import (
    FrozenMlpPolicyModel,
    LEARNED_BASELINE_SCHEMA_VERSION,
    MLP_LEARNED_FEATURE_NAMES,
    MLP_LEARNED_MODEL_ID,
    load_mlp_learned_spec,
)
from microbench.rl.adapters import ModelPredictPolicyAdapter
from microbench.rl.envs import DaaParallelEnv
from microbench.rl.policy_spec import RL_POLICY_SPEC_SCHEMA_VERSION, load_policy_spec, resolve_policy_artifact_path
from microbench.rl.rollout import RL_ROLLOUT_FIELDS, rollout_parallel_env
from microbench.rl.validation_matrix import run_rl_validation_matrix
from microbench.metrics import append_result, write_summary
from microbench.runner import run_episode
from microbench.scenarios import materialize_official_suite
from microbench.tools.baseline_report import score_v0
from microbench.tools.baseline_validation_matrix import (
    ValidationLane,
    prepare_validation_lane_scenarios,
    selected_validation_lanes,
)
from microbench.types import RunSpec


CLOSED_LOOP_TRAINING_SCHEMA_VERSION = "0.1"
CLOSED_LOOP_POLICY_NAME = "closed_loop_mlp_learned"
CLOSED_LOOP_TRAINABLE_PARAMETER_CHOICES = ("output_head", "all_layers")
CLOSED_LOOP_TRAINING_LANE_PROFILE_CHOICES = ("validation", "broad_3d_stress", "validation_plus_broad_3d")
CLOSED_LOOP_HOLDOUT_PROFILE_CHOICES = ("none", "broad_3d_stress")
CLOSED_LOOP_HOLDOUT_SCORE_TOLERANCE = 1.0
CLOSED_LOOP_BROAD_3D_HOLDOUT_SCENARIOS = (
    "sphere_swap_3d_medium",
    "dense_swarm_3d_hard",
    "merge_3d_hard",
    "sensor_volume_3d_hard",
    "noncooperative_intruder_3d_hard",
)
CLOSED_LOOP_BROAD_3D_TRAINING_LANES = (
    ValidationLane(
        lane_id="sphere_swap_3d_training",
        category="sphere_swap_3d",
        suite="official_3d_stress",
        scenario="sphere_swap_3d_medium",
        comm_profile="ideal_50hz",
        n_agents=6,
        seed=0,
        duration_s=18.0,
        purpose="Closed-loop learned-policy 3D altitude/lateral swap training lane.",
        expected_failure_modes=("vertical_layer_conflict", "reciprocal_yield_oscillation", "late_lateral_yield"),
    ),
    ValidationLane(
        lane_id="dense_swarm_3d_training",
        category="dense_swarm_3d",
        suite="official_3d_stress",
        scenario="dense_swarm_3d_hard",
        comm_profile="degraded_20hz",
        n_agents=8,
        seed=0,
        duration_s=18.0,
        purpose="Closed-loop learned-policy dense 3D swarm training lane under degraded communication.",
        expected_failure_modes=("dense_center_conflict", "throughput_collapse", "stale_intent"),
    ),
    ValidationLane(
        lane_id="merge_3d_training",
        category="merge_3d",
        suite="official_3d_stress",
        scenario="merge_3d_hard",
        comm_profile="degraded_20hz",
        n_agents=6,
        seed=0,
        duration_s=18.0,
        purpose="Closed-loop learned-policy 3D merge training lane with bottleneck pressure.",
        expected_failure_modes=("late_merge", "vertical_squeeze", "bottleneck_deadlock"),
    ),
    ValidationLane(
        lane_id="sensor_volume_3d_training",
        category="sensor_volume_3d",
        suite="official_3d_stress",
        scenario="sensor_volume_3d_hard",
        comm_profile="degraded_20hz",
        n_agents=6,
        seed=0,
        duration_s=18.0,
        purpose="Closed-loop learned-policy 3D sensing-volume training lane under stale fused tracks.",
        expected_failure_modes=("fov_blind_spot", "stale_track_collision", "message_sensor_disagreement"),
    ),
    ValidationLane(
        lane_id="noncooperative_intruder_3d_training",
        category="noncooperative_intruder_3d",
        suite="official_3d_stress",
        scenario="noncooperative_intruder_3d_hard",
        comm_profile="degraded_20hz",
        n_agents=6,
        seed=0,
        duration_s=18.0,
        purpose="Closed-loop learned-policy intruder-geometry training lane; all agents share the candidate RL policy.",
        expected_failure_modes=("noncooperative_intruder", "late_yield", "priority_inversion"),
    ),
)
CLOSED_LOOP_BROAD_3D_TRAINING_LANE_IDS = tuple(lane.lane_id for lane in CLOSED_LOOP_BROAD_3D_TRAINING_LANES)
CLOSED_LOOP_OBJECTIVE_DEFAULTS = {
    "collision_tick_penalty": 120.0,
    "near_miss_tick_penalty": 12.0,
    "clearance_penalty": 20.0,
    "mission_penalty": 60.0,
    "reward_weight": 1.0,
    "min_clearance_m": 0.0,
    "max_collision_ticks": 0,
    "max_near_miss_ticks": None,
}
CLOSED_LOOP_CANDIDATE_FIELDS = (
    "candidate_id",
    "generation",
    "candidate_index",
    "parent_candidate_id",
    "accepted",
    "feasible",
    "score",
    "score_delta_vs_best_before",
    "collision_ticks",
    "near_miss_ticks",
    "min_clearance_m",
    "completion_rate_mean",
    "total_reward_mean",
    "api_error_count",
    "finite",
    "sigma",
)
CLOSED_LOOP_EPISODE_FIELDS = (
    "candidate_id",
    "generation",
    "candidate_index",
    "accepted",
    "feasible",
    *RL_ROLLOUT_FIELDS,
)
CLOSED_LOOP_HOLDOUT_COMPARISON_FIELDS = (
    "label",
    "run_count",
    "scenario_count",
    "collision_episodes",
    "near_miss_episodes",
    "completion_rate_mean",
    "min_sep_min_row_m",
    "min_sep_p05_row_min_m",
    "score_v0_mean",
    "score_v0_worst",
    "planner_ms_p95_max",
    "results_csv",
    "summary_csv",
)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return path


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _round_or_none(value: Any, *, digits: int = 6) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return round(out, digits)


def _check(name: str, ok: bool, details: dict[str, Any] | None = None, *, severity: str = "gate") -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "severity": str(severity), "details": details or {}}


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(fields))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})
    return path


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _finite_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(out):
        return None
    return out


def _finite_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for row in rows:
        value = _finite_float(row.get(key))
        if value is not None:
            values.append(value)
    return values


def _mean_or_none(values: list[float], *, digits: int = 6) -> float | None:
    if not values:
        return None
    return round(float(sum(values) / len(values)), digits)


def _min_or_none(values: list[float], *, digits: int = 6) -> float | None:
    if not values:
        return None
    return round(float(min(values)), digits)


def _max_or_none(values: list[float], *, digits: int = 6) -> float | None:
    if not values:
        return None
    return round(float(max(values)), digits)


def _holdout_profile(value: str) -> str:
    normalized = str(value).strip()
    if normalized not in CLOSED_LOOP_HOLDOUT_PROFILE_CHOICES:
        raise ValueError("holdout_profile must be one of " + ",".join(CLOSED_LOOP_HOLDOUT_PROFILE_CHOICES))
    return normalized


def _holdout_str_list(values: tuple[str, ...] | list[str] | None, default: tuple[str, ...]) -> list[str]:
    return [str(value).strip() for value in (default if values is None else values) if str(value).strip()]


def _holdout_int_list(values: tuple[int, ...] | list[int] | None, default: tuple[int, ...]) -> list[int]:
    return [int(value) for value in (default if values is None else values)]


def _training_lane_profile(value: str) -> str:
    normalized = str(value).strip()
    if normalized not in CLOSED_LOOP_TRAINING_LANE_PROFILE_CHOICES:
        raise ValueError("lane_profile must be one of " + ",".join(CLOSED_LOOP_TRAINING_LANE_PROFILE_CHOICES))
    return normalized


def _closed_loop_training_lane_map() -> dict[str, ValidationLane]:
    lanes = [*selected_validation_lanes(None), *CLOSED_LOOP_BROAD_3D_TRAINING_LANES]
    return {lane.lane_id: lane for lane in lanes}


def _dedupe_lanes(lanes: list[ValidationLane]) -> list[ValidationLane]:
    out: list[ValidationLane] = []
    seen: set[str] = set()
    for lane in lanes:
        if lane.lane_id in seen:
            continue
        out.append(lane)
        seen.add(lane.lane_id)
    return out


def selected_closed_loop_training_lanes(
    lanes: tuple[str, ...] | list[str] | None = None,
    *,
    lane_profile: str = "validation",
) -> list[ValidationLane]:
    """Return closed-loop training lanes, including optional broad 3D stress lanes."""

    profile = _training_lane_profile(lane_profile)
    if lanes is None:
        validation = selected_validation_lanes(None)
        broad = list(CLOSED_LOOP_BROAD_3D_TRAINING_LANES)
        if profile == "validation":
            return validation
        if profile == "broad_3d_stress":
            return broad
        return _dedupe_lanes([*validation, *broad])

    lane_ids = [str(lane_id).strip() for lane_id in lanes if str(lane_id).strip()]
    by_id = _closed_loop_training_lane_map()
    unknown = sorted(set(lane_ids) - set(by_id))
    if unknown:
        raise ValueError(
            "Unknown closed-loop training lane(s): "
            + ",".join(unknown)
            + "; expected validation lanes or "
            + ",".join(CLOSED_LOOP_BROAD_3D_TRAINING_LANE_IDS)
        )
    return [by_id[lane_id] for lane_id in lane_ids]


def _canonical_eval_lanes_for_training_lanes(lanes: list[ValidationLane]) -> list[str]:
    canonical_ids = [lane.lane_id for lane in selected_validation_lanes(None)]
    selected_ids = {lane.lane_id for lane in lanes}
    overlap = [lane_id for lane_id in canonical_ids if lane_id in selected_ids]
    return overlap if overlap else canonical_ids


def _materialize_holdout_scenarios(out_dir: Path, scenario_ids: list[str]) -> dict[str, Path]:
    generated = materialize_official_suite("official_3d_stress", out_dir, overwrite=True)
    by_id = {path.stem: path for path in generated["scenario_paths"]}
    missing = sorted(set(scenario_ids) - set(by_id))
    if missing:
        raise ValueError(f"unknown broad 3D holdout scenario(s): {','.join(missing)}")
    return {scenario_id: by_id[scenario_id] for scenario_id in scenario_ids}


def _holdout_agent_methods(scenario_id: str, n_agents: int) -> list[str] | None:
    if scenario_id != "noncooperative_intruder_3d_hard":
        return None
    return ["baseline_goal"] + ["learned_policy_spec"] * max(0, int(n_agents) - 1)


def _holdout_run_specs(
    *,
    run_dir: Path,
    policy_spec: str | Path,
    scenario_paths: dict[str, Path],
    scenario_ids: list[str],
    seeds: list[int],
    comm_profiles: list[str],
    n_agents: int,
    max_runs: int | None,
) -> list[RunSpec]:
    specs: list[RunSpec] = []
    for scenario_id in scenario_ids:
        for comm_profile in comm_profiles:
            for seed in seeds:
                specs.append(
                    RunSpec(
                        scenario_path=str(scenario_paths[scenario_id]),
                        method="learned_policy_spec",
                        n_agents=int(n_agents),
                        seed=int(seed),
                        comm_profile=str(comm_profile),
                        out_dir=str(run_dir),
                        save_trace=False,
                        agent_methods=_holdout_agent_methods(scenario_id, int(n_agents)),
                        policy_spec=str(policy_spec),
                    )
                )
    if max_runs is None:
        return specs
    return specs[: max(0, int(max_runs))]


def _holdout_summary(label: str, run_dir: Path) -> dict[str, Any]:
    results_csv = run_dir / "results.csv"
    summary_csv = run_dir / "summary.csv"
    result_rows = _read_csv_rows(results_csv)
    summary_rows = _read_csv_rows(summary_csv)
    scored_rows: list[dict[str, Any]] = []
    scores: list[float] = []
    for row in summary_rows:
        projected = dict(row)
        score = score_v0(projected)
        projected["score_v0"] = score
        scored_rows.append(projected)
        if score is not None:
            scores.append(float(score))
    collision_episodes = sum(int(_finite_float(row.get("collision_episode")) or 0) for row in result_rows)
    near_miss_episodes = sum(int(_finite_float(row.get("near_miss_episode")) or 0) for row in result_rows)
    return {
        "label": str(label),
        "run_count": int(len(result_rows)),
        "summary_row_count": int(len(summary_rows)),
        "scenario_count": len({str(row.get("scenario", "")) for row in result_rows if str(row.get("scenario", ""))}),
        "comm_profiles": sorted({str(row.get("comm_profile", "")) for row in result_rows if str(row.get("comm_profile", ""))}),
        "n_agents": sorted({int(float(row["N"])) for row in result_rows if _finite_float(row.get("N")) is not None}),
        "collision_episodes": int(collision_episodes),
        "near_miss_episodes": int(near_miss_episodes),
        "completion_rate_mean": _mean_or_none(_finite_values(result_rows, "completion_rate")),
        "min_sep_min_row_m": _min_or_none(_finite_values(result_rows, "min_sep_min_m")),
        "min_sep_p05_row_min_m": _min_or_none(_finite_values(result_rows, "min_sep_p05_m")),
        "score_v0_mean": _mean_or_none(scores),
        "score_v0_worst": _max_or_none(scores),
        "planner_ms_p95_max": _max_or_none(_finite_values(result_rows, "planner_ms_per_tick_per_agent_p95")),
        "results_csv": str(results_csv),
        "summary_csv": str(summary_csv),
        "scored_summary_rows": scored_rows,
    }


def _comparison_delta(tuned: Any, base: Any) -> float | None:
    tuned_f = _finite_float(tuned)
    base_f = _finite_float(base)
    if tuned_f is None or base_f is None:
        return None
    return round(float(tuned_f - base_f), 6)


def _run_holdout_policy(label: str, run_dir: Path, specs: list[RunSpec]) -> dict[str, Any]:
    run_dir.mkdir(parents=True, exist_ok=True)
    for spec in specs:
        append_result(run_dir, run_episode(spec))
    write_summary(run_dir)
    return _holdout_summary(label, run_dir)


def _run_broad_3d_holdout(
    *,
    out_dir: Path,
    base_policy_spec: str | Path,
    tuned_policy_spec: str | Path,
    scenarios: list[str],
    seeds: list[int],
    comm_profiles: list[str],
    n_agents: int,
    max_runs: int | None,
    score_tolerance: float,
    allow_safety_regression: bool,
    allow_score_regression: bool,
) -> dict[str, Any]:
    if int(n_agents) < 2:
        raise ValueError("holdout_n_agents must be >= 2")
    if not scenarios:
        raise ValueError("holdout_scenarios must not be empty when holdout is enabled")
    if not seeds:
        raise ValueError("holdout_seeds must not be empty when holdout is enabled")
    if not comm_profiles:
        raise ValueError("holdout_comm_profiles must not be empty when holdout is enabled")
    if float(score_tolerance) < 0.0 or not math.isfinite(float(score_tolerance)):
        raise ValueError("holdout score_tolerance must be finite and >= 0")

    scenario_dir = out_dir / "_closed_loop_holdout_scenarios" / "official_3d_stress"
    scenario_paths = _materialize_holdout_scenarios(scenario_dir, scenarios)
    base_dir = out_dir / "base"
    tuned_dir = out_dir / "tuned"
    base_specs = _holdout_run_specs(
        run_dir=base_dir,
        policy_spec=base_policy_spec,
        scenario_paths=scenario_paths,
        scenario_ids=scenarios,
        seeds=seeds,
        comm_profiles=comm_profiles,
        n_agents=int(n_agents),
        max_runs=max_runs,
    )
    tuned_specs = _holdout_run_specs(
        run_dir=tuned_dir,
        policy_spec=tuned_policy_spec,
        scenario_paths=scenario_paths,
        scenario_ids=scenarios,
        seeds=seeds,
        comm_profiles=comm_profiles,
        n_agents=int(n_agents),
        max_runs=max_runs,
    )
    expected_runs = len(base_specs)
    base_summary = _run_holdout_policy("base", base_dir, base_specs)
    tuned_summary = _run_holdout_policy("tuned", tuned_dir, tuned_specs)

    comparison_csv = _write_csv(
        out_dir / "comparison_summary.csv",
        [base_summary, tuned_summary],
        CLOSED_LOOP_HOLDOUT_COMPARISON_FIELDS,
    )
    no_collision_regression = int(tuned_summary["collision_episodes"]) <= int(base_summary["collision_episodes"])
    no_near_miss_regression = int(tuned_summary["near_miss_episodes"]) <= int(base_summary["near_miss_episodes"])
    base_min_sep = _finite_float(base_summary.get("min_sep_min_row_m"))
    tuned_min_sep = _finite_float(tuned_summary.get("min_sep_min_row_m"))
    clearance_not_worse = base_min_sep is not None and tuned_min_sep is not None and tuned_min_sep >= base_min_sep - 1e-9
    base_score = _finite_float(base_summary.get("score_v0_mean"))
    tuned_score = _finite_float(tuned_summary.get("score_v0_mean"))
    score_not_worse = base_score is not None and tuned_score is not None and tuned_score <= base_score + float(score_tolerance)
    checks = [
        _check(
            "holdout_runs_completed",
            expected_runs > 0 and base_summary["run_count"] == expected_runs and tuned_summary["run_count"] == expected_runs,
            {"expected_runs_per_policy": expected_runs, "base_runs": base_summary["run_count"], "tuned_runs": tuned_summary["run_count"]},
        ),
        _check(
            "holdout_no_collision_regression",
            bool(no_collision_regression or allow_safety_regression),
            {
                "base_collision_episodes": base_summary["collision_episodes"],
                "tuned_collision_episodes": tuned_summary["collision_episodes"],
                "allow_safety_regression": bool(allow_safety_regression),
            },
            severity="behavior",
        ),
        _check(
            "holdout_no_near_miss_regression",
            bool(no_near_miss_regression or allow_safety_regression),
            {
                "base_near_miss_episodes": base_summary["near_miss_episodes"],
                "tuned_near_miss_episodes": tuned_summary["near_miss_episodes"],
                "allow_safety_regression": bool(allow_safety_regression),
            },
            severity="behavior",
        ),
        _check(
            "holdout_clearance_not_worse",
            bool(clearance_not_worse or allow_safety_regression),
            {
                "base_min_sep_min_row_m": base_summary.get("min_sep_min_row_m"),
                "tuned_min_sep_min_row_m": tuned_summary.get("min_sep_min_row_m"),
                "delta_m": _comparison_delta(tuned_summary.get("min_sep_min_row_m"), base_summary.get("min_sep_min_row_m")),
                "allow_safety_regression": bool(allow_safety_regression),
            },
            severity="behavior",
        ),
        _check(
            "holdout_score_not_worse",
            bool(score_not_worse or allow_score_regression),
            {
                "base_score_v0_mean": base_summary.get("score_v0_mean"),
                "tuned_score_v0_mean": tuned_summary.get("score_v0_mean"),
                "delta": _comparison_delta(tuned_summary.get("score_v0_mean"), base_summary.get("score_v0_mean")),
                "tolerance": float(score_tolerance),
                "allow_score_regression": bool(allow_score_regression),
            },
            severity="behavior",
        ),
    ]
    promotion_candidate = all(check["ok"] for check in checks)
    report = {
        "profile": "broad_3d_stress",
        "suite": "official_3d_stress",
        "scenarios": list(scenarios),
        "seeds": [int(seed) for seed in seeds],
        "comm_profiles": list(comm_profiles),
        "n_agents": int(n_agents),
        "max_runs": None if max_runs is None else int(max_runs),
        "score_tolerance": float(score_tolerance),
        "expected_runs_per_policy": int(expected_runs),
        "base": base_summary,
        "tuned": tuned_summary,
        "comparison_csv": str(comparison_csv),
        "checks": checks,
        "promotion_candidate": bool(promotion_candidate),
        "promotion_note": "Tuned policy must not regress base policy safety, clearance, or score_v0 on the selected broad 3D holdout rows.",
    }
    _write_json(out_dir / "comparison_report.json", report)
    return report


def _copy_with_output_head(spec: dict[str, Any], vector: np.ndarray) -> dict[str, Any]:
    out = copy.deepcopy(spec)
    hidden_dim = int(out["hidden_dim"])
    expected = 3 * hidden_dim + 3
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    if vec.shape != (expected,):
        raise ValueError(f"MLP output-head vector must have shape {(expected,)}, got {vec.shape}")
    w2 = vec[: 3 * hidden_dim].reshape(3, hidden_dim)
    b2 = vec[3 * hidden_dim :]
    out["layer2_weights"] = w2.round(8).tolist()
    out["layer2_bias"] = b2.round(8).tolist()
    return out


def _output_head_vector(spec: dict[str, Any]) -> np.ndarray:
    hidden_dim = int(spec["hidden_dim"])
    w2 = np.asarray(spec["layer2_weights"], dtype=np.float64)
    b2 = np.asarray(spec["layer2_bias"], dtype=np.float64)
    if w2.shape != (3, hidden_dim):
        raise ValueError(f"MLP layer2 weights must have shape {(3, hidden_dim)}, got {w2.shape}")
    if b2.shape != (3,):
        raise ValueError(f"MLP layer2 bias must have shape (3,), got {b2.shape}")
    return np.concatenate([w2.reshape(-1), b2.reshape(-1)], axis=0)


def _trainable_parameters(value: str) -> str:
    normalized = str(value).strip()
    if normalized not in CLOSED_LOOP_TRAINABLE_PARAMETER_CHOICES:
        raise ValueError(
            "trainable_parameters must be one of "
            + ",".join(CLOSED_LOOP_TRAINABLE_PARAMETER_CHOICES)
        )
    return normalized


def _parameter_blocks(spec: dict[str, Any], trainable_parameters: str) -> list[tuple[str, tuple[int, ...], np.ndarray]]:
    def block(name: str) -> tuple[str, tuple[int, ...], np.ndarray]:
        values = np.asarray(spec[name], dtype=np.float64)
        return name, tuple(values.shape), values

    mode = _trainable_parameters(trainable_parameters)
    if mode == "output_head":
        return [block("layer2_weights"), block("layer2_bias")]
    return [
        block("layer1_weights"),
        block("layer1_bias"),
        block("layer2_weights"),
        block("layer2_bias"),
    ]


def _parameter_vector(spec: dict[str, Any], trainable_parameters: str) -> np.ndarray:
    blocks = _parameter_blocks(spec, trainable_parameters)
    return np.concatenate([values.reshape(-1) for _name, _shape, values in blocks], axis=0)


def _copy_with_parameter_vector(spec: dict[str, Any], vector: np.ndarray, trainable_parameters: str) -> dict[str, Any]:
    out = copy.deepcopy(spec)
    blocks = _parameter_blocks(out, trainable_parameters)
    vec = np.asarray(vector, dtype=np.float64).reshape(-1)
    expected = sum(int(np.prod(shape)) for _name, shape, _values in blocks)
    if vec.shape != (expected,):
        raise ValueError(f"MLP parameter vector must have shape {(expected,)}, got {vec.shape}")
    cursor = 0
    for name, shape, _values in blocks:
        size = int(np.prod(shape))
        out[name] = vec[cursor : cursor + size].reshape(shape).round(8).tolist()
        cursor += size
    return out


def _load_base_mlp_spec(policy_spec: str | Path) -> tuple[dict[str, Any], Path, dict[str, Any]]:
    spec_path = Path(policy_spec)
    wrapper = load_policy_spec(spec_path)
    if str(wrapper.get("adapter")) != "mlp_json":
        raise ValueError("closed-loop fine-tuning currently supports only mlp_json policy specs")
    artifact_path = resolve_policy_artifact_path(spec_path, wrapper)
    if artifact_path is None:
        raise ValueError("mlp_json policy spec requires artifact_path")
    model_spec = load_mlp_learned_spec(artifact_path)
    return model_spec, artifact_path, wrapper


def _seed_list_for_lane(lane: ValidationLane, seed_override: list[int] | None) -> list[int]:
    return [int(lane.seed)] if seed_override is None else [int(seed) for seed in seed_override]


def _objective_config(
    *,
    collision_tick_penalty: float,
    near_miss_tick_penalty: float,
    clearance_penalty: float,
    mission_penalty: float,
    reward_weight: float,
    min_clearance_m: float,
    max_collision_ticks: int,
    max_near_miss_ticks: int | None,
    allow_near_miss_regression: bool,
) -> dict[str, Any]:
    values = {
        "collision_tick_penalty": float(collision_tick_penalty),
        "near_miss_tick_penalty": float(near_miss_tick_penalty),
        "clearance_penalty": float(clearance_penalty),
        "mission_penalty": float(mission_penalty),
        "reward_weight": float(reward_weight),
        "min_clearance_m": float(min_clearance_m),
    }
    invalid = [key for key, value in values.items() if not math.isfinite(value)]
    if invalid:
        raise ValueError(f"closed-loop objective values must be finite: {','.join(invalid)}")
    if any(values[key] < 0.0 for key in ("collision_tick_penalty", "near_miss_tick_penalty", "clearance_penalty", "mission_penalty")):
        raise ValueError("closed-loop objective penalties must be nonnegative")
    if int(max_collision_ticks) < 0:
        raise ValueError("max_collision_ticks must be >= 0")
    if max_near_miss_ticks is not None and int(max_near_miss_ticks) < 0:
        raise ValueError("max_near_miss_ticks must be >= 0 when provided")
    return {
        **values,
        "max_collision_ticks": int(max_collision_ticks),
        "max_near_miss_ticks": None if max_near_miss_ticks is None else int(max_near_miss_ticks),
        "allow_near_miss_regression": bool(allow_near_miss_regression),
        "description": (
            "closed-loop rollout objective with hard collision/clearance constraints, optional near-miss guardrail, "
            "mission-completion penalty, and optional reward bonus"
        ),
    }


def _aggregate_rollout_rows(
    rows: list[dict[str, Any]],
    *,
    objective: dict[str, Any],
    base_metrics: dict[str, Any] | None = None,
) -> dict[str, Any]:
    api_error_count = sum(1 for row in rows if str(row.get("api_error") or ""))
    finite = all(bool(row.get("finite_observations")) and bool(row.get("finite_rewards")) for row in rows)
    collision_ticks = int(sum(int(row.get("collision_ticks", 0) or 0) for row in rows))
    near_miss_ticks = int(sum(int(row.get("near_miss_ticks", 0) or 0) for row in rows))
    clearances = [_round_or_none(row.get("final_min_sep_m")) for row in rows]
    finite_clearances = [float(value) for value in clearances if value is not None]
    min_clearance = min(finite_clearances) if finite_clearances else float("-inf")
    completion_values = [float(row.get("completion_rate", 0.0) or 0.0) for row in rows]
    reward_values = [float(row.get("total_reward", 0.0) or 0.0) for row in rows]
    completion_mean = float(sum(completion_values) / max(1, len(completion_values)))
    reward_mean = float(sum(reward_values) / max(1, len(reward_values)))
    clearance_shortfall = sum(max(0.0, float(objective["min_clearance_m"]) - float(value)) for value in finite_clearances)
    if not finite_clearances:
        clearance_shortfall += 100.0

    score = (
        float(objective["collision_tick_penalty"]) * float(collision_ticks)
        + float(objective["near_miss_tick_penalty"]) * float(near_miss_ticks)
        + float(objective["clearance_penalty"]) * float(clearance_shortfall)
        + float(objective["mission_penalty"]) * (1.0 - completion_mean)
        - float(objective["reward_weight"]) * reward_mean
    )
    feasible = bool(
        api_error_count == 0
        and finite
        and collision_ticks <= int(objective["max_collision_ticks"])
        and min_clearance >= float(objective["min_clearance_m"])
    )
    max_near_miss = objective.get("max_near_miss_ticks")
    if max_near_miss is not None:
        feasible = feasible and near_miss_ticks <= int(max_near_miss)
    if base_metrics is not None and not bool(objective.get("allow_near_miss_regression", False)):
        feasible = feasible and near_miss_ticks <= int(base_metrics.get("near_miss_ticks", near_miss_ticks))
    if base_metrics is not None:
        feasible = feasible and collision_ticks <= int(base_metrics.get("collision_ticks", collision_ticks))

    return {
        "score": round(float(score), 6),
        "feasible": bool(feasible),
        "collision_ticks": collision_ticks,
        "near_miss_ticks": near_miss_ticks,
        "min_clearance_m": _round_or_none(min_clearance),
        "completion_rate_mean": _round_or_none(completion_mean),
        "total_reward_mean": _round_or_none(reward_mean),
        "api_error_count": int(api_error_count),
        "finite": bool(finite),
        "row_count": int(len(rows)),
    }


def _evaluate_candidate(
    *,
    candidate_id: str,
    generation: int,
    candidate_index: int,
    spec: dict[str, Any],
    lanes: list[ValidationLane],
    scenario_paths: dict[str, Path],
    seeds: list[int] | None,
    max_steps: int | None,
    objective: dict[str, Any],
    base_metrics: dict[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    policy = ModelPredictPolicyAdapter(FrozenMlpPolicyModel(spec), deterministic=True, clip=True)
    rows: list[dict[str, Any]] = []
    for lane in lanes:
        for seed in _seed_list_for_lane(lane, seeds):
            env = DaaParallelEnv(
                scenario_path=str(scenario_paths[lane.lane_id]),
                n_agents=int(lane.n_agents),
                seed=int(seed),
                comm_profile=str(lane.comm_profile),
            )
            try:
                row = rollout_parallel_env(
                    env,
                    policy,
                    seed=int(seed),
                    max_steps=max_steps,
                    metadata={
                        "suite": str(lane.suite),
                        "scenario": Path(str(scenario_paths[lane.lane_id])).stem,
                        "policy": str(candidate_id),
                        "n_agents": int(lane.n_agents),
                        "comm_profile": str(lane.comm_profile),
                    },
                )
            except Exception as exc:  # pragma: no cover - failure path.
                row = {
                    "suite": str(lane.suite),
                    "scenario": Path(str(scenario_paths[lane.lane_id])).stem,
                    "dimension": "unknown",
                    "policy": str(candidate_id),
                    "n_agents": int(lane.n_agents),
                    "seed": int(seed),
                    "comm_profile": str(lane.comm_profile),
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
            rows.append(
                {
                    "candidate_id": str(candidate_id),
                    "generation": int(generation),
                    "candidate_index": int(candidate_index),
                    **row,
                }
            )
    aggregate = _aggregate_rollout_rows(rows, objective=objective, base_metrics=base_metrics)
    aggregate.update(
        {
            "candidate_id": str(candidate_id),
            "generation": int(generation),
            "candidate_index": int(candidate_index),
        }
    )
    return aggregate, rows


def _candidate_sort_key(row: dict[str, Any]) -> tuple[int, float, int, int, float]:
    return (
        0 if bool(row.get("feasible")) else 1,
        float(row.get("score", float("inf"))),
        int(row.get("collision_ticks", 10**9)),
        int(row.get("near_miss_ticks", 10**9)),
        -float(row.get("completion_rate_mean", 0.0) or 0.0),
    )


def _policy_spec_payload(*, artifact_name: str, policy_name: str) -> dict[str, Any]:
    return {
        "schema_version": RL_POLICY_SPEC_SCHEMA_VERSION,
        "policy_name": str(policy_name),
        "description": "Closed-loop fine-tuned MLP policy trained through the DAA Microbench RL rollout interface.",
        "adapter": "mlp_json",
        "artifact_path": str(artifact_name),
        "deterministic": True,
        "clip": True,
    }


def _training_disclosure(
    *,
    base_policy_spec: str | Path,
    base_model_artifact: str | Path,
    lanes: list[ValidationLane],
    lane_profile: str,
    seeds: list[int] | None,
    objective: dict[str, Any],
    candidate_count: int,
    accepted_count: int,
    trainable_parameters: str,
    holdout_config: dict[str, Any],
) -> dict[str, Any]:
    updated = (
        "MLP output layer weights and bias only; feature extractor and normalization are inherited"
        if trainable_parameters == "output_head"
        else "all MLP layer weights and biases; feature normalization is inherited"
    )
    return {
        "schema_version": CLOSED_LOOP_TRAINING_SCHEMA_VERSION,
        "recipe": "python -m microbench.cli learned-closed-loop-finetune",
        "source": "DAA Microbench closed-loop PettingZoo-style RL rollouts",
        "base_policy_spec": str(base_policy_spec),
        "base_model_artifact": str(base_model_artifact),
        "public_observations_only": True,
        "privileged_global_state": False,
        "lane_profile": str(lane_profile),
        "training_lanes": [lane.lane_id for lane in lanes],
        "training_scenarios": [lane.scenario for lane in lanes],
        "training_seed_override": None if seeds is None else [int(seed) for seed in seeds],
        "rollout_policy": "candidate_mlp_json_policy",
        "objective": objective,
        "candidate_count": int(candidate_count),
        "accepted_generation_count": int(accepted_count),
        "optimizer": "bounded evolutionary MLP parameter search",
        "trainable_parameters": str(trainable_parameters),
        "updated_parameters": updated,
        "holdout": holdout_config,
        "external_data": "none",
        "pretrained_models": "base DAA Microbench mlp_json policy spec",
        "hardware": "local CPU",
    }


def fine_tune_closed_loop_policy(
    *,
    out_dir: str | Path,
    base_policy_spec: str | Path,
    lanes: tuple[str, ...] | list[str] | None = None,
    lane_profile: str = "validation",
    seeds: tuple[int, ...] | list[int] | None = None,
    train_max_steps: int | None = 12,
    generations: int = 2,
    population_size: int = 8,
    trainable_parameters: str = "output_head",
    sigma: float = 0.03,
    sigma_decay: float = 0.5,
    min_delta: float = 1e-6,
    collision_tick_penalty: float = CLOSED_LOOP_OBJECTIVE_DEFAULTS["collision_tick_penalty"],
    near_miss_tick_penalty: float = CLOSED_LOOP_OBJECTIVE_DEFAULTS["near_miss_tick_penalty"],
    clearance_penalty: float = CLOSED_LOOP_OBJECTIVE_DEFAULTS["clearance_penalty"],
    mission_penalty: float = CLOSED_LOOP_OBJECTIVE_DEFAULTS["mission_penalty"],
    reward_weight: float = CLOSED_LOOP_OBJECTIVE_DEFAULTS["reward_weight"],
    min_clearance_m: float = CLOSED_LOOP_OBJECTIVE_DEFAULTS["min_clearance_m"],
    max_collision_ticks: int = int(CLOSED_LOOP_OBJECTIVE_DEFAULTS["max_collision_ticks"]),
    max_near_miss_ticks: int | None = None,
    allow_near_miss_regression: bool = False,
    eval_lanes: tuple[str, ...] | list[str] | None = None,
    eval_max_steps: int | None = 12,
    holdout_profile: str = "none",
    holdout_scenarios: tuple[str, ...] | list[str] | None = None,
    holdout_seeds: tuple[int, ...] | list[int] | None = None,
    holdout_comm_profiles: tuple[str, ...] | list[str] | None = None,
    holdout_n_agents: int = 6,
    holdout_max_runs: int | None = None,
    holdout_score_tolerance: float = CLOSED_LOOP_HOLDOUT_SCORE_TOLERANCE,
    allow_holdout_safety_regression: bool = False,
    allow_holdout_score_regression: bool = False,
    policy_name: str = CLOSED_LOOP_POLICY_NAME,
    seed: int = 37,
    overwrite: bool = False,
    run_validation: bool = True,
) -> dict[str, Any]:
    """Fine-tune a portable MLP learned policy through closed-loop rollouts."""

    out = Path(out_dir)
    model_path = out / "closed_loop_mlp_policy.json"
    spec_path = out / "policy_spec.json"
    report_path = out / "closed_loop_training_report.json"
    candidate_csv = out / "candidate_summary.csv"
    episode_csv = out / "candidate_episodes.csv"
    validation_dir = out / "rl_validation_matrix"
    holdout_dir = out / "broad_3d_holdout"
    if not bool(overwrite):
        existing = [path for path in (model_path, spec_path, report_path, candidate_csv, episode_csv) if path.exists()]
        if validation_dir.exists():
            existing.append(validation_dir)
        if holdout_dir.exists():
            existing.append(holdout_dir)
        if existing:
            raise RuntimeError(f"closed-loop fine-tuning output already exists: {', '.join(str(path) for path in existing)}")
    elif out.exists():
        for path in (model_path, spec_path, report_path, candidate_csv, episode_csv):
            if path.exists():
                path.unlink()
        if validation_dir.exists():
            shutil.rmtree(validation_dir)
        if holdout_dir.exists():
            shutil.rmtree(holdout_dir)
    out.mkdir(parents=True, exist_ok=True)

    if int(generations) < 0:
        raise ValueError("generations must be >= 0")
    if int(population_size) < 1:
        raise ValueError("population_size must be >= 1")
    if float(sigma) < 0.0 or not math.isfinite(float(sigma)):
        raise ValueError("sigma must be finite and >= 0")
    if float(sigma_decay) <= 0.0 or not math.isfinite(float(sigma_decay)):
        raise ValueError("sigma_decay must be finite and > 0")
    if holdout_max_runs is not None and int(holdout_max_runs) < 0:
        raise ValueError("holdout_max_runs must be >= 0 when provided")
    if float(holdout_score_tolerance) < 0.0 or not math.isfinite(float(holdout_score_tolerance)):
        raise ValueError("holdout_score_tolerance must be finite and >= 0")

    trainable_parameters = _trainable_parameters(trainable_parameters)
    holdout_profile = _holdout_profile(holdout_profile)
    selected_holdout_scenarios = _holdout_str_list(holdout_scenarios, CLOSED_LOOP_BROAD_3D_HOLDOUT_SCENARIOS)
    selected_holdout_seeds = _holdout_int_list(holdout_seeds, (0, 1, 2))
    selected_holdout_comm_profiles = _holdout_str_list(holdout_comm_profiles, ("ideal_50hz", "degraded_20hz"))
    holdout_config = {
        "profile": str(holdout_profile),
        "scenarios": selected_holdout_scenarios if holdout_profile != "none" else [],
        "seeds": selected_holdout_seeds if holdout_profile != "none" else [],
        "comm_profiles": selected_holdout_comm_profiles if holdout_profile != "none" else [],
        "n_agents": int(holdout_n_agents),
        "max_runs": None if holdout_max_runs is None else int(holdout_max_runs),
        "score_tolerance": float(holdout_score_tolerance),
        "allow_safety_regression": bool(allow_holdout_safety_regression),
        "allow_score_regression": bool(allow_holdout_score_regression),
    }
    base_spec, base_artifact, wrapper_spec = _load_base_mlp_spec(base_policy_spec)
    selected = selected_closed_loop_training_lanes(list(lanes) if lanes is not None else None, lane_profile=str(lane_profile))
    seed_override = None if seeds is None else [int(value) for value in seeds]
    scenario_paths = prepare_validation_lane_scenarios(out_dir=out / "_closed_loop_training_scenarios", lanes=selected)
    objective = _objective_config(
        collision_tick_penalty=float(collision_tick_penalty),
        near_miss_tick_penalty=float(near_miss_tick_penalty),
        clearance_penalty=float(clearance_penalty),
        mission_penalty=float(mission_penalty),
        reward_weight=float(reward_weight),
        min_clearance_m=float(min_clearance_m),
        max_collision_ticks=int(max_collision_ticks),
        max_near_miss_ticks=max_near_miss_ticks,
        allow_near_miss_regression=bool(allow_near_miss_regression),
    )

    rng = np.random.default_rng(int(seed) + 17041)
    base_vector = _parameter_vector(base_spec, trainable_parameters)
    best_vector = base_vector.copy()
    best_spec = _copy_with_parameter_vector(base_spec, best_vector, trainable_parameters)
    base_metrics, base_episode_rows = _evaluate_candidate(
        candidate_id="base",
        generation=0,
        candidate_index=0,
        spec=best_spec,
        lanes=selected,
        scenario_paths=scenario_paths,
        seeds=seed_override,
        max_steps=train_max_steps,
        objective=objective,
        base_metrics=None,
    )
    base_metrics["feasible"] = bool(
        base_metrics["api_error_count"] == 0
        and base_metrics["finite"]
        and int(base_metrics["collision_ticks"]) <= int(objective["max_collision_ticks"])
        and float(base_metrics.get("min_clearance_m") or float("-inf")) >= float(objective["min_clearance_m"])
    )
    if objective.get("max_near_miss_ticks") is not None:
        base_metrics["feasible"] = bool(base_metrics["feasible"] and int(base_metrics["near_miss_ticks"]) <= int(objective["max_near_miss_ticks"]))

    candidate_rows: list[dict[str, Any]] = [
        {
            **base_metrics,
            "parent_candidate_id": "",
            "accepted": True,
            "score_delta_vs_best_before": 0.0,
            "sigma": 0.0,
        }
    ]
    episode_rows: list[dict[str, Any]] = [
        {**row, "accepted": True, "feasible": bool(base_metrics["feasible"])} for row in base_episode_rows
    ]
    best_metrics = dict(base_metrics)
    best_candidate_id = "base"
    accepted_count = 0
    current_sigma = float(sigma)

    for generation in range(1, int(generations) + 1):
        generation_rows: list[tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]]] = []
        best_score_before = float(best_metrics["score"])
        for candidate_index in range(int(population_size)):
            candidate_id = f"g{generation:03d}_c{candidate_index:03d}"
            noise = rng.normal(0.0, current_sigma, size=best_vector.shape)
            vector = best_vector + noise
            candidate_spec = _copy_with_parameter_vector(base_spec, vector, trainable_parameters)
            metrics, rows = _evaluate_candidate(
                candidate_id=candidate_id,
                generation=generation,
                candidate_index=candidate_index,
                spec=candidate_spec,
                lanes=selected,
                scenario_paths=scenario_paths,
                seeds=seed_override,
                max_steps=train_max_steps,
                objective=objective,
                base_metrics=base_metrics,
            )
            metrics.update(
                {
                    "parent_candidate_id": best_candidate_id,
                    "accepted": False,
                    "score_delta_vs_best_before": _round_or_none(float(metrics["score"]) - best_score_before),
                    "sigma": _round_or_none(current_sigma),
                }
            )
            generation_rows.append((metrics, vector, rows))

        chosen: tuple[dict[str, Any], np.ndarray, list[dict[str, Any]]] | None = None
        for metrics, vector, rows in sorted(generation_rows, key=lambda item: _candidate_sort_key(item[0])):
            if bool(metrics.get("feasible")) and float(metrics["score"]) < best_score_before - float(min_delta):
                chosen = (metrics, vector, rows)
                break
        if chosen is not None:
            chosen_metrics, chosen_vector, chosen_rows = chosen
            best_vector = chosen_vector.copy()
            best_spec = _copy_with_parameter_vector(base_spec, best_vector, trainable_parameters)
            best_metrics = dict(chosen_metrics)
            best_candidate_id = str(chosen_metrics["candidate_id"])
            accepted_count += 1
            for metrics, _vector, rows in generation_rows:
                accepted = str(metrics["candidate_id"]) == best_candidate_id
                metrics["accepted"] = bool(accepted)
                candidate_rows.append(metrics)
                episode_rows.extend({**row, "accepted": bool(accepted), "feasible": bool(metrics["feasible"])} for row in rows)
            _ = chosen_rows
        else:
            candidate_rows.extend(metrics for metrics, _vector, _rows in generation_rows)
            for metrics, _vector, rows in generation_rows:
                episode_rows.extend({**row, "accepted": False, "feasible": bool(metrics["feasible"])} for row in rows)
            current_sigma *= float(sigma_decay)

    final_spec = copy.deepcopy(best_spec)
    final_spec["display_name"] = "Closed-loop fine-tuned MLP learned-policy baseline"
    training = _training_disclosure(
        base_policy_spec=base_policy_spec,
        base_model_artifact=base_artifact,
        lanes=selected,
        lane_profile=str(lane_profile),
        seeds=seed_override,
        objective=objective,
        candidate_count=len(candidate_rows),
        accepted_count=accepted_count,
        trainable_parameters=trainable_parameters,
        holdout_config=holdout_config,
    )
    training.update(
        {
            "base_metrics": base_metrics,
            "best_metrics": best_metrics,
            "best_candidate_id": best_candidate_id,
            "base_policy_name": wrapper_spec.get("policy_name"),
        }
    )
    prior_training = final_spec.get("training")
    if isinstance(prior_training, dict):
        training["base_training"] = prior_training
    final_spec["training"] = training
    _write_json(model_path, final_spec)
    _write_json(spec_path, _policy_spec_payload(artifact_name=model_path.name, policy_name=str(policy_name)))
    _write_csv(candidate_csv, candidate_rows, CLOSED_LOOP_CANDIDATE_FIELDS)
    _write_csv(episode_csv, episode_rows, CLOSED_LOOP_EPISODE_FIELDS)

    validation_report = None
    if run_validation:
        validation_report = run_rl_validation_matrix(
            out_dir=validation_dir,
            policy_spec=spec_path,
            lanes=list(eval_lanes) if eval_lanes is not None else _canonical_eval_lanes_for_training_lanes(selected),
            max_steps=eval_max_steps,
        )

    holdout_report = None
    if holdout_profile == "broad_3d_stress":
        holdout_report = _run_broad_3d_holdout(
            out_dir=holdout_dir,
            base_policy_spec=base_policy_spec,
            tuned_policy_spec=spec_path,
            scenarios=selected_holdout_scenarios,
            seeds=selected_holdout_seeds,
            comm_profiles=selected_holdout_comm_profiles,
            n_agents=int(holdout_n_agents),
            max_runs=holdout_max_runs,
            score_tolerance=float(holdout_score_tolerance),
            allow_safety_regression=bool(allow_holdout_safety_regression),
            allow_score_regression=bool(allow_holdout_score_regression),
        )
        final_spec["training"]["holdout_result"] = {
            "profile": holdout_report["profile"],
            "promotion_candidate": bool(holdout_report["promotion_candidate"]),
            "comparison_csv": holdout_report["comparison_csv"],
            "comparison_report": str(holdout_dir / "comparison_report.json"),
            "base": {k: v for k, v in holdout_report["base"].items() if k != "scored_summary_rows"},
            "tuned": {k: v for k, v in holdout_report["tuned"].items() if k != "scored_summary_rows"},
        }
        _write_json(model_path, final_spec)

    no_collision_regression = int(best_metrics["collision_ticks"]) <= int(base_metrics["collision_ticks"])
    no_near_miss_regression = int(best_metrics["near_miss_ticks"]) <= int(base_metrics["near_miss_ticks"])
    clearance_not_worse = float(best_metrics.get("min_clearance_m") or float("-inf")) >= float(base_metrics.get("min_clearance_m") or float("-inf")) - 1e-9
    checks = [
        _check(
            "base_policy_supported",
            base_spec.get("schema_version") == LEARNED_BASELINE_SCHEMA_VERSION and base_spec.get("model_id") == MLP_LEARNED_MODEL_ID,
            {"base_policy_spec": str(base_policy_spec), "base_model_artifact": str(base_artifact), "adapter": wrapper_spec.get("adapter")},
        ),
        _check("candidate_evaluations_ran", len(candidate_rows) >= 1, {"candidate_count": len(candidate_rows), "episode_rows": len(episode_rows)}),
        _check("candidate_metrics_finite", all(math.isfinite(float(row.get("score", float("nan")))) for row in candidate_rows), {}),
        _check("output_policy_written", bool(model_path.exists() and spec_path.exists()), {"model_artifact": str(model_path), "policy_spec": str(spec_path)}),
        _check(
            "final_no_collision_regression",
            no_collision_regression,
            {"base_collision_ticks": base_metrics["collision_ticks"], "best_collision_ticks": best_metrics["collision_ticks"]},
            severity="behavior",
        ),
        _check(
            "final_no_near_miss_regression",
            no_near_miss_regression,
            {"base_near_miss_ticks": base_metrics["near_miss_ticks"], "best_near_miss_ticks": best_metrics["near_miss_ticks"]},
            severity="behavior",
        ),
        _check(
            "final_clearance_not_worse",
            clearance_not_worse,
            {"base_min_clearance_m": base_metrics.get("min_clearance_m"), "best_min_clearance_m": best_metrics.get("min_clearance_m")},
            severity="behavior",
        ),
    ]
    if validation_report is not None:
        checks.append(
            _check(
                "validation_matrix_gate_pass",
                bool(validation_report.get("ok")),
                {"run_count": validation_report.get("run_count"), "behavior_pass": validation_report.get("behavior_pass")},
            )
        )
    if holdout_report is not None:
        checks.extend(holdout_report["checks"])

    gate_pass = all(check["ok"] for check in checks if check["severity"] == "gate")
    behavior_pass = all(check["ok"] for check in checks if check["severity"] == "behavior")
    promotion_candidate = bool(
        gate_pass
        and behavior_pass
        and holdout_report is not None
        and bool(holdout_report.get("promotion_candidate"))
    )
    report = {
        "schema_version": CLOSED_LOOP_TRAINING_SCHEMA_VERSION,
        "ok": bool(gate_pass),
        "behavior_pass": bool(behavior_pass),
        "promotion_candidate": bool(promotion_candidate),
        "promotion_status": "candidate" if promotion_candidate else ("not_evaluated_without_holdout" if holdout_report is None else "review_required"),
        "policy_name": str(policy_name),
        "out_dir": str(out),
        "base_policy_spec": str(base_policy_spec),
        "base_model_artifact": str(base_artifact),
        "model_artifact": str(model_path),
        "policy_spec": str(spec_path),
        "training_report": str(report_path),
        "candidate_summary_csv": str(candidate_csv),
        "candidate_episodes_csv": str(episode_csv),
        "training_lanes": [asdict(lane) for lane in selected],
        "seeds": None if seed_override is None else [int(value) for value in seed_override],
        "lane_profile": str(lane_profile),
        "train_max_steps": None if train_max_steps is None else int(train_max_steps),
        "generations": int(generations),
        "population_size": int(population_size),
        "trainable_parameters": str(trainable_parameters),
        "sigma": float(sigma),
        "sigma_decay": float(sigma_decay),
        "min_delta": float(min_delta),
        "objective": objective,
        "candidate_count": len(candidate_rows),
        "accepted_generation_count": int(accepted_count),
        "best_candidate_id": best_candidate_id,
        "base_metrics": base_metrics,
        "best_metrics": best_metrics,
        "validation_matrix": validation_report,
        "holdout": holdout_report,
        "checks": checks,
    }
    _write_json(report_path, report)
    return report


__all__ = [
    "CLOSED_LOOP_CANDIDATE_FIELDS",
    "CLOSED_LOOP_BROAD_3D_HOLDOUT_SCENARIOS",
    "CLOSED_LOOP_EPISODE_FIELDS",
    "CLOSED_LOOP_HOLDOUT_PROFILE_CHOICES",
    "CLOSED_LOOP_HOLDOUT_SCORE_TOLERANCE",
    "CLOSED_LOOP_OBJECTIVE_DEFAULTS",
    "CLOSED_LOOP_POLICY_NAME",
    "CLOSED_LOOP_BROAD_3D_TRAINING_LANE_IDS",
    "CLOSED_LOOP_TRAINING_LANE_PROFILE_CHOICES",
    "CLOSED_LOOP_TRAINABLE_PARAMETER_CHOICES",
    "CLOSED_LOOP_TRAINING_SCHEMA_VERSION",
    "fine_tune_closed_loop_policy",
    "selected_closed_loop_training_lanes",
]
