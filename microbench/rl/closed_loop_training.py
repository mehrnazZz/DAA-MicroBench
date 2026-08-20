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
from microbench.tools.baseline_validation_matrix import (
    ValidationLane,
    prepare_validation_lane_scenarios,
    selected_validation_lanes,
)


CLOSED_LOOP_TRAINING_SCHEMA_VERSION = "0.1"
CLOSED_LOOP_POLICY_NAME = "closed_loop_mlp_learned"
CLOSED_LOOP_TRAINABLE_PARAMETER_CHOICES = ("output_head", "all_layers")
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
    seeds: list[int] | None,
    objective: dict[str, Any],
    candidate_count: int,
    accepted_count: int,
    trainable_parameters: str,
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
        "external_data": "none",
        "pretrained_models": "base DAA Microbench mlp_json policy spec",
        "hardware": "local CPU",
    }


def fine_tune_closed_loop_policy(
    *,
    out_dir: str | Path,
    base_policy_spec: str | Path,
    lanes: tuple[str, ...] | list[str] | None = None,
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
    if not bool(overwrite):
        existing = [path for path in (model_path, spec_path, report_path, candidate_csv, episode_csv) if path.exists()]
        if validation_dir.exists():
            existing.append(validation_dir)
        if existing:
            raise RuntimeError(f"closed-loop fine-tuning output already exists: {', '.join(str(path) for path in existing)}")
    elif out.exists():
        for path in (model_path, spec_path, report_path, candidate_csv, episode_csv):
            if path.exists():
                path.unlink()
        if validation_dir.exists():
            shutil.rmtree(validation_dir)
    out.mkdir(parents=True, exist_ok=True)

    if int(generations) < 0:
        raise ValueError("generations must be >= 0")
    if int(population_size) < 1:
        raise ValueError("population_size must be >= 1")
    if float(sigma) < 0.0 or not math.isfinite(float(sigma)):
        raise ValueError("sigma must be finite and >= 0")
    if float(sigma_decay) <= 0.0 or not math.isfinite(float(sigma_decay)):
        raise ValueError("sigma_decay must be finite and > 0")

    trainable_parameters = _trainable_parameters(trainable_parameters)
    base_spec, base_artifact, wrapper_spec = _load_base_mlp_spec(base_policy_spec)
    selected = selected_validation_lanes(list(lanes) if lanes is not None else None)
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
        seeds=seed_override,
        objective=objective,
        candidate_count=len(candidate_rows),
        accepted_count=accepted_count,
        trainable_parameters=trainable_parameters,
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
            lanes=list(eval_lanes) if eval_lanes is not None else [lane.lane_id for lane in selected],
            max_steps=eval_max_steps,
        )

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

    gate_pass = all(check["ok"] for check in checks if check["severity"] == "gate")
    behavior_pass = all(check["ok"] for check in checks if check["severity"] == "behavior")
    report = {
        "schema_version": CLOSED_LOOP_TRAINING_SCHEMA_VERSION,
        "ok": bool(gate_pass),
        "behavior_pass": bool(behavior_pass),
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
        "checks": checks,
    }
    _write_json(report_path, report)
    return report


__all__ = [
    "CLOSED_LOOP_CANDIDATE_FIELDS",
    "CLOSED_LOOP_EPISODE_FIELDS",
    "CLOSED_LOOP_OBJECTIVE_DEFAULTS",
    "CLOSED_LOOP_POLICY_NAME",
    "CLOSED_LOOP_TRAINABLE_PARAMETER_CHOICES",
    "CLOSED_LOOP_TRAINING_SCHEMA_VERSION",
    "fine_tune_closed_loop_policy",
]
