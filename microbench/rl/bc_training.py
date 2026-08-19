from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np

from microbench.learned import (
    LEARNED_BASELINE_SCHEMA_VERSION,
    MLP_LEARNED_FEATURE_NAMES,
    MLP_LEARNED_MODEL_ID,
    observation_to_mlp_features,
)
from microbench.rl.envs import DaaParallelEnv
from microbench.rl.policy_spec import RL_POLICY_SPEC_SCHEMA_VERSION
from microbench.rl.validation_matrix import run_rl_validation_matrix
from microbench.tools.baseline_validation_matrix import (
    ValidationLane,
    prepare_validation_lane_scenarios,
    selected_validation_lanes,
)


BC_TRAINING_SCHEMA_VERSION = "0.1"
BC_POLICY_NAME = "bc_mlp_learned"
BC_TEACHER_NAME = "local_avoidance_teacher_v0"
DEFAULT_BC_LANES = (
    "head_on",
    "crossing",
    "urban_obstacle",
    "communication_delay",
    "high_n_dense_merge",
)


def _normalize(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.zeros(3, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def _teacher_action_from_features(features: np.ndarray) -> np.ndarray:
    """Transparent local DAA teacher over public learned-policy features."""

    x = np.asarray(features, dtype=np.float32).reshape(-1)
    goal = _normalize(x[0:3])
    ego_vel = np.clip(x[3:6], -1.0, 1.0)
    avoid = np.clip(x[6:9], -1.0, 1.0)
    rel_vel = np.clip(x[9:12], -1.0, 1.0)
    threat = float(np.clip(x[12], 0.0, 1.0))
    neighbor_frac = float(np.clip(x[13], 0.0, 1.0))

    lateral = np.cross(goal, avoid).astype(np.float32)
    lateral_norm = float(np.linalg.norm(lateral))
    if lateral_norm > 1e-6:
        lateral = lateral / lateral_norm
    else:
        lateral = np.zeros(3, dtype=np.float32)

    goal_gain = 1.0 - 0.55 * threat
    avoid_gain = 1.25 + 1.65 * threat + 0.35 * neighbor_frac
    action = (
        goal_gain * goal
        - 0.18 * ego_vel
        + avoid_gain * avoid
        - 0.10 * rel_vel
        + 0.18 * threat * lateral
    )
    norm = float(np.linalg.norm(action))
    if norm > 1.0:
        action = action / norm
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _teacher_action_from_observation(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    top_k = max(0, (obs.shape[0] - 17) // 9)
    features = observation_to_mlp_features(obs, top_k=top_k)
    return features, _teacher_action_from_features(features)


def _fit_random_feature_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    hidden_dim: int,
    ridge: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(int(seed) + 1009)
    x = np.asarray(features, dtype=np.float64)
    y = np.arctanh(np.clip(np.asarray(labels, dtype=np.float64), -0.999, 0.999))
    w1 = rng.normal(0.0, 0.75 / math.sqrt(max(1, x.shape[1])), size=(int(hidden_dim), x.shape[1]))
    b1 = rng.normal(0.0, 0.08, size=(int(hidden_dim),))
    hidden = np.tanh(x @ w1.T + b1)
    hidden_aug = np.concatenate([hidden, np.ones((hidden.shape[0], 1), dtype=np.float64)], axis=1)
    reg = float(ridge) * np.eye(hidden_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(hidden_aug.T @ hidden_aug + reg, hidden_aug.T @ y)
    w2 = coef[:-1, :].T
    b2 = coef[-1, :]
    pred = np.tanh(hidden @ w2.T + b2)
    rmse = float(np.sqrt(np.mean((pred - labels) ** 2)))
    return w1.astype(float), b1.astype(float), w2.astype(float), b2.astype(float), rmse


def _seed_list_for_lane(lane: ValidationLane, seeds: list[int] | None) -> list[int]:
    return [int(lane.seed)] if seeds is None else [int(seed) for seed in seeds]


def _rollout_training_lane(
    *,
    scenario_path: Path,
    lane: ValidationLane,
    seed: int,
    max_steps: int,
    rollout_noise_std: float,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    rng = np.random.default_rng(int(seed) + 7301)
    env = DaaParallelEnv(
        scenario_path=str(scenario_path),
        n_agents=int(lane.n_agents),
        seed=int(seed),
        comm_profile=str(lane.comm_profile),
    )
    features: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    steps = 0
    try:
        observations, infos = env.reset(seed=int(seed))
        _ = infos
        while env.agents and steps < int(max_steps):
            actions: dict[str, np.ndarray] = {}
            for agent in env.agents:
                x, label = _teacher_action_from_observation(observations[agent])
                features.append(x)
                labels.append(label)
                action = label
                if rollout_noise_std > 0.0:
                    action = np.clip(
                        action + rng.normal(0.0, float(rollout_noise_std), size=3).astype(np.float32),
                        -1.0,
                        1.0,
                    )
                actions[agent] = action.astype(np.float32)
            observations, rewards, terminations, truncations, infos = env.step(actions)
            _ = rewards, terminations, truncations, infos
            steps += 1
    finally:
        env.close()

    return features, labels, {
        "lane_id": lane.lane_id,
        "category": lane.category,
        "scenario": str(scenario_path),
        "seed": int(seed),
        "n_agents": int(lane.n_agents),
        "comm_profile": lane.comm_profile,
        "steps": int(steps),
        "samples": int(len(features)),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _model_spec(
    *,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    hidden_dim: int,
    ridge: float,
    seed: int,
    fit_rmse: float,
    training_rows: list[dict[str, Any]],
    lanes: list[ValidationLane],
    seeds: list[int] | None,
    max_steps: int,
    rollout_noise_std: float,
    sample_count: int,
) -> dict[str, Any]:
    return {
        "schema_version": LEARNED_BASELINE_SCHEMA_VERSION,
        "model_id": MLP_LEARNED_MODEL_ID,
        "display_name": "Behavior-cloned MLP learned-policy baseline",
        "model_type": "mlp_tanh_policy",
        "hidden_dim": int(hidden_dim),
        "action_shape": [3],
        "input_features": list(MLP_LEARNED_FEATURE_NAMES),
        "hidden_activation": "tanh",
        "output_activation": "tanh",
        "layer1_weights": np.asarray(w1).round(8).tolist(),
        "layer1_bias": np.asarray(b1).round(8).tolist(),
        "layer2_weights": np.asarray(w2).round(8).tolist(),
        "layer2_bias": np.asarray(b2).round(8).tolist(),
        "training": {
            "schema_version": BC_TRAINING_SCHEMA_VERSION,
            "recipe": "python -m microbench.cli train-learned-bc",
            "source": "DAA Microbench RL validation-matrix rollouts",
            "teacher_policy": BC_TEACHER_NAME,
            "public_observations_only": True,
            "privileged_global_state": False,
            "training_lanes": [lane.lane_id for lane in lanes],
            "training_scenarios": [lane.scenario for lane in lanes],
            "training_seed_override": None if seeds is None else [int(seed_value) for seed_value in seeds],
            "rollout_policy": "teacher_with_optional_clipped_action_noise",
            "max_steps_per_episode": int(max_steps),
            "rollout_noise_std": float(rollout_noise_std),
            "random_feature_seed": int(seed),
            "samples": int(sample_count),
            "episodes": int(len(training_rows)),
            "feature_dim": int(len(MLP_LEARNED_FEATURE_NAMES)),
            "hidden_dim": int(hidden_dim),
            "ridge": float(ridge),
            "fit_rmse": round(float(fit_rmse), 8),
            "lane_rows": training_rows,
        },
    }


def _policy_spec(*, artifact_name: str, policy_name: str) -> dict[str, Any]:
    return {
        "schema_version": RL_POLICY_SPEC_SCHEMA_VERSION,
        "policy_name": str(policy_name),
        "description": "Behavior-cloned MLP policy trained from DAA Microbench public RL observations.",
        "adapter": "mlp_json",
        "artifact_path": str(artifact_name),
        "deterministic": True,
        "clip": True,
    }


def train_behavior_cloned_policy(
    *,
    out_dir: str | Path,
    lanes: tuple[str, ...] | list[str] | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    max_steps: int = 64,
    hidden_dim: int = 32,
    ridge: float = 1e-4,
    rollout_noise_std: float = 0.03,
    eval_lanes: tuple[str, ...] | list[str] | None = None,
    eval_max_steps: int | None = 12,
    policy_name: str = BC_POLICY_NAME,
    seed: int = 29,
    overwrite: bool = False,
    run_validation: bool = True,
) -> dict[str, Any]:
    """Train a portable behavior-cloned MLP policy from canonical RL lanes."""

    out = Path(out_dir)
    model_path = out / "bc_mlp_policy.json"
    spec_path = out / "policy_spec.json"
    report_path = out / "bc_training_report.json"
    validation_dir = out / "rl_validation_matrix"
    if not bool(overwrite):
        checked_paths = (
            model_path,
            spec_path,
            report_path,
            validation_dir / "rl_validation_matrix_episodes.csv",
        )
        existing = [path for path in checked_paths if path.exists()]
        if existing:
            raise RuntimeError(f"behavior-cloning output already exists: {', '.join(str(path) for path in existing)}")
    elif validation_dir.exists():
        shutil.rmtree(validation_dir)

    selected = selected_validation_lanes(list(lanes) if lanes is not None else list(DEFAULT_BC_LANES))
    seed_override = None if seeds is None else [int(seed_value) for seed_value in seeds]
    scenario_paths = prepare_validation_lane_scenarios(out_dir=out / "_bc_training_scenarios", lanes=selected)

    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    training_rows: list[dict[str, Any]] = []
    for lane in selected:
        for seed_value in _seed_list_for_lane(lane, seed_override):
            lane_features, lane_labels, row = _rollout_training_lane(
                scenario_path=scenario_paths[lane.lane_id],
                lane=lane,
                seed=int(seed_value),
                max_steps=int(max_steps),
                rollout_noise_std=float(rollout_noise_std),
            )
            feature_rows.extend(lane_features)
            label_rows.extend(lane_labels)
            training_rows.append(row)

    if not feature_rows:
        raise RuntimeError("behavior-cloning data collection produced no samples")

    features = np.vstack(feature_rows).astype(np.float32)
    labels = np.vstack(label_rows).astype(np.float32)
    w1, b1, w2, b2, fit_rmse = _fit_random_feature_mlp(
        features,
        labels,
        seed=int(seed),
        hidden_dim=int(hidden_dim),
        ridge=float(ridge),
    )
    model_payload = _model_spec(
        w1=w1,
        b1=b1,
        w2=w2,
        b2=b2,
        hidden_dim=int(hidden_dim),
        ridge=float(ridge),
        seed=int(seed),
        fit_rmse=fit_rmse,
        training_rows=training_rows,
        lanes=selected,
        seeds=seed_override,
        max_steps=int(max_steps),
        rollout_noise_std=float(rollout_noise_std),
        sample_count=int(features.shape[0]),
    )
    spec_payload = _policy_spec(artifact_name=model_path.name, policy_name=str(policy_name))
    _write_json(model_path, model_payload)
    _write_json(spec_path, spec_payload)

    validation_report = None
    if run_validation:
        validation_report = run_rl_validation_matrix(
            out_dir=validation_dir,
            policy_spec=spec_path,
            lanes=list(eval_lanes) if eval_lanes is not None else [lane.lane_id for lane in selected],
            max_steps=eval_max_steps,
        )

    checks = [
        {
            "name": "samples_collected",
            "ok": bool(features.shape[0] > 0),
            "details": {"samples": int(features.shape[0])},
        },
        {
            "name": "finite_dataset",
            "ok": bool(np.all(np.isfinite(features)) and np.all(np.isfinite(labels))),
            "details": {"feature_shape": list(features.shape), "label_shape": list(labels.shape)},
        },
        {
            "name": "fit_rmse_finite",
            "ok": math.isfinite(float(fit_rmse)),
            "details": {"fit_rmse": round(float(fit_rmse), 8)},
        },
        {
            "name": "policy_spec_written",
            "ok": bool(spec_path.exists() and model_path.exists()),
            "details": {"policy_spec": str(spec_path), "model_artifact": str(model_path)},
        },
    ]
    if validation_report is not None:
        checks.append(
            {
                "name": "validation_matrix_gate_pass",
                "ok": bool(validation_report.get("ok")),
                "details": {
                    "policy": validation_report.get("policy"),
                    "run_count": validation_report.get("run_count"),
                    "behavior_pass": validation_report.get("behavior_pass"),
                },
            }
        )

    report = {
        "schema_version": BC_TRAINING_SCHEMA_VERSION,
        "ok": all(check["ok"] for check in checks),
        "policy_name": str(policy_name),
        "teacher_policy": BC_TEACHER_NAME,
        "public_observations_only": True,
        "privileged_global_state": False,
        "out_dir": str(out),
        "model_artifact": str(model_path),
        "policy_spec": str(spec_path),
        "training_report": str(report_path),
        "sample_count": int(features.shape[0]),
        "feature_dim": int(features.shape[1]),
        "label_dim": int(labels.shape[1]),
        "hidden_dim": int(hidden_dim),
        "fit_rmse": round(float(fit_rmse), 8),
        "training_rows": training_rows,
        "validation_matrix": validation_report,
        "checks": checks,
    }
    _write_json(report_path, report)
    return report


__all__ = [
    "BC_POLICY_NAME",
    "BC_TEACHER_NAME",
    "BC_TRAINING_SCHEMA_VERSION",
    "DEFAULT_BC_LANES",
    "train_behavior_cloned_policy",
]
