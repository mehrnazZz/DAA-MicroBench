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
BC_EVIDENCE_SCHEMA_VERSION = "0.1"
BC_POLICY_NAME = "bc_mlp_learned"
BC_TEACHER_NAME = "local_lateral_avoidance_teacher_v0"
DEFAULT_BC_LANES = (
    "head_on",
    "crossing",
    "urban_obstacle",
    "communication_delay",
    "high_n_dense_merge",
)
BC_FIXTURE_BUNDLE_CONFIGS = (
    {"label": "tiny", "method": "learned_tiny", "policy": "tiny_learned"},
    {"label": "mlp", "method": "learned_mlp", "policy": "mlp_learned"},
)
BC_FEATURE_NORMALIZATION_CHOICES = ("standard", "none")
BC_FEATURE_NORMALIZATION_CLIP = 5.0
BC_FEATURE_NORMALIZATION_EPS = 1e-6


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

    avoid_lateral = avoid - float(np.dot(avoid, goal)) * goal
    if float(np.linalg.norm(avoid_lateral)) < 1e-6 and threat > 0.0:
        reference = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(goal, reference))) > 0.9:
            reference = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
        avoid_lateral = np.cross(reference, goal).astype(np.float32) * threat
    avoid_lateral = _normalize(avoid_lateral) * min(1.0, float(np.linalg.norm(avoid)))

    rel_vel_lateral = rel_vel - float(np.dot(rel_vel, goal)) * goal
    lateral = _normalize(avoid_lateral)

    goal_gain = max(0.35, 1.0 - 0.45 * threat)
    avoid_gain = 1.10 + 1.40 * threat + 0.30 * neighbor_frac
    action = (
        goal_gain * goal
        - 0.18 * ego_vel
        + avoid_gain * avoid_lateral
        - 0.08 * rel_vel_lateral
        + 0.12 * threat * lateral
    )
    forward = float(np.dot(action, goal))
    min_forward = 0.20 + 0.25 * (1.0 - threat)
    if forward < min_forward:
        action += (min_forward - forward) * goal
    norm = float(np.linalg.norm(action))
    if norm > 1.0:
        action = action / norm
    return np.clip(action, -1.0, 1.0).astype(np.float32)


def _teacher_action_from_observation(observation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    obs = np.asarray(observation, dtype=np.float32).reshape(-1)
    top_k = max(0, (obs.shape[0] - 17) // 9)
    features = observation_to_mlp_features(obs, top_k=top_k)
    return features, _teacher_action_from_features(features)


def _feature_normalization_payload(features: np.ndarray, *, mode: str = "standard") -> dict[str, Any]:
    normalized_mode = str(mode or "standard").strip().lower()
    if normalized_mode not in BC_FEATURE_NORMALIZATION_CHOICES:
        raise ValueError(
            f"Unsupported BC feature normalization {mode!r}; "
            f"expected one of {','.join(BC_FEATURE_NORMALIZATION_CHOICES)}"
        )
    if normalized_mode == "none":
        return {
            "mode": "none",
            "description": "none; public RL observation features are consumed directly",
        }

    x = np.asarray(features, dtype=np.float32)
    mean = np.mean(x, axis=0).astype(np.float32)
    scale = np.std(x, axis=0).astype(np.float32)
    scale = np.where(scale < BC_FEATURE_NORMALIZATION_EPS, 1.0, scale).astype(np.float32)
    return {
        "mode": "standard",
        "mean": mean.round(8).tolist(),
        "scale": scale.round(8).tolist(),
        "clip": float(BC_FEATURE_NORMALIZATION_CLIP),
        "epsilon": float(BC_FEATURE_NORMALIZATION_EPS),
        "description": "per-feature mean/std fitted on public training features; inference applies the stored transform",
    }


def _apply_feature_normalization(features: np.ndarray, payload: dict[str, Any] | None) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    if not payload or str(payload.get("mode", "none")) == "none":
        return x
    if str(payload.get("mode")) != "standard":
        raise ValueError(f"Unsupported feature normalization mode {payload.get('mode')!r}")
    mean = np.asarray(payload.get("mean"), dtype=np.float32).reshape(-1)
    scale = np.asarray(payload.get("scale"), dtype=np.float32).reshape(-1)
    if mean.shape != (x.shape[1],) or scale.shape != (x.shape[1],):
        raise ValueError(
            f"Feature normalization shape mismatch: mean={mean.shape}, scale={scale.shape}, feature_dim={x.shape[1]}"
        )
    normalized = (x - mean.reshape(1, -1)) / np.maximum(scale.reshape(1, -1), BC_FEATURE_NORMALIZATION_EPS)
    clip_value = payload.get("clip")
    if clip_value is not None:
        normalized = np.clip(normalized, -float(clip_value), float(clip_value))
    return normalized.astype(np.float32)


def _fit_random_feature_mlp(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    hidden_dim: int,
    ridge: float,
    sample_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    rng = np.random.default_rng(int(seed) + 1009)
    x = np.asarray(features, dtype=np.float64)
    y = np.arctanh(np.clip(np.asarray(labels, dtype=np.float64), -0.999, 0.999))
    weights = None
    if sample_weights is not None:
        weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if weights.shape != (x.shape[0],):
            raise ValueError(f"sample_weights must have shape {(x.shape[0],)}, got {weights.shape}")
        if not np.all(np.isfinite(weights)):
            raise ValueError("sample_weights must be finite")
        weights = np.clip(weights, 0.0, None)
        mean_weight = float(np.mean(weights))
        if mean_weight <= 1e-12:
            raise ValueError("sample_weights must contain at least one positive value")
        weights = weights / mean_weight
    w1 = rng.normal(0.0, 0.75 / math.sqrt(max(1, x.shape[1])), size=(int(hidden_dim), x.shape[1]))
    b1 = rng.normal(0.0, 0.08, size=(int(hidden_dim),))
    hidden = np.tanh(x @ w1.T + b1)
    hidden_aug = np.concatenate([hidden, np.ones((hidden.shape[0], 1), dtype=np.float64)], axis=1)
    solve_x = hidden_aug
    solve_y = y
    if weights is not None:
        sqrt_weights = np.sqrt(weights).reshape(-1, 1)
        solve_x = hidden_aug * sqrt_weights
        solve_y = y * sqrt_weights
    reg = float(ridge) * np.eye(hidden_aug.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    coef = np.linalg.solve(solve_x.T @ solve_x + reg, solve_x.T @ solve_y)
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


def _remove_known_evidence_outputs(out: Path) -> None:
    for name in (
        "training",
        "bc_bundle",
        "tiny_bundle",
        "mlp_bundle",
        "bc_manifest_overlay.json",
        "learned_policy_leaderboard.json",
        "learned_policy_leaderboard.csv",
        "learned_policy_diagnostics.json",
        "learned_policy_diagnostics.csv",
        "learned_policy_diagnostics.md",
        "learned_bc_evidence.json",
    ):
        path = out / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


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
    feature_normalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    normalization_payload = feature_normalization or _feature_normalization_payload(
        np.zeros((1, len(MLP_LEARNED_FEATURE_NAMES)), dtype=np.float32),
        mode="none",
    )
    return {
        "schema_version": LEARNED_BASELINE_SCHEMA_VERSION,
        "model_id": MLP_LEARNED_MODEL_ID,
        "display_name": "Behavior-cloned MLP learned-policy baseline",
        "model_type": "mlp_tanh_policy",
        "hidden_dim": int(hidden_dim),
        "action_shape": [3],
        "input_features": list(MLP_LEARNED_FEATURE_NAMES),
        "feature_normalization": normalization_payload,
        "hidden_activation": "tanh",
        "output_activation": "tanh",
        "postprocess": {
            "goal_forward_floor": True,
            "min_forward_base": 0.2,
            "min_forward_free_boost": 0.25,
            "normalize_max_norm": True,
        },
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
            "observation_normalization": normalization_payload.get("description"),
            "action_post_processing": "goal-direction forward floor plus unit-norm clamp before action-space clipping",
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


def _manifest_overlay_from_training_report(report: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in report.get("training_rows", []) if isinstance(row, dict)]
    seeds = sorted({int(row.get("seed", 0) or 0) for row in rows})
    lane_ids = sorted({str(row.get("lane_id")) for row in rows if row.get("lane_id")})
    scenario_names = sorted({Path(str(row.get("scenario", ""))).stem for row in rows if row.get("scenario")})
    environment_steps = sum(int(row.get("steps", 0) or 0) for row in rows)
    samples = int(report.get("sample_count", 0) or 0)
    normalization = dict(report.get("feature_normalization", {}) or {})
    return {
        "dependencies": {
            "inference_packages": [
                {
                    "name": "numpy",
                    "source": "pip",
                    "version": ">=1.24",
                    "purpose": "dependency-free JSON MLP inference",
                }
            ],
            "notes": "Generated by learned-bc-evidence from the behavior-cloning training report.",
        },
        "training_disclosure": {
            "training_scenarios": scenario_names,
            "training_suites": [f"rl_validation_matrix:{','.join(lane_ids)}"] if lane_ids else ["rl_validation_matrix"],
            "environment_steps": int(environment_steps),
            "random_seeds": seeds,
            "observation_normalization": normalization.get("description")
            or "none; public RL observation features are consumed directly",
            "action_post_processing": (
                "artifact-declared goal-direction forward floor and unit-norm clamp, "
                "then normalized velocity action clipped by the DAA Microbench action contract"
            ),
            "reward_configuration": "not used; supervised behavior cloning from transparent local-avoidance teacher labels",
            "external_data": "none",
            "pretrained_models": "none",
            "hardware": "local CPU",
            "teacher_policy": BC_TEACHER_NAME,
            "agent_samples": samples,
            "public_observations_only": bool(report.get("public_observations_only", True)),
            "privileged_global_state": bool(report.get("privileged_global_state", False)),
        },
        "inference_disclosure": {
            "deterministic": True,
            "uses_external_services": False,
            "external_services": [],
            "runtime_notes": "deterministic local CPU inference from portable JSON weights",
        },
        "review_notes": {
            "privileged_information": "none",
            "intended_category": "behavior_cloned_public_observation_baseline",
        },
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
    feature_normalization: str = "standard",
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
    normalization_payload = _feature_normalization_payload(features, mode=str(feature_normalization))
    fit_features = _apply_feature_normalization(features, normalization_payload)
    w1, b1, w2, b2, fit_rmse = _fit_random_feature_mlp(
        fit_features,
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
        feature_normalization=normalization_payload,
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
            "details": {
                "feature_shape": list(features.shape),
                "label_shape": list(labels.shape),
                "feature_normalization": normalization_payload.get("mode"),
            },
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
        "feature_normalization": normalization_payload,
        "fit_rmse": round(float(fit_rmse), 8),
        "training_rows": training_rows,
        "validation_matrix": validation_report,
        "checks": checks,
    }
    _write_json(report_path, report)
    return report


def build_behavior_cloned_policy_evidence(
    *,
    out_dir: str | Path,
    lanes: tuple[str, ...] | list[str] | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    max_steps: int = 64,
    hidden_dim: int = 32,
    ridge: float = 1e-4,
    rollout_noise_std: float = 0.03,
    feature_normalization: str = "standard",
    eval_lanes: tuple[str, ...] | list[str] | None = None,
    eval_max_steps: int | None = 12,
    policy_name: str = BC_POLICY_NAME,
    seed: int = 29,
    bundle_suite: str = "official_smoke_generated",
    bundle_n_agents: int = 4,
    bundle_seeds: tuple[int, ...] | list[int] | None = None,
    bundle_max_steps: int | None = 12,
    bundle_max_runs: int | None = 1,
    include_fixture_bundles: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Train, bundle, and compare a behavior-cloned learned policy."""

    from microbench.rl.learned_diagnostics import write_learned_policy_diagnostics
    from microbench.rl.learned_leaderboard import write_learned_policy_leaderboard
    from microbench.rl.submission_bundle import run_learned_policy_submission_bundle

    out = Path(out_dir)
    evidence_path = out / "learned_bc_evidence.json"
    if evidence_path.exists() and not bool(overwrite):
        raise RuntimeError(f"learned BC evidence output already exists: {evidence_path}")
    if bool(overwrite):
        _remove_known_evidence_outputs(out)
    out.mkdir(parents=True, exist_ok=True)

    training = train_behavior_cloned_policy(
        out_dir=out / "training",
        lanes=lanes,
        seeds=seeds,
        max_steps=int(max_steps),
        hidden_dim=int(hidden_dim),
        ridge=float(ridge),
        rollout_noise_std=float(rollout_noise_std),
        feature_normalization=str(feature_normalization),
        eval_lanes=eval_lanes,
        eval_max_steps=eval_max_steps,
        policy_name=str(policy_name),
        seed=int(seed),
        overwrite=bool(overwrite),
        run_validation=True,
    )
    manifest_overlay = _manifest_overlay_from_training_report(training)
    manifest_overlay_path = _write_json(out / "bc_manifest_overlay.json", manifest_overlay)

    seed_list = [int(value) for value in (bundle_seeds if bundle_seeds is not None else (0,))]
    bundles: dict[str, dict[str, Any]] = {}
    bc_bundle_dir = out / "bc_bundle"
    bundles["bc"] = run_learned_policy_submission_bundle(
        out_dir=bc_bundle_dir,
        method="learned_policy_spec",
        policy="goal_direction",
        policy_spec=str(training["policy_spec"]),
        suite=str(bundle_suite),
        n_agents=int(bundle_n_agents),
        seeds=seed_list,
        max_steps=bundle_max_steps,
        max_runs=bundle_max_runs,
        submission_manifest=manifest_overlay_path,
    )

    if include_fixture_bundles:
        for config in BC_FIXTURE_BUNDLE_CONFIGS:
            bundles[str(config["label"])] = run_learned_policy_submission_bundle(
                out_dir=out / f"{config['label']}_bundle",
                method=str(config["method"]),
                policy=str(config["policy"]),
                suite=str(bundle_suite),
                n_agents=int(bundle_n_agents),
                seeds=seed_list,
                max_steps=bundle_max_steps,
                max_runs=bundle_max_runs,
            )

    bundle_paths = [str(bc_bundle_dir)]
    if include_fixture_bundles:
        bundle_paths.extend(str(out / f"{config['label']}_bundle") for config in BC_FIXTURE_BUNDLE_CONFIGS)
    leaderboard = write_learned_policy_leaderboard(
        bundles=bundle_paths,
        out=out / "learned_policy_leaderboard.json",
    )
    diagnostics = write_learned_policy_diagnostics(
        bundles=bundle_paths,
        out=out / "learned_policy_diagnostics.json",
    )

    checks = [
        {
            "name": "training_ok",
            "ok": bool(training.get("ok")),
            "details": {"sample_count": training.get("sample_count"), "fit_rmse": training.get("fit_rmse")},
        },
        {
            "name": "bc_bundle_ok",
            "ok": bool(bundles["bc"].get("ok")),
            "details": {"path": str(bc_bundle_dir), "policy": bundles["bc"].get("policy")},
        },
        {
            "name": "leaderboard_ok",
            "ok": bool(leaderboard.get("ok")),
            "details": {
                "bundle_count": leaderboard.get("bundle_count"),
                "leaderboard_path": leaderboard.get("leaderboard_path"),
            },
        },
        {
            "name": "diagnostics_ok",
            "ok": bool(diagnostics.get("ok")),
            "details": {
                "bundle_count": diagnostics.get("bundle_count"),
                "diagnostics_path": diagnostics.get("diagnostics_path"),
            },
        },
    ]
    for label, bundle in bundles.items():
        if label == "bc":
            continue
        checks.append(
            {
                "name": f"{label}_fixture_bundle_ok",
                "ok": bool(bundle.get("ok")),
                "details": {"policy": bundle.get("policy"), "method": bundle.get("method")},
            }
        )

    report = {
        "schema_version": BC_EVIDENCE_SCHEMA_VERSION,
        "ok": all(check["ok"] for check in checks),
        "out_dir": str(out),
        "training": {
            "report": training.get("training_report"),
            "policy_spec": training.get("policy_spec"),
            "model_artifact": training.get("model_artifact"),
            "sample_count": training.get("sample_count"),
            "fit_rmse": training.get("fit_rmse"),
            "feature_normalization": training.get("feature_normalization", {}).get("mode"),
        },
        "manifest_overlay": str(manifest_overlay_path),
        "bundle_paths": {label: str(out / f"{label}_bundle") for label in bundles},
        "bundles": {
            label: {
                "ok": bool(bundle.get("ok")),
                "method": bundle.get("method"),
                "policy": bundle.get("policy"),
                "suite": bundle.get("suite"),
                "run_count": bundle.get("planner_sweep", {}).get("run_count"),
                "rl_validation_matrix": bundle.get("rl_validation_matrix"),
            }
            for label, bundle in bundles.items()
        },
        "leaderboard": leaderboard,
        "diagnostics": diagnostics,
        "checks": checks,
    }
    _write_json(evidence_path, report)
    return report


__all__ = [
    "BC_POLICY_NAME",
    "BC_TEACHER_NAME",
    "BC_EVIDENCE_SCHEMA_VERSION",
    "BC_TRAINING_SCHEMA_VERSION",
    "BC_FIXTURE_BUNDLE_CONFIGS",
    "BC_FEATURE_NORMALIZATION_CHOICES",
    "DEFAULT_BC_LANES",
    "build_behavior_cloned_policy_evidence",
    "train_behavior_cloned_policy",
]
