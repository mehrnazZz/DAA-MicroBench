from __future__ import annotations

import json
import math
from pathlib import Path
import shutil
from typing import Any

import numpy as np

from microbench.learned import observation_to_mlp_features
from microbench.learned.tiny_linear import OBS_BASE_DIM, OBS_NEIGHBOR_DIM
from microbench.rl.bc_training import (
    BC_FIXTURE_BUNDLE_CONFIGS,
    BC_POLICY_NAME,
    BC_TEACHER_NAME,
    BC_TRAINING_SCHEMA_VERSION,
    _fit_random_feature_mlp,
    _model_spec,
    _policy_spec,
)
from microbench.rl.learned_dataset import (
    LEARNED_DATASET_SCHEMA_VERSION,
    LEARNED_DATASET_TEACHER_POLICY,
    export_learned_policy_dataset,
)
from microbench.rl.learned_diagnostics import write_learned_policy_diagnostics
from microbench.rl.learned_leaderboard import write_learned_policy_leaderboard
from microbench.rl.submission_bundle import run_learned_policy_submission_bundle
from microbench.rl.validation_matrix import run_rl_validation_matrix
from microbench.tools.baseline_validation_matrix import ValidationLane, selected_validation_lanes


LEARNED_HARD_LANE_LOOP_SCHEMA_VERSION = "0.1"
LEARNED_DATASET_BC_TRAINING_SOURCE = "DAA Microbench learned-dataset-export shards"
DEFAULT_HARD_LANE_FALLBACK = ("head_on", "crossing", "urban_obstacle")
HARD_DIAGNOSTIC_LABELS = (
    "unsafe",
    "needs_training",
    "fast_but_close",
    "safe_but_slow",
    "balanced_limited_evidence",
)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return path


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON artifact must contain an object: {path}")
    return payload


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def _remove_known_loop_outputs(out: Path) -> None:
    for name in (
        "hard_lane_dataset",
        "training",
        "bc_bundle",
        "tiny_bundle",
        "mlp_bundle",
        "input_diagnostics.json",
        "input_diagnostics.csv",
        "input_diagnostics.md",
        "dataset_manifest_overlay.json",
        "learned_policy_leaderboard.json",
        "learned_policy_leaderboard.csv",
        "learned_policy_diagnostics.json",
        "learned_policy_diagnostics.csv",
        "learned_policy_diagnostics.md",
        "learned_hard_lane_loop.json",
    ):
        path = out / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def _resolve_output_path(raw: str | Path, *, root: Path) -> Path:
    path = Path(raw)
    if path.is_absolute():
        return path
    if path.exists():
        return path
    return root / path


def _diagnostics_payload(diagnostics: str | Path | dict[str, Any] | None) -> tuple[dict[str, Any] | None, Path | None]:
    if diagnostics is None:
        return None, None
    if isinstance(diagnostics, dict):
        return dict(diagnostics), None
    path = Path(diagnostics)
    if path.is_dir():
        path = path / "learned_policy_diagnostics.json"
    return _read_json(path), path


def _lane_match_lookup() -> dict[str, str]:
    lookup: dict[str, str] = {}
    for lane in selected_validation_lanes(None):
        values = {
            lane.lane_id,
            lane.category,
            lane.scenario,
            Path(lane.scenario).stem,
            Path(lane.scenario).name,
        }
        for value in values:
            key = str(value).strip().lower()
            if key:
                lookup[key] = lane.lane_id
    return lookup


def _match_lane_id(value: Any, lookup: dict[str, str]) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    key = text.lower()
    if key in lookup:
        return lookup[key]
    stem = Path(text).stem.lower()
    if stem in lookup:
        return lookup[stem]
    for known, lane_id in lookup.items():
        if known and (known in key or key in known):
            return lane_id
    return None


def _fallback_selection(fallback_lanes: tuple[str, ...] | list[str] | None, max_lanes: int) -> list[str]:
    lanes = [lane.lane_id for lane in selected_validation_lanes(list(fallback_lanes) if fallback_lanes is not None else list(DEFAULT_HARD_LANE_FALLBACK))]
    return lanes[: max(1, int(max_lanes))]


def select_hard_lanes_from_diagnostics(
    diagnostics: str | Path | dict[str, Any] | None = None,
    *,
    fallback_lanes: tuple[str, ...] | list[str] | None = None,
    max_lanes: int = 3,
) -> dict[str, Any]:
    """Select canonical learned-validation lanes from diagnostics rows."""

    payload, diagnostics_path = _diagnostics_payload(diagnostics)
    rows = list(payload.get("rows", [])) if isinstance(payload, dict) else []
    lookup = _lane_match_lookup()
    label_priority = {label: idx for idx, label in enumerate(HARD_DIAGNOSTIC_LABELS)}

    def row_key(row: dict[str, Any]) -> tuple[int, int, float, str]:
        label = str(row.get("diagnostic_label") or "")
        rank = row.get("diagnostic_rank")
        try:
            rank_int = int(rank)
        except (TypeError, ValueError):
            rank_int = 10**9
        try:
            balanced = float(row.get("balanced_score"))
        except (TypeError, ValueError):
            balanced = float("inf")
        return (label_priority.get(label, 10**6), rank_int, balanced, str(row.get("policy") or ""))

    selected: list[str] = []
    reasons: list[dict[str, Any]] = []
    for row in sorted([dict(row) for row in rows if isinstance(row, dict)], key=row_key):
        label = str(row.get("diagnostic_label") or "")
        if label not in label_priority:
            continue
        for field in ("worst_rl_lane", "worst_scenario", "category", "lane_id"):
            lane_id = _match_lane_id(row.get(field), lookup)
            if lane_id is None or lane_id in selected:
                continue
            selected.append(lane_id)
            reasons.append(
                {
                    "lane_id": lane_id,
                    "source_field": field,
                    "source_value": row.get(field),
                    "diagnostic_label": label,
                    "primary_failure": row.get("primary_failure"),
                    "policy": row.get("policy"),
                    "method": row.get("method"),
                }
            )
            break
        if len(selected) >= max(1, int(max_lanes)):
            break

    fallback_used = False
    if not selected:
        fallback_used = True
        selected = _fallback_selection(fallback_lanes, max_lanes)
        reasons = [
            {
                "lane_id": lane_id,
                "source_field": "fallback_lanes",
                "source_value": ",".join(selected),
                "diagnostic_label": None,
                "primary_failure": "no_hard_lane_diagnostics",
            }
            for lane_id in selected
        ]

    return {
        "schema_version": LEARNED_HARD_LANE_LOOP_SCHEMA_VERSION,
        "selected_lanes": selected,
        "fallback_used": bool(fallback_used),
        "fallback_lanes": _fallback_selection(fallback_lanes, max_lanes),
        "max_lanes": int(max_lanes),
        "diagnostics_path": None if diagnostics_path is None else str(diagnostics_path),
        "diagnostic_rows_seen": len(rows),
        "hard_labels": list(HARD_DIAGNOSTIC_LABELS),
        "reasons": reasons,
    }


def _load_dataset_manifest(dataset_manifest: str | Path | dict[str, Any]) -> tuple[dict[str, Any], Path | None, Path]:
    if isinstance(dataset_manifest, dict):
        report = dict(dataset_manifest)
        manifest_path = Path(str(report.get("manifest", ""))) if report.get("manifest") else None
        root = Path(str(report.get("out_dir", ".")))
        return report, manifest_path, root
    path = Path(dataset_manifest)
    if path.is_dir():
        path = path / "learned_dataset_manifest.json"
    report = _read_json(path)
    return report, path, path.parent


def _dataset_training_rows(shard_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    order: list[tuple[str, int]] = []
    grouped: dict[tuple[str, int], dict[str, Any]] = {}
    for row in shard_rows:
        key = (str(row["lane_id"]), int(row["seed"]))
        if key not in grouped:
            order.append(key)
            grouped[key] = {
                "lane_id": str(row["lane_id"]),
                "category": str(row["category"]),
                "scenario": str(row["scenario"]),
                "seed": int(row["seed"]),
                "n_agents": int(row["n_agents"]),
                "comm_profile": str(row["comm_profile"]),
                "steps": 0,
                "samples": 0,
            }
        grouped[key]["steps"] = max(int(grouped[key]["steps"]), int(row["step"]) + 1)
        grouped[key]["samples"] += 1
    return [grouped[key] for key in order]


def _load_shard_features_and_labels(report: dict[str, Any], *, root: Path) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]], list[Path]]:
    feature_rows: list[np.ndarray] = []
    label_rows: list[np.ndarray] = []
    shard_rows: list[dict[str, Any]] = []
    loaded_shards: list[Path] = []

    for raw_shard in report.get("shards", []) or []:
        shard = _resolve_output_path(raw_shard, root=root)
        with np.load(shard, allow_pickle=False) as data:
            observations = np.asarray(data["observations"], dtype=np.float32)
            actions = np.asarray(data["actions"], dtype=np.float32)
            valid_dims = np.asarray(
                data["observation_valid_dim"] if "observation_valid_dim" in data.files else np.full((actions.shape[0],), observations.shape[1]),
                dtype=np.int32,
            )
            lane_ids = np.asarray(data["lane_id"]).astype(str)
            categories = np.asarray(data["category"]).astype(str)
            scenarios = np.asarray(data["scenario"]).astype(str)
            seeds = np.asarray(data["seed"], dtype=np.int32)
            n_agents = np.asarray(data["n_agents"], dtype=np.int32)
            comm_profiles = np.asarray(data["comm_profile"]).astype(str)
            steps = np.asarray(data["step"], dtype=np.int32) if "step" in data.files else np.zeros((actions.shape[0],), dtype=np.int32)

            for idx in range(actions.shape[0]):
                valid_dim = int(valid_dims[idx])
                obs = observations[idx, :valid_dim]
                top_k = max(0, (valid_dim - OBS_BASE_DIM) // OBS_NEIGHBOR_DIM)
                feature_rows.append(observation_to_mlp_features(obs, top_k=top_k))
                label_rows.append(np.clip(actions[idx], -1.0, 1.0).astype(np.float32))
                shard_rows.append(
                    {
                        "lane_id": lane_ids[idx],
                        "category": categories[idx],
                        "scenario": scenarios[idx],
                        "seed": int(seeds[idx]),
                        "n_agents": int(n_agents[idx]),
                        "comm_profile": comm_profiles[idx],
                        "step": int(steps[idx]),
                    }
                )
        loaded_shards.append(shard)

    if not feature_rows:
        raise RuntimeError("learned dataset shards contained no samples")
    return (
        np.vstack(feature_rows).astype(np.float32),
        np.vstack(label_rows).astype(np.float32),
        _dataset_training_rows(shard_rows),
        loaded_shards,
    )


def _selected_lanes_from_training_rows(training_rows: list[dict[str, Any]], report: dict[str, Any]) -> list[ValidationLane]:
    lane_ids: list[str] = []
    for lane in report.get("lanes", []) or []:
        if isinstance(lane, dict) and lane.get("lane_id") and str(lane["lane_id"]) not in lane_ids:
            lane_ids.append(str(lane["lane_id"]))
    for row in training_rows:
        lane_id = str(row.get("lane_id") or "")
        if lane_id and lane_id not in lane_ids:
            lane_ids.append(lane_id)
    return selected_validation_lanes(lane_ids)


def _manifest_overlay_from_dataset_training_report(report: dict[str, Any]) -> dict[str, Any]:
    rows = [dict(row) for row in report.get("training_rows", []) if isinstance(row, dict)]
    seeds = sorted({int(row.get("seed", 0) or 0) for row in rows})
    lane_ids = sorted({str(row.get("lane_id")) for row in rows if row.get("lane_id")})
    scenario_names = sorted({Path(str(row.get("scenario", ""))).stem for row in rows if row.get("scenario")})
    environment_steps = sum(int(row.get("steps", 0) or 0) for row in rows)
    samples = int(report.get("sample_count", 0) or 0)
    source_policy = str(report.get("source_policy") or report.get("teacher_policy") or "dataset_action_labels")
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
            "notes": "Generated by learned-hard-lane-loop from learned-dataset-export shards.",
        },
        "training_disclosure": {
            "training_scenarios": scenario_names,
            "training_suites": [f"learned_hard_lane_loop:{','.join(lane_ids)}"] if lane_ids else ["learned_hard_lane_loop"],
            "environment_steps": int(environment_steps),
            "random_seeds": seeds,
            "observation_normalization": "none; public RL observation features are consumed directly",
            "action_post_processing": (
                "artifact-declared goal-direction forward floor and unit-norm clamp, "
                "then normalized velocity action clipped by the DAA Microbench action contract"
            ),
            "reward_configuration": "not used; supervised behavior cloning from dataset action labels",
            "external_data": "none; input labels came from DAA Microbench learned-dataset-export shards",
            "pretrained_models": "none",
            "hardware": "local CPU",
            "teacher_policy": report.get("teacher_policy"),
            "source_policy": source_policy,
            "action_source": report.get("action_source"),
            "source_dataset_manifest": report.get("dataset_manifest"),
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
            "intended_category": "hard_lane_behavior_cloned_public_observation_baseline",
        },
    }


def train_behavior_cloned_policy_from_dataset(
    *,
    out_dir: str | Path,
    dataset_manifest: str | Path | dict[str, Any],
    hidden_dim: int = 32,
    ridge: float = 1e-4,
    eval_lanes: tuple[str, ...] | list[str] | None = None,
    eval_max_steps: int | None = 12,
    policy_name: str = BC_POLICY_NAME,
    seed: int = 29,
    overwrite: bool = False,
    run_validation: bool = True,
) -> dict[str, Any]:
    """Train the portable BC MLP from exported learned dataset shards."""

    out = Path(out_dir)
    model_path = out / "bc_mlp_policy.json"
    spec_path = out / "policy_spec.json"
    report_path = out / "bc_dataset_training_report.json"
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
            raise RuntimeError(f"dataset BC output already exists: {', '.join(str(path) for path in existing)}")
    elif validation_dir.exists():
        shutil.rmtree(validation_dir)

    report, manifest_path, dataset_root = _load_dataset_manifest(dataset_manifest)
    features, labels, training_rows, loaded_shards = _load_shard_features_and_labels(report, root=dataset_root)
    selected_lanes = _selected_lanes_from_training_rows(training_rows, report)
    seed_override = sorted({int(row.get("seed", 0) or 0) for row in training_rows})

    w1, b1, w2, b2, fit_rmse = _fit_random_feature_mlp(
        features,
        labels,
        seed=int(seed),
        hidden_dim=int(hidden_dim),
        ridge=float(ridge),
    )
    max_steps = report.get("max_steps")
    if max_steps is None:
        max_steps = max((int(row.get("steps", 0) or 0) for row in training_rows), default=0)
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
        lanes=selected_lanes,
        seeds=seed_override,
        max_steps=int(max_steps),
        rollout_noise_std=0.0,
        sample_count=int(features.shape[0]),
    )
    training_block = model_payload["training"]
    training_block.update(
        {
            "recipe": "python -m microbench.cli learned-hard-lane-loop",
            "source": LEARNED_DATASET_BC_TRAINING_SOURCE,
            "teacher_policy": report.get("teacher_policy") or (BC_TEACHER_NAME if report.get("action_source") == "bc_teacher" else None),
            "source_policy": report.get("policy"),
            "action_source": report.get("action_source"),
            "source_dataset_schema_version": report.get("schema_version"),
            "source_dataset_manifest": None if manifest_path is None else str(manifest_path),
            "source_dataset_shards": [str(path) for path in loaded_shards],
            "rollout_policy": "dataset_action_labels",
            "rollout_noise_std": 0.0,
        }
    )
    spec_payload = _policy_spec(artifact_name=model_path.name, policy_name=str(policy_name))
    _write_json(model_path, model_payload)
    _write_json(spec_path, spec_payload)

    validation_report = None
    if run_validation:
        validation_report = run_rl_validation_matrix(
            out_dir=validation_dir,
            policy_spec=spec_path,
            lanes=list(eval_lanes) if eval_lanes is not None else [lane.lane_id for lane in selected_lanes],
            max_steps=eval_max_steps,
        )

    finite_dataset = bool(np.all(np.isfinite(features)) and np.all(np.isfinite(labels)))
    manifest_sample_count = int(report.get("sample_count", 0) or 0)
    checks = [
        _check("dataset_schema_supported", report.get("schema_version") == LEARNED_DATASET_SCHEMA_VERSION, {"schema_version": report.get("schema_version")}),
        _check("dataset_export_ok", bool(report.get("ok")), {"manifest": None if manifest_path is None else str(manifest_path)}),
        _check("dataset_shards_present", bool(loaded_shards), {"shard_count": len(loaded_shards)}),
        _check("samples_loaded", int(features.shape[0]) > 0, {"samples": int(features.shape[0]), "manifest_samples": manifest_sample_count}),
        _check("sample_count_matches_manifest", manifest_sample_count in {0, int(features.shape[0])}, {"samples": int(features.shape[0]), "manifest_samples": manifest_sample_count}),
        _check("finite_dataset", finite_dataset, {"feature_shape": list(features.shape), "label_shape": list(labels.shape)}),
        _check("fit_rmse_finite", math.isfinite(float(fit_rmse)), {"fit_rmse": round(float(fit_rmse), 8)}),
        _check("policy_spec_written", bool(spec_path.exists() and model_path.exists()), {"policy_spec": str(spec_path), "model_artifact": str(model_path)}),
    ]
    if validation_report is not None:
        checks.append(
            _check(
                "validation_matrix_gate_pass",
                bool(validation_report.get("ok")),
                {
                    "policy": validation_report.get("policy"),
                    "run_count": validation_report.get("run_count"),
                    "behavior_pass": validation_report.get("behavior_pass"),
                },
            )
        )

    training_report = {
        "schema_version": BC_TRAINING_SCHEMA_VERSION,
        "ok": all(check["ok"] for check in checks),
        "training_source": LEARNED_DATASET_BC_TRAINING_SOURCE,
        "policy_name": str(policy_name),
        "teacher_policy": report.get("teacher_policy") or (BC_TEACHER_NAME if report.get("action_source") == "bc_teacher" else None),
        "source_policy": report.get("policy"),
        "action_source": report.get("action_source"),
        "public_observations_only": bool(report.get("public_observations_only", True)),
        "privileged_global_state": bool(report.get("privileged_global_state", False)),
        "out_dir": str(out),
        "dataset_manifest": None if manifest_path is None else str(manifest_path),
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
    _write_json(report_path, training_report)
    return training_report


def run_learned_hard_lane_loop(
    *,
    out_dir: str | Path,
    diagnostics: str | Path | dict[str, Any] | None = None,
    bundles: tuple[str | Path, ...] | list[str | Path] | None = None,
    fallback_lanes: tuple[str, ...] | list[str] | None = None,
    max_lanes: int = 3,
    dataset_policy: str = LEARNED_DATASET_TEACHER_POLICY,
    dataset_policy_spec: str | Path | None = None,
    dataset_max_steps: int | None = 64,
    dataset_shard_size: int = 50000,
    save_replay: bool = False,
    hidden_dim: int = 32,
    ridge: float = 1e-4,
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
    """Diagnose hard lanes, export shards, train from them, and rerun evidence."""

    out = Path(out_dir)
    report_path = out / "learned_hard_lane_loop.json"
    if report_path.exists() and not bool(overwrite):
        raise RuntimeError(f"learned hard-lane loop output already exists: {report_path}")
    if bool(overwrite) and out.exists():
        _remove_known_loop_outputs(out)
    out.mkdir(parents=True, exist_ok=True)

    input_diagnostics: dict[str, Any] | None = None
    input_diagnostics_path: Path | None = None
    if diagnostics is not None:
        input_diagnostics, input_diagnostics_path = _diagnostics_payload(diagnostics)
        written_diagnostics = _write_json(out / "input_diagnostics.json", input_diagnostics or {})
        input_diagnostics_path = input_diagnostics_path or written_diagnostics
    elif bundles:
        input_diagnostics = write_learned_policy_diagnostics(
            bundles=[str(bundle) for bundle in bundles],
            out=out / "input_diagnostics.json",
        )
        input_diagnostics_path = out / "input_diagnostics.json"

    selection = select_hard_lanes_from_diagnostics(
        input_diagnostics,
        fallback_lanes=fallback_lanes,
        max_lanes=int(max_lanes),
    )
    selected_lanes = list(selection["selected_lanes"])

    dataset = export_learned_policy_dataset(
        out_dir=out / "hard_lane_dataset",
        policy=str(dataset_policy),
        policy_spec=dataset_policy_spec,
        lanes=selected_lanes,
        max_steps=dataset_max_steps,
        shard_size=int(dataset_shard_size),
        save_replay=bool(save_replay),
        overwrite=bool(overwrite),
    )
    training = train_behavior_cloned_policy_from_dataset(
        out_dir=out / "training",
        dataset_manifest=str(dataset["manifest"]),
        hidden_dim=int(hidden_dim),
        ridge=float(ridge),
        eval_lanes=eval_lanes if eval_lanes is not None else selected_lanes,
        eval_max_steps=eval_max_steps,
        policy_name=str(policy_name),
        seed=int(seed),
        overwrite=bool(overwrite),
        run_validation=True,
    )

    manifest_overlay = _manifest_overlay_from_dataset_training_report(training)
    manifest_overlay_path = _write_json(out / "dataset_manifest_overlay.json", manifest_overlay)

    seed_list = [int(value) for value in (bundle_seeds if bundle_seeds is not None else (0,))]
    built_bundles: dict[str, dict[str, Any]] = {}
    bc_bundle_dir = out / "bc_bundle"
    built_bundles["bc"] = run_learned_policy_submission_bundle(
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
            built_bundles[str(config["label"])] = run_learned_policy_submission_bundle(
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
    final_diagnostics = write_learned_policy_diagnostics(
        bundles=bundle_paths,
        out=out / "learned_policy_diagnostics.json",
    )

    checks = [
        _check("hard_lanes_selected", bool(selected_lanes), {"selected_lanes": selected_lanes}),
        _check("dataset_ok", bool(dataset.get("ok")), {"sample_count": dataset.get("sample_count"), "shards": len(dataset.get("shards", []))}),
        _check("training_ok", bool(training.get("ok")), {"sample_count": training.get("sample_count"), "fit_rmse": training.get("fit_rmse")}),
        _check("bc_bundle_ok", bool(built_bundles["bc"].get("ok")), {"path": str(bc_bundle_dir), "policy": built_bundles["bc"].get("policy")}),
        _check("leaderboard_ok", bool(leaderboard.get("ok")), {"bundle_count": leaderboard.get("bundle_count")}),
        _check("diagnostics_ok", bool(final_diagnostics.get("ok")), {"bundle_count": final_diagnostics.get("bundle_count")}),
    ]
    for label, bundle in built_bundles.items():
        if label == "bc":
            continue
        checks.append(_check(f"{label}_fixture_bundle_ok", bool(bundle.get("ok")), {"policy": bundle.get("policy"), "method": bundle.get("method")}))

    report = {
        "schema_version": LEARNED_HARD_LANE_LOOP_SCHEMA_VERSION,
        "ok": all(check["ok"] for check in checks),
        "out_dir": str(out),
        "input_diagnostics": None if input_diagnostics_path is None else str(input_diagnostics_path),
        "selection": selection,
        "dataset": {
            "manifest": dataset.get("manifest"),
            "policy": dataset.get("policy"),
            "action_source": dataset.get("action_source"),
            "sample_count": dataset.get("sample_count"),
            "episode_count": dataset.get("episode_count"),
            "shards": dataset.get("shards"),
        },
        "training": {
            "report": training.get("training_report"),
            "policy_spec": training.get("policy_spec"),
            "model_artifact": training.get("model_artifact"),
            "sample_count": training.get("sample_count"),
            "fit_rmse": training.get("fit_rmse"),
            "training_source": training.get("training_source"),
        },
        "manifest_overlay": str(manifest_overlay_path),
        "bundle_paths": {
            label: str(out / f"{label}_bundle")
            for label in built_bundles
        },
        "bundles": {
            label: {
                "ok": bool(bundle.get("ok")),
                "method": bundle.get("method"),
                "policy": bundle.get("policy"),
                "suite": bundle.get("suite"),
                "run_count": bundle.get("planner_sweep", {}).get("run_count"),
                "rl_validation_matrix": bundle.get("rl_validation_matrix"),
            }
            for label, bundle in built_bundles.items()
        },
        "leaderboard": leaderboard,
        "diagnostics": final_diagnostics,
        "checks": checks,
    }
    _write_json(report_path, report)
    return report


__all__ = [
    "DEFAULT_HARD_LANE_FALLBACK",
    "HARD_DIAGNOSTIC_LABELS",
    "LEARNED_DATASET_BC_TRAINING_SOURCE",
    "LEARNED_HARD_LANE_LOOP_SCHEMA_VERSION",
    "run_learned_hard_lane_loop",
    "select_hard_lanes_from_diagnostics",
    "train_behavior_cloned_policy_from_dataset",
]
