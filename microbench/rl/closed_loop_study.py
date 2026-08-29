from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from microbench.rl.closed_loop_training import (
    CLOSED_LOOP_BROAD_3D_HOLDOUT_SCENARIOS,
    CLOSED_LOOP_HOLDOUT_SCORE_TOLERANCE,
    CLOSED_LOOP_OBJECTIVE_DEFAULTS,
    CLOSED_LOOP_PER_LANE_CLEARANCE_TOLERANCE_M,
    CLOSED_LOOP_POLICY_NAME,
    CLOSED_LOOP_TRAINING_LANE_PROFILE_CHOICES,
    fine_tune_closed_loop_policy,
)
from microbench.rl.learned_diagnostics import write_learned_policy_diagnostics
from microbench.rl.learned_leaderboard import write_learned_policy_leaderboard
from microbench.rl.submission_bundle import run_learned_policy_submission_bundle


LEARNED_CLOSED_LOOP_STUDY_SCHEMA_VERSION = "0.1"
LEARNED_CLOSED_LOOP_STUDY_REPORT_FILENAME = "learned_closed_loop_study_report.json"
LEARNED_CLOSED_LOOP_STUDY_MANIFEST_FILENAME = "study_manifest.json"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": str(name), "ok": bool(ok), "details": details or {}}


def _cleanup_study_outputs(out: Path) -> None:
    for path in (
        out / "training",
        out / "bundle",
    ):
        if path.exists():
            shutil.rmtree(path)
    for path in (
        out / LEARNED_CLOSED_LOOP_STUDY_MANIFEST_FILENAME,
        out / LEARNED_CLOSED_LOOP_STUDY_REPORT_FILENAME,
        out / "learned_leaderboard.json",
        out / "learned_leaderboard.csv",
        out / "learned_diagnostics.json",
        out / "learned_diagnostics.csv",
        out / "learned_diagnostics.md",
    ):
        if path.exists():
            path.unlink()


def _existing_study_outputs(out: Path) -> list[Path]:
    candidates = [
        out / "training",
        out / "bundle",
        out / LEARNED_CLOSED_LOOP_STUDY_MANIFEST_FILENAME,
        out / LEARNED_CLOSED_LOOP_STUDY_REPORT_FILENAME,
        out / "learned_leaderboard.json",
        out / "learned_leaderboard.csv",
        out / "learned_diagnostics.json",
        out / "learned_diagnostics.csv",
        out / "learned_diagnostics.md",
    ]
    return [path for path in candidates if path.exists()]


def _summary_metrics(metrics: dict[str, Any] | None) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "score": metrics.get("score"),
        "collision_ticks": metrics.get("collision_ticks"),
        "near_miss_ticks": metrics.get("near_miss_ticks"),
        "min_clearance_m": metrics.get("min_clearance_m"),
        "completion_rate_mean": metrics.get("completion_rate_mean"),
        "total_reward_mean": metrics.get("total_reward_mean"),
    }


def _holdout_summary(holdout: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(holdout, dict):
        return None
    return {
        "profile": holdout.get("profile"),
        "promotion_candidate": bool(holdout.get("promotion_candidate")),
        "expected_runs_per_policy": holdout.get("expected_runs_per_policy"),
        "comparison_csv": holdout.get("comparison_csv"),
        "base": {
            key: (holdout.get("base") or {}).get(key)
            for key in (
                "run_count",
                "collision_episodes",
                "near_miss_episodes",
                "completion_rate_mean",
                "min_sep_min_row_m",
                "min_sep_p05_row_min_m",
                "score_v0_mean",
                "score_v0_worst",
            )
        },
        "tuned": {
            key: (holdout.get("tuned") or {}).get(key)
            for key in (
                "run_count",
                "collision_episodes",
                "near_miss_episodes",
                "completion_rate_mean",
                "min_sep_min_row_m",
                "min_sep_p05_row_min_m",
                "score_v0_mean",
                "score_v0_worst",
            )
        },
    }


def _training_summary(training: dict[str, Any]) -> dict[str, Any]:
    return {
        "ok": bool(training.get("ok")),
        "behavior_pass": bool(training.get("behavior_pass")),
        "promotion_candidate": bool(training.get("promotion_candidate")),
        "promotion_status": training.get("promotion_status"),
        "policy_name": training.get("policy_name"),
        "policy_spec": training.get("policy_spec"),
        "model_artifact": training.get("model_artifact"),
        "training_report": training.get("training_report"),
        "candidate_summary_csv": training.get("candidate_summary_csv"),
        "candidate_episodes_csv": training.get("candidate_episodes_csv"),
        "candidate_lane_summary_csv": training.get("candidate_lane_summary_csv"),
        "candidate_count": training.get("candidate_count"),
        "accepted_generation_count": training.get("accepted_generation_count"),
        "best_candidate_id": training.get("best_candidate_id"),
        "search_strategy": training.get("search_strategy"),
        "search_plan": training.get("search_plan"),
        "antithetic_sampling": training.get("antithetic_sampling"),
        "require_per_lane_safety": training.get("require_per_lane_safety"),
        "per_lane_clearance_tolerance_m": training.get("per_lane_clearance_tolerance_m"),
        "base_metrics": _summary_metrics(training.get("base_metrics")),
        "best_metrics": _summary_metrics(training.get("best_metrics")),
        "holdout": _holdout_summary(training.get("holdout")),
        "failed_gate_checks": [
            check.get("name")
            for check in training.get("checks", [])
            if check.get("severity") == "gate" and not check.get("ok")
        ],
        "failed_behavior_checks": [
            check.get("name")
            for check in training.get("checks", [])
            if check.get("severity") == "behavior" and not check.get("ok")
        ],
    }


def _bundle_summary(bundle: dict[str, Any], bundle_dir: Path) -> dict[str, Any]:
    return {
        "ok": bool(bundle.get("ok")),
        "method": bundle.get("method"),
        "policy": bundle.get("policy"),
        "suite": bundle.get("suite"),
        "bundle_json": str(bundle_dir / "learned_submission_bundle.json"),
        "planner_sweep": {
            "run_count": (bundle.get("planner_sweep") or {}).get("run_count"),
            "planned_run_count": (bundle.get("planner_sweep") or {}).get("planned_run_count"),
            "results_csv": (bundle.get("planner_sweep") or {}).get("results_csv"),
            "summary_csv": (bundle.get("planner_sweep") or {}).get("summary_csv"),
        },
        "rl_validation_matrix": bundle.get("rl_validation_matrix"),
        "failed_checks": [check.get("name") for check in bundle.get("checks", []) if not check.get("ok")],
    }


def _failed_behavior_checks(training: dict[str, Any]) -> list[str]:
    return [
        str(check.get("name"))
        for check in training.get("checks", [])
        if check.get("severity") == "behavior" and not check.get("ok")
    ]


def _recommendation(*, training: dict[str, Any], ok: bool) -> str:
    if not ok:
        return "fix_artifacts"
    if bool(training.get("promotion_candidate")):
        if int(training.get("accepted_generation_count", 0) or 0) <= 0 or str(training.get("best_candidate_id") or "") == "base":
            return "holdout_passed_no_update"
        return "promote"
    failed = _failed_behavior_checks(training)
    if any(name.startswith("holdout_") or name.startswith("final_") for name in failed):
        return "reject_regression"
    return "holdout_review_required"


def run_learned_closed_loop_study(
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
    search_strategy: str = "single_stage",
    stage1_generations: int | None = None,
    stage2_generations: int | None = None,
    antithetic_sampling: bool = False,
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
    require_per_lane_safety: bool = False,
    per_lane_clearance_tolerance_m: float = CLOSED_LOOP_PER_LANE_CLEARANCE_TOLERANCE_M,
    eval_lanes: tuple[str, ...] | list[str] | None = None,
    eval_max_steps: int | None = 12,
    holdout_profile: str = "broad_3d_stress",
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
    run_validation: bool = True,
    bundle_method: str = "learned_policy_spec",
    bundle_suite: str = "official_smoke_generated",
    bundle_root: str | Path = ".",
    bundle_n_agents: int = 4,
    bundle_seeds: tuple[int, ...] | list[int] | None = None,
    bundle_max_steps: int | None = 12,
    bundle_max_runs: int | None = 1,
    bundle_save_trace: bool = False,
    submission_manifest: str | Path | None = None,
    comparison_bundles: tuple[str | Path, ...] | list[str | Path] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run training, holdout, bundle review, leaderboard, and diagnostics as one study."""

    out = Path(out_dir)
    if overwrite:
        _cleanup_study_outputs(out)
    else:
        existing = _existing_study_outputs(out)
        if existing:
            raise RuntimeError(f"learned closed-loop study output already exists: {', '.join(str(path) for path in existing)}")
    out.mkdir(parents=True, exist_ok=True)

    training_dir = out / "training"
    bundle_dir = out / "bundle"
    leaderboard_json = out / "learned_leaderboard.json"
    leaderboard_csv = out / "learned_leaderboard.csv"
    diagnostics_json = out / "learned_diagnostics.json"
    diagnostics_csv = out / "learned_diagnostics.csv"
    diagnostics_md = out / "learned_diagnostics.md"
    manifest_path = out / LEARNED_CLOSED_LOOP_STUDY_MANIFEST_FILENAME
    report_path = out / LEARNED_CLOSED_LOOP_STUDY_REPORT_FILENAME

    training_report = fine_tune_closed_loop_policy(
        out_dir=training_dir,
        base_policy_spec=base_policy_spec,
        lanes=lanes,
        lane_profile=str(lane_profile),
        seeds=seeds,
        train_max_steps=train_max_steps,
        generations=int(generations),
        population_size=int(population_size),
        trainable_parameters=str(trainable_parameters),
        search_strategy=str(search_strategy),
        stage1_generations=stage1_generations,
        stage2_generations=stage2_generations,
        antithetic_sampling=bool(antithetic_sampling),
        sigma=float(sigma),
        sigma_decay=float(sigma_decay),
        min_delta=float(min_delta),
        collision_tick_penalty=float(collision_tick_penalty),
        near_miss_tick_penalty=float(near_miss_tick_penalty),
        clearance_penalty=float(clearance_penalty),
        mission_penalty=float(mission_penalty),
        reward_weight=float(reward_weight),
        min_clearance_m=float(min_clearance_m),
        max_collision_ticks=int(max_collision_ticks),
        max_near_miss_ticks=max_near_miss_ticks,
        allow_near_miss_regression=bool(allow_near_miss_regression),
        require_per_lane_safety=bool(require_per_lane_safety),
        per_lane_clearance_tolerance_m=float(per_lane_clearance_tolerance_m),
        eval_lanes=eval_lanes,
        eval_max_steps=eval_max_steps,
        holdout_profile=str(holdout_profile),
        holdout_scenarios=holdout_scenarios or CLOSED_LOOP_BROAD_3D_HOLDOUT_SCENARIOS,
        holdout_seeds=holdout_seeds,
        holdout_comm_profiles=holdout_comm_profiles,
        holdout_n_agents=int(holdout_n_agents),
        holdout_max_runs=holdout_max_runs,
        holdout_score_tolerance=float(holdout_score_tolerance),
        allow_holdout_safety_regression=bool(allow_holdout_safety_regression),
        allow_holdout_score_regression=bool(allow_holdout_score_regression),
        policy_name=str(policy_name),
        seed=int(seed),
        overwrite=False,
        run_validation=bool(run_validation),
    )

    bundle_report = run_learned_policy_submission_bundle(
        out_dir=bundle_dir,
        method=str(bundle_method),
        policy="tiny_learned",
        policy_spec=training_report["policy_spec"],
        suite=str(bundle_suite),
        root=bundle_root,
        n_agents=int(bundle_n_agents),
        seeds=[int(seed_value) for seed_value in (bundle_seeds if bundle_seeds is not None else (0,))],
        max_steps=bundle_max_steps,
        max_runs=bundle_max_runs,
        save_trace=bool(bundle_save_trace),
        submission_manifest=submission_manifest,
    )

    bundle_paths: list[str | Path] = list(comparison_bundles or [])
    bundle_paths.append(bundle_dir)
    leaderboard_report = write_learned_policy_leaderboard(
        bundles=bundle_paths,
        out=leaderboard_json,
        csv_out=leaderboard_csv,
    )
    diagnostics_report = write_learned_policy_diagnostics(
        bundles=bundle_paths,
        out=diagnostics_json,
        csv_out=diagnostics_csv,
        markdown_out=diagnostics_md,
    )

    checks = [
        _check(
            "closed_loop_training_ran",
            bool(training_report.get("ok")),
            {
                "training_report": training_report.get("training_report"),
                "failed_gate_checks": [
                    check.get("name")
                    for check in training_report.get("checks", [])
                    if check.get("severity") == "gate" and not check.get("ok")
                ],
            },
        ),
        _check(
            "learned_submission_bundle_ran",
            bool(bundle_report.get("ok")),
            {
                "bundle": str(bundle_dir),
                "failed_checks": [check.get("name") for check in bundle_report.get("checks", []) if not check.get("ok")],
            },
        ),
        _check(
            "learned_leaderboard_written",
            bool(leaderboard_report.get("ok")) and leaderboard_json.exists() and leaderboard_csv.exists(),
            {"leaderboard": str(leaderboard_json), "csv": str(leaderboard_csv)},
        ),
        _check(
            "learned_diagnostics_written",
            bool(diagnostics_report.get("ok")) and diagnostics_json.exists() and diagnostics_csv.exists() and diagnostics_md.exists(),
            {"diagnostics": str(diagnostics_json), "csv": str(diagnostics_csv), "markdown": str(diagnostics_md)},
        ),
    ]
    ok = all(check["ok"] for check in checks)
    recommendation = _recommendation(training=training_report, ok=ok)
    promotion_candidate = bool(ok and training_report.get("promotion_candidate"))
    artifacts = {
        "study_manifest": str(manifest_path),
        "study_report": str(report_path),
        "training_dir": str(training_dir),
        "training_report": str(training_report.get("training_report")),
        "policy_spec": str(training_report.get("policy_spec")),
        "model_artifact": str(training_report.get("model_artifact")),
        "candidate_lane_summary": str(training_report.get("candidate_lane_summary_csv")),
        "bundle_dir": str(bundle_dir),
        "bundle_report": str(bundle_dir / "learned_submission_bundle.json"),
        "leaderboard_json": str(leaderboard_json),
        "leaderboard_csv": str(leaderboard_csv),
        "diagnostics_json": str(diagnostics_json),
        "diagnostics_csv": str(diagnostics_csv),
        "diagnostics_markdown": str(diagnostics_md),
    }
    configuration = {
        "base_policy_spec": str(base_policy_spec),
        "lane_profile": str(lane_profile),
        "lanes": None if lanes is None else [str(lane) for lane in lanes],
        "seeds": None if seeds is None else [int(value) for value in seeds],
        "train_max_steps": None if train_max_steps is None else int(train_max_steps),
        "generations": int(generations),
        "population_size": int(population_size),
        "trainable_parameters": str(trainable_parameters),
        "search_strategy": str(search_strategy),
        "stage1_generations": None if stage1_generations is None else int(stage1_generations),
        "stage2_generations": None if stage2_generations is None else int(stage2_generations),
        "antithetic_sampling": bool(antithetic_sampling),
        "require_per_lane_safety": bool(require_per_lane_safety),
        "per_lane_clearance_tolerance_m": float(per_lane_clearance_tolerance_m),
        "holdout_profile": str(holdout_profile),
        "holdout_scenarios": [str(value) for value in (holdout_scenarios or CLOSED_LOOP_BROAD_3D_HOLDOUT_SCENARIOS)],
        "holdout_seeds": None if holdout_seeds is None else [int(value) for value in holdout_seeds],
        "holdout_comm_profiles": None if holdout_comm_profiles is None else [str(value) for value in holdout_comm_profiles],
        "holdout_n_agents": int(holdout_n_agents),
        "holdout_max_runs": None if holdout_max_runs is None else int(holdout_max_runs),
        "bundle_method": str(bundle_method),
        "bundle_suite": str(bundle_suite),
        "bundle_n_agents": int(bundle_n_agents),
        "bundle_seeds": [int(seed_value) for seed_value in (bundle_seeds if bundle_seeds is not None else (0,))],
        "bundle_max_steps": None if bundle_max_steps is None else int(bundle_max_steps),
        "bundle_max_runs": None if bundle_max_runs is None else int(bundle_max_runs),
        "comparison_bundles": [str(path) for path in (comparison_bundles or [])],
    }
    manifest = {
        "schema_version": LEARNED_CLOSED_LOOP_STUDY_SCHEMA_VERSION,
        "workflow": "learned-closed-loop-study",
        "recommendation": recommendation,
        "promotion_candidate": promotion_candidate,
        "ok": ok,
        "configuration": configuration,
        "artifacts": artifacts,
        "checks": checks,
    }
    checks.append(
        _check(
            "study_manifest_written",
            True,
            {"study_manifest": str(manifest_path), "schema_version": LEARNED_CLOSED_LOOP_STUDY_SCHEMA_VERSION},
        )
    )
    ok = all(check["ok"] for check in checks)
    recommendation = _recommendation(training=training_report, ok=ok)
    promotion_candidate = bool(ok and training_report.get("promotion_candidate"))
    manifest.update(
        {
            "recommendation": recommendation,
            "promotion_candidate": promotion_candidate,
            "ok": ok,
            "checks": checks,
        }
    )
    _write_json(manifest_path, manifest)

    report = {
        "schema_version": LEARNED_CLOSED_LOOP_STUDY_SCHEMA_VERSION,
        "ok": ok,
        "promotion_candidate": promotion_candidate,
        "recommendation": recommendation,
        "out_dir": str(out),
        "configuration": configuration,
        "artifacts": artifacts,
        "training": _training_summary(training_report),
        "bundle": _bundle_summary(bundle_report, bundle_dir),
        "leaderboard": {
            "ok": bool(leaderboard_report.get("ok")),
            "bundle_count": leaderboard_report.get("bundle_count"),
            "reviewable_count": leaderboard_report.get("reviewable_count"),
            "leaderboard_candidate_count": leaderboard_report.get("leaderboard_candidate_count"),
            "leaderboard_path": leaderboard_report.get("leaderboard_path"),
            "leaderboard_csv": leaderboard_report.get("leaderboard_csv"),
            "rows": [
                {
                    "policy": row.get("policy"),
                    "lineage_label": row.get("lineage_label"),
                    "promotion_stage": row.get("promotion_stage"),
                    "score_v0_mean": row.get("score_v0_mean"),
                    "recommendation": row.get("recommendation"),
                }
                for row in leaderboard_report.get("rows", [])
            ],
        },
        "diagnostics": {
            "ok": bool(diagnostics_report.get("ok")),
            "diagnostics_path": diagnostics_report.get("diagnostics_path"),
            "diagnostics_csv": diagnostics_report.get("diagnostics_csv"),
            "diagnostics_markdown": diagnostics_report.get("diagnostics_markdown"),
            "summary": diagnostics_report.get("summary"),
            "findings": diagnostics_report.get("findings"),
        },
        "checks": checks,
    }
    _write_json(report_path, report)
    return report


__all__ = [
    "LEARNED_CLOSED_LOOP_STUDY_MANIFEST_FILENAME",
    "LEARNED_CLOSED_LOOP_STUDY_REPORT_FILENAME",
    "LEARNED_CLOSED_LOOP_STUDY_SCHEMA_VERSION",
    "CLOSED_LOOP_TRAINING_LANE_PROFILE_CHOICES",
    "run_learned_closed_loop_study",
]
