from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from microbench.acceptance import check_acceptance
from microbench.metrics import append_result, write_summary
from microbench.planners import planner_metadata
from microbench.rl.calibration import run_rl_policy_calibration
from microbench.rl.evaluate import run_rl_policy_smoke
from microbench.rl.freeze import run_rl_freeze_check
from microbench.rl.schema import interface_contract
from microbench.runner import run_episode
from microbench.scenarios import materialize_official_suite, suite_defaults
from microbench.types import RunSpec


LEARNED_SUBMISSION_BUNDLE_SCHEMA_VERSION = "0.1"


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _method_metadata(method: str) -> dict[str, Any] | None:
    by_method = {entry["method"]: entry for entry in planner_metadata(include_aliases=False)}
    return by_method.get(str(method))


def _run_planner_sweep(
    *,
    out_dir: Path,
    suite: str,
    method: str,
    max_runs: int | None,
    save_trace: bool,
) -> dict[str, Any]:
    if (out_dir / "results.csv").exists():
        raise RuntimeError(f"planner sweep output already exists: {out_dir / 'results.csv'}")

    generated = materialize_official_suite(
        suite,
        out_dir / "_generated_scenarios" / suite,
        overwrite=True,
    )
    defaults = suite_defaults(suite)
    scenarios = [Path(path) for path in generated["scenario_paths"]]
    n_agents = [int(value) for value in defaults["n_agents"]]
    seeds = [int(value) for value in defaults["seeds"]]
    comm_profiles = [str(value) for value in defaults["comm_profiles"]]

    specs: list[RunSpec] = []
    for scenario in scenarios:
        for comm_profile in comm_profiles:
            for n in n_agents:
                for seed in seeds:
                    specs.append(
                        RunSpec(
                            scenario_path=str(scenario),
                            method=str(method),
                            n_agents=int(n),
                            seed=int(seed),
                            comm_profile=str(comm_profile),
                            out_dir=str(out_dir),
                            save_trace=bool(save_trace),
                        )
                    )

    planned_run_count = len(specs)
    if max_runs is not None:
        specs = specs[: max(0, int(max_runs))]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = run_episode(spec)
        append_result(out_dir, row)
        rows.append(row)
    summary_csv = write_summary(out_dir)
    results_csv = out_dir / "results.csv"

    acceptance = check_acceptance(
        summary_csv=summary_csv,
        results_csv=results_csv,
        suite_manifest=generated["manifest_path"],
        methods=[str(method)],
    )

    guardrail_total = 0
    finite_metric_violations: list[dict[str, Any]] = []
    for row in rows:
        for field in ("planner_timeout_count", "planner_error_count", "planner_fallback_count"):
            guardrail_total += int(float(row.get(field, 0) or 0))
        for field in ("collision_episode", "completion_rate", "min_sep_min_m", "planner_ms_per_tick_per_agent_p95"):
            if not _finite(row.get(field)):
                finite_metric_violations.append(
                    {
                        "scenario": row.get("scenario"),
                        "comm_profile": row.get("comm_profile"),
                        "N": row.get("N"),
                        "seed": row.get("seed"),
                        "field": field,
                        "value": row.get(field),
                    }
                )

    return {
        "suite": str(suite),
        "method": str(method),
        "planned_run_count": int(planned_run_count),
        "run_count": len(rows),
        "max_runs": None if max_runs is None else int(max_runs),
        "results_csv": str(results_csv),
        "summary_csv": str(summary_csv),
        "result_schema_json": str(out_dir / "result_schema.json"),
        "suite_manifest": str(generated["manifest_path"]),
        "acceptance_json": str(out_dir / "acceptance.json"),
        "acceptance": acceptance,
        "guardrail_total": int(guardrail_total),
        "finite_metric_violations": finite_metric_violations[:20],
        "rows": [
            {
                "method": row.get("method"),
                "scenario": row.get("scenario"),
                "comm_profile": row.get("comm_profile"),
                "N": row.get("N"),
                "seed": row.get("seed"),
                "collision_episode": row.get("collision_episode"),
                "completion_rate": row.get("completion_rate"),
                "min_sep_min_m": row.get("min_sep_min_m"),
                "planner_ms_per_tick_per_agent_p95": row.get("planner_ms_per_tick_per_agent_p95"),
                "planner_timeout_count": row.get("planner_timeout_count"),
                "planner_error_count": row.get("planner_error_count"),
                "planner_fallback_count": row.get("planner_fallback_count"),
            }
            for row in rows
        ],
    }


def run_learned_policy_submission_bundle(
    *,
    out_dir: str | Path,
    method: str = "learned_tiny",
    policy: str = "tiny_learned",
    suite: str = "official_smoke_generated",
    root: str | Path = ".",
    n_agents: int = 4,
    seeds: tuple[int, ...] | list[int] | None = None,
    max_steps: int | None = None,
    max_runs: int | None = None,
    save_trace: bool = False,
) -> dict[str, Any]:
    """Create a reproducible learned-policy submission artifact bundle."""

    out = Path(out_dir)
    bundle_path = out / "learned_submission_bundle.json"
    if bundle_path.exists():
        raise RuntimeError(f"learned submission bundle already exists: {bundle_path}")
    out.mkdir(parents=True, exist_ok=True)

    seed_list = [int(seed) for seed in (seeds if seeds is not None else (0,))]
    contract = interface_contract(top_k=8)
    freeze = run_rl_freeze_check(root=root)
    smoke = run_rl_policy_smoke(
        out_dir=out / "rl_smoke",
        policy=str(policy),
        n_agents=int(n_agents),
        seeds=seed_list,
        max_steps=max_steps,
    )
    calibration = run_rl_policy_calibration(
        out_dir=out / "rl_calibration",
        policy=str(policy),
        n_agents=int(n_agents),
        seeds=seed_list,
        max_steps=max_steps,
    )
    planner_sweep = _run_planner_sweep(
        out_dir=out / "planner_sweep",
        suite=str(suite),
        method=str(method),
        max_runs=max_runs,
        save_trace=bool(save_trace),
    )

    _write_json(out / "rl_contract.json", contract)
    _write_json(out / "rl_freeze_check.json", freeze)
    _write_json(out / "rl_smoke.json", smoke)
    _write_json(out / "rl_calibration.json", calibration)
    _write_json(Path(planner_sweep["acceptance_json"]), planner_sweep["acceptance"])

    meta = _method_metadata(str(method))
    artifact_paths = {
        "rl_contract": str(out / "rl_contract.json"),
        "rl_freeze_check": str(out / "rl_freeze_check.json"),
        "rl_smoke": str(out / "rl_smoke.json"),
        "rl_smoke_episodes": str(out / "rl_smoke" / "rl_smoke_episodes.csv"),
        "rl_calibration": str(out / "rl_calibration.json"),
        "rl_calibration_episodes": str(out / "rl_calibration" / "rl_calibration_episodes.csv"),
        "planner_results": planner_sweep["results_csv"],
        "planner_summary": planner_sweep["summary_csv"],
        "planner_result_schema": planner_sweep["result_schema_json"],
        "planner_suite_manifest": planner_sweep["suite_manifest"],
        "planner_acceptance": planner_sweep["acceptance_json"],
    }
    missing_artifacts = [name for name, path in artifact_paths.items() if not Path(path).exists()]
    checks = [
        _check("method_metadata_present", meta is not None, {"method": str(method)}),
        _check("method_marked_learned", bool(meta and meta.get("learned")), {"learned": None if meta is None else meta.get("learned")}),
        _check("rl_freeze_check_ok", bool(freeze.get("ok")), {"failed": [c["name"] for c in freeze.get("checks", []) if not c.get("ok")]}),
        _check("rl_smoke_ok", bool(smoke.get("ok")), {"failed": [c["name"] for c in smoke.get("checks", []) if not c.get("ok")]}),
        _check(
            "rl_calibration_ok",
            bool(calibration.get("ok")),
            {"failed": [c["name"] for c in calibration.get("checks", []) if not c.get("ok")]},
        ),
        _check(
            "planner_sweep_ran",
            int(planner_sweep["run_count"]) > 0,
            {"run_count": planner_sweep["run_count"], "planned_run_count": planner_sweep["planned_run_count"]},
        ),
        _check(
            "planner_sweep_guardrails_clear",
            int(planner_sweep["guardrail_total"]) == 0,
            {"guardrail_total": planner_sweep["guardrail_total"]},
        ),
        _check(
            "planner_sweep_metrics_finite",
            not planner_sweep["finite_metric_violations"],
            {"violations": planner_sweep["finite_metric_violations"]},
        ),
        _check(
            "planner_acceptance_no_failures",
            bool(planner_sweep["acceptance"].get("ok")),
            {
                "status": planner_sweep["acceptance"].get("status"),
                "rules_failed": planner_sweep["acceptance"].get("rules_failed"),
            },
        ),
        _check("expected_artifacts_present", not missing_artifacts, {"missing": missing_artifacts}),
    ]

    report = {
        "schema_version": LEARNED_SUBMISSION_BUNDLE_SCHEMA_VERSION,
        "ok": all(check["ok"] for check in checks),
        "method": str(method),
        "policy": str(policy),
        "suite": str(suite),
        "root": str(root),
        "n_agents": int(n_agents),
        "seeds": seed_list,
        "max_steps": None if max_steps is None else int(max_steps),
        "max_runs": None if max_runs is None else int(max_runs),
        "method_metadata": meta,
        "artifacts": artifact_paths,
        "checks": checks,
        "planner_sweep": {
            key: value
            for key, value in planner_sweep.items()
            if key not in {"acceptance"}
        },
        "acceptance": planner_sweep["acceptance"],
    }
    _write_json(bundle_path, report)
    return report
