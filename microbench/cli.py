from __future__ import annotations

import argparse
import glob
import json
import platform
from pathlib import Path
import subprocess
import sys
import tempfile
from tqdm import tqdm

from microbench.acceptance import check_acceptance
from microbench.config import builtin_scenario_paths, load_defaults, resolve_config_path
from microbench.planners import list_methods, planner_metadata
from microbench.types import RunSpec
from microbench.runner import run_episode
from microbench.metrics import append_result, write_summary
from microbench.replay import export_foxglove_comparison_mcap, export_foxglove_mcap, render_episode_report
from microbench.dataset import generate_dataset, expand_scenarios, expand_list, sanity_check_shard
from microbench.logging import wandb_logger
from microbench.rl.calibration import run_rl_policy_calibration
from microbench.rl.bc_training import build_behavior_cloned_policy_evidence, train_behavior_cloned_policy
from microbench.rl.evaluate import run_rl_policy_smoke
from microbench.rl.freeze import run_rl_freeze_check
from microbench.rl.learned_leaderboard import write_learned_policy_leaderboard
from microbench.rl.policies import POLICY_NAMES
from microbench.rl.schema import interface_contract
from microbench.rl.submission_bundle import (
    review_learned_policy_submission_bundle,
    run_learned_policy_submission_bundle,
    validate_learned_policy_submission_bundle,
    validate_learned_submission_manifest,
)
from microbench.rl.submission_schema_check import run_learned_submission_schema_check
from microbench.rl.validation_matrix import run_rl_validation_matrix
from microbench.tools import (
    DEFAULT_ADVANCED_COMPARISON_COMM_PROFILE,
    DEFAULT_ADVANCED_COMPARISON_DURATION_S,
    DEFAULT_ADVANCED_COMPARISON_METHODS,
    DEFAULT_ADVANCED_COMPARISON_MCAP,
    DEFAULT_ADVANCED_COMPARISON_N_AGENTS,
    DEFAULT_ADVANCED_COMPARISON_PLANNER_PRESET,
    DEFAULT_ADVANCED_COMPARISON_SCENARIO,
    DEFAULT_ADVANCED_COMPARISON_SEED,
    DEFAULT_EXTERNAL_REFERENCE_COMM_PROFILES,
    DEFAULT_EXTERNAL_REFERENCE_N_AGENTS,
    DEFAULT_EXTERNAL_REFERENCE_SCENARIOS,
    DEFAULT_EXTERNAL_REFERENCE_SEEDS,
    DEFAULT_HIGH_VOLUME_COMM_PROFILES,
    DEFAULT_HIGH_VOLUME_DURATION_S,
    DEFAULT_HIGH_VOLUME_N_AGENTS,
    DEFAULT_HIGH_VOLUME_PLANNER_PRESET,
    DEFAULT_HIGH_VOLUME_RUN_TIMEOUT_S,
    DEFAULT_HIGH_VOLUME_SCALE_SPAWN_PROFILE,
    DEFAULT_HIGH_VOLUME_SCENARIOS,
    DEFAULT_HIGH_VOLUME_SEEDS,
    DEFAULT_OPTIMIZER_REVIEW_SUITES,
    DEFAULT_SCALE_BENCHMARK_METHODS,
    DEFAULT_LATENCY_BUDGET_MS,
    OPTIMIZER_REVIEW_METHODS,
    MAX_RUNS_STRATEGIES,
    SCALE_SPAWN_PROFILES,
    build_external_reference_bundle,
    build_baseline_audit,
    build_current_schema_candidate,
    compare_current_schema_golden,
    mine_worst_cases,
    run_advanced_baseline_comparison,
    run_baseline_leaderboard,
    run_baseline_validation_matrix,
    run_baseline_behavior_smoke,
    run_baseline_reference_evidence,
    run_baseline_promotion_calibration,
    run_baseline_stable_review,
    run_high_volume_evidence,
    run_optimizer_suite_review,
    run_scale_benchmark,
    validate_external_reference_manifest,
    write_high_volume_leaderboard,
    write_baseline_report,
    write_current_schema_golden,
)
from microbench.scenarios import (
    list_official_suites,
    materialize_official_suite,
    suite_registry_dicts,
    suite_defaults,
    validate_scenario_file,
    validate_suite_manifest_file,
)

CANONICAL_SCENARIOS = [
    "config/scenarios/corridor.yaml",
    "config/scenarios/intersection.yaml",
    "config/scenarios/funnel.yaml",
    "config/scenarios/ring.yaml",
    "config/scenarios/crowd_swap.yaml",
    "config/scenarios/weather_event.yaml",
]

CANONICAL_3D_SCENARIOS = [
    "config/scenarios/stacked_swap_3d.yaml",
    "config/scenarios/layered_funnel_3d.yaml",
    "config/scenarios/layered_intersection_3d.yaml",
    "config/scenarios/weather_vertical_event_3d.yaml",
    "config/scenarios/vertical_crossing_obstacles_3d.yaml",
    "config/scenarios/urban_airspace_3d.yaml",
    "config/scenarios/urban_throughput_3d.yaml",
]

CANONICAL_PERCEPTION_SCENARIOS = [
    "config/scenarios/perception_sensor_occlusion.yaml",
    "config/scenarios/perception_fused_degraded.yaml",
    "config/scenarios/perception_stale_tracks.yaml",
]


def _parse_int_list(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        p = part.strip()
        if not p:
            continue
        if ":" in p:
            a, b = p.split(":", 1)
            start = int(a)
            end = int(b)
            step = 1 if end >= start else -1
            out.extend(list(range(start, end + step, step)))
        else:
            out.append(int(p))
    return out


def _parse_str_list(spec: str) -> list[str]:
    return [x.strip() for x in spec.split(",") if x.strip()]


def _parse_label_trace_args(items: list[str]) -> tuple[list[str], list[str]]:
    labels: list[str] = []
    traces: list[str] = []
    for item in items:
        if "=" not in item:
            raise ValueError(f"trace entries must use label=path form: {item!r}")
        label, path = item.split("=", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"trace entries must include a non-empty label and path: {item!r}")
        labels.append(label)
        traces.append(path)
    return labels, traces


def _expand_scenarios(spec: str) -> list[str]:
    scenarios: list[str] = []
    for token in _parse_str_list(spec):
        matches = sorted(glob.glob(token))
        if matches:
            scenarios.extend(matches)
        else:
            scenarios.append(str(resolve_config_path(token)))
    return scenarios


def _resolve_scenario_list(paths: list[str]) -> list[str]:
    return [str(resolve_config_path(p)) for p in paths]


def _git_commit() -> str | None:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        return out if out else None
    except Exception:
        return None


def _add_wandb_flags(p: argparse.ArgumentParser) -> None:
    p.add_argument("--wandb", action="store_true", help="Enable Weights & Biases logging")
    p.add_argument("--wandb-project", default="daa-microbench")
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group", default=None)
    p.add_argument("--wandb-name", default=None)
    p.add_argument("--wandb-tags", default=None, help="Comma-separated list of tags")
    p.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    p.add_argument("--wandb-upload-results", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--wandb-upload-traces", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wandb-upload-replays", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--wandb-upload-dataset", action=argparse.BooleanOptionalAction, default=False)


def _build_run_config(
    *,
    out_dir: str,
    suite: str,
    methods: list[str],
    scenarios: list[str],
    n_agents: list[int],
    seeds: list[int],
    comm_profiles: list[str],
    defaults: dict,
    policy_spec: str | None = None,
) -> dict:
    ncfg = defaults.get("neighbors", {})
    dcfg = defaults.get("dynamics", {})
    run_id = Path(out_dir).name
    git_commit = _git_commit()
    method_name = methods[0] if len(methods) == 1 else "multi"
    cfg = {
        "run_id": run_id,
        "suite": suite,
        "method_name": method_name,
        "method_version": git_commit,
        "methods": methods,
        "scenarios": [Path(s).stem for s in scenarios],
        "N_list": n_agents,
        "seed_min": min(seeds) if seeds else None,
        "seed_max": max(seeds) if seeds else None,
        "seed_count": len(seeds),
        "comm_profiles": comm_profiles,
        "policy_spec": policy_spec,
        "dt_s": float(defaults.get("sim", {}).get("dt_s", 0.02)),
        "duration_s_default": float(defaults.get("sim", {}).get("duration_s", 60.0)),
        "top_k": int(ncfg.get("top_k", 8)),
        "range_m": float(ncfg.get("range_m", 30.0)),
        "threat_metric": str(ncfg.get("threat_metric", "ttc")),
        "ttc_horizon_s": float(ncfg.get("ttc_horizon_s", 6.0)),
        "v_max_mps_default": float(dcfg.get("v_max_mps", 3.0)),
        "a_max_mps2_default": float(dcfg.get("a_max_mps2", 2.0)),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_model": platform.processor() or None,
        "git_commit": git_commit,
    }
    return cfg


def _run_once(args) -> dict:
    defaults = load_defaults()
    comm = args.comm or defaults.get("comm", {}).get("profile", "ideal_50hz")
    out_dir = args.out_dir or defaults.get("logging", {}).get("out_dir", "runs")
    save_trace_default = bool(defaults.get("logging", {}).get("save_trace", False))
    save_trace = save_trace_default if args.save_trace is None else bool(args.save_trace)
    spec = RunSpec(
        scenario_path=args.scenario,
        method=args.method,
        n_agents=int(args.n),
        seed=int(args.seed),
        comm_profile=comm,
        out_dir=out_dir,
        save_trace=save_trace,
        agent_methods=_parse_str_list(args.agent_methods) if args.agent_methods else None,
        policy_spec=args.policy_spec,
    )
    row = run_episode(spec)
    append_result(out_dir, row)
    write_summary(out_dir)
    return row


def _run_sweep(args) -> None:
    defaults = load_defaults()
    comm_default = defaults.get("comm", {}).get("profile", "ideal_50hz")
    comm_profiles = _parse_str_list(args.comm) if args.comm else [comm_default]
    out_dir = args.out_dir or defaults.get("logging", {}).get("out_dir", "runs")
    save_trace = bool(defaults.get("logging", {}).get("save_trace", False))

    scenarios = _expand_scenarios(args.scenarios)
    methods = _parse_str_list(args.methods)
    seeds = _parse_int_list(args.seeds)
    n_agents_list = _parse_int_list(args.n)

    specs: list[RunSpec] = []
    for scenario in scenarios:
        for method in methods:
            for comm in comm_profiles:
                for n_agents in n_agents_list:
                    for seed in seeds:
                        specs.append(
                            RunSpec(
                                scenario_path=scenario,
                                method=method,
                                n_agents=n_agents,
                                seed=seed,
                                comm_profile=comm,
                                out_dir=out_dir,
                                save_trace=save_trace,
                                policy_spec=args.policy_spec,
                            )
                        )

    run_cfg = _build_run_config(
        out_dir=out_dir,
        suite="custom_sweep",
        methods=methods,
        scenarios=scenarios,
        n_agents=n_agents_list,
        seeds=seeds,
        comm_profiles=comm_profiles,
        defaults=defaults,
        policy_spec=args.policy_spec,
    )
    wb_run = wandb_logger.init_run(args, run_cfg)
    try:
        for spec in tqdm(specs, desc="sweep", unit="run"):
            row = run_episode(spec)
            append_result(out_dir, row)

        summary_path = write_summary(out_dir)
        run_dir = Path(out_dir)
        wandb_logger.log_summary(
            wb_run,
            summary_csv_path=summary_path,
            results_csv_path=run_dir / "results.csv",
            extra_artifacts_paths={
                "upload_results": bool(args.wandb_upload_results),
                "upload_traces": bool(args.wandb_upload_traces),
                "upload_replays": bool(args.wandb_upload_replays),
                "traces_dir": run_dir / "worst_cases",
                "replays_dir": run_dir / "worst_cases",
                "worst_cases_index": run_dir / "worst_cases" / "index.csv",
            },
            metrics_dict=None,
        )
    finally:
        wandb_logger.finish(wb_run)


def _run_specs(specs: list[RunSpec], out_dir: str) -> None:
    for spec in tqdm(specs, desc="sweep", unit="run"):
        row = run_episode(spec)
        append_result(out_dir, row)
    write_summary(out_dir)


def _run_canonical_sweep(args) -> None:
    defaults = load_defaults()
    out_dir = args.out_dir or defaults.get("logging", {}).get("out_dir", "runs")
    save_trace = bool(defaults.get("logging", {}).get("save_trace", False))

    suite = args.suite
    stretch = bool(args.stretch)
    include_bursty = bool(args.include_bursty)

    official_suite_names = set(list_official_suites())
    if suite in official_suite_names:
        generated_dir = Path(out_dir) / "_generated_scenarios" / suite
        generated = materialize_official_suite(suite, generated_dir, overwrite=True, stretch=stretch)
        scenarios = [str(p) for p in generated["scenario_paths"]]
        defaults_for_suite = suite_defaults(suite, stretch=stretch)
        methods = _parse_str_list(args.methods) if args.methods else list(defaults_for_suite["default_methods"])
        n_agents = [int(x) for x in defaults_for_suite["n_agents"]]
        seeds = [int(x) for x in defaults_for_suite["seeds"]]
        comm_profiles = [str(x) for x in defaults_for_suite["comm_profiles"]]
    elif suite == "primary":
        scenarios = _resolve_scenario_list(CANONICAL_SCENARIOS)
        methods = _parse_str_list(args.methods or "")
        if not methods:
            raise ValueError("canonical-sweep --suite primary requires --methods")
        n_agents = [10, 20, 50] + ([100] if stretch else [])
        seeds = list(range(0, 100 if stretch else 50))
        comm_profiles = ["ideal_50hz", "realistic_v2v_50hz", "degraded_20hz"]
    elif suite == "baseline_sanity":
        scenarios = _resolve_scenario_list(CANONICAL_SCENARIOS)
        methods = _parse_str_list(args.methods) if args.methods else ["baseline_goal", "orca_heuristic"]
        n_agents = [10, 20] + ([100] if stretch else [])
        seeds = list(range(0, 100 if stretch else 20))
        comm_profiles = ["ideal_50hz", "realistic_v2v_50hz"]
    elif suite == "three_d":
        scenarios = _resolve_scenario_list(CANONICAL_3D_SCENARIOS)
        methods = _parse_str_list(args.methods) if args.methods else ["orca_heuristic"]
        n_agents = [4, 8] + ([16] if stretch else [])
        seeds = list(range(0, 20 if stretch else 10))
        comm_profiles = ["ideal_50hz"]
    elif suite == "perception_stress":
        scenarios = _resolve_scenario_list(CANONICAL_PERCEPTION_SCENARIOS)
        methods = _parse_str_list(args.methods) if args.methods else ["priority_yield"]
        n_agents = [6, 10] + ([20] if stretch else [])
        seeds = list(range(0, 20 if stretch else 10))
        comm_profiles = ["ideal_50hz", "degraded_20hz"]
    else:
        raise ValueError(f"Unknown suite: {suite}")

    if include_bursty and "bursty_stress_50hz" not in comm_profiles:
        comm_profiles.append("bursty_stress_50hz")

    specs: list[RunSpec] = []
    for scenario in scenarios:
        for method in methods:
            for comm in comm_profiles:
                for n in n_agents:
                    for seed in seeds:
                        specs.append(
                            RunSpec(
                                scenario_path=scenario,
                                method=method,
                                n_agents=n,
                                seed=seed,
                                comm_profile=comm,
                                out_dir=out_dir,
                                save_trace=save_trace,
                                policy_spec=args.policy_spec,
                            )
                        )

    if args.max_runs is not None:
        specs = specs[: max(0, int(args.max_runs))]

    if args.print_plan:
        print("canonical sweep plan:")
        print(f"  suite: {suite}")
        print(f"  methods: {','.join(methods)}")
        print(f"  scenarios: {','.join(scenarios)}")
        print(f"  N: {','.join(str(x) for x in n_agents)}")
        print(f"  seeds: {seeds[0]}:{seeds[-1]}")
        print(f"  comm: {','.join(comm_profiles)}")
        print(f"  total_runs: {len(specs)}")
        if args.no_run:
            return

    run_cfg = _build_run_config(
        out_dir=out_dir,
        suite=suite,
        methods=methods,
        scenarios=scenarios,
        n_agents=n_agents,
        seeds=seeds,
        comm_profiles=comm_profiles,
        defaults=defaults,
        policy_spec=args.policy_spec,
    )
    wb_run = wandb_logger.init_run(args, run_cfg)
    try:
        _run_specs(specs, out_dir)
        run_dir = Path(out_dir)
        wandb_logger.log_summary(
            wb_run,
            summary_csv_path=run_dir / "summary.csv",
            results_csv_path=run_dir / "results.csv",
            extra_artifacts_paths={
                "upload_results": bool(args.wandb_upload_results),
                "upload_traces": bool(args.wandb_upload_traces),
                "upload_replays": bool(args.wandb_upload_replays),
                "traces_dir": run_dir / "worst_cases",
                "replays_dir": run_dir / "worst_cases",
                "worst_cases_index": run_dir / "worst_cases" / "index.csv",
            },
            metrics_dict=None,
        )
    finally:
        wandb_logger.finish(wb_run)


def _print_validation_report(report) -> None:
    status = "ok" if report.ok else "FAIL"
    print(f"{status}: {report.kind} {report.path}")
    for warning in report.warnings:
        print(f"  warning: {warning}")
    for error in report.errors:
        print(f"  error: {error}")


def _validate_scenarios(args) -> None:
    reports = []
    scenario_paths: list[str] = []
    manifest_paths: list[str] = []
    generated_suites: list[str] = []

    if args.scenario:
        scenario_paths.extend(_expand_scenarios(args.scenario))
    if args.all_builtins:
        scenario_paths.extend(builtin_scenario_paths())
    if args.suite_manifest:
        manifest_paths.extend(_expand_scenarios(args.suite_manifest))
    if args.generated_suite:
        generated_suites.extend(_parse_str_list(args.generated_suite))
    if args.all_generated_suites:
        generated_suites.extend(list_official_suites())

    if not scenario_paths and not manifest_paths and not generated_suites:
        scenario_paths.extend(builtin_scenario_paths())

    for path in dict.fromkeys(scenario_paths):
        reports.append(validate_scenario_file(path))
    for path in dict.fromkeys(manifest_paths):
        reports.append(validate_suite_manifest_file(path))

    unknown_suites = sorted(set(generated_suites) - set(list_official_suites()))
    if unknown_suites:
        raise SystemExit(f"Unknown generated suite(s): {','.join(unknown_suites)}")
    if generated_suites:
        with tempfile.TemporaryDirectory(prefix="daa_suite_validate_") as td:
            for suite in dict.fromkeys(generated_suites):
                generated = materialize_official_suite(suite, Path(td) / suite, overwrite=True)
                reports.append(validate_suite_manifest_file(generated["manifest_path"]))

    for report in reports:
        if not args.quiet or not report.ok:
            _print_validation_report(report)
    failed = [r for r in reports if not r.ok]
    if failed:
        raise SystemExit(f"validation failed: {len(failed)} artifact(s) had errors")

    scenarios = sum(1 for r in reports if r.kind == "scenario")
    manifests = sum(1 for r in reports if r.kind == "suite_manifest")
    print(f"validation: PASS scenarios={scenarios} suite_manifests={manifests}")


def _list_suites(args) -> None:
    entries = suite_registry_dicts()
    if args.json:
        print(json.dumps(entries, indent=2, sort_keys=True))
        return

    print("suite,status,source,dimensions,scenario_count,acceptance_rules,default_methods,description")
    for entry in entries:
        print(
            ",".join(
                [
                    entry["suite"],
                    entry["status"],
                    entry["source"],
                    "+".join(entry["dimensions"]),
                    str(len(entry["scenarios"])),
                    str(entry["acceptance_rule_count"]),
                    "+".join(entry["default_methods"]) if entry["default_methods"] else "-",
                    entry["description"],
                ]
            )
        )


def _check_acceptance(args) -> None:
    report = check_acceptance(
        summary_csv=args.summary,
        results_csv=args.results,
        suite_manifest=args.suite_manifest,
        methods=_parse_str_list(args.methods) if args.methods else None,
        scenarios=_parse_str_list(args.scenarios) if args.scenarios else None,
        comm_profiles=_parse_str_list(args.comm_profiles) if args.comm_profiles else None,
        n_agents=[str(x) for x in _parse_int_list(args.n)] if args.n else None,
    )
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(
            "acceptance: "
            f"{report['status']} suite={report['suite']} rules={report['rules_total']} "
            f"passed={report['rules_passed']} warnings={report['rules_warned']} "
            f"failed={report['rules_failed']} skipped={report['rules_skipped']}"
        )
        for check in report["checks"]:
            status = check["status"]
            if status == "pass" and not args.verbose:
                continue
            if status == "skipped" and not args.verbose:
                continue
            print(
                f"{status}: {check.get('name')} "
                f"scope={check.get('scope')} method={check.get('method')} "
                f"scenario={check.get('scenario')} comm={check.get('comm_profile')} "
                f"N={check.get('n_agents')} metric={check.get('metric')} "
                f"{check.get('operator')} {check.get('value')} matched={check.get('matched_rows')}"
            )
            for violation in check.get("violations", [])[:10]:
                print(
                    "  row: "
                    f"method={violation.get('method')} scenario={violation.get('scenario')} "
                    f"comm={violation.get('comm_profile')} N={violation.get('N')} "
                    f"observed={violation.get('observed')} reason={violation.get('reason')}"
                )
            if len(check.get("violations", [])) > 10:
                print(f"  ... {len(check['violations']) - 10} more violation(s)")
    if not report["ok"]:
        raise SystemExit(f"acceptance failed: {report['rules_failed']} rule(s) failed")


def _baseline_report(args) -> None:
    out = write_baseline_report(
        summary_csv=args.summary,
        results_csv=args.results,
        suite=args.suite,
        out=args.out,
        generated_by=args.generated_by,
    )
    print(f"done: baseline comparison report saved to {out}")


def _baseline_leaderboard(args) -> None:
    suites = list_official_suites() if args.suites == "all" else _parse_str_list(args.suites)
    report = run_baseline_leaderboard(
        out_dir=args.out_dir,
        suites=suites,
        methods=_parse_str_list(args.methods) if args.methods else None,
        n_agents=_parse_int_list(args.n) if args.n else None,
        seeds=_parse_int_list(args.seeds) if args.seeds else None,
        comm_profiles=_parse_str_list(args.comm) if args.comm else None,
        max_runs=args.max_runs,
        max_runs_strategy=str(args.max_runs_strategy),
        stretch=bool(args.stretch),
        resume=bool(args.resume),
        max_wall_time_s=args.max_wall_time_s,
        run_timeout_s=args.run_timeout_s,
    )
    run_cfg = {
        "run_id": Path(args.out_dir).name,
        "suite": "baseline_leaderboard",
        "method_name": "baseline_leaderboard",
        "method_version": _git_commit(),
        "suites": [suite["suite"] for suite in report.get("suites", [])],
        "methods": report.get("methods", []),
        "schema_version": report.get("schema_version"),
        "complete": bool(report.get("complete", False)),
        "selected_complete": bool(report.get("selected_complete", False)),
        "timeout_run_count": int(report.get("timeout_run_count") or 0),
        "max_runs_strategy": report.get("max_runs_strategy"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "git_commit": _git_commit(),
    }
    wb_run = wandb_logger.init_run(args, run_cfg)
    try:
        wandb_logger.log_baseline_leaderboard(
            wb_run,
            report,
            upload_results=bool(args.wandb_upload_results),
        )
    finally:
        wandb_logger.finish(wb_run)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "REVIEW"
        print(
            "baseline-leaderboard: "
            f"{status} suites={len(report['suites'])} methods={len(report['methods'])} "
            f"leaderboard={report['leaderboard_path']}"
        )
        for suite in report["suites"]:
            truncated = " truncated" if suite["truncated_by_max_runs"] else ""
            incomplete = " incomplete" if not suite.get("selected_complete", False) else ""
            stopped = " wall-time-stop" if suite.get("stopped_by_wall_time") else ""
            timeouts = f" timeouts={suite.get('timeout_run_count', 0)}" if suite.get("timeout_run_count", 0) else ""
            print(
                f"  {suite['suite']}: runs={suite['selected_completed_count']}/{suite['selected_run_count']} "
                f"planned={suite['planned_run_count']} ok={suite['ok']}"
                f"{truncated}{incomplete}{stopped}{timeouts} report={suite['report_path']}"
            )
        if report["aggregate_ranking"]:
            best = report["aggregate_ranking"][0]
            print(f"  best_score_v0: rank=1 method={best['method']} score={best['score_v0_mean']}")

    if args.require_pass and not report["ok"]:
        failed = [suite["suite"] for suite in report["suites"] if not suite["ok"]]
        raise SystemExit(f"baseline leaderboard acceptance failed: {','.join(failed)}")
    if args.require_complete and not report["complete"]:
        incomplete = [suite["suite"] for suite in report["suites"] if not suite.get("complete", False)]
        raise SystemExit(f"baseline leaderboard incomplete: {','.join(incomplete)}")


def _baseline_audit(args) -> None:
    report = build_baseline_audit(root=args.root)
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        summary = report["summary"]
        print(
            "baseline-audit: "
            f"public_alpha_ready={report['public_alpha_ready']} "
            f"stable_v1_ready={report['stable_v1_ready']} "
            f"methods={summary['method_count']} "
            f"reference_ready={summary['public_alpha_reference_ready_count']} "
            f"experimental={summary['experimental_runnable_count']}"
        )
        for entry in report["methods"]:
            blockers = ",".join(entry["blockers"]) if entry["blockers"] else "-"
            print(
                f"{entry['method']}: readiness={entry['readiness']} "
                f"role={entry['role']} status={entry['status']} "
                f"fidelity={entry.get('fidelity')} blockers={blockers}"
            )
    if args.require_public_alpha_ready and not report["public_alpha_ready"]:
        raise SystemExit("baseline audit failed: public-alpha reference baseline blockers present")
    if args.require_stable_v1_ready and not report["stable_v1_ready"]:
        raise SystemExit("baseline audit failed: stable-v1 baseline blockers present")


def _baseline_smoke(args) -> None:
    report = run_baseline_behavior_smoke(
        out_dir=args.out_dir,
        methods=_parse_str_list(args.methods) if args.methods else None,
        scenario_ids=_parse_str_list(args.scenarios) if args.scenarios else None,
        n_agents=int(args.n),
        seed=int(args.seed),
        comm_profile=str(args.comm),
    )
    report_path = Path(args.out_dir) / "baseline_smoke.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(
            f"baseline-smoke: {status} "
            f"runs={report['run_count']} methods={len(report['methods'])} "
            f"scenarios={','.join(report['scenario_ids'])}"
        )
        for check in report["checks"]:
            check_status = "ok" if check["ok"] else "FAIL"
            print(f"  {check_status}: {check['name']}")
        print(f"  report: {report_path}")

    if args.require_pass and not report["ok"]:
        failed = [check["name"] for check in report["checks"] if not check["ok"]]
        raise SystemExit(f"baseline smoke failed: {','.join(failed)}")


def _baseline_promotion(args) -> None:
    report = run_baseline_promotion_calibration(
        out_dir=args.out_dir,
        root=args.root,
        methods=_parse_str_list(args.methods) if args.methods else None,
        behavior_report=args.behavior_report,
        include_experimental_suite=not args.skip_experimental_suite,
    )
    report_path = Path(args.out_dir) / "baseline_promotion.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        summary = report["summary"]
        print(
            "baseline-promotion: "
            f"public_alpha_calibrated={report['public_alpha_calibrated']} "
            f"stable_v1_ready={report['stable_v1_ready']} "
            f"calibrated={summary['calibration_ready_count']}/{summary['method_count']} "
            f"stable={summary['stable_v1_ready_count']}/{summary['method_count']}"
        )
        for entry in report["methods_detail"]:
            blockers = ",".join(entry["stable_v1_blockers"]) if entry["stable_v1_blockers"] else "-"
            print(
                f"{entry['method']}: calibrated={entry['calibration_ready']} "
                f"stable_v1={entry['stable_v1_ready']} blockers={blockers}"
            )
        print(f"  report: {report_path}")

    if args.require_calibrated and not report["public_alpha_calibrated"]:
        blockers = {
            entry["method"]: entry["calibration_blockers"]
            for entry in report["methods_detail"]
            if entry["calibration_blockers"]
        }
        raise SystemExit(f"baseline promotion calibration failed: {blockers}")
    if args.require_stable_v1_ready and not report["stable_v1_ready"]:
        blockers = report["summary"]["stable_v1_blockers"]
        raise SystemExit(f"baseline promotion stable-v1 blockers present: {blockers}")


def _baseline_evidence(args) -> None:
    report = run_baseline_reference_evidence(
        mpc_profile_iters=int(args.mpc_profile_iters),
        mpc_p95_max_ms=float(args.max_mpc_p95_ms),
        optimizer_profile_iters=int(args.optimizer_profile_iters),
        optimizer_p95_max_ms=float(args.max_optimizer_p95_ms),
        artifact_dir=args.out_dir,
        save_optimizer_traces=bool(args.save_optimizer_traces),
    )
    report_path = Path(args.out_dir) / "baseline_evidence.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        summary = report["summary"]
        print(
            "baseline-evidence: "
            f"{status} checks={summary['check_count']} failed={summary['failed_count']} "
            f"methods={','.join(report['methods'])}"
        )
        for check in report["checks"]:
            check_status = "ok" if check["ok"] else "FAIL"
            print(f"  {check_status}: {check['method']} {check['name']}")
        print(f"  report: {report_path}")

    if args.require_pass and not report["ok"]:
        failed = [f"{check['method']}:{check['name']}" for check in report["checks"] if not check["ok"]]
        raise SystemExit(f"baseline evidence checks failed: {','.join(failed)}")


def _validate_external_reference(args) -> None:
    report = validate_external_reference_manifest(
        manifest=args.manifest,
        require_artifacts=bool(args.require_artifacts),
    )

    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(
            "validate-external-reference: "
            f"{status} reference={report['reference_id']} "
            f"method={report['related_microbench_method']} fidelity={report['fidelity']}"
        )
        for error in report["errors"]:
            print(f"  error: {error}")
        for warning in report["warnings"]:
            print(f"  warning: {warning}")

    if args.require_pass and not report["ok"]:
        raise SystemExit(f"external reference manifest failed validation: {report['errors']}")


def _external_reference_bundle(args) -> None:
    report = build_external_reference_bundle(
        method_family=str(args.method_family),
        out_dir=args.out_dir,
        scenarios=_parse_str_list(args.scenarios) if args.scenarios else None,
        suites=_parse_str_list(args.suites) if args.suites else None,
        n_agents=_parse_int_list(args.n) if args.n else None,
        seeds=_parse_int_list(args.seeds) if args.seeds else None,
        comm_profiles=_parse_str_list(args.comm) if args.comm else None,
        reference_id=args.reference_id,
        related_microbench_method=args.related_microbench_method,
        implementation_name=args.implementation_name,
        source_url=args.source_url,
        license_text=args.license,
        upstream_commit=str(args.upstream_commit),
        runner_type=str(args.runner_type),
        runner_command=str(args.runner_command),
        runner_working_dir=str(args.runner_working_dir),
        overwrite=bool(args.overwrite),
    )

    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(
            "external-reference-bundle: "
            f"{status} reference={report['reference_id']} "
            f"rows={report['run_count']} scenarios={report['scenario_count']} "
            f"bundle={report['out_dir']}"
        )
        print(f"  manifest: {report['manifest']}")
        print(f"  adapter_plan: {report['adapter_plan_md']}")
        print(f"  run_matrix: {report['run_matrix_csv']}")
        print(f"  results_template: {report['results_template']}")
        print(f"  post_run_validation: {report['post_run_validation_command']}")

    if args.require_pass and not report["ok"]:
        raise SystemExit(f"external reference bundle failed validation: {report['validation']['errors']}")


def _advanced_baseline_comparison(args) -> None:
    duration_s = None if args.full_duration else float(args.duration_s)
    report = run_advanced_baseline_comparison(
        out_dir=args.out_dir,
        scenario=args.scenario,
        methods=_parse_str_list(args.methods) if args.methods else None,
        n_agents=int(args.n),
        seed=int(args.seed),
        comm_profile=str(args.comm),
        duration_s=duration_s,
        save_traces=bool(args.save_traces),
        export_foxglove_mcap=bool(args.export_foxglove_mcap),
        foxglove_mcap_path=args.foxglove_mcap_out,
        mcap_trail_frames=int(args.mcap_trail_frames),
        mcap_max_sensing_links=int(args.mcap_max_sensing_links),
        mcap_compression=str(args.mcap_compression),
        planner_preset=str(args.planner_preset),
    )

    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "REVIEW"
        print(
            "advanced-baseline-comparison: "
            f"{status} runs={report['run_count']}/{report['planned_run_count']} "
            f"scenario={report['scenario']} preset={report['planner_preset']} "
            f"report={report['report_path']}"
        )
        for row in report["ranking"]:
            print(
                f"  rank={row['rank']} method={row['method']} score_v0={row.get('score_v0')} "
                f"collision_rate={row.get('collision_episode_rate')} completion={row.get('completion_rate_mean')} "
                f"min_sep={row.get('min_sep_min_mean')}"
            )
        if report.get("foxglove_mcap"):
            print(f"  foxglove_mcap: {report['foxglove_mcap']['path']}")

    if args.require_pass and not report["ok"]:
        raise SystemExit(
            "advanced baseline comparison failed: "
            f"checks={report['checks']} guardrails={report['guardrail_failures']} "
            f"nonfinite={report['nonfinite_methods']}"
        )


def _optimizer_suite_review(args) -> None:
    suites = list_official_suites() if args.suites == "all" else _parse_str_list(args.suites)
    report = run_optimizer_suite_review(
        out_dir=args.out_dir,
        suites=suites,
        methods=_parse_str_list(args.methods) if args.methods else None,
        n_agents=_parse_int_list(args.n) if args.n else None,
        seeds=_parse_int_list(args.seeds) if args.seeds else None,
        comm_profiles=_parse_str_list(args.comm) if args.comm else None,
        max_runs=args.max_runs,
        max_runs_strategy=str(args.max_runs_strategy),
        stretch=bool(args.stretch),
        resume=bool(args.resume),
        max_wall_time_s=args.max_wall_time_s,
        run_timeout_s=args.run_timeout_s,
        max_trace_cases=int(args.max_trace_cases),
        save_review_traces=bool(args.save_review_traces),
        trace_max_steps=int(args.trace_max_steps),
        guardrail_retries=int(args.guardrail_retries),
    )

    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "REVIEW"
        findings = report["findings"]
        print(
            "optimizer-suite-review: "
            f"{status} suites={len(report['suites'])} methods={len(report['methods'])} "
            f"selected_complete={report['selected_complete']} report={report['report_path']}"
        )
        print(
            "  findings: "
            f"collisions={findings['collision_episode_rows']} "
            f"negative_clearance={findings['negative_clearance_rows']} "
            f"incomplete={findings['incomplete_episode_rows']} "
            f"guardrails={findings['guardrail_rows']} "
            f"persistent_guardrails={findings['persistent_guardrail_rows']} "
            f"dimensions={','.join(findings['dimensions_covered']) or '-'}"
        )
        for entry in report["method_summaries"]:
            print(
                f"  {entry['method']}: runs={entry['run_count']} "
                f"collision_rate={entry['collision_episode_rate']} "
                f"completion={entry['completion_rate_mean']} "
                f"worst_min_sep={entry['min_sep_min_worst_m']} "
                f"planner_p95_max={entry['planner_ms_p95_max']}"
            )
        if report["review_cases"]:
            print("  review_cases:")
            for case in report["review_cases"]:
                trace_status = f" trace={case.get('trace_status')}" if case.get("trace_status") else ""
                print(
                    f"    {case['suite']} {case['scenario']} {case['method']} "
                    f"N={case['N']} seed={case['seed']} comm={case['comm_profile']}{trace_status}"
                )

    if args.require_pass and not report["ok"]:
        raise SystemExit(
            "optimizer suite review failed: "
            f"official_acceptance_ok={report['official_acceptance_ok']} "
            f"selected_complete={report['selected_complete']} findings={report['findings']}"
        )
    if args.require_complete and not report["publication_complete"]:
        raise SystemExit("optimizer suite review incomplete for publication-scale claims")


def _scale_benchmark(args) -> None:
    report = run_scale_benchmark(
        out_dir=args.out_dir,
        scenarios=_expand_scenarios(args.scenarios),
        methods=_parse_str_list(args.methods) if args.methods else None,
        n_agents=_parse_int_list(args.n) if args.n else None,
        seeds=_parse_int_list(args.seeds) if args.seeds else None,
        comm_profiles=_parse_str_list(args.comm) if args.comm else None,
        max_runs=args.max_runs,
        max_runs_strategy=str(args.max_runs_strategy),
        resume=bool(args.resume),
        max_wall_time_s=args.max_wall_time_s,
        run_timeout_s=args.run_timeout_s,
        duration_s=args.duration_s,
        save_traces=bool(args.save_traces),
        scale_spawn_profile=str(args.scale_spawn_profile),
        planner_preset=str(args.planner_preset),
        plan_only=bool(args.plan_only),
    )

    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "REVIEW"
        print(
            "scale-benchmark: "
            f"{status} runs={report['selected_completed_count']}/{report['selected_run_count']} "
            f"planned={report['planned_run_count']} timeouts={report['timeout_run_count']} "
            f"summary={report['scale_summary_csv']}"
        )
        for row in report["method_summaries"]:
            print(
                f"  rank={row['rank']} method={row['method']} "
                f"max_completed_N={row.get('max_completed_N')} max_clean_N={row.get('max_clean_N')} "
                f"timeout_rate={row.get('timeout_rate_mean')} "
                f"collision_rate={row.get('collision_episode_rate_mean')}"
            )

    if args.require_complete and not report["complete"]:
        raise SystemExit("scale benchmark incomplete for publication-scale claims")
    if args.require_pass and not report["ok"]:
        raise SystemExit(
            "scale benchmark failed: "
            f"selected_complete={report['selected_complete']} "
            f"timeouts={report['timeout_run_count']} stopped={report['stopped_by_wall_time']}"
        )


def _high_volume_leaderboard(args) -> None:
    scale_summaries: list[str] = []
    for item in args.scale_summary:
        scale_summaries.extend(_parse_str_list(item))
    report = write_high_volume_leaderboard(
        scale_summaries=scale_summaries,
        out=args.out,
        latency_budget_ms=float(args.latency_budget_ms),
        generated_by="python -m microbench.cli high-volume-leaderboard",
    )

    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        top = report["overall_ranking"][0] if report["overall_ranking"] else {}
        print(
            "high-volume-leaderboard: "
            f"methods={report['method_count']} cells={report['scale_cell_count']} "
            f"top={top.get('method')} score={top.get('overall_score')} "
            f"report={args.out} csv={report['leaderboard_csv']}"
        )
        for axis, rows in report["axis_rankings"].items():
            leader = rows[0] if rows else {}
            print(f"  {axis}: {leader.get('method')} score={leader.get('score')}")


def _high_volume_evidence(args) -> None:
    report = run_high_volume_evidence(
        out_dir=args.out_dir,
        scenarios=_expand_scenarios(args.scenarios) if args.scenarios else None,
        methods=_parse_str_list(args.methods) if args.methods else None,
        n_agents=_parse_int_list(args.n) if args.n else None,
        seeds=_parse_int_list(args.seeds) if args.seeds else None,
        comm_profiles=_parse_str_list(args.comm) if args.comm else None,
        max_runs=args.max_runs,
        max_runs_strategy=str(args.max_runs_strategy),
        resume=bool(args.resume),
        max_wall_time_s=args.max_wall_time_s,
        run_timeout_s=args.run_timeout_s,
        duration_s=args.duration_s,
        save_traces=bool(args.save_traces),
        scale_spawn_profile=str(args.scale_spawn_profile),
        planner_preset=str(args.planner_preset),
        latency_budget_ms=float(args.latency_budget_ms),
        plan_only=bool(args.plan_only),
    )

    if args.json:
        print(json.dumps(report, allow_nan=False, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "REVIEW"
        top = report["overall_ranking"][0] if report["overall_ranking"] else {}
        print(
            "high-volume-evidence: "
            f"{status} runs={report['selected_completed_count']}/{report['selected_run_count']} "
            f"planned={report['planned_run_count']} timeouts={report['timeout_run_count']} "
            f"top={top.get('method')} report={report['report_path']}"
        )
        if report["high_volume_leaderboard_path"]:
            print(
                f"  leaderboard={report['high_volume_leaderboard_path']} "
                f"csv={report['high_volume_leaderboard_csv']}"
            )
            for axis, rows in report["axis_rankings"].items():
                leader = rows[0] if rows else {}
                print(f"  {axis}: {leader.get('method')} score={leader.get('score')}")
        else:
            print("  leaderboard=skipped plan_only=True")

    if args.require_complete and not report["complete"]:
        raise SystemExit("high-volume evidence incomplete for publication-scale claims")
    if args.require_pass and not report["ok"]:
        raise SystemExit(
            "high-volume evidence failed: "
            f"selected_complete={report['selected_complete']} "
            f"timeouts={report['timeout_run_count']} stopped={report['stopped_by_wall_time']}"
        )


def _baseline_review(args) -> None:
    duration_s = None if args.full_duration else float(args.duration_s)
    report = run_baseline_stable_review(
        out_dir=args.out_dir,
        root=args.root,
        methods=_parse_str_list(args.methods) if args.methods else None,
        lanes=_parse_str_list(args.lanes) if args.lanes else None,
        duration_s=duration_s,
        max_runs=args.max_runs,
        plan_only=bool(args.plan_only),
    )
    report_path = Path(args.out_dir) / "baseline_review.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["review_checks_pass"] else "REVIEW"
        if report["plan_only"]:
            status = "PLAN"
        print(
            "baseline-review: "
            f"{status} runs={report['run_count']}/{report['planned_run_count']} "
            f"methods={len(report['methods'])} lanes={len(report['lanes'])}"
        )
        for entry in report["methods_detail"]:
            failed = ",".join(check["name"] for check in entry["failed_checks"]) if entry["failed_checks"] else "-"
            print(
                f"{entry['method']}: checks={entry['review_checks_pass']} "
                f"recommendation={entry['metadata_recommendation']} failed={failed}"
            )
        print(f"  report: {report_path}")

    if args.require_pass and not report["plan_only"] and not report["review_checks_pass"]:
        failed = {
            entry["method"]: entry["failed_checks"]
            for entry in report["methods_detail"]
            if entry["failed_checks"] or not entry["review_checks_pass"]
        }
        raise SystemExit(f"baseline review checks failed: {failed}")


def _baseline_validation_matrix(args) -> None:
    report = run_baseline_validation_matrix(
        out_dir=args.out_dir,
        methods=_parse_str_list(args.methods) if args.methods else None,
        lanes=_parse_str_list(args.lanes) if args.lanes else None,
        duration_s=args.duration_s,
        max_runs=args.max_runs,
        plan_only=bool(args.plan_only),
    )
    report_path = Path(args.out_dir) / "baseline_validation_matrix.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PLAN" if report["plan_only"] else ("PASS" if report["ok"] else "FAIL")
        print(
            "baseline-validation-matrix: "
            f"{status} runs={report['run_count']}/{report['selected_run_count']} "
            f"methods={len(report['methods'])} lanes={len(report['lanes'])} "
            f"behavior_pass={report['behavior_pass']}"
        )
        print(f"  report: {report_path}")

    if args.require_pass and not report["plan_only"] and not report["ok"]:
        failed = [check for check in report["checks"] if check["severity"] == "gate" and not check["ok"]]
        raise SystemExit(f"baseline validation matrix gate checks failed: {failed}")
    if args.require_behavior_pass and not report["plan_only"] and not report["behavior_pass"]:
        failed = [check for check in report["checks"] if check["severity"] == "behavior" and not check["ok"]]
        raise SystemExit(f"baseline validation matrix behavior checks failed: {failed}")


def _rl_smoke(args) -> None:
    report = run_rl_policy_smoke(
        out_dir=args.out_dir,
        policy=str(args.policy),
        policy_spec=args.policy_spec,
        scenario_ids=_parse_str_list(args.scenarios) if args.scenarios else None,
        n_agents=int(args.n),
        seeds=_parse_int_list(args.seeds),
        comm_profile=str(args.comm),
        max_steps=args.max_steps,
    )
    report_path = Path(args.out_dir) / "rl_smoke.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(
            f"rl-smoke: {status} policy={report['policy']} runs={report['run_count']} "
            f"scenarios={','.join(report['scenario_ids'])} dimensions={'+'.join(report['dimensions'])}"
        )
        for check in report["checks"]:
            check_status = "ok" if check["ok"] else "FAIL"
            print(f"  {check_status}: {check['name']}")
        print(f"  report: {report_path}")

    if args.require_pass and not report["ok"]:
        failed = [check["name"] for check in report["checks"] if not check["ok"]]
        raise SystemExit(f"RL smoke failed: {','.join(failed)}")


def _rl_calibration(args) -> None:
    report = run_rl_policy_calibration(
        out_dir=args.out_dir,
        policy=str(args.policy),
        policy_spec=args.policy_spec,
        n_agents=int(args.n),
        seeds=_parse_int_list(args.seeds),
        max_steps=args.max_steps,
    )
    report_path = Path(args.out_dir) / "rl_calibration.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(
            f"rl-calibration: {status} policy={report['policy']} runs={report['run_count']} "
            f"bands={'+'.join(report['bands'])}"
        )
        for check in report["checks"]:
            check_status = "ok" if check["ok"] else "FAIL"
            print(f"  {check_status}: {check['name']}")
        print(f"  report: {report_path}")

    if args.require_pass and not report["ok"]:
        failed = [check["name"] for check in report["checks"] if not check["ok"]]
        raise SystemExit(f"RL calibration failed: {','.join(failed)}")


def _rl_validation_matrix(args) -> None:
    report = run_rl_validation_matrix(
        out_dir=args.out_dir,
        policy=str(args.policy),
        policy_spec=args.policy_spec,
        lanes=_parse_str_list(args.lanes) if args.lanes else None,
        seeds=_parse_int_list(args.seeds) if args.seeds else None,
        duration_s=args.duration_s,
        n_agents=args.n,
        max_steps=args.max_steps,
        plan_only=bool(args.plan_only),
    )
    report_path = Path(args.out_dir) / "rl_validation_matrix.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PLAN" if report["plan_only"] else ("PASS" if report["ok"] else "FAIL")
        print(
            f"rl-validation-matrix: {status} policy={report['policy']} "
            f"runs={report['run_count']}/{report['planned_run_count']} "
            f"lanes={len(report['lanes'])} behavior_pass={report['behavior_pass']}"
        )
        for check in report["checks"]:
            check_status = "ok" if check["ok"] else "FAIL"
            print(f"  {check_status}: {check['name']} [{check['severity']}]")
        print(f"  report: {report_path}")

    if args.require_pass and not report["plan_only"] and not report["ok"]:
        failed = [check["name"] for check in report["checks"] if check["severity"] == "gate" and not check["ok"]]
        raise SystemExit(f"RL validation matrix failed: {','.join(failed)}")
    if args.require_behavior_pass and not report["plan_only"] and not report["behavior_pass"]:
        failed = [check["name"] for check in report["checks"] if check["severity"] == "behavior" and not check["ok"]]
        raise SystemExit(f"RL validation matrix behavior checks failed: {','.join(failed)}")


def _train_learned_bc(args) -> None:
    report = train_behavior_cloned_policy(
        out_dir=args.out_dir,
        lanes=_parse_str_list(args.lanes) if args.lanes else None,
        seeds=_parse_int_list(args.seeds) if args.seeds else None,
        max_steps=int(args.max_steps),
        hidden_dim=int(args.hidden_dim),
        ridge=float(args.ridge),
        rollout_noise_std=float(args.rollout_noise_std),
        eval_lanes=_parse_str_list(args.eval_lanes) if args.eval_lanes else None,
        eval_max_steps=args.eval_max_steps,
        policy_name=str(args.policy_name),
        seed=int(args.seed),
        overwrite=bool(args.overwrite),
        run_validation=not bool(args.skip_validation),
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        validation = report.get("validation_matrix") or {}
        print(
            f"train-learned-bc: {status} policy={report['policy_name']} "
            f"samples={report['sample_count']} hidden_dim={report['hidden_dim']} "
            f"fit_rmse={report['fit_rmse']}"
        )
        print(f"  policy_spec: {report['policy_spec']}")
        print(f"  model_artifact: {report['model_artifact']}")
        if validation:
            print(
                f"  validation_matrix: ok={validation.get('ok')} "
                f"runs={validation.get('run_count')}/{validation.get('planned_run_count')} "
                f"behavior_pass={validation.get('behavior_pass')}"
            )

    if args.require_pass and not report["ok"]:
        failed = [check["name"] for check in report["checks"] if not check["ok"]]
        raise SystemExit(f"behavior-cloned learned-policy training failed: {','.join(failed)}")


def _learned_bc_evidence(args) -> None:
    report = build_behavior_cloned_policy_evidence(
        out_dir=args.out_dir,
        lanes=_parse_str_list(args.lanes) if args.lanes else None,
        seeds=_parse_int_list(args.seeds) if args.seeds else None,
        max_steps=int(args.max_steps),
        hidden_dim=int(args.hidden_dim),
        ridge=float(args.ridge),
        rollout_noise_std=float(args.rollout_noise_std),
        eval_lanes=_parse_str_list(args.eval_lanes) if args.eval_lanes else None,
        eval_max_steps=args.eval_max_steps,
        policy_name=str(args.policy_name),
        seed=int(args.seed),
        bundle_suite=str(args.suite),
        bundle_n_agents=int(args.n),
        bundle_seeds=_parse_int_list(args.bundle_seeds) if args.bundle_seeds else None,
        bundle_max_steps=args.bundle_max_steps,
        bundle_max_runs=args.max_runs,
        include_fixture_bundles=not bool(args.skip_fixtures),
        overwrite=bool(args.overwrite),
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        training = report.get("training", {})
        leaderboard = report.get("leaderboard", {})
        print(
            f"learned-bc-evidence: {status} policy={args.policy_name} "
            f"samples={training.get('sample_count')} fit_rmse={training.get('fit_rmse')} "
            f"bundles={leaderboard.get('bundle_count')}"
        )
        print(f"  policy_spec: {training.get('policy_spec')}")
        print(f"  bc_bundle: {report.get('bundle_paths', {}).get('bc')}")
        print(f"  leaderboard: {leaderboard.get('leaderboard_path')}")
        print(f"  csv: {leaderboard.get('leaderboard_csv')}")

    if args.require_pass and not report["ok"]:
        failed = [check["name"] for check in report["checks"] if not check["ok"]]
        raise SystemExit(f"learned BC evidence failed: {','.join(failed)}")


def _rl_contract(args) -> None:
    report = interface_contract(top_k=int(args.top_k))
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
    if args.json or not args.out:
        print(payload)
    else:
        print(f"done: RL interface contract saved to {args.out}")


def _rl_freeze_check(args) -> None:
    report = run_rl_freeze_check(root=args.root)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(f"rl-freeze-check: {status} interface={report['interface_version']}")
        for check in report["checks"]:
            check_status = "ok" if check["ok"] else "FAIL"
            print(f"  {check_status}: {check['name']}")
        if args.out:
            print(f"  report: {args.out}")

    if args.require_pass and not report["ok"]:
        failed = [check["name"] for check in report["checks"] if not check["ok"]]
        raise SystemExit(f"RL freeze check failed: {','.join(failed)}")


def _learned_submission_schema_check(args) -> None:
    report = run_learned_submission_schema_check(root=args.root)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(f"learned-submission-schema-check: {status} schema={report['schema_version']}")
        for check in report["checks"]:
            check_status = "ok" if check["ok"] else "FAIL"
            print(f"  {check_status}: {check['name']}")
        if args.out:
            print(f"  report: {args.out}")

    if args.require_pass and not report["ok"]:
        failed = [check["name"] for check in report["checks"] if not check["ok"]]
        raise SystemExit(f"learned submission schema check failed: {','.join(failed)}")


def _learned_submission_bundle(args) -> None:
    report = run_learned_policy_submission_bundle(
        out_dir=args.out_dir,
        method=str(args.method),
        policy=str(args.policy),
        policy_spec=args.policy_spec,
        suite=str(args.suite),
        root=args.root,
        n_agents=int(args.n),
        seeds=_parse_int_list(args.seeds),
        max_steps=args.max_steps,
        max_runs=args.max_runs,
        save_trace=bool(args.save_trace),
        submission_manifest=args.submission_manifest,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(
            f"learned-submission-bundle: {status} method={report['method']} policy={report['policy']} "
            f"suite={report['suite']} runs={report['planner_sweep']['run_count']}"
        )
        for check in report["checks"]:
            check_status = "ok" if check["ok"] else "FAIL"
            print(f"  {check_status}: {check['name']}")
        print(f"  report: {Path(args.out_dir) / 'learned_submission_bundle.json'}")

    if args.require_pass and not report["ok"]:
        failed = [check["name"] for check in report["checks"] if not check["ok"]]
        raise SystemExit(f"learned submission bundle failed: {','.join(failed)}")


def _validate_learned_bundle(args) -> None:
    report = validate_learned_policy_submission_bundle(bundle=args.bundle)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(
            f"validate-learned-bundle: {status} method={report.get('method')} "
            f"policy={report.get('policy')} suite={report.get('suite')}"
        )
        for check in report["checks"]:
            check_status = "ok" if check["ok"] else "FAIL"
            print(f"  {check_status}: {check['name']}")
        print(f"  bundle: {report['bundle_json']}")

    if args.require_pass and not report["ok"]:
        failed = [check["name"] for check in report["checks"] if not check["ok"]]
        raise SystemExit(f"learned bundle validation failed: {','.join(failed)}")


def _validate_learned_manifest(args) -> None:
    report = validate_learned_submission_manifest(
        manifest=args.manifest,
        bundle_root=args.bundle_root,
        allow_undisclosed=bool(args.allow_undisclosed),
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        policy = report.get("policy") or {}
        benchmark = report.get("benchmark") or {}
        print(
            f"validate-learned-manifest: {status} policy={policy.get('name')} "
            f"method={policy.get('method')} suite={benchmark.get('suite')}"
        )
        for check in report["checks"]:
            check_status = "ok" if check["ok"] else "FAIL"
            print(f"  {check_status}: {check['name']}")
        if report.get("unknown_fields"):
            print(f"  unknown_fields: {','.join(report['unknown_fields'])}")
        print(f"  manifest: {report['manifest']}")

    if args.require_pass and not report["ok"]:
        failed = [check["name"] for check in report["checks"] if not check["ok"]]
        raise SystemExit(f"learned manifest validation failed: {','.join(failed)}")


def _review_learned_bundle(args) -> None:
    report = review_learned_policy_submission_bundle(bundle=args.bundle)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        score = report.get("score_v0", {})
        safety = report.get("dimensions", {}).get("safety", {})
        mission = report.get("dimensions", {}).get("mission", {})
        compute = report.get("dimensions", {}).get("compute", {})
        print(
            f"review-learned-bundle: {status} method={report.get('method')} "
            f"policy={report.get('policy')} suite={report.get('suite')} "
            f"recommendation={report.get('recommendation')}"
        )
        print(
            f"  runs: {report.get('run_count')}/{report.get('planned_run_count')} "
            f"summary_rows={report.get('summary_row_count')} result_rows={report.get('result_row_count')}"
        )
        print(
            f"  score_v0: mean={score.get('mean')} best={score.get('best')} worst={score.get('worst')}"
        )
        print(
            "  safety: "
            f"collision_episodes={safety.get('collision_episode_count')} "
            f"collision_rate_mean={safety.get('collision_episode_rate_mean')} "
            f"min_sep_p05_min_m={safety.get('min_sep_p05_min_m')}"
        )
        print(
            "  mission: "
            f"completion_rate_mean={mission.get('completion_rate_mean')} "
            f"deadlock_time_pct_mean={mission.get('deadlock_time_pct_mean')}"
        )
        print(
            "  compute: "
            f"planner_ms_p95_max={compute.get('planner_ms_p95_max')} "
            f"timeouts={compute.get('planner_timeout_count')} "
            f"errors={compute.get('planner_error_count')} "
            f"fallbacks={compute.get('planner_fallback_count')}"
        )
        if report.get("limitations"):
            print(f"  limitations: {','.join(report['limitations'])}")
        if args.out:
            print(f"  report: {args.out}")

    if args.require_pass and not report["ok"]:
        failed = [check["name"] for check in report["checks"] if not check["ok"]]
        raise SystemExit(f"learned bundle review failed: {','.join(failed)}")


def _learned_leaderboard(args) -> None:
    report = write_learned_policy_leaderboard(
        bundles=list(args.bundle),
        out=args.out,
        csv_out=args.csv_out,
    )

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(
            f"learned-leaderboard: {status} bundles={report['bundle_count']} "
            f"reviewable={report['reviewable_count']} candidates={report['leaderboard_candidate_count']}"
        )
        for row in report["rows"][:10]:
            print(
                "  "
                f"rank={row.get('development_rank')} method={row.get('method')} "
                f"policy={row.get('policy')} score={row.get('score_v0_mean')} "
                f"recommendation={row.get('recommendation')} ok={row.get('ok')}"
            )
        print(f"  leaderboard={report['leaderboard_path']}")
        print(f"  csv={report['leaderboard_csv']}")

    if args.require_pass and not report["ok"]:
        failed = [row["bundle"] for row in report["rows"] if not row.get("ok")]
        raise SystemExit(f"learned leaderboard failed bundles: {','.join(failed)}")


def _golden_current_schema(args) -> None:
    if args.update and args.candidate:
        raise SystemExit("--update cannot be combined with --candidate")

    if args.update:
        out = write_current_schema_golden(args.golden_dir)
        report = compare_current_schema_golden(candidate_dir=out, golden_dir=args.golden_dir)
    elif args.candidate:
        report = compare_current_schema_golden(candidate_dir=args.candidate, golden_dir=args.golden_dir)
    else:
        with tempfile.TemporaryDirectory(prefix="daa_current_schema_check_") as td:
            candidate = build_current_schema_candidate(td)
            report = compare_current_schema_golden(candidate_dir=candidate, golden_dir=args.golden_dir)

    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        status = "PASS" if report["ok"] else "FAIL"
        print(
            f"current-schema golden: {status} "
            f"schema={report['schema_version']} candidate={report['candidate_dir']} golden={report['golden_dir']}"
        )
        if report["mismatches"]:
            for mismatch in report["mismatches"][:20]:
                print(
                    f"  {mismatch.get('reason')}: file={mismatch.get('file')} "
                    f"row={mismatch.get('row')} field={mismatch.get('field')} "
                    f"expected={mismatch.get('expected')} actual={mismatch.get('actual')}"
                )
            if len(report["mismatches"]) > 20:
                print(f"  ... {len(report['mismatches']) - 20} more mismatch(es)")
    if not report["ok"]:
        raise SystemExit(f"current-schema golden check failed: {len(report['mismatches'])} mismatch(es)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="microbench")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_run = sub.add_parser("run", help="Run a single episode")
    p_run.add_argument("--scenario", required=True)
    p_run.add_argument("--method", required=True)
    p_run.add_argument("--n", required=True, type=int)
    p_run.add_argument("--seed", required=True, type=int)
    p_run.add_argument("--comm", default=None)
    p_run.add_argument("--out-dir", default=None)
    p_run.add_argument(
        "--save-trace",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Save trace_episode.jsonl for Foxglove export and episode reports",
    )
    p_run.add_argument("--policy-spec", default=None, help="JSON/YAML external policy spec for --method learned_policy_spec")
    p_run.add_argument(
        "--agent-methods",
        default=None,
        help="Optional comma-separated methods for heterogeneous agents; length must be 1 or N",
    )

    p_sweep = sub.add_parser("sweep", help="Run a parameter sweep")
    p_sweep.add_argument("--scenarios", required=True, help="Comma-separated list and/or globs")
    p_sweep.add_argument("--methods", required=True, help="Comma-separated planner methods")
    p_sweep.add_argument("--seeds", required=True, help="Comma list and/or ranges, e.g. 0:9")
    p_sweep.add_argument("--n", required=True, help="Comma list and/or ranges of agent counts")
    p_sweep.add_argument("--comm", default=None, help="One or more comm profiles (comma-separated)")
    p_sweep.add_argument("--out-dir", default=None)
    p_sweep.add_argument("--policy-spec", default=None, help="JSON/YAML external policy spec for learned_policy_spec runs")
    _add_wandb_flags(p_sweep)

    p_episode_report = sub.add_parser("episode-report", help="Render a multi-panel HTML episode analysis report")
    p_episode_report.add_argument("--trace", required=True, help="Path to trace_episode.jsonl or trace_collision_*.jsonl")
    p_episode_report.add_argument("--out", required=True, help="Output HTML path")
    p_episode_report.add_argument(
        "--max-frames",
        type=int,
        default=800,
        help="Maximum frames embedded in the report; <=0 keeps all frames",
    )
    p_episode_report.add_argument(
        "--plotly-source",
        choices=("auto", "inline", "cdn"),
        default="auto",
        help="How to load Plotly: inline when installed, force inline, or CDN fallback",
    )

    p_foxglove = sub.add_parser("foxglove-export", help="Export a trace as a Foxglove-compatible MCAP log")
    p_foxglove.add_argument("--trace", required=True, help="Path to trace_episode.jsonl or trace_collision_*.jsonl")
    p_foxglove.add_argument("--out", required=True, help="Output MCAP path")
    p_foxglove.add_argument("--trail-frames", type=int, default=200, help="Number of history frames in trail entities")
    p_foxglove.add_argument("--max-sensing-links", type=int, default=200, help="Maximum sensing/V2V links per frame")
    p_foxglove.add_argument("--compression", choices=("none", "lz4", "zstd"), default="zstd", help="MCAP chunk compression")

    p_foxglove_cmp = sub.add_parser(
        "foxglove-comparison-export",
        help="Export multiple traces into one namespaced Foxglove MCAP log",
    )
    p_foxglove_cmp.add_argument(
        "--trace",
        action="append",
        required=True,
        help="Trace entry in label=trace_episode.jsonl form; repeat once per comparison panel",
    )
    p_foxglove_cmp.add_argument("--out", required=True, help="Output MCAP path")
    p_foxglove_cmp.add_argument(
        "--topic-root",
        default="/daa/comparison",
        help="Root topic prefix for namespaced scene topics",
    )
    p_foxglove_cmp.add_argument("--trail-frames", type=int, default=200, help="Number of history frames in trail entities")
    p_foxglove_cmp.add_argument("--max-sensing-links", type=int, default=200, help="Maximum sensing/V2V links per frame")
    p_foxglove_cmp.add_argument(
        "--compression",
        choices=("none", "lz4", "zstd"),
        default="zstd",
        help="MCAP chunk compression",
    )

    p_ds = sub.add_parser("generate-dataset", help="Generate diffusion training dataset shards")
    p_ds.add_argument("--scenario", required=True, help="Scenario path(s) or globs (comma-separated)")
    p_ds.add_argument("--method", default="orca_heuristic")
    p_ds.add_argument("--n", default=None, help="Agent counts, e.g. 10,20,50 (preferred)")
    p_ds.add_argument("--N", default=None, help="Deprecated alias for --n")
    p_ds.add_argument("--seeds", required=True, help="Seed list/range, e.g. 0:99")
    p_ds.add_argument("--T", type=int, default=20, help="Number of planning steps in U0")
    p_ds.add_argument("--dt-plan-s", type=float, default=0.10, help="Action sequence sampling dt")
    p_ds.add_argument("--horizon_steps", type=int, default=None, help="Deprecated alias for --T")
    p_ds.add_argument("--goal-dist-cap", type=float, default=60.0)
    p_ds.add_argument("--comm", default=None, help="Comma-separated comm profile list")
    p_ds.add_argument("--out-dir", default="datasets")
    p_ds.add_argument("--shard-size", type=int, default=50000)
    p_ds.add_argument("--quality-filter", choices=["none", "collision_free", "safe_expert"], default="safe_expert")
    p_ds.add_argument("--filter-min-sep-m", type=float, default=0.2)

    p_sc = sub.add_parser("sanity-check-dataset", help="Sanity check one dataset shard")
    p_sc.add_argument("--shard", required=True)
    p_sc.add_argument("--out-plot", default=None, help="Optional histogram path")

    p_cs = sub.add_parser("canonical-sweep", help="Run a canonical benchmark suite")
    p_cs.add_argument(
        "--suite",
        required=True,
        choices=["primary", "baseline_sanity", "three_d", "perception_stress"] + list_official_suites(),
    )
    p_cs.add_argument(
        "--methods",
        default=None,
        help="Comma-separated methods (required for primary; generated/3D suites have registry defaults)",
    )
    p_cs.add_argument("--out-dir", default=None)
    p_cs.add_argument("--stretch", action="store_true", help="Enable stretch settings (N=100 and more seeds)")
    p_cs.add_argument("--include-bursty", action="store_true", help="Include bursty_stress_50hz comm profile")
    p_cs.add_argument("--print-plan", action="store_true", help="Print resolved run matrix before execution")
    p_cs.add_argument("--no-run", action="store_true", help="Only print matrix; do not execute")
    p_cs.add_argument("--max-runs", type=int, default=None, help="Optional cap for debugging/smoke tests")
    p_cs.add_argument("--policy-spec", default=None, help="JSON/YAML external policy spec for learned_policy_spec runs")
    _add_wandb_flags(p_cs)

    p_ms = sub.add_parser("materialize-suite", help="Write generated official suite scenarios and manifest")
    p_ms.add_argument("--suite", required=True, choices=list_official_suites())
    p_ms.add_argument("--out-dir", required=True)
    p_ms.add_argument("--overwrite", action="store_true")
    p_ms.add_argument("--stretch", action="store_true", help="Write stretch run-matrix recommendations into manifest")
    p_ms.add_argument("--print-plan", action="store_true")

    p_val = sub.add_parser("validate-scenarios", help="Validate scenario YAMLs and generated suite manifests")
    p_val.add_argument("--scenario", default=None, help="Scenario path(s) or globs (comma-separated)")
    p_val.add_argument("--suite-manifest", default=None, help="Suite manifest path(s) or globs (comma-separated)")
    p_val.add_argument("--generated-suite", default=None, help="Generated suite id(s), comma-separated")
    p_val.add_argument("--all-builtins", action="store_true", help="Validate config/scenarios/*.yaml")
    p_val.add_argument("--all-generated-suites", action="store_true", help="Materialize and validate all generated suites")
    p_val.add_argument("--quiet", action="store_true", help="Only print failures and final summary")

    p_ls = sub.add_parser("list-suites", help="List known benchmark suites and registry status")
    p_ls.add_argument("--json", action="store_true", help="Emit suite registry as JSON")

    p_acc = sub.add_parser("check-acceptance", help="Evaluate suite acceptance rules against result CSVs")
    p_acc.add_argument("--summary", required=True, help="Path to summary.csv")
    p_acc.add_argument("--suite-manifest", required=True, help="Path to generated suite_manifest.yaml")
    p_acc.add_argument("--results", default=None, help="Optional path to results.csv for results-scoped rules")
    p_acc.add_argument("--methods", default=None, help="Only evaluate rules/rows for these methods")
    p_acc.add_argument("--scenarios", default=None, help="Only evaluate rules/rows for these scenario ids")
    p_acc.add_argument(
        "--comm-profiles",
        "--comm",
        dest="comm_profiles",
        default=None,
        help="Only evaluate rules/rows for these comm profiles",
    )
    p_acc.add_argument("--n", default=None, help="Only evaluate rules/rows for these agent counts")
    p_acc.add_argument("--json", action="store_true", help="Emit machine-readable acceptance report")
    p_acc.add_argument("--verbose", action="store_true", help="Print pass/skipped checks as well as failures/warnings")

    p_br = sub.add_parser("baseline-report", help="Build a compact baseline comparison report from summary.csv")
    p_br.add_argument("--summary", required=True, help="Path to summary.csv")
    p_br.add_argument("--suite", required=True, help="Suite name used for the comparison run")
    p_br.add_argument("--out", required=True, help="Output report JSON path")
    p_br.add_argument("--results", default=None, help="Optional results.csv path for run_count")
    p_br.add_argument("--generated-by", default=None, help="Optional reproducibility command or label")

    p_bl = sub.add_parser("baseline-leaderboard", help="Run baselines across official suites and write leaderboard JSON")
    p_bl.add_argument("--out-dir", required=True, help="Fresh output directory for leaderboard artifacts")
    p_bl.add_argument(
        "--suites",
        default="all",
        help="Comma-separated official suite ids, or 'all' for every generated official suite",
    )
    p_bl.add_argument(
        "--methods",
        default=None,
        help="Comma-separated methods; defaults to serious built-in baseline methods",
    )
    p_bl.add_argument("--n", default=None, help="Optional agent-count override list/range, e.g. 4,8")
    p_bl.add_argument("--seeds", default=None, help="Optional seed override list/range, e.g. 0:2")
    p_bl.add_argument("--comm", default=None, help="Optional comm-profile override list")
    p_bl.add_argument("--max-runs", type=int, default=None, help="Optional per-suite run cap for smoke checks")
    p_bl.add_argument(
        "--max-runs-strategy",
        choices=MAX_RUNS_STRATEGIES,
        default="prefix",
        help="How to choose runs when --max-runs truncates a suite matrix",
    )
    p_bl.add_argument("--resume", action="store_true", help="Resume from existing per-suite results.csv rows")
    p_bl.add_argument(
        "--max-wall-time-s",
        type=float,
        default=None,
        help="Optional global wall-clock budget; writes partial progress when exceeded",
    )
    p_bl.add_argument(
        "--run-timeout-s",
        type=float,
        default=None,
        help="Optional hard per-episode wall-clock timeout for leaderboard jobs",
    )
    p_bl.add_argument("--stretch", action="store_true", help="Use stretch suite defaults")
    _add_wandb_flags(p_bl)
    p_bl.add_argument("--json", action="store_true", help="Emit machine-readable leaderboard summary")
    p_bl.add_argument("--require-pass", action="store_true", help="Fail if generated-suite acceptance fails")
    p_bl.add_argument("--require-complete", action="store_true", help="Fail if the selected leaderboard matrix is incomplete")

    p_ba = sub.add_parser("baseline-audit", help="Audit built-in baseline metadata, docs, tests, and suite coverage")
    p_ba.add_argument("--root", default=".", help="Repository root used for docs/tests coverage checks")
    p_ba.add_argument("--json", action="store_true", help="Emit machine-readable audit report")
    p_ba.add_argument(
        "--require-public-alpha-ready",
        action="store_true",
        help="Fail if required public-alpha reference baselines are blocked",
    )
    p_ba.add_argument(
        "--require-stable-v1-ready",
        action="store_true",
        help="Fail if any baseline still has stable-v1 promotion blockers",
    )

    p_bs = sub.add_parser("baseline-smoke", help="Run compact 2D/3D behavioral smoke checks for baselines")
    p_bs.add_argument("--out-dir", required=True, help="Fresh output directory for smoke results")
    p_bs.add_argument(
        "--methods",
        default=None,
        help="Comma-separated methods; defaults to all non-template built-in baselines",
    )
    p_bs.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated generated smoke scenario ids; defaults to one 2D and one 3D case",
    )
    p_bs.add_argument("--n", type=int, default=4, help="Agent count for each smoke episode")
    p_bs.add_argument("--seed", type=int, default=0, help="Seed for each smoke episode")
    p_bs.add_argument("--comm", default="ideal_50hz", help="Communication profile for each smoke episode")
    p_bs.add_argument("--json", action="store_true", help="Emit machine-readable smoke report")
    p_bs.add_argument("--require-pass", action="store_true", help="Fail if any smoke check fails")

    p_bp = sub.add_parser("baseline-promotion", help="Calibrate experimental baselines and report promotion blockers")
    p_bp.add_argument("--out-dir", required=True, help="Fresh output directory for promotion calibration artifacts")
    p_bp.add_argument("--root", default=".", help="Repository root used for docs/tests coverage checks")
    p_bp.add_argument(
        "--methods",
        default=None,
        help="Comma-separated promotion candidates; defaults to cbf_qp,mpc_local,negotiation_yield",
    )
    p_bp.add_argument(
        "--behavior-report",
        default=None,
        help="Optional existing baseline_smoke.json to reuse instead of rerunning behavior smoke",
    )
    p_bp.add_argument(
        "--skip-experimental-suite",
        action="store_true",
        help="Skip the generated experimental CBF/MPC calibration suite",
    )
    p_bp.add_argument("--json", action="store_true", help="Emit machine-readable promotion report")
    p_bp.add_argument("--require-calibrated", action="store_true", help="Fail if public-alpha calibration evidence is missing")
    p_bp.add_argument("--require-stable-v1-ready", action="store_true", help="Fail if stable-v1 promotion blockers remain")

    p_be = sub.add_parser(
        "baseline-evidence",
        help="Run targeted CBF/MPC/NMPC/BVC tube-DMPC/dynamic tube-DMPC/RMADER/EGO-Swarm/VO/RVO reference-evidence checks",
    )
    p_be.add_argument("--out-dir", required=True, help="Fresh output directory for evidence artifacts")
    p_be.add_argument("--mpc-profile-iters", type=int, default=20, help="Dense 3D MPC timing samples")
    p_be.add_argument("--max-mpc-p95-ms", type=float, default=50.0, help="Allowed dense 3D MPC p95 per-call latency")
    p_be.add_argument("--optimizer-profile-iters", type=int, default=8, help="Dense 3D optimizer-grade timing samples")
    p_be.add_argument(
        "--max-optimizer-p95-ms",
        type=float,
        default=80.0,
        help="Allowed dense 3D optimizer-grade p95 per-call latency",
    )
    p_be.add_argument(
        "--save-optimizer-traces",
        action="store_true",
        help="Write trace_episode.jsonl artifacts for mpc_nonlinear and ego_swarm_opt evidence episodes",
    )
    p_be.add_argument("--json", action="store_true", help="Emit machine-readable evidence report")
    p_be.add_argument("--require-pass", action="store_true", help="Fail if any evidence check fails")

    p_ext = sub.add_parser(
        "validate-external-reference",
        help="Validate an external official/reference-baseline manifest without executing external code",
    )
    p_ext.add_argument("--manifest", required=True, help="External reference manifest YAML/JSON path")
    p_ext.add_argument(
        "--require-artifacts",
        action="store_true",
        help="Require declared artifact paths such as results.csv to exist and match the core result fields",
    )
    p_ext.add_argument("--json", action="store_true", help="Emit machine-readable validation report")
    p_ext.add_argument("--require-pass", action="store_true", help="Fail if the manifest has validation errors")

    p_ext_bundle = sub.add_parser(
        "external-reference-bundle",
        help="Create a portable scenario/run-matrix bundle for an official external implementation",
    )
    p_ext_bundle.add_argument("--method-family", required=True, help="External method family, e.g. rmader")
    p_ext_bundle.add_argument("--out-dir", required=True, help="Fresh output directory for the bundle")
    p_ext_bundle.add_argument(
        "--scenarios",
        default=",".join(DEFAULT_EXTERNAL_REFERENCE_SCENARIOS),
        help="Comma-separated scenario ids/paths to copy into the bundle",
    )
    p_ext_bundle.add_argument(
        "--suites",
        default=None,
        help="Optional comma-separated generated suite ids to materialize into the bundle",
    )
    p_ext_bundle.add_argument(
        "--n",
        default=",".join(str(x) for x in DEFAULT_EXTERNAL_REFERENCE_N_AGENTS),
        help="Comma-separated agent counts or ranges, e.g. 4,8 or 4:8",
    )
    p_ext_bundle.add_argument(
        "--seeds",
        default=",".join(str(x) for x in DEFAULT_EXTERNAL_REFERENCE_SEEDS),
        help="Comma-separated seeds or ranges",
    )
    p_ext_bundle.add_argument(
        "--comm",
        default=",".join(DEFAULT_EXTERNAL_REFERENCE_COMM_PROFILES),
        help="Comma-separated communication profiles",
    )
    p_ext_bundle.add_argument("--reference-id", default=None, help="External reference id for the manifest")
    p_ext_bundle.add_argument(
        "--related-microbench-method",
        default=None,
        help="Canonical built-in method this official reference should be compared against",
    )
    p_ext_bundle.add_argument("--implementation-name", default=None, help="External implementation display name")
    p_ext_bundle.add_argument("--source-url", default=None, help="External implementation source URL")
    p_ext_bundle.add_argument("--license", default=None, help="External implementation license")
    p_ext_bundle.add_argument(
        "--upstream-commit",
        default="<external-repo-commit>",
        help="External implementation commit/tag to record in the manifest",
    )
    p_ext_bundle.add_argument(
        "--runner-type",
        choices=("external_process", "docker", "ros", "manual_import"),
        default="external_process",
        help="How the external reference is expected to run",
    )
    p_ext_bundle.add_argument(
        "--runner-command",
        default="<external command that consumes run_matrix.csv and scenarios/*.yaml>",
        help="Command note for the external runner",
    )
    p_ext_bundle.add_argument(
        "--runner-working-dir",
        default="<external-workspace>",
        help="External runner working directory note",
    )
    p_ext_bundle.add_argument("--overwrite", action="store_true", help="Replace existing bundle files")
    p_ext_bundle.add_argument("--json", action="store_true", help="Emit machine-readable bundle report")
    p_ext_bundle.add_argument("--require-pass", action="store_true", help="Fail if the generated manifest is invalid")

    p_abc = sub.add_parser(
        "advanced-baseline-comparison",
        help="Run a compact shared 3D comparison lane for advanced baselines",
    )
    p_abc.add_argument("--out-dir", required=True, help="Fresh output directory for comparison artifacts")
    p_abc.add_argument(
        "--scenario",
        default=DEFAULT_ADVANCED_COMPARISON_SCENARIO,
        help="Scenario path or bundled scenario id",
    )
    p_abc.add_argument(
        "--methods",
        default=None,
        help=(
            "Comma-separated methods; defaults to "
            + ",".join(DEFAULT_ADVANCED_COMPARISON_METHODS)
        ),
    )
    p_abc.add_argument("--n", type=int, default=DEFAULT_ADVANCED_COMPARISON_N_AGENTS, help="Agent count")
    p_abc.add_argument("--seed", type=int, default=DEFAULT_ADVANCED_COMPARISON_SEED, help="Scenario seed")
    p_abc.add_argument("--comm", default=DEFAULT_ADVANCED_COMPARISON_COMM_PROFILE, help="Communication profile")
    p_abc.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_ADVANCED_COMPARISON_DURATION_S,
        help="Scenario duration override; use --full-duration for the scenario default",
    )
    p_abc.add_argument("--full-duration", action="store_true", help="Use the scenario's configured duration")
    p_abc.add_argument("--save-traces", action="store_true", help="Save per-method episode traces")
    p_abc.add_argument(
        "--export-foxglove-mcap",
        action="store_true",
        help="Save traces and export one panelized Foxglove MCAP comparison file",
    )
    p_abc.add_argument(
        "--foxglove-mcap-out",
        default=None,
        help=f"Optional MCAP output path; defaults to <out-dir>/{DEFAULT_ADVANCED_COMPARISON_MCAP}",
    )
    p_abc.add_argument("--mcap-trail-frames", type=int, default=200, help="Foxglove trail history frames")
    p_abc.add_argument("--mcap-max-sensing-links", type=int, default=200, help="Maximum sensing/V2V links per frame")
    p_abc.add_argument(
        "--mcap-compression",
        choices=("none", "lz4", "zstd"),
        default="zstd",
        help="MCAP chunk compression for the comparison export",
    )
    p_abc.add_argument(
        "--planner-preset",
        default=DEFAULT_ADVANCED_COMPARISON_PLANNER_PRESET,
        help="Planner config preset for comparison rows; use 'scale' for scale-tuned optimizer visualizations",
    )
    p_abc.add_argument("--json", action="store_true", help="Emit machine-readable comparison report")
    p_abc.add_argument(
        "--require-pass",
        action="store_true",
        help="Fail if the comparison artifact is incomplete or has planner guardrail errors",
    )

    p_osr = sub.add_parser(
        "optimizer-suite-review",
        help="Run optimizer-grade baselines across official suites and write a review report",
    )
    p_osr.add_argument("--out-dir", required=True, help="Fresh output directory for optimizer review artifacts")
    p_osr.add_argument(
        "--suites",
        default=",".join(DEFAULT_OPTIMIZER_REVIEW_SUITES),
        help="Comma-separated official suite ids, or 'all' for every generated official suite",
    )
    p_osr.add_argument(
        "--methods",
        default=None,
        help="Comma-separated methods; defaults to " + ",".join(OPTIMIZER_REVIEW_METHODS),
    )
    p_osr.add_argument("--n", default=None, help="Optional agent-count override list/range, e.g. 4,8")
    p_osr.add_argument("--seeds", default=None, help="Optional seed override list/range, e.g. 0:2")
    p_osr.add_argument("--comm", default=None, help="Optional comm-profile override list")
    p_osr.add_argument("--max-runs", type=int, default=None, help="Optional per-suite run cap for smoke checks")
    p_osr.add_argument(
        "--max-runs-strategy",
        choices=MAX_RUNS_STRATEGIES,
        default="balanced",
        help="How to choose runs when --max-runs truncates a suite matrix",
    )
    p_osr.add_argument("--resume", action="store_true", help="Resume from existing per-suite results.csv rows")
    p_osr.add_argument(
        "--max-wall-time-s",
        type=float,
        default=None,
        help="Optional global wall-clock budget; writes partial progress when exceeded",
    )
    p_osr.add_argument(
        "--run-timeout-s",
        type=float,
        default=None,
        help="Optional hard per-episode wall-clock timeout for optimizer review jobs",
    )
    p_osr.add_argument("--stretch", action="store_true", help="Use stretch suite defaults")
    p_osr.add_argument(
        "--max-trace-cases",
        type=int,
        default=4,
        help="Number of worst-case episodes to include as Foxglove review candidates",
    )
    p_osr.add_argument(
        "--save-review-traces",
        action="store_true",
        help="Rerun review cases with full trace_episode.jsonl output for Foxglove export",
    )
    p_osr.add_argument(
        "--trace-max-steps",
        type=int,
        default=4000,
        help="Maximum full-trace frames saved for --save-review-traces",
    )
    p_osr.add_argument(
        "--guardrail-retries",
        type=int,
        default=1,
        help="Retry rows with planner guardrails this many times to separate transient spikes from persistent failures",
    )
    p_osr.add_argument("--json", action="store_true", help="Emit machine-readable optimizer review report")
    p_osr.add_argument("--require-pass", action="store_true", help="Fail if selected optimizer review checks fail")
    p_osr.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless the selected suites are complete enough for publication-scale claims",
    )

    p_scale = sub.add_parser(
        "scale-benchmark",
        help="Run methods across scenario files and fleet-size ladders with timeout-aware scale reporting",
    )
    p_scale.add_argument("--out-dir", required=True, help="Fresh output directory for scale artifacts")
    p_scale.add_argument(
        "--scenarios",
        required=True,
        help="Comma-separated scenario paths/ids/globs, e.g. config/scenarios/stacked_swap_3d.yaml,config/scenarios/urban_conflict_3d.yaml,config/scenarios/urban_throughput_3d.yaml",
    )
    p_scale.add_argument(
        "--methods",
        default=None,
        help="Comma-separated methods; defaults to " + ",".join(DEFAULT_SCALE_BENCHMARK_METHODS),
    )
    p_scale.add_argument("--n", default="4,8,16,30", help="Agent-count ladder, e.g. 4,8,16,30")
    p_scale.add_argument("--seeds", default="0", help="Seed list/range, e.g. 0:2")
    p_scale.add_argument("--comm", default="realistic_v2v_50hz", help="Communication profile list")
    p_scale.add_argument("--max-runs", type=int, default=None, help="Optional run cap for development checkpoints")
    p_scale.add_argument(
        "--max-runs-strategy",
        choices=MAX_RUNS_STRATEGIES,
        default="balanced",
        help="How to choose runs when --max-runs truncates the scenario/method/N matrix",
    )
    p_scale.add_argument("--resume", action="store_true", help="Resume from existing scale results.csv rows")
    p_scale.add_argument(
        "--max-wall-time-s",
        type=float,
        default=None,
        help="Optional global wall-clock budget; writes partial progress when exceeded",
    )
    p_scale.add_argument(
        "--run-timeout-s",
        type=float,
        default=None,
        help="Optional hard per-episode wall-clock timeout for scale jobs",
    )
    p_scale.add_argument(
        "--duration-s",
        type=float,
        default=None,
        help="Optional duration override applied to copied scale scenario files",
    )
    p_scale.add_argument("--save-traces", action="store_true", help="Save full traces for scale runs")
    p_scale.add_argument(
        "--scale-spawn-profile",
        choices=SCALE_SPAWN_PROFILES,
        default="none",
        help="Optional spawn adaptation for copied scale scenario files; dense widens four-way lanes for larger N",
    )
    p_scale.add_argument(
        "--planner-preset",
        default="default",
        help="Planner config preset for scale rows; use 'scale' for bounded high-N optimizer effort",
    )
    p_scale.add_argument("--plan-only", action="store_true", help="Write prepared scenario files and report the planned matrix")
    p_scale.add_argument("--json", action="store_true", help="Emit machine-readable scale benchmark report")
    p_scale.add_argument("--require-pass", action="store_true", help="Fail if selected scale runs are incomplete or timed out")
    p_scale.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless the full requested scale matrix completed without hard timeouts",
    )

    p_hvl = sub.add_parser(
        "high-volume-leaderboard",
        help="Build a multi-axis leaderboard from one or more scale_summary.csv artifacts",
    )
    p_hvl.add_argument(
        "--scale-summary",
        action="append",
        required=True,
        help="Path or comma-separated paths to scale_summary.csv; repeat for multiple scale runs",
    )
    p_hvl.add_argument("--out", required=True, help="Output leaderboard JSON path; a sibling CSV is also written")
    p_hvl.add_argument(
        "--latency-budget-ms",
        type=float,
        default=DEFAULT_LATENCY_BUDGET_MS,
        help="Planner p95 latency budget used for the runtime/scale axis",
    )
    p_hvl.add_argument("--json", action="store_true", help="Emit the full leaderboard report as JSON")

    p_hve = sub.add_parser(
        "high-volume-evidence",
        help="Run the default high-volume scale suite and package the leaderboard evidence artifacts",
    )
    p_hve.add_argument("--out-dir", required=True, help="Output directory for evidence, scale runs, and leaderboard files")
    p_hve.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated scenario paths/ids/globs; defaults to " + ",".join(DEFAULT_HIGH_VOLUME_SCENARIOS),
    )
    p_hve.add_argument(
        "--methods",
        default=None,
        help="Comma-separated methods; defaults to " + ",".join(DEFAULT_SCALE_BENCHMARK_METHODS),
    )
    p_hve.add_argument(
        "--n",
        default=",".join(str(v) for v in DEFAULT_HIGH_VOLUME_N_AGENTS),
        help="Agent-count list/range; default is the high-volume N=30 evidence cell",
    )
    p_hve.add_argument(
        "--seeds",
        default=",".join(str(v) for v in DEFAULT_HIGH_VOLUME_SEEDS),
        help="Seed list/range for evidence rows",
    )
    p_hve.add_argument(
        "--comm",
        default=",".join(DEFAULT_HIGH_VOLUME_COMM_PROFILES),
        help="Communication profile list for evidence rows",
    )
    p_hve.add_argument("--max-runs", type=int, default=None, help="Optional run cap for development checkpoints")
    p_hve.add_argument(
        "--max-runs-strategy",
        choices=MAX_RUNS_STRATEGIES,
        default="balanced",
        help="How to choose runs when --max-runs truncates the scenario/method/N matrix",
    )
    p_hve.add_argument("--resume", action="store_true", help="Resume from an existing evidence output directory")
    p_hve.add_argument(
        "--max-wall-time-s",
        type=float,
        default=None,
        help="Optional global wall-clock budget; writes partial progress when exceeded",
    )
    p_hve.add_argument(
        "--run-timeout-s",
        type=float,
        default=DEFAULT_HIGH_VOLUME_RUN_TIMEOUT_S,
        help="Hard per-episode wall-clock timeout for evidence jobs",
    )
    p_hve.add_argument(
        "--duration-s",
        type=float,
        default=DEFAULT_HIGH_VOLUME_DURATION_S,
        help="Duration override applied to copied evidence scenario files",
    )
    p_hve.add_argument("--save-traces", action="store_true", help="Save full traces for evidence runs")
    p_hve.add_argument(
        "--scale-spawn-profile",
        choices=SCALE_SPAWN_PROFILES,
        default=DEFAULT_HIGH_VOLUME_SCALE_SPAWN_PROFILE,
        help="Spawn adaptation for copied evidence scenario files",
    )
    p_hve.add_argument(
        "--planner-preset",
        default=DEFAULT_HIGH_VOLUME_PLANNER_PRESET,
        help="Planner config preset for evidence rows",
    )
    p_hve.add_argument(
        "--latency-budget-ms",
        type=float,
        default=DEFAULT_LATENCY_BUDGET_MS,
        help="Planner p95 latency budget used for the runtime/scale leaderboard axis",
    )
    p_hve.add_argument("--plan-only", action="store_true", help="Prepare the evidence matrix without running episodes")
    p_hve.add_argument("--json", action="store_true", help="Emit machine-readable evidence report")
    p_hve.add_argument("--require-pass", action="store_true", help="Fail if selected evidence runs are incomplete or timed out")
    p_hve.add_argument(
        "--require-complete",
        action="store_true",
        help="Fail unless the full requested evidence matrix completed without hard timeouts",
    )

    p_brv = sub.add_parser("baseline-review", help="Run optional longer stable-metadata review lanes for baseline candidates")
    p_brv.add_argument("--out-dir", required=True, help="Fresh output directory for review artifacts")
    p_brv.add_argument("--root", default=".", help="Repository root used for docs/tests coverage checks")
    p_brv.add_argument(
        "--methods",
        default=None,
        help="Comma-separated promotion candidates; defaults to cbf_qp,mpc_local,negotiation_yield",
    )
    p_brv.add_argument(
        "--lanes",
        default=None,
        help="Comma-separated review lane ids; defaults to all stable-metadata prep lanes",
    )
    p_brv.add_argument(
        "--duration-s",
        type=float,
        default=20.0,
        help="Scenario duration override for review lanes; use --full-duration for official generated durations",
    )
    p_brv.add_argument("--full-duration", action="store_true", help="Use official generated scenario durations")
    p_brv.add_argument("--max-runs", type=int, default=None, help="Run only the first K planned review episodes")
    p_brv.add_argument("--plan-only", action="store_true", help="Write/print the review plan without running episodes")
    p_brv.add_argument("--json", action="store_true", help="Emit machine-readable review report")
    p_brv.add_argument("--require-pass", action="store_true", help="Fail if any executed review check fails")

    p_bvm = sub.add_parser(
        "baseline-validation-matrix",
        help="Plan or run per-baseline validation lanes for key encounter families",
    )
    p_bvm.add_argument("--out-dir", required=True, help="Fresh output directory for validation artifacts")
    p_bvm.add_argument(
        "--methods",
        default=None,
        help="Comma-separated methods; defaults to serious built-in baselines",
    )
    p_bvm.add_argument(
        "--lanes",
        default=None,
        help="Comma-separated lane ids; defaults to head_on,crossing,urban_obstacle,communication_delay,high_n_dense_merge",
    )
    p_bvm.add_argument("--duration-s", type=float, default=None, help="Override all lane durations")
    p_bvm.add_argument("--max-runs", type=int, default=None, help="Run only the first K planned validation episodes")
    p_bvm.add_argument("--plan-only", action="store_true", help="Write/print the validation plan without running episodes")
    p_bvm.add_argument("--json", action="store_true", help="Emit machine-readable validation report")
    p_bvm.add_argument("--require-pass", action="store_true", help="Fail if hard gate checks fail")
    p_bvm.add_argument(
        "--require-behavior-pass",
        action="store_true",
        help="Fail if behavior-evidence checks fail; useful for candidate promotion, not lower-bound rows",
    )

    p_rl = sub.add_parser("rl-smoke", help="Run compact PettingZoo/Gymnasium wrapper smoke checks")
    p_rl.add_argument("--out-dir", required=True, help="Fresh output directory for RL smoke artifacts")
    p_rl.add_argument("--policy", choices=POLICY_NAMES, default="goal_direction", help="Built-in smoke policy")
    p_rl.add_argument("--policy-spec", default=None, help="Optional JSON/YAML external policy spec; overrides --policy")
    p_rl.add_argument(
        "--scenarios",
        default=None,
        help="Comma-separated generated smoke scenario ids; defaults to one 2D and one 3D case",
    )
    p_rl.add_argument("--n", type=int, default=4, help="Agent count for each RL smoke episode")
    p_rl.add_argument("--seeds", default="0", help="Seed list/range, e.g. 0:2")
    p_rl.add_argument("--comm", default="ideal_50hz", help="Communication profile for each RL smoke episode")
    p_rl.add_argument("--max-steps", type=int, default=None, help="Optional cap for each episode")
    p_rl.add_argument("--json", action="store_true", help="Emit machine-readable RL smoke report")
    p_rl.add_argument("--require-pass", action="store_true", help="Fail if any RL smoke check fails")

    p_rlcal = sub.add_parser("rl-calibration", help="Run compact 3D/degraded RL policy calibration lanes")
    p_rlcal.add_argument("--out-dir", required=True, help="Fresh output directory for RL calibration artifacts")
    p_rlcal.add_argument("--policy", choices=POLICY_NAMES, default="goal_direction", help="Built-in calibration policy")
    p_rlcal.add_argument("--policy-spec", default=None, help="Optional JSON/YAML external policy spec; overrides --policy")
    p_rlcal.add_argument("--n", type=int, default=4, help="Agent count for each RL calibration episode")
    p_rlcal.add_argument("--seeds", default="0", help="Seed list/range, e.g. 0:2")
    p_rlcal.add_argument("--max-steps", type=int, default=None, help="Optional cap for each episode")
    p_rlcal.add_argument("--json", action="store_true", help="Emit machine-readable RL calibration report")
    p_rlcal.add_argument("--require-pass", action="store_true", help="Fail if any RL calibration check fails")

    p_rlvm = sub.add_parser(
        "rl-validation-matrix",
        help="Run a learned policy or policy spec on the canonical validation-matrix lanes",
    )
    p_rlvm.add_argument("--out-dir", required=True, help="Fresh output directory for RL validation artifacts")
    p_rlvm.add_argument("--policy", choices=POLICY_NAMES, default="goal_direction", help="Built-in learned-policy wrapper policy")
    p_rlvm.add_argument("--policy-spec", default=None, help="Optional JSON/YAML external policy spec; overrides --policy")
    p_rlvm.add_argument(
        "--lanes",
        default=None,
        help="Comma-separated lane ids; defaults to head_on,crossing,urban_obstacle,communication_delay,high_n_dense_merge",
    )
    p_rlvm.add_argument("--seeds", default=None, help="Optional seed list/range override; defaults to each lane's canonical seed")
    p_rlvm.add_argument("--duration-s", type=float, default=None, help="Override all lane durations")
    p_rlvm.add_argument("--n", type=int, default=None, help="Override agent count for every lane")
    p_rlvm.add_argument("--max-steps", type=int, default=None, help="Optional cap for each episode")
    p_rlvm.add_argument("--plan-only", action="store_true", help="Write/print the validation plan without running episodes")
    p_rlvm.add_argument("--json", action="store_true", help="Emit machine-readable RL validation report")
    p_rlvm.add_argument("--require-pass", action="store_true", help="Fail if hard interface/rollout gates fail")
    p_rlvm.add_argument(
        "--require-behavior-pass",
        action="store_true",
        help="Fail if behavior-evidence checks fail; useful for policy promotion, not wrapper health",
    )

    p_bc = sub.add_parser(
        "train-learned-bc",
        help="Train a dependency-free behavior-cloned MLP policy from public RL validation-lane observations",
    )
    p_bc.add_argument("--out-dir", required=True, help="Output directory for model, policy spec, and training report")
    p_bc.add_argument(
        "--lanes",
        default=None,
        help="Comma-separated training lane ids; defaults to the canonical learned-policy validation lanes",
    )
    p_bc.add_argument("--seeds", default=None, help="Optional seed list/range for every lane; defaults to each lane seed")
    p_bc.add_argument("--max-steps", type=int, default=64, help="Teacher rollout steps per lane/seed")
    p_bc.add_argument("--hidden-dim", type=int, default=32, help="Random-feature MLP hidden dimension")
    p_bc.add_argument("--ridge", type=float, default=1e-4, help="Ridge regularization for the output layer fit")
    p_bc.add_argument(
        "--rollout-noise-std",
        type=float,
        default=0.03,
        help="Optional clipped Gaussian action noise during teacher rollouts",
    )
    p_bc.add_argument(
        "--eval-lanes",
        default=None,
        help="Comma-separated validation lane ids for the trained spec; defaults to the training lanes",
    )
    p_bc.add_argument("--eval-max-steps", type=int, default=12, help="Optional validation rollout step cap")
    p_bc.add_argument("--policy-name", default="bc_mlp_learned", help="Policy name written to policy_spec.json")
    p_bc.add_argument("--seed", type=int, default=29, help="Random-feature model seed")
    p_bc.add_argument("--overwrite", action="store_true", help="Overwrite existing model/spec/report files")
    p_bc.add_argument("--skip-validation", action="store_true", help="Skip post-training RL validation matrix")
    p_bc.add_argument("--json", action="store_true", help="Emit machine-readable training report")
    p_bc.add_argument("--require-pass", action="store_true", help="Fail if training or validation gates fail")

    p_bce = sub.add_parser(
        "learned-bc-evidence",
        help="Train, bundle, and compare a behavior-cloned learned policy against frozen learned fixtures",
    )
    p_bce.add_argument("--out-dir", required=True, help="Output directory for training, bundles, and leaderboard")
    p_bce.add_argument(
        "--lanes",
        default=None,
        help="Comma-separated training lane ids; defaults to the canonical learned-policy validation lanes",
    )
    p_bce.add_argument("--seeds", default=None, help="Optional training seed list/range for every lane")
    p_bce.add_argument("--max-steps", type=int, default=64, help="Teacher rollout steps per training lane/seed")
    p_bce.add_argument("--hidden-dim", type=int, default=32, help="Random-feature MLP hidden dimension")
    p_bce.add_argument("--ridge", type=float, default=1e-4, help="Ridge regularization for the output layer fit")
    p_bce.add_argument("--rollout-noise-std", type=float, default=0.03, help="Clipped Gaussian teacher-rollout action noise")
    p_bce.add_argument("--eval-lanes", default=None, help="Comma-separated validation lane ids for the trained spec")
    p_bce.add_argument("--eval-max-steps", type=int, default=12, help="Validation rollout step cap for the trained spec")
    p_bce.add_argument("--policy-name", default="bc_mlp_learned", help="Policy name written to the trained policy spec")
    p_bce.add_argument("--seed", type=int, default=29, help="Random-feature model seed")
    p_bce.add_argument("--suite", default="official_smoke_generated", help="Generated suite for learned-submission bundles")
    p_bce.add_argument("--n", type=int, default=4, help="Agent count for learned-submission bundle RL wrapper checks")
    p_bce.add_argument("--bundle-seeds", default=None, help="Seed list/range for learned-submission bundle checks")
    p_bce.add_argument("--bundle-max-steps", type=int, default=12, help="Step cap for bundle RL wrapper checks")
    p_bce.add_argument("--max-runs", type=int, default=1, help="Planner-sweep run cap for each learned bundle")
    p_bce.add_argument("--skip-fixtures", action="store_true", help="Only bundle the trained BC policy, not tiny/MLP fixtures")
    p_bce.add_argument("--overwrite", action="store_true", help="Overwrite known evidence artifacts in --out-dir")
    p_bce.add_argument("--json", action="store_true", help="Emit machine-readable evidence report")
    p_bce.add_argument("--require-pass", action="store_true", help="Fail if training, bundles, or leaderboard gates fail")

    p_rlc = sub.add_parser("rl-contract", help="Print the versioned RL action/observation/reward contract")
    p_rlc.add_argument("--top-k", type=int, default=8, help="Neighbor slots used to compute observation shape")
    p_rlc.add_argument("--out", default=None, help="Optional JSON output path")
    p_rlc.add_argument("--json", action="store_true", help="Emit machine-readable contract JSON")

    p_rlf = sub.add_parser("rl-freeze-check", help="Check stable-v1 RL interface freeze criteria")
    p_rlf.add_argument("--root", default=".", help="Repository root containing docs and examples")
    p_rlf.add_argument("--out", default=None, help="Optional JSON output path")
    p_rlf.add_argument("--json", action="store_true", help="Emit machine-readable freeze-check JSON")
    p_rlf.add_argument("--require-pass", action="store_true", help="Fail if any freeze criterion check fails")

    p_lssc = sub.add_parser(
        "learned-submission-schema-check",
        help="Check learned-submission schema readiness criteria",
    )
    p_lssc.add_argument("--root", default=".", help="Repository root containing docs, examples, and schemas")
    p_lssc.add_argument("--out", default=None, help="Optional JSON output path")
    p_lssc.add_argument("--json", action="store_true", help="Emit machine-readable schema-check JSON")
    p_lssc.add_argument("--require-pass", action="store_true", help="Fail if any schema readiness check fails")

    p_lsb = sub.add_parser(
        "learned-submission-bundle",
        help="Create RL and planner artifacts for a learned-policy submission",
    )
    p_lsb.add_argument("--out-dir", required=True, help="Fresh output directory for learned-policy submission artifacts")
    p_lsb.add_argument("--method", default="learned_tiny", help="Planner method to evaluate for official CSV artifacts")
    p_lsb.add_argument("--policy", choices=POLICY_NAMES, default="tiny_learned", help="RL policy to evaluate for wrapper artifacts")
    p_lsb.add_argument("--policy-spec", default=None, help="Optional JSON/YAML external policy spec for RL wrapper artifacts; overrides --policy")
    p_lsb.add_argument("--suite", default="official_smoke_generated", choices=list_official_suites(), help="Generated suite for planner CSV artifacts")
    p_lsb.add_argument("--root", default=".", help="Repository root used for freeze-check docs/examples")
    p_lsb.add_argument("--n", type=int, default=4, help="Agent count for RL wrapper smoke/calibration artifacts")
    p_lsb.add_argument("--seeds", default="0", help="Seed list/range for RL wrapper artifacts, e.g. 0:2")
    p_lsb.add_argument("--max-steps", type=int, default=None, help="Optional cap for each RL wrapper episode")
    p_lsb.add_argument("--max-runs", type=int, default=None, help="Optional cap for planner sweep episodes")
    p_lsb.add_argument("--save-trace", action="store_true", help="Save traces for planner sweep rows")
    p_lsb.add_argument(
        "--submission-manifest",
        default=None,
        help="Optional JSON disclosure overrides merged into learned_submission_manifest.json",
    )
    p_lsb.add_argument("--json", action="store_true", help="Emit machine-readable bundle report")
    p_lsb.add_argument("--require-pass", action="store_true", help="Fail if any bundle check fails")

    p_vlm = sub.add_parser("validate-learned-manifest", help="Validate a learned-policy submission manifest")
    p_vlm.add_argument("--manifest", required=True, help="Path to learned_submission_manifest.json or a draft manifest")
    p_vlm.add_argument("--bundle-root", default=None, help="Optional bundle root for artifact path/hash checks")
    p_vlm.add_argument(
        "--allow-undisclosed",
        action="store_true",
        help="Allow undisclosed training/inference fields while drafting the manifest",
    )
    p_vlm.add_argument("--json", action="store_true", help="Emit machine-readable manifest validation report")
    p_vlm.add_argument("--require-pass", action="store_true", help="Fail if any manifest validation check fails")

    p_vlb = sub.add_parser("validate-learned-bundle", help="Validate an existing learned-policy submission bundle")
    p_vlb.add_argument("--bundle", required=True, help="Path to a bundle directory or learned_submission_bundle.json")
    p_vlb.add_argument("--json", action="store_true", help="Emit machine-readable validation report")
    p_vlb.add_argument("--require-pass", action="store_true", help="Fail if any bundle validation check fails")

    p_rlb = sub.add_parser("review-learned-bundle", help="Summarize an existing learned-policy bundle for review")
    p_rlb.add_argument("--bundle", required=True, help="Path to a bundle directory or learned_submission_bundle.json")
    p_rlb.add_argument("--out", default=None, help="Optional JSON review output path")
    p_rlb.add_argument("--json", action="store_true", help="Emit machine-readable review report")
    p_rlb.add_argument("--require-pass", action="store_true", help="Fail if the bundle is not structurally reviewable")

    p_llb = sub.add_parser("learned-leaderboard", help="Build a development leaderboard from learned-policy bundles")
    p_llb.add_argument(
        "--bundle",
        action="append",
        required=True,
        help="Path to a bundle directory or learned_submission_bundle.json; repeat for multiple policies",
    )
    p_llb.add_argument("--out", required=True, help="Output JSON leaderboard path")
    p_llb.add_argument("--csv-out", default=None, help="Optional CSV output path; defaults beside --out")
    p_llb.add_argument("--json", action="store_true", help="Emit machine-readable leaderboard report")
    p_llb.add_argument("--require-pass", action="store_true", help="Fail if any bundle is not structurally reviewable")

    p_golden = sub.add_parser("golden-current-schema", help="Check or regenerate the current result-schema fixture")
    p_golden.add_argument("--golden-dir", default="golden/current_schema", help="Path to checked-in fixture")
    p_golden.add_argument("--candidate", default=None, help="Compare an existing candidate directory instead of running")
    p_golden.add_argument("--update", action="store_true", help="Overwrite the fixture with a fresh deterministic run")
    p_golden.add_argument("--json", action="store_true", help="Emit machine-readable comparison report")

    p_hc = sub.add_parser("mine-hard-cases", help="Collect trace artifacts for worst episodes")
    p_hc.add_argument("--results", required=True, help="Path to runs/<id>/results.csv")
    p_hc.add_argument("--top-k", type=int, default=20, help="Number of episodes to collect")

    p_lm = sub.add_parser("list-methods", help="List available planner methods")
    p_lm.add_argument("--json", action="store_true", help="Emit planner metadata as JSON")
    p_lm.add_argument("--include-aliases", action="store_true", help="Include compatibility aliases")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.cmd == "run":
        row = _run_once(args)
        print(
            f"done: scenario={row['scenario']} method={row['method']} n={row['N']} seed={row['seed']} "
            f"collisions={row['collisions']} completion={row['completion_rate']:.3f}"
        )
        return
    if args.cmd == "sweep":
        _run_sweep(args)
        print("done: sweep complete")
        return
    if args.cmd == "episode-report":
        out = render_episode_report(
            args.trace,
            args.out,
            max_frames=args.max_frames,
            plotly_source=args.plotly_source,
        )
        print(f"done: episode report saved to {out}")
        return
    if args.cmd == "foxglove-export":
        try:
            out = export_foxglove_mcap(
                args.trace,
                args.out,
                trail_frames=args.trail_frames,
                max_sensing_links=args.max_sensing_links,
                compression=args.compression,
            )
        except RuntimeError as exc:
            raise SystemExit(str(exc))
        print(f"done: Foxglove MCAP saved to {out}")
        return
    if args.cmd == "foxglove-comparison-export":
        try:
            labels, traces = _parse_label_trace_args(args.trace)
            out = export_foxglove_comparison_mcap(
                traces,
                args.out,
                labels=labels,
                topic_root=args.topic_root,
                trail_frames=args.trail_frames,
                max_sensing_links=args.max_sensing_links,
                compression=args.compression,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(str(exc))
        print(f"done: Foxglove comparison MCAP saved to {out}")
        return
    if args.cmd == "generate-dataset":
        defaults = load_defaults()
        comm_default = defaults.get("comm", {}).get("profile", "ideal_50hz")
        comm_profiles = expand_list(args.comm) if args.comm else [comm_default]
        scenarios = expand_scenarios(args.scenario)
        n_spec = args.n if args.n is not None else args.N
        if n_spec is None:
            raise ValueError("generate-dataset requires --n (or deprecated --N)")
        n_agents = _parse_int_list(n_spec)
        seeds = _parse_int_list(args.seeds)
        T = int(args.horizon_steps) if args.horizon_steps is not None else int(args.T)
        shards = generate_dataset(
            scenarios=scenarios,
            method=args.method,
            n_agents_list=n_agents,
            seeds=seeds,
            T=T,
            dt_plan_s=float(args.dt_plan_s),
            out_dir=args.out_dir,
            comm_profiles=comm_profiles,
            shard_size=int(args.shard_size),
            goal_dist_cap=float(args.goal_dist_cap),
            quality_filter=str(args.quality_filter),
            filter_min_sep_m=float(args.filter_min_sep_m),
        )
        print(f"done: dataset generated with {len(shards)} shard(s) in {args.out_dir}")
        return
    if args.cmd == "sanity-check-dataset":
        stats = sanity_check_shard(args.shard, out_plot=args.out_plot)
        print("dataset sanity check:")
        for k in sorted(stats):
            print(f"  {k}: {stats[k]}")
        return
    if args.cmd == "canonical-sweep":
        _run_canonical_sweep(args)
        if not args.no_run:
            print("done: canonical sweep complete")
        return
    if args.cmd == "materialize-suite":
        generated = materialize_official_suite(
            args.suite,
            args.out_dir,
            overwrite=bool(args.overwrite),
            stretch=bool(args.stretch),
        )
        manifest = generated["manifest"]
        dims = sorted({s["dimension"] for s in manifest["scenarios"]})
        print(
            f"done: materialized suite={args.suite} scenarios={len(generated['scenario_paths'])} "
            f"dimensions={','.join(dims)} manifest={generated['manifest_path']}"
        )
        if args.print_plan:
            print(f"  methods: {','.join(manifest['default_methods'])}")
            print(f"  N: {','.join(str(x) for x in manifest['n_agents'])}")
            print(f"  seeds: {','.join(str(x) for x in manifest['seeds'])}")
            print(f"  comm: {','.join(manifest['comm_profiles'])}")
        return
    if args.cmd == "validate-scenarios":
        _validate_scenarios(args)
        return
    if args.cmd == "list-suites":
        _list_suites(args)
        return
    if args.cmd == "check-acceptance":
        _check_acceptance(args)
        return
    if args.cmd == "baseline-report":
        _baseline_report(args)
        return
    if args.cmd == "baseline-leaderboard":
        _baseline_leaderboard(args)
        return
    if args.cmd == "baseline-audit":
        _baseline_audit(args)
        return
    if args.cmd == "baseline-smoke":
        _baseline_smoke(args)
        return
    if args.cmd == "baseline-promotion":
        _baseline_promotion(args)
        return
    if args.cmd == "baseline-evidence":
        _baseline_evidence(args)
        return
    if args.cmd == "validate-external-reference":
        _validate_external_reference(args)
        return
    if args.cmd == "external-reference-bundle":
        _external_reference_bundle(args)
        return
    if args.cmd == "advanced-baseline-comparison":
        _advanced_baseline_comparison(args)
        return
    if args.cmd == "optimizer-suite-review":
        _optimizer_suite_review(args)
        return
    if args.cmd == "scale-benchmark":
        _scale_benchmark(args)
        return
    if args.cmd == "high-volume-leaderboard":
        _high_volume_leaderboard(args)
        return
    if args.cmd == "high-volume-evidence":
        _high_volume_evidence(args)
        return
    if args.cmd == "baseline-review":
        _baseline_review(args)
        return
    if args.cmd == "baseline-validation-matrix":
        _baseline_validation_matrix(args)
        return
    if args.cmd == "rl-smoke":
        _rl_smoke(args)
        return
    if args.cmd == "rl-calibration":
        _rl_calibration(args)
        return
    if args.cmd == "rl-validation-matrix":
        _rl_validation_matrix(args)
        return
    if args.cmd == "train-learned-bc":
        _train_learned_bc(args)
        return
    if args.cmd == "learned-bc-evidence":
        _learned_bc_evidence(args)
        return
    if args.cmd == "rl-contract":
        _rl_contract(args)
        return
    if args.cmd == "rl-freeze-check":
        _rl_freeze_check(args)
        return
    if args.cmd == "learned-submission-schema-check":
        _learned_submission_schema_check(args)
        return
    if args.cmd == "learned-submission-bundle":
        _learned_submission_bundle(args)
        return
    if args.cmd == "validate-learned-manifest":
        _validate_learned_manifest(args)
        return
    if args.cmd == "validate-learned-bundle":
        _validate_learned_bundle(args)
        return
    if args.cmd == "review-learned-bundle":
        _review_learned_bundle(args)
        return
    if args.cmd == "learned-leaderboard":
        _learned_leaderboard(args)
        return
    if args.cmd == "golden-current-schema":
        _golden_current_schema(args)
        return
    if args.cmd == "mine-hard-cases":
        out = mine_worst_cases(results_csv=args.results, top_k=args.top_k)
        print(
            f"done: mined {out['selected']} episodes, copied artifacts for {out['copied']} "
            f"into {out['worst_dir']}"
        )
        return
    if args.cmd == "list-methods":
        if args.json:
            print(json.dumps(planner_metadata(include_aliases=bool(args.include_aliases)), indent=2, sort_keys=True))
            return
        for m in list_methods(include_aliases=bool(args.include_aliases)):
            print(m)
        return


if __name__ == "__main__":
    main()
