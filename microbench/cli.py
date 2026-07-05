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
from microbench.config import load_defaults
from microbench.planners import list_methods, planner_metadata
from microbench.types import RunSpec
from microbench.runner import run_episode
from microbench.metrics import append_result, write_summary
from microbench.replay import render_interactive_trace, render_trace
from microbench.dataset import generate_dataset, expand_scenarios, expand_list, sanity_check_shard
from microbench.logging import wandb_logger
from microbench.tools import (
    build_current_schema_candidate,
    compare_current_schema_golden,
    mine_worst_cases,
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


def _expand_scenarios(spec: str) -> list[str]:
    scenarios: list[str] = []
    for token in _parse_str_list(spec):
        matches = sorted(glob.glob(token))
        if matches:
            scenarios.extend(matches)
        else:
            scenarios.append(token)
    return scenarios


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
    save_trace = bool(defaults.get("logging", {}).get("save_trace", False))
    spec = RunSpec(
        scenario_path=args.scenario,
        method=args.method,
        n_agents=int(args.n),
        seed=int(args.seed),
        comm_profile=comm,
        out_dir=out_dir,
        save_trace=save_trace,
        agent_methods=_parse_str_list(args.agent_methods) if args.agent_methods else None,
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
        scenarios = CANONICAL_SCENARIOS
        methods = _parse_str_list(args.methods or "")
        if not methods:
            raise ValueError("canonical-sweep --suite primary requires --methods")
        n_agents = [10, 20, 50] + ([100] if stretch else [])
        seeds = list(range(0, 100 if stretch else 50))
        comm_profiles = ["ideal_50hz", "realistic_v2v_50hz", "degraded_20hz"]
    elif suite == "baseline_sanity":
        scenarios = CANONICAL_SCENARIOS
        methods = _parse_str_list(args.methods) if args.methods else ["baseline_goal", "orca_heuristic"]
        n_agents = [10, 20] + ([100] if stretch else [])
        seeds = list(range(0, 100 if stretch else 20))
        comm_profiles = ["ideal_50hz", "realistic_v2v_50hz"]
    elif suite == "three_d":
        scenarios = CANONICAL_3D_SCENARIOS
        methods = _parse_str_list(args.methods) if args.methods else ["orca_heuristic"]
        n_agents = [6, 10] + ([20] if stretch else [])
        seeds = list(range(0, 20 if stretch else 10))
        comm_profiles = ["ideal_50hz"]
    elif suite == "perception_stress":
        scenarios = CANONICAL_PERCEPTION_SCENARIOS
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
        scenario_paths.extend(sorted(glob.glob("config/scenarios/*.yaml")))
    if args.suite_manifest:
        manifest_paths.extend(_expand_scenarios(args.suite_manifest))
    if args.generated_suite:
        generated_suites.extend(_parse_str_list(args.generated_suite))
    if args.all_generated_suites:
        generated_suites.extend(list_official_suites())

    if not scenario_paths and not manifest_paths and not generated_suites:
        scenario_paths.extend(sorted(glob.glob("config/scenarios/*.yaml")))

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
    _add_wandb_flags(p_sweep)

    p_replay = sub.add_parser("replay", help="Render a saved episode or collision trace")
    p_replay.add_argument("--trace", required=True, help="Path to trace_episode.jsonl or trace_collision_*.jsonl")
    p_replay.add_argument("--out", required=True, help="Output media path (.gif/.mp4)")
    p_replay.add_argument("--fps", type=int, default=25)
    p_replay.add_argument("--tail", type=int, default=25, help="Trail length in frames")
    p_replay.add_argument("--max-sensed", type=int, default=8, help="Max sensed-neighbor links per focus agent")
    p_replay.add_argument(
        "--show-sensed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show sensed-neighbor links and msgAge labels",
    )

    p_replay_html = sub.add_parser("replay-interactive", help="Render an interactive HTML replay")
    p_replay_html.add_argument("--trace", required=True, help="Path to trace_episode.jsonl or trace_collision_*.jsonl")
    p_replay_html.add_argument("--out", required=True, help="Output HTML path")
    p_replay_html.add_argument("--tail", type=int, default=40, help="Trail length in frames")
    p_replay_html.add_argument("--max-sensed", type=int, default=8, help="Max sensed-neighbor links per focus agent")
    p_replay_html.add_argument(
        "--show-sensed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show sensed-neighbor links",
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
    if args.cmd == "replay":
        out = render_trace(
            args.trace,
            args.out,
            fps=args.fps,
            tail=args.tail,
            show_sensed=args.show_sensed,
            max_sensed_per_agent=args.max_sensed,
        )
        print(f"done: replay saved to {out}")
        return
    if args.cmd == "replay-interactive":
        out = render_interactive_trace(
            args.trace,
            args.out,
            tail=args.tail,
            show_sensed=args.show_sensed,
            max_sensed_per_agent=args.max_sensed,
        )
        print(f"done: interactive replay saved to {out}")
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
