from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from microbench.acceptance import check_acceptance
from microbench.metrics import append_result, write_summary
from microbench.runner import run_episode
from microbench.scenarios import list_official_suites, materialize_official_suite, suite_defaults
from microbench.tools.baseline_report import build_baseline_report
from microbench.types import RunSpec


BASELINE_LEADERBOARD_SCHEMA_VERSION = "0.1"

SERIOUS_BASELINE_METHODS = (
    "baseline_goal",
    "orca_heuristic",
    "orca_with_staleness",
    "velocity_obstacle",
    "reciprocal_velocity_obstacle",
    "cbf_qp",
    "mpc_local",
    "priority_yield",
    "negotiation_yield",
    "learned_tiny",
)


def _as_list(values: tuple[str, ...] | list[str] | None, default: tuple[str, ...]) -> list[str]:
    return [str(v).strip() for v in (values if values is not None else default) if str(v).strip()]


def _rel(path: str | Path, root: Path) -> str:
    p = Path(path)
    try:
        return str(p.relative_to(root))
    except ValueError:
        return str(p)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return round(sum(values) / len(values), 6)


def _aggregate_method_scores(suite_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_method: dict[str, list[float]] = {}
    episodes: dict[str, int] = {}
    suites: dict[str, set[str]] = {}
    for suite_report in suite_reports:
        suite_id = str(suite_report["suite"])
        report = suite_report["report"]
        for entry in report.get("method_summaries", []):
            method = str(entry.get("method"))
            score = entry.get("score_v0_mean")
            if isinstance(score, (int, float)) and not isinstance(score, bool):
                by_method.setdefault(method, []).append(float(score))
            episodes[method] = episodes.get(method, 0) + int(entry.get("episodes") or 0)
            suites.setdefault(method, set()).add(suite_id)

    ranking = []
    for method, scores in sorted(by_method.items()):
        ranking.append(
            {
                "method": method,
                "score_v0_mean": _mean(scores),
                "score_v0_best_suite": round(min(scores), 6) if scores else None,
                "score_v0_worst_suite": round(max(scores), 6) if scores else None,
                "suite_count": len(suites.get(method, set())),
                "episodes": int(episodes.get(method, 0)),
            }
        )
    ranking.sort(key=lambda row: (float("inf") if row["score_v0_mean"] is None else float(row["score_v0_mean"]), row["method"]))
    for rank, row in enumerate(ranking, start=1):
        row["rank"] = rank
    return ranking


def _run_suite(
    *,
    out_dir: Path,
    suite: str,
    methods: list[str],
    n_agents: list[int] | None,
    seeds: list[int] | None,
    comm_profiles: list[str] | None,
    max_runs: int | None,
    stretch: bool,
) -> dict[str, Any]:
    suite_dir = out_dir / suite
    run_dir = suite_dir / "runs"
    generated_dir = suite_dir / "_generated_scenarios" / suite
    generated = materialize_official_suite(suite, generated_dir, overwrite=True, stretch=stretch)
    defaults = suite_defaults(suite, stretch=stretch)
    scenario_paths = [Path(path) for path in generated["scenario_paths"]]
    n_values = list(n_agents if n_agents is not None else [int(x) for x in defaults["n_agents"]])
    seed_values = list(seeds if seeds is not None else [int(x) for x in defaults["seeds"]])
    comm_values = list(comm_profiles if comm_profiles is not None else [str(x) for x in defaults["comm_profiles"]])

    specs: list[RunSpec] = []
    for scenario_path in scenario_paths:
        for method in methods:
            for comm in comm_values:
                for n in n_values:
                    for seed in seed_values:
                        specs.append(
                            RunSpec(
                                scenario_path=str(scenario_path),
                                method=str(method),
                                n_agents=int(n),
                                seed=int(seed),
                                comm_profile=str(comm),
                                out_dir=str(run_dir),
                                save_trace=False,
                            )
                        )

    planned_run_count = len(specs)
    if max_runs is not None:
        specs = specs[: max(0, int(max_runs))]

    rows: list[dict[str, Any]] = []
    for spec in specs:
        row = run_episode(spec)
        append_result(run_dir, row)
        rows.append(row)

    summary_csv = write_summary(run_dir)
    results_csv = run_dir / "results.csv"
    report = build_baseline_report(
        summary_csv=summary_csv,
        results_csv=results_csv if results_csv.exists() else None,
        suite=suite,
        generated_by="python -m microbench.cli baseline-leaderboard",
    )
    report_path = suite_dir / "baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    acceptance = check_acceptance(
        summary_csv=summary_csv,
        results_csv=results_csv if results_csv.exists() else None,
        suite_manifest=generated["manifest_path"],
        methods=methods,
    )
    acceptance_path = suite_dir / "acceptance.json"
    acceptance_path.write_text(json.dumps(acceptance, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    return {
        "suite": suite,
        "ok": bool(acceptance.get("ok", False)),
        "planned_run_count": planned_run_count,
        "run_count": len(rows),
        "methods": methods,
        "scenario_count": len(scenario_paths),
        "n_agents": n_values,
        "seeds": seed_values,
        "comm_profiles": comm_values,
        "report": report,
        "report_path": str(report_path),
        "acceptance": acceptance,
        "acceptance_path": str(acceptance_path),
        "results_csv": str(results_csv),
        "summary_csv": str(summary_csv),
        "suite_manifest": str(generated["manifest_path"]),
        "truncated_by_max_runs": max_runs is not None and len(rows) < planned_run_count,
    }


def run_baseline_leaderboard(
    *,
    out_dir: str | Path,
    suites: tuple[str, ...] | list[str] | None = None,
    methods: tuple[str, ...] | list[str] | None = None,
    n_agents: tuple[int, ...] | list[int] | None = None,
    seeds: tuple[int, ...] | list[int] | None = None,
    comm_profiles: tuple[str, ...] | list[str] | None = None,
    max_runs: int | None = None,
    stretch: bool = False,
) -> dict[str, Any]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suite_values = _as_list(suites, tuple(list_official_suites()))
    method_values = _as_list(methods, SERIOUS_BASELINE_METHODS)
    official = set(list_official_suites())
    unknown = sorted(set(suite_values) - official)
    if unknown:
        raise ValueError(f"unknown official suite(s): {','.join(unknown)}")

    suite_reports = [
        _run_suite(
            out_dir=out,
            suite=suite,
            methods=method_values,
            n_agents=list(n_agents) if n_agents is not None else None,
            seeds=list(seeds) if seeds is not None else None,
            comm_profiles=list(comm_profiles) if comm_profiles is not None else None,
            max_runs=max_runs,
            stretch=bool(stretch),
        )
        for suite in suite_values
    ]

    aggregate_ranking = _aggregate_method_scores(suite_reports)
    report = {
        "schema_version": BASELINE_LEADERBOARD_SCHEMA_VERSION,
        "ok": all(entry["ok"] for entry in suite_reports),
        "out_dir": str(out),
        "suites": [
            {
                "suite": entry["suite"],
                "ok": entry["ok"],
                "planned_run_count": entry["planned_run_count"],
                "run_count": entry["run_count"],
                "scenario_count": entry["scenario_count"],
                "report_path": _rel(entry["report_path"], out),
                "acceptance_path": _rel(entry["acceptance_path"], out),
                "results_csv": _rel(entry["results_csv"], out),
                "summary_csv": _rel(entry["summary_csv"], out),
                "suite_manifest": _rel(entry["suite_manifest"], out),
                "truncated_by_max_runs": entry["truncated_by_max_runs"],
            }
            for entry in suite_reports
        ],
        "methods": method_values,
        "aggregate_ranking": aggregate_ranking,
        "score_note": "score_v0 follows docs/LEADERBOARD.md; use per-suite rankings for official comparisons.",
    }
    leaderboard_path = out / "baseline_leaderboard.json"
    report["leaderboard_path"] = str(leaderboard_path)
    leaderboard_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report
