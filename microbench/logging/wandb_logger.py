from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
import math
from typing import Any

from microbench.metrics.io import RESULT_SCHEMA_FILENAME


def _warn(msg: str) -> None:
    print(f"[wandb] warning: {msg}")


def _get(args: Any, key: str, default: Any = None) -> Any:
    if isinstance(args, dict):
        return args.get(key, default)
    return getattr(args, key, default)


def _to_float(v: Any) -> float | None:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(out):
        return None
    return out


def _mean(vals: list[float]) -> float:
    if not vals:
        return float("nan")
    return float(sum(vals) / len(vals))


def _read_csv(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.exists():
        return []
    with p.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _read_json(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def compute_overall_metrics(summary_csv_path: str | Path) -> dict[str, float]:
    rows = _read_csv(summary_csv_path)
    if not rows:
        return {}

    coll = [_to_float(r.get("collision_rate")) for r in rows]
    comp = [_to_float(r.get("completion_rate_mean")) for r in rows]
    sep = [_to_float(r.get("min_sep_p05_mean")) for r in rows]
    ttg = [_to_float(r.get("mean_time_to_goal_mean")) for r in rows]
    dead = [_to_float(r.get("deadlock_time_pct_mean")) for r in rows]
    pms = [_to_float(r.get("planner_ms_mean")) for r in rows]

    coll = [x for x in coll if x is not None]
    comp = [x for x in comp if x is not None]
    sep = [x for x in sep if x is not None]
    ttg = [x for x in ttg if x is not None]
    dead = [x for x in dead if x is not None]
    pms = [x for x in pms if x is not None]

    return {
        "overall_collision_rate_mean": _mean(coll),
        "overall_completion_rate_mean": _mean(comp),
        "overall_min_sep_p05_mean": _mean(sep),
        "overall_time_to_goal_mean": _mean(ttg),
        "overall_deadlock_pct_mean": _mean(dead),
        "overall_planner_ms_mean": _mean(pms),
    }


def init_run(args: Any, run_config: dict[str, Any]) -> Any | None:
    if not bool(_get(args, "wandb", False)):
        return None

    mode = _get(args, "wandb_mode", None)
    if mode is None:
        mode = "online"
    if mode == "disabled":
        return None

    try:
        import wandb  # type: ignore
    except Exception as exc:
        _warn(f"wandb import failed ({exc}); continuing without W&B logging")
        return None

    method = str(run_config.get("method_name", "method"))
    suite = str(run_config.get("suite", "sweep"))
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M")
    name = _get(args, "wandb_name", None) or f"{method}_{suite}_{timestamp}"
    group = _get(args, "wandb_group", None) or method
    tags_raw = _get(args, "wandb_tags", None)
    tags = [t.strip() for t in str(tags_raw).split(",") if t.strip()] if tags_raw else []
    comm_profiles = run_config.get("comm_profiles", [])
    n_list = run_config.get("N_list", [])
    git_commit = run_config.get("git_commit")
    auto_tags = [
        f"suite:{suite}",
        f"planner:{method}",
    ]
    if comm_profiles:
        auto_tags.append(f"comm:{'+'.join(str(x) for x in comm_profiles)}")
    if n_list:
        auto_tags.append(f"N:{'-'.join(str(x) for x in n_list)}")
    if git_commit:
        auto_tags.append(f"commit:{str(git_commit)[:8]}")
    tags = auto_tags + tags

    try:
        run = wandb.init(
            project=_get(args, "wandb_project", "daa-microbench"),
            entity=_get(args, "wandb_entity", None),
            group=group,
            name=name,
            tags=tags,
            mode=mode,
            config=run_config,
        )
        return run
    except Exception as exc:
        _warn(f"wandb.init failed ({exc}); continuing without W&B logging")
        return None


def _rows_to_table(wandb: Any, rows: list[dict[str, Any]], table_name: str) -> tuple[str, Any] | None:
    if not rows:
        return None
    cols = list(rows[0].keys())
    table = wandb.Table(columns=cols)
    for r in rows:
        table.add_data(*[r.get(c) for c in cols])
    return table_name, table


def _log_path_as_artifact(run: Any, wandb: Any, name: str, type_name: str, path: Path) -> None:
    if not path.exists():
        return
    art = wandb.Artifact(name=name, type=type_name)
    if path.is_dir():
        art.add_dir(str(path))
    else:
        art.add_file(str(path))
    run.log_artifact(art)


def _resolve_artifact_path(path_value: Any, *, root: Path) -> Path | None:
    if path_value in (None, ""):
        return None
    path = Path(str(path_value))
    if not path.is_absolute():
        path = root / path
    return path


def _add_artifact_file(artifact: Any, path: Path | None, *, root: Path) -> None:
    if path is None or not path.exists() or not path.is_file():
        return
    try:
        name = str(path.relative_to(root))
    except ValueError:
        name = path.name
    artifact.add_file(str(path), name=name)


def _sanitize_artifact_name(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.-")
    cleaned = "".join(ch if ch in allowed else "_" for ch in value)
    return cleaned.strip("._-") or "leaderboard"


def _leaderboard_root(report: dict[str, Any]) -> Path:
    out_dir = report.get("out_dir") or "."
    return Path(str(out_dir))


def build_leaderboard_wandb_payload(report: dict[str, Any]) -> dict[str, Any]:
    """Build table rows for publishing a baseline leaderboard to W&B.

    The local JSON/CSV files remain the canonical benchmark artifact. This
    payload is a dashboard-friendly projection of those same files.
    """

    root = _leaderboard_root(report)
    aggregate_rows = [dict(row) for row in report.get("aggregate_ranking", []) if isinstance(row, dict)]
    suite_rows: list[dict[str, Any]] = []
    suite_method_rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    artifact_paths: list[Path] = []

    leaderboard_path = _resolve_artifact_path(report.get("leaderboard_path"), root=root)
    if leaderboard_path is not None:
        artifact_paths.append(leaderboard_path)

    for suite in report.get("suites", []):
        if not isinstance(suite, dict):
            continue
        suite_id = str(suite.get("suite", ""))
        suite_rows.append(
            {
                "suite": suite_id,
                "ok": bool(suite.get("ok", False)),
                "complete": bool(suite.get("complete", False)),
                "selected_complete": bool(suite.get("selected_complete", False)),
                "planned_run_count": suite.get("planned_run_count"),
                "selected_run_count": suite.get("selected_run_count"),
                "selected_completed_count": suite.get("selected_completed_count"),
                "run_count": suite.get("run_count"),
                "timeout_run_count": suite.get("timeout_run_count"),
                "scenario_count": suite.get("scenario_count"),
                "truncated_by_max_runs": bool(suite.get("truncated_by_max_runs", False)),
                "stopped_by_wall_time": bool(suite.get("stopped_by_wall_time", False)),
                "report_path": suite.get("report_path"),
                "summary_csv": suite.get("summary_csv"),
                "results_csv": suite.get("results_csv"),
            }
        )

        for key in (
            "report_path",
            "acceptance_path",
            "progress_path",
            "results_csv",
            "summary_csv",
            "suite_manifest",
        ):
            resolved = _resolve_artifact_path(suite.get(key), root=root)
            if resolved is not None:
                artifact_paths.append(resolved)
        summary_csv = _resolve_artifact_path(suite.get("summary_csv"), root=root)
        if summary_csv is not None:
            artifact_paths.append(summary_csv.with_name(RESULT_SCHEMA_FILENAME))

        suite_report_path = _resolve_artifact_path(suite.get("report_path"), root=root)
        suite_report = _read_json(suite_report_path) if suite_report_path is not None else {}
        for method_row in suite_report.get("method_summaries", []):
            if not isinstance(method_row, dict):
                continue
            suite_method_rows.append({"suite": suite_id, **method_row})
        for row in suite_report.get("rows", []):
            if not isinstance(row, dict):
                continue
            component_rows.append({"suite": suite_id, **row})

    return {
        "aggregate_rows": aggregate_rows,
        "suite_rows": suite_rows,
        "suite_method_rows": suite_method_rows,
        "component_rows": component_rows,
        "artifact_paths": artifact_paths,
    }


def log_baseline_leaderboard(
    run: Any | None,
    report: dict[str, Any],
    *,
    upload_results: bool = True,
) -> None:
    if run is None:
        return

    try:
        import wandb  # type: ignore
    except Exception:
        _warn("wandb unavailable during leaderboard logging; skipping")
        return

    try:
        payload = build_leaderboard_wandb_payload(report)
        aggregate_rows = payload["aggregate_rows"]
        best = aggregate_rows[0] if aggregate_rows else {}
        run.log(
            {
                "leaderboard_ok": int(bool(report.get("ok", False))),
                "leaderboard_complete": int(bool(report.get("complete", False))),
                "leaderboard_selected_complete": int(bool(report.get("selected_complete", False))),
                "leaderboard_timeout_run_count": int(report.get("timeout_run_count") or 0),
                "leaderboard_suite_count": len(report.get("suites", [])),
                "leaderboard_method_count": len(report.get("methods", [])),
                "leaderboard_best_score_v0": _to_float(best.get("score_v0_mean")),
            }
        )

        for table_name, rows in (
            ("leaderboard_aggregate", aggregate_rows),
            ("leaderboard_suites", payload["suite_rows"]),
            ("leaderboard_suite_methods", payload["suite_method_rows"]),
            ("leaderboard_components", payload["component_rows"]),
        ):
            table = _rows_to_table(wandb, rows, table_name)
            if table is not None:
                run.log({table[0]: table[1]})

        if bool(upload_results):
            root = _leaderboard_root(report)
            run_id = str(run.config.get("run_id", Path(str(report.get("out_dir", "leaderboard"))).name))
            art = wandb.Artifact(
                name=_sanitize_artifact_name(f"daa_microbench_leaderboard_{run_id}"),
                type="daa-microbench-leaderboard",
            )
            seen: set[str] = set()
            for path in payload["artifact_paths"]:
                key = str(path)
                if key in seen:
                    continue
                seen.add(key)
                _add_artifact_file(art, path, root=root)
            run.log_artifact(art)
    except Exception as exc:
        _warn(f"leaderboard logging failed ({exc}); continuing")


def log_summary(
    run: Any | None,
    summary_csv_path: str | Path,
    results_csv_path: str | Path,
    extra_artifacts_paths: dict[str, Any] | None = None,
    metrics_dict: dict[str, float] | None = None,
) -> None:
    if run is None:
        return

    extra = extra_artifacts_paths or {}
    try:
        import wandb  # type: ignore
    except Exception:
        _warn("wandb unavailable during summary logging; skipping")
        return

    try:
        summary_rows = _read_csv(summary_csv_path)
        metrics = metrics_dict or compute_overall_metrics(summary_csv_path)
        if metrics:
            run.log(metrics)

        leaderboard = _rows_to_table(wandb, summary_rows, "leaderboard")
        if leaderboard is not None:
            run.log({leaderboard[0]: leaderboard[1]})

        worst_index = Path(extra.get("worst_cases_index", ""))
        if worst_index.exists():
            top_fail_rows = _read_csv(worst_index)
            top_fail_rows = top_fail_rows[:20]
            top_failures = _rows_to_table(wandb, top_fail_rows, "top_failures")
            if top_failures is not None:
                run.log({top_failures[0]: top_failures[1]})

        if bool(extra.get("upload_results", True)):
            method = str(run.config.get("method_name", "multi"))
            run_id = str(run.config.get("run_id", run.id))
            art = wandb.Artifact(name=f"microbench_results_{method}_{run_id}", type="microbench-results")
            sp = Path(summary_csv_path)
            rp = Path(results_csv_path)
            schema_path = sp.with_name(RESULT_SCHEMA_FILENAME)
            if sp.exists():
                art.add_file(str(sp))
            if rp.exists():
                art.add_file(str(rp))
            if schema_path.exists():
                art.add_file(str(schema_path))
            run.log_artifact(art)

        if bool(extra.get("upload_traces", False)):
            traces_dir = Path(extra.get("traces_dir", ""))
            if traces_dir.exists():
                method = str(run.config.get("method_name", "multi"))
                run_id = str(run.config.get("run_id", run.id))
                _log_path_as_artifact(
                    run,
                    wandb,
                    name=f"microbench_traces_{method}_{run_id}",
                    type_name="microbench-traces",
                    path=traces_dir,
                )

        if bool(extra.get("upload_replays", False)):
            replays_dir = Path(extra.get("replays_dir", ""))
            if replays_dir.exists():
                method = str(run.config.get("method_name", "multi"))
                run_id = str(run.config.get("run_id", run.id))
                _log_path_as_artifact(
                    run,
                    wandb,
                    name=f"microbench_replays_{method}_{run_id}",
                    type_name="microbench-replays",
                    path=replays_dir,
                )
    except Exception as exc:
        _warn(f"summary logging failed ({exc}); continuing")


def finish(run: Any | None) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception as exc:
        _warn(f"run.finish failed ({exc})")
