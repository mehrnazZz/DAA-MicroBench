from __future__ import annotations

import csv
import datetime as dt
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
