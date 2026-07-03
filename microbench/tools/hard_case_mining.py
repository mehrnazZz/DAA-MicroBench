from __future__ import annotations

import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any

from microbench.metrics import episode_dir_name


def _to_float(v: Any, default: float) -> float:
    try:
        out = float(v)
    except (TypeError, ValueError):
        return default
    if math.isnan(out):
        return default
    return out


def _to_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _row_str(row: dict[str, Any], keys: list[str], default: str = "") -> str:
    for k in keys:
        if k in row and row.get(k) not in (None, ""):
            return str(row.get(k))
    return default


def _row_int(row: dict[str, Any], keys: list[str], default: int = 0) -> int:
    for k in keys:
        if k in row and row.get(k) not in (None, ""):
            return _to_int(row.get(k), default)
    return default


def _episode_dirs_for_row(run_dir: Path, row: dict[str, Any]) -> list[Path]:
    scenario = _row_str(row, ["scenario", "scenario_id"], "")
    method = _row_str(row, ["method"], "")
    n_agents = _row_int(row, ["N", "n_agents"], 0)
    seed = _row_int(row, ["seed"], 0)
    comm_profile = _row_str(row, ["comm_profile", "comm"], "")

    episodes_root = run_dir / "episodes"
    name_comm = episode_dir_name(
        scenario=scenario,
        method=method,
        n_agents=n_agents,
        seed=seed,
        comm_profile=comm_profile or None,
    )
    name_legacy = episode_dir_name(
        scenario=scenario,
        method=method,
        n_agents=n_agents,
        seed=seed,
        comm_profile=None,
    )
    return [episodes_root / name_comm, episodes_root / name_legacy]


def _severity_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    collisions = _to_float(row.get("collisions"), 0.0)
    near_misses = _to_float(row.get("near_misses"), 0.0)
    min_sep_min = _to_float(row.get("min_sep_min_m"), float("inf"))
    min_sep_p05 = _to_float(row.get("min_sep_p05_m"), float("inf"))
    # Worst first: more collisions, then more near misses, then smaller separations.
    return (collisions, near_misses, -min_sep_min, -min_sep_p05)


def mine_worst_cases(results_csv: str | Path, top_k: int = 20) -> dict[str, Any]:
    results_path = Path(results_csv)
    if not results_path.exists():
        raise FileNotFoundError(f"results.csv not found: {results_path}")
    if results_path.name != "results.csv":
        raise ValueError(f"Expected a results.csv path, got: {results_path}")

    run_dir = results_path.parent
    with results_path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    if not rows:
        worst_dir = run_dir / "worst_cases"
        worst_dir.mkdir(parents=True, exist_ok=True)
        return {"run_dir": str(run_dir), "worst_dir": str(worst_dir), "selected": 0, "copied": 0}

    ranked = sorted(rows, key=_severity_key, reverse=True)[: max(0, int(top_k))]
    worst_dir = run_dir / "worst_cases"
    if worst_dir.exists():
        shutil.rmtree(worst_dir)
    worst_dir.mkdir(parents=True, exist_ok=True)

    copied = 0
    out_rows: list[dict[str, Any]] = []
    for rank, row in enumerate(ranked, start=1):
        scenario = _row_str(row, ["scenario", "scenario_id"], "unknown")
        method = _row_str(row, ["method"], "unknown")
        n_agents = _row_int(row, ["N", "n_agents"], -1)
        seed = _row_int(row, ["seed"], -1)
        comm_profile = _row_str(row, ["comm_profile", "comm"], "unknown")
        case_name = f"rank_{rank:02d}_{scenario}_{method}_n{n_agents}_seed{seed}_comm_{comm_profile}"
        case_dir = worst_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)

        episode_dir = None
        for candidate in _episode_dirs_for_row(run_dir, row):
            if candidate.exists():
                episode_dir = candidate
                break

        copied_files: list[str] = []
        if episode_dir is not None:
            for p in sorted(episode_dir.glob("trace_collision_*.jsonl")):
                shutil.copy2(p, case_dir / p.name)
                copied_files.append(p.name)
            events = episode_dir / "events.jsonl"
            if events.exists():
                shutil.copy2(events, case_dir / events.name)
                copied_files.append(events.name)
            for p in sorted(episode_dir.glob("trace_collision_*.npz")):
                shutil.copy2(p, case_dir / p.name)
                copied_files.append(p.name)
            if copied_files:
                copied += 1

        with (case_dir / "episode_row.json").open("w", encoding="utf-8") as f:
            json.dump(row, f, indent=2)
        with (case_dir / "copied_files.json").open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "episode_dir": str(episode_dir) if episode_dir is not None else None,
                    "copied_files": copied_files,
                },
                f,
                indent=2,
            )

        out_row = dict(row)
        out_row["rank"] = rank
        out_row["copied_file_count"] = len(copied_files)
        out_rows.append(out_row)

    fields = ["rank", "copied_file_count"] + [k for k in rows[0].keys() if k not in {"rank", "copied_file_count"}]
    with (worst_dir / "index.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(out_rows)

    return {
        "run_dir": str(run_dir),
        "worst_dir": str(worst_dir),
        "selected": len(ranked),
        "copied": copied,
    }
