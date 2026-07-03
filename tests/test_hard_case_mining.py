from __future__ import annotations

import csv
from pathlib import Path

from microbench.tools import mine_worst_cases


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_mine_worst_cases_copies_ranked_artifacts(tmp_path: Path) -> None:
    run_dir = tmp_path / "runs" / "demo"
    results = run_dir / "results.csv"
    episodes = run_dir / "episodes"
    rows = [
        {
            "run_id": "demo",
            "method": "baseline_goal",
            "scenario": "intersection",
            "comm_profile": "ideal_50hz",
            "N": 10,
            "seed": 0,
            "collisions": 10,
            "near_misses": 2,
            "min_sep_min_m": -0.8,
            "min_sep_p05_m": -0.5,
        },
        {
            "run_id": "demo",
            "method": "baseline_goal",
            "scenario": "intersection",
            "comm_profile": "ideal_50hz",
            "N": 10,
            "seed": 1,
            "collisions": 3,
            "near_misses": 20,
            "min_sep_min_m": -0.2,
            "min_sep_p05_m": 0.1,
        },
        {
            "run_id": "demo",
            "method": "orca_expert",
            "scenario": "intersection",
            "comm_profile": "ideal_50hz",
            "N": 10,
            "seed": 2,
            "collisions": 0,
            "near_misses": 1,
            "min_sep_min_m": 0.05,
            "min_sep_p05_m": 0.2,
        },
    ]
    _write_csv(results, rows)

    ep0 = episodes / "intersection_baseline_goal_n10_seed0_comm_ideal_50hz"
    ep0.mkdir(parents=True, exist_ok=True)
    (ep0 / "events.jsonl").write_text('{"type":"collision"}\n', encoding="utf-8")
    (ep0 / "trace_collision_0_1_t1.00.jsonl").write_text('{"kind":"frame"}\n', encoding="utf-8")

    ep1 = episodes / "intersection_baseline_goal_n10_seed1_comm_ideal_50hz"
    ep1.mkdir(parents=True, exist_ok=True)
    (ep1 / "events.jsonl").write_text('{"type":"near_miss"}\n', encoding="utf-8")

    out = mine_worst_cases(results_csv=results, top_k=2)
    worst_dir = Path(out["worst_dir"])
    assert out["selected"] == 2
    assert worst_dir.exists()
    assert (worst_dir / "index.csv").exists()

    rank1 = next(worst_dir.glob("rank_01_*"))
    assert (rank1 / "events.jsonl").exists()
    assert (rank1 / "trace_collision_0_1_t1.00.jsonl").exists()

