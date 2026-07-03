#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-runs/ci_sanity}"
SCENARIO="config/scenarios/corridor.yaml"
METHOD="baseline_goal"
COMM="ideal_50hz"
N="10"
SEEDS="0:2"

python -m microbench.cli sweep \
  --scenarios "${SCENARIO}" \
  --methods "${METHOD}" \
  --seeds "${SEEDS}" \
  --n "${N}" \
  --comm "${COMM}" \
  --out-dir "${OUT_DIR}"

export OUT_DIR
python - <<'PY'
import csv
import os
from pathlib import Path

out_dir = Path(os.environ.get("OUT_DIR", "runs/ci_sanity"))
results = out_dir / "results.csv"
summary = out_dir / "summary.csv"
if not results.exists() or not summary.exists():
    raise SystemExit("Missing results.csv or summary.csv")

rows = list(csv.DictReader(results.open()))
if len(rows) != 3:
    raise SystemExit(f"Expected 3 episodes (seeds 0..2), got {len(rows)}")

print("ci_sanity: PASS")
print(f"results: {results}")
print(f"summary: {summary}")
PY
