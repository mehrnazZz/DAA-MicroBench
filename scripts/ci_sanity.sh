#!/usr/bin/env bash
set -euo pipefail

OUT_DIR="${1:-runs/ci_sanity}"
METHOD="baseline_goal"

python -m microbench.cli canonical-sweep \
  --suite official_smoke_generated \
  --methods "${METHOD}" \
  --out-dir "${OUT_DIR}"

python -m microbench.cli validate-scenarios \
  --all-builtins \
  --all-generated-suites \
  --quiet

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
    raise SystemExit(f"Expected 3 generated smoke episodes, got {len(rows)}")

print("ci_sanity: PASS")
print(f"results: {results}")
print(f"summary: {summary}")
PY
