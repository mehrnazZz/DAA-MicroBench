# Baseline Comparison Fixture

This folder contains a compact, path-independent comparison report for a tiny baseline calibration run.

The fixture is intentionally not a leaderboard. The generated scenarios use an 8 second duration override, so completion can be low even for useful baselines. Use this report to inspect method wiring, collision behavior, planner cost, and guardrail counts.

## Regeneration

```bash
python -m microbench.cli canonical-sweep \
  --suite official_experimental_baselines \
  --methods baseline_goal,orca_heuristic,cbf_qp,mpc_local \
  --out-dir /tmp/daa_baseline_comparison

python -m microbench.cli baseline-report \
  --summary /tmp/daa_baseline_comparison/summary.csv \
  --results /tmp/daa_baseline_comparison/results.csv \
  --suite official_experimental_baselines \
  --out golden/baseline_comparison/report.json \
  --generated-by "python -m microbench.cli canonical-sweep --suite official_experimental_baselines --methods baseline_goal,orca_heuristic,cbf_qp,mpc_local"
```

## Contents

- `report.json`: deterministic JSON projection of selected `summary.csv` metrics plus method-level aggregates.

Timing fields are machine-dependent. Treat them as calibration evidence, not exact regression targets.
