# Current-Schema Golden Smoke Bundle

This folder freezes a tiny deterministic smoke reference for the current leaderboard CSV schema.

## Contents
- `results.csv`: per-episode outputs with explicit collision semantics.
- `summary.csv`: grouped leaderboard summary.

## Source Commands
```bash
python -m microbench.cli run \
  --scenario config/scenarios/corridor.yaml \
  --method baseline_goal \
  --n 4 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir golden/current_schema

python -m microbench.cli run \
  --scenario config/scenarios/corridor.yaml \
  --method mixed \
  --agent-methods baseline_goal,template,baseline_goal,template \
  --n 4 \
  --seed 1 \
  --comm ideal_50hz \
  --out-dir golden/current_schema
```

## Purpose
- Exercises the current result and summary schemas.
- Exercises both single-method and heterogeneous-agent method labels.
- Stays fast enough to regenerate during normal development.
- Timing columns are useful for smoke checks but should not be compared bit-for-bit across machines.
