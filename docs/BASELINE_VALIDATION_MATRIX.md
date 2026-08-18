# Baseline Validation Matrix

`baseline-validation-matrix` is the compact per-baseline behavior matrix for the encounter families we care about most before stable-v1 promotion:

- `head_on`: reciprocal planar DAA conflict.
- `crossing`: dense planar right-of-way/deadlock conflict.
- `urban_obstacle`: hand-authored 3D city/building conflict.
- `communication_delay`: degraded V2V and fused-sensing exposure.
- `high_n_dense_merge`: higher-N 3D merge bottleneck.

Inspect the planned matrix before running it:

```bash
python -m microbench.cli baseline-validation-matrix \
  --out-dir runs_baseline_validation_matrix \
  --plan-only
```

Run a compact development slice:

```bash
python -m microbench.cli baseline-validation-matrix \
  --out-dir runs_baseline_validation_matrix_dev \
  --methods orca_heuristic,mpc_local,learned_tiny \
  --lanes head_on,communication_delay \
  --max-runs 4 \
  --require-pass
```

Run the full default matrix:

```bash
python -m microbench.cli baseline-validation-matrix \
  --out-dir runs_baseline_validation_matrix \
  --require-pass
```

The report writes `baseline_validation_matrix.json`, `results.csv`, and `summary.csv`.

`ok` is a hard gate: finite metrics, no planner exceptions, no soft timeouts, and no fallback commands. `behavior_pass` is separate evidence for collision-free behavior, nonnegative clearance, mission progress, and degraded-observation exposure. Keep that separation: lower-bound/template rows may legitimately fail behavior checks while still proving their plumbing is healthy.

Learned-policy rows are included in the same matrix so the PettingZoo/Gymnasium interface, observation contract, action conversion, and submission bundles can be validated against the same encounter families as classical planners.

For direct learned-policy or external policy-spec checks through the PettingZoo-style wrapper, use:

```bash
python -m microbench.cli rl-validation-matrix \
  --out-dir runs_rl_validation_matrix \
  --policy-spec examples/external_policy_spec.json \
  --require-pass
```

That command reuses these same lanes but reports RL rollout/interface health rather than planner `results.csv` leaderboard metrics.
