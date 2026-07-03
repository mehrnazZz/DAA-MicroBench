# Benchmark Result Submission

Use this template when submitting DAA Microbench results.

## Required Summary

```text
Method:
Method version / commit:
DAA Microbench commit:
Suite:
Command:
Hardware:
Python version:
Dependencies:
Uses learned weights: yes/no
Uses V2V: yes/no
Uses intent: yes/no
Uses agent messages: yes/no
Uses local sensing: yes/no
```

## Required Artifacts

Attach or link:

- `results.csv`
- `summary.csv`
- `_generated_scenarios/<suite>/suite_manifest.yaml` when using a generated official suite
- any changed scenario/config files
- any planner source or package version needed to reproduce
- representative failure traces for nonzero collision or near-miss-heavy results

Run validation before submitting:

```bash
python -m microbench.cli validate-scenarios \
  --scenario path/to/custom_or_official.yaml \
  --suite-manifest path/to/suite_manifest.yaml
```

## Reproduction Command

Paste the exact command. Example:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_alpha \
  --methods your_method \
  --out-dir runs_your_method_official_alpha
```

Use `python -m microbench.cli list-suites` to confirm suite status, source, and default run matrix.
Use `python -m microbench.cli list-suites --json` to inspect generated-suite acceptance rules.
Generated suite manifests preserve the same `acceptance` metadata and should be submitted with results.

## Result Tables

Include the relevant rows from `summary.csv`. At minimum include:

- `collision_episode_rate`
- `unique_collision_pairs_mean`
- `collision_pair_ticks_mean`
- `min_sep_p05_mean`
- `completion_rate_mean`
- `mean_time_to_goal_mean`
- `deadlock_time_pct_mean`
- `planner_ms_p95`
- `obs_neighbors_mean`
- `obs_v2v_fraction_mean`
- `obs_sensor_fraction_mean`
- `obs_stale_fraction_mean`
- `obs_empty_fraction_mean`

## Disclosure Checklist

- [ ] I did not use simulator ground truth outside `PlannerInput`.
- [ ] I did not change shared neighbor, collision, dynamics, comm, or perception settings for only my method.
- [ ] I included all runs, including failures.
- [ ] I disclosed learned weights or external services.
- [ ] I included enough config and command detail to reproduce the result.

## Notes

Use [docs/LEADERBOARD.md](LEADERBOARD.md) for ranking policy and interpretation.
