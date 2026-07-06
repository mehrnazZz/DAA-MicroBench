---
name: Benchmark result
about: Submit a reproducible benchmark result
title: "[Result] "
labels: benchmark-result
---

## Summary

- Method:
- Method version / commit:
- DAA Microbench commit:
- Suite:
- Hardware:
- Python version:

## Command

```bash
# exact command
```

## Artifacts

- `results.csv`:
- `summary.csv`:
- `result_schema.json`:
- `_generated_scenarios/<suite>/suite_manifest.yaml`:
- changed configs or scenarios:
- representative traces or mined worst cases:

## Key Metrics

Paste the relevant `summary.csv` rows or table here.

Minimum fields:

- `collision_episode_rate`
- `unique_collision_pairs_mean`
- `collision_pair_ticks_mean`
- `min_sep_p05_mean`
- `completion_rate_mean`
- `mean_time_to_goal_mean`
- `planner_ms_p95`
- `planner_timeout_count_mean`
- `planner_error_count_mean`
- `planner_fallback_count_mean`

## Validation

```bash
python -m microbench.cli validate-scenarios ...
python -m microbench.cli check-acceptance ...
```

## Disclosure Checklist

- [ ] I did not use simulator ground truth outside `PlannerInput`.
- [ ] I did not change shared benchmark settings for only my method.
- [ ] I included all runs, including failures.
- [ ] I disclosed learned weights and external services.
- [ ] I disclosed whether the method uses V2V, intent, agent messages, local sensing, or learned weights.
- [ ] I disclosed any nonzero planner timeout/error/fallback counts.
- [ ] I included enough detail to reproduce the result.

See `docs/RESULT_SUBMISSION.md` and `docs/LEADERBOARD.md`.
