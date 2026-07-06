## Summary

What changed and why?

## Type

- [ ] Bug fix
- [ ] Planner or baseline change
- [ ] Scenario or suite change
- [ ] Metric, schema, or leaderboard change
- [ ] Documentation-only change
- [ ] Tooling/CI/packaging change

## Benchmark Contract

- [ ] I did not introduce privileged simulator information into planner code.
- [ ] I did not change shared timestep, collision, dynamics, comm, perception, or neighbor settings for only one method.
- [ ] I documented any public planner API, metric, suite, or result-schema change.
- [ ] I updated `result_schema.json` / current-schema golden fixtures if CSV fields or semantic metric values changed.
- [ ] I disclosed any expected nonzero planner timeout/error/fallback counts.

## Verification

Paste commands run:

```bash
python -m pytest -q
bash scripts/ci_sanity.sh
python -m microbench.cli golden-current-schema --golden-dir golden/current_schema
```

## Result Artifacts

If this changes planner behavior, scenarios, or metrics, attach or link:

- `results.csv`
- `summary.csv`
- `result_schema.json`
- generated `suite_manifest.yaml` when relevant
- representative traces or mined worst cases

## Notes For Reviewers

Anything surprising, risky, or intentionally deferred?
