---
name: Planner submission
about: Propose or discuss a planner implementation
title: "[Planner] "
labels: planner
---

## Summary

- Method name:
- Planner family: reactive / geometric / optimization / learned / hybrid / other
- Intended role: illustrative / reference baseline / experimental baseline / external submission

## Public Contract

- [ ] Uses only public `PlannerInput` fields.
- [ ] Returns finite shape `(3,)` commands or `PlannerOutput`.
- [ ] Handles both planar and non-planar inputs, or clearly documents dimensional limits.
- [ ] Does not mutate shared benchmark settings for only this method.

## Information Sources

- Uses V2V: yes/no
- Uses local sensing: yes/no
- Uses intent: yes/no
- Uses agent messages: yes/no
- Uses learned weights: yes/no
- Uses external services: yes/no

## Reproduction

```bash
# command used for smoke run
```

## Results

Attach or paste:

- `results.csv`
- `summary.csv`
- `result_schema.json`
- relevant traces for collisions, near misses, or planner guardrail events

## Guardrail Counts

- `planner_timeout_count_mean`:
- `planner_error_count_mean`:
- `planner_fallback_count_mean`:

## Notes

See `docs/DESIGN_V1.md` and `docs/PLANNER_API.md`.
