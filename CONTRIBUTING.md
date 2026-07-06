# Contributing to DAA Microbench

Thanks for helping make DAA Microbench more useful and trustworthy. The benchmark is meant to be small, reproducible, and strict about fairness.

## Development Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
python -m pytest -q
```

Optional extras:

```bash
pip install -e ".[ml]"
pip install -e ".[opt]"
```

## Contribution Types

Good first contributions:

- scenario documentation and schema examples
- new tests for existing planner contracts
- clearer failure traces or replay diagnostics
- baseline planner improvements with before/after results
- bug fixes that preserve benchmark comparability

Larger contributions should open an issue first:

- new official scenario suites
- new leaderboard metrics
- changes to collision, completion, or observation semantics
- new planner APIs
- changes to the canonical sweep protocol

For the public benchmark contract, use [docs/DESIGN_V1.md](docs/DESIGN_V1.md). For planner implementation details, use [docs/PLANNER_API.md](docs/PLANNER_API.md).
Use the GitHub issue templates for benchmark results, planner submissions, scenario proposals, bugs, and feature requests.

## Benchmark Fairness Rules

Planner submissions must not:

- read simulator ground truth beyond `PlannerInput`
- mutate shared scenario, neighbor, dynamics, or collision settings
- use method-specific timesteps, radii, sensing ranges, or communication profiles
- inspect future states, goals of other agents beyond observed messages, or trace files during an episode

If a method requires extra information, model it explicitly as perception, V2V, intent, or agent messages.

## Pull Request Checklist

Before opening a PR:

- run `python -m pytest -q`
- run `bash scripts/ci_sanity.sh /private/tmp/daa_microbench_ci_sanity`
- update README or docs when user-facing behavior changes
- add or update tests for new planner, metric, scenario, or schema behavior
- keep result schema changes explicit in `microbench/metrics/io.py`
- run `python -m microbench.cli golden-current-schema --golden-dir golden/current_schema` for schema or metric-semantics changes
- avoid unrelated refactors in benchmark-critical code paths

## Scenario Contributions

New scenarios should include:

- a clear `scenario.description`
- intended failure mode or benchmark purpose
- whether it is planar or 3D
- relevant perception/comm assumptions
- expected baseline behavior if known

Scenarios proposed for official suites should include a smoke result from at least one baseline.

## Baseline Planner Contributions

Baseline planners should be:

- deterministic for fixed seed/config
- isolated from privileged state
- documented in the README or a planner docstring
- accompanied by at least one focused test

Name expert planners carefully. If a method is heuristic, call it heuristic.

## Result Submissions

For benchmark result submissions, follow [docs/RESULT_SUBMISSION.md](docs/RESULT_SUBMISSION.md).
