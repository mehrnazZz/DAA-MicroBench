# DAA Microbench Documentation

Start here when navigating the public docs.

## Core Contract

- [DESIGN_V1.md](DESIGN_V1.md): benchmark scope, planner contract, allowed/forbidden information, agentic definition, schema policy, and v1 stability expectations.
- [PLANNER_API.md](PLANNER_API.md): planner implementation tutorial, lifecycle, `PlannerInput`, guardrails, registration, heterogeneous runs, and example planner.

## Running And Interpreting Benchmarks

- [SCENARIO_SUITES.md](SCENARIO_SUITES.md): official/generated suite registry, suite materialization, validation, and acceptance metadata.
- [BASELINES.md](BASELINES.md): built-in baseline roles, limitations, recommended comparison sets, and promotion criteria.
- [LEADERBOARD.md](LEADERBOARD.md): ranking policy, primary metrics, result categories, reproducibility rules, and review policy.
- [RESULT_SUBMISSION.md](RESULT_SUBMISSION.md): result submission template, required artifacts, validation commands, and disclosure checklist.
- [PUBLIC_ALPHA_NOTES.md](PUBLIC_ALPHA_NOTES.md): current public-alpha status, known limitations, and reproducibility commands.
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md): public-alpha release checks, packaging smoke, CI expectations, and release-note requirements.

## Contributing

- Use the GitHub issue templates for bugs, feature requests, benchmark results, planner submissions, and scenario proposals.
- Use the pull request template checklist when changing planner APIs, metrics, schemas, suites, or benchmark-critical behavior.
- See [../CONTRIBUTING.md](../CONTRIBUTING.md) for setup, fairness rules, and PR expectations.

## Useful Commands

List methods:

```bash
python -m microbench.cli list-methods --json --include-aliases
```

List suites:

```bash
python -m microbench.cli list-suites --json
```

Run CI-style sanity:

```bash
bash scripts/ci_sanity.sh
```

Run installed-package smoke:

```bash
bash scripts/package_smoke.sh
```

Run the public-alpha readiness dry run:

```bash
bash scripts/release_readiness.sh
```

Audit baseline readiness:

```bash
python -m microbench.cli baseline-audit --require-public-alpha-ready
```

Run baseline behavior smoke:

```bash
python -m microbench.cli baseline-smoke --out-dir runs_baseline_smoke --require-pass
```

Calibrate experimental baseline promotion status:

```bash
python -m microbench.cli baseline-promotion --out-dir runs_baseline_promotion --require-calibrated
```

Plan optional longer stable-metadata review rows:

```bash
python -m microbench.cli baseline-review --out-dir runs_baseline_review --plan-only
```

Check the current result-schema fixture:

```bash
python -m microbench.cli golden-current-schema --golden-dir golden/current_schema
```

Smoke-test the public example planner:

```bash
python -m pytest tests/test_public_docs_examples.py -q
```
