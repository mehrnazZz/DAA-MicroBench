# DAA Microbench Documentation

Start here when navigating the public docs.

## Core Contract

- [DESIGN_V1.md](DESIGN_V1.md): benchmark scope, planner contract, allowed/forbidden information, agentic definition, schema policy, and v1 stability expectations.
- [PLANNER_API.md](PLANNER_API.md): planner implementation tutorial, lifecycle, `PlannerInput`, guardrails, registration, heterogeneous runs, and example planner.

## Running And Interpreting Benchmarks

- [SCENARIO_SUITES.md](SCENARIO_SUITES.md): official/generated suite registry, suite materialization, validation, and acceptance metadata.
- [BASELINES.md](BASELINES.md): built-in baseline roles, limitations, recommended comparison sets, and promotion criteria.
- [LEADERBOARD.md](LEADERBOARD.md): ranking policy, primary metrics, result categories, reproducibility rules, and review policy.
- [RL_INTERFACE.md](RL_INTERFACE.md): PettingZoo/Gymnasium-style wrappers for learning researchers.
- [LEARNED_POLICY_ADOPTION.md](LEARNED_POLICY_ADOPTION.md): exported-policy specs, model adapters, planner CSV generation, and learned bundle review.
- [RL_STABLE_V1_FREEZE.md](RL_STABLE_V1_FREEZE.md): stable-v1 RL interface freeze criteria, compatibility policy, and learned-policy artifact expectations.
- [RESULT_SUBMISSION.md](RESULT_SUBMISSION.md): result submission template, required artifacts, validation commands, and disclosure checklist.
- [PUBLIC_ALPHA_NOTES.md](PUBLIC_ALPHA_NOTES.md): current public-alpha status, known limitations, and reproducibility commands.
- [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md): public-alpha release checks, packaging smoke, CI expectations, and release-note requirements.

## Contributing

- Use the GitHub issue templates for bugs, feature requests, benchmark results, planner submissions, learned-policy submissions, and scenario proposals.
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

Run targeted CBF/MPC reference-evidence checks:

```bash
python -m microbench.cli baseline-evidence --out-dir runs_baseline_evidence --require-pass
```

Plan optional longer stable-metadata review rows:

```bash
python -m microbench.cli baseline-review --out-dir runs_baseline_review --plan-only
```

Smoke-test the RL interface:

```bash
python -m pytest tests/test_rl_interface.py -q
python -m microbench.cli rl-smoke --out-dir runs_rl_smoke --require-pass
python -m microbench.cli rl-smoke --out-dir runs_external_rl_smoke --policy-spec examples/external_policy_spec.json --require-pass
python -m microbench.cli rl-smoke --out-dir runs_external_model_predict_smoke --policy-spec examples/external_policy_model_predict_spec.json --max-steps 3 --require-pass
python -m microbench.cli run --scenario config/scenarios/stacked_swap_3d.yaml --method learned_policy_spec --policy-spec examples/external_policy_spec.json --n 4 --seed 0 --comm ideal_50hz --out-dir runs_external_policy_planner
python -m microbench.cli rl-smoke --out-dir runs_rl_tiny_learned --policy tiny_learned --require-pass
python -m microbench.cli rl-calibration --out-dir runs_rl_calibration --require-pass
python -m microbench.cli rl-contract --json
python -m microbench.cli rl-freeze-check --require-pass --json
python -m microbench.cli validate-learned-manifest --manifest examples/learned_submission_manifest_template.json --require-pass
python -m microbench.cli learned-submission-bundle --out-dir runs_learned_bundle --method learned_tiny --policy tiny_learned --require-pass
python -m microbench.cli learned-submission-bundle --out-dir runs_external_learned_bundle --method learned_policy_spec --policy-spec examples/external_policy_spec.json --require-pass
python -m microbench.cli validate-learned-bundle --bundle runs_learned_bundle --require-pass
python -m microbench.cli review-learned-bundle --bundle runs_learned_bundle --require-pass
```

Check the current result-schema fixture:

```bash
python -m microbench.cli golden-current-schema --golden-dir golden/current_schema
```

Smoke-test the public example planner:

```bash
python -m pytest tests/test_public_docs_examples.py -q
```

Run the learned-policy adapter example:

```bash
python examples/rl_external_policy_adapter.py --max-steps 100
```

Smoke-test the external policy-spec loader:

```bash
python -m microbench.cli rl-smoke --out-dir runs_external_rl_smoke --policy-spec examples/external_policy_spec.json --require-pass
python -m microbench.cli rl-smoke --out-dir runs_external_model_predict_smoke --policy-spec examples/external_policy_model_predict_spec.json --max-steps 3 --require-pass
python -m microbench.cli run --scenario config/scenarios/stacked_swap_3d.yaml --method learned_policy_spec --policy-spec examples/external_policy_spec.json --n 4 --seed 0 --comm ideal_50hz --out-dir runs_external_policy_planner
```

Regenerate a compatible tiny learned-policy weight artifact:

```bash
python examples/rl_train_tiny_linear_policy.py --out /tmp/tiny_linear_policy.json
```

Build a learned-policy submission bundle:

```bash
python -m microbench.cli validate-learned-manifest --manifest examples/learned_submission_manifest_template.json --require-pass
python -m microbench.cli learned-submission-bundle --out-dir runs_learned_bundle --method learned_tiny --policy tiny_learned --require-pass
python -m microbench.cli learned-submission-bundle --out-dir runs_external_learned_bundle --method learned_policy_spec --policy-spec examples/external_policy_spec.json --require-pass
python -m microbench.cli validate-learned-bundle --bundle runs_learned_bundle --require-pass
python -m microbench.cli review-learned-bundle --bundle runs_learned_bundle --require-pass
```
