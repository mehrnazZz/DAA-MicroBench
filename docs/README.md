# DAA Microbench Documentation

Start here when navigating the public docs.

## Core Contract

- [DESIGN_V1.md](DESIGN_V1.md): benchmark scope, planner contract, allowed/forbidden information, agentic definition, schema policy, and v1 stability expectations.
- [PLANNER_API.md](PLANNER_API.md): planner implementation tutorial, lifecycle, `PlannerInput`, guardrails, registration, heterogeneous runs, and example planner.

## Running And Interpreting Benchmarks

- [SCENARIO_SUITES.md](SCENARIO_SUITES.md): official/generated suite registry, suite materialization, validation, and acceptance metadata.
- [BASELINES.md](BASELINES.md): built-in baseline roles, limitations, recommended comparison sets, and promotion criteria.
- [BASELINE_FIDELITY.md](BASELINE_FIDELITY.md): implementation-fidelity tiers, provenance matrix, and external-reference manifest workflow.
- [LEADERBOARD.md](LEADERBOARD.md): ranking policy, primary metrics, result categories, reproducibility rules, and review policy.
- [RL_INTERFACE.md](RL_INTERFACE.md): PettingZoo/Gymnasium-style wrappers for learning researchers.
- [LEARNED_POLICY_ADOPTION.md](LEARNED_POLICY_ADOPTION.md): exported-policy specs, model adapters, planner CSV generation, and learned bundle review.
- [LEARNED_SUBMISSION_SCHEMAS.md](LEARNED_SUBMISSION_SCHEMAS.md): learned-submission JSON Schemas, compatibility policy, and full-manifest vs overlay guidance.
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

Validate an external official/reference manifest:

```bash
python -m microbench.cli validate-external-reference --manifest examples/external_reference_rmader_manifest.yaml --json
```

Prepare a portable bundle for an official external implementation run:

```bash
python -m microbench.cli external-reference-bundle --method-family rmader --out-dir runs_external_references/rmader_official_bundle --scenarios urban_conflict_3d,urban_throughput_3d,stacked_swap_3d --n 4,8 --seeds 2 --comm realistic_v2v_50hz --runner-type ros
```

Run baseline behavior smoke:

```bash
python -m microbench.cli baseline-smoke --out-dir runs_baseline_smoke --require-pass
```

Calibrate experimental baseline promotion status:

```bash
python -m microbench.cli baseline-promotion --out-dir runs_baseline_promotion --require-calibrated
```

Run targeted CBF/MPC/NMPC/BVC tube-DMPC/dynamic tube-DMPC/RMADER/EGO-Swarm/VO/RVO reference-evidence checks:

```bash
python -m microbench.cli baseline-evidence --out-dir runs_baseline_evidence --require-pass
```

Add compact Foxglove-ready optimizer trace artifacts:

```bash
python -m microbench.cli baseline-evidence --out-dir runs_optimizer_evidence --save-optimizer-traces --require-pass
```

Run the compact shared 3D advanced-baseline comparison:

```bash
python -m microbench.cli advanced-baseline-comparison --out-dir runs_advanced_baseline_comparison --require-pass
```

Generate a single panelized Foxglove MCAP for optimizer comparison on the urban-throughput map:

```bash
python -m microbench.cli advanced-baseline-comparison --out-dir runs_urban_throughput_comparison --scenario config/scenarios/urban_throughput_3d.yaml --methods dmpc_best_response,bvc_tube_dmpc,dynamic_tube_dmpc,mpc_nonlinear,ego_swarm_opt,rmader --n 8 --seed 2 --comm realistic_v2v_50hz --duration-s 20 --planner-preset scale --export-foxglove-mcap
```

Run a capped optimizer-grade suite review for NMPC versus EGO-Swarm optimization:

```bash
python -m microbench.cli optimizer-suite-review --out-dir runs_optimizer_suite_review --max-runs 4 --require-pass
```

The optimizer review retries planner-guardrail rows once by default and records retry evidence in `optimizer_suite_review.json`, so transient local runtime spikes stay visible without being confused with persistent baseline failures.

Run a fleet-size scaling ladder:

```bash
python -m microbench.cli scale-benchmark --out-dir runs_scale_benchmark --scenarios config/scenarios/stacked_swap_3d.yaml,config/scenarios/urban_conflict_3d.yaml,config/scenarios/urban_throughput_3d.yaml --n 4,8,16,30 --seeds 2 --comm realistic_v2v_50hz --duration-s 60 --scale-spawn-profile dense --planner-preset scale --run-timeout-s 360
```

The scale report writes `scale_summary.csv` and `scale_benchmark.json`, recording completed rows, hard timeouts, guardrail pressure, collision rates, completion, final goal distance, progress fraction, and planner latency by scenario, method, and N. `--planner-preset scale` bounds optimizer effort for large-fleet studies, including NMPC/DMPC/BVC/RMADER command-level velocity/yield guarding, RMADER dynamic-hull/cached-validation/seed-budget limits, and EGO-Swarm optimized receding cached-plan reuse plus command-level velocity/yield guarding; omit it for the fuller default planner settings.

Package the default N=30 high-volume evidence run:

```bash
python -m microbench.cli high-volume-evidence --out-dir runs_high_volume_evidence --require-pass
```

Use `--plan-only` to inspect the prepared matrix before launching the full run; the command writes `high_volume_evidence.json` plus the generated scale and leaderboard artifacts.

Turn scale summaries into a high-volume leaderboard:

```bash
python -m microbench.cli high-volume-leaderboard --scale-summary runs_scale_benchmark/scale_summary.csv --out runs_scale_benchmark/high_volume_leaderboard.json
```

Run the all-suite baseline leaderboard:

```bash
python -m microbench.cli baseline-leaderboard --out-dir runs_baseline_leaderboard --suites all --require-pass --require-complete
```

Export a saved trace to a Foxglove-compatible MCAP log:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/stacked_swap_3d.yaml \
  --method your_method \
  --n 10 \
  --seed 0 \
  --comm ideal_50hz \
  --save-trace \
  --out-dir runs/<run_id>

python -m microbench.cli foxglove-export \
  --trace runs/<run_id>/episodes/<episode_dir>/trace_episode.jsonl \
  --out runs/<run_id>/episode.mcap
```

Install `daa-microbench[foxglove]` first. The MCAP contains `/tf`, environment/static scene entities, agent scene entities, executed trails, sensing links, sensor/range volumes, future intent trajectories when present, frame diagnostics, and events for Foxglove Studio.

For a single file that compares several baselines in Foxglove panels:

```bash
python -m microbench.cli foxglove-comparison-export \
  --trace mpc_nonlinear=runs/<run_id>/episodes/<mpc_episode>/trace_episode.jsonl \
  --trace ego_swarm_opt=runs/<run_id>/episodes/<ego_episode>/trace_episode.jsonl \
  --out runs/<run_id>/baseline_comparison.mcap
```

The comparison export writes per-method topics under `/daa/comparison/<method>/...` and shared namespaced transforms on `/tf`.

`advanced-baseline-comparison --export-foxglove-mcap` saves the per-method traces and writes the same combined `baseline_comparison.mcap` automatically. It also writes `comparison_manifest.json`; use `--planner-preset scale` when recording a visual comparison that should match scale-tuned optimizer settings.

Render a multi-panel episode report from a saved trace:

```bash
python -m microbench.cli episode-report \
  --trace runs/<run_id>/episodes/<episode_dir>/trace_episode.jsonl \
  --out runs/<run_id>/episode_report.html
```

Use `--plotly-source inline` after installing `daa-microbench[viz]` for a single-file report that works offline.

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
python -m microbench.cli learned-submission-schema-check --require-pass
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
python -m microbench.cli learned-submission-schema-check --require-pass
python -m microbench.cli learned-submission-bundle --out-dir runs_learned_bundle --method learned_tiny --policy tiny_learned --require-pass
python -m microbench.cli learned-submission-bundle --out-dir runs_external_learned_bundle --method learned_policy_spec --policy-spec examples/external_policy_spec.json --require-pass
python -m microbench.cli validate-learned-bundle --bundle runs_learned_bundle --require-pass
python -m microbench.cli review-learned-bundle --bundle runs_learned_bundle --require-pass
```
