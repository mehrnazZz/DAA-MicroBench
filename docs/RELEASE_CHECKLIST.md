# Public Alpha Release Checklist

Use this checklist before tagging a public alpha, announcing a leaderboard run, or inviting external planner submissions.

## Required Checks

- [ ] Working tree is clean.
- [ ] `python -m pytest -q` passes.
- [ ] `bash scripts/ci_sanity.sh` passes.
- [ ] `bash scripts/package_smoke.sh` passes.
- [ ] `bash scripts/release_readiness.sh` passes.
- [ ] `DAA_REQUIRE_CLEAN=1 bash scripts/release_readiness.sh` passes before tagging.
- [ ] `python -m microbench.cli golden-current-schema --golden-dir golden/current_schema` passes.
- [ ] `python -m microbench.cli baseline-audit --require-public-alpha-ready` passes.
- [ ] `python -m microbench.cli baseline-smoke --out-dir runs_baseline_smoke --require-pass` passes from a fresh output directory.
- [ ] `python -m microbench.cli baseline-promotion --out-dir runs_baseline_promotion --require-calibrated` passes, while `--require-stable-v1-ready` remains blocked until promotion work is complete.
- [ ] `python -m microbench.cli baseline-evidence --out-dir runs_baseline_evidence --require-pass` passes for targeted CBF/MPC/NMPC/EGO-Swarm/VO/RVO reference-evidence checks.
- [ ] `python -m microbench.cli baseline-evidence --out-dir runs_optimizer_evidence --save-optimizer-traces --require-pass` writes compact Foxglove-ready optimizer trace JSONL artifacts for qualitative review.
- [ ] `python -m microbench.cli advanced-baseline-comparison --out-dir runs_advanced_baseline_comparison --require-pass` passes for the compact shared 3D advanced-baseline lane.
- [ ] `python -m microbench.cli rl-smoke --out-dir runs_rl_smoke --require-pass` passes for PettingZoo/Gymnasium wrapper health.
- [ ] `python -m microbench.cli rl-smoke --out-dir runs_external_rl_smoke --policy-spec examples/external_policy_spec.json --require-pass` passes for external policy-spec loading.
- [ ] `python -m microbench.cli rl-smoke --out-dir runs_external_model_predict_smoke --policy-spec examples/external_policy_model_predict_spec.json --max-steps 3 --require-pass` passes for import-based external model loading.
- [ ] `python -m microbench.cli run --scenario config/scenarios/stacked_swap_3d.yaml --method learned_policy_spec --policy-spec examples/external_policy_spec.json --n 4 --seed 0 --comm ideal_50hz --out-dir runs_external_policy_planner` passes for external policy-spec planner CSV generation.
- [ ] `python -m microbench.cli rl-smoke --out-dir runs_rl_tiny_learned --policy tiny_learned --require-pass` passes for the frozen learned-policy fixture.
- [ ] `python -m microbench.cli rl-calibration --out-dir runs_rl_calibration --require-pass` passes for compact 3D/degraded RL wrapper exposure.
- [ ] `python -m microbench.cli rl-contract --json` prints the current RL interface contract.
- [ ] `python -m microbench.cli rl-freeze-check --require-pass --json` passes and writes a stable-v1 readiness artifact when preparing learned-policy submissions or v1 candidates.
- [ ] `python -m microbench.cli validate-learned-manifest --manifest examples/learned_submission_manifest_template.json --require-pass` passes for the learned manifest template.
- [ ] Learned submission schemas are packaged under `microbench/bundled_config/schemas/` and documented in `docs/LEARNED_SUBMISSION_SCHEMAS.md`.
- [ ] `python -m microbench.cli learned-submission-schema-check --require-pass` passes for schema packaging, template validity, docs coverage, and overlay guidance.
- [ ] `python -m microbench.cli learned-submission-bundle --out-dir runs_learned_bundle --method learned_tiny --policy tiny_learned --require-pass` passes for the frozen learned-policy fixture.
- [ ] `python -m microbench.cli learned-submission-bundle --out-dir runs_external_learned_bundle --method learned_policy_spec --policy-spec examples/external_policy_spec.json --require-pass` passes for an external policy-spec planner submission.
- [ ] Learned bundles contain `learned_submission_manifest.json` with artifact hashes and policy-spec provenance.
- [ ] `python -m microbench.cli validate-learned-bundle --bundle runs_learned_bundle --require-pass` passes for the frozen learned-policy fixture bundle.
- [ ] `python -m microbench.cli review-learned-bundle --bundle runs_learned_bundle --require-pass` summarizes the frozen learned-policy fixture bundle.
- [ ] Optional when `.[rl]` is installed: `python -m pytest tests/test_rl_optional_integrations.py -q` passes.
- [ ] GitHub Actions CI is green for Python 3.10, 3.11, and 3.12.
- [ ] `docs/README.md`, `docs/PUBLIC_ALPHA_NOTES.md`, `docs/DESIGN_V1.md`, `docs/PLANNER_API.md`, `docs/SCENARIO_SUITES.md`, `docs/BASELINES.md`, `docs/LEADERBOARD.md`, `docs/RL_INTERFACE.md`, `docs/RL_STABLE_V1_FREEZE.md`, and `docs/RESULT_SUBMISSION.md` reflect the release behavior.
- [ ] Issue templates and pull request template still match the current public contract.
- [ ] Generated official suites validate with `python -m microbench.cli validate-scenarios --all-builtins --all-generated-suites --quiet`.
- [ ] Any metric/schema change has an updated `result_schema.json` and current-schema golden fixture.

## Release Notes

Before tagging, write a short release note with:

- supported Python versions
- official suite status and any pre-v1 caveats
- result schema version
- leaderboard policy version
- known limitations
- exact reproduction commands for the smoke and package checks
- whether the release is a public alpha or stable v1

## Do Not Release If

- official smoke acceptance fails
- planner guardrail counts are unexpectedly nonzero in smoke runs
- installed-wheel smoke cannot run from outside the source checkout
- docs describe metrics, scenarios, or planner inputs that differ from the code
- any public template asks for artifacts that the runner cannot produce
