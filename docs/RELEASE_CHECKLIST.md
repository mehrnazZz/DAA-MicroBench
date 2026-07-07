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
- [ ] `python -m microbench.cli baseline-evidence --out-dir runs_baseline_evidence --require-pass` passes for targeted CBF/MPC reference-evidence checks.
- [ ] `python -m microbench.cli rl-smoke --out-dir runs_rl_smoke --require-pass` passes for PettingZoo/Gymnasium wrapper health.
- [ ] `python -m microbench.cli rl-smoke --out-dir runs_rl_tiny_learned --policy tiny_learned --require-pass` passes for the frozen learned-policy fixture.
- [ ] `python -m microbench.cli rl-calibration --out-dir runs_rl_calibration --require-pass` passes for compact 3D/degraded RL wrapper exposure.
- [ ] `python -m microbench.cli rl-contract --json` prints the current RL interface contract.
- [ ] `python -m microbench.cli rl-freeze-check --require-pass --json` passes and writes a stable-v1 readiness artifact when preparing learned-policy submissions or v1 candidates.
- [ ] `python -m microbench.cli learned-submission-bundle --out-dir runs_learned_bundle --method learned_tiny --policy tiny_learned --require-pass` passes for the frozen learned-policy fixture.
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
