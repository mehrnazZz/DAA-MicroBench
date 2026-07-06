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
- [ ] GitHub Actions CI is green for Python 3.10, 3.11, and 3.12.
- [ ] `docs/README.md`, `docs/PUBLIC_ALPHA_NOTES.md`, `docs/DESIGN_V1.md`, `docs/PLANNER_API.md`, `docs/SCENARIO_SUITES.md`, `docs/BASELINES.md`, `docs/LEADERBOARD.md`, and `docs/RESULT_SUBMISSION.md` reflect the release behavior.
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
