# Public Alpha Notes

Status date: 2026-07-06

DAA Microbench is ready for public-alpha evaluation: the repository has a documented planner contract, generated 2D/3D/agentic scenario suites, baseline metadata, result-schema manifests, issue templates, package smoke checks, and GitHub Actions CI.

This is not yet a stable v1 release. The public contract is intended to be reviewable and usable, but official suite membership, acceptance bands, experimental baselines, RL observation/reward wrappers, and leaderboard scoring may still change before v1. In particular, `cbf_qp` and `mpc_local` are runnable experimental baselines, and `negotiation_yield` is a pre-v1 agentic reference rather than a stable-v1 leaderboard anchor.

## Supported Environment

- Python 3.10, 3.11, and 3.12
- Core install: `pip install -e .`
- Result schema version: `0.4.0`
- License: Apache-2.0

## Public Alpha Suites

- `official_smoke_generated`: tiny 2D/3D/agentic smoke coverage for CI and quick checks.
- `official_alpha`: pre-v1 generated suite mixing planar and 3D DAA families.
- `official_3d_stress`: volumetric, dense-swarm, vertical, and noncooperative 3D DAA stress cases.
- `official_agentic_stress`: heterogeneous priorities, multi-intruder/noncooperative traffic, partial sensing, and agentic coordination.
- `official_experimental_baselines`: compact calibration suite for experimental CBF-QP and MPC-local baselines.
- `official_promotion_calibration`: compact 3D and degraded sensing/communication calibration suite used by `baseline-promotion` and `rl-calibration`.

## Reproducibility Commands

Run the complete public-alpha dry run:

```bash
bash scripts/release_readiness.sh
```

Before tagging or announcing a release, require a clean working tree:

```bash
DAA_REQUIRE_CLEAN=1 bash scripts/release_readiness.sh
```

Individual checks:

```bash
python -m pytest -q
bash scripts/ci_sanity.sh
bash scripts/package_smoke.sh
python -m microbench.cli golden-current-schema --golden-dir golden/current_schema
python -m microbench.cli validate-scenarios --all-builtins --all-generated-suites --quiet
python -m microbench.cli baseline-audit --require-public-alpha-ready
python -m microbench.cli baseline-smoke --out-dir runs_baseline_smoke --require-pass
python -m microbench.cli baseline-promotion --out-dir runs_baseline_promotion --require-calibrated
python -m microbench.cli baseline-evidence --out-dir runs_baseline_evidence --require-pass
python -m microbench.cli rl-smoke --out-dir runs_rl_smoke --require-pass
python -m microbench.cli rl-smoke --out-dir runs_rl_tiny_learned --policy tiny_learned --require-pass
python -m microbench.cli rl-calibration --out-dir runs_rl_calibration --require-pass
python -m microbench.cli rl-contract --json
python -m microbench.cli rl-freeze-check --require-pass --json
python -m microbench.cli learned-submission-bundle --out-dir runs_learned_bundle --method learned_tiny --policy tiny_learned --require-pass
python -m microbench.cli validate-learned-bundle --bundle runs_learned_bundle --require-pass
python -m microbench.cli review-learned-bundle --bundle runs_learned_bundle --require-pass
```

## Known Public Alpha Limitations

- `orca_heuristic` and `orca_with_staleness` are geometric reference heuristics, not expert or certified DAA controllers.
- `cbf_qp`, `mpc_local`, and `negotiation_yield` pass compact public-alpha promotion calibration, including 3D/degraded lanes, but still have stable-v1 promotion blockers; do not treat them as stable-v1 leaderboard anchors yet.
- `learned_tiny` is a frozen tiny learned-model fixture for adapter, disclosure, and CSV-plumbing tests; it is not a competitive learned DAA baseline.
- `cbf_qp` and `mpc_local` also pass the longer stable-metadata prep lanes in `baseline-review`, but they remain experimental until the reference-role decision, CBF validation, and MPC compute/stress characterization are stronger.
- `baseline-evidence` adds targeted CBF fallback/solver-status checks and dense-3D MPC profiling; it is evidence for review, not a stable-v1 promotion by itself.
- The PettingZoo/Gymnasium-style RL interface is available for public-alpha experimentation, but observation vectors and reward defaults are not stable-v1 contracts yet.
- `rl-smoke` checks wrapper API health and 2D/3D coverage, not policy quality or leaderboard safety.
- `rl-calibration` adds compact 3D/degraded wrapper exposure for learned-policy submissions, but it is not a leaderboard score.
- `rl-contract` publishes schema-versioned action, observation, and reward metadata for adapter authors, but those versions are still pre-v1.
- `rl-freeze-check` publishes a machine-readable stable-v1 readiness checklist for the RL interface, but passing it does not make this public alpha a stable v1 release.
- `tiny_learned` is available as a built-in RL smoke policy and maps to the same frozen model family as the planner method `learned_tiny`.
- `learned-submission-bundle` creates the standard learned-policy artifact folder, including RL contract/freeze/smoke/calibration reports and official planner CSVs.
- `validate-learned-bundle` reviews an existing learned-policy bundle without rerunning simulations and checks required artifacts, parseability, passing RL reports, planner acceptance, and nonempty planner CSVs.
- `review-learned-bundle` summarizes an existing learned-policy bundle into safety, mission, compute, communication, observation, and v0-score fields for manual leaderboard review.
- Learned-policy submissions should include `learned_submission_bundle.json` or equivalent `rl_contract.json`, `rl_freeze_check.json`, `rl_smoke.json`, `rl_calibration.json`, weight/version disclosures, and training scenario disclosure.
- The benchmark models local planning and simplified dynamics; it is not a full flight stack, airspace model, PX4/ROS simulator, or certification tool.
- Generated official suites are pre-v1 and may be adjusted as external users stress-test the benchmark.
- Leaderboard policy and scoring dimensions are documented, but public submissions should still be reviewed manually during alpha.

## Recommended Announcement Scope

Invite early users to:

- inspect the planner contract in `docs/DESIGN_V1.md` and `docs/PLANNER_API.md`
- run `official_smoke_generated` and one 3D suite
- submit planner/scenario/result feedback through GitHub issue templates
- report confusing metrics, missing docs, or suite cases that feel too easy or too artificial
