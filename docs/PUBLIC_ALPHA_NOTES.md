# Public Alpha Notes

Status date: 2026-07-06

DAA Microbench is ready for public-alpha evaluation: the repository has a documented planner contract, generated 2D/3D/agentic scenario suites, baseline metadata, result-schema manifests, issue templates, package smoke checks, and GitHub Actions CI.

This is not yet a stable v1 release. The public contract is intended to be reviewable and usable, but official suite membership, acceptance bands, experimental baselines, and leaderboard scoring may still change before v1. In particular, `cbf_qp`, `mpc_local`, and `negotiation_yield` are runnable experimental baselines, not mature leaderboard anchors.

## Supported Environment

- Python 3.10, 3.11, and 3.12
- Core install: `pip install -e .`
- Result schema version: `0.4.0`
- License: Apache-2.0

## Public Alpha Suites

- `official_smoke_generated`: tiny 2D/3D/agentic smoke coverage for CI and quick checks.
- `official_alpha`: pre-v1 generated suite mixing planar and 3D DAA families.
- `official_3d_stress`: volumetric and vertical DAA stress cases.
- `official_agentic_stress`: heterogeneous priorities, noncooperative traffic, partial sensing, and agentic coordination.
- `official_experimental_baselines`: compact calibration suite for experimental CBF-QP and MPC-local baselines.
- `official_promotion_calibration`: compact 3D and degraded sensing/communication calibration suite used by `baseline-promotion`.

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
```

## Known Public Alpha Limitations

- `orca_heuristic` and `orca_with_staleness` are geometric reference heuristics, not expert or certified DAA controllers.
- `cbf_qp`, `mpc_local`, and `negotiation_yield` pass compact public-alpha promotion calibration, including 3D/degraded lanes, but still have stable-v1 promotion blockers; do not treat them as leaderboard anchors yet.
- The benchmark models local planning and simplified dynamics; it is not a full flight stack, airspace model, PX4/ROS simulator, or certification tool.
- Generated official suites are pre-v1 and may be adjusted as external users stress-test the benchmark.
- Leaderboard policy and scoring dimensions are documented, but public submissions should still be reviewed manually during alpha.

## Recommended Announcement Scope

Invite early users to:

- inspect the planner contract in `docs/DESIGN_V1.md` and `docs/PLANNER_API.md`
- run `official_smoke_generated` and one 3D suite
- submit planner/scenario/result feedback through GitHub issue templates
- report confusing metrics, missing docs, or suite cases that feel too easy or too artificial
