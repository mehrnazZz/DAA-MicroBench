# Benchmark Result Submission

Use this template when submitting DAA Microbench results.

Submissions are reviewed against the public benchmark contract in [DESIGN_V1.md](DESIGN_V1.md) and the planner API rules in [PLANNER_API.md](PLANNER_API.md).

## Required Summary

```text
Method:
Method version / commit:
DAA Microbench commit:
Suite:
Command:
Hardware:
Python version:
Dependencies:
Uses learned weights: yes/no
Uses V2V: yes/no
Uses intent: yes/no
Uses agent messages: yes/no
Uses local sensing: yes/no
Uses RL interface: yes/no
```

## Required Artifacts

Attach or link:

- `results.csv`
- `summary.csv`
- `result_schema.json`
- `_generated_scenarios/<suite>/suite_manifest.yaml` when using a generated official suite
- any changed scenario/config files
- any planner source or package version needed to reproduce
- `learned_submission_bundle.json` for learned/RL policy submissions when using the bundle command
- `learned_submission_manifest.json` for learned/RL policy provenance, dependencies, artifact hashes, and training/inference disclosure
- `learned_bundle_review.json` for learned/RL policy submissions when using the reviewer command
- `policy_spec.json` and `policy_artifacts/` for learned/RL policy submissions loaded through `--policy-spec`
- `rl_contract.json`, `rl_freeze_check.json`, `rl_smoke.json`, and `rl_calibration.json` for learned/RL policy submissions
- representative failure traces for nonzero collision or near-miss-heavy results

`result_schema.json` is generated next to the CSV files and records the explicit result schema version plus the ordered `results.csv` and `summary.csv` fields. Results without this sidecar should be treated as legacy or unofficial unless the benchmark commit unambiguously identifies the schema.

Run validation before submitting:

```bash
python -m microbench.cli validate-scenarios \
  --scenario path/to/custom_or_official.yaml \
  --suite-manifest path/to/suite_manifest.yaml
```

Run acceptance checks for generated suites:

```bash
python -m microbench.cli check-acceptance \
  --summary path/to/summary.csv \
  --results path/to/results.csv \
  --suite-manifest path/to/suite_manifest.yaml
```

For learned/RL policies, validate your disclosure draft and then build the standard artifact bundle:

```bash
python -m microbench.cli validate-learned-manifest \
  --manifest path/to/learned_submission_manifest.json \
  --require-pass
```

```bash
python -m microbench.cli learned-submission-bundle \
  --out-dir runs_learned_bundle \
  --method learned_policy_spec \
  --policy-spec path/to/policy_spec.json \
  --submission-manifest path/to/submission_manifest_overrides.json \
  --require-pass
```

Use `--method learned_policy_spec --policy-spec ...` when your learned policy should be evaluated through the standard planner-sweep CSV path. The same spec is also used for RL wrapper smoke/calibration artifacts in the bundle. Use `--submission-manifest` to fill in training/inference disclosures; otherwise the generated manifest marks unknown fields as `undisclosed` for reviewer follow-up. Start from `examples/learned_submission_manifest_template.json` when possible. See [LEARNED_POLICY_ADOPTION.md](LEARNED_POLICY_ADOPTION.md) for a concrete exported-policy example using `model_predict`, `callable`, copied policy artifacts, validation, and reviewer summaries.

Then validate the saved bundle before attaching it:

```bash
python -m microbench.cli validate-learned-bundle \
  --bundle runs_learned_bundle \
  --require-pass
```

Generate the reviewer summary to attach or paste into the submission:

```bash
python -m microbench.cli review-learned-bundle \
  --bundle runs_learned_bundle \
  --out runs_learned_bundle/learned_bundle_review.json \
  --require-pass
```

## Reproduction Command

Paste the exact command. Example:

```bash
python -m microbench.cli canonical-sweep \
  --suite official_alpha \
  --methods your_method \
  --out-dir runs_your_method_official_alpha
```

Use `python -m microbench.cli list-suites` to confirm suite status, source, and default run matrix.
Use `python -m microbench.cli list-suites --json` to inspect generated-suite acceptance rules.
Use `python -m microbench.cli list-methods --json --include-aliases` or [BASELINES.md](BASELINES.md) to confirm whether a submitted method is a canonical baseline or compatibility alias.
Generated suite manifests preserve the same `acceptance` metadata and should be submitted with results.

## Result Tables

Include the relevant rows from `summary.csv`. At minimum include:

- `collision_episode_rate`
- `unique_collision_pairs_mean`
- `collision_pair_ticks_mean`
- `min_sep_p05_mean`
- `completion_rate_mean`
- `mean_time_to_goal_mean`
- `deadlock_time_pct_mean`
- `planner_ms_p95`
- `planner_timeout_count_mean`
- `planner_error_count_mean`
- `planner_fallback_count_mean`
- `obs_neighbors_mean`
- `obs_v2v_fraction_mean`
- `obs_sensor_fraction_mean`
- `obs_stale_fraction_mean`
- `obs_empty_fraction_mean`

## Disclosure Checklist

- [ ] I did not use simulator ground truth outside `PlannerInput`.
- [ ] I did not change shared neighbor, collision, dynamics, comm, or perception settings for only my method.
- [ ] I included all runs, including failures.
- [ ] I disclosed learned weights or external services.
- [ ] For learned/RL policies, I included the learned submission bundle or the equivalent RL contract, freeze check, smoke report, and calibration report.
- [ ] For learned/RL policies, I included `learned_bundle_review.json` or pasted the reviewer summary.
- [ ] For learned/RL policies, I ran `validate-learned-manifest --manifest path/to/manifest.json --require-pass` after filling disclosure fields.
- [ ] For learned/RL policies, I ran `validate-learned-bundle --bundle runs_learned_bundle --require-pass` or validated the equivalent artifacts manually.
- [ ] For learned/RL policies, I ran `review-learned-bundle --bundle runs_learned_bundle --require-pass` or included an equivalent safety/mission/compute summary.
- [ ] For learned/RL policies, I reviewed `learned_submission_manifest.json` and filled in training/inference disclosure fields rather than leaving material fields `undisclosed`.
- [ ] I included enough config and command detail to reproduce the result.

## Notes

Use [docs/LEADERBOARD.md](LEADERBOARD.md) for ranking policy and interpretation. Learned-policy submissions can use the dedicated GitHub issue template for training, weights, and RL-contract disclosures.
