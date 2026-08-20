# Learned Policy Adoption

This guide shows the shortest path from an exported policy artifact to DAA Microbench planner CSVs.

## Public Contract

External learned policies should use only the public RL observation/action contract:

```bash
python -m microbench.cli rl-contract --json
```

The action is a finite normalized `(3,)` desired world-frame velocity. The simulator clips it to `[-1, 1]`, forces `y = 0` in planar scenarios, scales by each drone's `v_max`, and applies the normal dynamics limits.

## Example Files

The dependency-free examples are:

- `examples/exported_linear_policy.json`: tiny exported model metadata and coefficients.
- `examples/exported_policy.py`: inference wrapper with `predict(...)`, `make_model(...)`, and `callable_policy(...)`.
- `examples/external_policy_model_predict_spec.json`: `model_predict` spec using a Python factory and `factory_kwargs`.
- `examples/external_policy_callable_spec.json`: `callable` spec using `callable_policy(observation, info)`.
- `examples/rl_train_tiny_linear_policy.py` and `examples/rl_train_mlp_policy.py`: deterministic recipes for the bundled learned-policy fixtures.
- `python -m microbench.cli train-learned-bc`: behavior-cloned MLP trainer over real DAA Microbench RL validation-lane observations.
- `examples/learned_submission_manifest_template.json`: full reviewer-ready manifest template.
- `examples/learned_submission_manifest_overlay_example.json`: compact disclosure overlay for `--submission-manifest`.

The `model_predict` pattern is the recommended shape for real exported models because the factory can load weights, construct preprocessing objects, and own inference state:

```json
{
  "schema_version": "0.1",
  "policy_name": "external_model_predict_fixture",
  "adapter": "model_predict",
  "factory": "exported_policy:make_model",
  "pythonpath": ["."],
  "artifact_path": "exported_linear_policy.json",
  "factory_kwargs": {
    "artifact_path": "exported_linear_policy.json"
  },
  "deterministic": true,
  "clip": true
}
```

Relative `pythonpath`, `artifact_path`, and path-like `factory_kwargs` such as `artifact_path` resolve from the spec file. Import-based specs execute Python code, so only run specs from trusted sources.

The built-in behavior-cloning trainer writes an `mlp_json` spec that avoids custom import code:

```bash
python -m microbench.cli train-learned-bc \
  --out-dir runs_bc_mlp_policy \
  --lanes head_on,crossing,urban_obstacle,communication_delay,high_n_dense_merge \
  --eval-lanes head_on,crossing,urban_obstacle \
  --require-pass
```

Its output `runs_bc_mlp_policy/policy_spec.json` can be passed anywhere a learned policy spec is accepted. The generated `bc_training_report.json` records training lanes, seeds, sample counts, teacher policy, fit error, and validation-matrix evidence. The trained artifact declares its inference guardrail, currently a goal-direction forward-progress floor plus unit-norm clamp, so disclose it as behavior cloning with post-processing, not as privileged imitation from simulator truth.

To export reusable public observation/action shards for your own imitation-learning or offline-RL loop:

```bash
python -m microbench.cli learned-dataset-export \
  --out-dir runs_learned_dataset \
  --lanes head_on,crossing,urban_obstacle \
  --max-steps 64 \
  --save-replay \
  --require-pass
```

By default this uses the same transparent `bc_teacher` as the built-in trainer. Pass `--policy tiny_learned`, `--policy mlp_learned`, or `--policy-spec path/to/policy_spec.json` when you want shards from another policy. The export writes `learned_dataset_manifest.json`, `learned_dataset_episodes.csv`, compressed `shards/shard_*.npz`, and optional `replay/*.jsonl` files. Shards contain `observations`, `actions`, `next_observations`, rewards, termination flags, lane metadata, and collision/near-miss diagnostics under the public RL contract.

For an end-to-end development comparison, use:

```bash
python -m microbench.cli learned-bc-evidence \
  --out-dir runs_bc_mlp_evidence \
  --lanes head_on,crossing,urban_obstacle \
  --max-runs 1 \
  --require-pass
```

The evidence command trains the BC policy, generates a manifest overlay with training disclosure, packages the trained spec through `learned-submission-bundle`, packages `learned_tiny` and `learned_mlp` for side-by-side context, and writes learned leaderboard plus diagnostics JSON/CSV/Markdown reports. Keep the output directory with any learned-policy claim because the leaderboard and diagnostic rows are derived from those bundles.

To turn those diagnostic labels into a reproducible retraining slice:

```bash
python -m microbench.cli learned-hard-lane-loop \
  --out-dir runs_hard_lane_loop \
  --diagnostics runs_bc_mlp_evidence/learned_policy_diagnostics.json \
  --target-policy bc_mlp_learned \
  --fallback-lanes urban_obstacle,communication_delay,high_n_dense_merge \
  --mix-lanes head_on,crossing,urban_obstacle,communication_delay,high_n_dense_merge,dense_swarm_hard_negative \
  --sample-weighting safety \
  --sample-selection hard_negative_windows \
  --max-lanes 3 \
  --max-runs 1 \
  --require-pass
```

The hard-lane loop selects canonical validation lanes from `unsafe`, `needs_training`, `fast_but_close`, `safe_but_slow`, and limited-evidence diagnostics, exports `learned-dataset-export` shards, trains the BC MLP from those shards, packages the trained policy, and writes a fresh learned leaderboard plus diagnostics report. Use `--target-policy` when the diagnostics file contains comparison fixtures but you only want to retrain one policy; `--fallback-lanes` can fill the remaining hard-lane budget with richer 3D/degraded lanes, `--mix-lanes` adds broad replay lanes so focused retraining does not erase general behavior, and `--sample-weighting safety` emphasizes collision, near-miss, and low-clearance samples in the supervised fit. Add `dense_swarm_hard_negative` to `--mix-lanes` only when you intentionally want the generated dense 3D swarm hard-negative training lane; it is not part of the default validation matrix. Use `--sample-selection hard_negative_windows` with that lane to keep only its hard-event or closest-approach temporal windows. New BC artifacts store per-feature mean/std normalization plus the sample-weighting and sample-selection recipes by default.

For a closed-loop learned-policy iteration path, start from any portable `mlp_json` policy and run:

```bash
python -m microbench.cli learned-closed-loop-finetune \
  --out-dir runs_closed_loop_finetune \
  --base-policy-spec runs_bc_mlp_policy/policy_spec.json \
  --lanes head_on,crossing,urban_obstacle,communication_delay,high_n_dense_merge \
  --trainable-parameters all_layers \
  --generations 4 \
  --population-size 12 \
  --train-max-steps 24 \
  --require-pass
```

The fine-tuner evaluates candidates inside `DaaParallelEnv`, records every candidate rollout in CSV, and writes a new portable `mlp_json` policy only after applying collision, clearance, and near-miss guardrails. Use `--trainable-parameters output_head` for conservative audits and `all_layers` for stronger learned-baseline development. Treat this as the benchmark-native closed-loop training spine for stronger learned baselines; it is still public-alpha infrastructure.

## Health Gates

Smoke-test the wrapper API on 2D and 3D generated scenarios:

```bash
python -m microbench.cli rl-smoke \
  --out-dir runs_external_model_predict_smoke \
  --policy-spec examples/external_policy_model_predict_spec.json \
  --max-steps 3 \
  --require-pass
```

Run compact 3D/degraded calibration:

```bash
python -m microbench.cli rl-calibration \
  --out-dir runs_external_model_predict_calibration \
  --policy-spec examples/external_policy_model_predict_spec.json \
  --max-steps 3 \
  --require-pass
```

Run the learned-policy validation matrix on the same canonical encounter families used for baseline validation:

```bash
python -m microbench.cli rl-validation-matrix \
  --out-dir runs_external_model_predict_validation_matrix \
  --policy-spec examples/external_policy_model_predict_spec.json \
  --max-steps 3 \
  --require-pass
```

The matrix covers head-on, crossing, 3D urban obstacle, communication-delay, and high-N dense-merge lanes. It is an interface/rollout health artifact first; use benchmark planner CSVs and leaderboard reports for final policy quality claims.

## Planner CSVs

Evaluate the same spec through the standard planner path:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/stacked_swap_3d.yaml \
  --method learned_policy_spec \
  --policy-spec examples/external_policy_model_predict_spec.json \
  --n 4 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_external_model_predict_planner
```

For official generated suites, use the bundle command. It writes planner CSVs, RL smoke/calibration/validation-matrix reports, acceptance output, a portable `policy_spec.json`, and copied `policy_artifacts/` when the spec declares a file artifact:

```bash
python -m microbench.cli learned-submission-bundle \
  --out-dir runs_external_model_predict_bundle \
  --method learned_policy_spec \
  --policy-spec examples/external_policy_model_predict_spec.json \
  --max-runs 1 \
  --max-steps 3 \
  --require-pass
```

For real submissions, pass a JSON disclosure overlay to populate the generated `learned_submission_manifest.json`. Start with `examples/learned_submission_manifest_overlay_example.json` when you only need to fill disclosure fields:

```json
{
  "training_disclosure": {
    "training_suites": ["my_training_suite"],
    "environment_steps": 1000000,
    "observation_normalization": "running mean/std fitted on training rollouts",
    "reward_configuration": {"progress": 1.0, "collision": -10.0},
    "external_data": "none",
    "pretrained_models": "none",
    "hardware": "1x local GPU"
  },
  "inference_disclosure": {
    "uses_external_services": false,
    "runtime_notes": "deterministic CPU inference"
  },
  "dependencies": {
    "inference_packages": [
      {"name": "numpy", "version": ">=1.24"}
    ]
  },
  "review_notes": {
    "privileged_information": "none"
  }
}
```

The fuller reviewer-ready shape is available at `examples/learned_submission_manifest_template.json`, and the packaged schema reference is in [LEARNED_SUBMISSION_SCHEMAS.md](LEARNED_SUBMISSION_SCHEMAS.md). Validate the schema/docs release gate and then validate a full draft before running the heavier bundle command:

```bash
python -m microbench.cli learned-submission-schema-check --require-pass
```

```bash
python -m microbench.cli validate-learned-manifest \
  --manifest examples/learned_submission_manifest_template.json \
  --require-pass
```

Use `--allow-undisclosed` only while drafting. A reviewer-ready manifest should declare inference package names and versions/specifiers, deterministic/runtime behavior, training suites or scenarios, seeds, reward configuration, observation normalization, external data, pretrained models, and any privileged-information caveats. `--submission-manifest` accepts an overlay that is merged into the generated manifest; `validate-learned-manifest` expects the full manifest shape.

```bash
python -m microbench.cli learned-submission-bundle \
  --out-dir runs_external_model_predict_bundle \
  --method learned_policy_spec \
  --policy-spec examples/external_policy_model_predict_spec.json \
  --submission-manifest path/to/submission_manifest_overrides.json \
  --max-runs 1 \
  --max-steps 3 \
  --require-pass
```

Validate and summarize the bundle:

```bash
python -m microbench.cli validate-learned-bundle \
  --bundle runs_external_model_predict_bundle \
  --require-pass

python -m microbench.cli review-learned-bundle \
  --bundle runs_external_model_predict_bundle \
  --out runs_external_model_predict_bundle/learned_bundle_review.json \
  --require-pass
```

Compare two or more learned-policy bundles as a development leaderboard table:

```bash
python -m microbench.cli learned-leaderboard \
  --bundle runs_learned_bundle \
  --bundle runs_external_model_predict_bundle \
  --out runs_learned_leaderboard/learned_policy_leaderboard.json \
  --require-pass

python -m microbench.cli learned-diagnostics \
  --bundle runs_learned_bundle \
  --bundle runs_external_model_predict_bundle \
  --out runs_learned_diagnostics/learned_policy_diagnostics.json \
  --require-pass
```

The leaderboard command writes JSON plus a sibling CSV and combines planner `summary.csv` score fields with RL validation-matrix lane evidence. It also reports true row-level clearance (`min_sep_min_row_m`, `min_sep_p05_row_min_m`) next to summary-derived clearance (`min_sep_min_summary_mean_min_m`, `min_sep_p05_summary_mean_min_m`) so a single close or colliding run is visible during review. The diagnostics command writes JSON/CSV/Markdown labels such as `safe_but_slow`, `fast_but_close`, and `balanced`, including the weakest scenario/lane and next suggested action. Treat both as reviewer evidence; final learned-policy claims should still include the underlying bundle artifacts.

## Submission Manifest Checklist

For review, include:

- exact DAA Microbench commit and policy source commit
- `learned_submission_manifest.json`
- `policy_spec.json` and any `policy_artifacts/`
- inference dependency versions and whether inference is deterministic
- training scenarios/suites, seeds, number of environment steps, reward configuration, and observation normalization
- `rl_contract.json`, `rl_freeze_check.json`, `rl_smoke.json`, `rl_calibration.json`, `rl_validation_matrix.json`
- `rl_validation_matrix/rl_validation_matrix_episodes.csv`
- planner `results.csv`, `summary.csv`, `result_schema.json`, generated `suite_manifest.yaml`, and `acceptance.json`
- reviewer output from `review-learned-bundle`

Do not use simulator truth outside the public observation/info surfaces. If a policy uses extra state, global positions not present in the observation, or offline labels from a privileged simulator, disclose that clearly and do not compare it as a standard local DAA planner.
