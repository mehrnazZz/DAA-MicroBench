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

For official generated suites, use the bundle command. It writes planner CSVs, RL reports, acceptance output, a portable `policy_spec.json`, and copied `policy_artifacts/` when the spec declares a file artifact:

```bash
python -m microbench.cli learned-submission-bundle \
  --out-dir runs_external_model_predict_bundle \
  --method learned_policy_spec \
  --policy-spec examples/external_policy_model_predict_spec.json \
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

## Submission Manifest Checklist

For review, include:

- exact DAA Microbench commit and policy source commit
- `policy_spec.json` and any `policy_artifacts/`
- inference dependency versions and whether inference is deterministic
- training scenarios/suites, seeds, number of environment steps, reward configuration, and observation normalization
- `rl_contract.json`, `rl_freeze_check.json`, `rl_smoke.json`, `rl_calibration.json`
- planner `results.csv`, `summary.csv`, `result_schema.json`, generated `suite_manifest.yaml`, and `acceptance.json`
- reviewer output from `review-learned-bundle`

Do not use simulator truth outside the public observation/info surfaces. If a policy uses extra state, global positions not present in the observation, or offline labels from a privileged simulator, disclose that clearly and do not compare it as a standard local DAA planner.
