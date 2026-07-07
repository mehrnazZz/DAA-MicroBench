# RL Stable-V1 Freeze Criteria

The public-alpha RL interface is intentionally versioned before it is stable. Stable v1 means outside policy authors can train against the interface without silent observation, action, or reward changes.

Run the machine-readable readiness check:

```bash
python -m microbench.cli rl-freeze-check --require-pass --json
```

## Stable-v1 freeze criteria

- The `rl-contract` output is the source of truth for interface, action, observation, and reward schema versions.
- Action output remains a finite `float32` vector with shape `(3,)`, normalized bounds `[-1, 1]`, and desired world-frame velocity semantics.
- Observation output remains a finite `float32` local vector: 17 ego fields followed by padded top-k neighbor blocks of width 9.
- Observations do not expose privileged global simulator state.
- Neighbor fields remain ordered as present flag, relative position, relative velocity, neighbor radius, and message age.
- Default reward terms remain documented as progress, time, collision, near miss, and goal bonus.
- Default reward is a training convenience only; leaderboard comparisons continue to use benchmark metrics, `results.csv`, `summary.csv`, and schema sidecars.
- `rl-smoke`, `rl-calibration`, `rl-contract`, `rl-freeze-check`, release readiness, and optional `tests/test_rl_optional_integrations.py` are the required interface health gates.
- Dependency-free learned-policy adapter examples remain runnable from a source checkout.

## Compatibility policy

Before stable v1, schema versions may change while the interface is still public-alpha. After stable v1:

- Patch releases must not reorder, remove, or reinterpret observation fields.
- Patch releases must not change action shape, bounds, or semantics.
- Patch releases must not change default reward weights without a reward schema version bump.
- Additive observation fields require a new observation schema version and migration notes.
- Any breaking RL interface change requires a new interface version, release-note callout, and updated adapter examples.

## Required artifacts for learned-policy submissions

Learned-policy submissions should include:

- `learned_submission_bundle.json`
- `rl_contract.json`
- `rl_freeze_check.json`
- `rl_smoke.json`
- `rl_calibration.json`
- training-data and reward-shaping disclosure
- inference dependency and weight/version disclosure
- official benchmark `results.csv`, `summary.csv`, suite manifest, and result schema sidecar for leaderboard claims
