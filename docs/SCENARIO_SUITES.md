# Scenario Suites

DAA Microbench separates generated official suites, legacy hand-written suites, development suites, and custom experiments. This page is the human-readable registry; the CLI source of truth is:

```bash
python -m microbench.cli list-suites
python -m microbench.cli list-suites --json
```

The JSON form includes machine-readable `acceptance` metadata for generated suites.
See [BASELINES.md](BASELINES.md) for baseline roles and recommended method sets.

## Registry Status

| Status | Meaning |
|---|---|
| `pre_v1_official` | Generated suites intended for public alpha comparisons. APIs and exact run matrices may still change before v1. |
| `legacy_official` | Older hand-written canonical suite retained for continuity. |
| `development` | Useful for debugging or focused development, but not a single leaderboard ranking. |
| `smoke` | Fast sanity check suite for local regression checks. |
| `custom` | User-defined scenarios or modified official scenarios. Label results clearly. |

## Generated Official Suites

### `official_smoke_generated`

Fast generated smoke suite for CI, local install checks, and public examples. It intentionally covers one planar case, one true 3D volumetric case, and one agentic heterogeneous-priority case with a tiny default run matrix.

Families:
- `head_on_2d_easy`
- `sphere_swap_3d_medium`
- `heterogeneous_priority_crossing_3d_medium`

Default matrix:
- methods: `baseline_goal`, `orca_heuristic`, `priority_yield`
- N: `4`
- seeds: `0`
- comm: `ideal_50hz`
- generated scenario duration override: `8.0s`

### `official_alpha`

Mixed planar and non-planar public-alpha suite. It includes simple planar conflicts plus 3D stress cases so new planners are not overfit to flat geometry.

Families:
- `head_on_2d_easy`
- `crossing_2d_medium`
- `funnel_2d_hard`
- `sphere_swap_3d_medium`
- `merge_3d_hard`
- `vertical_crossing_3d_hard`
- `heterogeneous_priority_crossing_3d_medium`
- `sensor_volume_3d_hard`

### `official_3d_stress`

Generated 3D suite for volumetric, vertical, partial-sensing, merge, overtake, and noncooperative stress testing.
Default methods: `orca_heuristic`, `orca_with_staleness`.

Families:
- `sphere_swap_3d_medium`
- `merge_3d_hard`
- `overtake_3d_medium`
- `vertical_crossing_3d_hard`
- `sensor_volume_3d_hard`
- `noncooperative_intruder_3d_hard`
- `heterogeneous_priority_crossing_3d_medium`

### `official_agentic_stress`

Generated 3D suite focused on decentralized/agentic behavior: heterogeneous priorities, noncooperative traffic, partial sensing, intent/messages, and degraded communication.
Default methods: `priority_yield`, `negotiation_yield`, `orca_heuristic`, `orca_with_staleness`.

Families:
- `heterogeneous_priority_crossing_3d_medium`
- `noncooperative_intruder_3d_hard`
- `sensor_volume_3d_hard`
- `vertical_crossing_3d_hard`

## Hand-Written Suites

| Suite | Status | Purpose |
|---|---|---|
| `primary` | `legacy_official` | Older planar canonical suite retained for continuity. |
| `baseline_sanity` | `smoke` | Fast planar sanity check for baseline behavior. |
| `three_d` | `development` | Hand-written 3D scenarios for non-planar debugging. |
| `perception_stress` | `development` | Sensor-only and fused-perception stress tests. |

## Acceptance Metadata

Generated suite manifests include:

```yaml
acceptance:
  schema_version: "0.1"
  rules:
    - name: orca_heuristic_smoke_runtime
      scope: summary
      method: orca_heuristic
      metric: planner_ms_p95
      operator: <=
      value: 25.0
      severity: smoke
```

The validator checks rule schema, operator names, and metric names against `summary.csv` / `results.csv` fields. Evaluate the rules against a run with:

```bash
python -m microbench.cli check-acceptance \
  --summary runs_official_smoke_generated/summary.csv \
  --results runs_official_smoke_generated/results.csv \
  --suite-manifest runs_official_smoke_generated/_generated_scenarios/official_smoke_generated/suite_manifest.yaml
```

Use `--methods`, `--scenarios`, `--comm-profiles`, or `--n` when a run intentionally covers only part of a suite. `required` and `smoke` rule failures exit nonzero; `warning` and `informational` rule violations are reported without failing the command.

`official_smoke_generated` includes calibrated smoke bands for baseline runtime, ORCA runtime, priority-yield message delivery, and zero planner guardrail events. The expected path-independent acceptance report lives in `golden/acceptance/official_smoke_generated_acceptance.json`.
`official_3d_stress` includes pre-v1 informational checks for the `orca_heuristic` and `orca_with_staleness` 3D reference rows.

`orca_heuristic` is the canonical ORCA-like reference baseline name. `orca_expert` is still accepted by the planner registry as a compatibility alias for older scripts and legacy result folders.
Use `orca_with_staleness` when you want an explicit stale-aware ORCA-like preset for degraded communication or stale sensor-track comparisons.

## Validation

Validate built-in and generated suites before submitting results:

```bash
python -m microbench.cli validate-scenarios \
  --all-builtins \
  --all-generated-suites
```

Generated suites write portable scenario YAMLs and `suite_manifest.yaml`:

```bash
python -m microbench.cli materialize-suite \
  --suite official_smoke_generated \
  --out-dir generated_official_smoke \
  --print-plan
```

For generated official suites, include `_generated_scenarios/<suite>/suite_manifest.yaml` with submitted results.
