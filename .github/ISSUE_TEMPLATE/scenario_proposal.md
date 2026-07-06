---
name: Scenario proposal
about: Propose a new scenario or official-suite candidate
title: "[Scenario] "
labels: scenario
---

## Summary

- Scenario name:
- Proposed suite/category:
- Dimensions: 2D / 3D / mixed
- Primary stressor: crossing / merge / overtake / obstacle / comm / perception / agentic / other

## Purpose

What DAA behavior or failure mode should this scenario expose?

## Scenario Design

- Agent count(s):
- Seed range:
- Comm profile(s):
- Perception mode:
- Obstacles:
- Roles/priorities/capabilities/failure modes:

## Expected Baseline Behavior

Describe expected behavior for at least one built-in method.

## Validation

```bash
python -m microbench.cli validate-scenarios --scenario path/to/scenario.yaml
```

## Acceptance Criteria

If this should become an official-suite scenario, propose any acceptance checks or smoke bands.

## Artifacts

Attach scenario YAML, run command, and any relevant traces/results.

See `docs/SCENARIO_SUITES.md` and `docs/DESIGN_V1.md`.
