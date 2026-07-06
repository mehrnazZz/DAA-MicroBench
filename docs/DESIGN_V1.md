# DAA Microbench v1 Design Contract

This document defines the intended public contract for DAA Microbench v1.

DAA Microbench v1 is a fast, deterministic benchmark for local multi-drone detect-and-avoid planners under limited sensing, V2V impairment, intent sharing, and decentralized decision-making. It is designed to compare local planning behavior, not to replace high-fidelity flight simulation, certification analysis, or hardware validation.

## Scope

DAA Microbench evaluates planners that repeatedly receive one ego-local observation and return a desired velocity command. The simulator applies shared dynamics, communication, perception, collision, metric, and logging rules.

In scope:

- multi-agent spherical collision and near-miss evaluation
- planar and non-planar 3D scenarios
- per-agent planner instances and memory
- heterogeneous planners, priorities, capabilities, missions, and failure modes
- V2V odometry impairment, intent messages, and agent messages
- local sensor and fused sensor/V2V observation modes
- reproducible official suites, baselines, traces, metrics, and result schemas

Out of scope:

- full flight-stack, autopilot, ROS, PX4, or actuator modeling
- certified aircraft performance envelopes
- airspace procedure compliance beyond scenario-defined rules
- global trajectory planning as the primary task
- learned-policy training infrastructure beyond benchmark/dataset interfaces

## Planner Definition

A planner is a deterministic local policy implementation that satisfies the public `PlannerInput -> velocity command` contract. It may be classical, optimization-based, learned, heuristic, or hybrid.

Each drone owns one planner instance unless a user explicitly implements shared state outside the benchmark. Per-agent state should live in instance attributes or `planner_input.agent_context.memory`.

Public return forms:

- `np.ndarray` with finite shape `(3,)`
- `PlannerOutput(v_cmd=..., intent_out=..., messages_out=..., debug_info=...)`

The command is desired world-frame velocity in meters per second. The simulator applies shared speed and acceleration clamps.

## Allowed Information

Planners may use only information reachable from `PlannerInput`:

- `ego`: ego state, radius, limits, goal, and progress flags
- `goal_dir`: current world-frame unit direction to goal
- `neighbors`: selected local tracks after scenario perception/V2V/top-k rules
- `obstacles`: static AABB obstacles exposed to planners
- `neighbor_intents`: intent messages aligned with selected neighbors
- `messages`: delivered agent messages for the current tick
- `agent_context`: agent id, method, seed, role, priority, capabilities, mission, failure modes, and memory
- `dt`, `t`, and `planar`

Planners may also use their own fixed model weights, solver state, and deterministic random streams seeded through `reset`.

## Forbidden Information

Planners must not use:

- simulator ground truth for non-observed agents
- future states, future messages, or future collisions
- private engine objects, global scenario internals, recorder artifacts, traces, or result files during an episode
- method-specific changes to timestep, collision radius, communication profile, perception model, neighbor range, or top-k ranking
- different official scenario files for one method in a comparison

If a method needs extra information, model it as perception, V2V, intent, or agent messages so every method can be compared under explicit assumptions.

## Agentic Contract

Agentic in DAA Microbench means decentralized per-drone behavior with local state and imperfect information.

Required properties:

- one planner instance per drone
- per-agent lifecycle and deterministic per-agent seeds
- persistent per-agent memory
- heterogeneous methods and profiles in one episode
- delayed, lossy, rate-limited, replayable communication
- local perception and stale belief handling

Scenario authors can encode roles, priorities, capabilities, missions, and failure modes through `agents:` blocks.

## Scenarios And Suites

Official pre-v1 generated suites are listed by:

```bash
python -m microbench.cli list-suites
```

Generated suites materialize scenario YAML plus a `suite_manifest.yaml` that records run matrix and acceptance metadata. Submitted official-suite results should include this manifest.

Current suite categories:

- `official_smoke_generated`: tiny CI/public smoke suite
- `official_alpha`: pre-v1 mixed planar/3D suite
- `official_3d_stress`: volumetric and vertical 3D stress suite
- `official_agentic_stress`: heterogeneous-priority and noncooperative agentic stress suite
- `official_experimental_baselines`: calibration lane for experimental baselines

Custom suites are allowed, but must be labeled separately from official results.

## Metrics And Schema

Run directories include:

- `results.csv`: per-episode rows
- `summary.csv`: grouped leaderboard rows
- `result_schema.json`: schema id, version, and ordered fields

Safety is the first gate. Primary safety metrics are collision episode rate, unique collision pairs, collision pair ticks, time to first collision, and separation margins. Mission, observation, communication, compute, and planner guardrail metrics explain the tradeoffs behind safety.

Current result schema version: `0.4.0`.

Schema changes must update:

- `microbench/metrics/io.py`
- `golden/current_schema/`
- `docs/LEADERBOARD.md`
- `docs/RESULT_SUBMISSION.md`

Use:

```bash
python -m microbench.cli golden-current-schema --golden-dir golden/current_schema
```

## Guardrails

Planner guardrails make failures visible instead of silently crashing or corrupting dynamics.

- Exceptions and invalid outputs increment `planner_error_count`.
- Soft timeouts increment `planner_timeout_count`.
- Every guardrail replacement increments `planner_fallback_count`.
- Fallback commands are deterministic and move away from observed risk when possible.

Nonzero guardrail counts should be disclosed in submissions and treated as benchmark penalties.

## Reproducibility

An official result should include:

- DAA Microbench commit hash
- method name and version
- full command
- `results.csv`, `summary.csv`, `result_schema.json`
- generated `suite_manifest.yaml` when relevant
- hardware, Python version, dependencies, and learned-weight disclosure

Before submitting, run scenario validation, acceptance checks, and the current-schema golden check.

## What v1 Stability Means

For v1, the following should be stable:

- public planner input/output contract
- result schema versioning process
- official suite manifest format
- leaderboard interpretation policy
- guardrail semantics
- replay and golden fixture smoke checks

The benchmark may still grow new scenarios, baselines, optional interfaces, and learning wrappers after v1, but changes that affect comparability should be explicit and versioned.
