# Planner API Tutorial

This guide shows how to write a local planner for DAA Microbench.

For the design contract and fairness rules, read [DESIGN_V1.md](DESIGN_V1.md).

## Minimal Planner

A planner implements:

```python
def reset(self, seed: int) -> None:
    ...

def compute_cmd(self, planner_input: PlannerInput) -> np.ndarray | PlannerOutput:
    ...
```

The command must be finite shape `(3,)` in world-frame meters per second.

```python
from __future__ import annotations

import numpy as np
from microbench.types import PlannerInput


class MyPlanner:
    def reset(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def compute_cmd(self, planner_input: PlannerInput) -> np.ndarray:
        ego = planner_input.ego
        return np.asarray(planner_input.goal_dir, dtype=float) * float(ego.v_max)
```

## Rich Lifecycle

Newer planners may accept the richer reset signature:

```python
def reset(self, agent_id: int, seed: int, config: dict) -> None:
    self.agent_id = agent_id
    self.seed = seed
    self.role = config.get("role")
    self.priority = int(config.get("priority", agent_id))
```

The engine supports old and new signatures for compatibility. Each drone gets its own planner object.

## PlannerInput

Use only public `PlannerInput` fields:

- `ego`
- `goal_dir`
- `neighbors`
- `obstacles`
- `neighbor_intents`
- `messages`
- `agent_context`
- `dt`
- `t`
- `planar`

Do not read engine internals, trace files, result files, future state, or global truth for unobserved agents.

## Per-Agent Memory

Use `planner_input.agent_context.memory` for persistent state:

```python
def compute_cmd(self, planner_input):
    ctx = planner_input.agent_context
    if ctx is not None:
        ctx.memory["ticks"] = int(ctx.memory.get("ticks", 0)) + 1
```

This memory is keyed to one agent context and persists across ticks in the episode.

## Intent And Agent Messages

Return `PlannerOutput` when you need extra outputs:

```python
from microbench.types import PlannerOutput

return PlannerOutput(
    v_cmd=v_cmd,
    intent_out=None,
    messages_out=[],
    debug_info={"mode": "nominal"},
)
```

Intent messages go through the intent channel. Agent messages go through the rate-limited/delayed/lossy V2V message bus and appear in `PlannerInput.messages`.

## Guardrails

The benchmark validates planner output:

- return shape must be `(3,)`
- values must be finite
- exceptions are caught
- soft timeouts are counted after the call returns

Failures are scored through:

- `planner_timeout_count`
- `planner_error_count`
- `planner_fallback_count`

Guardrail replacements use deterministic fallback commands, so failed planners still produce complete result rows.

## Registering A Planner

The current CLI uses the built-in registry in `microbench/planners/__init__.py`.

To add a planner for local development:

1. Copy your planner module into `microbench/planners/`.
2. Import the class in `microbench/planners/__init__.py`.
3. Add a factory entry to `_FACTORIES`.
4. Add `PlannerMetadata` in `_METADATA`.
5. Run `python -m microbench.cli list-methods --json`.

For quick Python experiments without changing the registry, instantiate `EpisodeEngine` with a `planner_factory` callback. The official CLI and submitted benchmark results should use registered method names.

## Example Planner

See [examples/simple_external_planner.py](../examples/simple_external_planner.py). It is intentionally small and uses only public `PlannerInput` fields.

Smoke-test the example directly:

```bash
python -m pytest tests/test_public_docs_examples.py -q
```

After registration, run:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/corridor.yaml \
  --method your_method_name \
  --n 4 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_your_method_smoke
```

Then inspect `summary.csv` and confirm planner guardrail counts are zero.

## Heterogeneous Episodes

Run one planner per drone:

```bash
python -m microbench.cli run \
  --scenario config/scenarios/corridor.yaml \
  --method mixed \
  --agent-methods baseline_goal,template,baseline_goal,template \
  --n 4 \
  --seed 0 \
  --comm ideal_50hz \
  --out-dir runs_heterogeneous_example
```

When `--agent-methods` has `N` entries, each drone receives the method at the same index. The result row method label becomes `mixed[...]`.

## Submission Checklist

Before comparing or submitting a planner:

- run at least `official_smoke_generated`
- include `results.csv`, `summary.csv`, and `result_schema.json`
- disclose whether the method uses V2V, intent, agent messages, local sensing, learned weights, or external services
- include all failed runs
- disclose any nonzero guardrail counts
- follow [RESULT_SUBMISSION.md](RESULT_SUBMISSION.md)
