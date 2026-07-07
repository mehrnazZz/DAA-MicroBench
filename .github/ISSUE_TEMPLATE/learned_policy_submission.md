---
name: Learned policy submission
about: Submit or discuss a learned/RL policy for DAA Microbench
title: "[Learned Policy] "
labels: learned-policy, benchmark-result
---

## Summary

- Policy name:
- Training algorithm:
- Training code commit:
- DAA Microbench commit:
- Intended category: exploratory / external submission / candidate baseline

## RL Interface Contract

- [ ] I ran `python -m microbench.cli rl-contract --json` and used the reported action/observation/reward schema.
- [ ] I ran `python -m microbench.cli rl-freeze-check --require-pass --json` and attached the report.
- [ ] I ran `python -m microbench.cli rl-smoke --out-dir runs_rl_smoke --require-pass`.
- [ ] I ran `python -m microbench.cli rl-calibration --out-dir runs_rl_calibration --require-pass`.
- [ ] I ran `python -m microbench.cli learned-submission-bundle --out-dir runs_learned_bundle --method <method> --policy <policy> --require-pass` or attached equivalent artifacts.
- [ ] I did not use simulator ground truth outside the public RL observation and info surfaces.
- [ ] I disclosed any reward shaping beyond the default public-alpha reward.

## Training Disclosure

- Training scenarios/suites:
- Number of environment steps:
- Random seeds:
- Observation normalization:
- Action post-processing:
- Reward configuration:
- External simulators, datasets, or pretrained models:
- Hardware used for training:

## Weights And Dependencies

- Weight artifact location/version:
- Inference dependencies:
- Uses external services at inference time: yes/no
- Deterministic inference with fixed seed: yes/no

## Evaluation Artifacts

Attach or link:

- `results.csv`
- `summary.csv`
- `result_schema.json`
- `_generated_scenarios/<suite>/suite_manifest.yaml`
- `learned_submission_bundle.json`
- `rl_contract.json`
- `rl_freeze_check.json`
- `rl_smoke.json`
- `rl_calibration.json`
- representative traces for collisions, near misses, deadlocks, or planner guardrail events

## Commands

```bash
# training command

# evaluation command

# validation commands
python -m microbench.cli validate-scenarios ...
python -m microbench.cli check-acceptance ...
python -m microbench.cli rl-freeze-check --require-pass --json
python -m microbench.cli rl-smoke --out-dir runs_rl_smoke --require-pass
python -m microbench.cli rl-calibration --out-dir runs_rl_calibration --require-pass
python -m microbench.cli rl-contract --json
python -m microbench.cli learned-submission-bundle --out-dir runs_learned_bundle --method <method> --policy <policy> --require-pass
```

## Notes

See `docs/RL_INTERFACE.md`, `docs/RL_STABLE_V1_FREEZE.md`, `docs/RESULT_SUBMISSION.md`, and `docs/LEADERBOARD.md`.
