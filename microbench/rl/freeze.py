from __future__ import annotations

from pathlib import Path
from typing import Any

from microbench.rl.schema import (
    RL_ACTION_SCHEMA_VERSION,
    RL_INTERFACE_VERSION,
    RL_OBSERVATION_SCHEMA_VERSION,
    RL_REWARD_SCHEMA_VERSION,
    action_schema,
    interface_contract,
    observation_schema,
    reward_schema,
)


RL_FREEZE_CHECK_SCHEMA_VERSION = "0.1"
RL_STABLE_V1_FREEZE_CRITERIA = (
    {
        "name": "versioned_contract",
        "description": "RL interface, action, observation, and reward schemas are versioned and emitted by rl-contract.",
    },
    {
        "name": "action_contract_frozen",
        "description": "Stable v1 action semantics keep normalized finite float32 desired-velocity actions with shape (3,) and bounds [-1, 1].",
    },
    {
        "name": "observation_contract_frozen",
        "description": "Stable v1 observation layout keeps fixed float32 local ego fields followed by padded top-k neighbor blocks.",
    },
    {
        "name": "no_privileged_observation",
        "description": "RL observations expose local planner information only, not simulator-wide privileged state.",
    },
    {
        "name": "reward_contract_documented",
        "description": "Default training reward terms are documented and clearly separated from leaderboard metrics.",
    },
    {
        "name": "wrapper_health_gates",
        "description": "rl-smoke, rl-calibration, and optional PettingZoo/Gymnasium integration checks are documented release gates.",
    },
    {
        "name": "adapter_example_available",
        "description": "A dependency-free learned-policy adapter example is available for external policy authors.",
    },
)


def _check(name: str, ok: bool, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {"name": name, "ok": bool(ok), "details": details or {}}


def run_rl_freeze_check(*, root: str | Path = ".") -> dict[str, Any]:
    """Return a machine-readable stable-v1 RL interface freeze checklist."""

    repo = Path(root)
    action = action_schema()
    observation = observation_schema(top_k=8)
    reward = reward_schema()
    contract = interface_contract(top_k=8)
    rl_doc = (repo / "docs" / "RL_INTERFACE.md")
    freeze_doc = (repo / "docs" / "RL_STABLE_V1_FREEZE.md")
    adapter_example = repo / "examples" / "rl_external_policy_adapter.py"
    release_checklist = repo / "docs" / "RELEASE_CHECKLIST.md"

    doc_text = rl_doc.read_text(encoding="utf-8") if rl_doc.exists() else ""
    freeze_text = freeze_doc.read_text(encoding="utf-8") if freeze_doc.exists() else ""
    checklist_text = release_checklist.read_text(encoding="utf-8") if release_checklist.exists() else ""

    checks = [
        _check(
            "versioned_contract",
            contract["interface_version"] == RL_INTERFACE_VERSION
            and action["schema_version"] == RL_ACTION_SCHEMA_VERSION
            and observation["schema_version"] == RL_OBSERVATION_SCHEMA_VERSION
            and reward["schema_version"] == RL_REWARD_SCHEMA_VERSION,
            {
                "interface_version": RL_INTERFACE_VERSION,
                "action_schema_version": RL_ACTION_SCHEMA_VERSION,
                "observation_schema_version": RL_OBSERVATION_SCHEMA_VERSION,
                "reward_schema_version": RL_REWARD_SCHEMA_VERSION,
            },
        ),
        _check(
            "action_contract_frozen",
            action["shape"] == [3]
            and action["dtype"] == "float32"
            and float(action["low"]) == -1.0
            and float(action["high"]) == 1.0
            and "desired" in str(action["semantics"]),
            {"action": action},
        ),
        _check(
            "observation_contract_frozen",
            observation["shape"] == [89]
            and observation["base_dim"] == 17
            and observation["neighbor_dim"] == 9
            and observation["top_k"] == 8
            and observation["layout"]["neighbors"][0] == 17,
            {"shape": observation["shape"], "base_dim": observation["base_dim"], "neighbor_dim": observation["neighbor_dim"]},
        ),
        _check(
            "no_privileged_observation",
            observation["privileged_global_state"] is False,
            {"privileged_global_state": observation["privileged_global_state"]},
        ),
        _check(
            "reward_contract_documented",
            bool(reward["terms"])
            and "leaderboard" in str(reward["leaderboard_note"]).lower()
            and "Default reward" in freeze_text,
            {"terms": [term["name"] for term in reward["terms"]]},
        ),
        _check(
            "wrapper_health_gates",
            "rl-smoke" in checklist_text
            and "rl-calibration" in checklist_text
            and "tests/test_rl_optional_integrations.py" in checklist_text,
        ),
        _check(
            "adapter_example_available",
            adapter_example.exists()
            and "ModelPredictPolicyAdapter" in adapter_example.read_text(encoding="utf-8"),
            {"path": str(adapter_example)},
        ),
        _check(
            "freeze_docs_linked",
            freeze_doc.exists()
            and "stable-v1 freeze criteria" in doc_text.lower()
            and "rl-freeze-check" in checklist_text,
            {"path": str(freeze_doc)},
        ),
    ]

    return {
        "schema_version": RL_FREEZE_CHECK_SCHEMA_VERSION,
        "interface_version": RL_INTERFACE_VERSION,
        "ok": all(check["ok"] for check in checks),
        "criteria": [dict(item) for item in RL_STABLE_V1_FREEZE_CRITERIA],
        "checks": checks,
    }
