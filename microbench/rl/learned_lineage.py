from __future__ import annotations

import json
from pathlib import Path
from typing import Any


LEARNED_POLICY_LINEAGE_SCHEMA_VERSION = "0.1"

LEARNED_POLICY_LINEAGE_FIELDS = (
    "training_lineage",
    "lineage_label",
    "promotion_stage",
    "holdout_profile",
    "holdout_promotion_candidate",
    "training_recipe",
    "trainable_parameters",
    "sample_selection",
    "sample_weighting",
)


def _get(mapping: dict[str, Any], *keys: str) -> Any:
    current: Any = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _bundle_root(bundle: str | Path, review: dict[str, Any]) -> Path:
    raw = review.get("bundle_root")
    if raw:
        return Path(str(raw))
    path = Path(bundle)
    return path.parent if path.is_file() else path


def _policy_artifact_path(bundle: str | Path, review: dict[str, Any]) -> Path | None:
    artifacts = _get(review, "validation", "artifacts")
    if not isinstance(artifacts, dict):
        return None
    raw = artifacts.get("policy_artifact")
    if not raw:
        return None
    path = Path(str(raw))
    candidates = [path]
    if not path.is_absolute():
        candidates.append(_bundle_root(bundle, review) / path)
    for candidate in candidates:
        if candidate.exists() and candidate.is_file():
            return candidate
    return None


def _policy_model_payload(bundle: str | Path, review: dict[str, Any]) -> dict[str, Any]:
    path = _policy_artifact_path(bundle, review)
    if path is None:
        return {}
    try:
        payload = _read_json(path)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _mode(value: Any) -> str | None:
    if isinstance(value, dict):
        raw = value.get("mode")
        return None if raw is None else str(raw)
    if value is None:
        return None
    return str(value)


def classify_learned_policy_lineage(*, bundle: str | Path, review: dict[str, Any]) -> dict[str, Any]:
    """Classify a learned-policy bundle by training provenance and promotion evidence."""

    method = str(review.get("method") or "")
    policy = str(review.get("policy") or "")
    model = _policy_model_payload(bundle, review)
    training = model.get("training") if isinstance(model.get("training"), dict) else {}
    recipe = str(training.get("recipe") or "")
    holdout_result = training.get("holdout_result") if isinstance(training.get("holdout_result"), dict) else {}
    holdout_config = training.get("holdout") if isinstance(training.get("holdout"), dict) else {}
    holdout_profile = holdout_result.get("profile") or holdout_config.get("profile")
    holdout_promotion = holdout_result.get("promotion_candidate")
    sample_selection = _mode(training.get("sample_selection"))
    sample_weighting = _mode(training.get("sample_weighting"))

    if "learned-closed-loop-finetune" in recipe or policy == "closed_loop_mlp_learned":
        if holdout_promotion is True:
            lineage_label = "closed_loop_holdout_passed"
            promotion_stage = "holdout_passed"
        elif holdout_result:
            lineage_label = "closed_loop_holdout_review"
            promotion_stage = "holdout_review_required"
        else:
            lineage_label = "closed_loop_no_holdout"
            promotion_stage = "no_holdout"
        training_lineage = "closed_loop_finetuned"
    elif "learned-hard-lane-loop" in recipe or sample_selection not in {None, "all"} or policy == "bc_mlp_hard_lane":
        training_lineage = "hard_lane_behavior_cloned"
        lineage_label = "hard_lane_bc"
        promotion_stage = "not_applicable"
    elif "train-learned-bc" in recipe or policy.startswith("bc_mlp"):
        training_lineage = "behavior_cloned"
        lineage_label = "bc_only"
        promotion_stage = "not_applicable"
    elif method in {"learned_tiny", "learned_mlp"} or policy in {"tiny_learned", "mlp_learned"}:
        training_lineage = "frozen_fixture"
        lineage_label = "frozen_fixture"
        promotion_stage = "not_applicable"
    elif method == "learned_policy_spec" or _get(review, "submission_manifest", "policy", "policy_spec"):
        training_lineage = "external_policy_spec"
        lineage_label = "external_or_unknown"
        promotion_stage = "not_applicable"
    else:
        training_lineage = "unknown"
        lineage_label = "unknown"
        promotion_stage = "not_applicable"

    return {
        "schema_version": LEARNED_POLICY_LINEAGE_SCHEMA_VERSION,
        "training_lineage": training_lineage,
        "lineage_label": lineage_label,
        "promotion_stage": promotion_stage,
        "holdout_profile": holdout_profile,
        "holdout_promotion_candidate": None if holdout_promotion is None else bool(holdout_promotion),
        "training_recipe": recipe or None,
        "trainable_parameters": training.get("trainable_parameters"),
        "sample_selection": sample_selection,
        "sample_weighting": sample_weighting,
    }


__all__ = [
    "LEARNED_POLICY_LINEAGE_FIELDS",
    "LEARNED_POLICY_LINEAGE_SCHEMA_VERSION",
    "classify_learned_policy_lineage",
]
