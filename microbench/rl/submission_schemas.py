from __future__ import annotations

import json
from importlib import resources
from typing import Any


LEARNED_SUBMISSION_MANIFEST_SCHEMA_FILE = "learned_submission_manifest.schema.json"
LEARNED_SUBMISSION_BUNDLE_SCHEMA_FILE = "learned_submission_bundle.schema.json"
LEARNED_BUNDLE_REVIEW_SCHEMA_FILE = "learned_bundle_review.schema.json"


def load_submission_schema(filename: str) -> dict[str, Any]:
    """Load a bundled learned-submission JSON Schema."""

    root = resources.files("microbench.bundled_config").joinpath("schemas")
    with root.joinpath(filename).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"schema must be a JSON object: {filename}")
    return payload


def validate_with_schema_subset(payload: Any, schema: dict[str, Any]) -> list[str]:
    """Validate the JSON-Schema subset used by bundled submission schemas.

    DAA Microbench keeps runtime dependencies small, so this covers the schema
    keywords we publish for learned-submission artifacts instead of depending
    on a full JSON Schema engine.
    """

    errors: list[str] = []

    def _type_name(value: Any) -> str:
        if value is None:
            return "null"
        if isinstance(value, bool):
            return "boolean"
        if isinstance(value, int):
            return "integer"
        if isinstance(value, float):
            return "number"
        if isinstance(value, str):
            return "string"
        if isinstance(value, list):
            return "array"
        if isinstance(value, dict):
            return "object"
        return type(value).__name__

    def _matches_type(value: Any, expected: str) -> bool:
        if expected == "null":
            return value is None
        if expected == "boolean":
            return isinstance(value, bool)
        if expected == "integer":
            return isinstance(value, int) and not isinstance(value, bool)
        if expected == "number":
            return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
        if expected == "string":
            return isinstance(value, str)
        if expected == "array":
            return isinstance(value, list)
        if expected == "object":
            return isinstance(value, dict)
        return True

    def _validate(value: Any, node: dict[str, Any], path: str) -> None:
        if "anyOf" in node:
            branch_errors: list[list[str]] = []
            for branch in node["anyOf"]:
                before = len(errors)
                _validate(value, branch, path)
                branch_errors.append(errors[before:])
                del errors[before:]
                if not branch_errors[-1]:
                    break
            else:
                errors.append(f"{path}: does not match any allowed schema")
            return

        if "const" in node and value != node["const"]:
            errors.append(f"{path}: expected constant {node['const']!r}, got {value!r}")
            return
        if "enum" in node and value not in node["enum"]:
            errors.append(f"{path}: expected one of {node['enum']!r}, got {value!r}")
            return

        expected_type = node.get("type")
        if expected_type is not None:
            expected = expected_type if isinstance(expected_type, list) else [expected_type]
            if not any(_matches_type(value, item) for item in expected):
                errors.append(f"{path}: expected type {'/'.join(expected)}, got {_type_name(value)}")
                return

        if isinstance(value, str):
            min_length = node.get("minLength")
            if min_length is not None and len(value) < int(min_length):
                errors.append(f"{path}: expected string length >= {min_length}")
            pattern = node.get("pattern")
            if pattern is not None:
                import re

                if re.match(str(pattern), value) is None:
                    errors.append(f"{path}: does not match pattern {pattern!r}")

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            minimum = node.get("minimum")
            if minimum is not None and value < minimum:
                errors.append(f"{path}: expected value >= {minimum}")

        if isinstance(value, list):
            min_items = node.get("minItems")
            if min_items is not None and len(value) < int(min_items):
                errors.append(f"{path}: expected at least {min_items} item(s)")
            item_schema = node.get("items")
            if isinstance(item_schema, dict):
                for idx, item in enumerate(value):
                    _validate(item, item_schema, f"{path}[{idx}]")

        if isinstance(value, dict):
            required = node.get("required", [])
            for key in required:
                if key not in value:
                    errors.append(f"{path}.{key}: required property is missing")

            properties = node.get("properties", {})
            for key, child_schema in properties.items():
                if key in value and isinstance(child_schema, dict):
                    _validate(value[key], child_schema, f"{path}.{key}")

            additional = node.get("additionalProperties", True)
            extra_keys = set(value) - set(properties)
            if additional is False:
                for key in sorted(extra_keys):
                    errors.append(f"{path}.{key}: additional property is not allowed")
            elif isinstance(additional, dict):
                for key in sorted(extra_keys):
                    _validate(value[key], additional, f"{path}.{key}")

    _validate(payload, schema, "$")
    return errors
