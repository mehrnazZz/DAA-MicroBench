# Learned Submission Schemas

DAA Microbench publishes machine-readable JSON Schemas for learned/RL policy submission artifacts. The bundled schemas live under `microbench/bundled_config/schemas/` and are included in packaged installs.

## Schema Files

- `learned_submission_manifest.schema.json`: full `learned_submission_manifest.json` disclosure/provenance file.
- `learned_submission_bundle.schema.json`: top-level `learned_submission_bundle.json` report written by `learned-submission-bundle`.
- `learned_bundle_review.schema.json`: reviewer summary written by `review-learned-bundle`.

All three schemas currently use artifact schema version `0.1`. The JSON Schema `$schema` declaration is Draft 2020-12.

## Full Manifest Vs Overlay

`validate-learned-manifest` expects a full reviewer-ready `learned_submission_manifest.json`. Start from:

```bash
examples/learned_submission_manifest_template.json
```

The `learned-submission-bundle --submission-manifest` flag accepts an overlay file, not necessarily a full manifest. The overlay is deep-merged into the generated manifest so users can fill disclosures without duplicating generated artifact hashes and benchmark metadata. Use `examples/learned_submission_manifest_overlay_example.json` as a compact overlay starting point.

Recommended flow:

```bash
python -m microbench.cli learned-submission-schema-check --require-pass

python -m microbench.cli validate-learned-manifest \
  --manifest examples/learned_submission_manifest_template.json \
  --require-pass

python -m microbench.cli learned-submission-bundle \
  --out-dir runs_learned_bundle \
  --method learned_policy_spec \
  --policy-spec path/to/policy_spec.json \
  --submission-manifest path/to/submission_manifest_overrides.json \
  --require-pass

python -m microbench.cli validate-learned-bundle \
  --bundle runs_learned_bundle \
  --require-pass

python -m microbench.cli review-learned-bundle \
  --bundle runs_learned_bundle \
  --require-pass
```

## Compatibility Policy

For schema `0.1`:

- Required top-level fields should remain stable during public alpha unless a schema version is bumped.
- New optional fields may be added.
- Validators should tolerate unknown fields for forward-compatible review metadata.
- Legacy bundles that predate `learned_submission_manifest.json` remain structurally valid, but reviewer output flags `legacy_bundle_without_submission_manifest`.
- Reviewer-ready manifests should not leave material training, inference, dependency, external-service, or privileged-information fields as `undisclosed`.

## Schema Changelog

### 0.1

- Added `learned_submission_manifest.schema.json` for reviewer-ready learned-policy disclosure.
- Added `learned_submission_bundle.schema.json` for bundle reports produced by `learned-submission-bundle`.
- Added `learned_bundle_review.schema.json` for reviewer summaries produced by `review-learned-bundle`.
- Added `learned-submission-schema-check` as the release gate for schema packaging, version constants, docs, template validity, and overlay guidance.

## Programmatic Use

```python
from microbench.rl import (
    LEARNED_SUBMISSION_MANIFEST_SCHEMA_FILE,
    load_submission_schema,
    validate_with_schema_subset,
)

schema = load_submission_schema(LEARNED_SUBMISSION_MANIFEST_SCHEMA_FILE)
errors = validate_with_schema_subset(payload, schema)
```

The internal checker validates the JSON Schema subset used by the bundled schemas. Projects that already depend on a full JSON Schema implementation can also validate these files directly with Draft 2020-12 tooling.
