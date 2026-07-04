# Acceptance Golden Fixtures

This folder stores small, path-independent acceptance report fixtures for generated suites.

`official_smoke_generated_acceptance.json` captures the expected rule set and pass counts for the generated smoke suite after running its default methods against its generated manifest.

The fixture intentionally omits machine-specific paths and observed timing values. It is meant to catch accidental rule drift, missing checks, and changed acceptance semantics while keeping runtime-sensitive values in the suite manifest thresholds.
