# ANSYS Vertical-Flap Validation Tools Package Goal 2026-06-30

## Objective

Plan a future migration that moves reusable ANSYS vertical-flap Fluent reference validation logic out of `validation_runs/.../scripts` into a stable tools package while keeping the current script paths as thin wrappers.

## Proposed Package

Target package:

`tools/validation/ansys_vertical_flap/`

Proposed modules:

- `fluent_reference_contract_schema.py`
- `fluent_source_export_schema.py`
- `fluent_reference_collection.py`
- `fluent_parity.py`
- `fluent_artifact_policy.py`

## Constraints

- Do not change artifact semantics during the migration.
- Keep `validation_runs/ansys_vertical_flap_fsi/scripts/*.py` as stable wrappers for existing workflow commands.
- Keep tests unchanged until the wrapper layer is in place.
- Do not move generated artifacts.
- Do not use the migration to alter solver physics or claim Fluent parity.

## Acceptance Criteria For The Future Migration

- Existing CLI commands continue to work.
- Existing workflow paths continue to work.
- Focused tests pass without needing artifact expectation rewrites.
- Wrapper scripts contain no business logic beyond argument parsing and delegation.
- The reusable tool package has direct unit coverage.
