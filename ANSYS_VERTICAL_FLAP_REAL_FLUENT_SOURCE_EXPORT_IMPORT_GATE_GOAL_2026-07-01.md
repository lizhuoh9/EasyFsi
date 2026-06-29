# ANSYS Vertical Flap Real Fluent Source Export Import Gate Goal - 2026-07-01

## Objective

Add a machine-readable real Fluent source export import gate before importing
any official Fluent data or running any EasyFsi solver comparison.

The current repository does not contain the complete provenance-backed Fluent
source exports needed to promote the ANSYS vertical-flap reference contract. The
only committed `source_exports` files are schema-only CSVs plus incomplete
metadata, and `official_web/fluent_tip_displacement_web_final.csv` is public web
evidence rather than a complete Fluent report export. This task must not invent
missing Fluent data.

The implementation must make the current state explicit:

- real Fluent import is blocked;
- collection remains pending/incomplete;
- active manifest promotion remains blocked;
- solver evaluation must not be treated as ready;
- Fluent parity remains unclaimed.

## Branch

- Current branch:
  `codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`
- Current base lineage:
  `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- This branch currently contains:
  - `dd51edd test: add fluent source provenance reproducer`
  - `641b4de fix: reject non-fluent source export provenance`

## In Scope

- Add a structured `real_fluent_import_gate` payload to the Fluent reference
  collection validator output.
- Add focused integration tests for the gate:
  - current committed schema-only exports are blocked;
  - explicit synthetic/test-source allowance cannot count as real Fluent import
    readiness;
  - a temp-only, provenance-complete Fluent-style source bundle can make the gate
    ready without touching repo artifacts.
- Keep the existing collection validator, source export schema, and artifact
  policy semantics fail-closed.

## Out of Scope

- Do not import real Fluent source exports unless complete, provenance-backed
  files are actually available.
- Do not use public tutorial/web data as numeric parity truth.
- Do not run EasyFsi solver comparison as part of this task.
- Do not change solver physics, FSI coupling, pressure projection, marker force
  scatter, or any simulation-core behavior.
- Do not set `fluent_parity_claimed=true`.
- Do not loosen tolerances or complete the reference contract artificially.
- Do not migrate this branch onto the validation-tools package branch in this
  task; note the future forward-port requirement if that branch lands first.

## Required Gate Contract

The new gate should be emitted from
`validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py`
as `payload["real_fluent_import_gate"]`.

Required fields:

- `gate_schema_version`: stable string such as
  `ansys_vertical_flap_real_fluent_import_gate_v1`
- `status`: `ready_for_real_fluent_import` only when all checks are clear,
  otherwise `blocked_real_fluent_import_incomplete`
- `can_import_real_fluent_reference`: boolean
- `can_run_solver_evaluation`: boolean; true only when the import gate is ready
- `fluent_parity_claimed`: always false in this collection-stage gate
- `source_exports`: per-CSV status rows with artifact, metric group, source
  path, file status, final-step status, metric status, schema blockers, and
  readiness
- `metadata`: metadata file/provenance status, semantic mismatch list, and
  readiness
- `candidate_contract_status`
- `promotion_status`
- `blockers`: stable machine-readable blockers explaining why readiness is
  blocked

The gate must be blocked when:

- any required CSV is schema-only, missing, has missing final step, has missing
  metric values, or has disallowed source provenance;
- required metadata is incomplete;
- `allow_test_sources=True` was used, even if synthetic temp data made the
  candidate contract complete;
- the candidate contract is not `fluent_reference_complete`;
- the active manifest promotion is not `ready_for_versioned_contract_promotion`.

## TDD Plan

### Red

Add tests first. The initial RED should fail because `real_fluent_import_gate`
does not exist yet in collection payloads.

Test cases:

1. Current committed source exports:
   - run `run_with_paths(...)` against the committed `source_exports` root while
     writing diagnostics and active manifest to a temp directory;
   - assert `real_fluent_import_gate.status` is
     `blocked_real_fluent_import_incomplete`;
   - assert `can_import_real_fluent_reference=false`;
   - assert `can_run_solver_evaluation=false`;
   - assert source exports are not ready and metadata is not ready.

2. Synthetic/test-source allowance:
   - build complete synthetic temp source exports;
   - use a temp contract with complete tolerances;
   - run collection with `allow_test_sources=True`;
   - assert collection may be complete for temp-only comparison mechanics, but
     `real_fluent_import_gate` stays blocked by
     `test_source_allowance_enabled`.

3. Provenance-complete Fluent-style temp bundle:
   - build temp source CSVs with headers and finite final-step values;
   - rewrite final-step `source` values to a Fluent-style provenance string that
     does not contain disallowed terms;
   - write complete metadata with expected step/time and conventions;
   - use complete tolerances in a temp contract;
   - assert the gate is `ready_for_real_fluent_import`.

### Green

Implement the minimal collection-runner gate helper and include its output in
the payload. Keep existing field names and artifact semantics stable.

### Verification

Run:

```powershell
python -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\fluent_source_export_schema.py `
  tests\integration\test_ansys_vertical_flap_real_fluent_source_exports.py
```

Run focused tests:

```powershell
python -m unittest -v `
  tests.integration.test_ansys_vertical_flap_real_fluent_source_exports `
  tests.integration.test_ansys_vertical_flap_fluent_reference_synthetic_pipeline `
  tests.integration.test_ansys_vertical_flap_fluent_source_export_schema `
  tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts `
  tests.integration.test_ansys_vertical_flap_fluent_reference_contract_schema `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic `
  tests.integration.test_ansys_vertical_flap_fluent_artifact_policy
```

Run artifact and whitespace guards:

```powershell
git diff --check
git diff --name-only validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics
git diff --name-only validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics
```

No generated artifact roots should change unless the gate intentionally
regenerates diagnostics and the diff is reviewed.

## Completion Criteria

- A RED test demonstrates the missing `real_fluent_import_gate` field before the
  implementation.
- The gate is present after implementation and correctly blocks current
  schema-only committed exports.
- Synthetic/test-source allowance does not mark real import readiness.
- A complete Fluent-style temp bundle can mark import readiness without changing
  repo artifacts.
- Existing source export schema tests and Fluent parity fail-closed tests remain
  green.
- No solver code, generated Fluent values, or parity claim is changed.
- Commit and push only after local verification passes.
