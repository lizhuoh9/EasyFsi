# ANSYS Vertical Flap Metadata Public Tutorial Provenance Guard Goal - 2026-07-01

## Objective

Close the remaining metadata provenance gap before any real Fluent source export
import: metadata that points at public tutorial pages, web contour screenshots,
or official web baseline artifacts must not satisfy `real_fluent_import_gate`,
even when CSV `source` fields look like real Fluent run exports.

This is a small hardening task on the existing official Fluent evaluation
branch. It must preserve the current fail-closed behavior and must not import or
invent Fluent reference data.

## Current Context

Branch:

`codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`

Current remote HEAD before this task:

`4d1eca25cc7072e1e504fb82a2204961f52d8e13`

Existing safeguards already present:

- CSV `source` values reject EasyFsi, HIBM-MPM, synthetic, fixture,
  placeholder, public tutorial, not-Fluent-truth, validation_runs, and
  not_collected provenance.
- Metadata provenance rejects EasyFsi, HIBM-MPM, synthetic, fixture,
  placeholder, not-Fluent-truth, validation_runs, and not_collected provenance.
- `real_fluent_import_gate` stays blocked for schema-only exports, test-source
  allowance, incomplete contracts, and blocked active manifest promotion.

Remaining gap:

- Metadata currently does not explicitly reject public tutorial/web contour/web
  baseline provenance as a numeric Fluent reference source.

## Required Behavior

Add fail-closed handling for metadata field values containing:

- `public tutorial`
- `web tutorial`
- `tutorial page`
- `web contour`
- `official web baseline`

If any of those terms appear in observed metadata fields and
`allow_test_sources=False`, the collection validator must:

- set `metadata_check.provenance_status` to `incomplete`;
- set `metadata_check.blocker` to
  `fluent_reference_metadata_disallowed_provenance`;
- include the matching terms in `metadata_check.disallowed_provenance`;
- keep `candidate_status` as `fluent_reference_collection_pending`;
- keep `candidate_contract_status` as `fluent_reference_incomplete`;
- keep `real_fluent_import_gate.status` as
  `blocked_real_fluent_import_incomplete`;
- keep `real_fluent_import_gate.can_import_real_fluent_reference=false`;
- keep `real_fluent_import_gate.can_run_solver_evaluation=false`;
- keep `real_fluent_import_gate.fluent_parity_claimed=false`.

Legitimate future metadata may mention the ANSYS tutorial as setup context, but
the numeric source document/provenance must be a real Fluent run/archive/export,
not a public tutorial page, web contour, or web baseline artifact.

## In Scope

- `validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py`
- `tests/integration/test_ansys_vertical_flap_real_fluent_source_exports.py`
- This goal file

## Out of Scope

- Do not import real Fluent CSVs.
- Do not edit existing committed source export templates.
- Do not regenerate Fluent reference diagnostics or parity diagnostics.
- Do not run EasyFsi solver comparison.
- Do not change solver logic.
- Do not set `fluent_parity_claimed=true`.
- Do not change tolerance semantics or active contract promotion behavior.

## TDD Plan

### Red

Add a focused integration test:

1. Build a temporary complete source export bundle using the existing synthetic
   fixture helper.
2. Rewrite every CSV `source` field to a clean Fluent-style source string so the
   CSV source gate passes.
3. Write complete metadata whose `Source document` or export procedure points at
   a public tutorial/web contour/official web baseline as the numeric source.
4. Run `run_with_paths(...)` with default `allow_test_sources=False`.
5. Assert the collection remains blocked by
   `fluent_reference_metadata_disallowed_provenance`.

Before implementation, this test should fail because the metadata disallowed
term list does not contain public tutorial/web contour/web baseline terms.

### Green

Extend the metadata disallowed provenance term list with the public tutorial/web
source terms above. Do not change broader collection, contract, or artifact
semantics.

## Verification

Run:

```powershell
python -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py `
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

Run guards:

```powershell
git diff --check
git diff --name-only validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics
git diff --name-only validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics
```

## Completion Criteria

- The RED test fails for the intended missing public tutorial metadata guard.
- The GREEN implementation blocks public tutorial/web contour/web baseline
  metadata provenance.
- The existing clean Fluent-style temp bundle remains able to satisfy
  `real_fluent_import_gate`.
- Existing HIBM-MPM metadata blocker and source provenance tests remain green.
- No solver logic, generated Fluent data, or parity claim changes.
- Commit and push after local verification passes.
