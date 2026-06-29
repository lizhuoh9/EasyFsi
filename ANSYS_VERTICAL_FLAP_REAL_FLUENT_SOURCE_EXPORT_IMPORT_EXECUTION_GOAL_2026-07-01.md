# ANSYS Vertical-Flap Real Fluent Source Export Import Execution Goal - 2026-07-01

## Objective

Execute the next real Fluent import step for the ANSYS vertical-flap reference workflow without fabricating Fluent data. The current branch is `codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`, and the reviewed remote HEAD before this task was `95931a0fb4367b0aabf0b89a0513afae8d9b9c98` (`test: add real fluent export readiness gate`).

The latest review says the provenance/readiness gates are complete and the next step should be the real Fluent CSV plus metadata import. The implementation must therefore provide a concrete import execution path and tests, while preserving the hard boundary that only real provenance-backed ANSYS Fluent report exports may make `real_fluent_import_gate` ready.

## Current Data Availability Check

Before this goal was written, the workspace and adjacent `D:\working\squid robot` tree were checked for the four target real Fluent export filenames:

- `fluent_tip_displacement_history.csv`
- `fluent_force_history.csv`
- `fluent_flow_balance_history.csv`
- `fluent_pressure_summary_history.csv`

Only the committed schema-only files were found under:

`validation_runs/ansys_vertical_flap_fsi/fluent_reference/source_exports/`

Those files currently contain headers only. No complete real Fluent report export bundle and no complete real Fluent metadata file were found in the attachment directory or adjacent working tree. The only `official_web` CSV is web-derived tip displacement evidence and is explicitly not acceptable as real Fluent source truth.

Therefore this task must not write made-up metric rows into the committed source exports. The correct execution work in this checkout is to add a real import tool plus tests that make the import step deterministic once the private/provenance-backed Fluent bundle is provided.

## Non-Negotiable Boundaries

- Do not change solver logic.
- Do not run EasyFsi solver comparison.
- Do not run `run_ansys_vertical_flap_generic_solver.py`.
- Do not run `run_traction_selected_formulation_fluent_parity.py`.
- Do not claim Fluent parity.
- Keep `fluent_parity_claimed=false` in every new path and artifact.
- Do not use public tutorial, web contour, official web baseline, EasyFsi, HIBM-MPM, synthetic, fixture, placeholder, `validation_runs`, or `not_collected` provenance as real Fluent truth.
- Do not import the single `official_web/fluent_tip_displacement_web_final.csv` as a real Fluent report export.
- Do not regenerate solver/parity diagnostics.
- If no complete real Fluent bundle is supplied, keep the committed repository in the current schema-only fail-closed state.

## Required Import Execution Surface

Add an import execution module:

`validation_runs/ansys_vertical_flap_fsi/scripts/import_real_fluent_source_exports.py`

The module must:

- Accept an input bundle directory containing the four required CSVs and `fluent_metadata_2026-06-28.md`.
- Validate all four CSVs before copying anything into the destination.
- Validate complete metadata before copying anything into the destination.
- Reject schema-only bundles.
- Reject missing files.
- Reject wrong headers.
- Reject missing final `step=50`.
- Reject final time different from `0.025`.
- Reject missing or non-finite metric values.
- Reject empty source strings.
- Reject disallowed source provenance terms.
- Reject metadata with missing required fields.
- Reject metadata with disallowed provenance terms.
- Copy the validated source exports into the target source export directory only after the whole bundle passes.
- Run the collection validator after copy when requested.
- Require `real_fluent_import_gate.status=ready_for_real_fluent_import` after a successful import when validation is requested.
- Require `real_fluent_import_gate.fluent_parity_claimed=false`.
- Return or print a machine-readable summary that includes copied files, source and destination directories, collection status, gate status, and blockers.

The default destination should remain:

`validation_runs/ansys_vertical_flap_fsi/fluent_reference/source_exports/`

The default collection output should remain:

`validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics/`

The module should also expose pure functions for tests so integration tests can run in temporary directories without modifying committed source exports.

## Required Tests

Add focused integration tests:

`tests/integration/test_ansys_vertical_flap_real_fluent_source_export_import.py`

The tests must be TDD-oriented and prove:

1. Current committed schema-only source exports are rejected by the import preflight as not real Fluent rows.
2. A missing input file causes a precise import failure before any destination file is overwritten.
3. A source CSV with disallowed provenance such as `official web baseline` is rejected.
4. Metadata with public tutorial or local artifact provenance is rejected.
5. A complete temporary real-Fluent-style bundle imports into a temporary destination and makes the collection validator return:
   - `candidate_status=fluent_reference_collection_complete`
   - `candidate_contract_status=fluent_reference_complete`
   - `schema_validation.validated_metric_count=5`
   - `schema_validation.required_metric_count=5`
   - `schema_validation.missing_required_metrics=[]`
   - `promotion_status=ready_for_versioned_contract_promotion`
   - `real_fluent_import_gate.status=ready_for_real_fluent_import`
   - `real_fluent_import_gate.can_import_real_fluent_reference=true`
   - `real_fluent_import_gate.can_run_solver_evaluation=true`
   - `real_fluent_import_gate.fluent_parity_claimed=false`
   - `real_fluent_import_gate.blockers=[]`

The positive test may use a temporary local fixture with clearly test-scoped numeric values, but it must not write those values to the committed source export directory. The committed repository must remain schema-only unless a real external Fluent bundle is explicitly supplied.

## Expected Command Behavior

The importer should support a command shape like:

```powershell
python validation_runs\ansys_vertical_flap_fsi\scripts\import_real_fluent_source_exports.py `
  --input-dir <path-to-real-fluent-export-bundle> `
  --destination-dir validation_runs\ansys_vertical_flap_fsi\fluent_reference\source_exports `
  --run-collection-validator
```

If the input bundle is incomplete or not real Fluent provenance, the command must fail non-zero and report blockers. If the bundle is complete, it may update source exports and collection diagnostics, but must still leave `fluent_parity_claimed=false`.

## Verification Plan

Run at minimum:

```powershell
python -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\import_real_fluent_source_exports.py `
  tests\integration\test_ansys_vertical_flap_real_fluent_source_export_import.py
```

```powershell
python -m unittest -v `
  tests.integration.test_ansys_vertical_flap_real_fluent_source_export_import `
  tests.integration.test_ansys_vertical_flap_real_fluent_source_export_artifacts `
  tests.integration.test_ansys_vertical_flap_real_fluent_source_exports `
  tests.integration.test_ansys_vertical_flap_fluent_source_export_schema `
  tests.integration.test_ansys_vertical_flap_fluent_reference_contract_schema `
  tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts `
  tests.integration.test_ansys_vertical_flap_fluent_artifact_policy
```

Also run:

```powershell
git diff --check
```

Verify that no solver logic, solver diagnostics, parity diagnostics, or committed source export CSV values changed unless a complete real external Fluent bundle was actually supplied.

## Acceptance Criteria

- This detailed goal file exists before implementation.
- A short active goal references this file.
- The importer module exists and can be used later with a real Fluent bundle.
- Tests prove incomplete/schema-only inputs fail before copy.
- Tests prove disallowed provenance fails before copy.
- Tests prove a complete temporary real-Fluent-style bundle can make the collection validator ready without claiming parity.
- Current committed source exports remain schema-only because no real external Fluent bundle was available in this checkout.
- No fake Fluent source rows or metadata are committed.
- Relevant tests pass locally in default mode.
- The finished changes are committed and pushed to the current remote branch.
