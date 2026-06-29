# ANSYS Vertical-Flap Real Fluent Source Export Import Goal - 2026-07-01

## Objective

Import provenance-backed real ANSYS Fluent vertical-flap FSI report exports into the Fluent reference collection contract when real exports are available, and otherwise add a fail-closed, opt-in artifact-readiness test surface that proves the current committed schema-only exports cannot be mistaken for real Fluent truth.

This goal follows the latest remote review for branch `codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01` at HEAD `3c68b1c1ee254e2d5bd9de4e9b6f609d3c950fcc`.

## Non-Negotiable Scope Boundaries

- Do not change EasyFsi solver logic.
- Do not run solver comparison.
- Do not run `run_ansys_vertical_flap_generic_solver.py`.
- Do not run `run_traction_selected_formulation_fluent_parity.py`.
- Do not set `fluent_parity_claimed=true`.
- Do not treat public tutorial, official web contour, EasyFsi, HIBM-MPM, synthetic, fixture, placeholder, `validation_runs`, or `not_collected` provenance as real Fluent source truth.
- Do not fabricate numeric Fluent values to make the collection green.
- Do not claim GitHub Actions or remote CI green unless a current workflow-run connector provides evidence.

## Current Repository State Observed Before Implementation

The committed source export files under `validation_runs/ansys_vertical_flap_fsi/fluent_reference/source_exports/` are currently schema-only:

- `fluent_tip_displacement_history.csv` contains only its header.
- `fluent_force_history.csv` contains only its header.
- `fluent_flow_balance_history.csv` contains only its header.
- `fluent_pressure_summary_history.csv` contains only its header.
- `fluent_metadata_2026-06-28.md` keeps every required metadata field as `MISSING`.

Because of that state, the implementation must not promote the current committed source exports to real Fluent readiness unless real exported rows and complete real Fluent metadata are actually present.

## Required Real Fluent Source Export Contract

When real exports are available, exactly these four CSV files must live under `validation_runs/ansys_vertical_flap_fsi/fluent_reference/source_exports/`:

- `fluent_tip_displacement_history.csv`
- `fluent_force_history.csv`
- `fluent_flow_balance_history.csv`
- `fluent_pressure_summary_history.csv`

Each CSV must satisfy:

- It has the exact committed schema header.
- It has a final row with `step=50`.
- The final-row time is `time_s=0.025`.
- All metric values required by the collection validator are finite numbers.
- Every `source` field is non-empty.
- Every `source` field identifies a real ANSYS Fluent run/report export.
- No `source` field contains disallowed provenance terms:
  - `easyfsi`
  - `hibm-mpm`
  - `synthetic`
  - `fixture`
  - `placeholder`
  - `not fluent truth`
  - `validation_runs`
  - `not_collected`
  - `public tutorial`
  - `web tutorial`
  - `tutorial page`
  - `web contour`
  - `official web baseline`

## Required Metadata Contract

`validation_runs/ansys_vertical_flap_fsi/fluent_reference/source_exports/fluent_metadata_2026-06-28.md` must include complete real Fluent provenance for:

- `Source document`
- `Fluent run id`
- `Export author`
- `Export date`
- `Fluent version`
- `mesh/domain source`
- `geometry units`
- `material model`
- `boundary conditions`
- `time step`
- `number of steps`
- `coupling settings if applicable`
- `export procedure`
- `who/when/how generated`
- `force_z_positive`
- `flow_rate_positive`
- `pressure_reference`
- `displacement_definition`

Metadata must not cite public tutorial pages, web contours, local HIBM-MPM reruns, EasyFsi outputs, or validation artifact archives as numeric Fluent truth.

## Required Validator Outcome When Real Exports Are Present

After real exports and metadata are present, running only:

```powershell
python validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py
```

must produce:

- `candidate_status = fluent_reference_collection_complete`
- `candidate_contract_status = fluent_reference_complete`
- `schema_validation.validated_metric_count = 5`
- `schema_validation.required_metric_count = 5`
- `schema_validation.missing_required_metrics = []`
- `promotion_status = ready_for_versioned_contract_promotion`
- `real_fluent_import_gate.status = ready_for_real_fluent_import`
- `real_fluent_import_gate.can_import_real_fluent_reference = true`
- `real_fluent_import_gate.can_run_solver_evaluation = true`
- `real_fluent_import_gate.fluent_parity_claimed = false`
- `real_fluent_import_gate.blockers = []`

If any of these are not true, do not run solver comparison and do not claim parity.

## Required Test Strategy

Add a focused integration test file for real Fluent source export artifact readiness:

`tests/integration/test_ansys_vertical_flap_real_fluent_source_export_artifacts.py`

The test surface must cover:

- The four expected source CSV names.
- Exact headers for every CSV.
- Final `step=50` row detection.
- Final `time_s=0.025` validation.
- Non-empty source provenance.
- Disallowed source provenance rejection.
- Finite metric values.
- Metadata completeness.
- Metadata disallowed provenance must be empty for real readiness.
- Collection candidate complete when real exports are required and present.
- `real_fluent_import_gate.status=ready_for_real_fluent_import` when real exports are required and present.
- `real_fluent_import_gate.fluent_parity_claimed=false` even when import readiness is achieved.
- Active manifest promotion readiness when the contract is complete.
- Existing `CHECKSUMS.sha256` entries match the generated collection diagnostics when diagnostics are regenerated.

## Default Behavior Without Real Exports

If the current repository still has only schema-only/header-only exports, default tests must not fail CI merely because proprietary or uncommitted real Fluent exports are absent.

Default behavior must assert the current fail-closed state:

- Current committed source exports remain schema-only.
- Current committed metadata remains incomplete.
- `candidate_status` remains `fluent_reference_collection_pending`.
- `real_fluent_import_gate.status` remains `blocked_real_fluent_import_incomplete`.
- `real_fluent_import_gate.can_import_real_fluent_reference` remains `false`.
- `real_fluent_import_gate.can_run_solver_evaluation` remains `false`.
- `real_fluent_import_gate.fluent_parity_claimed` remains `false`.

Use the opt-in environment variable:

```text
EASYFSI_REQUIRE_REAL_FLUENT_EXPORTS=1
```

When this variable is set, missing or schema-only real exports must fail the new artifact-readiness tests. This gives maintainers a hard gate for local/private real Fluent import validation without turning public CI red before real exports are committed or mounted.

## Implementation Plan

1. Add the detailed goal file first and create a short active goal that references it.
2. Add the new real Fluent source export artifact readiness tests before production changes.
3. Run the new opt-in test with `EASYFSI_REQUIRE_REAL_FLUENT_EXPORTS=1` while the current repository is schema-only; it must fail for the intended reason.
4. If real Fluent exports are absent, do not edit CSV data or metadata. Implement only the missing test/readiness surface and default fail-closed behavior.
5. If real Fluent exports are present, import them and complete metadata, then regenerate only Fluent reference collection diagnostics.
6. Re-run the new test in default mode and relevant existing integration tests.
7. Verify no solver/parity diagnostics were changed unless the goal explicitly regenerated Fluent reference collection diagnostics after real source import.
8. Commit the RED test/readiness contract separately from any implementation changes when practical.
9. Push the completed branch only after verification.

## Acceptance Criteria

- A repo-local detailed goal file exists and is the implementation contract.
- A short active goal references this file.
- New tests exist for real Fluent source export artifact readiness.
- Current schema-only repository state stays blocked by default.
- Opt-in strict mode fails if real exports are absent or schema-only.
- If real exports are not available in this checkout, no fake import is committed.
- If real exports are available, collection diagnostics are regenerated and prove real import readiness without claiming solver parity.
- `fluent_parity_claimed` remains false in all new tests and artifacts.
- Relevant tests pass locally.
- `git diff --check` passes.
- Final commit is pushed to the tracked remote branch.
