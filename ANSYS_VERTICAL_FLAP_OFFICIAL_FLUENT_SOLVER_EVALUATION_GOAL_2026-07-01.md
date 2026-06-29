# ANSYS Vertical Flap Official Fluent Solver Evaluation Goal - 2026-07-01

## Objective

Start the official Fluent solver evaluation line on a new branch without expanding
the existing validation-tools package PR. This first pass must harden the Fluent
source export intake contract before any solver comparison, artifact promotion,
or Fluent parity claim is allowed.

The immediate deliverable is a fail-closed provenance guard for ANSYS
vertical-flap Fluent source exports:

- Fluent source exports may only promote a candidate reference contract when the
  final-step metric rows are finite, schema-valid, and backed by source strings
  that are not EasyFsi, HIBM-MPM, synthetic, placeholder, public tutorial, or
  derived validation artifacts.
- The existing synthetic pipeline tests may continue to exercise temporary
  parity comparison mechanics, but synthetic fixtures must not be accepted as
  production Fluent source exports when running the collection validator.
- Missing real Fluent exports must keep `fluent_reference_incomplete`,
  `fluent_reference_collection_pending`, `blocked_reference_incomplete`, and
  `fluent_parity_claimed=false`.
- The current committed source export templates are schema-only and must remain
  honest until real Fluent source exports with complete provenance are provided.

## Branch and Base

- Work branch:
  `codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`
- Base branch:
  `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Do not add more commits to
  `codex/ansys-vertical-flap-validation-tools-package-2026-06-30` for this
  evaluation line.

## Files In Scope

- `validation_runs/ansys_vertical_flap_fsi/scripts/fluent_source_export_schema.py`
- `validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py`
- `tests/integration/test_ansys_vertical_flap_fluent_reference_synthetic_pipeline.py`
- `tests/integration/test_ansys_vertical_flap_fluent_reference_collection_artifacts.py`
- New focused tests under `tests/integration/` if they better isolate the guard
- This goal document

## Files Out of Scope

- Solver physics, FSI coupling, pressure projection, marker force scatter, and
  runtime solver logic
- Existing generated Fluent parity artifacts unless regeneration is required by
  the source-export guard and the diff is reviewed
- Fluent reference metric values, tolerance calibration, or official source
  export imports
- Public tutorial data promotion
- Any change that turns `fluent_parity_claimed` to `true`
- Any claim that GitHub Actions is green without an observed run URL/status

## Required Behavior

1. The production source export validator must reject final-step rows whose
   `source` field is missing or contains disallowed provenance terms such as:
   `EasyFsi`, `HIBM-MPM`, `synthetic`, `fixture`, `placeholder`, `public
   tutorial`, `not fluent truth`, `validation_runs`, or `not_collected`.
2. Rejected source rows must keep `metric_status=missing` and produce a blocker
   that makes the collection validator preserve the incomplete contract state.
3. The collection matrix must surface rejected source provenance through the
   existing `schema_blockers` / source check payloads without changing the
   established top-level missing-reference blocker vocabulary unless a test
   requires a precise new blocker.
4. The temporary synthetic pipeline must still be able to exercise parity
   comparison logic when explicitly requested by tests. This should be done via
   a non-default test-only option, not by weakening production intake behavior.
5. Existing committed schema-only source exports must still pass the existing
   artifact tests as incomplete/pending.

## TDD Plan

### Red

Add a focused integration test that builds otherwise complete temporary source
exports, rewrites the final-step `source` fields to an EasyFsi/HIBM-MPM-derived
placeholder, and runs the collection validator through `run_with_paths`.

The expected red assertion before implementation is:

- `candidate_status` must remain `fluent_reference_collection_pending`
- `candidate_contract_status` must remain `fluent_reference_incomplete`
- `schema_validation.validated_metric_count` must be less than 5
- source checks must include a provenance-related schema blocker
- no metric can be promoted from an EasyFsi/HIBM-MPM source row

### Green

Implement the minimal production validator change:

- Reject disallowed final-step source strings by default.
- Add an explicit test-only allowance for the synthetic fixture pipeline so the
  existing temp-only parity comparison test remains useful.
- Keep all artifact paths repo-relative and avoid absolute-path leakage.

### Refactor

Keep helper functions small and local to the source-export schema or collection
runner. Avoid broad rewrites and avoid changing existing public artifact field
names unless tests prove a contract gap.

## Verification Commands

Run at minimum:

```powershell
python -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\fluent_source_export_schema.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py `
  tests\integration\test_ansys_vertical_flap_fluent_reference_synthetic_pipeline.py `
  tests\integration\test_ansys_vertical_flap_fluent_reference_collection_artifacts.py
```

```powershell
python -m unittest -v `
  tests.integration.test_ansys_vertical_flap_fluent_reference_synthetic_pipeline `
  tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts `
  tests.integration.test_ansys_vertical_flap_fluent_reference_contract_schema `
  tests.integration.test_ansys_vertical_flap_fluent_source_export_schema `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic `
  tests.integration.test_ansys_vertical_flap_fluent_artifact_policy
```

Also run:

```powershell
git diff --check
git diff --name-only validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics
git diff --name-only validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics
```

Generated artifact roots should remain unchanged unless the source-export guard
intentionally regenerates diagnostics and those regenerated files are reviewed.

## Completion Criteria

- The new RED test fails for the intended reason before production code changes.
- The same test passes after the guard is implemented.
- Existing schema-only committed source export artifact tests remain green.
- Synthetic temp-only parity mechanics remain green through an explicit
  test-only allowance.
- No solver logic or Fluent parity claim is changed.
- Branch is committed and pushed only after local verification passes.
