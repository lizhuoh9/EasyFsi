# ANSYS Vertical Flap Real Fluent Metadata Provenance Guard Goal - 2026-07-01

## Objective

Harden the real Fluent source-export import path so that provenance-complete
CSV rows cannot be promoted if the companion metadata reveals that the source is
actually a local EasyFsi, HIBM-MPM, synthetic, fixture, placeholder, or archived
validation artifact.

The current branch already rejects disallowed CSV `source` values and emits a
machine-readable `real_fluent_import_gate`. This task closes the remaining
metadata loophole: a CSV row could use a Fluent-looking `source` string while
`fluent_metadata_2026-06-28.md` still documents a local HIBM-MPM archive or
other non-Fluent provenance.

## Why This Is Required

Local directories in this checkout include files with names such as
`official_ansys_fluent...`, but the archived README explicitly says those are
local HIBM-MPM reruns or Fluent-style renders, not ANSYS Fluent report exports.
Those artifacts are useful diagnostic evidence, but they must not be imported
as Fluent reference truth.

The gate must therefore validate both:

- per-CSV final-step `source` provenance; and
- metadata provenance fields.

## In Scope

- Update `validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py`.
- Add tests to `tests/integration/test_ansys_vertical_flap_real_fluent_source_exports.py`.
- Keep existing source-export schema tests, synthetic temp-only comparison
  tests, and Fluent parity fail-closed tests green.
- Preserve current schema-only committed source exports as blocked/incomplete.

## Out of Scope

- Do not import or fabricate real Fluent source exports.
- Do not use local HIBM-MPM archives, Fluent-style renders, or public tutorial
  data as numeric Fluent truth.
- Do not run EasyFsi solver comparison.
- Do not change solver physics, FSI coupling, pressure projection, or generated
  solver artifacts.
- Do not set `fluent_parity_claimed=true`.
- Do not regenerate committed Fluent collection diagnostics unless the diff is
  intentional and reviewed.

## Required Behavior

1. Production metadata validation must scan required metadata field values for
   disallowed provenance terms:
   - `easyfsi`
   - `hibm-mpm`
   - `synthetic`
   - `fixture`
   - `placeholder`
   - `not fluent truth`
   - `validation_runs`
   - `not_collected`
2. If disallowed metadata provenance is found and `allow_test_sources=False`:
   - `metadata_check.provenance_status` must be `incomplete`;
   - `metadata_check.blocker` must be
     `fluent_reference_metadata_disallowed_provenance`;
   - `candidate_contract_status` must remain `fluent_reference_incomplete`;
   - `real_fluent_import_gate.status` must remain
     `blocked_real_fluent_import_incomplete`;
   - `real_fluent_import_gate.can_import_real_fluent_reference` must be false;
   - `real_fluent_import_gate.can_run_solver_evaluation` must be false.
3. `allow_test_sources=True` remains a temp-only test path. It may keep
   synthetic comparison tests usable, but the real import gate must still block
   real readiness through `test_source_allowance_enabled`.
4. A temp-only Fluent-style bundle with clean metadata must still satisfy the
   import gate, proving the guard does not block legitimate Fluent provenance.

## TDD Plan

### Red

Add an integration test that:

- builds a complete temp source-export bundle;
- rewrites CSV source strings to a Fluent-style source so CSV provenance passes;
- writes complete metadata that contains local HIBM-MPM archive provenance;
- runs the collection validator with default `allow_test_sources=False`;
- expects the collection to remain blocked by
  `fluent_reference_metadata_disallowed_provenance`.

Before implementation this should fail because metadata provenance is currently
treated as complete when all fields are non-missing and step/time values match.

### Green

Implement a minimal metadata-provenance scan in the collection validator:

- add stable blocker name and details;
- expose disallowed metadata matches in `metadata_check`;
- carry the metadata blocker into `real_fluent_import_gate.metadata`;
- leave existing fail-closed blocker behavior unchanged for missing metadata.

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

Run artifact and whitespace guards:

```powershell
git diff --check
git diff --name-only validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics
git diff --name-only validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics
```

## Completion Criteria

- The RED test fails for the intended metadata provenance loophole.
- The GREEN implementation blocks HIBM-MPM/EasyFsi/synthetic metadata
  provenance while preserving test-only synthetic comparison mechanics.
- Current committed source exports stay fail-closed.
- No solver logic, generated Fluent values, or parity claim changes.
- Local verification passes before commit and push.
