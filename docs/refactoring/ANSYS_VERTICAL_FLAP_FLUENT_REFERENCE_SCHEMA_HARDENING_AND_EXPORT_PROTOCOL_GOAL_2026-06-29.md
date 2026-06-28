# ANSYS Vertical Flap Fluent Reference Schema Hardening And Export Protocol Goal - 2026-06-29

## Source Context

This goal follows review of remote branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25` at commit
`17c5e3d51c0fcd060e9fe501a4484b7481f0ccbe`
(`validation: add fluent reference contract schema gate`).

The previous stage added the first Fluent reference contract schema gate:

- collection artifacts now include `schema_validation`;
- parity artifacts now expose `fluent_parity_claimed=false`;
- production parity resolution uses schema validation, not only a
  `contract_status` string;
- the real checked-in Fluent contract remains fail-closed because the
  repository does not contain ANSYS Fluent-generated force, flow, or pressure
  time-history exports;
- the workflow runs the Fluent reference contract schema test.

GitHub Actions run visibility was still unavailable to the reviewer. Treat the
remote CI URL/run id as an external merge blocker, not as a local implementation
blocker.

## Short Active Goal Reference

Use this compact active goal:

```text
Implement docs/refactoring/ANSYS_VERTICAL_FLAP_FLUENT_REFERENCE_SCHEMA_HARDENING_AND_EXPORT_PROTOCOL_GOAL_2026-06-29.md:
harden the ANSYS vertical-flap Fluent reference workflow from a basic fail-closed
schema gate into a contract-v1 gate with schema-bypass regression coverage,
metric unit/source/extraction-method validation, tolerance comparator validation,
active-manifest validation, source-export CSV validation, public tutorial evidence
mapping, Fluent export protocol docs, artifact overclaim scans, and review-map
documentation; keep the real contract incomplete, keep fluent_parity_claimed=false,
regenerate artifacts/checksums, verify focused tests, commit, and push branch
solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25.
```

## Objective

Harden the Fluent reference schema gate so future Fluent data can be added
without allowing a fake or malformed complete reference to enter parity
comparison. This goal does not collect new Fluent measurements and must not
claim Fluent parity.

The desired end state is:

```text
checked-in real Fluent reference contract:
  contract_status = fluent_reference_incomplete
  fluent_parity_claimed = false
  missing force/flow/pressure Fluent exports remain blockers

synthetic complete contracts in tests:
  must satisfy contract-v1 schema before comparison
  may prove pass/fail comparison behavior

future real Fluent exports:
  must provide exact CSV schemas, complete provenance, units, sources,
  extraction methods, comparators, tolerances, and manifest SHA validation
```

## Non-Negotiable Boundaries

- Do not fabricate Fluent displacement, force, flow, or pressure values.
- Do not use EasyFsi or local HIBM-MPM outputs as Fluent source truth.
- Do not use public tutorial contour colors as numeric force/flow/pressure truth.
- Do not switch the real active contract to `fluent_reference_complete`.
- Do not retire `no_fluent_parity_claim`.
- Do not change solver, pressure-pair provider, selected formulation, material,
  geometry, or boundary-condition runtime behavior.
- Do not run Fluent or require heavy CFD in CI.
- Keep generated artifacts deterministic except for any intentionally separate
  non-deterministic manifest, if added later.

## Phase A - Schema Bypass Regression Coverage

Add a negative parity comparison test proving that a reference contract with
`contract_status="fluent_reference_complete"` still stays blocked when the
schema validation result is incomplete.

Target file:

```text
tests/integration/test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic.py
```

Required behavior:

- construct a synthetic contract whose status string says complete;
- make at least one required metric invalid or missing;
- pass an explicit schema validation result with
  `contract_status="fluent_reference_incomplete"` into `_parity_metrics(...)`;
- assert `_candidate_status(...)` returns
  `fluent_parity_blocked_reference_incomplete`;
- assert blockers include `fluent_reference_incomplete` and
  `no_fluent_parity_claim`.

## Phase B - Contract V1 Schema Hardening

Upgrade `fluent_reference_contract_schema.py` from a minimum field checker into
a versioned contract-v1 validator.

Required top-level fields:

```text
schema_version
contract_id
case
contract_status
source_provenance
geometry
material
simulation
sign_conventions
displacement_definition
sampling_definitions
reference_metrics
tolerances
comparison_policy
```

Expected schema version:

```text
ansys_vertical_flap_fluent_reference_contract_v1
```

New or strengthened blockers:

```text
fluent_reference_schema_version_missing
fluent_reference_contract_id_missing
fluent_reference_sampling_definitions_incomplete
fluent_reference_comparison_policy_incomplete
```

### Metric Payload Contract

Every available reference metric must include:

```json
{
  "status": "available",
  "value": 0.0,
  "unit": "...",
  "source": "...",
  "extraction_method": "...",
  "time_s": 0.025
}
```

Required metric units:

```text
tip_displacement_m -> m
max_displacement_m -> m
force_z_N -> N
flow_rate_m3s -> m3/s
pressure_range_pa -> Pa
```

Validation must fail closed on missing unit, wrong unit, missing source,
missing extraction method, non-finite value, missing final time, or final time
mismatch.

### Tolerance Payload Contract

Every available tolerance must include:

```json
{
  "status": "available",
  "value": 0.1,
  "comparator": "relative_error",
  "source": "validation policy v1",
  "rationale": "..."
}
```

Allowed comparators:

```text
relative_error
absolute_error
range_contains
sign_matches
report_only
```

Pressure tolerance must use `absolute_error` or `range_contains`. Unsupported
comparators, missing comparator, missing source, or missing rationale must fail
closed.

## Phase C - Active Contract Manifest Hardening

Make the active manifest a stricter entry point for parity.

Target files:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py
validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_selected_formulation_fluent_parity.py
tests/integration/test_ansys_vertical_flap_fluent_reference_collection_artifacts.py
tests/integration/test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts.py
```

Manifest should include:

```json
{
  "manifest_schema_version": "active_fluent_reference_contract_manifest_v1",
  "active_contract": "...",
  "active_contract_sha256": "...",
  "candidate_contract": "...",
  "candidate_contract_sha256": "...",
  "promotion_status": "...",
  "recommended_action": "...",
  "promotion_blockers": [],
  "active_contract_schema_validation": {},
  "candidate_contract_schema_validation": {}
}
```

Parity runner requirements:

- validate `manifest_schema_version`;
- reject absolute active contract paths;
- reject `..` traversal paths;
- reject active contract paths outside the expected Fluent reference directory;
- reject stale `active_contract_sha256`;
- treat a manifest that claims complete while schema validation is incomplete
  as fail-closed;
- keep fallback behavior only for missing older manifests, and keep fallback
  incomplete.

## Phase D - Fluent Source Export CSV Validator

Add a reusable source export CSV validator to make the distinction between
schema-only CSVs and real data rows explicit.

Suggested file:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/fluent_source_export_schema.py
```

Required API:

```python
validate_source_export_csv(path, expected_header, required_final_step=50, expected_final_time=0.025)
```

Required result fields:

```json
{
  "file_status": "schema_only",
  "row_count": 0,
  "observed_columns": [],
  "reference_values": {},
  "blockers": []
}
```

Required statuses:

```text
schema_only
present_complete
present_header_mismatch
present_missing_final_step
present_final_time_mismatch
present_missing_source
present_missing_metric_value
missing_file
```

Add synthetic temp-file tests for header-only, wrong header, missing final step,
final-time mismatch, empty source, missing metric value, and complete row.

## Phase E - Fluent Reference Export Protocol

Add a human-readable protocol for future real Fluent data collection.

Target file:

```text
docs/validation/ANSYS_VERTICAL_FLAP_FLUENT_REFERENCE_EXPORT_PROTOCOL_2026-06-29.md
```

The protocol must define:

- Fluent version, case source, mesh/source provenance requirements;
- required run setup: 50 steps, `dt=0.0005 s`, lower-symmetry half-domain;
- required report definitions:
  - tip total displacement;
  - max solid displacement;
  - flap force z;
  - outlet flow rate;
  - pressure min/max/range;
- exact CSV filenames and schemas:
  - `fluent_tip_displacement_history.csv`;
  - `fluent_force_history.csv`;
  - `fluent_flow_balance_history.csv`;
  - `fluent_pressure_summary_history.csv`;
  - `fluent_metadata_2026-06-28.md`;
- required units and sign conventions;
- explicit non-use policy for EasyFsi and HIBM-MPM outputs;
- explicit non-claim policy for parity;
- regeneration commands for collection and parity artifacts after real CSVs are
  committed.

Add a protocol artifact test that asserts the document names all required CSVs,
mentions 50 steps and `0.0005`, and contains the non-use/non-claim language.

## Phase F - Public Tutorial Evidence Map

Add a machine-readable map that records what the public ANSYS tutorial supports
and what it does not support.

Target file:

```text
validation_runs/ansys_vertical_flap_fsi/fluent_reference/source_exports/public_tutorial_evidence_map.json
```

Required structure:

```json
{
  "source": "Ansys Fluent v251 tutorial",
  "source_url": "...",
  "available_public_reference": {
    "geometry": true,
    "material": true,
    "boundary_conditions": true,
    "time_step": true,
    "step_count": true,
    "velocity_contour_range": true,
    "displacement_contour_range": true
  },
  "not_available_public_reference": {
    "force_z_time_history": true,
    "flow_rate_time_history": true,
    "pressure_range_time_history": true,
    "csv_exports": true
  },
  "use_policy": "metadata_only_not_parity_truth"
}
```

Collection matrix should reference this map and must still not mark force,
flow, or pressure metrics as available.

## Phase G - Schema-Validated Comparison Fixtures

Update comparison logic tests so synthetic complete contracts pass the schema
validator before comparison.

Required behavior:

- call `validate_fluent_reference_contract(...)` on the synthetic complete
  contract;
- assert the schema result is complete;
- pass schema validation into `_parity_metrics(...)`;
- add failed metric tests for displacement, flow sign, and pressure mismatch if
  time permits.

## Phase H - Explicit Comparison State Separation

Keep `candidate_status` simple:

```text
fluent_parity_blocked_reference_incomplete
fluent_parity_failed
fluent_parity_validated
```

Add a separate `comparison_status` if useful:

```text
not_run_reference_incomplete
compared_failed
compared_passed
```

`fluent_parity_claimed` must remain derived only from:

```text
candidate_status == fluent_parity_validated
```

## Phase I - Artifact Hygiene And Overclaim Scan

Add a workflow artifact-only scan that searches generated Fluent reference and
parity artifacts for accidental validated-parity claims while the real contract
is incomplete.

Scan only generated artifact directories, not tests or goal documents:

```text
validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics
validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics
```

Also add a test or validator rule:

```text
if any Fluent source export CSV has data rows:
  metadata provenance must be complete
  schema validation must not contain a provenance blocker
```

## Phase J - Branch Review Map

Add a review map for the large cumulative branch.

Target file:

```text
docs/refactoring/BRANCH_REVIEW_MAP_2026-06-29.md
```

Required sections:

```text
1. Large layout moves
2. Generic solver boundary
3. ANSYS selected formulation artifacts
4. Runtime pressure-pair audit
5. Fluent reference schema gate
6. What does not claim parity
7. Test matrix
8. Generated artifacts and checksums
9. Suggested PR split
```

## Verification Plan

Use the trusted interpreter:

```text
D:\working\taichi\env\python.exe
```

Required commands:

```text
D:\working\taichi\env\python.exe -m py_compile validation_runs/ansys_vertical_flap_fsi/scripts/fluent_reference_contract_schema.py validation_runs/ansys_vertical_flap_fsi/scripts/fluent_source_export_schema.py validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_selected_formulation_fluent_parity.py tests/integration/test_ansys_vertical_flap_fluent_reference_contract_schema.py tests/integration/test_ansys_vertical_flap_fluent_source_export_schema.py tests/integration/test_ansys_vertical_flap_fluent_reference_export_protocol.py tests/integration/test_ansys_vertical_flap_fluent_reference_collection_artifacts.py tests/integration/test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts.py tests/integration/test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic.py
D:\working\taichi\env\python.exe validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py
D:\working\taichi\env\python.exe validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_selected_formulation_fluent_parity.py
D:\working\taichi\env\python.exe -m unittest -v tests.integration.test_ansys_vertical_flap_fluent_reference_contract_schema tests.integration.test_ansys_vertical_flap_fluent_source_export_schema tests.integration.test_ansys_vertical_flap_fluent_reference_export_protocol tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic tests.integration.test_ansys_vertical_flap_generic_solver_artifacts tests.solvers.test_pressure_sample_pair_provider_contract tests.contracts.test_generic_fsi_solver_architecture
git diff --check
```

Manual artifact checks:

```text
collection matrix:
  candidate_status = fluent_reference_collection_pending
  candidate_contract_status = fluent_reference_incomplete
  validated_metric_count remains below required_metric_count
  public reference use policy = metadata_only_not_parity_truth

parity matrix:
  candidate_status = fluent_parity_blocked_reference_incomplete
  fluent_parity_claimed = false
  active blockers include fluent_reference_incomplete and no_fluent_parity_claim
```

## Push Condition

Push only after:

- this detailed goal file exists and the active goal references it;
- schema bypass and contract-v1 tests pass;
- source export schema tests pass;
- export protocol test passes;
- collection and parity artifacts are regenerated and checksummed;
- runtime/generic regression tests still pass;
- `git diff --check` has no whitespace errors;
- staged diff has no secrets;
- commit is made and pushed to
  `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.
