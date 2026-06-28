# ANSYS Vertical Flap Fluent Reference Collection Goal - 2026-06-28

## Baseline

The current branch is `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.
The current remote-backed baseline is commit
`0876dadce1dd977edffe57414f96d66e04c0f19c`, which intentionally keeps Fluent
parity fail-closed:

- `candidate_status = fluent_parity_blocked_reference_incomplete`
- active blockers are `fluent_reference_incomplete` and `no_fluent_parity_claim`
- `historical_blockers_retired = []`
- no Fluent reference values are fabricated
- no heavy coupled EasyFsi runner is re-run for the parity artifact
- no solver, material, geometry, selected formulation, or coupling behavior is
  changed

That baseline already has a selected formulation, fixed-solid selected
formulation evidence, 5-step smoke evidence, 10/30/50-step coupled validation
evidence, and a Fluent parity artifact that correctly refuses to claim parity
while the Fluent reference contract is incomplete.

## Objective

Add an honest Fluent reference collection layer that can accept future
provenance-backed Fluent exports, validate their schema and provenance, compute
reference metrics, and keep the parity contract fail-closed when those exports
are missing or incomplete.

This goal moves the current blocker from the parity runner into the
reference-data layer without changing solver physics. The result must make it
clear that the remaining work is collecting traceable Fluent displacement,
force, flow/outlet, and pressure data, not changing the selected EasyFsi
formulation.

## Required Scope For This Patch

This patch must implement the lightweight, repository-verifiable collection
surface:

1. Harden the current incomplete-reference parity artifact tests so they lock
   the exact blocked status, active blockers, row status, parity status, and
   summary non-overclaim text.
2. Add this detailed goal file as the source of truth for the next step.
3. Add a Fluent reference `source_exports` directory with schema files that
   define the expected CSV headers and metadata fields, while keeping all
   metric values absent until real Fluent exports are available.
4. Add a lightweight Fluent reference collection validator script. The script
   must validate source export presence, headers, final step coverage,
   provenance completeness, units, finite final metrics, and contract fields.
   It must not run Fluent, Taichi, or EasyFsi.
5. Add fail-closed reference collection diagnostics. Missing exports or
   incomplete provenance must produce an explicit blocked/pending status rather
   than a complete contract.
6. Add integration tests for the reference collection artifacts and validator
   output.
7. Add the validator and artifact tests to the existing GitHub workflow cheap
   checks.
8. Keep the existing Fluent parity runner output in the
   `fluent_parity_blocked_reference_incomplete` state until a future committed
   reference contract is complete and backed by real Fluent exports.

## Non-Goals

This patch must not:

- change selected formulation logic
- change solver/coupling code
- change material parameters
- change geometry
- change boundary conditions
- re-run the heavy selected-formulation 50-step coupled validation
- re-run Fluent
- hand-fill reference metric values without source artifacts
- infer reference values from EasyFsi artifacts
- mark `fluent_reference_contract_2026-06-27.json` complete
- retire `no_fluent_parity_claim`
- emit `fluent_parity_validated`
- claim point-by-point Fluent parity

## Reference Source Export Contract

The reference export directory is:

```text
validation_runs/ansys_vertical_flap_fsi/fluent_reference/source_exports/
```

The directory must contain schema files for these future source artifacts:

```text
fluent_metadata_2026-06-28.md
fluent_tip_displacement_history.csv
fluent_force_history.csv
fluent_flow_balance_history.csv
fluent_pressure_summary_history.csv
```

The source files are allowed to be missing, empty, or schema-only in this
patch. That state must remain blocked and must not become a complete contract.

### Tip Displacement CSV

Required columns:

```text
step,time_s,tip_displacement_x_m,tip_displacement_y_m,tip_displacement_z_m,tip_displacement_norm_m,max_displacement_m,source
```

The final row for step 50 is required before the displacement reference can be
considered available.

### Force CSV

Required columns:

```text
step,time_s,force_x_N,force_y_N,force_z_N,primary_force_z_N,secondary_force_z_N,source
```

The final row for step 50 is required before the force reference can be
considered available.

### Flow Balance CSV

Required columns:

```text
step,time_s,inlet_flow_rate_m3s,outlet_flow_rate_m3s,pressure_outlet_flux_m3s,velocity_outlet_flux_m3s,source
```

The final row for step 50 is required before the flow/outlet reference can be
considered available.

### Pressure Summary CSV

Required columns:

```text
step,time_s,pressure_min_pa,pressure_max_pa,pressure_range_pa,source
```

The final row for step 50 is required before the pressure reference can be
considered available.

### Metadata Markdown

Required provenance fields:

```text
Fluent version
mesh/domain source
geometry units
material model
boundary conditions
time step
number of steps
coupling settings if applicable
export procedure
who/when/how generated
```

If any of these are missing or left as placeholders, the collection status must
remain incomplete.

## Validator Contract

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py
```

The script must:

- read only committed local source export files and the current reference
  contract
- validate all required source export paths and headers
- validate that the final required step is present when data rows exist
- validate that `time_step_s = 0.0005`, `step_count = 50`, and
  `total_time_s = 0.025` are consistently represented
- validate provenance completeness without trusting placeholder values
- compute reference metrics only from source rows when all required inputs are
  complete
- emit blocked diagnostics when source exports are missing or schema-only
- write deterministic JSON, CSV, Markdown, and checksum artifacts
- exit successfully for an honest incomplete state so CI can enforce the
  artifact contract without requiring Fluent data

The script must not:

- launch Fluent
- launch Taichi
- run any EasyFsi simulation
- use network access
- mutate solver inputs
- mark the existing 2026-06-27 contract complete

## Diagnostics Contract

The validator output directory is:

```text
validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics/
```

Required output files:

```text
fluent_reference_collection_matrix.json
fluent_reference_collection_matrix.csv
fluent_reference_collection_summary.md
fluent_reference_collection_candidate_contract.json
CHECKSUMS.sha256
```

When reference exports are missing or incomplete, the candidate matrix must
contain:

```text
candidate_status = fluent_reference_collection_pending
candidate_blockers includes:
  fluent_displacement_reference_missing
  fluent_force_reference_missing
  fluent_flow_reference_missing
  fluent_pressure_reference_missing
  fluent_reference_provenance_incomplete
candidate_contract_status = fluent_reference_incomplete
```

The candidate contract must keep missing metrics explicit:

```text
tip_displacement_m = missing/null
max_displacement_m = missing/null
force_z_N = missing/null
flow_rate_m3s = missing/null
pressure_range_pa = missing/null
```

If future source exports become complete, the validator may produce:

```text
candidate_status = fluent_reference_collection_complete
candidate_contract_status = fluent_reference_complete
```

Only then may a later parity patch use those values for actual comparison.

## Fluent Parity Boundary

The existing parity artifact must remain blocked until the collection validator
has produced a provenance-backed complete contract and the parity runner has
been deliberately updated to compare against it.

Allowed current parity state:

```text
candidate_status = fluent_parity_blocked_reference_incomplete
candidate_blockers = fluent_reference_incomplete, no_fluent_parity_claim
historical_blockers_retired = []
```

Forbidden current parity states:

```text
fluent_parity_validated
no_fluent_parity_claim retired
Fluent parity validated
```

## Tests

Add or update tests so they prove:

- current parity artifacts remain fail-closed
- the parity row has `run_status = blocked`
- the parity row has
  `parity_status = fluent_parity_blocked_reference_incomplete`
- `historical_blockers_retired = []`
- the summary contains "does not claim Fluent parity"
- the summary does not contain `fluent_parity_validated`
- source export schema files exist
- required source export headers are exact
- metadata provenance requirements are explicit
- collection diagnostics exist
- collection diagnostics are deterministic and checksum-backed
- the candidate contract remains incomplete while source data is missing
- no source export schema-only state can be mistaken for reference completion

## Workflow

Update `.github/workflows/ansys-vertical-flap-validation.yml` to include only
cheap checks:

```text
py_compile validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py
unittest tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts
unittest tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts
```

The workflow must not run Fluent or the heavy coupled simulation.

## Acceptance Criteria

This goal is complete when:

- the goal file is committed
- the collection source export schema files are committed
- the collection validator script is committed
- collection diagnostics are generated and committed
- integration tests cover both parity fail-closed behavior and collection
  fail-closed behavior
- the GitHub workflow compiles the validator and runs the relevant artifact
  tests
- local verification passes with the trusted Python interpreter
- `git diff --check` passes, allowing only pre-existing CRLF warnings if any
- the final commit is pushed to the same remote branch

## Verification Commands

Use the trusted local interpreter:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py `
  tests\integration\test_ansys_vertical_flap_fluent_reference_collection_artifacts.py `
  tests\integration\test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts.py

& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py

& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py

& 'D:\working\taichi\env\python.exe' -m unittest -v `
  tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts

git diff --check
```

## Completion Statement

After completion, it must still be true that Fluent parity is not claimed. The
only honest next physical-data step is to collect provenance-backed Fluent
exports and rerun the collection validator against those committed files.
