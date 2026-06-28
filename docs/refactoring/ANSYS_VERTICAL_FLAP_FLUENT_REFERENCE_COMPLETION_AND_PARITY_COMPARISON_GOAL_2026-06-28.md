# ANSYS Vertical Flap Fluent Reference Completion And Parity Comparison Goal - 2026-06-28

## Baseline

The active branch is
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.
The current remote-backed baseline is commit
`0f1e50d60aa199a8daea44bf8a50fd5e369c18d6`, which added a fail-closed
Fluent reference collection contract.

Current committed state:

- selected-formulation coupled step50 evidence is passed
- Fluent reference collection is pending
- Fluent parity remains blocked because the reference contract is incomplete
- source export CSV files are schema-only
- Fluent metadata provenance fields are still `MISSING`
- `fluent_reference_contract_2026-06-27.json` is still
  `fluent_reference_incomplete`
- `no_fluent_parity_claim` is still active

This is correct. The next change must not turn this into a Fluent parity claim
without real Fluent source exports, complete provenance, tolerances, and actual
error computations.

## Objective

Prepare the code and artifacts for the next real Fluent-data step by adding the
missing complete-reference and parity-comparison machinery while keeping the
current repository state fail-closed.

This patch should make it possible for a future commit containing real Fluent
source exports to flow through:

```text
source_exports complete
metadata provenance complete
tolerances complete
collection validator emits fluent_reference_collection_complete
versioned complete contract can be promoted intentionally
parity runner computes real errors
parity runner emits validated or failed, not pending_comparison
```

Until those source exports exist, the current artifacts must still say:

```text
collection: fluent_reference_collection_pending
parity: fluent_parity_blocked_reference_incomplete
```

## Required Scope For This Patch

This patch must:

1. Add this detailed goal file and use it as the active Codex goal reference.
2. Extend the Fluent reference collection contract with explicit comparison
   metadata needed for a future complete contract:
   - displacement definition
   - force sign convention
   - flow-rate sign convention
   - pressure reference convention
   - source contract recommendation / active contract promotion boundary
3. Keep current metadata placeholders incomplete and auditable.
4. Strengthen the collection validator so it can distinguish:
   - schema-only source exports
   - complete metric source exports but incomplete provenance
   - complete source/provenance but incomplete tolerances
   - fully complete reference contract
5. Keep the validator output deterministic and checksum-backed.
6. Extend the Fluent parity runner so that once a complete contract is provided,
   it computes actual displacement, force, flow/outlet, and pressure errors
   instead of returning `pending_comparison`.
7. Add tests for the complete-contract comparison logic using synthetic in-memory
   fixtures, without writing fake Fluent source values into committed reference
   exports.
8. Keep current artifact tests proving the checked-in repository remains
   fail-closed.
9. Add the new comparison-logic test to the existing cheap workflow.

## Non-Goals

This patch must not:

- populate real Fluent CSV values unless provenance-backed source data already
  exists in the workspace
- fabricate Fluent displacement, force, flow, or pressure values
- change EasyFsi solver logic
- change selected formulation logic
- change geometry, material, boundary conditions, or timestep
- promote `fluent_reference_contract_2026-06-27.json` to complete
- create `fluent_reference_contract_2026-06-28.json` with fake values
- retire `no_fluent_parity_claim`
- claim `fluent_parity_validated`
- run Fluent
- run the heavy EasyFsi coupled validation

## Reference Completion Contract

The source exports directory remains:

```text
validation_runs/ansys_vertical_flap_fsi/fluent_reference/source_exports/
```

The current source export CSVs are intentionally schema-only. A future
data-population commit may add rows, but this patch must not invent them.

The collection validator should emit comparison metadata in the candidate
contract, even while incomplete:

```json
"displacement_definition": {
  "metric": "tip_displacement_norm_m",
  "source_step50_metric": "tip_mean_displacement_m",
  "point": "flap tip centerline or documented Fluent export location",
  "status": "missing"
}
```

```json
"sign_conventions": {
  "force_z_positive": "",
  "flow_rate_positive": "",
  "pressure_reference": "",
  "status": "missing"
}
```

When future metadata fills these fields, `status` may become `complete`.

The validator should also preserve a promotion boundary:

```json
"active_contract_recommendation": {
  "recommended_action": "keep_current_incomplete_contract",
  "current_contract": "...fluent_reference_contract_2026-06-27.json",
  "candidate_contract": "...validation_diagnostics/fluent_reference_collection_candidate_contract.json",
  "reason": "source exports, provenance, or tolerances are incomplete"
}
```

If a future candidate becomes complete, the recommendation may change to:

```json
"recommended_action": "promote_versioned_contract"
```

but this patch should not perform that promotion.

## Parity Comparison Contract

The parity runner currently reads:

```text
validation_runs/ansys_vertical_flap_fsi/fluent_reference/fluent_reference_contract_2026-06-27.json
```

That path remains unchanged in this patch.

When the referenced contract is incomplete, behavior remains unchanged:

```text
candidate_status = fluent_parity_blocked_reference_incomplete
candidate_blockers = fluent_reference_incomplete, no_fluent_parity_claim
historical_blockers_retired = []
```

When a future complete contract is provided, the parity runner must compute:

### Displacement

Compare:

```text
source_step50_tip_mean_displacement_m
source_step50_max_displacement_m
```

against:

```text
fluent_tip_displacement_m
fluent_max_displacement_m
```

using:

```text
tip_displacement_relative
max_displacement_relative
```

The displacement gate passes only if both errors pass.

### Force

Compare:

```text
source_step50_marker_force_z_N
```

against:

```text
fluent_force_z_N
```

using:

```text
force_z_relative
```

The metric must report both relative error and force sign agreement. Near-zero
Fluent force must use a stable denominator floor.

### Flow / Outlet

Compare the committed step50 outlet flux:

```text
zmin_velocity_outlet_flux_m3s
```

against:

```text
fluent_flow_rate_m3s
```

using:

```text
flow_rate_relative
```

The metric must report relative error and flow sign agreement.

### Pressure

Compute:

```text
source_pressure_range_pa = pressure_max_pa - pressure_min_pa
```

and compare against:

```text
fluent_pressure_range_pa
```

using:

```text
pressure_sanity_absolute
```

This is an absolute gate unless a later source-backed contract defines a
different pressure convention.

### Metadata

For a complete contract, metadata gate must require:

- `contract_status = fluent_reference_complete`
- `source_provenance.status = complete`
- `simulation.step_count = 50`
- `simulation.time_step_s = 0.0005`
- `simulation.total_time_s = 0.025`
- comparison metadata is present

For the current incomplete contract, metadata may remain `passed` so that the
artifact stays blocked by reference incompleteness rather than reporting a
spurious mismatch.

## Expected Current Artifact State After This Patch

Collection artifacts must remain:

```text
candidate_status = fluent_reference_collection_pending
candidate_contract_status = fluent_reference_incomplete
```

Parity artifacts must remain:

```text
candidate_status = fluent_parity_blocked_reference_incomplete
candidate_blockers = fluent_reference_incomplete, no_fluent_parity_claim
historical_blockers_retired = []
```

The parity metrics may include additional comparison fields, but the checked-in
status must not change.

## Tests

Update or add tests so they cover:

- checked-in collection artifacts remain pending/incomplete
- checked-in parity artifacts remain blocked/incomplete
- comparison metadata exists in the candidate contract
- current missing comparison metadata is not treated as complete
- synthetic complete contracts produce actual relative/absolute errors
- synthetic all-pass contracts produce `fluent_parity_validated`
- synthetic mismatched contracts produce `fluent_parity_failed`
- `no_fluent_parity_claim` is retired only for synthetic validated complete
  parity, not for current checked-in incomplete parity
- near-zero denominator logic is deterministic
- pressure range absolute error is computed from source pressure max/min

Synthetic tests must not write fake Fluent values into committed
`source_exports/`.

## Workflow

Update `.github/workflows/ansys-vertical-flap-validation.yml` to include the
new comparison-logic test in cheap checks. The workflow must still not run
Fluent or heavy EasyFsi coupled validation.

## Verification Commands

Use the trusted local interpreter:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py `
  tests\integration\test_ansys_vertical_flap_fluent_reference_collection_artifacts.py `
  tests\integration\test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts.py `
  tests\integration\test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic.py

& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py

& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py

& 'D:\working\taichi\env\python.exe' -m unittest -v `
  tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic

git diff --check
```

## Completion Criteria

This goal is complete when:

- this goal file is committed
- the collection candidate contract includes explicit comparison metadata and
  promotion recommendation fields
- the parity runner computes actual comparison errors for complete contracts
- synthetic tests prove validated and failed complete-contract paths
- current checked-in artifacts remain fail-closed
- workflow cheap checks include the new tests
- local verification passes
- the final commit is pushed to the active remote branch

## Completion Statement

After this patch, the repository should be ready for real Fluent source exports
without pretending those exports already exist. The only valid next physical
step remains collecting provenance-backed Fluent data and then promoting a
versioned complete contract.
