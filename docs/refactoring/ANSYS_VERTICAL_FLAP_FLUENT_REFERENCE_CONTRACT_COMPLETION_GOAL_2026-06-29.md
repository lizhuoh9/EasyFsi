# ANSYS Vertical Flap Fluent Reference Contract Completion Goal - 2026-06-29

## Source Context

This goal follows the remote review of branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25` at commit
`62d5b1b69c14436eae744b434c0347bbe7470f5b`.

The previous stage closed the runtime pressure-pair audit gap:

- focused runtime pressure-pair and generic architecture tests now run in CI;
- generated generic artifacts are scanned for transition-only states;
- `RuntimeAnchoredCellPairProvider` implements the generic provider contract;
- `pressure_sample_pair_map.json` records 24 runtime-generated pairs, checksum,
  pair-map SHA, and marker geometry SHA;
- ANSYS generic adapter metadata now reports the real
  `cartesian-3d-half-domain` runtime domain;
- Fluent parity remains unclaimed and fail-closed.

The remaining scientific blocker is the Fluent reference contract. Current
artifacts intentionally report:

```text
fluent_parity_claimed = false
fluent_parity_status = blocked_reference_incomplete
```

This goal moves the Fluent reference from "blocked because incomplete" to
"complete as a machine-checkable reference contract", while preserving the rule
that Fluent parity is not claimed until comparison gates explicitly pass.

## Short Active Goal Reference

Use this compact active goal:

```text
Implement docs/refactoring/ANSYS_VERTICAL_FLAP_FLUENT_REFERENCE_CONTRACT_COMPLETION_GOAL_2026-06-29.md:
add a machine-checkable Fluent reference contract with provenance, geometry,
material, time-integration, metric units, sign conventions, tolerances, and
schema validation; keep incomplete/malformed contracts fail-closed; regenerate
Fluent reference artifacts; update parity comparison to consume only complete
contracts while keeping fluent_parity_claimed=false unless comparison gates
pass; verify focused tests, artifact checksums, no parity overclaiming; commit
and push branch solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25.
```

## Objective

Complete the ANSYS vertical-flap Fluent reference contract so it is explicit,
machine-checkable, provenance-backed, and safe to use as the input to parity
comparison logic.

The output must not fabricate Fluent data and must not claim Fluent parity just
because the reference contract is complete. A complete reference only permits
comparison; parity remains gated by comparison results.

## Evidence Boundary Found During Implementation

Before editing solver or artifact state, inspect the committed sources for
actual Fluent-exported reference values. The current repository contains:

- the official ANSYS tutorial/web contour baseline under
  `docs/validation/ANSYS_VERTICAL_FLAP_OFFICIAL_WEB_BASELINE_2026-06-25.md`;
- a checked-in web reference CSV with final displacement scale and velocity
  contour range only;
- a local high-resolution HIBM-MPM rerun archive under
  `validation/ansys-fluent-official-half-domain-hibm-mpm-2026-06-25/`.

That archive explicitly states it is a local HIBM-MPM rerun of the Fluent setup,
not an ANSYS Fluent solve, and does not include ANSYS Fluent-generated
time-history reports. Therefore this implementation must not promote
`contract_status` to `fluent_reference_complete` unless provenance-backed
numeric Fluent exports exist for displacement, force, flow, and pressure.

For the current commit, the honest completion target is:

- add the reusable machine-checkable schema validator;
- make collection artifacts expose validator output and blockers;
- make parity artifacts consume only schema-complete active contracts;
- keep the current real contract fail-closed because force, flow, and pressure
  Fluent reference values are not present;
- prove with synthetic tests that a complete contract can validate and enter
  comparison logic when real source data is later supplied.

## Scope

### In Scope

- Add a reference contract schema for ANSYS vertical-flap Fluent data.
- Record required provenance:
  - source name and URL;
  - extraction method or source artifact;
  - reference artifact version;
  - units;
  - sign conventions;
  - sampling definitions;
  - time integration;
  - geometry and material metadata.
- Record required reference metrics:
  - `tip_displacement_m`;
  - `max_displacement_m`;
  - `force_z_N`;
  - `flow_rate_m3s`;
  - `pressure_range_pa`.
- Record tolerances and comparator modes for every reference metric.
- Add schema validation that fails closed on missing provenance, units, sign
  conventions, tolerances, time-step mismatch, step-count mismatch, or metric
  name/unit mismatch.
- Regenerate Fluent reference collection artifacts so they expose
  `contract_status = fluent_reference_complete` when the contract is complete.
- Update parity comparison logic so only a complete reference contract can enter
  comparison gates.
- Keep `fluent_parity_claimed = false` unless all comparison gates pass.
- Keep the EasyFsi generic solver artifacts separate from the Fluent reference
  contract artifacts.
- Add focused integration tests for incomplete, malformed, synthetic-complete,
  and real-complete reference contracts.

### Out Of Scope

- Do not claim Fluent parity automatically.
- Do not alter EasyFsi runtime pressure-pair generation.
- Do not modify the generic solver runtime behavior.
- Do not invent unpublished Fluent measurements.
- Do not fold Fluent reference data into the generic EasyFsi matrix.
- Do not remove existing fail-closed parity tests.
- Do not require heavy Fluent or coupled CFD execution in CI.

## Required Artifact Layout

Keep generated outputs separated by role:

```text
validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/
  EasyFsi generic solver outputs only.

validation_runs/ansys_vertical_flap_fsi/fluent_reference_contract/
  Fluent reference extraction, provenance, units, sign conventions, tolerances,
  and schema validation outputs.

validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity/
  EasyFsi-vs-Fluent comparison outputs only.
```

## Fluent Reference Contract Requirements

Create a canonical contract JSON, for example:

```text
validation_runs/ansys_vertical_flap_fsi/fluent_reference_contract/
  fluent_reference_contract.json
```

Required top-level fields:

```json
{
  "case": "ansys_vertical_flap_fsi",
  "contract_status": "fluent_reference_complete",
  "source_provenance": {},
  "geometry": {},
  "materials": {},
  "time_integration": {},
  "sign_conventions": {},
  "sampling_definitions": {},
  "reference_metrics": {},
  "tolerances": {},
  "comparison_policy": {}
}
```

### Provenance

Required fields:

```text
source_name
source_url
source_version
extraction_method
reference_artifact_version
created_by
```

The extraction method must be explicit. If the data is tutorial-derived rather
than raw Fluent output, state that plainly. Do not hide source limitations.

### Geometry And Materials

Required geometry fields:

```text
duct_length_m
duct_height_m
modeled_domain
modeled_height_m
flap_height_m
flap_thickness_m
flap_streamwise_min_m
flap_streamwise_max_m
```

Required material fields:

```text
fluid.material
fluid.inlet_velocity_mps
solid.material
solid.density_kgm3
solid.young_modulus_pa
solid.poisson_ratio
```

### Time Integration

Required fields:

```text
step_count = 50
time_step_s = 5.0e-4
total_time_s = 0.025
```

### Sign Conventions

Required fields must specify:

```text
coordinate_axes
positive_displacement
positive_force
pressure_reference
flow_rate_positive_direction
```

### Sampling Definitions

Required fields must define:

```text
tip_displacement
max_displacement
force_z
flow_rate
pressure_range
```

Each sampling definition must state location or aggregation, units, and whether
the value is signed, absolute, maximum, or range-based.

### Reference Metrics

Required metrics and units:

```text
tip_displacement_m
max_displacement_m
force_z_N
flow_rate_m3s
pressure_range_pa
```

The contract may use currently available tutorial/reference values and explicit
placeholder policies only when the missing value remains fail-closed. A metric
included in `reference_metrics` must be numeric and finite.

### Tolerances

Each metric must have:

```text
absolute_tolerance
relative_tolerance
comparator
```

Comparators should be explicit, such as:

```text
absolute_error
relative_error
range_contains
sign_matches
```

## Validation Rules

Implement a reusable validator that returns a structured result, for example:

```python
validate_fluent_reference_contract(contract: Mapping[str, Any]) -> dict[str, Any]
```

Required behavior:

- complete valid contract returns `contract_status = fluent_reference_complete`;
- incomplete contract returns `contract_status = fluent_reference_incomplete`;
- missing provenance fails closed;
- missing sign conventions fail closed;
- missing tolerances fail closed;
- missing or non-finite metric values fail closed;
- metric unit/name mismatch fails closed;
- `step_count` mismatch fails closed;
- `time_step_s` mismatch fails closed;
- malformed JSON or wrong types fail closed;
- validation result includes `blockers`, `validated_metric_count`, and
  `required_metric_count`.

## Parity Comparison Rules

Update parity comparison logic so:

- `fluent_reference_incomplete` blocks comparison;
- only `fluent_reference_complete` can reach comparison gates;
- `fluent_parity_claimed` remains `false` if any metric comparison fails;
- a complete synthetic contract can be used in tests to prove pass/fail logic;
- real generated artifacts must not claim parity unless actual comparison gates
  pass;
- near-zero denominator handling remains deterministic.

## Required Tests

Add or update:

```text
tests/integration/test_ansys_vertical_flap_fluent_reference_contract_schema.py
tests/integration/test_ansys_vertical_flap_fluent_reference_collection_artifacts.py
tests/integration/test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts.py
tests/integration/test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic.py
```

Required coverage:

- incomplete contract remains fail-closed;
- missing provenance fails closed;
- missing sign convention fails closed;
- missing tolerance fails closed;
- metric unit/name mismatch fails closed;
- step-count mismatch fails closed;
- time-step mismatch fails closed;
- complete synthetic contract validates as complete;
- complete synthetic contract may enter comparison logic;
- complete contract plus failed metric comparison still keeps no parity claim;
- real generated contract has complete schema and checksums;
- real parity artifact does not claim parity unless comparison gates pass.

## Official Artifact Requirements

The Fluent reference contract artifact should report:

```text
contract_status = fluent_reference_complete
validated_metric_count = required_metric_count
missing_required_metrics = []
blockers = []
```

The parity artifact may change from `blocked_reference_incomplete` to a
comparison result only if the reference contract is complete, but it must still
preserve:

```text
fluent_parity_claimed = false
```

unless every comparison gate passes.

## Verification Plan

Use:

```text
D:\working\taichi\env\python.exe
```

Run:

```text
D:\working\taichi\env\python.exe -m py_compile validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_selected_formulation_fluent_parity.py tests/integration/test_ansys_vertical_flap_fluent_reference_contract_schema.py tests/integration/test_ansys_vertical_flap_fluent_reference_collection_artifacts.py tests/integration/test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts.py tests/integration/test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic.py
D:\working\taichi\env\python.exe validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py
D:\working\taichi\env\python.exe validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_selected_formulation_fluent_parity.py
D:\working\taichi\env\python.exe -m unittest -v tests.integration.test_ansys_vertical_flap_fluent_reference_contract_schema tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic
D:\working\taichi\env\python.exe -m unittest -v tests.integration.test_ansys_vertical_flap_generic_solver_artifacts tests.solvers.test_pressure_sample_pair_provider_contract tests.contracts.test_generic_fsi_solver_architecture
git diff --check
```

Also verify manually:

```text
fluent_reference_contract.json:
  contract_status = fluent_reference_complete
  validated_metric_count = required_metric_count
  blockers = []

fluent parity matrix:
  reference contract is consumed only after schema validation
  fluent_parity_claimed = false unless all gates pass
```

## Push Condition

Push only after:

- this goal file exists and active goal references it;
- Fluent reference contract schema tests pass;
- generated Fluent reference artifacts are checksummed;
- parity artifacts remain honest and do not overclaim;
- runtime pressure-pair/generic solver regressions still pass;
- `git diff --check` passes except benign Windows CRLF warnings;
- staged diff has been reviewed for any fabricated Fluent data or parity
  overclaim.
