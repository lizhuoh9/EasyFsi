# Generic FSI Solver Boundary Goal - 2026-06-28

## Source context

This goal follows the remote review of commit
`3ed8674634a2ed2f11ace3a9886291872c4d4465` on branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.

The Fluent reference and parity line is now correctly fail-closed:

- the active Fluent reference contract points at an incomplete reference;
- promotion remains blocked while required metrics, provenance, comparison
  metadata, and tolerances are missing;
- the parity runner refuses to claim Fluent parity while the active reference is
  incomplete;
- cheap workflow checks cover the collection/parity runners and committed
  artifact contracts;
- no heavy Fluent or coupled EasyFsi run is placed in CI.

The next useful step is no longer another Fluent reference schema patch. The
missing Fluent data must come from an external Fluent export line. This goal
therefore moves the main implementation back to the generic solver runtime:
selected formulation behavior must become a runtime capability that the solver
can generate, run, diagnose, and export without relying on ANSYS-specific
artifact replay as the official path.

## Objective

Introduce the first generic FSI solver runtime boundary for the ANSYS vertical
flap validation case.

The implementation must provide a small, explicit generic API surface and an
official-case runner that can execute the ANSYS vertical flap selected
formulation through that generic boundary, export EasyFsi histories in the same
schema family as the Fluent source exports, and preserve honest diagnostics
about any remaining debug/replay-only paths.

## Required repository goal behavior

This markdown file is the source of truth for the detailed goal. The active
short goal should reference this path rather than trying to inline the full
contract:

```text
Implement docs/refactoring/GENERIC_FSI_SOLVER_BOUNDARY_GOAL_2026-06-28.md:
add the first generic FSI solver runtime boundary, keep ANSYS vertical flap as
an adapter/official case, generate pressure sampling policy diagnostics at
runtime, export EasyFsi Fluent-comparable histories, verify focused tests and
artifacts, then push the branch.
```

## Scope

### In scope

- Add generic solver data structures for a stable runtime boundary:
  - `FsiProblem`
  - `FsiSolverConfig`
  - `FluidDomain`
  - `SolidBody`
  - `InterfaceSurface`
  - `SurfaceRegion`
  - `PressureSamplePairProvider`
  - `OneSidedPressurePolicy`
  - `SurfaceRegionPolicy`
  - `TractionConfig`
  - `PressureSamplingConfig`
  - `DiagnosticsConfig`
  - `FsiRunResult`
- Add a generic `solve_fsi(problem, solver_config, diagnostics_config)` entry
  point that returns structured diagnostics and output artifact paths.
- Keep the first implementation deliberately adapter-backed where needed, but
  expose that status explicitly in diagnostics instead of hiding it.
- Add an ANSYS vertical flap adapter/preset that builds a `FsiProblem` and
  solver configuration without putting `ANSYS`, `Fluent`, `vertical flap`,
  `primary/secondary`, or committed validation artifact filenames into generic
  core logic.
- Add a runtime pressure-pair policy contract with explicit modes:
  - `runtime_anchored_cell_pair`
  - `normal_ladder`
  - `replay_from_diagnostics`
- Treat `replay_from_diagnostics` as debug/regression-only. It may exist for
  transition, but it must not be presented as the official generic runtime
  solution.
- Add a generic official-case runner:
  `validation_runs/ansys_vertical_flap_fsi/scripts/run_ansys_vertical_flap_generic_solver.py`.
- Write generic solver diagnostics under:
  `validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/`.
- Export EasyFsi histories aligned to the Fluent-comparable schemas:
  - `easyfsi_tip_displacement_history.csv`
  - `easyfsi_force_history.csv`
  - `easyfsi_flow_balance_history.csv`
  - `easyfsi_pressure_summary_history.csv`
- Add or update focused tests that prove the generic runner/artifacts are
  present, schema-valid, and honestly report whether runtime generation is
  active or still transition-backed.
- Keep the implementation compatible with cheap CI checks.

### Out of scope

- Do not fabricate Fluent source data.
- Do not mark the Fluent reference contract complete.
- Do not retire `no_fluent_parity_claim`.
- Do not claim Fluent parity from EasyFsi-only artifacts.
- Do not put heavy Fluent or long coupled validation runs into CI.
- Do not rewrite broad solver physics outside the smallest path needed for this
  generic runtime boundary.
- Do not delete existing selected-formulation validation artifacts unless the
  new runner supersedes them with tested artifacts.

## Generic core boundary constraints

Generic core modules must not hard-code case identity. In particular, generic
core code must not contain:

```text
ANSYS
Fluent
vertical flap
primary/secondary case hardcode
fixed_solid_selected_*_markers.json
validation artifact JSON filenames as required runtime inputs
```

Case-specific naming, file paths, metadata, and validation thresholds belong in
adapters, presets, validation runners, or tests.

## Runtime pressure-pair policy requirements

The generic pressure sampling policy must describe how marker pressure samples
are obtained at runtime.

Required structured fields include:

```text
mode
region_id
inside_cell
outside_cell
sample_status
fallback_status
diagnostic_reason
pair_map_sha256
```

For this first stage, the implementation may bridge from existing selected
formulation artifacts only when it reports that bridge as transition/debug
state. The official direction must be clear: runtime generation is the target,
artifact replay is not the final solver path.

## ANSYS vertical flap adapter requirements

The ANSYS vertical flap case must be represented as an adapter or preset that is
responsible for:

- geometry identity;
- material properties;
- boundary condition labels;
- interface region IDs;
- selected formulation preset values;
- artifact and export directories for this validation case.

It must not be responsible for bypassing generic guards or hiding selected
formulation special cases inside generic core.

## Generic official-case runner requirements

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_ansys_vertical_flap_generic_solver.py
```

The runner should call the generic boundary in the following shape:

```python
problem = AnsysVerticalFlapProblem(...)
result = solve_fsi(problem, solver_config, diagnostics_config)
```

The runner must write:

```text
validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/
  generic_solver_selected_formulation_matrix.json
  generic_solver_selected_formulation_history.json
  generic_solver_selected_formulation_summary.md
  easyfsi_tip_displacement_history.csv
  easyfsi_force_history.csv
  easyfsi_flow_balance_history.csv
  easyfsi_pressure_summary_history.csv
```

The matrix JSON must clearly distinguish:

- generic API was invoked;
- selected formulation preset was configured through adapter/config;
- pressure pair policy mode;
- whether runtime generation or transition replay supplied pairs;
- whether this is EasyFsi-only validation;
- whether Fluent parity is still blocked by incomplete Fluent reference data.

## EasyFsi Fluent-comparable export schemas

The EasyFsi exports must use these headers:

```text
step,time_s,tip_displacement_x_m,tip_displacement_y_m,tip_displacement_z_m,tip_displacement_norm_m,max_displacement_m,source
step,time_s,force_x_N,force_y_N,force_z_N,primary_force_z_N,secondary_force_z_N,source
step,time_s,inlet_flow_rate_m3s,outlet_flow_rate_m3s,pressure_outlet_flux_m3s,velocity_outlet_flux_m3s,source
step,time_s,pressure_min_pa,pressure_max_pa,pressure_range_pa,source
```

The source value for this stage should identify the EasyFsi generic solver
runner, not Fluent.

## Acceptance criteria

The implementation is complete for this stage only if all of the following are
true:

- the detailed goal file exists and the active short goal references it;
- the generic API surface exists and is covered by focused tests or artifact
  tests;
- ANSYS vertical flap uses an adapter/preset to call the generic API;
- the generic official-case runner exists and can regenerate the committed
  generic solver artifacts;
- matrix diagnostics state that this is EasyFsi generic solver validation, not
  Fluent parity;
- any transition use of replay/diagnostic artifacts is reported explicitly;
- EasyFsi exports are written with the Fluent-comparable headers listed above;
- committed tests validate the matrix, history, summary, and CSV schemas;
- existing Fluent reference/parity tests remain fail-closed and green;
- verification uses the repository's reliable Taichi Python interpreter:
  `D:\working\taichi\env\python.exe`;
- the final branch is committed and pushed only after verification succeeds.

## First-stage numerical gates

If the runner performs a real 50-step selected-formulation run, the expected
gates are:

```text
completed_step_count = 50
invalid_marker_count_max = 0
sample_pair_fallback_count_max = 0
one_sided_marker_count_min = 24
force_action_reaction_residual <= 1e-8
```

If the first implementation can only replay previously accepted diagnostics,
the matrix must not claim that these runtime gates were freshly validated. It
must instead mark the runtime derivation as incomplete/transition-backed and
keep the next action visible.

## Verification plan

Run the focused local checks:

```text
D:\working\taichi\env\python.exe -m py_compile validation_runs/ansys_vertical_flap_fsi/scripts/run_ansys_vertical_flap_generic_solver.py
D:\working\taichi\env\python.exe validation_runs/ansys_vertical_flap_fsi/scripts/run_ansys_vertical_flap_generic_solver.py
D:\working\taichi\env\python.exe -m unittest -v tests.integration.test_ansys_vertical_flap_generic_solver_artifacts
D:\working\taichi\env\python.exe -m unittest -v tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic
git diff --check
```

Broader validation is optional for this stage unless the implementation touches
shared solver behavior beyond the new generic boundary and adapter.

## Push condition

Only push after:

- the goal markdown exists;
- implementation and tests match this contract;
- generated artifacts are committed;
- focused verification passes;
- the final diff has been reviewed for accidental Fluent parity overclaiming.
