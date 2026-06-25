# ANSYS Vertical-Flap Sustained Flow Driver Goal

Date: 2026-06-25

## Context

The previous commit `0b9fe8eb4fa985403ca7d4b61c19378cb908b548`
established a reviewable flow-collapse diagnostic baseline for the ANSYS
vertical-flap case. The committed artifacts show that the coarse formal runner
can collapse even when the solid is fixed and marker feedback is disabled.

The most important observations from that baseline are:

- `preflow_only_01` and `preflow_only_02` produce official-like acceleration,
  with p999 velocity around `22.38` and `25.04 m/s`.
- By `preflow_only_05`, `preflow_only_10`, and `preflow_only_20`, p999 drops to
  about `12.65`, `10.30`, and `11.01 m/s`.
- `feedback_on_step10` and `feedback_off_step10` both collapse to about
  `10.31` and `10.30 m/s`, so marker feedback is not the primary suspect.
- `reset_pressure_every_step_step10` improves final p999 to about `17.80 m/s`,
  but still does not keep the flow in the official-like range.
- `reinitialize_inlet_each_step_step10` keeps final p999 around `22.98 m/s` and
  final peak around `29.45 m/s`, but it does so by diagnostic full-field
  reinitialization and must not be treated as a final physical model.

The next step is to convert the diagnostic success of repeated inlet
reinitialization into an explicit, physically interpretable sustained inlet /
flow-predictor path.

## Objective

Implement and validate a minimal sustained-flow driver layer for the ANSYS
vertical-flap formal runner.

The driver layer must answer this question with committed code and artifacts:

```text
Can a sustained inlet / predictor flow path replace the diagnostic full-field
reinitialize-inlet control while keeping 10-step p999 velocity >= 20 m/s?
```

The result must be honest about scope:

- This is an EasyFsi solver diagnostic, not a Fluent parity claim.
- This is a 10-step coarse formal-runner diagnostic, not an L3 50-step run.
- This does not tune solid material, damping, support radius, marker count, or
  feedback weights.

## Non-Goals

Do not do any of the following in this step:

- Do not tune solid parameters.
- Do not tune material parameters or damping.
- Do not use `flow_reinitialize_inlet_each_step` as a final physical model.
- Do not claim point-by-point Fluent parity.
- Do not run L3 50-step validation.
- Do not hide solver behavior in the case layer when a runner/core contract is
  the correct place to expose the behavior.

## Required Design

### Explicit Driver Modes

Add an explicit flow-driver selector to `VerticalFlapFsiConfig`:

```python
flow_driver_mode: str = "projection_only"
```

The initial accepted modes are:

```text
projection_only
reinitialize_inlet_each_step_diagnostic
sustained_boundary_inlet
sustained_volume_source_inlet
sustained_inlet_predictor
sharp_hibm_mpm_reference
```

The existing boolean `flow_reinitialize_inlet_each_step` can remain for
compatibility, but it must be treated as diagnostic-only. The matrix should
report the explicit mode so future reviewers do not have to infer the physical
path from scattered booleans.

### Unified Flow Advance Entry Point

Add a runner-local helper that owns the per-step fluid advance logic, for
example:

```python
_flow_advance_current_step(fluid, config, step_index, preflow_history)
```

This helper should route behavior by `flow_driver_mode`:

```text
projection_only:
  Use the current projection-only path.

reinitialize_inlet_each_step_diagnostic:
  Reinitialize inlet/full computed flow before projection.
  This is a diagnostic upper bound only.

sustained_boundary_inlet:
  Reapply the zmax inlet Dirichlet boundary before projection without resetting
  the interior velocity field and without resetting pressure unless configured.

sustained_volume_source_inlet:
  Reapply inlet boundary and add a sustained zmax inlet volume/source driver
  before projection without resetting the full velocity field.

sustained_inlet_predictor:
  Use the minimal available physically interpretable predictor path. In this
  step it may be implemented as the strongest non-full-reset sustained inlet
  driver available in the current formal runner, but it must be reported
  separately from the diagnostic full-field reset.

sharp_hibm_mpm_reference:
  Reserved for a later switch to the sharp HIBM-MPM reference path. In this
  step, it may be rejected explicitly as unsupported rather than silently doing
  projection-only work.
```

### Boundary Driver

Implement the minimal non-full-reset boundary driver:

```text
Each step:
  refresh zmax velocity Dirichlet active/value/weight
  keep obstacle cells from receiving inlet velocity constraints
  do not alter interior velocity
  do not reset pressure unless flow_reset_pressure_each_step is configured
```

### Volume/Source Driver

Use existing `CartesianFluidSolver` source infrastructure where possible.
The expected minimum is to call the solver's zmax inlet volume source support
before projection and report the related source/outlet diagnostics.

At minimum, record per-step and final fields where available:

```text
source_volume_flux_m3s
positive_source_volume_flux_m3s
abs_source_volume_flux_m3s
zmin_pressure_outlet_flux_m3s
zmin_velocity_outlet_flux_m3s
pressure_outlet_flux_ratio
```

If a field is unavailable from the current projection report, record it as
blank/zero consistently and add source-level tests for the report schema rather
than inventing fake values.

## Diagnostic Matrix

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_sustained_flow_driver_matrix.py
```

It must write:

```text
validation_runs/ansys_vertical_flap_fsi/sustained_flow_driver_diagnostics/
  sustained_flow_driver_matrix.json
  sustained_flow_driver_matrix.csv
  sustained_flow_driver_matrix_summary.md
```

The first matrix should include:

```text
projection_only_step10
reinitialize_inlet_each_step_step10
sustained_boundary_inlet_step10
sustained_volume_source_inlet_step10
sustained_inlet_predictor_step10
sustained_inlet_predictor_feedback_off_step10
reset_pressure_every_step_step10
```

Each row must record at least:

```text
flow_driver_mode
step_count
apply_marker_feedback_to_fluid
flow_reset_pressure_each_step
flow_reinitialize_inlet_each_step
final_velocity_peak_mps
final_velocity_p99_mps
final_velocity_p999_mps
max_velocity_p999_mps
collapse_ratio_p999
projection_l2
projection_max_abs
source_volume_flux_m3s
positive_source_volume_flux_m3s
abs_source_volume_flux_m3s
zmin_pressure_outlet_flux_m3s
zmin_velocity_outlet_flux_m3s
pressure_outlet_flux_ratio
marker_force_z_N
tip_dz_final_m
stress_invalid_marker_count
scatter_invalid_marker_count
feedback_invalid_marker_count
flow_status
```

## Decision Logic

The matrix summary must state one of these outcomes:

```text
projection_only collapses, reinitialize succeeds, sustained mode succeeds:
  Sustained inlet/predictor fix is effective enough for the next coarse 50-step gate.

projection_only collapses, reinitialize succeeds, sustained mode still collapses:
  Sustained mode is not yet equivalent to real inlet momentum; investigate
  source/outlet/projection coupling before any 50-step run.

all sustained modes fail, sharp path one-step succeeds:
  Formal runner should be downgraded to source-level/contract/quick-smoke use;
  official physical validation should move to the sharp HIBM-MPM path.
```

## Success Criteria

The commit is acceptable when all of the following are true:

- A detailed goal file exists and is referenced by the active goal.
- The default `projection_only` behavior remains backward compatible.
- `flow_reinitialize_inlet_each_step` remains diagnostic-only and is explicitly
  reported as such.
- The sustained driver modes are explicit and visible in reports/artifacts.
- The 10-step matrix runs and is committed.
- The matrix answers whether a non-full-reset sustained driver keeps p999
  velocity at or above `20 m/s`.
- The result does not claim Fluent parity.
- Focused tests and compile checks pass.
- Relevant code, docs, tests, scripts, and artifacts are committed and pushed.

## Verification Commands

Use the local Taichi Python environment:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile cases\ansys_vertical_flap_fsi.py benchmarks\official\solid_mpm_fsi_runner.py tools\validation\print_ansys_vertical_flap_diagnostics.py validation_runs\ansys_vertical_flap_fsi\scripts\run_sustained_flow_driver_matrix.py tests\integration\test_ansys_vertical_flap_sustained_flow_driver_artifacts.py
```

```powershell
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_sustained_flow_driver_matrix.py
```

```powershell
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.integration.test_ansys_vertical_flap_sustained_flow_driver_artifacts tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_preflow_controls_are_exposed_without_changing_default_smoke tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_diagnostic_flow_controls_are_explicit_and_default_safe tests.tools.test_ansys_vertical_flap_diagnostics
```

Also run:

```powershell
git diff --check
```

## Remote CI Evidence

After push, record any visible GitHub Actions run for the branch/commit in the
verification artifact if the run can be queried from this environment. If the
run cannot be queried because of authentication, network, or tooling limits,
record that limitation explicitly and keep local verification commands as the
evidence basis.

## Expected Commit

Use:

```text
fix: add sustained inlet predictor diagnostics for ANSYS flap
```
