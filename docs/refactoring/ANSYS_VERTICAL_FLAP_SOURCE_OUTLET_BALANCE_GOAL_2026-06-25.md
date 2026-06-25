# ANSYS Vertical-Flap Source/Outlet Balance Goal

Date: 2026-06-25

## Context

The current remote head before this task is
`15f7db40f04bec99f611c2be327d04d4a0e861bc`. The sustained-flow diagnostic
baseline introduced explicit flow driver modes and showed that:

- `projection_only_step10` still collapses to final p999 about `10.31 m/s`.
- `reinitialize_inlet_each_step_step10` reaches final p999 about `22.98 m/s`
  with final peak about `29.45 m/s`, but it is a diagnostic full-field reset.
- `sustained_boundary_inlet_step10` still collapses, so refreshing only zmax
  Dirichlet velocity is insufficient.
- `sustained_volume_source_inlet_step10` and `sustained_inlet_predictor_step10`
  restore p999 to about `31-32 m/s`, but over-accelerate the peak to about
  `40-41 m/s`.
- The source/outlet diagnostics show a strong reporting and balance question:
  source flux is about `6.0e-4 m^3/s`, zmin velocity outlet flux is close to
  that scale, while zmin pressure outlet flux is only about `2.6-2.7e-5 m^3/s`.

The next step is not a 50-step validation run. The next step is to parameterize
and diagnose source strength and outlet balance so the formal runner can either
find a non-full-reset 10-step candidate or honestly report that no candidate was
found.

## Objective

Implement a source/outlet balance diagnostic baseline for the ANSYS vertical-flap
formal runner.

The implementation must answer:

```text
Can a parameterized non-full-reset inlet source driver keep 10-step p999 in the
20-29 m/s range without pushing peak velocity above 35 m/s, while keeping
interface invalid counts at zero and reporting source/outlet flux balance?
```

## Non-Goals

This task must not:

- Run a 50-step coarse validation.
- Run L-level or L3 validation.
- Tune solid parameters.
- Tune material, damping, support radius, marker count, or feedback weights.
- Treat `flow_reinitialize_inlet_each_step` as a physical model.
- Claim Fluent parity.
- Hide source/outlet calibration in generated artifacts without code-level
  configuration and tests.

## Required Code Changes

### Config

Extend `VerticalFlapFsiConfig` with explicit source/outlet controls:

```python
flow_inlet_source_strength: float = 1.0
flow_inlet_source_ramp_steps: int = 0
flow_inlet_source_profile: str = "constant"
flow_pressure_outlet_enabled: bool = True
flow_outlet_balance_policy: str = "report_only"
```

Initial accepted source profiles:

```text
constant
linear_ramp
```

Initial accepted outlet balance policies:

```text
report_only
```

This task should not implement automatic closed-loop source scaling as a final
physical policy. If later needed, `scale_source_to_outlet` can be added in a
separate validated step.

### Source Strength

The sustained source path currently applies:

```python
normal_velocity_mps = -config.inlet_velocity_mps
```

Change it to:

```python
normal_velocity_mps = -config.inlet_velocity_mps * source_factor
```

where `source_factor` is derived from:

```text
flow_inlet_source_strength
flow_inlet_source_profile
flow_inlet_source_ramp_steps
step_index
```

For `constant`, the factor is simply `flow_inlet_source_strength`.

For `linear_ramp`, the factor should increase from the first step to the target
strength over `flow_inlet_source_ramp_steps`. A ramp value must be bounded in
`[0, flow_inlet_source_strength]`. If `flow_inlet_source_ramp_steps <= 0`, the
profile behaves like `constant`.

### Flow Report

Each step must report at least:

```text
flow_inlet_source_strength
flow_inlet_source_profile
flow_inlet_source_ramp_steps
flow_inlet_source_factor
flow_inlet_source_normal_velocity_mps
flow_pressure_outlet_enabled
flow_outlet_balance_policy
source_volume_flux_m3s
positive_source_volume_flux_m3s
abs_source_volume_flux_m3s
zmin_pressure_outlet_flux_m3s
zmin_velocity_outlet_flux_m3s
pressure_outlet_flux_ratio
velocity_outlet_flux_ratio
```

The matrix output must keep pressure outlet flux and velocity outlet flux
separate. Do not treat one as a substitute for the other without saying so.

### Predictor Naming

`sustained_inlet_predictor` is currently equivalent to source-driven projection.
This task must make that explicit in reports by adding:

```text
flow_predictor_applied
flow_predictor_note
```

Until a real predictor/advection step exists, `sustained_inlet_predictor` should
report `flow_predictor_applied = false` and a note explaining that it is using
the source-driven projection path in this diagnostic baseline.

## Diagnostic Matrix

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_source_outlet_balance_matrix.py
```

It must write:

```text
validation_runs/ansys_vertical_flap_fsi/source_outlet_balance_diagnostics/
  source_strength_sweep.json
  source_strength_sweep.csv
  outlet_balance_sweep.json
  outlet_balance_sweep.csv
  source_outlet_balance_summary.md
  verification_source_outlet_balance_2026-06-25.md
```

### Source Strength Sweep

Run 10-step `sustained_volume_source_inlet` with feedback enabled and no full
field reset:

```text
source_strength = 0.20, 0.30, 0.40, 0.50, 0.60, 0.75, 1.00
flow_inlet_source_profile = constant
step_count = 10
```

Each row must record:

```text
scenario
run_status
flow_driver_mode
source_strength
source_profile
source_ramp_steps
source_factor_final
source_normal_velocity_final_mps
flow_pressure_outlet_enabled
flow_outlet_balance_policy
source_volume_flux_m3s
positive_source_volume_flux_m3s
abs_source_volume_flux_m3s
zmin_pressure_outlet_flux_m3s
zmin_velocity_outlet_flux_m3s
pressure_outlet_flux_ratio
velocity_outlet_flux_ratio
final_velocity_peak_mps
final_velocity_p99_mps
final_velocity_p999_mps
max_velocity_p999_mps
projection_l2
projection_max_abs
marker_force_z_N
tip_dz_final_m
stress_invalid_marker_count
scatter_invalid_marker_count
feedback_invalid_marker_count
candidate_status
```

### Outlet Balance Sweep

Run report-only outlet diagnostics around the best source-strength candidates.
At minimum include:

```text
projection_only_baseline_step10
diagnostic_reinitialize_upper_bound_step10
selected_source_strength_step10
selected_source_strength_reset_pressure_step10
selected_source_strength_ramp5_step10
```

If no selected candidate exists, choose the closest non-full-reset row and mark
it as `closest_no_candidate`.

The sweep must report both pressure and velocity outlet flux ratios.

## Candidate Logic

A `candidate` row must satisfy:

```text
run_status == completed
flow_driver_uses_full_velocity_reset == false
20 <= final_velocity_p999_mps <= 29
final_velocity_peak_mps <= 35
stress_invalid_marker_count == 0
scatter_invalid_marker_count == 0
feedback_invalid_marker_count == 0
```

Rows that use `reinitialize_inlet_each_step_diagnostic` must never be considered
candidate rows, even if their velocity range looks good.

The summary must state:

```text
best_candidate = <scenario or none>
candidate_status = candidate_found | no_candidate
primary_observation = ...
current_best_hypothesis = ...
next_action = ...
```

If no candidate exists, the next action must not be a 50-step run.

## Tests

Add:

```text
tests/integration/test_ansys_vertical_flap_source_outlet_balance_artifacts.py
```

Tests must require:

- source strength sweep has all requested strengths.
- at least one strength is below `1.0`.
- rows include source/outlet flux fields and velocity/projection fields.
- summary explicitly records `best_candidate` or `none`.
- full-field diagnostic rows are excluded from candidate selection.
- artifact CSV row counts match JSON rows.

Update source-level tests for:

- new config defaults.
- source factor calculation for constant and ramp profiles.
- predictor note fields showing no real predictor is applied yet.

## Verification

Run at least:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile cases\ansys_vertical_flap_fsi.py benchmarks\official\solid_mpm_fsi_runner.py tools\validation\print_ansys_vertical_flap_diagnostics.py validation_runs\ansys_vertical_flap_fsi\scripts\run_source_outlet_balance_matrix.py tests\integration\test_ansys_vertical_flap_source_outlet_balance_artifacts.py
```

```powershell
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_source_outlet_balance_matrix.py
```

```powershell
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.integration.test_ansys_vertical_flap_source_outlet_balance_artifacts tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_preflow_controls_are_exposed_without_changing_default_smoke tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_sustained_flow_driver_modes_are_explicit_and_default_safe tests.tools.test_ansys_vertical_flap_diagnostics
```

Also run:

```powershell
git diff --check
```

## Remote CI Evidence

Try to query visible GitHub Actions evidence after push. If GitHub CLI is still
unauthenticated or remote run data is unavailable, record that limitation in the
verification artifact. Do not fabricate remote CI evidence.

## Expected Commit

Use:

```text
fix: calibrate ANSYS flap source outlet balance
```
