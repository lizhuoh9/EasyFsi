# ANSYS Vertical-Flap Source Candidate STEP20 Goal - 2026-06-25

## Goal Summary

Validate whether the current 10-step non-full-reset source candidate for the
ANSYS vertical-flap formal runner remains credible over 20 steps. The previous
source/outlet balance run identified `source_strength=0.75`, constant profile,
pressure outlet enabled, and no full-field velocity reset as the best 10-step
coarse candidate. This goal must test that candidate over 20 steps before any
50-step run is attempted.

This is a source/outlet flow-gate diagnostic. It is not a Fluent parity claim,
not a solid-parameter tuning pass, and not a full validation run.

## Source Evidence From Previous Step

Remote branch:

```text
solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25
```

Observed remote branch HEAD before this goal:

```text
4d3a2c0966d0b5360a915e297e7a4ee50f583802
```

Implementation commit for the prior source/outlet balance step:

```text
02723dd54643f79da5fda6e3b9ed559eee22e993
```

Prior verification recorded that the local GitHub CLI was not authenticated, so
remote Actions run evidence was not available from this workstation. That
limitation must remain explicit in the verification trail until real remote run
evidence is available.

Previous 10-step source/outlet result:

```text
best_candidate = source_strength_0p75_step10
candidate_status = candidate_found
source_strength = 0.75
source_profile = constant
flow_driver_mode = sustained_volume_source_inlet
full-field reset = false
pressure outlet enabled = true
final_velocity_p999_mps = 22.1755
final_velocity_peak_mps = 28.8742
velocity_outlet_flux_ratio = 0.9603
invalid marker counts = 0
```

Known counterexamples from the previous run:

```text
source_strength_1p00_step10 over-accelerated:
  p999 = 31.85 m/s
  peak = 41.35 m/s

source_strength_0p20 through 0p60 step10 stayed below the p999 gate.

source_strength_0p75_ramp5_step10 was too weak:
  p999 = 17.30 m/s

source_strength_0p75_reset_pressure_step10 remained a candidate but had a
velocity outlet flux ratio farther from 1.0 than the no-reset candidate.
```

## Non-Goals

Do not run a 50-step candidate in this goal.

Do not claim Fluent parity.

Do not tune solid parameters, material constants, marker placement, support
radius, grid resolution, or the official geometry.

Do not treat `sustained_inlet_predictor` as a real predictor/advection path.
The current predictor-labeled path is still a diagnostic source-driven
projection path and reports `flow_predictor_applied=false`.

Do not use full-field reinitialize rows as pass candidates. They are upper-bound
diagnostics only.

Do not collapse pressure outlet flux and velocity outlet flux into one metric.

## Required Files

Create or update these files:

```text
docs/refactoring/ANSYS_VERTICAL_FLAP_SOURCE_CANDIDATE_STEP20_GOAL_2026-06-25.md
validation_runs/ansys_vertical_flap_fsi/scripts/run_source_candidate_step20_matrix.py
validation_runs/ansys_vertical_flap_fsi/source_candidate_step20_diagnostics/
tests/integration/test_ansys_vertical_flap_source_candidate_step20_artifacts.py
docs/VALIDATION.md
.github/workflows/ansys-vertical-flap-validation.yml
```

The verification artifact should be:

```text
validation_runs/ansys_vertical_flap_fsi/source_candidate_step20_diagnostics/verification_source_candidate_step20_2026-06-25.md
```

## Required Matrix

The minimum STEP20 matrix is:

```text
projection_only_step20_baseline
diagnostic_reinitialize_step20_upper_bound
source_0p70_constant_step20
source_0p75_constant_step20
source_0p80_constant_step20
source_0p75_reset_pressure_step20
source_0p75_ramp2_step20
source_0p80_ramp2_step20
```

The run may also include:

```text
source_0p75_ramp5_step20
```

if runtime remains reasonable, because the prior 10-step ramp5 case was too weak
and is useful as a slow-ramp contrast.

## Required Candidate Gate

A STEP20 row can be a candidate only if all of these are true:

```text
run_status = completed
flow_driver_uses_full_velocity_reset = false
20 <= final_velocity_p999_mps <= 29
final_velocity_peak_mps <= 35
max_velocity_peak_mps <= 40
0.80 <= velocity_outlet_flux_ratio <= 1.20
stress_invalid_marker_count = 0
scatter_invalid_marker_count = 0
feedback_invalid_marker_count = 0
marker_force_z_N < 0
tip_dz_final_m < 0
```

Rows using full-field reinitialize must be marked `diagnostic_excluded`.

Rows with velocity p999 below 20 must be marked `below_p999_gate`.

Rows with p999 above 29, final peak above 35, or max peak above 40 must be marked
`over_accelerated`.

Rows with velocity outlet flux ratio outside [0.80, 1.20] must not be candidates
and should be classified as an outlet balance failure when no stronger failure
already applies.

Rows with nonzero invalid interface counts must be classified as
`invalid_interface`.

Rows with non-negative marker force z or non-negative tip displacement z must be
classified as a force or displacement sign failure.

## Required Mass-Balance Interpretation

The matrix output and summary must explicitly record:

```text
mass_balance_primary_metric = velocity_outlet_flux_ratio
pressure_outlet_flux_interpretation = diagnostic_only_until_pressure_outlet_model_reviewed
```

Rationale: the current 10-step candidate has velocity outlet ratio close to 1.0,
while pressure outlet ratio is small and negative. Until the pressure-outlet
model is reviewed, pressure outlet ratio is diagnostic evidence, not the primary
pass/fail mass-balance gate.

## Required Per-Step History

The STEP20 run must preserve per-step history for at least the selected
`source_0p75_constant_step20` path. It is acceptable and preferred to emit
history for every matrix row if runtime and artifact size remain small.

Each history row should include:

```text
scenario
step
source_factor
source_normal_velocity_mps
velocity_peak_mps
velocity_p999_mps
velocity_outlet_flux_ratio
pressure_outlet_flux_ratio
projection_l2
projection_max_abs
marker_force_z_N
tip_dz_m
stress_invalid_marker_count
scatter_invalid_marker_count
feedback_invalid_marker_count
```

At minimum, create:

```text
source_strength_0p75_step20_history.csv
```

Prefer a consolidated JSON history artifact as well.

The purpose of the history is to determine whether the candidate is stable over
20 steps, or whether it only happens to end inside the gate after an early
overshoot or collapse.

## Required Output Artifacts

Write:

```text
source_candidate_step20_matrix.json
source_candidate_step20_matrix.csv
source_candidate_step20_summary.md
source_candidate_step20_history.json
source_strength_0p75_step20_history.csv
verification_source_candidate_step20_2026-06-25.md
```

The JSON payload must include:

```text
case
purpose
step_count
rows
best_candidate
candidate_status
mass_balance_primary_metric
pressure_outlet_flux_interpretation
primary_observation
current_best_hypothesis
next_action
```

The verification markdown must state:

```text
No 50-step run was performed.
No Fluent parity claim is made.
No solid parameters were tuned.
Full-field reset rows are diagnostic-only.
```

## Expected Outcomes

If at least one non-full-reset row passes the STEP20 candidate gate:

```text
candidate_status = candidate_found
next_action = run a coarse 50-step flow-gate candidate only after reviewing the
              STEP20 per-step history
```

If no non-full-reset row passes the STEP20 candidate gate:

```text
candidate_status = no_candidate
next_action = stop before 50-step and refine source/outlet model or switch the
              physical validation path to sharp HIBM-MPM
```

Do not hide a failure by selecting the nearest row as a pass candidate. A nearest
row may be reported as a diagnostic fallback, but it must not set
`candidate_status = candidate_found`.

## Required Tests

Add an integration test:

```text
tests/integration/test_ansys_vertical_flap_source_candidate_step20_artifacts.py
```

The test must verify:

```text
the STEP20 matrix JSON exists and is valid
all required scenarios are present
no row has step_count other than 20
the matrix does not include any 50-step scenario
full-field diagnostic rows are excluded from candidate selection
mass_balance_primary_metric is velocity_outlet_flux_ratio
pressure outlet flux interpretation is diagnostic-only
velocity outlet flux ratio is present on every row
pressure outlet flux ratio is present on every row
candidate rows, if any, satisfy the full STEP20 candidate gate
source_0p75 constant history exists
the history contains per-step velocity, outlet, projection, force, displacement,
and invalid-count fields
the verification markdown states that no 50-step run and no Fluent parity claim
were made
```

Update the ANSYS workflow so the new artifact test runs in CI.

## Verification Commands

Use the local Taichi environment for runtime validation:

```powershell
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_source_candidate_step20_matrix.py
& 'D:\working\taichi\env\python.exe' -m py_compile validation_runs\ansys_vertical_flap_fsi\scripts\run_source_candidate_step20_matrix.py tests\integration\test_ansys_vertical_flap_source_candidate_step20_artifacts.py
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.integration.test_ansys_vertical_flap_source_candidate_step20_artifacts
& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_official_half_domain_archive_consistency tests.integration.test_ansys_vertical_flap_postrepair_artifacts tests.integration.test_ansys_vertical_flap_flow_collapse_artifacts tests.integration.test_ansys_vertical_flap_sustained_flow_driver_artifacts tests.integration.test_ansys_vertical_flap_source_outlet_balance_artifacts tests.integration.test_ansys_vertical_flap_source_candidate_step20_artifacts -v
git diff --check
```

Also run a changed-file credential scan before commit.

## Git And Push Requirements

Commit message:

```text
fix: validate ANSYS flap source candidate at 20 steps
```

Push to the current tracked GitHub branch:

```text
solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25
```

After push, attempt to query remote GitHub Actions. If the local GitHub CLI is
not authenticated, record the exact limitation in the verification artifact and
push that documentation update as well.
