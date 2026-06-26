# ANSYS Vertical-Flap Hydrodynamic Load Diagnostic Goal - 2026-06-26

## Background

The previous pushed commit is:

```text
5f5093b16ddf76528612df66e298ec31642ea228
fix: correct ANSYS preflow release scheduling
```

That commit correctly fixed the preflow source schedule indexing, extracted
shared temporal gates, regenerated fixed-solid STEP30 flow diagnostics, and
added the coupled preflow-release STEP20 matrix. The new evidence changed the
next question. The source/outlet flow path is now stable enough to diagnose,
but the fixed flap hydrodynamic marker force can still change sign even before
the MPM solid is released.

The next goal is therefore not coupling subiterations. The next goal is to make
the fluid-to-marker hydrodynamic load measurable, face-resolved, residual
checked, and archived in a fixed-solid diagnostic matrix.

## Current Evidence

The committed preflow-release STEP20 artifacts report:

```text
best_preflow_release_candidate = none
promotion_candidate_count = 0
best_release_flow_candidate = no_preflow_release20_source_0p80_ramp2
```

All release rows pass the release-flow temporal gate, but all combined/coupling
rows fail. That alone is not sufficient to blame MPM coupling because fixed
solid preflow already shows marker-force sign changes.

Important observed examples from the committed artifacts:

```text
preflow30_release20_source_0p80_ramp2:
  fixed-solid preflow force changes from negative near step 28 to positive by
  preflow step 30.

preflow20_release20_source_0p80_ramp2:
  force is negative before release and for early release rows, then becomes
  positive from release step 9 through step 20 while tip displacement remains
  negative.

preflow20_release20_source_0p80_ramp2_feedback_off:
  marker feedback disabled does not remove the force sign failure.

preflow20_release20_source_0p80_ramp2_phase_local:
  source ramp restart is correctly detected, but it also does not remove the
  force sign failure.
```

This means the current first-order diagnostic question is:

```text
Does the current dual-face marker, two-sided-pressure sampling, and traction
integration produce a correct, stable, explainable hydrodynamic load on the
fixed flap?
```

## Scope For This Commit

This commit should implement the minimum reviewable hydrodynamic-load
diagnostic surface. It should not attempt the later A/B/C traction formulation
matrix, Fluent force-history import, or coupling subiterations.

Required in scope:

1. Add a detailed goal file, this file, and reference it from the active goal.
2. Extend the formal ANSYS vertical-flap runner reports with hydrodynamic-load
   fields that are already available from the HIBM-MPM core reports.
3. Split the two marker faces into primary and secondary region IDs for
   reporting while preserving marker positions, normals, areas, force scatter,
   surface feedback, and total marker count.
4. Preserve total-load behavior while making face-resolved marker force,
   valid/invalid marker counts, and traction diagnostics visible.
5. Fill marker and scatter action-reaction residual fields with real report
   values instead of empty CSV cells.
6. Rename the preflow-release index-only continuity field so it no longer
   overclaims state-field continuity.
7. Fix the `best_release_flow_candidate` ranking so the flow candidate is
   ranked by flow metrics only, not by coupling settling.
8. Tighten the preflow-release promotion gate so it also checks root
   displacement and marker/scatter action-reaction residuals before a future
   row can become promotion-ready.
9. Add a fixed-solid hydrodynamic-load temporal matrix and archive artifacts.
10. Update tests, workflow, and validation docs.
11. Commit and push to the GitHub remote after verification passes.

Out of scope for this commit:

```text
coupled 50-step runs
L2/L3 matrices
solid material tuning
damping/support/gate relaxation
full-field reinitialize promotion
Fluent parity claims
traction formulation A/B/C implementation
Fluent force-history import
coupling subiterations
```

## Phase 1 - Runner Report Completeness

Modify:

```text
benchmarks/official/solid_mpm_fsi_runner.py
```

The per-step FSI history, fixed-solid preflow history, preflow-only report, and
final report must expose the following fields where available:

```text
marker_force_z_N
fluid_reaction_force_z_N
marker_action_reaction_residual_N
scatter_action_reaction_residual_N
root_max_displacement_m

primary_face_force_n
secondary_face_force_n
primary_face_force_z_N
secondary_face_force_z_N
primary_face_marker_count
secondary_face_marker_count
primary_face_valid_marker_count
secondary_face_valid_marker_count
primary_face_invalid_marker_count
secondary_face_invalid_marker_count
max_abs_traction_pa
two_sided_pressure_marker_count
one_sided_pressure_marker_count
```

Implementation constraints:

- Prefer values directly from `HibmMpmSurfaceMarkerForceReport` and
  `HibmMpmMpmForceScatterReport`.
- Use marker arrays only for diagnostics that are not already aggregated by the
  core report, such as maximum traction magnitude and pressure-sampling counts.
- Do not change pressure sampling math, force scatter math, surface feedback
  math, MPM stepping, or source/outlet driver behavior.
- Do not hide unavailable values as successful zeros. If a value is truly not
  available for a diagnostic-only row, report an explicit blank or a documented
  not-applicable status.

## Phase 2 - Face-Resolved Marker Region Reporting

Current marker construction creates two streamwise faces but assigns all
markers to `PRIMARY_REGION_ID`.

Change the ANSYS vertical-flap benchmark runner so:

```text
+z / inlet-facing face  -> PRIMARY_REGION_ID
-z / outlet-facing face -> SECONDARY_REGION_ID
```

This must only change the marker region labels for diagnostics and the existing
core region-aware aggregation. It must not change:

```text
marker positions
marker normals
marker areas
marker velocities
force scatter behavior
surface feedback behavior
total force computation
```

Tests must assert:

```text
primary markers = marker_count
secondary markers = marker_count
total markers = 2 * marker_count
primary_force + secondary_force == total_force
primary and secondary valid/invalid counts are independently reportable
```

If a core API requires the secondary region ID for row assembly or scatter, pass
the actual `SECONDARY_REGION_ID`. If a path intentionally remains primary-only,
document and test why it does not affect this diagnostic.

## Phase 3 - Preflow-Release Matrix Corrections

Modify:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_preflow_release_step20_matrix.py
```

Rename the overclaiming field:

```text
preflow_release_state_continuity_ok
-> preflow_release_index_continuity_ok
```

Keep:

```text
preflow_release_source_factor_continuity_ok
```

Add:

```text
first_release_pressure_reset
first_release_full_field_reinitialized
```

These should describe the first release step and must be filled from runner
report fields where possible. If a scenario does not have preflow or release
history, use explicit blank/not-applicable values.

Fix release candidate selection:

```text
best_release_flow_candidate
```

must use a pure flow penalty. It may use:

```text
final_velocity_p999_mps
max_velocity_peak_mps
release_flow_last_window_min_p999_mps
release_flow_last_window_mean_outlet_ratio
release_flow_temporal_status strict/soft preference
```

It must not use:

```text
release_first_permanently_valid_step
release_coupling_settling_status
force/tip sign fields
```

If useful, add:

```text
best_release_coupling_candidate
best_release_promotion_candidate
```

but do not weaken promotion semantics.

Tighten promotion candidate checks. A future row can only be
`promotion_ready` if all current checks pass and:

```text
root_max_displacement_m <= 1e-8
marker_action_reaction_residual_N <= tolerance
scatter_action_reaction_residual_N <= tolerance
```

Use a clear tolerance constant in the runner. If current artifacts do not yet
promote any row, tests must still assert the gate fields are present and would
block missing/large values.

## Phase 4 - Fixed-Solid Hydrodynamic-Load Matrix

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_fixed_solid_load_temporal_matrix.py
```

Write artifacts under:

```text
validation_runs/ansys_vertical_flap_fsi/fixed_solid_load_temporal_diagnostics/
```

Required scenarios:

```text
fixed_load_0p75_constant_step60
fixed_load_0p80_constant_step60
fixed_load_0p75_ramp2_step60
fixed_load_0p80_ramp2_step60
projection_only_step60_baseline
diagnostic_reinitialize_step60_upper_bound
```

Use isolated worker subprocesses, following the existing fixed-solid and
preflow-release matrix pattern. Record worker return code, timeout status,
elapsed time, stdout log, and stderr log. Force-add committed worker logs if
they are ignored by `*.log` but referenced by artifact tests.

Each per-step history must include:

```text
scenario
step
flow_phase
flow_step_index_local
flow_step_index_global
flow_source_schedule_step_index
flow_source_schedule_scope
source_factor
source_normal_velocity_mps

velocity_peak_mps
velocity_p999_mps
velocity_outlet_flux_ratio
pressure_outlet_flux_ratio
pressure_min_pa
pressure_max_pa
projection_l2
projection_max_abs

total_force_z_N
primary_face_force_z_N
secondary_face_force_z_N
fluid_reaction_force_z_N
marker_action_reaction_residual_N
scatter_action_reaction_residual_N
primary_face_valid_marker_count
secondary_face_valid_marker_count
primary_face_invalid_marker_count
secondary_face_invalid_marker_count
max_abs_traction_pa
two_sided_pressure_marker_count
one_sided_pressure_marker_count
stress_invalid_marker_count
```

The summary/matrix row must include:

```text
force_z_min_N
force_z_max_N
force_z_mean_N
force_z_rms_N
force_z_zero_crossing_count
force_z_negative_fraction
last20_force_z_mean_N
last20_force_z_min_N
last20_force_z_max_N
last20_force_z_negative_fraction
last20_primary_face_force_z_mean_N
last20_secondary_face_force_z_mean_N
last20_marker_action_reaction_residual_max_N
last20_scatter_action_reaction_residual_max_N
flow_temporal_status
hydrodynamic_load_status
hydrodynamic_load_fail_reasons
```

The fixed-solid hydrodynamic-load matrix must not require every force sample to
be negative. Its job is to answer whether the force is:

```text
mostly negative and bounded
near-zero mean
periodically sign-changing
positive-dominated
invalid because marker/residual diagnostics fail
```

Recommended load gate for this diagnostic:

```text
invalid marker count = 0
last-window mean force_z < 0
negative-force fraction >= 0.8
zero-crossing count finite and reported
marker/scatter action-reaction residuals within tolerance
primary + secondary force decomposition available
```

The gate may report `load_temporal_failed`; it must not be tuned to manufacture
a pass.

## Phase 5 - Tests

Add or update tests:

```text
tests/cases/test_ansys_vertical_flap_fsi.py
tests/tools/test_ansys_vertical_flap_temporal_gate.py
tests/integration/test_ansys_vertical_flap_fixed_solid_load_artifacts.py
tests/integration/test_ansys_vertical_flap_preflow_release_step20_artifacts.py
```

Required assertions:

```text
marker region split creates primary/secondary face counts
force decomposition primary + secondary == total
fixed-solid force history has no empty residual columns
fixed-solid force history records sign statistics and face statistics
promotion gate blocks missing or excessive root/residual values
best_release_flow_candidate is selected by pure flow penalty
preflow_release_index_continuity_ok replaces state_continuity naming
first_release_pressure_reset and first_release_full_field_reinitialized exist
no 50-step, Fluent parity, solid tuning, or gate relaxation claim appears
```

## Phase 6 - Documentation And Workflow

Update:

```text
docs/VALIDATION.md
.github/workflows/ansys-vertical-flap-validation.yml
```

`docs/VALIDATION.md` must explain:

```text
fixed-solid STEP30 proves source/outlet flow only
fixed-solid load STEP60 is the hydrodynamic-load diagnostic
preflow-release failures cannot yet be blamed entirely on MPM coupling
current scope does not claim Fluent parity
```

Workflow must compile the new script and run the new artifact test without
running long simulations in CI.

## Phase 7 - Verification Commands

Run at minimum:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  benchmarks\official\solid_mpm_fsi_runner.py `
  tools\validation\ansys_vertical_flap_temporal_gates.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_fixed_solid_load_temporal_matrix.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_preflow_release_step20_matrix.py `
  tests\cases\test_ansys_vertical_flap_fsi.py `
  tests\integration\test_ansys_vertical_flap_fixed_solid_load_artifacts.py

& 'D:\working\taichi\env\python.exe' -m unittest `
  tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests `
  tests.tools.test_ansys_vertical_flap_temporal_gate `
  -v

& 'D:\working\taichi\env\python.exe' `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_fixed_solid_load_temporal_matrix.py

& 'D:\working\taichi\env\python.exe' `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_preflow_release_step20_matrix.py `
  --reclassify-existing

& 'D:\working\taichi\env\python.exe' -m unittest `
  tests.integration.test_ansys_vertical_flap_fixed_solid_load_artifacts `
  tests.integration.test_ansys_vertical_flap_preflow_release_step20_artifacts `
  -v

git diff --check
```

If runtime becomes excessive, reduce only the new fixed-solid load matrix
worker timeout or scenario count after documenting the deviation. Do not
replace real solver artifacts with synthetic data.

## Completion Criteria

The goal is complete only when:

```text
face-resolved force fields are present and tested
fixed-solid load histories include non-empty force/residual fields
fixed-solid load summary reports sign, mean, RMS, zero-crossing, and face stats
preflow-release matrix naming and candidate ranking are corrected
promotion gate includes root and residual checks
new artifacts are generated from real EasyFsi solver runs
docs and workflow are updated
verification commands pass
all relevant code, docs, logs, and artifacts are committed and pushed
```

Final reporting must include:

```text
new commit SHA
remote branch
verification commands and pass/fail status
fixed-solid hydrodynamic load conclusion
any remaining blocked scope
```
