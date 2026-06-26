# ANSYS Vertical-Flap Traction Pressure-Probe Observability Goal - 2026-06-26

## Source Review

This goal follows the review of commit:

```text
90184da9b73128ffc3819a78036e4c5c04a80dcf
fix: harden ANSYS traction formulation gates
```

That commit correctly established the current fixed-solid traction matrix
state:

```text
baseline_scenario = dual_two_sided_offset0p51_pressure_only
reference_formulation_candidate = none
candidate_status = no_reference_formulation_candidate
```

The current candidate blockers are:

```text
pressure_probe_diagnostics_incomplete
required_formulation_unsupported
dual_face_one_sided_unsupported
dual_two_sided_offset_sensitivity_above_tolerance
```

The reclassified artifacts correctly record the offset pathology:

```text
offset_force_ratio_min = 0.06751512975451793
offset_force_ratio_max = 1.9507284752819152
flow_snapshot_identity_status = flow_metrics_match_completed_rows
```

This means the previous commit achieved its immediate purpose: the current
0.51-cell dual/two-sided row is only a comparison baseline and cannot be
promoted to a reference traction formulation.

## Physical State To Preserve

No new coupled simulation or GPU matrix is required for the first commit under
this goal. The physical conclusion remains:

1. `dual_physical_faces + two_sided_pressure_jump` is extremely sensitive to
   marker offset.
2. `dual_two_sided_offset0p51_pressure_only` is a baseline, not a reference.
3. `dual_physical_faces + one_sided_surface_pressure` is not safely implemented
   because the current core exposes only one one-sided pressure region.
4. Per-face pressure/probe diagnostics are not yet exposed.
5. No reference formulation can be selected.
6. Coupled STEP30/STEP50/50-step validation remains blocked.
7. No Fluent parity claim may be made.

## Immediate Commit Scope

The immediate implementation target is:

```text
fix: fail closed ANSYS traction diagnostics
```

This commit is intentionally small and gate-focused. It must not implement the
full pressure-probe observability feature yet. It must not change the numeric
traction formula, pressure solve, flow update, marker feedback, MPM scatter, or
MPM advance. It must not rerun the GPU traction matrix unless a local artifact
regeneration bug makes reclassification impossible.

Required immediate changes:

1. Make direct runner validation reject unsupported fixed-solid traction
   formulations.
2. Keep matrix unsupported-row handling intact: matrix orchestration may still
   call `traction_formulation_supported()` before the runner and archive an
   unsupported row without executing the solver.
3. Make completed-row quality gates fail closed when required diagnostics are
   missing or non-finite.
4. Stop treating missing residual/invalid-count fields as zero.
5. Stop treating missing total force or marker-count fields as acceptable.
6. Add explicit blockers for missing diagnostics:

   ```text
   invalid_marker_count_missing
   force_decomposition_residual_missing
   marker_action_reaction_residual_missing
   scatter_action_reaction_residual_missing
   total_force_missing
   marker_count_missing
   ```

7. Preserve existing above-tolerance blockers:

   ```text
   invalid_marker_count_nonzero
   force_decomposition_residual_above_tolerance
   marker_action_reaction_residual_above_tolerance
   scatter_action_reaction_residual_above_tolerance
   full_field_reset_used
   pressure_probe_diagnostics_incomplete
   formulation_disagreement_above_tolerance
   flow_snapshot_identity_mismatch
   ```

8. Add source-level tests for direct runner rejection of unsupported fixed-solid
   formulations.
9. Add gate-level tests for missing diagnostics on completed rows.
10. Add gate-level tests for full-field reset, formulation disagreement, and
    flow snapshot mismatch.
11. Reclassify existing traction formulation artifacts after the script change.
12. Update validation docs if the fail-closed behavior changes the documented
    contract.
13. Verify with the fast local workflow and push only after the checks pass.

## Immediate Non-Goals

The fail-closed commit must not implement these items:

```text
stress_inside_pressure_pa
stress_outside_pressure_pa
stress_pressure_jump_pa
stress_fluid_side_pressure_pa
stress_reference_pressure_pa
stress_inside_pressure_found
stress_outside_pressure_found
stress_inside_probe_rung
stress_outside_probe_rung
stress_inside_probe_distance_m
stress_outside_probe_distance_m
stress_inside_probe_cell
stress_outside_probe_cell
stress_probe_mode
stress_invalid_reason
t_pressure_gamma_pa
t_viscous_gamma_pa
t_gamma_pa
per-marker one_sided_fluid_side_normal_sign
dual-face physical one-sided formula
single-snapshot NPZ resampling
flow_snapshot_sha256
STEP60/STEP120/STEP200 reruns
coupled STEP30/STEP50
Fluent force-history import
Fluent parity claim
coupling subiterations
```

## Direct Runner Fail-Closed Contract

`benchmarks/official/solid_mpm_fsi_runner.py` must enforce the same support
surface regardless of whether the call originates from the matrix script or a
direct case invocation.

The runner must reject:

```text
dual_physical_faces + one_sided_surface_pressure
single_mid_surface + one_sided_surface_pressure
```

The runner must continue to allow fixed-solid diagnostic variants that are
supported:

```text
dual_physical_faces + two_sided_pressure_jump + offset sweep + step_count=0
single_mid_surface + two_sided_pressure_jump + step_count=0
dual_physical_faces + two_sided_pressure_jump + viscous + step_count=0
```

Non-default supported formulations must remain forbidden when `step_count > 0`.

## Matrix Gate Fail-Closed Contract

For every completed row, the gate must require these fields to be present,
finite where numeric, and physically valid:

```text
total_force_z_N
total_marker_count
primary_face_invalid_marker_count
secondary_face_invalid_marker_count
force_decomposition_residual_N
marker_action_reaction_residual_N
scatter_action_reaction_residual_N
```

Missing or non-finite values must block candidate promotion. They must not be
converted to zero by `_float_or_zero()`.

Invalid marker counts must block as follows:

```text
missing or non-finite -> invalid_marker_count_missing
nonzero              -> invalid_marker_count_nonzero
```

Residuals must block as follows:

```text
missing or non-finite -> <residual>_missing
above tolerance       -> <residual>_above_tolerance
```

The row force and marker-count checks must block as:

```text
missing/non-finite total force   -> total_force_missing
missing/non-finite marker count  -> marker_count_missing
marker count <= 0                -> marker_count_missing
```

## Required Tests

Add or extend tests so the following are executable, not only documented:

```text
direct fixed-solid dual-one-sided runner config is rejected
direct fixed-solid single-mid one-sided runner config is rejected
completed row with missing marker residual is blocked
completed row with missing scatter residual is blocked
completed row with missing force decomposition residual is blocked
completed row with missing invalid marker count is blocked
completed row with missing total force is blocked
completed row with missing marker count is blocked
completed row with full-field reset is blocked
completed row with formulation disagreement > 10% is blocked
completed rows with mismatched flow snapshots are blocked
stable synthetic rows still promote only when all blockers are absent
```

## Artifact Reclassification

The existing traction artifacts under:

```text
validation_runs/ansys_vertical_flap_fsi/traction_formulation_diagnostics/
```

must be reclassified with:

```powershell
& 'D:\working\taichi\env\python.exe' `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_formulation_validation_matrix.py `
  --reclassify-existing
```

This must only read the existing matrix/history and rewrite derived artifacts.
It must not launch worker subprocesses or rerun the GPU matrix.

Expected candidate result remains:

```text
reference_formulation_candidate = none
candidate_status = no_reference_formulation_candidate
```

## Fast Verification List

At minimum, run:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  benchmarks\official\solid_mpm_fsi_runner.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_formulation_validation_matrix.py `
  tests\cases\test_ansys_vertical_flap_fsi.py `
  tests\integration\test_ansys_vertical_flap_traction_formulation_artifacts.py

& 'D:\working\taichi\env\python.exe' -m unittest -v `
  tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_traction_formulation_controls_are_explicit_and_report_unsupported

& 'D:\working\taichi\env\python.exe' -m unittest -v `
  tests.integration.test_ansys_vertical_flap_traction_formulation_artifacts

git diff --check
```

If time and environment allow, also run the ANSYS validation workflow subset
documented in `.github/workflows/ansys-vertical-flap-validation.yml`.

## Later Pressure-Probe Observability Scope

After the fail-closed commit, the next physics-bearing work is pressure-probe
observability. That later stage must expose per-marker evidence from the core,
not infer it after the fact in Python reports.

The later core work should expose:

```text
stress_inside_pressure_pa
stress_outside_pressure_pa
stress_pressure_jump_pa
stress_fluid_side_pressure_pa
stress_reference_pressure_pa
stress_inside_pressure_found
stress_outside_pressure_found
stress_inside_probe_rung
stress_outside_probe_rung
stress_inside_probe_distance_m
stress_outside_probe_distance_m
stress_inside_probe_cell
stress_outside_probe_cell
stress_probe_mode
stress_invalid_reason
```

The later traction decomposition must expose:

```text
t_pressure_gamma_pa
t_viscous_gamma_pa
t_gamma_pa
```

and enforce:

```text
t_pressure_gamma_pa + t_viscous_gamma_pa == t_gamma_pa
```

Both the pressure-only fast path and the viscous/general path must fill the same
diagnostic fields. The runner must aggregate them per face without reading
private fields directly.

## Later One-Sided Traction Scope

Only after pressure-probe evidence explains the current offset pathology should
the project implement physical per-face one-sided traction.

The later per-marker attributes should include:

```text
pressure_sampling_mode
one_sided_fluid_side_normal_sign
one_sided_reference_pressure_pa
```

The one-sided formula should be:

```text
traction = -p_fluid * normal
```

`region_id` may be used for diagnostics, but it must not decide physical
sampling direction.

## Later Snapshot Resampling Scope

The long-term matrix must stop comparing separate flow runs as if they were the
same field. It should:

```text
run one fixed-solid flow
save pressure / velocity / obstacle / coordinates
resample multiple formulations against that same snapshot
record flow_snapshot_sha256 per row
```

Recommended artifacts:

```text
step020_fields.npz
step040_fields.npz
step060_fields.npz
snapshot_manifest.json
CHECKSUMS.sha256
```

## Later Per-Formulation Gate Scope

After one-sided traction is implemented, the global gate must become a
per-formulation gate:

```text
dual_two_sided:
    rejected_offset_sensitive

dual_one_sided:
    candidate / rejected

single_mid_two_sided:
    candidate / rejected
```

Reference selection must come from formulations that pass their own probe
sensitivity, snapshot consistency, residual, marker validity, and analytical
invariant checks. It must not always return the baseline scenario.

## Done Criteria For This Goal's Immediate Commit

The immediate commit is done only when:

1. unsupported direct runner configurations fail before solver execution;
2. completed rows with missing diagnostics add blockers instead of passing;
3. existing artifacts still report `reference_formulation_candidate = none`;
4. tests cover the new fail-closed paths;
5. docs describe the stricter gate honestly;
6. no coupled 50-step or Fluent parity claim is introduced;
7. local verification passes;
8. the commit is pushed to the configured GitHub remote.
