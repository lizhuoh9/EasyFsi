# ANSYS Vertical-Flap Traction Probe Observability Matrix Goal - 2026-06-26

## Source Review

This goal follows the review of current HEAD:

```text
8d803d18033f20d62fe5671403fab652a1055fa8
fix: complete ANSYS traction probe diagnostics
```

That commit completed the Phase 0 diagnostics hardening requested after
`99c1a81d1816bfe4d35483bbefc3c3b78adc0cb4`:

1. Marker-level pressure probe diagnostics now expose machine-readable pressure,
   rung, multiplier, grid-coordinate, nearest-cell, fluid-weight, invalid-reason,
   and traction-decomposition evidence.
2. Face-level diagnostics now expose pressure missing counts, found counts,
   base/inside/outside means, rung histograms, unique nearest-cell counts,
   distance stats, and decomposition residuals.
3. The traction formulation gate is fail-closed when pressure/probe/decomposition
   evidence is missing.
4. New CUDA targeted tests exercise real Taichi values rather than source-text
   contracts.
5. The old matrix artifacts remain correctly fail-closed because they were
   produced before the new observability fields existed.

The next step is not another reclassification and not a one-sided traction
implementation. The next step is to run the fixed-solid traction matrix again
with the new diagnostics active, archive per-marker probe evidence, and explain
why the current dual/two-sided pressure-jump formulation is extremely sensitive
to marker/probe offset.

## Objective

Create and run a new ANSYS vertical-flap traction probe observability matrix:

```text
validation: archive ANSYS traction probe observability
```

The run must use the local HIBM-MPM solver and the official ANSYS vertical-flap
fixed-solid diagnostic setup. It must produce a new artifact directory, separate
from the older `traction_formulation_diagnostics`, containing matrix rows,
histories, worker logs, marker-level probe JSON files, checksums, and a written
mechanism explanation for the offset pathology.

The expected final validation conclusion remains:

```text
reference_formulation_candidate = none
```

This is correct until physical dual-face one-sided traction exists and has been
validated against a shared flow snapshot.

## Non-Goals

This task must not:

```text
implement dual-face one-sided traction
select a reference traction formulation
run coupled STEP30, STEP50, or 50-step validation
claim Fluent parity
change the fluid or solid formulas
retune solid stiffness, damping, support radius, source strength, or mesh settings
change marker geometry defaults
split marker-surface offset from pressure-probe start offset
relax offset sensitivity gates
overwrite the old traction_formulation_diagnostics directory
convert old artifacts into fake new probe evidence
put the long GPU matrix into the default GitHub workflow
```

The output is evidence for the next physics change, not a physics fix.

## New Artifact Directory

Write new outputs under:

```text
validation_runs/ansys_vertical_flap_fsi/
  traction_probe_observability_diagnostics/
```

Do not overwrite:

```text
validation_runs/ansys_vertical_flap_fsi/
  traction_formulation_diagnostics/
```

The old directory represents pre-observability artifacts plus fail-closed
reclassification. The new directory represents a real post-observability GPU
rerun.

Required files:

```text
traction_probe_observability_matrix.json
traction_probe_observability_matrix.csv
traction_probe_observability_history.json
traction_probe_observability_summary.md
verification_traction_probe_observability_2026-06-26.md
CHECKSUMS.sha256

histories/
  <scenario>_history.csv

marker_diagnostics/
  <scenario>_step020_markers.json

worker_logs/
  <scenario>_stdout.log
  <scenario>_stderr.log

failures/
  <scenario>_failure.json
```

## Scenarios

Run these supported scenarios with isolated worker subprocesses:

```text
dual_two_sided_offset0p25_pressure_only
dual_two_sided_offset0p51_pressure_only
dual_two_sided_offset1p00_pressure_only
single_mid_two_sided_offset0p00_pressure_only
dual_two_sided_offset0p51_viscous_air
```

Keep this unsupported row, but do not execute a solver worker for it:

```text
dual_one_sided_offset0p51_pressure_only = unsupported
```

All supported scenarios must use the same fixed-solid diagnostic envelope:

```text
step_count = 0
preflow_steps = 20
apply_marker_feedback_to_fluid = false
flow_driver_mode = sustained_volume_source_inlet
flow_inlet_source_strength = 0.80
flow_inlet_source_profile = linear_ramp
flow_inlet_source_ramp_steps = 2
flow_inlet_source_schedule_scope = global
```

This remains a fixed-solid traction diagnostic. It is not a coupled moving-solid
FSI validation.

## Runner Contract

Add a new script:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/
  run_traction_probe_observability_matrix.py
```

The script may reuse constants, scenario configs, row hydration, baseline
comparisons, and candidate-gate functions from:

```text
run_traction_formulation_validation_matrix.py
```

but it must write to the new observability directory and must export marker
diagnostics JSON for every completed supported scenario.

The script should support:

```text
--single-scenario <name>
--single-output <path>
```

so each scenario can run in an isolated subprocess. It should also support the
default full matrix mode.

## Marker Diagnostics JSON Contract

For every completed supported scenario, write:

```text
marker_diagnostics/<scenario>_step020_markers.json
```

Each file must contain:

```text
scenario
preflow_step = 20
marker_count
markers
```

Every marker entry must include:

```text
marker_index
region_id
position_m
normal

valid
invalid_reason_code
invalid_reason

base_pressure_found
inside_pressure_found
outside_pressure_found

base_pressure_pa
inside_pressure_pa
outside_pressure_pa
pressure_jump_pa

fluid_side_pressure_defined
fluid_side_pressure_pa
reference_pressure_pa

inside_probe_ladder_mode
outside_probe_ladder_mode
inside_probe_rung
outside_probe_rung
inside_probe_multiplier
outside_probe_multiplier
inside_probe_distance_m
outside_probe_distance_m
inside_probe_grid_coordinate
outside_probe_grid_coordinate
inside_probe_nearest_cell
outside_probe_nearest_cell
inside_probe_fluid_weight
outside_probe_fluid_weight

pressure_traction_pa
viscous_traction_pa
total_traction_pa
traction_decomposition_residual_pa
```

The file should also include face-level diagnostic fields for the scenario so
reviewers can connect marker-level evidence to matrix row fields.

## Summary Explanation Contract

The summary must explain mechanism, not just ratios.

It must include sections for:

```text
offset0p25
offset0p51
offset1p00
```

For `offset0p25`, the summary must state whether primary and secondary faces
both sampled a near-complete pressure jump and whether that duplicates load
across the two physical faces.

For `offset0p51`, the summary must state whether the primary face sampled the
dominant jump and whether the secondary face sampled same-side or nearly equal
inside/outside pressure.

For `offset1p00`, the summary must state whether probes moved away from the
thin-wall jump or sampled the same pressure region, and whether nearest-cell
evidence supports that conclusion.

The summary must keep the scope explicit:

```text
fixed-solid traction probe observability only; no coupled 50-step or Fluent parity claim
```

## Expected Candidate Gate Behavior

After the real rerun:

```text
reference_formulation_candidate = none
candidate_status = no_reference_formulation_candidate
```

The following blockers should remain:

```text
dual_face_one_sided_unsupported
dual_two_sided_offset_sensitivity_above_tolerance
```

The following old-observability blockers should disappear from the new payload
if the rerun succeeded:

```text
pressure_probe_diagnostics_incomplete
primary_pressure_probe_diagnostics_incomplete
secondary_pressure_probe_diagnostics_incomplete
probe_rung_diagnostics_incomplete
probe_cell_diagnostics_incomplete
traction_decomposition_missing
```

If the new matrix unexpectedly selects a reference formulation, stop and fix
the gate. Do not accept a candidate from dual/two-sided offset-sensitive rows.

## Tests

Add:

```text
tests/integration/
  test_ansys_vertical_flap_traction_probe_observability_artifacts.py
```

Tests must verify:

```text
all supported scenarios completed
the unsupported one-sided scenario was archived but not run
each completed scenario has marker diagnostics JSON
each completed history has 20 rows
matrix pressure/probe fields are populated
inside/outside found counts match valid counts for completed two-sided rows
rung histograms are non-empty
unique nearest-cell counts are positive
traction decomposition residuals are within tolerance
reference_formulation_candidate remains none
candidate blockers do not include pressure_probe_diagnostics_incomplete
candidate blockers still include dual_face_one_sided_unsupported
candidate blockers still include dual_two_sided_offset_sensitivity_above_tolerance
offset0p25 ratio remains high enough to show duplication
offset1p00 ratio remains low enough to show lost/attenuated jump
summary contains offset0p25, offset0p51, and offset1p00 mechanism explanations
```

Use artifact tests for archived evidence; do not make the default test suite
rerun the long GPU matrix.

## Workflow

Update the ANSYS workflow only for cheap checks:

```text
py_compile includes run_traction_probe_observability_matrix.py
artifact tests include test_ansys_vertical_flap_traction_probe_observability_artifacts
```

Do not add the long matrix execution to default push or pull request jobs.

## Verification

Before push, run at least:

```powershell
& "D:/TOOL/Anaconda/python.exe" -m py_compile `
  benchmarks/official/solid_mpm_fsi_runner.py `
  validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_probe_observability_matrix.py `
  tests/integration/test_ansys_vertical_flap_traction_probe_observability_artifacts.py

& "D:/TOOL/Anaconda/python.exe" `
  validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_probe_observability_matrix.py

& "D:/TOOL/Anaconda/python.exe" -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_probe_observability_artifacts -v
```

Also run the existing focused traction artifact tests to ensure the old archive
contract still holds:

```powershell
& "D:/TOOL/Anaconda/python.exe" -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_formulation_artifacts -v
```

Run `git diff --check` before commit.

## Acceptance Criteria

The task is complete only when:

1. The new goal file exists and the short Codex goal references it.
2. The new observability runner exists and can run isolated workers.
3. The new artifact directory exists and contains the required matrix, history,
   marker diagnostics, summary, verification, logs, and checksums.
4. Five supported scenarios completed from a real run.
5. Unsupported one-sided scenario is archived as unsupported and not run.
6. Marker JSON contains complete pressure/probe/decomposition evidence.
7. Summary explains offset0p25, offset0p51, and offset1p00 mechanisms.
8. Candidate remains fail-closed for one-sided unsupported and dual/two-sided
   offset sensitivity.
9. The new artifact tests pass.
10. README has been checked for consistency.
11. Changes are committed and pushed to the GitHub remote.

