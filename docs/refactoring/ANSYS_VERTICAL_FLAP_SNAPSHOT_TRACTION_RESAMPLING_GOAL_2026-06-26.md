# ANSYS Vertical-Flap Snapshot Traction Resampling Goal - 2026-06-26

## Source Review

This goal follows the review that was written against commit:

```text
8488848d9302f7c05ffb8fd59342aec9d0a7e36f
validation: save ANSYS traction shared snapshot
```

The review correctly identified a provenance issue in that commit's shared
snapshot manifest. That issue has already been addressed by the current HEAD:

```text
db8e0c08262614ea27a69a1828a961a71bdc83f1
validation: refresh ANSYS traction shared snapshot provenance
```

The current shared snapshot manifest records:

```text
commit_sha = 8488848d9302f7c05ffb8fd59342aec9d0a7e36f
source_commit = 8488848d9302f7c05ffb8fd59342aec9d0a7e36f
field_sha256 = 3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968
candidate_status = snapshot_only_no_reference_selection
```

That means the immediate provenance blocker is resolved. The next required step
is to resample the existing supported traction formulations from exactly this
same pressure/velocity field, instead of rerunning a separate fixed-solid flow
for each formulation row.

## Objective

Implement the first shared-snapshot traction resampling matrix:

```text
validation: add ANSYS traction snapshot resampling matrix
```

The runner must load:

```text
validation_runs/ansys_vertical_flap_fsi/
  traction_shared_snapshot_diagnostics/step020_fields.npz
validation_runs/ansys_vertical_flap_fsi/
  traction_shared_snapshot_diagnostics/snapshot_manifest.json
```

Then it must rebuild a sampling-only local EasyFsi flow state and evaluate the
current supported traction formulations on the exact same archived pressure and
velocity arrays.

The main proof this task must establish is:

```text
every completed formulation row references the same flow_snapshot_sha256
```

This is not a reference-formulation selection step. It is the evidence bridge
between the fixed-solid observability work and later one-sided traction work.

## Explicit Non-Goals

This task must not:

```text
implement dual-face one-sided traction
select a reference traction formulation
change traction formulas
change marker defaults
change pressure-probe defaults
split marker offset from pressure-probe offset
run a new fixed-solid preflow per formulation
run coupled STEP30, STEP50, or 50-step FSI
claim Fluent parity
change fluid solver physics
change solid solver physics
retune source strength, mesh, support radius, stiffness, or damping
overwrite traction_formulation_diagnostics
overwrite traction_probe_observability_diagnostics
overwrite traction_shared_snapshot_diagnostics
put a GPU resampling run into the default GitHub workflow
```

The runner may use Taichi/CUDA to rebuild solver objects for sampling, but it
must not advance the flow or solid.

## New Artifact Directory

Write outputs under:

```text
validation_runs/ansys_vertical_flap_fsi/
  traction_snapshot_resampling_diagnostics/
```

Required files:

```text
traction_snapshot_resampling_matrix.json
traction_snapshot_resampling_matrix.csv
traction_snapshot_resampling_history.json
traction_snapshot_resampling_summary.md
verification_snapshot_resampling_2026-06-26.md
CHECKSUMS.sha256
marker_diagnostics/
  <scenario>_markers.json
```

The directory must remain separate from:

```text
traction_formulation_diagnostics/
traction_probe_observability_diagnostics/
traction_shared_snapshot_diagnostics/
```

## Runner Contract

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/
  run_traction_snapshot_resampling_matrix.py
```

The runner must be directly callable:

```powershell
& "D:/TOOL/Anaconda/python.exe" `
  validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_snapshot_resampling_matrix.py
```

The runner must:

```text
load shared snapshot NPZ and manifest
validate field_sha256 against the NPZ bytes
validate pressure/velocity/obstacle shape contract
rebuild a sampling-only CartesianFluidSolver
write pressure, velocity, obstacle, and grid coordinates into that solver
rebuild markers for each formulation scenario
sample marker tractions without advancing flow or solid
write matrix JSON/CSV, marker diagnostics, summary, verification, and checksums
return nonzero on missing fields, hash mismatch, shape mismatch, or incomplete diagnostics
```

The runner should reuse existing ANSYS vertical-flap configuration and existing
traction scenario definitions where possible. It must keep solver behavior in
the core and use case/validation code only as orchestration.

## Scenario Contract

The first version must resample these scenarios:

```text
dual_two_sided_offset0p25_pressure_only
dual_two_sided_offset0p51_pressure_only
dual_two_sided_offset1p00_pressure_only
single_mid_two_sided_offset0p00_pressure_only
dual_two_sided_offset0p51_viscous_air
dual_one_sided_offset0p51_pressure_only
```

The one-sided scenario must remain:

```text
run_status = unsupported
worker_mode = not_run
```

Do not silently emulate one-sided traction through another formulation.

## Row Contract

Each matrix row must include at least:

```text
scenario
run_status
marker_layout
pressure_sampling_mode
include_viscous_traction
viscosity_pa_s
marker_face_offset_cells
flow_snapshot_sha256
flow_snapshot_path
flow_snapshot_source_commit
flow_snapshot_preflow_steps
flow_snapshot_grid_nodes
flow_snapshot_pressure_min_pa
flow_snapshot_pressure_max_pa
flow_snapshot_velocity_peak_mps
flow_snapshot_velocity_p999_mps
marker_geometry_sha256
marker_diagnostics_json
total_marker_count
primary_face_marker_count
secondary_face_marker_count
primary_face_valid_marker_count
secondary_face_valid_marker_count
primary_face_invalid_marker_count
secondary_face_invalid_marker_count
primary_face_force_z_N
secondary_face_force_z_N
total_force_z_N
primary_plus_secondary_force_z_N
force_decomposition_residual_N
primary_face_pressure_complete_marker_count
secondary_face_pressure_complete_marker_count
primary_face_inside_pressure_found_marker_count
secondary_face_inside_pressure_found_marker_count
primary_face_outside_pressure_found_marker_count
secondary_face_outside_pressure_found_marker_count
primary_face_inside_probe_rung_histogram
secondary_face_inside_probe_rung_histogram
primary_face_outside_probe_rung_histogram
secondary_face_outside_probe_rung_histogram
primary_face_inside_unique_nearest_cell_count
secondary_face_inside_unique_nearest_cell_count
primary_face_outside_unique_nearest_cell_count
secondary_face_outside_unique_nearest_cell_count
primary_face_traction_decomposition_max_abs_residual_pa
secondary_face_traction_decomposition_max_abs_residual_pa
status_reason
scope_limit
```

For unsupported rows, numeric formulation fields may be empty, but the row must
still record the shared snapshot identity and a clear unsupported reason.

## Matrix-Level Contract

`traction_snapshot_resampling_matrix.json` must include:

```text
case = ansys-vertical-flap-fsi
purpose = shared-snapshot traction formulation resampling
scope_limit = shared-snapshot traction resampling only; no coupled 50-step or Fluent parity claim
input_snapshot_manifest
input_snapshot_npz
flow_snapshot_sha256
flow_snapshot_source_commit
flow_snapshot_identity_status
completed_formulation_count
unsupported_formulation_count
reference_formulation_candidate = none
candidate_status = no_reference_formulation_candidate
candidate_blockers
rows
```

Required candidate blockers:

```text
dual_face_one_sided_unsupported
dual_two_sided_offset_sensitivity_above_tolerance
```

The matrix must verify:

```text
all completed rows have identical flow_snapshot_sha256
the shared flow_snapshot_sha256 equals the shared snapshot manifest field_sha256
```

Expected offset pathology should still be visible:

```text
dual_two_sided_offset0p25 force_ratio_to_baseline > 1.5
dual_two_sided_offset1p00 force_ratio_to_baseline < 0.2
```

If those ratios differ materially from the prior observability matrix, the
summary must explain the likely difference instead of hiding it.

## Marker Diagnostics Contract

Each completed scenario must write:

```text
marker_diagnostics/<scenario>_markers.json
```

The marker JSON must include:

```text
scenario
flow_snapshot_sha256
flow_snapshot_source_commit
marker_count
marker_required_fields
face_diagnostics
row_force_summary
markers
```

Each marker must preserve the existing pressure-probe diagnostics fields used by
the observability artifact tests, including:

```text
inside_pressure_found
outside_pressure_found
inside_probe_rung
outside_probe_rung
inside_probe_nearest_cell
outside_probe_nearest_cell
pressure_traction_pa
viscous_traction_pa
total_traction_pa
traction_decomposition_residual_pa
```

## Summary Contract

`traction_snapshot_resampling_summary.md` must state:

```text
same pressure/velocity field was used for every completed row
flow_snapshot_sha256
source snapshot manifest path
completed scenarios
unsupported scenarios
reference_formulation_candidate = none
candidate blockers
offset0p25 / offset0p51 / offset1p00 interpretation
does not claim Fluent parity
does not run coupled 50-step FSI
next intended step = split marker offset from pressure-probe start offset
```

## Tests

Add:

```text
tests/integration/
  test_ansys_vertical_flap_traction_snapshot_resampling_artifacts.py
```

Tests must verify:

```text
resampling artifact directory exists
matrix JSON/CSV/history/summary/verification/checksums exist
shared snapshot artifacts exist
matrix flow_snapshot_sha256 equals shared snapshot manifest field_sha256
all completed rows use the same flow_snapshot_sha256
one-sided row is unsupported/not_run
completed rows have marker diagnostics JSON
marker diagnostics include required probe fields
traction decomposition residual is bounded
reference_formulation_candidate = none
candidate blockers include dual_face_one_sided_unsupported
candidate blockers include dual_two_sided_offset_sensitivity_above_tolerance
summary states same pressure/velocity field
summary denies Fluent parity and coupled 50-step FSI
CHECKSUMS covers matrix, summary, verification, and marker diagnostics
```

These tests must not rerun the GPU resampling runner.

## Workflow

Update the ANSYS validation workflow only for cheap checks:

```text
py_compile includes run_traction_snapshot_resampling_matrix.py
artifact consistency tests include test_ansys_vertical_flap_traction_snapshot_resampling_artifacts
```

Do not add the GPU resampling run itself to default push or pull request jobs.

## Verification Commands

Before commit, run at least:

```powershell
& "D:/TOOL/Anaconda/python.exe" -m py_compile `
  validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_snapshot_resampling_matrix.py `
  tests/integration/test_ansys_vertical_flap_traction_snapshot_resampling_artifacts.py

& "D:/TOOL/Anaconda/python.exe" `
  validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_snapshot_resampling_matrix.py

& "D:/TOOL/Anaconda/python.exe" -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_snapshot_resampling_artifacts -v

& "D:/TOOL/Anaconda/python.exe" -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_shared_snapshot_artifacts -v

& "D:/TOOL/Anaconda/python.exe" -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_probe_observability_artifacts -v

git diff --check
```

Also read `README.md` and update it only if this task changes user-facing
behavior already documented there.

## Acceptance Criteria

The task is complete only when:

1. This detailed goal file exists and the short Codex goal references it.
2. `run_traction_snapshot_resampling_matrix.py` exists.
3. The runner loads the committed shared snapshot NPZ/manifest and validates
   the NPZ hash.
4. The runner writes all required resampling artifacts.
5. Completed rows all reference the exact same `flow_snapshot_sha256`.
6. The one-sided row stays unsupported/not_run.
7. `reference_formulation_candidate` remains `none`.
8. Candidate blockers still include unsupported one-sided traction and
   offset-sensitive dual/two-sided traction.
9. Artifact tests pass without rerunning GPU work.
10. Previous shared snapshot and observability artifact tests still pass.
11. README has been checked for consistency.
12. Changes are committed and pushed to the GitHub remote.
