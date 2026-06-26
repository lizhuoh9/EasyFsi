# ANSYS Vertical-Flap Shared Traction Snapshot Goal - 2026-06-26

## Source Review

This goal follows the review of current remote HEAD:

```text
b163e0a7d7b0ee2117580bd48f41c7f1fc1108c2
validation: archive ANSYS traction probe observability
```

That commit completed the fixed-solid traction probe observability evidence
loop. It did not merely expose pressure-probe diagnostics in code; it reran the
fixed-solid matrix, archived marker-level pressure/probe/decomposition evidence,
and explained why the current dual physical faces plus two-sided pressure-jump
formulation is not a reliable reference formulation:

```text
offset0p25: pressure jump is effectively duplicated on both physical faces
offset0p51: primary face carries the dominant jump while secondary is near zero
offset1p00: probes no longer straddle the thin-wall jump cleanly
```

The remaining blockers are real:

```text
required_formulation_unsupported
dual_face_one_sided_unsupported
dual_two_sided_offset_sensitivity_above_tolerance
```

The next step is not one-sided traction and not coupled FSI. The next step is to
save one fixed-solid flow snapshot that later formulation comparisons can share.

## Objective

Create a real shared fixed-solid ANSYS vertical-flap flow snapshot:

```text
validation: save ANSYS traction shared snapshot
```

The snapshot must capture the pressure, velocity, obstacle, grid, and relevant
configuration state from one fixed-solid source/outlet run at preflow step 20.
Future formulation resampling must be able to load this snapshot and sample
different traction formulations from the exact same pressure/velocity field.

This commit should establish the snapshot artifact contract and tests only. It
must not compare multiple formulations on the snapshot yet.

## Explicit Non-Goals

This task must not:

```text
implement dual-face one-sided traction
select a reference traction formulation
run snapshot resampling matrix
run coupled STEP30, STEP50, or 50-step validation
claim Fluent parity
change fluid or solid solver formulas
change marker defaults
change pressure-probe defaults
split marker position from pressure-probe start offset
retune stiffness, damping, support radius, source strength, or mesh settings
relax offset sensitivity gates
overwrite traction_probe_observability_diagnostics
put a long GPU run into the default GitHub workflow
```

This is a data artifact and contract step. The conclusion must remain that no
reference traction formulation has been selected.

## New Artifact Directory

Write outputs under:

```text
validation_runs/ansys_vertical_flap_fsi/
  traction_shared_snapshot_diagnostics/
```

Required files:

```text
step020_fields.npz
snapshot_manifest.json
snapshot_summary.md
verification_shared_snapshot_2026-06-26.md
CHECKSUMS.sha256
```

The directory must be separate from:

```text
traction_formulation_diagnostics/
traction_probe_observability_diagnostics/
```

Those older directories remain historical evidence. This new directory is the
shared field source for later resampling work.

## Runner Contract

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/
  run_traction_shared_snapshot.py
```

The runner must execute one fixed-solid source/outlet diagnostic using the same
configuration envelope as the observability matrix:

```text
case = ansys-vertical-flap-fsi
step_count = 0
preflow_steps = 20
apply_marker_feedback_to_fluid = false
flow_driver_mode = sustained_volume_source_inlet
flow_inlet_source_strength = 0.80
flow_inlet_source_profile = linear_ramp
flow_inlet_source_ramp_steps = 2
flow_inlet_source_schedule_scope = global
```

It should use the local solver path, not Fluent. It should be callable directly:

```powershell
& "D:/TOOL/Anaconda/python.exe" `
  validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_shared_snapshot.py
```

The runner should return nonzero if the snapshot is missing required fields, if
fields contain non-finite values, or if the fixed-solid run does not complete
20 preflow steps.

## NPZ Field Contract

`step020_fields.npz` must contain at least:

```text
pressure
velocity
obstacle
cell_face_x_m
cell_face_y_m
cell_face_z_m
cell_center_x_m
cell_center_y_m
cell_center_z_m
cell_width_x_m
cell_width_y_m
cell_width_z_m
```

If a separate sampling obstacle exists in the fixed-solid path, include it as:

```text
sampling_obstacle
```

The NPZ should also include compact numeric metadata arrays:

```text
grid_nodes
preflow_step
dt_s
inlet_velocity_mps
source_strength
source_ramp_steps
```

All numeric field arrays must be finite. `obstacle` and optional
`sampling_obstacle` may be integer arrays and must have the same grid shape as
`pressure`.

Expected shape contract:

```text
pressure.shape == obstacle.shape == tuple(grid_nodes)
velocity.shape == tuple(grid_nodes) + (3,)
```

## Manifest Contract

`snapshot_manifest.json` must include at least:

```text
case
purpose
scope_limit
commit_sha
source_commit
runner
created_at_utc
preflow_steps
step_count
flow_driver_mode
flow_inlet_source_strength
flow_inlet_source_profile
flow_inlet_source_ramp_steps
flow_inlet_source_schedule_scope
grid_nodes
field_path
field_sha256
field_arrays
pressure_min_pa
pressure_max_pa
velocity_peak_mps
velocity_p999_mps
velocity_outlet_flux_ratio
pressure_outlet_flux_ratio
flow_projection_report
marker_geometry
marker_geometry_sha256
reference_formulation_candidate
candidate_status
non_goal_statement
```

The manifest must explicitly state:

```text
reference_formulation_candidate = none
candidate_status = snapshot_only_no_reference_selection
scope_limit = fixed-solid shared flow snapshot only; no coupled 50-step or Fluent parity claim
```

The `field_sha256` should be computed from the actual NPZ file bytes after the
file is written. `CHECKSUMS.sha256` should include the NPZ, manifest, summary,
and verification files.

## Marker Geometry Contract

The manifest should record the marker geometry used by the baseline fixed-solid
setup, enough for later snapshot resampling scripts to confirm that the same
geometry family is being used:

```text
marker_layout
marker_face_offset_cells
pressure_sampling_mode
traction_include_viscous
traction_viscosity_pa_s
marker_count
marker_count_per_face
marker_face_count
flap_box_m
grid_nodes
solid_particle_counts
```

Also write:

```text
marker_geometry_sha256
```

computed from a stable JSON encoding of the marker geometry block.

## Summary Contract

`snapshot_summary.md` must be short but explicit. It should include:

```text
scope limit
field path
field sha256
grid shape
pressure range
velocity peak/p999
why this snapshot exists
what this snapshot does not prove
next intended step = snapshot resampling matrix
```

It must say that the snapshot removes the remaining ambiguity from prior
observability work:

```text
future formulation rows can be sampled from the exact same pressure/velocity field
```

It must also say that no reference formulation is selected.

## Tests

Add:

```text
tests/integration/
  test_ansys_vertical_flap_traction_shared_snapshot_artifacts.py
```

Tests must verify:

```text
artifact directory exists
step020_fields.npz exists
snapshot_manifest.json exists
CHECKSUMS.sha256 exists
NPZ contains required arrays
pressure/velocity arrays are finite
obstacle shape matches pressure
velocity shape is pressure shape + (3,)
manifest field_sha256 matches the NPZ bytes
manifest records preflow_steps = 20
manifest records flow_driver_mode = sustained_volume_source_inlet
manifest records source_strength = 0.80
manifest records reference_formulation_candidate = none
manifest records candidate_status = snapshot_only_no_reference_selection
manifest scope denies coupled 50-step and Fluent parity
marker_geometry_sha256 is present
summary says no reference formulation is selected
```

These tests are artifact tests. They must not rerun the GPU snapshot.

## Workflow

Update the ANSYS workflow only for cheap checks:

```text
py_compile includes run_traction_shared_snapshot.py
archive/artifact consistency tests include test_ansys_vertical_flap_traction_shared_snapshot_artifacts
```

Do not add the GPU snapshot run itself to default push or pull request jobs.

## Verification Commands

Before commit, run at least:

```powershell
& "D:/TOOL/Anaconda/python.exe" -m py_compile `
  validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_shared_snapshot.py `
  tests/integration/test_ansys_vertical_flap_traction_shared_snapshot_artifacts.py

& "D:/TOOL/Anaconda/python.exe" `
  validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_shared_snapshot.py

& "D:/TOOL/Anaconda/python.exe" -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_shared_snapshot_artifacts -v
```

Also run the previous observability artifact tests:

```powershell
& "D:/TOOL/Anaconda/python.exe" -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_probe_observability_artifacts -v
```

Run `git diff --check` and check the README contract before commit.

## Acceptance Criteria

The task is complete only when:

1. This detailed goal file exists and the short Codex goal references it.
2. `run_traction_shared_snapshot.py` exists and writes the snapshot artifacts.
3. `step020_fields.npz` contains the required pressure, velocity, obstacle, and
   grid arrays.
4. `snapshot_manifest.json` includes field hash, marker geometry hash, fixed
   source/outlet config, and explicit no-reference/no-Fluent-parity scope.
5. `CHECKSUMS.sha256` includes all new snapshot artifacts.
6. Artifact tests pass.
7. Previous observability artifact tests still pass.
8. README has been checked for consistency.
9. Changes are committed and pushed to the GitHub remote.
