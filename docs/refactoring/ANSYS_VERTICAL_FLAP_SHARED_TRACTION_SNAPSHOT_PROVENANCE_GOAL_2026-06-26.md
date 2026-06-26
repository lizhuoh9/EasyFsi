# ANSYS Vertical-Flap Shared Snapshot Provenance Goal - 2026-06-26

## Source Review

This goal follows the review of current remote HEAD:

```text
8488848d9302f7c05ffb8fd59342aec9d0a7e36f
validation: save ANSYS traction shared snapshot
```

That commit successfully added the fixed-solid shared flow snapshot contract and
archived a real `step020_fields.npz` for later traction formulation resampling.
The committed snapshot is numerically useful and the artifact test coverage is
sound:

```text
case = ansys-vertical-flap-fsi
step_count = 0
preflow_steps = 20
apply_marker_feedback_to_fluid = false
flow_driver_mode = sustained_volume_source_inlet
flow_inlet_source_strength = 0.80
flow_inlet_source_profile = linear_ramp
flow_inlet_source_ramp_steps = 2
reference_formulation_candidate = none
candidate_status = snapshot_only_no_reference_selection
```

The remaining issue is artifact provenance. The current committed manifest was
generated before commit `8488848...` existed, so it records:

```text
commit_sha = b163e0a7d7b0ee2117580bd48f41c7f1fc1108c2
source_commit = b163e0a7d7b0ee2117580bd48f41c7f1fc1108c2
```

That older commit did not yet contain the committed snapshot runner and
`export_final_flow_snapshot` path. The field data are not invalid, but a
reviewer reading only the manifest can wrongly infer that `b163e0a...` already
contained the snapshot exporter.

## Objective

Refresh the ANSYS vertical-flap shared snapshot provenance:

```text
validation: refresh ANSYS traction shared snapshot provenance
```

Rerun the existing committed snapshot runner from current source HEAD
`8488848...`, update the snapshot artifacts, and add a focused artifact test
guard so the archived manifest no longer points at the pre-runner commit.

The intended evidence chain after this task is:

```text
source_commit / commit_sha = a commit that already contains:
  - run_traction_shared_snapshot.py
  - export_final_flow_snapshot
  - snapshot artifact tests
artifact carrier commit = this task's new commit
```

The manifest `commit_sha` and `source_commit` represent the Git HEAD used by the
runner when generating the snapshot. They do not have to equal the later commit
that carries the regenerated artifact files, because that would be
self-referential. They must, however, point to a commit that contains the
runner/export code actually used to create the snapshot.

## Explicit Non-Goals

This task must not:

```text
implement run_traction_snapshot_resampling_matrix.py
implement dual-face one-sided traction
select a reference traction formulation
run coupled STEP30, STEP50, or 50-step FSI
claim Fluent parity
change solver physics
change marker defaults
change pressure-probe defaults
retune mesh, source strength, support radius, stiffness, or damping
modify old traction_formulation_diagnostics
modify old traction_probe_observability_diagnostics
put a long GPU run into the default GitHub workflow
```

This is a provenance refresh only. Shared-snapshot resampling comes next, after
the snapshot provenance is clean.

## Artifact Directory

Refresh files under:

```text
validation_runs/ansys_vertical_flap_fsi/
  traction_shared_snapshot_diagnostics/
```

Required files remain:

```text
step020_fields.npz
snapshot_manifest.json
snapshot_summary.md
verification_shared_snapshot_2026-06-26.md
CHECKSUMS.sha256
```

The refresh must leave these older evidence directories untouched:

```text
traction_formulation_diagnostics/
traction_probe_observability_diagnostics/
```

## Runner Contract

Use the existing runner:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_shared_snapshot.py
```

Run it directly from the current checkout:

```powershell
& "D:/TOOL/Anaconda/python.exe" `
  validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_shared_snapshot.py
```

The runner must still write a fixed-solid source/outlet snapshot with:

```text
step_count = 0
preflow_steps = 20
flow_driver_mode = sustained_volume_source_inlet
flow_inlet_source_strength = 0.80
flow_inlet_source_profile = linear_ramp
flow_inlet_source_ramp_steps = 2
flow_inlet_source_schedule_scope = global
export_final_flow_snapshot = true
```

## Provenance Contract

`snapshot_manifest.json` must no longer point at:

```text
b163e0a7d7b0ee2117580bd48f41c7f1fc1108c2
```

It must continue to include:

```text
commit_sha
source_commit
runner
created_at_utc
field_sha256
marker_geometry_sha256
reference_formulation_candidate
candidate_status
scope_limit
```

The provenance requirements are:

```text
commit_sha == source_commit
commit_sha is a 40-character Git SHA
source_commit is not b163e0a7d7b0ee2117580bd48f41c7f1fc1108c2
runner path points at run_traction_shared_snapshot.py
candidate_status remains snapshot_only_no_reference_selection
reference_formulation_candidate remains none
scope_limit still denies coupled 50-step and Fluent parity
```

The regenerated artifact will normally record:

```text
commit_sha = 8488848d9302f7c05ffb8fd59342aec9d0a7e36f
source_commit = 8488848d9302f7c05ffb8fd59342aec9d0a7e36f
```

because that is the committed source state used by the runner before this
provenance-refresh commit is created.

## Field Contract

The refresh must preserve the existing field contract:

```text
pressure
velocity
obstacle
sampling_obstacle, if present
cell_face_x_m
cell_face_y_m
cell_face_z_m
cell_center_x_m
cell_center_y_m
cell_center_z_m
cell_width_x_m
cell_width_y_m
cell_width_z_m
grid_nodes
preflow_step
dt_s
inlet_velocity_mps
source_strength
source_ramp_steps
velocity_outlet_flux_ratio
pressure_outlet_flux_ratio
```

Expected shape contract remains:

```text
pressure.shape == obstacle.shape == tuple(grid_nodes)
velocity.shape == tuple(grid_nodes) + (3,)
sampling_obstacle.shape == pressure.shape, when present
```

All numeric field arrays must remain finite.

## Test Contract

Update:

```text
tests/integration/
  test_ansys_vertical_flap_traction_shared_snapshot_artifacts.py
```

Add or extend a focused test that fails on the currently stale provenance and
passes after the refresh. It should check:

```text
manifest["commit_sha"] == manifest["source_commit"]
manifest["commit_sha"] is a 40-character SHA string
manifest["commit_sha"] != b163e0a7d7b0ee2117580bd48f41c7f1fc1108c2
manifest["runner"] references run_traction_shared_snapshot.py
manifest field_sha256 still matches the NPZ bytes
manifest candidate_status still snapshot_only_no_reference_selection
summary still says no reference formulation is selected
```

The test must not rerun the GPU snapshot.

## Workflow Contract

The existing GitHub workflow already compiles:

```text
run_traction_shared_snapshot.py
```

and runs:

```text
test_ansys_vertical_flap_traction_shared_snapshot_artifacts
```

Do not add the GPU snapshot generation run to default CI.

## Verification Commands

Before commit, run at least:

```powershell
& "D:/TOOL/Anaconda/python.exe" -m py_compile `
  tests/integration/test_ansys_vertical_flap_traction_shared_snapshot_artifacts.py

& "D:/TOOL/Anaconda/python.exe" `
  validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_shared_snapshot.py

& "D:/TOOL/Anaconda/python.exe" -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_shared_snapshot_artifacts -v

& "D:/TOOL/Anaconda/python.exe" -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_probe_observability_artifacts -v

git diff --check
```

Also read `README.md` and keep it unchanged unless the provenance refresh
changes user-facing behavior documented there.

## Acceptance Criteria

The task is complete only when:

1. This detailed goal file exists and the short Codex goal references it.
2. The shared snapshot artifact test contains a provenance guard.
3. The current stale manifest with `b163e0a...` would fail that guard.
4. `run_traction_shared_snapshot.py` has been rerun from a committed source SHA
   that contains the snapshot runner/export path.
5. `snapshot_manifest.json` records a non-`b163e0a...` `commit_sha` and
   `source_commit`.
6. The NPZ field contract still passes.
7. `CHECKSUMS.sha256` matches the refreshed artifacts.
8. Previous observability artifact tests still pass.
9. README has been checked for consistency.
10. Changes are committed and pushed to the GitHub remote.
