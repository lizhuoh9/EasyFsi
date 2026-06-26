# ANSYS Vertical-Flap Traction Probe Diagnostics Hardening Goal - 2026-06-26

## Source Review

This goal follows the review of commit:

```text
99c1a81d1816bfe4d35483bbefc3c3b78adc0cb4
fix: expose ANSYS traction pressure probes
```

That commit established the correct pressure-probe observability skeleton:

1. `HibmMpmSurfaceMarkers` now stores pressure and traction decomposition
   fields beside `t_gamma_pa`.
2. `stress_marker_diagnostics()` exposes per-marker pressure-probe evidence.
3. `stress_face_diagnostics()` exposes public face-level aggregates, and the
   formal ANSYS runner no longer reads private Taichi stress fields directly.
4. The traction formulation matrix archives new pressure-probe columns and
   keeps old artifacts fail-closed instead of fabricating data.
5. Candidate promotion still correctly reports:

   ```text
   reference_formulation_candidate = none
   ```

The review also found that this is still not enough to audit the offset
pathology. The next commit must harden the diagnostics contract and add real
numeric tests before any new reference formulation can be selected or any
coupled validation can resume.

## Objective

Complete the small observability-hardening step for ANSYS vertical-flap
traction probes:

```text
fix: complete ANSYS traction probe diagnostics
```

The goal is to make pressure-probe diagnostics machine-auditable and
numerically testable. The commit must answer whether each valid marker has
complete pressure evidence, where each pressure probe sampled, whether the
inside/outside means excluded sentinel values, and whether:

```text
total traction = pressure traction + viscous traction
```

holds at marker and face levels.

## Immediate Scope

This goal is limited to Phase 0 from the review. It should produce one focused
code/test/docs commit. It should not rerun the expensive traction matrix unless
all cheap checks pass and the user explicitly asks for the real rerun in this
same turn.

Required immediate changes:

1. Add missing public marker diagnostics fields.
2. Add missing public face diagnostics aggregates.
3. Fix face means so pressure sentinel values are not averaged when the
   corresponding `*_pressure_found` flag is false.
4. Clarify probe-cell naming as nearest-cell diagnostics, not full trilinear
   stencil evidence.
5. Add ladder-mode and multiplier diagnostics so rung histograms are not
   interpreted across incompatible ladders.
6. Clarify `fluid_side_pressure_pa` semantics for two-sided mode rather than
   presenting it as a unique physical fluid side.
7. Make `set_marker_tractions_pa()` consistent with the new decomposition
   fields.
8. Add fail-closed matrix blockers for incomplete probe, cell, rung, pressure
   count, and traction-decomposition evidence.
9. Add numeric tests that exercise real Taichi values rather than only checking
   source text.
10. Add the new targeted tests to the ANSYS workflow using CPU-safe test
    methods.

## Explicit Non-Goals

This commit must not:

```text
select a reference traction formulation
claim Fluent parity
run coupled STEP30, STEP50, or 50-step validation
implement dual-face one-sided traction
change the traction formula
change marker geometry defaults
split marker position from pressure-probe start offset
unify pressure-only and general probe ladders
write shared flow snapshots
retune stiffness, damping, support radius, source strength, or mesh settings
relax offset sensitivity gates
convert old artifacts into fake new pressure-probe evidence
```

If an artifact was produced before these fields existed, it must stay
fail-closed until a real rerun writes the new evidence.

## Core/API Contract

Extend `simulation_core/coupling/hibm_mpm/core.py` through the existing
`HibmMpmSurfaceMarkers` diagnostics path. Do not create a parallel sidecar API.

### Marker Diagnostics

`stress_marker_diagnostics()` must include at least:

```text
invalid_reason_code
invalid_reason

base_pressure_found
inside_pressure_found
outside_pressure_found
base_pressure_pa
inside_pressure_pa
outside_pressure_pa
pressure_jump_pa

fluid_side_pressure_pa
fluid_side_pressure_defined
reference_pressure_pa

inside_probe_rung
outside_probe_rung
inside_probe_ladder_mode
outside_probe_ladder_mode
inside_probe_multiplier
outside_probe_multiplier
inside_probe_distance_m
outside_probe_distance_m

inside_probe_cell
outside_probe_cell
inside_probe_nearest_cell
outside_probe_nearest_cell
inside_probe_grid_coordinate
outside_probe_grid_coordinate
inside_probe_fluid_weight
outside_probe_fluid_weight

pressure_traction_pa
viscous_traction_pa
total_traction_pa
traction_decomposition_residual_pa
```

Compatibility rule:

```text
keep existing names such as inside_probe_cell and outside_probe_cell
add nearest-cell aliases instead of removing old keys
```

Sentinel rule:

```text
pressure value       = 0.0 and valid only when found flag is true
probe rung           = -1
probe multiplier     = 0.0 or another explicit unset sentinel
probe distance       = -1.0
probe cell           = (-1, -1, -1)
grid coordinate      = (-1.0, -1.0, -1.0)
fluid weight         = 0.0 unless the sampled cell is known to be fluid
invalid reason code  = explicit integer code
```

### Face Diagnostics

`stress_face_diagnostics()` must aggregate primary and secondary faces with
layout-compatible semantics. Each face payload should include:

```text
marker_count
valid_marker_count
invalid_marker_count

pressure_complete_marker_count
pressure_missing_marker_count
base_pressure_found_marker_count
inside_pressure_found_marker_count
outside_pressure_found_marker_count

mean_base_pressure_pa
mean_inside_pressure_pa
mean_outside_pressure_pa
mean_pressure_jump_pa
mean_fluid_side_pressure_pa
mean_reference_pressure_pa

inside_probe_rung_histogram
outside_probe_rung_histogram
inside_probe_distance_min_m
inside_probe_distance_mean_m
inside_probe_distance_max_m
outside_probe_distance_min_m
outside_probe_distance_mean_m
outside_probe_distance_max_m

inside_unique_nearest_cell_count
outside_unique_nearest_cell_count

mean_pressure_traction_z_pa
mean_viscous_traction_z_pa
mean_total_traction_z_pa
traction_decomposition_max_abs_residual_pa
traction_decomposition_invalid_marker_count
```

Mean rule:

```text
mean_inside_pressure_pa  uses only markers with inside_pressure_found=true
mean_outside_pressure_pa uses only markers with outside_pressure_found=true
mean_base_pressure_pa    uses only markers with base_pressure_found=true
```

Missing values must remain blank or `None` at report level. They must not be
converted to numeric zero in a way that could pass gates accidentally.

### Fluid-Side Semantics

For two-sided pressure-jump mode, there is no unique physical fluid side for
single-mid surfaces, and dual physical faces need explicit side metadata before
the field can mean a physical one-sided water pressure. Therefore the public
diagnostics must expose either:

```text
fluid_side_pressure_defined = false
```

for two-sided mode, or a clearly named non-physical alias such as:

```text
selected_water_pressure_pa
```

The code may keep `fluid_side_pressure_pa` for compatibility, but reports and
docs must not treat it as the definitive physical one-sided pressure in
two-sided mode.

### Probe Cell Semantics

The current diagnostic cell is:

```text
floor(grid_coordinate + 0.5)
```

It is a nearest rounded cell for debugging. It is not the full trilinear
stencil. The code must therefore expose:

```text
inside_probe_nearest_cell
outside_probe_nearest_cell
```

and documentation/tests must avoid implying that this single cell is the whole
interpolation stencil.

### Rung Semantics

Pressure-only and general/two-sided paths currently use different ladders:

```text
pressure-only: multiplier = 1, 2, 3
general:       multiplier = 1.0, 1.5, 2.0, 2.5, 3.0
```

Do not unify the ladders in this commit because that may change numerical
behavior. Instead, expose:

```text
probe_ladder_mode
probe_multiplier
```

so rung histograms remain interpretable.

### External Traction Semantics

`set_marker_tractions_pa()` must not leave public diagnostics internally
inconsistent. Use one explicit convention:

```text
external total traction => pressure_traction = total, viscous_traction = 0
external total traction => decomposition residual = 0
probe evidence remains unset unless it was sampled later
```

If the code chooses an `unknown_decomposition` flag instead, that flag must be
public and gates must treat it fail-closed. Prefer the first convention for the
current small hardening commit.

## Matrix/Gate Contract

Extend `validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_formulation_validation_matrix.py`.

New or preserved matrix columns should include layout-safe face fields:

```text
primary_face_pressure_missing_marker_count
secondary_face_pressure_missing_marker_count
primary_face_base_pressure_found_marker_count
secondary_face_base_pressure_found_marker_count
primary_face_inside_pressure_found_marker_count
secondary_face_inside_pressure_found_marker_count
primary_face_outside_pressure_found_marker_count
secondary_face_outside_pressure_found_marker_count

primary_face_mean_base_pressure_pa
secondary_face_mean_base_pressure_pa
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
primary_face_traction_decomposition_invalid_marker_count
secondary_face_traction_decomposition_invalid_marker_count
```

Add fail-closed blockers:

```text
pressure_complete_count_inconsistent
probe_rung_diagnostics_incomplete
probe_cell_diagnostics_incomplete
traction_decomposition_missing
traction_decomposition_above_tolerance
```

For two-sided completed rows, require:

```text
pressure_complete_marker_count == valid_marker_count
inside_pressure_found_marker_count == valid_marker_count
outside_pressure_found_marker_count == valid_marker_count
rung diagnostics complete for required faces
nearest-cell diagnostics complete for required faces
traction decomposition residual finite and below tolerance
```

Layout rule:

```text
dual_physical_faces:
    require primary and secondary face evidence

single_mid_surface:
    require primary face evidence only
    do not block only because secondary face fields are blank
```

Tolerance rule:

Use a tight absolute tolerance for traction decomposition residuals. The
expected value for exact decomposition is zero; use only a small numerical
tolerance needed for floating-point conversion.

## Test Contract

Add a dedicated numeric test module:

```text
tests/solvers/test_hibm_traction_probe_diagnostics.py
```

The tests must exercise real values, not just `inspect.getsource()`.

Required tests:

1. Uniform pressure:

   ```text
   two-sided pressure jump = 0
   inside/outside pressure are found and correct
   traction decomposition residual = 0
   ```

2. Piecewise pressure jump:

   ```text
   inside = 5 Pa
   outside = 1 Pa
   jump = 4 Pa
   traction sign is correct for the configured normal
   rung/distance/nearest-cell are not sentinel
   ```

3. Missing probe:

   ```text
   found=false where expected
   rung=-1
   distance=-1
   nearest cell=(-1,-1,-1)
   invalid_reason_code and invalid_reason agree
   ```

4. Pressure-only and general path equivalence:

   ```text
   with viscosity=0 and the same simple pressure field,
   pressure traction values match within tolerance
   ```

5. Base viscous split:

   ```text
   known linear velocity gradient
   total traction = pressure traction + viscous traction
   residual = 0 within tolerance
   ```

6. Reset semantics:

   ```text
   a second stress sampling call cannot retain the first call's pressure,
   rung, nearest-cell, or traction decomposition data
   ```

7. Public face API:

   ```text
   found counts, missing counts, rung histograms,
   unique nearest-cell counts, pressure means, and residual max
   match the marker diagnostics
   ```

Extend existing artifact/gate tests to cover:

```text
pressure_complete_count_inconsistent
probe_rung_diagnostics_incomplete
probe_cell_diagnostics_incomplete
traction_decomposition_missing
traction_decomposition_above_tolerance
single_mid_surface does not require secondary evidence
dual_physical_faces requires secondary evidence
```

## CI Contract

Update:

```text
.github/workflows/ansys-vertical-flap-validation.yml
```

Add only targeted test methods from
`tests/solvers/test_hibm_traction_probe_diagnostics.py`. Do not add the whole
GPU-heavy `tests.solvers.test_hibm` module to the workflow.

Important runtime constraint discovered during implementation:

```text
simulation_core/runtime.py intentionally rejects arch="cpu"
```

Therefore the workflow test step must be CUDA-gated. On default GitHub Windows
runners without CUDA, these tests must skip honestly rather than pretending that
CPU solver coverage exists. A CUDA runner may opt in with:

```text
HIBM_RUN_CUDA_TRACTION_PROBE_TESTS=1
```

The workflow evidence state after this commit should be reported honestly:

```text
local focused checks passed
remote GitHub Actions status unavailable unless explicitly verified
```

## Artifact Policy

This Phase 0 commit may update derived matrix artifacts only through
`--reclassify-existing`. It must not claim that old histories contain new
pressure-probe evidence.

If existing rows predate the new fields, the correct status is still:

```text
pressure_probe_diagnostics_incomplete
```

and, where applicable:

```text
primary_pressure_probe_diagnostics_incomplete
secondary_pressure_probe_diagnostics_incomplete
probe_rung_diagnostics_incomplete
probe_cell_diagnostics_incomplete
traction_decomposition_missing
```

The real rerun belongs to a later commit:

```text
validation: archive ANSYS traction probe evidence
```

## Done Criteria

The goal is complete when:

1. Marker diagnostics expose the missing fields and keep sentinel semantics.
2. Face diagnostics aggregate counts, means, histograms, unique nearest cells,
   and traction decomposition residuals.
3. Face pressure means filter by corresponding found flags.
4. `set_marker_tractions_pa()` no longer creates inconsistent public
   decomposition data.
5. Matrix gate adds the new fail-closed blockers and keeps the candidate
   blocked.
6. Numeric tests prove simple pressure/probe/decomposition behavior.
7. The ANSYS workflow includes the targeted CPU-safe tests.
8. `py_compile`, focused solver tests, artifact/gate tests, and diff checks
   pass locally.
9. The commit is pushed to the current GitHub branch.

Expected final physical conclusion:

```text
reference_formulation_candidate = none
```

Expected implementation status:

```text
Phase 0 observability hardening complete
real pressure-probe matrix rerun still pending
dual-face one-sided traction still unsupported
coupled validation still blocked
```
