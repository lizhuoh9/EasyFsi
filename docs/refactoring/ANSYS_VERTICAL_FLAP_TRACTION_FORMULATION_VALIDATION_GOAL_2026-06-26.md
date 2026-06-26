# ANSYS Vertical-Flap Traction Formulation Validation Goal - 2026-06-26

## Background

The previous pushed commit is:

```text
e3ac937b4b0ebb7a75df900f7662d4c02f801b0e
fix: diagnose ANSYS flap hydrodynamic load
```

That commit made a useful step forward: the ANSYS vertical-flap validation now
archives fixed-solid STEP60 load histories, separates the two physical marker
faces into primary and secondary diagnostic regions, records marker/scatter
action-reaction residuals, and identifies an internal fixed-solid load
candidate:

```text
fixed_load_0p80_ramp2_step60
flow_temporal_strict
load_temporal_strict
last-20 force mean ~= -4.799e-05 N
last-20 negative-force fraction = 0.8
scatter residual max ~= 3.98e-12 N
```

This proves the fixed-solid source/outlet path can produce a bounded,
mostly-negative hydrodynamic marker force and that the already-computed marker
force is scattered to the MPM particles conservatively. It does not yet prove
that the traction formulation itself is correct.

The next blocking question is:

```text
Does the current dual-face marker + two-sided pressure traction formulation
avoid double counting, remain stable under small marker offset changes, and
produce a face-resolved total force that is robust enough to become the
reference fixed-flap load path?
```

## Scope For This Commit

This commit implements the first recommended follow-up:

```text
fix: validate ANSYS flap traction formulation
```

The scope is intentionally narrower than STEP120/STEP200 runs, Fluent force
history import, load-to-MPM replay, moving-interface diagnostics, or coupling
subiterations. This commit must add a reviewable, artifact-backed traction
formulation A/B/C diagnostic on top of the current fixed-solid STEP60 evidence.

Required in scope:

1. Add this detailed goal file and reference it from the active goal.
2. Add a non-synthetic ANSYS vertical-flap traction formulation diagnostic
   matrix.
3. Reuse one fixed-solid flow candidate path and resample marker traction under
   explicit formulation variants.
4. Compare at least the following formulations:

```text
A. dual physical faces + two-sided pressure jump
B. dual physical faces + one-sided surface pressure
C. single mid-surface + two-sided pressure jump
```

5. Parameterize the diagnostic surface so it can also run:

```text
marker_face_offset_cells: 0.25, 0.51, 1.00
include_viscous_traction: false, true
```

6. Archive per-formulation force, pressure, traction, marker-count, and
   residual statistics.
7. Add tests proving that diagnostic region splitting does not change the total
   marker force for the same loaded marker force field.
8. Add tests proving the new traction artifacts are reviewable and cannot claim
   Fluent parity.
9. Update validation documentation and CI compile/artifact checks.
10. Commit and push after verification passes.

Out of scope for this commit:

```text
coupled 50-step runs
STEP120 / STEP200 long windows
Fluent force-history import
Fluent parity claims
solid stiffness/damping/support tuning
gate relaxation
load-to-MPM replay
moving IBM interface diagnostics
dynamic mesh implementation
coupling subiterations
full-field reinitialize promotion
```

## Key Physical Constraints

The current STEP60 load gate is an internal diagnostic gate only. The new
traction matrix must not reinterpret it as Fluent validation.

These statements must remain true in code, docs, and artifacts:

```text
scatter residual ~= 0 proves conservative scatter of the computed marker force
scatter residual ~= 0 does not prove the marker traction is correct
negative-force fraction = 0.8 is an internal load candidate, not Fluent parity
diagnostic full-field/inlet reinitialize rows are never release candidates
pressure-only traction must be called pressure-only traction
viscous traction must be explicit when enabled
```

## Phase 1 - Runner Support For Traction Formulation Diagnostics

Modify, only as needed:

```text
benchmarks/official/solid_mpm_fsi_runner.py
```

The runner should expose a small diagnostic surface for building alternative
marker layouts and stress sampling modes without changing the production
coupled FSI path.

At minimum, support explicit diagnostic controls:

```text
marker_layout:
  dual_physical_faces
  single_mid_surface

pressure_sampling_mode:
  two_sided_pressure_jump
  one_sided_surface_pressure

include_viscous_traction:
  false
  true

marker_face_offset_cells:
  0.25
  0.51
  1.00
```

Implementation constraints:

- Do not silently change the default formal runner behavior.
- Do not change the pressure solve, flow driver, marker feedback, force
  scatter, MPM stepping, or coupled FSI loop.
- Prefer diagnostic helper functions and explicit config fields over hidden
  global switches.
- Preserve current marker positions, normals, areas, and region IDs for the
  default dual-face layout.
- For single-mid-surface diagnostics, construct a distinct diagnostic marker
  set and mark the output as non-production/reference-candidate-only until the
  artifacts justify promotion.
- If the core stress sampler cannot currently support a requested mode without
  solver-formula changes, implement a report-only artifact with
  `status = unsupported` for that mode rather than faking data.

## Phase 2 - Total-Force Invariance Test For Region Splitting

Add a strict regression test proving that splitting marker region IDs is
diagnostic-only for total force aggregation.

The test should use the same marker positions, normals, areas, and already
loaded marker force field twice:

```text
case 1: all markers in PRIMARY_REGION_ID
case 2: +z face PRIMARY_REGION_ID, -z face SECONDARY_REGION_ID
```

Required assertions:

```text
total marker count unchanged
total marker force unchanged within tight numerical tolerance
primary + secondary force == total force
primary/secondary marker counts match the two physical faces
```

This test should not require running a long fluid simulation.

## Phase 3 - Traction Matrix Script

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_formulation_validation_matrix.py
```

Write artifacts under:

```text
validation_runs/ansys_vertical_flap_fsi/traction_formulation_diagnostics/
```

Required scenarios:

```text
dual_two_sided_offset0p51_pressure_only
dual_one_sided_offset0p51_pressure_only
single_mid_two_sided_offset0p00_pressure_only
dual_two_sided_offset0p25_pressure_only
dual_two_sided_offset1p00_pressure_only
dual_two_sided_offset0p51_viscous_air
```

The script may use a short real fixed-solid preflow window when necessary to
obtain a valid flow/stress state, but it must keep the run diagnostic and cheap
enough for local review. Prefer reusing the already selected fixed-solid source
path:

```text
flow_driver_mode = sustained_volume_source_inlet
source_strength = 0.80
source_profile = linear_ramp
source_ramp_steps = 2
step_count = 0
apply_marker_feedback_to_fluid = false
```

The script must clearly report whether each row is:

```text
completed
unsupported
failed
```

Unsupported rows are acceptable only when the current core does not expose the
needed mode without physical-formula changes. Unsupported rows must still be
represented in the matrix and documentation.

## Phase 4 - Required Artifact Fields

Each matrix row must include:

```text
scenario
run_status
marker_layout
pressure_sampling_mode
include_viscous_traction
viscosity_pa_s
marker_face_offset_cells
preflow_steps
flow_driver_mode
source_strength
source_profile
source_ramp_steps

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
marker_action_reaction_residual_N
scatter_action_reaction_residual_N

primary_face_mean_pressure_pa
secondary_face_mean_pressure_pa
primary_face_mean_traction_z_pa
secondary_face_mean_traction_z_pa
max_abs_traction_pa
two_sided_pressure_marker_count
one_sided_pressure_marker_count

force_difference_from_reference_N
force_ratio_to_reference
face_force_ratio
status_reason
scope_limit
```

If pressure means cannot be computed from existing core outputs without
changing solver formulas, report them as unsupported/blank and explain that the
current implementation exposes force and traction counters but not per-face
pressure means.

## Phase 5 - Candidate And Gate Semantics

The matrix should identify:

```text
reference_formulation_candidate
candidate_status
supported_formulation_count
unsupported_formulation_count
```

Do not promote a formulation if:

```text
invalid markers are nonzero
primary + secondary != total
marker/scatter action-reaction residual exceeds tolerance
small offset changes cause order-one force changes
the row uses an unsupported/report-only mode
the row requires full-field reset
```

The current implementation may legitimately end with:

```text
reference_formulation_candidate = none
candidate_status = no_reference_formulation_candidate
```

if the A/B/C evidence is not yet strong enough.

## Phase 6 - Tests

Add or update tests:

```text
tests/cases/test_ansys_vertical_flap_fsi.py
tests/integration/test_ansys_vertical_flap_traction_formulation_artifacts.py
tests/tools/test_ansys_vertical_flap_temporal_gate.py  # only if shared gate helpers change
```

Required assertions:

```text
region split does not change total marker force for an identical force field
primary + secondary force equals total force in the new artifact rows
unsupported modes are explicit, not silently passing
pressure-only rows are labeled pressure-only
viscous-air rows are labeled with nonzero viscosity
no artifact claims Fluent parity
no artifact claims coupled 50-step success
diagnostic full-field reset is not promoted
```

## Phase 7 - Documentation And Workflow

Update:

```text
docs/VALIDATION.md
.github/workflows/ansys-vertical-flap-validation.yml
```

`docs/VALIDATION.md` must explain:

```text
STEP60 fixed-solid load matrix found an internal load candidate
traction formulation validation is the next diagnostic layer
scatter residual validates transfer of computed force, not traction correctness
pressure-only vs viscous traction is explicit
current scope does not claim Fluent parity
```

The workflow should compile the new matrix script and run artifact consistency
tests. It must not run long simulations by default.

## Phase 8 - Verification Commands

Run at minimum:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  benchmarks\official\solid_mpm_fsi_runner.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_formulation_validation_matrix.py `
  tests\cases\test_ansys_vertical_flap_fsi.py `
  tests\integration\test_ansys_vertical_flap_traction_formulation_artifacts.py

& 'D:\working\taichi\env\python.exe' -m unittest `
  tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_formal_runner_places_both_streamwise_marker_faces `
  tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_marker_force_report_fields_are_face_resolved `
  tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_region_split_preserves_total_marker_force `
  -v

& 'D:\working\taichi\env\python.exe' `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_formulation_validation_matrix.py

& 'D:\working\taichi\env\python.exe' -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_formulation_artifacts `
  -v

git diff --check
```

If the real matrix runtime is excessive, document the runtime and reduce only
the diagnostic preflow window or scenario count. Do not replace real solver
artifacts with synthetic data.

## Completion Criteria

The goal is complete only when:

```text
the traction formulation diagnostic matrix exists
required A/B/C rows are present or explicitly unsupported
the region split total-force invariance test passes
the artifacts contain face-resolved force decomposition and residuals
pressure-only and viscous-air semantics are explicit
docs and workflow are updated
verification commands pass
all relevant code, docs, logs, and generated artifacts are committed and pushed
```

Final reporting must include:

```text
new commit SHA
remote branch
verification commands and pass/fail status
traction formulation conclusion
unsupported modes or remaining blocked scope
whether push succeeded
```
