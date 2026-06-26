# ANSYS Vertical-Flap Traction Pressure-Probe Observability Implementation Goal - 2026-06-26

## Source Review

This goal follows the review of commit:

```text
3e19c47b23d6e068a6eb42205f5042508c6fc06c
fix: fail closed ANSYS traction diagnostics
```

That commit closed the previous gate bypasses:

1. direct runner calls now reject unsupported traction formulations before
   solver execution;
2. missing completed-row diagnostics now add blockers instead of being treated
   as zero;
3. source and artifact tests cover unsupported direct runner paths, missing
   diagnostics, full-field reset, formulation disagreement, flow snapshot
   mismatch, offset instability, and stable synthetic promotion.

The current physical conclusion remains unchanged:

```text
reference_formulation_candidate = none
```

The remaining physical blockers are:

```text
pressure_probe_diagnostics_incomplete
required_formulation_unsupported
dual_face_one_sided_unsupported
dual_two_sided_offset_sensitivity_above_tolerance
```

## Objective

Move from gate-only protection to pressure-probe observability. The core must
report what pressure each marker sampled, from which side, at which probe rung,
at which distance, and in which grid cell. This evidence is required before any
dual-face one-sided traction formula can be implemented safely.

## Immediate Commit Scope

The immediate implementation target is:

```text
fix: expose ANSYS traction pressure probes
```

This commit should be the smallest useful observability increment. It must
prefer extending the existing `HibmMpmSurfaceMarkers.stress_marker_diagnostics()`
path over creating a parallel diagnostic system.

Required immediate changes:

1. Extend existing marker stress diagnostics with explicit pressure-probe
   fields.
2. Preserve sentinel semantics for missing probes:

   ```text
   pressure value       = 0.0 and valid only when found flag is true
   probe rung           = -1
   probe distance       = -1
   probe cell           = (-1, -1, -1)
   probe mode           = explicit integer code
   invalid reason       = existing reason code where possible
   ```

3. Expose enough per-marker fields to answer:

   ```text
   base pressure
   inside pressure
   outside pressure
   pressure jump
   inside pressure found
   outside pressure found
   inside probe rung
   outside probe rung
   inside probe distance
   outside probe distance
   inside probe cell
   outside probe cell
   probe mode
   invalid reason
   ```

4. Add a public face-level diagnostic API or report helper so the formal runner
   does not need to read private Taichi fields for pressure/probe evidence.
5. Add runner/matrix fields that archive face-level pressure-probe evidence.
6. Make pressure completeness layout-aware:

   ```text
   dual physical faces:
       require primary and secondary pressure/probe evidence

   single mid-surface:
       require primary inside/outside pressure-jump evidence
       do not require nonexistent secondary pressure fields
   ```

7. Add fail-closed blockers for the remaining small gate issues:

   ```text
   full_field_reset_status_missing
   marker_count_inconsistent
   valid_invalid_marker_count_inconsistent
   marker_count_nonintegral
   primary_pressure_probe_diagnostics_incomplete
   secondary_pressure_probe_diagnostics_incomplete
   ```

8. Keep the current reference result blocked. This commit must not select a
   reference formulation.

## Immediate Non-Goals

The immediate commit must not:

```text
change the traction formula
implement dual-face one-sided traction
change marker geometry defaults
split marker position from probe start offset
run coupled STEP30/STEP50/50-step validation
claim Fluent parity
implement coupling subiterations
retune solid stiffness/damping/support radius
relax offset or force gates
```

If a full GPU rerun is too expensive for this commit, the commit may stop at
core/report/API/test coverage and document that archived matrix artifacts still
come from the prior run. It must not fabricate new probe evidence from old
artifacts.

## Existing Code To Use

Use the current HIBM-MPM diagnostic path:

```text
simulation_core/coupling/hibm_mpm/core.py
simulation_core/coupling/hibm_mpm/reports.py
HibmMpmSurfaceMarkers
HibmMpmFluidStressSampleReport
stress_marker_diagnostics()
benchmarks/official/solid_mpm_fsi_runner.py
```

The existing code already has partial pressure evidence such as found flags,
invalid reasons, marker positions, normals, and region IDs. Extend that path
instead of adding an unrelated sidecar format.

## Pressure-Probe Field Contract

Per marker diagnostics should include at least:

```text
base_pressure_pa
inside_pressure_pa
outside_pressure_pa
pressure_jump_pa
inside_pressure_found
outside_pressure_found
inside_probe_rung
outside_probe_rung
inside_probe_distance_m
outside_probe_distance_m
inside_probe_cell
outside_probe_cell
probe_mode
invalid_reason_code
invalid_reason
```

If the existing naming differs, keep existing names for compatibility and add
new explicit aliases rather than removing older keys.

## Face-Level Report Contract

The runner needs face-level aggregates:

```text
primary/secondary marker count
primary/secondary valid marker count
primary/secondary invalid marker count
primary/secondary pressure complete count
primary/secondary pressure missing count
primary/secondary mean base pressure
primary/secondary mean inside pressure
primary/secondary mean outside pressure
primary/secondary mean pressure jump
primary/secondary inside probe rung histogram
primary/secondary outside probe rung histogram
primary/secondary inside unique probe cell count
primary/secondary outside unique probe cell count
```

The matrix should archive enough of this information to explain the current
offset pathology in later reruns.

## Gate Contract

The candidate gate must remain fail-closed:

1. missing `flow_driver_uses_full_velocity_reset` is
   `full_field_reset_status_missing`;
2. true `flow_driver_uses_full_velocity_reset` is `full_field_reset_used`;
3. marker counts must be finite integers;
4. for dual faces, primary + secondary marker counts must equal total marker
   count;
5. valid + invalid must equal each face marker count;
6. pressure completeness must depend on marker layout;
7. missing pressure evidence must block reference promotion.

## Required Tests

Add or extend tests for:

```text
stress_marker_diagnostics includes pressure probe keys
sentinel probe values are explicit for missing probes
runner exposes face-level pressure probe aggregate keys
missing full-field reset status blocks candidate
non-integral marker count blocks candidate
primary + secondary marker count mismatch blocks candidate
valid + invalid marker count mismatch blocks candidate
single-mid completed row does not require secondary pressure fields
dual-face completed row requires both primary and secondary pressure fields
```

If practical, add a small analytical Taichi/core test for a simple pressure
field that verifies inside/outside pressure and pressure jump are reported.

## Artifact Policy

Do not fabricate pressure-probe values into old artifacts. Existing artifacts
may be reclassified only for gate-field changes. A real observability matrix
rerun is required before claiming archived per-marker probe evidence.

If this commit does not rerun the matrix, the verification text must say:

```text
core/report pressure probe observability added
existing traction matrix artifacts remain from the prior run
reference_formulation_candidate remains none
no coupled 50-step run was performed
no Fluent parity claim is made
```

## Verification

At minimum run:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  simulation_core\coupling\hibm_mpm\core.py `
  simulation_core\coupling\hibm_mpm\reports.py `
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

Also run any new focused core/report tests added for pressure-probe diagnostics.

## Done Criteria

This immediate commit is done only when:

1. the detailed goal file is committed;
2. the short active goal references this file;
3. pressure-probe evidence is exposed through existing core/report paths;
4. runner and matrix fields can consume that evidence;
5. gate completeness is layout-aware and fail-closed;
6. tests cover new probe keys and gate edge cases;
7. existing candidate remains `none`;
8. no coupled 50-step or Fluent parity claim is introduced;
9. local verification passes;
10. the commit is pushed to the configured GitHub remote.
