# ANSYS Vertical-Flap Traction Probe And One-Sided Validation Goal - 2026-06-26

## Source Review

This goal follows the review of commit:

```text
7a98186b3acc7ab105162ba8c608507c18569479
fix: validate ANSYS flap traction formulation
```

That commit proved the current fixed-solid traction matrix is useful and honest:

```text
supported_formulation_count = 5
unsupported_formulation_count = 1
reference_formulation_candidate = none
```

The archived matrix also exposed a blocking physical issue:

```text
dual two-sided offset 0.25 force ratio ~= 1.9507
dual two-sided offset 0.51 force ratio = 1.0000
dual two-sided offset 1.00 force ratio ~= 0.0675
single-mid two-sided force ratio ~= 0.9612
viscous-air force ratio ~= 0.9809
```

The current dual physical faces plus two-sided pressure jump formulation cannot
be promoted to a reference traction formulation. The candidate gate must encode
that conclusion explicitly so a future implementation of dual one-sided
sampling cannot accidentally turn the baseline into a false reference.

## High-Level Objective

Harden the ANSYS vertical-flap traction formulation validation path so it cannot
promote a traction formulation unless the matrix actually satisfies the
documented rule:

```text
all required formulation rows are supported, completed, conservative, and stable
```

This goal also documents the next required physics work:

1. expose pressure probe diagnostics,
2. separate physical marker position from pressure probe start offset,
3. implement per-face one-sided traction with an explicit fluid side,
4. compare dual one-sided and single-mid two-sided on identical flow snapshots.

## Immediate Commit Scope

The immediate implementation target is the first safe commit:

```text
fix: harden ANSYS traction formulation gates
```

This commit is intentionally gate/diagnostic-only. It must not change the
production pressure solve, flow update, marker feedback, MPM scatter, MPM
advance, or traction formula. It must not run coupled 50-step validation.

Required immediate changes:

1. Rename comparison semantics in the traction matrix from reference-oriented to
   baseline-oriented where appropriate.
2. Keep the output field `reference_formulation_candidate` for downstream
   compatibility, but ensure its value is selected only by the hardened gate.
3. Add matrix-level candidate blockers.
4. Add explicit offset-sensitivity fields.
5. Add formulation-agreement fields.
6. Add flow snapshot identity status based on already archived per-row flow
   metrics.
7. Make unsupported/failed rows block the candidate with explicit reasons.
8. Make current offset instability block the candidate even if unsupported rows
   disappear in the future.
9. Mark `single_mid_surface + one_sided_surface_pressure` unsupported because
   the fluid side is ambiguous without an explicit per-marker side.
10. Add stricter `traction_marker_face_offset_cells` validation.
11. Reclassify existing traction formulation artifacts without rerunning the
    GPU matrix unless the script requires it.
12. Update tests, docs, summary, verification, and workflow checks.
13. Commit and push after verification.

## Immediate Non-Goals

The immediate commit must not implement these items:

```text
core pressure probe fields
t_pressure_gamma_pa / t_viscous_gamma_pa split
per-marker pressure sampling mode fields
per-marker fluid side sign fields
dual-face physical one-sided formula
single-snapshot NPZ resampling
STEP60 rerun
STEP120 / STEP200
coupled 30-step or 50-step
Fluent force-history import
Fluent parity claim
coupling subiterations
solid stiffness/damping/support tuning
gate-threshold tuning after seeing new long-run results
```

## Candidate Gate Contract

The traction matrix must report:

```text
candidate_blockers
offset_sensitivity_status
offset_force_ratio_min
offset_force_ratio_max
offset_force_relative_span
formulation_agreement_status
dual_one_sided_vs_single_mid_relative_error
flow_snapshot_identity_status
```

`reference_formulation_candidate` must be:

```text
none
```

unless all of the following are true:

```text
all required scenarios are present
all required scenarios completed
no required scenario is unsupported
no required scenario failed
invalid marker count is zero for every completed row
force_decomposition_residual_N <= 1e-8 for every completed row
marker_action_reaction_residual_N <= 1e-8 for every completed row
scatter_action_reaction_residual_N <= 1e-8 for every completed row
no completed row uses a full-field reset
offset sensitivity passes
formulation agreement passes
flow snapshot identity is acceptable
pressure probe diagnostics are complete enough for promotion
```

For current artifacts, blockers must include at least:

```text
dual_face_one_sided_unsupported
dual_two_sided_offset_sensitivity_above_tolerance
pressure_probe_diagnostics_incomplete
```

## Offset Sensitivity Rule

For current matrix rows, compare completed pressure-only dual two-sided offset
rows against the baseline scenario:

```text
dual_two_sided_offset0p51_pressure_only
```

The baseline name should remain a comparison baseline, not a reference
formulation. The candidate gate must compute:

```text
offset_force_ratio_min
offset_force_ratio_max
offset_force_relative_span
```

The current data must fail the gate because:

```text
min ratio ~= 0.0675
max ratio ~= 1.9507
relative span is order-one
```

The implementation may use a fixed conservative tolerance for this immediate
gate:

```text
OFFSET_FORCE_RATIO_MIN = 0.90
OFFSET_FORCE_RATIO_MAX = 1.10
```

Do not relax this tolerance in this commit.

## Formulation Agreement Rule

For the current artifact set, dual one-sided is unsupported, so
`dual_one_sided_vs_single_mid_relative_error` should remain blank and the
agreement status should fail or be blocked for a clear reason.

If future artifacts contain completed rows for both:

```text
dual physical faces + one-sided surface pressure
single mid-surface + two-sided pressure jump
```

then the gate must compare their total force against each other. A future
promotion should require relative error less than or equal to 10%.

Do not treat `single_mid_surface + one_sided_surface_pressure` as supported
unless the caller explicitly specifies a physically meaningful fluid side.

## Flow Snapshot Identity Rule

The current matrix runs each formulation independently but records final flow
metrics. For the immediate commit, compute a report-only identity status from
the final archived metrics:

```text
velocity_peak / p999
velocity_outlet_flux_ratio
pressure_outlet_flux_ratio
pressure_min_pa
pressure_max_pa
```

The status should be:

```text
flow_metrics_match_completed_rows
flow_metrics_mismatch_completed_rows
flow_metrics_incomplete
```

This is not a substitute for future NPZ snapshot hashing; it is only a guard
against silently comparing different flow states.

## Support Matrix Contract

`traction_formulation_supported()` must distinguish these combinations:

```text
dual_physical_faces + two_sided_pressure_jump:
  supported

dual_physical_faces + one_sided_surface_pressure:
  unsupported_pending_per_face_fluid_side

single_mid_surface + two_sided_pressure_jump:
  supported

single_mid_surface + one_sided_surface_pressure:
  unsupported_ambiguous_fluid_side
```

Unsupported rows must still appear in the matrix and history CSV/JSON should
remain empty for those rows.

## Diagnostic Boundary

Non-default traction formulations are currently routed through the formal marker
builder used by the coupled runner. To prevent accidental production use,
validation must reject non-default traction formulations when:

```text
step_count > 0
```

unless a future explicit opt-in flag is added and tested. This immediate commit
should enforce the conservative rule:

```text
coupled FSI uses default dual physical faces + two-sided pressure jump + 0.51 cell offset + pressure-only
fixed-solid preflow diagnostics may vary traction formulation controls
```

If the current code path makes this too invasive, at minimum add source-level
tests and validation guards documenting the default-safe boundary.

## Tests Required For Immediate Commit

Add or update source tests for:

```text
unsupported row blocks candidate
failed row blocks candidate
offset ratio 1.95 blocks candidate
offset ratio 0.067 blocks candidate
all rows completed but offset unstable still blocks
stable synthetic A/B/C matrix can promote only when every blocker is absent
single-mid one-sided is unsupported
invalid marker counts block promotion
non-default coupled traction config is rejected
current artifacts expose candidate_blockers and offset fields
artifact summary/verification state the blockers
```

Keep tests focused and cheap. Do not require GPU simulation for gate logic.

## Artifact Reclassification

Run:

```powershell
& $python validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_formulation_validation_matrix.py --reclassify-existing
```

This should update:

```text
validation_runs/ansys_vertical_flap_fsi/traction_formulation_diagnostics/traction_formulation_matrix.json
validation_runs/ansys_vertical_flap_fsi/traction_formulation_diagnostics/traction_formulation_matrix.csv
validation_runs/ansys_vertical_flap_fsi/traction_formulation_diagnostics/traction_formulation_summary.md
validation_runs/ansys_vertical_flap_fsi/traction_formulation_diagnostics/verification_traction_formulation_2026-06-26.md
```

It should not rerun the expensive matrix unless the script cannot classify the
existing artifacts.

## Documentation Required

Update:

```text
docs/VALIDATION.md
```

Required documentation points:

```text
0.51 dual two-sided is a baseline, not a reference
current candidate is blocked by unsupported one-sided and offset instability
pressure probe diagnostics are incomplete
coupled 50-step remains out of scope
Fluent parity remains out of scope
future probe/one-sided work must happen before reference promotion
```

## Future Commit 2 - Pressure Probe Observability

After the immediate gate commit, add diagnostics that expose pressure sampling
details without changing formulas:

```text
stress_inside_pressure_pa
stress_outside_pressure_pa
stress_pressure_jump_pa
stress_fluid_side_pressure_pa
stress_reference_pressure_pa
stress_inside_pressure_found
stress_outside_pressure_found
stress_probe_mode
stress_inside_probe_rung
stress_outside_probe_rung
stress_inside_probe_cell
stress_outside_probe_cell
stress_invalid_reason
t_pressure_gamma_pa
t_viscous_gamma_pa
t_gamma_pa
```

The core invariant must be:

```text
t_pressure_gamma_pa + t_viscous_gamma_pa == t_gamma_pa
```

This future commit must explain the current offset pathology:

```text
why 0.25 cell samples both faces as full pressure jump
why 0.51 cell makes secondary near zero
why 1.00 cell loses both faces
```

## Future Commit 3 - Physical Dual-Face One-Sided Traction

After pressure probe observability exists, implement physically explicit
dual-face one-sided traction:

```text
pressure_sampling_mode
one_sided_fluid_side_normal_sign
one_sided_reference_pressure_pa
```

For dual physical faces:

```text
primary face normal = +z, fluid side = +normal
secondary face normal = -z, fluid side = +normal
```

The intended one-sided pressure traction is:

```text
traction = -p_fluid * normal
```

Required analytical tests for that future commit:

```text
uniform pressure cancels across two physical faces
pressure-jump test matches single-mid two-sided total
gauge shift does not change net force
primary/secondary region IDs remain diagnostic only
sampling mode does not depend on diagnostic region ID
```

## Future Matrix Redesign

After probe observability and physical one-sided traction are implemented,
replace independent formulation runs with:

```text
single fixed-solid flow solve
saved pressure/velocity/sampling snapshot
multiple traction resampling passes on the same snapshot
flow_snapshot_sha256 in every row
```

Suggested snapshots:

```text
step020_fields.npz
step040_fields.npz
step060_fields.npz
snapshot_manifest.json
```

Suggested probe start sweep:

```text
0.25, 0.375, 0.50, 0.625, 0.75, 1.00 cells
```

The future internal reference candidate may be selected only after dual
one-sided and single-mid two-sided agree across snapshots and probe starts.

## Done Criteria For This Immediate Commit

The immediate commit is complete when:

```text
goal file is committed
traction matrix script reports candidate_blockers
baseline wording replaces reference wording where appropriate
current artifacts reclassified with explicit blockers
source tests cover blocker logic
artifact tests cover new fields and blockers
validation docs updated
py_compile passes
focused unittest slice passes
git diff --check passes
commit is pushed to the current GitHub branch
```

Expected final commit message:

```text
fix: harden ANSYS traction formulation gates
```
