# ANSYS Vertical-Flap Flow-Collapse Diagnostic Goal - 2026-06-25

## Source Review

This goal implements the next step after commit
`6a2f55968fbccb829aee8bda19a237480edb0948`
(`fix: add ansys vertical flap postrepair validation`). That commit established
the reviewable ANSYS vertical-flap baseline, repaired coarse 50-step artifact
capture, fixed-solid preflow smoke reporting, local verification records, and a
CI contract workflow.

The current physical result is intentionally not a Fluent-parity claim. The
coarse 50-step runner completes but remains `FAIL_FLOW`: final velocity peak is
about `10.68 m/s` and final p999 velocity is about `10.19 m/s`, below the
official web contour range of `20-29 m/s`. Early coarse steps and one-step
preflow can still produce `20-29 m/s` local acceleration, so the next task is to
diagnose why the transient projection-only loop collapses back to inlet-scale
velocity.

## Primary Objective

Create a reviewable flow-collapse diagnostic baseline for the ANSYS
vertical-flap coarse runner. The output must answer, with committed artifacts
and tests, whether the velocity decay from `28-31 m/s` to `10-11 m/s` is caused
by the projection-only flow loop itself or by marker-feedback / solid-coupled
constraints.

## Non-Goals

- Do not tune solid material, solid damping, support radius, or substep counts
  to mask a flow failure.
- Do not run L3 50-step validation.
- Do not claim Fluent parity.
- Do not replace the official sharp HIBM-MPM validation path in this commit.
- Do not hardcode pressure, velocity, displacement, or force fields to fake a
  jet.

## Required Reporting Fixes

1. Diagnostics summary must stop reporting `markers=config.marker_count` as if
   it were the actual marker count. It must report:
   - `markers_per_face`
   - `markers_actual`
   - `markers`, kept as an actual-count alias only if compatibility requires it
2. Diagnostics summary must stop treating `preflow_steps=0` as if preflow
   "converged". It must report a status field:
   - `not_requested`
   - `converged`
   - `max_steps`
3. Post-repair artifact tests must assert that summary actual marker count
   equals the report/stage-check semantics (`24` for two faces of 12 markers).

## Required Diagnostic Controls

Add diagnostic-only controls to the formal ANSYS rectangular-solid runner:

1. `apply_marker_feedback_to_fluid`
   - default `True`
   - when `False`, solid force scatter and MPM advancement still run, but marker
     velocity feedback is not imposed as a fluid Dirichlet constraint before
     projection
2. `flow_reset_pressure_each_step`
   - default `False`
   - when `True`, each projection resets pressure before solve
3. `flow_reinitialize_inlet_each_step`
   - default `False`
   - when `True`, the diagnostic run re-applies the initialized inlet field
     before every projection
   - this is a diagnostic control only, not the final physical model
4. `step_count=0` with `preflow_steps>0` must be legal for preflow-only
   diagnostics and must return a report with `history=[]` instead of raising
   "did not advance"

These controls must be exposed on `VerticalFlapFsiConfig` and flow through the
formal runner, but the default smoke behavior must remain unchanged.

## Required Scripts

Add committed scripts under:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/
```

### run_preflow_only_sweep_after_halfdomain_repair.py

Runs fixed-solid preflow-only diagnostics for:

```text
preflow_steps = 1, 2, 5, 10, 20
step_count = 0
solid_advanced = false
feedback_applied = false
```

Writes:

```text
validation_runs/ansys_vertical_flap_fsi/flow_collapse_diagnostics/preflow_only_sweep/preflow_only_sweep.json
validation_runs/ansys_vertical_flap_fsi/flow_collapse_diagnostics/preflow_only_sweep/preflow_only_sweep.csv
```

Each row must include at least:

```text
scenario
preflow_steps
preflow_steps_completed
preflow_status
local_velocity_peak_mps
fluid_speed_p99_mps
fluid_speed_p999_mps
pressure_min_pa
pressure_max_pa
projection_l2
projection_max_abs
velocity_dirichlet_boundary_max_delta_mps
stress_invalid_marker_count
marker_force_z_N
solid_advanced
feedback_applied
```

### run_flow_collapse_diagnostic_matrix.py

Runs the minimal coarse diagnostic matrix:

```text
feedback_on_step10
feedback_off_step10
solver_fv_jacobi_1080_step10
solver_fv_cg_1080_step10
solver_fv_cg_4096_step10
reset_pressure_first_only_step10
reset_pressure_every_step_step10
reinitialize_inlet_each_step_step10
```

The matrix intentionally uses 10-step runs, not 50-step or L3 runs. It must
write:

```text
validation_runs/ansys_vertical_flap_fsi/flow_collapse_diagnostics/diagnostic_matrix/flow_collapse_diagnostic_matrix.json
validation_runs/ansys_vertical_flap_fsi/flow_collapse_diagnostics/diagnostic_matrix/flow_collapse_diagnostic_matrix.csv
validation_runs/ansys_vertical_flap_fsi/flow_collapse_diagnostics/diagnostic_matrix/flow_collapse_diagnostic_matrix_summary.md
```

Each scenario row must include:

```text
scenario
step_count
apply_marker_feedback_to_fluid
flow_pressure_solver
flow_projection_iterations
flow_reset_pressure_each_step
flow_reinitialize_inlet_each_step
final_velocity_peak_mps
final_velocity_p999_mps
max_velocity_peak_mps
max_velocity_p999_mps
collapse_ratio_peak
collapse_ratio_p999
flow_status
projection_l2
projection_max_abs
stress_invalid_marker_count
scatter_invalid_marker_count
feedback_invalid_marker_count
marker_force_z_N
tip_dz_final_m
elapsed_s
```

## Required Interpretation

The diagnostic summary must explicitly state:

```text
primary_observation
current_best_hypothesis
next_action
```

The expected answer from the currently available evidence is likely:

```text
projection-only formal runner is the primary suspect if preflow-only and
feedback-off runs also collapse; feedback constraints are the primary suspect
only if feedback-off remains in the 20-29 m/s p999 band while feedback-on
collapses.
```

The summary must report the result actually observed in committed artifacts,
not force this hypothesis.

## Required Tests

Add or update focused tests:

1. Diagnostics summary reports actual marker count and markers per face.
2. Preflow status is `not_requested` when preflow was not requested.
3. The formal runner accepts preflow-only `step_count=0` with preflow history.
4. Diagnostic controls are exposed on `VerticalFlapFsiConfig` without changing
   defaults.
5. Flow-collapse artifact tests assert that committed sweep/matrix artifacts
   exist, contain the required scenarios, and include enough fields to answer
   the projection-only vs feedback-collapse question.

## Required Verification

Run at minimum:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile cases\ansys_vertical_flap_fsi.py benchmarks\official\solid_mpm_fsi_runner.py tools\validation\print_ansys_vertical_flap_diagnostics.py validation_runs\ansys_vertical_flap_fsi\scripts\run_preflow_only_sweep_after_halfdomain_repair.py validation_runs\ansys_vertical_flap_fsi\scripts\run_flow_collapse_diagnostic_matrix.py
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_preflow_only_sweep_after_halfdomain_repair.py
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_flow_collapse_diagnostic_matrix.py
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.cases.test_ansys_vertical_flap_fsi tests.tools.test_ansys_vertical_flap_diagnostics tests.integration.test_ansys_vertical_flap_postrepair_artifacts tests.integration.test_ansys_vertical_flap_flow_collapse_artifacts
git diff --check
```

If a runtime command is too slow or fails, commit the failure JSON/log only if
it is materially useful and clearly label the run as incomplete. Do not convert
runtime failure into a false green test.

## Done Criteria

- A short Codex goal references this markdown file.
- Reporting ambiguity for marker count and preflow status is fixed.
- Diagnostic controls exist and default behavior remains unchanged.
- Preflow-only and flow-collapse matrix scripts exist.
- Diagnostic artifacts are committed under
  `validation_runs/ansys_vertical_flap_fsi/flow_collapse_diagnostics/`.
- Tests cover the reporting fixes and artifact schema.
- Verification commands and observed results are recorded.
- Work is committed and pushed to the tracked GitHub branch.
