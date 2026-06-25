# ANSYS Vertical Flap Closed-Loop Runner Result - 2026-06-25

## Scope

This note records the first solver-branch result after moving fluid projection into the ANSYS vertical-flap FSI loop.

The change targets runner closure only. It does not change ANSYS case parameters, reference values, tolerances, material constants, support radius defaults, or `simulation_core/`.

## Baseline Before This Branch

From the official-web baseline artifacts:

- status = FAIL_SOLID_HISTORY
- feedback_closure_status = OPEN_LOOP_LOAD_REUSE
- fluid_recomputed_after_feedback = false
- tip_dz_monotonic_violation_count = 23
- first_tip_dz_violation_step = 5
- max_tip_dz_rebound_m = 5.472451448440552e-06
- final displacement rel_error = 0.4699417795678494
- velocity_peak_mps = 28.15654945373535

## Result After Closed-Loop Runner Change

Generated from:

```text
validation_runs/ansys_vertical_flap_fsi/compare/easyfsi_summary.csv
validation_runs/ansys_vertical_flap_fsi/compare/stage_check.md
validation_runs/ansys_vertical_flap_fsi/compare/displacement_compare.csv
```

Current artifact-backed result:

- status = FAIL_FLOW
- feedback_closure_status = CLOSED_LOOP_RECOMPUTED_FLOW
- fluid_recomputed_after_feedback = true
- fluid_recompute_count = 50
- velocity_peak_mps = 10.41722297668457
- official velocity range = 20 to 29 m/s
- max_disp_m = 1.0789924090204295e-05
- ref_max_disp_m = 5.1e-05
- disp_relerr = 0.7884328609763864
- final displacement rel_error vs official web scale = 0.8056619301102129
- tip_dz_monotonic_violation_count = 23
- first_tip_dz_violation_step = 4
- max_tip_dz_rebound_m = 1.1056661605834961e-05
- tip_dz_sign_violation_count = 10
- scatter_invalid = 0
- feedback_invalid = 0
- root_max_disp_m = 0.0

## Interpretation

The runner now reports real per-step fluid recomputation:

```text
fluid_recomputed_after_feedback = true
feedback_closure_status = CLOSED_LOOP_RECOMPUTED_FLOW
fluid_recompute_count = 50
```

The source-level runner-loop contract tests now pass without `expectedFailure`.

This is not yet a physical validation pass. The regenerated artifacts show a new first failing gate:

```text
status = FAIL_FLOW
velocity_peak_mps = 10.41722297668457
```

The value is below the official web contour range of `20 to 29 m/s`. The solid history also remains non-monotone and now has positive streamwise sign violations.

## Remaining Work

Keep these artifact-level tests as expected failures until the artifacts satisfy them:

- no solid history rebound
- final displacement rel_error <= 0.20

The next solver step should inspect whether the per-step projection consumes marker feedback in a physically meaningful way. In particular, compare per-step:

- `local_velocity_peak_mps`
- `pressure_min_pa`
- `pressure_max_pa`
- `total_marker_force_z_N`
- `tip_mean_dz_m`
- marker feedback displacement

Closed-loop structure exists now, but the flow field and solid response still need physical repair.
