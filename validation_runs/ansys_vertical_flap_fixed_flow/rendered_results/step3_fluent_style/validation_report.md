# ANSYS Vertical Flap Fixed-Flow Step 3 Validation Report

## Scope

- Source: Step 2 solver output.
- No Fluent parity claim.
- No FSI claim.
- traction_shared_snapshot_diagnostics not used.
- This is a Fluent-style visualization of the Step 2 solver output, not a Fluent parity validation.

## Field Summary

| metric | value |
|---|---:|
| max_u | 37.5631832 |
| max_speed | 37.5631833 |
| centerline_max_u | 37.5631832 |
| mass_imbalance_rel | -0.012579369 |
| divergence_linf | 8316.92787 |
| divergence_l2 | 71.2591858 |
| poisson_residual_linf | 4.31529579e+09 |
| throat_max_u | 34.8702037 |
| throat_mean_u | 28.6564864 |

## Visual Outputs

- speed_full_fluent_scale_0_28p1: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/speed_full_fluent_scale_0_28p1.png`
- speed_full_autoscale: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/speed_full_autoscale.png`
- streamwise_minus_Uz_fluent_scale_0_28p1: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/streamwise_minus_Uz_fluent_scale_0_28p1.png`
- streamwise_minus_Uz_autoscale: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/streamwise_minus_Uz_autoscale.png`
- Uy_full: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/Uy_full.png`
- pressure_full: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/pressure_full.png`
- geometry_overlay: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/geometry_overlay.png`
- solver_history_plot: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/solver_history_plot.png`
- mass_balance_plot: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/mass_balance_plot.png`

## Profile Outputs

- centerline_streamwise_minus_Uz: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/centerline_streamwise_minus_Uz.csv`
- throat_profile_streamwise_minus_Uz: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/throat_profile_streamwise_minus_Uz.csv`
- downstream_profiles_streamwise_minus_Uz: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/downstream_profiles_streamwise_minus_Uz.csv`

## Quality Gates

| gate | status | reason |
|---|---|---|
| visual_candidate | pass | centerline jet exists |
| mass_quality | pass | final mass imbalance rel = 0.0125794 |
| incompressibility_quality | warn | divergence_l2=71.2592; divergence_linf=8316.93; poisson_residual_linf=4.3153e+09 |
| overall_status | diagnostic_only_not_parity | diagnostic_only_not_parity until parity data and solver convergence justify stronger claims |

## Interpretation

Current Step 2 produces a jet-like fixed-flap field, but the report keeps visual similarity separate from numerical convergence and official Fluent parity.
diagnostic_only_not_parity is the controlling status whenever divergence or pressure Poisson convergence remains outside the warning thresholds.

## Required Next Solver Improvement

- Improve pressure Poisson convergence.
- Add a divergence-reduction regression test.
- Compare uniform-initialized runs against the current jet-structured initialization.
- Introduce official Fluent numeric exports before any Fluent parity claim.
