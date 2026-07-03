# ANSYS Vertical Flap Fixed-Flow Step 3 Validation Report

## Scope

- Source: Step 4 stabilized solver output.
- No Fluent parity claim.
- No FSI claim.
- traction_shared_snapshot_diagnostics not used.
- This is a Fluent-style visualization of Step 4 stabilized solver output, not a Fluent parity validation.

## Field Summary

| metric | value |
|---|---:|
| max_u | 25.3180284 |
| max_speed | 26.8868206 |
| centerline_max_u | 18.5629764 |
| mass_imbalance_rel | 5.59517159e-16 |
| mass_imbalance_rel_raw | -0.143895068 |
| mass_imbalance_rel_corrected | 5.59517159e-16 |
| divergence_linf | 2584.41235 |
| divergence_l2 | 195.312418 |
| divergence_linf_excluding_near_solid | 2584.41235 |
| divergence_l2_excluding_near_solid | 195.469337 |
| poisson_residual_linf | 1.36449525e+09 |
| poisson_residual_linf_relative | 0.000994944433 |
| throat_max_u | 16.1719177 |
| throat_mean_u | 9.26410701 |

## Visual Outputs

- speed_full_fluent_scale_0_28p1: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/speed_full_fluent_scale_0_28p1.png`
- speed_full_autoscale: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/speed_full_autoscale.png`
- streamwise_minus_Uz_fluent_scale_0_28p1: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/streamwise_minus_Uz_fluent_scale_0_28p1.png`
- streamwise_minus_Uz_autoscale: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/streamwise_minus_Uz_autoscale.png`
- Uy_full: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/Uy_full.png`
- pressure_full: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/pressure_full.png`
- geometry_overlay: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/geometry_overlay.png`
- solver_history_plot: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/solver_history_plot.png`
- mass_balance_plot: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/mass_balance_plot.png`

## Profile Outputs

- centerline_streamwise_minus_Uz: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/centerline_streamwise_minus_Uz.csv`
- throat_profile_streamwise_minus_Uz: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/throat_profile_streamwise_minus_Uz.csv`
- downstream_profiles_streamwise_minus_Uz: `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/downstream_profiles_streamwise_minus_Uz.csv`

## Quality Gates

| gate | status | reason |
|---|---|---|
| visual_candidate | pass | centerline jet exists |
| mass_quality | pass | final mass imbalance rel = 5.59517e-16 |
| incompressibility_quality | warn | divergence_l2=195.312; divergence_linf=2584.41; divergence_l2_excluding_near_solid=195.469; divergence_linf_excluding_near_solid=2584.41; poisson_residual_linf=1.3645e+09; poisson_residual_linf_relative=0.000994944 |
| overall_status | diagnostic_only_not_parity | candidate_not_parity still means no Fluent parity claim without official numeric exports |

## Interpretation

Step 4 stabilized solver output produces a jet-like fixed-flap field, but the report keeps visual similarity separate from numerical convergence and official Fluent parity.
diagnostic_only_not_parity is the controlling status whenever divergence or pressure Poisson convergence remains outside the warning thresholds.

## Required Next Solver Improvement

- Improve pressure Poisson convergence.
- Add a divergence-reduction regression test.
- Compare uniform-initialized runs against the current jet-structured initialization.
- Introduce official Fluent numeric exports before any Fluent parity claim.
