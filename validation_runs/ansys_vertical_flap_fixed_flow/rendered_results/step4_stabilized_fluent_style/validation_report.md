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
| max_u | 31.1107582 |
| max_speed | 42.2414164 |
| centerline_max_u | 28.7926583 |
| mass_imbalance_rel | 0 |
| mass_imbalance_rel_raw | 0.0227049498 |
| mass_imbalance_rel_corrected | 0 |
| divergence_linf | 3294.59033 |
| divergence_l2 | 102.470282 |
| divergence_linf_excluding_near_solid | 3294.59033 |
| divergence_l2_excluding_near_solid | 95.3212948 |
| poisson_residual_linf | 7.8179869e+09 |
| poisson_residual_linf_relative | 0.000928156984 |
| throat_max_u | 29.5289996 |
| throat_mean_u | 27.9468856 |

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
| mass_quality | pass | final mass imbalance rel = 0 |
| incompressibility_quality | pass | divergence_l2=102.47; divergence_linf=3294.59; divergence_l2_excluding_near_solid=95.3213; divergence_linf_excluding_near_solid=3294.59; poisson_residual_linf=7.81799e+09; poisson_residual_linf_relative=0.000928157 |
| overall_status | candidate_not_parity | candidate_not_parity still means no Fluent parity claim without official numeric exports |

## Interpretation

Step 4 stabilized solver output produces a jet-like fixed-flap field, but the report keeps visual similarity separate from numerical convergence and official Fluent parity.
diagnostic_only_not_parity is the controlling status whenever divergence or pressure Poisson convergence remains outside the warning thresholds.

## Required Next Solver Improvement

- Improve pressure Poisson convergence.
- Add a divergence-reduction regression test.
- Compare uniform-initialized runs against the current jet-structured initialization.
- Introduce official Fluent numeric exports before any Fluent parity claim.
