[SETUP]
PASS geometry: duct_length=0.1, duct_height=0.04, flap_height=0.01, flap_thickness=0.003
PASS material: rho_s=1600, E=1000000, nu=0.47
PASS time: dt=0.0005, steps=1

[PREFLOW]
steps_requested = 1
steps_completed = 1
converged = false
status = max_steps
stop_reason = max_steps
history_rows = 1

[FLOW_ONLY]
velocity_peak_mps = 31.5817070007
velocity_p99_mps = 16.3495376015
velocity_p999_mps = 25.0433615437
official_range_mps = [20, 29]
pressure_min_pa = -400.465295526
pressure_max_pa = 14.0339470091
projection_l2 = 1103.35841006
projection_max_abs = 6400
pre_projection_l2 = 1064.06488112
post_boundary_l2 = 1103.35841006
velocity_dirichlet_boundary_max_delta_mps = 9.85337352753
diagnosis = check fluid solver / BC / obstacle / outlet / projection

[INTERFACE_FORCE]
valid_markers = 24
invalid_markers = 0
two_sided_pressure_markers = 24
force_N = [0, 0, -0.00190900991473]
expected_streamwise_sign = negative z
action_reaction_residual = 3.23285339471e-11
diagnosis = interface-force gate passed

[SOLID_RESPONSE]
root_max_disp_m = 0
tip_mean_disp_m = [0, 4.19793650508e-07, -1.43796205521e-06]
max_disp_m = 1.52154848365e-06
reference_m = 5.1e-05
relative_error = 0.970165716007
solid_substeps_selected = 1600
solid_estimated_cfl = 0.030641756372
tip_dz_final_m = -1.43796205521e-06
tip_dz_min_m = -1.43796205521e-06
tip_dz_max_m = -1.43796205521e-06
tip_dz_monotonic_violation_count = 0
first_tip_dz_violation_step =
max_tip_dz_rebound_m =
tip_dz_sign_violation_count = 0
diagnosis = solid-response gate passed

[FSI_FEEDBACK]
updated_markers = 24
invalid_markers = 0
max_marker_displacement_m = 3.37677261086e-06
fluid_recomputed_after_feedback = false
feedback_closure_status = OPEN_LOOP_OR_PREFEEDBACK_ONLY
fluid_projection_consumed_feedback = false
fluid_feedback_constraint_marker_count = 0
fluid_feedback_constraint_active_cell_count = 0
fluid_feedback_constraint_cleared_cell_count = 0
fluid_feedback_constraint_obstacle_cell_count = 0
fluid_feedback_constraint_non_obstacle_cell_count = 0
fluid_feedback_constraint_projection_participating_cell_count = 0
no_slip_residual_before_mps =
no_slip_residual_after_mps =
no_slip_target_residual_after_assembly_mps =
no_slip_projected_residual_after_projection_mps = 0
diagnosis = feedback is downstream of current failing gate

[COORDINATE_MAPPING]
Fluent x <-> EasyFsi z
Fluent y <-> EasyFsi y
Fluent out-of-plane <-> EasyFsi x
fluent_comparison = not run; no Fluent tip-displacement CSV supplied
