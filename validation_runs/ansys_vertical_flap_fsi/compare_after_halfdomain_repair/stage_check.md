[SETUP]
PASS geometry: duct_length=0.1, duct_height=0.04, flap_height=0.01, flap_thickness=0.003
PASS material: rho_s=1600, E=1000000, nu=0.47
PASS time: dt=0.0005, steps=50

[PREFLOW]
steps_requested = 0
steps_completed = 0
converged = true
stop_reason = not_requested
history_rows = 0

[FLOW_ONLY]
velocity_peak_mps = 10.6815423965
velocity_p99_mps = 10
velocity_p999_mps = 10.1898175573
official_range_mps = [20, 29]
pressure_min_pa = -135.271006881
pressure_max_pa = 17.6711305446
projection_l2 = 1143.85323502
projection_max_abs = 6400
pre_projection_l2 = 1145.91290613
post_boundary_l2 = 1143.85323502
velocity_dirichlet_boundary_max_delta_mps = 10.0719394684
diagnosis = check fluid solver / BC / obstacle / outlet / projection

[INTERFACE_FORCE]
valid_markers = 24
invalid_markers = 0
two_sided_pressure_markers = 24
force_N = [0, -1.67665585429e-08, -7.833711021e-05]
expected_streamwise_sign = negative z
action_reaction_residual = 1.82539069485e-12
diagnosis = interface-force gate passed

[SOLID_RESPONSE]
root_max_disp_m = 0
tip_mean_disp_m = [0, 9.72487032413e-06, -4.25707548857e-06]
max_disp_m = 1.15921211545e-05
reference_m = 5.1e-05
relative_error = 0.772703506774
solid_substeps_selected = 1600
solid_estimated_cfl = 0.030641756372
tip_dz_final_m = -4.25707548857e-06
tip_dz_min_m = -2.29850411415e-05
tip_dz_max_m = 1.18315219879e-05
tip_dz_monotonic_violation_count = 23
first_tip_dz_violation_step = 4
max_tip_dz_rebound_m = 9.9828466773e-06
tip_dz_sign_violation_count = 8
diagnosis = solid-response gate passed

[FSI_FEEDBACK]
updated_markers = 24
invalid_markers = 0
max_marker_displacement_m = 1.11917370305e-06
fluid_recomputed_after_feedback = true
feedback_closure_status = CLOSED_LOOP_RECOMPUTED_AFTER_FEEDBACK
fluid_projection_consumed_feedback = true
fluid_feedback_constraint_marker_count = 24
fluid_feedback_constraint_active_cell_count = 24
fluid_feedback_constraint_cleared_cell_count = 24
fluid_feedback_constraint_obstacle_cell_count = 0
fluid_feedback_constraint_non_obstacle_cell_count = 24
fluid_feedback_constraint_projection_participating_cell_count = 24
no_slip_residual_before_mps = 0.000675105431583
no_slip_residual_after_mps = 0
no_slip_target_residual_after_assembly_mps = 0
no_slip_projected_residual_after_projection_mps = 2.69509246209e-06
diagnosis = feedback is downstream of current failing gate

[COORDINATE_MAPPING]
Fluent x <-> EasyFsi z
Fluent y <-> EasyFsi y
Fluent out-of-plane <-> EasyFsi x
fluent_comparison = validation_runs\ansys_vertical_flap_fsi\official_web\fluent_tip_displacement_web_final.csv
