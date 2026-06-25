[SETUP]
PASS geometry: duct_length=0.1, duct_height=0.04, flap_height=0.01, flap_thickness=0.003
PASS material: rho_s=1600, E=1000000, nu=0.47
PASS time: dt=0.0005, steps=50

[FLOW_ONLY]
velocity_peak_mps = 10.4172229767
official_range_mps = [20, 29]
pressure_min_pa = -136.474249776
pressure_max_pa = 16.4704247666
projection_final_residual = 
projection_l2 = 1144.05692371
projection_max_abs = 6400
pre_projection_l2 = 1146.05694984
post_boundary_l2 = 1144.05692371
velocity_dirichlet_boundary_max_delta_mps = 10.0722332001
diagnosis = check fluid solver / BC / obstacle / outlet / projection

[INTERFACE_FORCE]
valid_markers = 12
invalid_markers = 0
two_sided_pressure_markers = 12
force_N = [0, 6.6429473796e-09, -5.13383553348e-05]
expected_streamwise_sign = negative z
action_reaction_residual = 2.40776045579e-13
diagnosis = interface-force gate passed

[SOLID_RESPONSE]
root_max_disp_m = 0
tip_mean_disp_m = [0, 9.83872450888e-06, -1.19674950838e-06]
max_disp_m = 1.07899240902e-05
reference_m = 5.1e-05
relative_error = 0.788432860976
tip_dz_final_m = -1.19674950838e-06
tip_dz_min_m = -2.60705128312e-05
tip_dz_max_m = 1.16592273116e-05
tip_dz_monotonic_violation_count = 23
first_tip_dz_violation_step = 4
max_tip_dz_rebound_m = 1.10566616058e-05
tip_dz_sign_violation_count = 10
diagnosis = solid-response gate passed

[FSI_FEEDBACK]
updated_markers = 12
invalid_markers = 0
max_marker_displacement_m = 1.02023579984e-06
fluid_recomputed_after_feedback = true
feedback_closure_status = CLOSED_LOOP_RECOMPUTED_AFTER_FEEDBACK
diagnosis = feedback is downstream of current failing gate

[COORDINATE_MAPPING]
Fluent x <-> EasyFsi z
Fluent y <-> EasyFsi y
Fluent out-of-plane <-> EasyFsi x
fluent_comparison = validation_runs\ansys_vertical_flap_fsi\official_web\fluent_tip_displacement_web_final.csv
