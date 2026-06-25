[SETUP]
PASS geometry: duct_length=0.1, duct_height=0.04, flap_height=0.01, flap_thickness=0.003
PASS material: rho_s=1600, E=1000000, nu=0.47
PASS time: dt=0.0005, steps=50

[FLOW_ONLY]
velocity_peak_mps = 28.1565494537
official_range_mps = [20, 29]
pressure_min_pa = -324.161210016
pressure_max_pa = 206.22111361
projection_final_residual =
diagnosis = flow gate passed

[INTERFACE_FORCE]
valid_markers = 12
invalid_markers = 0
two_sided_pressure_markers = 12
force_N = [0, -4.02049885774e-06, -0.00173535963378]
expected_streamwise_sign = negative z
action_reaction_residual = 2.7931894113e-11
diagnosis = interface-force gate passed

[SOLID_RESPONSE]
root_max_disp_m = 0
tip_mean_disp_m = [0, 8.67573544383e-06, -2.56029888988e-05]
max_disp_m = 2.82664586848e-05
reference_m = 5.1e-05
relative_error = 0.445755712063
tip_dz_final_m = -2.56029888988e-05
tip_dz_min_m = -3.39206308126e-05
tip_dz_max_m = -7.16745853424e-06
tip_dz_monotonic_violation_count = 23
first_tip_dz_violation_step = 5
max_tip_dz_rebound_m = 5.47245144844e-06
tip_dz_sign_violation_count = 0
diagnosis = check solid history monotonicity / load persistence / time integration

[FSI_FEEDBACK]
updated_markers = 12
invalid_markers = 0
max_marker_displacement_m = 1.60639956448e-07
fluid_recomputed_after_feedback = false
feedback_closure_status = OPEN_LOOP_LOAD_REUSE
diagnosis = feedback is downstream of current failing gate

[COORDINATE_MAPPING]
Fluent x <-> EasyFsi z
Fluent y <-> EasyFsi y
Fluent out-of-plane <-> EasyFsi x
fluent_comparison = validation_runs\ansys_vertical_flap_fsi\official_web\fluent_tip_displacement_web_final.csv
