# ANSYS Vertical-Flap Traction Formulation Diagnostics

baseline_scenario = dual_two_sided_offset0p51_pressure_only
reference_formulation_candidate = none
candidate_status = no_reference_formulation_candidate
candidate_blockers = pressure_probe_diagnostics_incomplete, required_formulation_unsupported, dual_face_one_sided_unsupported, dual_two_sided_offset_sensitivity_above_tolerance
offset_sensitivity_status = offset_sensitivity_above_tolerance
offset_force_ratio_min = 0.06751512975451793
offset_force_ratio_max = 1.9507284752819152
offset_force_relative_span = 1.8832133455273972
formulation_agreement_status = blocked_dual_one_sided_unsupported
dual_one_sided_vs_single_mid_relative_error = none
flow_snapshot_identity_status = flow_metrics_match_completed_rows
supported_formulation_count = 5
unsupported_formulation_count = 1
pressure_mean_status = not_exposed_by_current_core; force and traction counters are archived
scope_limit = fixed-solid traction formulation diagnostic only; no coupled 50-step or Fluent parity claim

candidate_rule = all A/B/C rows supported, completed, conservative, pressure-probe complete, offset-stable, formulation-agreeing, and flow-snapshot identical

## Matrix

| scenario | status | layout | pressure mode | viscous | total force z N | ratio to baseline | flow snapshot | reason |
|---|---|---|---|---|---|---|---|---|
| dual_two_sided_offset0p51_pressure_only | completed | dual_physical_faces | two_sided_pressure_jump | False | -0.00017959052273141572 | 1.0 | 31.6403141022|25.1116390228|0.984664705351|0.0225098139034|-4.47271580767|25.9990321205 | completed; pressure means not_exposed_by_current_core; force and traction counters are archived |
| dual_one_sided_offset0p51_pressure_only | unsupported | dual_physical_faces | one_sided_surface_pressure | False |  |  |  | dual-face one-sided pressure needs per-face one-sided region support; current core exposes one one_sided_pressure_region_id |
| single_mid_two_sided_offset0p00_pressure_only | completed | single_mid_surface | two_sided_pressure_jump | False | -0.0001726169401754124 | 0.9611695402967751 | 31.6403141022|25.1116390228|0.984664705351|0.0225098139034|-4.47271580767|25.9990321205 | completed; pressure means not_exposed_by_current_core; force and traction counters are archived |
| dual_two_sided_offset0p25_pressure_only | completed | dual_physical_faces | two_sided_pressure_jump | False | -0.00035033234658293674 | 1.9507284752819152 | 31.6403141022|25.1116390228|0.984664705351|0.0225098139034|-4.47271580767|25.9990321205 | completed; pressure means not_exposed_by_current_core; force and traction counters are archived |
| dual_two_sided_offset1p00_pressure_only | completed | dual_physical_faces | two_sided_pressure_jump | False | -1.2125077444893233e-05 | 0.06751512975451793 | 31.6403141022|25.1116390228|0.984664705351|0.0225098139034|-4.47271580767|25.9990321205 | completed; pressure means not_exposed_by_current_core; force and traction counters are archived |
| dual_two_sided_offset0p51_viscous_air | completed | dual_physical_faces | two_sided_pressure_jump | True | -0.00017616778972252313 | 0.9809414608475113 | 31.6403141022|25.1116390228|0.984664705351|0.0225098139034|-4.47271580767|25.9990321205 | completed; pressure means not_exposed_by_current_core; force and traction counters are archived |
