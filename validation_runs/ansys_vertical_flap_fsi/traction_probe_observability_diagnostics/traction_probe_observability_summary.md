# ANSYS Traction Probe Observability Summary - 2026-06-26

scope_limit = fixed-solid traction probe observability only; no coupled 50-step or Fluent parity claim

reference_formulation_candidate = none

candidate_status = no_reference_formulation_candidate

candidate_blockers = required_formulation_unsupported, dual_face_one_sided_unsupported, dual_two_sided_offset_sensitivity_above_tolerance

## offset0p25 mechanism

offset0p25 duplicates pressure jump across both physical faces: primary and secondary both report non-trivial two-sided jumps.

primary = {"inside_rung_histogram": {"1": 12}, "inside_unique_nearest_cell_count": 12, "marker_count": 12, "mean_inside_pressure_pa": 16.824130657940014, "mean_outside_pressure_pa": 22.646397118415724, "mean_pressure_jump_pa": -5.822266460475711, "mean_total_traction_z_pa": -5.822266460475711, "outside_rung_histogram": {"0": 12}, "outside_unique_nearest_cell_count": 12, "valid_marker_count": 12}

secondary = {"inside_rung_histogram": {"1": 12}, "inside_unique_nearest_cell_count": 12, "marker_count": 12, "mean_inside_pressure_pa": 22.578028809029252, "mean_outside_pressure_pa": 16.722550088402418, "mean_pressure_jump_pa": 5.855478720626835, "mean_total_traction_z_pa": -5.855478720626835, "outside_rung_histogram": {"0": 12}, "outside_unique_nearest_cell_count": 12, "valid_marker_count": 12}

## offset0p51 mechanism

offset0p51 leaves the secondary face near zero because its inside/outside probes sample nearly equal pressure regions; the primary face carries the dominant jump.

primary = {"inside_rung_histogram": {"1": 12}, "inside_unique_nearest_cell_count": 12, "marker_count": 12, "mean_inside_pressure_pa": 16.824130657785325, "mean_outside_pressure_pa": 22.672928395079996, "mean_pressure_jump_pa": -5.848797737294671, "mean_total_traction_z_pa": -5.848797737294671, "outside_rung_histogram": {"0": 12}, "outside_unique_nearest_cell_count": 12, "valid_marker_count": 12}

secondary = {"inside_rung_histogram": {"0": 12}, "inside_unique_nearest_cell_count": 12, "marker_count": 12, "mean_inside_pressure_pa": 16.82413065778235, "mean_outside_pressure_pa": 16.68657748613522, "mean_pressure_jump_pa": 0.13755317164713046, "mean_total_traction_z_pa": -0.13755317164713046, "outside_rung_histogram": {"0": 12}, "outside_unique_nearest_cell_count": 12, "valid_marker_count": 12}

## offset1p00 mechanism

offset1p00 loses the thin-wall pressure jump: nearest-cell and pressure evidence show probes no longer straddle the jump cleanly.

primary = {"inside_rung_histogram": {"0": 12}, "inside_unique_nearest_cell_count": 12, "marker_count": 12, "mean_inside_pressure_pa": 22.578028809045154, "mean_outside_pressure_pa": 22.741139099064124, "mean_pressure_jump_pa": -0.1631102900189664, "mean_total_traction_z_pa": -0.1631102900189664, "outside_rung_histogram": {"0": 12}, "outside_unique_nearest_cell_count": 12, "valid_marker_count": 12}

secondary = {"inside_rung_histogram": {"0": 12}, "inside_unique_nearest_cell_count": 12, "marker_count": 12, "mean_inside_pressure_pa": 16.824130657691935, "mean_outside_pressure_pa": 16.58307168933762, "mean_pressure_jump_pa": 0.24105896835431562, "mean_total_traction_z_pa": -0.24105896835431562, "outside_rung_histogram": {"0": 12}, "outside_unique_nearest_cell_count": 12, "valid_marker_count": 12}

## Conclusion

The observability rerun archives marker-level inside/outside pressure, probe rung, distance, nearest-cell, fluid-weight, and traction decomposition evidence. It explains the dual/two-sided offset sensitivity but does not select a reference formulation and does not claim Fluent parity.
