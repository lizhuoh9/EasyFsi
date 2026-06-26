# ANSYS Vertical-Flap Traction Formulation Verification

Date: 2026-06-26

This EasyFsi diagnostic keeps `step_count=0`, uses fixed-solid `preflow_steps=20`, disables marker feedback, and resamples the ANSYS vertical-flap marker traction under explicit marker-layout, pressure-sampling, offset, and viscous-traction controls.

## Result

baseline_scenario = dual_two_sided_offset0p51_pressure_only

reference_formulation_candidate = none

candidate_status = no_reference_formulation_candidate

candidate_blockers = pressure_probe_diagnostics_incomplete, primary_pressure_probe_diagnostics_incomplete, secondary_pressure_probe_diagnostics_incomplete, pressure_complete_count_inconsistent, probe_rung_diagnostics_incomplete, probe_cell_diagnostics_incomplete, traction_decomposition_missing, required_formulation_unsupported, dual_face_one_sided_unsupported, dual_two_sided_offset_sensitivity_above_tolerance

offset_sensitivity_status = offset_sensitivity_above_tolerance

offset_force_ratio_min = 0.06751512975451793

offset_force_ratio_max = 1.9507284752819152

offset_force_relative_span = 1.8832133455273972

formulation_agreement_status = blocked_dual_one_sided_unsupported

dual_one_sided_vs_single_mid_relative_error = none

flow_snapshot_identity_status = flow_metrics_match_completed_rows

supported_formulation_count = 5

unsupported_formulation_count = 1

The current matrix treats the 0.51-cell dual/two-sided row as a baseline scenario, not as a validated reference formulation. It does not promote a reference formulation when any required row is unsupported, failed, pressure-probe incomplete, offset-sensitive, formulation-disagreeing, or flow-snapshot divergent. The dual physical-face plus one-sided surface-pressure row is report-only because the current core exposes a single `one_sided_pressure_region_id`, not per-face one-sided region support.

## Runtime Finding

The core exposes force, traction, marker-count, stress-counter, action-reaction residual, per-face pressure-probe data, and per-face pressure means for this diagnostic. Candidate promotion remains fail-closed when the required layout-specific pressure fields are blank; status `exposed_by_core_stress_face_diagnostics; mean field is pressure_jump_pa`.

## Scope Limits

- No coupled FSI release was run.
- No 50-step run was performed.
- No Fluent force-history import was used.
- No Fluent parity claim is made.
- No solid material, damping, support-radius, or gate threshold was tuned.
- Unsupported pressure-sampling modes are archived as unsupported instead of faked.
