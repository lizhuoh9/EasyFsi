# ANSYS vertical-flap selected-formulation coupled smoke

## Scope

This artifact records selected-formulation coupled smoke preflight evidence. It does not claim 50-step validation and does not claim Fluent parity.

## Candidate decision

- candidate_status: `selected_formulation_coupled_smoke_pending`
- smoke_status: `blocked_invalid_marker_sampling`
- reference_formulation_candidate: `anchored_dual_face_pressure_pair_with_per_face_one_sided`
- pressure_pair_policy_candidate: `baseline_anchored_cell_pair`
- one_sided_pressure_policy_candidate: `per_face_mirrored`
- requested_step_count: `5`
- completed_step_count: `1`
- invalid_marker_count_max: `24.0`

## Active blockers

- coupled_fsi_validation_pending: requested 5-step selected-formulation smoke has not passed
- no_fluent_parity_claim: Fluent parity remains a later validation step
- blocked_invalid_marker_sampling: first coupled preflight did not satisfy smoke gates

## Files

- Matrix JSON: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_smoke_diagnostics/traction_selected_formulation_coupled_smoke_matrix.json`
- Matrix CSV: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_smoke_diagnostics/traction_selected_formulation_coupled_smoke_matrix.csv`
- History JSON: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_smoke_diagnostics/traction_selected_formulation_coupled_smoke_history.json`
- Scenario diagnostics: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_smoke_diagnostics/scenario_diagnostics`
- Checksums: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_smoke_diagnostics/CHECKSUMS.sha256`
