# ANSYS vertical-flap selected-formulation coupled smoke

## Scope

This artifact preserves a one-step anchor-injected preflight row and records a true requested five-step coupled smoke row. It does not claim 50-step validation and does not claim Fluent parity.

## Candidate decision

- candidate_status: `selected_formulation_coupled_smoke_passed`
- preflight_smoke_status: `blocked_requested_5step_not_completed`
- five_step_smoke_status: `passed`
- reference_formulation_candidate: `anchored_dual_face_pressure_pair_with_per_face_one_sided`
- pressure_pair_policy_candidate: `baseline_anchored_cell_pair`
- one_sided_pressure_policy_candidate: `per_face_mirrored`
- preflight_completed_step_count: `1`
- five_step_completed_step_count: `5`
- five_step_invalid_marker_count_max: `0.0`
- five_step_first_failed_step: ``
- five_step_first_failed_gate: ``

## Five-step history

step | invalid | one-sided | anchor selected | anchor fallback | force residual | max velocity | max pressure | max displacement
--- | --- | --- | --- | --- | --- | --- | --- | ---
1 | 0 | 24 | 24 | 0 | 1.947857e-10 | 2.814045e+01 | 3.240767e+02 | 6.181899e-06
2 | 0 | 24 | 24 | 0 | 4.648993e-12 | 3.176406e+01 | 4.004515e+02 | 2.005325e-05
3 | 0 | 24 | 24 | 0 | 1.662222e-11 | 2.888934e+01 | 4.009129e+02 | 2.293519e-05
4 | 0 | 24 | 24 | 0 | 3.340565e-11 | 2.282744e+01 | 4.073830e+02 | 1.808823e-05
5 | 0 | 24 | 24 | 0 | 4.275641e-11 | 1.649194e+01 | 4.320848e+02 | 8.770841e-06

## Active blockers

- long_coupled_validation_pending: 5-step smoke passed but 30/50-step coupled validation remains pending
- no_fluent_parity_claim: Fluent parity remains a later validation step

## Files

- Matrix JSON: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_smoke_diagnostics/traction_selected_formulation_coupled_smoke_matrix.json`
- Matrix CSV: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_smoke_diagnostics/traction_selected_formulation_coupled_smoke_matrix.csv`
- History JSON: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_smoke_diagnostics/traction_selected_formulation_coupled_smoke_history.json`
- Scenario diagnostics: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_smoke_diagnostics/scenario_diagnostics`
- Checksums: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_smoke_diagnostics/CHECKSUMS.sha256`
