# ANSYS vertical-flap selected-formulation coupled step50

## Scope

This artifact records staged requested 10/30/50-step selected-formulation coupled validation. It does not claim Fluent parity.

## Candidate decision

- candidate_status: `selected_formulation_coupled_step50_passed`
- reference_formulation_candidate: `anchored_dual_face_pressure_pair_with_per_face_one_sided`
- pressure_pair_policy_candidate: `baseline_anchored_cell_pair`
- one_sided_pressure_policy_candidate: `per_face_mirrored`
- first_failed_scenario: ``
- first_failed_step: ``
- first_failed_gate: ``
- source_5step_smoke_matrix: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_smoke_diagnostics/traction_selected_formulation_coupled_smoke_matrix.json`

## Stage rows

scenario | status | completed/requested | invalid | one-sided | anchor selected | fallback | force residual | velocity growth | pressure growth | displacement growth | sign flips | first failed gate
--- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | ---
selected_formulation_coupled_step10 | passed | 10/10 | 0.0 | 24 | 24 | 0.0 | 1.947857e-10 | 1.128769e+00 | 1.507373e+00 | 3.717952e+00 | 1 | 
selected_formulation_coupled_step30 | passed | 30/30 | 0.0 | 24 | 24 | 0.0 | 1.947857e-10 | 1.128769e+00 | 1.507373e+00 | 3.714813e+00 | 2 | 
selected_formulation_coupled_step50 | passed | 50/50 | 0.0 | 24 | 24 | 0.0 | 1.947857e-10 | 1.128769e+00 | 1.507373e+00 | 3.755496e+00 | 4 | 

## Active blockers

- no_fluent_parity_claim: Fluent parity remains a later validation step

## Files

- Matrix JSON: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_step50_diagnostics/traction_selected_formulation_coupled_step50_matrix.json`
- Matrix CSV: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_step50_diagnostics/traction_selected_formulation_coupled_step50_matrix.csv`
- History JSON: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_step50_diagnostics/traction_selected_formulation_coupled_step50_history.json`
- Scenario diagnostics: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_step50_diagnostics/scenario_diagnostics`
- Checksums: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_step50_diagnostics/CHECKSUMS.sha256`
