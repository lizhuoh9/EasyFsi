# ANSYS vertical-flap reference formulation selection

## Scope

This artifact selects a shared snapshot traction reference formulation candidate from already committed pressure-pair and per-face one-sided component evidence. It reuses marker-traction sampling evidence and does not advance the flow, the structure, or a coupled FSI loop.

It does not claim Fluent parity; fixed-solid and coupled validations remain pending.

## Candidate decision

- candidate_status: `reference_formulation_candidate_selected`
- reference_formulation_candidate: `anchored_dual_face_pressure_pair_with_per_face_one_sided`
- pressure_pair_policy_candidate: `baseline_anchored_cell_pair`
- one_sided_pressure_policy_candidate: `per_face_mirrored`
- shared snapshot SHA-256: `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`

## Active blockers

- sampling_only_no_coupled_fsi: selection reuses shared snapshot marker-traction sampling only
- no_fluent_parity_claim: Fluent parity remains a later coupled-validation step
- fixed_solid_regenerated_validation_pending: selected formulation has not been rerun on regenerated fixed-solid evidence
- coupled_fsi_validation_pending: selected formulation has not been advanced in coupled FSI

## Selection rows

| scenario | source scenario | component | policy | one-sided policy |
|---|---|---|---|---|
| reference_baseline_anchored_two_sided_probe0p51 | baseline_anchored_two_sided_probe0p51 | pressure_pair_preselection | baseline_anchored_cell_pair | disabled |
| reference_anchored_two_sided_probe0p00 | anchored_pair_dual_faces_probe0p00 | pressure_pair_preselection | baseline_anchored_cell_pair | disabled |
| reference_anchored_two_sided_probe0p25 | anchored_pair_dual_faces_probe0p25 | pressure_pair_preselection | baseline_anchored_cell_pair | disabled |
| reference_anchored_two_sided_probe0p375 | anchored_pair_dual_faces_probe0p375 | pressure_pair_preselection | baseline_anchored_cell_pair | disabled |
| reference_anchored_two_sided_probe0p625 | anchored_pair_dual_faces_probe0p625 | pressure_pair_preselection | baseline_anchored_cell_pair | disabled |
| reference_anchored_two_sided_probe1p00 | anchored_pair_dual_faces_probe1p00 | pressure_pair_preselection | baseline_anchored_cell_pair | disabled |
| reference_per_face_one_sided_probe0p51 | dual_one_sided_per_face_probe0p51 | per_face_one_sided_pressure | baseline_anchored_cell_pair | per_face_mirrored |
| reference_per_face_one_sided_probe0p625 | dual_one_sided_per_face_probe0p625 | per_face_one_sided_pressure | baseline_anchored_cell_pair | per_face_mirrored |
| reference_per_face_one_sided_probe1p00 | dual_one_sided_per_face_probe1p00 | per_face_one_sided_pressure | baseline_anchored_cell_pair | per_face_mirrored |

## Files

- Matrix JSON: `validation_runs/ansys_vertical_flap_fsi/traction_reference_formulation_selection_diagnostics/traction_reference_formulation_selection_matrix.json`
- Matrix CSV: `validation_runs/ansys_vertical_flap_fsi/traction_reference_formulation_selection_diagnostics/traction_reference_formulation_selection_matrix.csv`
- History JSON: `validation_runs/ansys_vertical_flap_fsi/traction_reference_formulation_selection_diagnostics/traction_reference_formulation_selection_history.json`
- Marker diagnostics: `validation_runs/ansys_vertical_flap_fsi/traction_reference_formulation_selection_diagnostics/marker_diagnostics`
- Checksums: `validation_runs/ansys_vertical_flap_fsi/traction_reference_formulation_selection_diagnostics/CHECKSUMS.sha256`
