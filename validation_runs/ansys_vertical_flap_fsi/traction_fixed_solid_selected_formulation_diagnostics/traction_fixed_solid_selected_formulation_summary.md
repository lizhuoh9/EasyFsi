# ANSYS vertical-flap fixed-solid selected formulation

## Scope

This artifact validates the selected reference formulation against committed fixed-solid source/load evidence while reusing the confirmed shared fixed-solid snapshot and its selected marker-traction rows.

It does not claim coupled FSI and does not claim Fluent parity.

## Candidate decision

- candidate_status: `fixed_solid_selected_formulation_validated`
- reference_formulation_candidate: `anchored_dual_face_pressure_pair_with_per_face_one_sided`
- pressure_pair_policy_candidate: `baseline_anchored_cell_pair`
- one_sided_pressure_policy_candidate: `per_face_mirrored`
- fixed_solid_snapshot_policy: `confirmed_shared_fixed_solid_snapshot_reused`
- fixed-solid snapshot SHA-256: `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`
- anchor map SHA-256: `9e7af66b2d5eac9ef8f4a609a347178413ef5a48a91877fcca84b2c08f9d7b88`

## Active blockers

- coupled_fsi_validation_pending: selected formulation has not been advanced in coupled FSI
- no_fluent_parity_claim: Fluent parity remains a later coupled-validation step

## Selection rows

| scenario | source scenario | component | policy | one-sided policy |
|---|---|---|---|---|
| fixed_solid_selected_baseline_probe0p51 | reference_baseline_anchored_two_sided_probe0p51 | pressure_pair_preselection | baseline_anchored_cell_pair | disabled |
| fixed_solid_selected_anchored_probe0p00 | reference_anchored_two_sided_probe0p00 | pressure_pair_preselection | baseline_anchored_cell_pair | disabled |
| fixed_solid_selected_anchored_probe0p25 | reference_anchored_two_sided_probe0p25 | pressure_pair_preselection | baseline_anchored_cell_pair | disabled |
| fixed_solid_selected_anchored_probe0p51 | reference_baseline_anchored_two_sided_probe0p51 | pressure_pair_preselection | baseline_anchored_cell_pair | disabled |
| fixed_solid_selected_anchored_probe0p625 | reference_anchored_two_sided_probe0p625 | pressure_pair_preselection | baseline_anchored_cell_pair | disabled |
| fixed_solid_selected_anchored_probe1p00 | reference_anchored_two_sided_probe1p00 | pressure_pair_preselection | baseline_anchored_cell_pair | disabled |
| fixed_solid_selected_per_face_one_sided_probe0p51 | reference_per_face_one_sided_probe0p51 | per_face_one_sided_pressure | baseline_anchored_cell_pair | per_face_mirrored |
| fixed_solid_selected_per_face_one_sided_probe0p625 | reference_per_face_one_sided_probe0p625 | per_face_one_sided_pressure | baseline_anchored_cell_pair | per_face_mirrored |
| fixed_solid_selected_per_face_one_sided_probe1p00 | reference_per_face_one_sided_probe1p00 | per_face_one_sided_pressure | baseline_anchored_cell_pair | per_face_mirrored |

## Files

- Matrix JSON: `validation_runs/ansys_vertical_flap_fsi/traction_fixed_solid_selected_formulation_diagnostics/traction_fixed_solid_selected_formulation_matrix.json`
- Matrix CSV: `validation_runs/ansys_vertical_flap_fsi/traction_fixed_solid_selected_formulation_diagnostics/traction_fixed_solid_selected_formulation_matrix.csv`
- History JSON: `validation_runs/ansys_vertical_flap_fsi/traction_fixed_solid_selected_formulation_diagnostics/traction_fixed_solid_selected_formulation_history.json`
- Marker diagnostics: `validation_runs/ansys_vertical_flap_fsi/traction_fixed_solid_selected_formulation_diagnostics/marker_diagnostics`
- Checksums: `validation_runs/ansys_vertical_flap_fsi/traction_fixed_solid_selected_formulation_diagnostics/CHECKSUMS.sha256`
