# ANSYS vertical-flap selected-formulation Fluent parity

## Scope

This artifact compares committed selected-formulation step50 evidence against the explicit Fluent reference contract. It does not claim Fluent parity while the reference contract is incomplete.

## Candidate decision

- candidate_status: `fluent_parity_blocked_reference_incomplete`
- active_blockers: `fluent_reference_incomplete, no_fluent_parity_claim`
- source_step50_candidate_status: `selected_formulation_coupled_step50_passed`
- reference_contract_status: `fluent_reference_incomplete`

## Metric gates

metric | gate status
--- | ---
displacement | blocked_reference_missing
force | blocked_reference_missing
flow_outlet | blocked_reference_missing
pressure | blocked_reference_missing
metadata | passed

## Files

- Matrix JSON: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/traction_selected_formulation_fluent_parity_matrix.json`
- Matrix CSV: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/traction_selected_formulation_fluent_parity_matrix.csv`
- History JSON: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/traction_selected_formulation_fluent_parity_history.json`
- Scenario diagnostics: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/scenario_diagnostics`
- Artifact manifest: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/ARTIFACT_MANIFEST.json`
- Checksums: `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/CHECKSUMS.sha256`
