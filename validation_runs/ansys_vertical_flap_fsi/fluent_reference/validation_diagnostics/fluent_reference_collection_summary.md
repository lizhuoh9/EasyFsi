# ANSYS vertical-flap Fluent reference collection

## Scope

This artifact validates committed Fluent reference source export schemas and provenance. It does not run Fluent, does not run EasyFsi, and does not claim Fluent parity.

## Candidate decision

- candidate_status: `fluent_reference_collection_pending`
- candidate_contract_status: `fluent_reference_incomplete`
- active_blockers: `fluent_displacement_reference_missing, fluent_force_reference_missing, fluent_flow_reference_missing, fluent_pressure_reference_missing, fluent_reference_provenance_incomplete`

## Source exports

artifact | file status | header status | final step | metric status
--- | --- | --- | --- | ---
fluent_tip_displacement_history.csv | schema_only | passed | missing_final_step | missing
fluent_force_history.csv | schema_only | passed | missing_final_step | missing
fluent_flow_balance_history.csv | schema_only | passed | missing_final_step | missing
fluent_pressure_summary_history.csv | schema_only | passed | missing_final_step | missing
fluent_metadata_2026-06-28.md | present_incomplete | not_applicable | not_applicable | incomplete

## Files

- Matrix JSON: `validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics/fluent_reference_collection_matrix.json`
- Matrix CSV: `validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics/fluent_reference_collection_matrix.csv`
- Candidate contract: `validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics/fluent_reference_collection_candidate_contract.json`
- Active contract manifest: `validation_runs/ansys_vertical_flap_fsi/fluent_reference/active_fluent_reference_contract.json`
- Public tutorial evidence map: `validation_runs/ansys_vertical_flap_fsi/fluent_reference/source_exports/public_tutorial_evidence_map.json`
- Artifact manifest: `validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics/ARTIFACT_MANIFEST.json`
- Checksums: `validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics/CHECKSUMS.sha256`
