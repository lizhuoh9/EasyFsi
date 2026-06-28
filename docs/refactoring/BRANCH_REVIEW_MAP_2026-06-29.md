# Branch Review Map 2026-06-29

Branch: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`

## Review Scope

This branch keeps the ANSYS vertical-flap Fluent parity workflow fail-closed while the real Fluent reference contract remains incomplete. The branch adds contract-v1 schema hardening, source-export CSV validation, active-manifest validation, public tutorial evidence policy, export protocol documentation, and CI checks that prevent overclaiming.

## Files To Review

- `validation_runs/ansys_vertical_flap_fsi/scripts/fluent_reference_contract_schema.py`
  - Contract v1 schema version, contract id, metric unit/source/extraction/time checks, tolerance comparator checks, sampling definitions, and comparison policy.
- `validation_runs/ansys_vertical_flap_fsi/scripts/fluent_source_export_schema.py`
  - Standalone CSV validator for headers, final step, final time, source provenance, and numeric metric values.
- `validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py`
  - Collection artifact generation, candidate contract schema validation, active manifest schema validation payload, public tutorial evidence map.
- `validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_selected_formulation_fluent_parity.py`
  - Active manifest schema/version/path/SHA validation and schema-status-gated parity claims.
- `validation_runs/ansys_vertical_flap_fsi/fluent_reference/fluent_reference_contract_2026-06-27.json`
  - Real contract remains incomplete and cannot retire `no_fluent_parity_claim`.
- `validation_runs/ansys_vertical_flap_fsi/fluent_reference/source_exports/public_tutorial_evidence_map.json`
  - Public ANSYS tutorial metadata policy: metadata only, not parity truth.
- `docs/validation/ANSYS_VERTICAL_FLAP_FLUENT_REFERENCE_EXPORT_PROTOCOL_2026-06-29.md`
  - Manual export protocol for future provenance-backed Fluent data collection.
- `.github/workflows/ansys-vertical-flap-validation.yml`
  - Focused tests and artifact overclaim scan.

## Expected State

- `fluent_parity_claimed` remains `false`.
- `candidate_status` remains `fluent_parity_blocked_reference_incomplete`.
- `fluent_reference_contract_2026-06-27.json` remains `fluent_reference_incomplete`.
- `active_fluent_reference_contract.json` uses `active_fluent_reference_contract_manifest_v1`.
- Public tutorial evidence is explicitly `metadata_only_not_parity_truth`.
- No generated artifact contains `fluent_parity_validated` or `Fluent parity validated` while the reference is incomplete.

## Focused Verification

Run these tests before approving or pushing:

```powershell
python -m py_compile validation_runs\ansys_vertical_flap_fsi\scripts\fluent_reference_contract_schema.py validation_runs\ansys_vertical_flap_fsi\scripts\fluent_source_export_schema.py validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py
python validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py
python validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py
python -m unittest tests.integration.test_ansys_vertical_flap_fluent_reference_contract_schema tests.integration.test_ansys_vertical_flap_fluent_source_export_schema tests.integration.test_ansys_vertical_flap_fluent_reference_export_protocol tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts -v
```
