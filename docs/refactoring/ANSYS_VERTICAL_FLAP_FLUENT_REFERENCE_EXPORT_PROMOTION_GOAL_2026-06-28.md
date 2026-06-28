# ANSYS Vertical Flap Fluent Reference Export Promotion Goal - 2026-06-28

## Baseline

The active branch is
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.
The current remote-backed baseline is commit
`a01fcac1e5050163bdf0746c1d5b3e81e0db0a7c`.

Current correct state:

- selected-formulation coupled step50 evidence is passed
- Fluent reference collection is pending
- Fluent parity remains blocked by incomplete reference data
- source export CSV files are schema-only
- metadata provenance and comparison conventions are still `MISSING`
- `fluent_reference_contract_2026-06-27.json` remains
  `fluent_reference_incomplete`
- `no_fluent_parity_claim` remains active
- no Fluent parity is claimed

The next physical validation step requires real Fluent source exports,
provenance, tolerances, and convention metadata. Those values are not present in
the repository yet, so this patch must not fabricate them.

## Objective

Add a safe promotion layer for future complete Fluent reference exports without
changing the current fail-closed status.

This patch should make the reference workflow explicit:

```text
schema-only source exports
  -> collection validator emits pending candidate contract
  -> active contract manifest keeps using current incomplete contract
  -> parity runner reads the active contract manifest
  -> parity remains blocked_reference_incomplete
```

When real Fluent exports are later committed, the same workflow should be able
to move to:

```text
complete source exports
  -> complete candidate contract
  -> promotion guard recommends versioned contract promotion
  -> active contract manifest can point to fluent_reference_contract_2026-06-28.json
  -> parity runner compares against that active contract
```

## Required Scope For This Patch

This patch must:

1. Add this detailed goal file and use it as the active Codex goal reference.
2. Add an active Fluent reference contract manifest under:

```text
validation_runs/ansys_vertical_flap_fsi/fluent_reference/active_fluent_reference_contract.json
```

3. Generate/update that manifest from the collection validator.
4. Keep the active manifest pointed at the existing incomplete contract while
   source exports remain schema-only:

```text
active_contract = fluent_reference_contract_2026-06-27.json
active_contract_status = fluent_reference_incomplete
promotion_status = blocked_reference_incomplete
recommended_action = keep_current_incomplete_contract
```

5. Record the candidate contract path, candidate contract SHA, active contract
   path, active contract SHA, promotion blockers, and whether
   `no_fluent_parity_claim` can be retired.
6. Make the Fluent parity runner resolve the active contract from the manifest
   instead of relying only on a hardcoded contract path.
7. Keep the fallback to `fluent_reference_contract_2026-06-27.json` if the
   manifest is missing, so older artifacts remain readable.
8. Strengthen tolerance validation so a tolerance is complete only when it has:
   - `status = available`
   - a finite numeric `value`
   - a non-placeholder `source`
9. Update collection artifact tests to verify the active manifest and promotion
   blockers.
10. Update parity artifact tests to verify the active manifest is referenced and
    the current active contract is still the incomplete 2026-06-27 contract.
11. Regenerate collection and parity artifacts.
12. Keep current artifacts fail-closed.

## Non-Goals

This patch must not:

- populate source export CSV rows without real Fluent data
- replace `MISSING` provenance fields with invented metadata
- invent tolerance values or tolerance sources
- create a fake complete `fluent_reference_contract_2026-06-28.json`
- switch the active contract to a complete contract
- retire `no_fluent_parity_claim`
- claim `fluent_parity_validated`
- run Fluent
- run heavy EasyFsi coupled validation
- change solver, selected formulation, material, geometry, or boundary logic

## Active Contract Manifest Contract

The active manifest must be deterministic JSON with repository-relative paths:

```json
{
  "case": "ansys_vertical_flap_fsi",
  "purpose": "active_fluent_reference_contract_manifest",
  "source_script": "validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py",
  "active_contract": "validation_runs/ansys_vertical_flap_fsi/fluent_reference/fluent_reference_contract_2026-06-27.json",
  "active_contract_sha256": "...",
  "active_contract_status": "fluent_reference_incomplete",
  "candidate_contract": "validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics/fluent_reference_collection_candidate_contract.json",
  "candidate_contract_sha256": "...",
  "candidate_contract_status": "fluent_reference_incomplete",
  "promotion_status": "blocked_reference_incomplete",
  "recommended_action": "keep_current_incomplete_contract",
  "promotion_blockers": [
    {"blocker": "fluent_displacement_reference_missing", "detail": "..."},
    {"blocker": "fluent_force_reference_missing", "detail": "..."},
    {"blocker": "fluent_flow_reference_missing", "detail": "..."},
    {"blocker": "fluent_pressure_reference_missing", "detail": "..."},
    {"blocker": "fluent_reference_provenance_incomplete", "detail": "..."},
    {"blocker": "fluent_reference_comparison_metadata_incomplete", "detail": "..."},
    {"blocker": "fluent_reference_tolerances_incomplete", "detail": "..."}
  ],
  "no_fluent_parity_claim_retired": false
}
```

The current manifest must not recommend validated parity. It must keep the
active contract on the incomplete 2026-06-27 contract.

If a future candidate contract is complete, the manifest may say:

```text
promotion_status = ready_for_versioned_contract_promotion
recommended_action = promote_versioned_contract
```

but the actual versioned contract promotion must remain a separate explicit
commit.

## Tolerance Provenance Contract

The collection validator currently checks tolerance availability. This patch
must also require tolerance source provenance.

Incomplete tolerance examples:

```json
{"status": "pending_reference", "value": null}
{"status": "available", "value": 0.1}
{"status": "available", "value": 0.1, "source": "MISSING"}
```

Complete tolerance example:

```json
{"status": "available", "value": 0.1, "source": "documented Fluent parity tolerance source"}
```

The current committed contract must remain incomplete because its tolerances are
still pending.

## Parity Runner Contract

The parity runner must:

- read `active_fluent_reference_contract.json` when present
- resolve `active_contract` as a repository-relative path
- verify that the active contract path is not absolute
- keep using the existing 2026-06-27 contract when that is what the manifest
  selects
- include active manifest path/SHA in matrix, history, row, and metadata
- keep the current candidate status as
  `fluent_parity_blocked_reference_incomplete`
- keep `no_fluent_parity_claim` active

## Tests

Update tests so they prove:

- active manifest exists
- active manifest uses repository-relative paths
- active manifest points to `fluent_reference_contract_2026-06-27.json`
- active contract SHA matches the file
- candidate contract SHA matches the diagnostics candidate contract
- promotion status is `blocked_reference_incomplete`
- recommended action is `keep_current_incomplete_contract`
- promotion blockers include missing displacement, force, flow, pressure,
  provenance, comparison metadata, and tolerances
- `no_fluent_parity_claim_retired = false`
- parity artifact records the active manifest path/SHA
- parity artifact still uses the incomplete active contract
- collection and parity summaries do not claim Fluent parity

## Verification Commands

Use the trusted local interpreter:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py `
  tests\integration\test_ansys_vertical_flap_fluent_reference_collection_artifacts.py `
  tests\integration\test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts.py `
  tests\integration\test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic.py

& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py

& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py

& 'D:\working\taichi\env\python.exe' -m unittest -v `
  tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic

git diff --check
```

## Acceptance Criteria

This goal is complete when:

- the goal file is committed
- the active manifest is committed and generated by the validator
- parity runner uses the active manifest
- tolerance provenance is required for completion
- current collection status remains `fluent_reference_collection_pending`
- current parity status remains `fluent_parity_blocked_reference_incomplete`
- tests and lightweight validation pass
- the final commit is pushed to the active remote branch

## Completion Statement

After this patch, the repository will have a safe path for future real Fluent
export promotion, but it will still honestly report that the current reference
data is incomplete and Fluent parity is not claimed.
