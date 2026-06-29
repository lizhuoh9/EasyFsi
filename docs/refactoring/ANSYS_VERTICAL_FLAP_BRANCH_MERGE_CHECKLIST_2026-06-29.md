# ANSYS Vertical-Flap Branch Merge Checklist 2026-06-29

Branch: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`

## Commit And CI

- Commit SHA: `PENDING_FINAL_COMMIT`
- GitHub Actions run URL / run id: `PENDING_REMOTE_CI_RUN`
- CI run URL: `PENDING_REMOTE_CI_RUN`
- Local interpreter path: `D:\working\taichi\env\python.exe`

## Claim Boundary

- `fluent_parity_claimed=false`
- `fluent_reference_incomplete`
- `no_fluent_parity_claim`
- No EasyFsi output is promoted as Fluent truth.
- No HIBM-MPM output is promoted as Fluent truth.
- Public tutorial evidence remains `metadata_only_not_parity_truth`.

## Required Commands

### Py Compile

```powershell
python -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\fluent_reference_contract_schema.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\fluent_source_export_schema.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\build_synthetic_fluent_reference_fixture.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\check_fluent_artifact_policy.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py `
  scripts\check_validation_artifact_hygiene.py
```

Result: `PENDING`

### Artifact Regeneration

```powershell
python validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py
python validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py
```

Result: `PENDING`

### Focused Unit Tests

```powershell
python -m unittest -v `
  tests.integration.test_ansys_vertical_flap_branch_review_docs `
  tests.integration.test_ansys_vertical_flap_fluent_reference_contract_schema `
  tests.integration.test_ansys_vertical_flap_fluent_source_export_schema `
  tests.integration.test_ansys_vertical_flap_fluent_reference_export_protocol `
  tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts `
  tests.integration.test_ansys_vertical_flap_fluent_reference_synthetic_pipeline `
  tests.integration.test_ansys_vertical_flap_fluent_artifact_policy `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic `
  tests.tools.test_validation_artifact_hygiene
```

Result: `PENDING`

### Policy And Hygiene

```powershell
python validation_runs\ansys_vertical_flap_fsi\scripts\check_fluent_artifact_policy.py
python scripts\check_validation_artifact_hygiene.py
```

Result: `PENDING`

### Diff And Secret Scan

```powershell
git diff --check
Select-String -Path <changed-files> -Pattern 'api[_-]?key','password','secret','token'
```

- `git diff --check` result: `PENDING`
- Secret scan result: `PENDING`

### Artifact Checksums

- Fluent reference collection `CHECKSUMS.sha256`: `PENDING`
- Fluent parity diagnostics `CHECKSUMS.sha256`: `PENDING`
- `ARTIFACT_MANIFEST.json` outputs match checksums: `PENDING`

## Reviewer Sign-Off

- [ ] Claim boundary reviewed.
- [ ] Generated artifacts reviewed.
- [ ] Synthetic-only data did not enter real artifact roots.
- [ ] Workflow includes focused tests and policy checks.
- [ ] GitHub Actions run URL / run id recorded when available.
- [ ] Ready to merge or split according to `BRANCH_REVIEW_MAP_2026-06-29.md`.
