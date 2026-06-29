# ANSYS Vertical-Flap Branch Merge Checklist 2026-06-29

Branch: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`

## Commit And CI

- Reviewed HEAD commit at checklist update: `25b8c60074f3cbcda4f24c611b97e2cf7fca6dc9`
- Artifact generation source commit: `c94332888fe09d792a119086a4969f78b03bb134`
- GitHub Actions run URL / run id: `BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK`
- CI run URL: `BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK`
- Remote CI evidence: `BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK`
- Remote CI source: `NOT_AVAILABLE_CONNECTOR_EMPTY`
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

Result: `PASSED_LOCAL`
Recorded local evidence: `PASSED_LOCAL`

### Artifact Regeneration

```powershell
python validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py
python validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py
```

Result: `PASSED_LOCAL`
Evidence: collection and parity artifacts regenerated from artifact generation source commit `c94332888fe09d792a119086a4969f78b03bb134` and ref `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`; the reviewed HEAD at checklist update was `25b8c60074f3cbcda4f24c611b97e2cf7fca6dc9`.

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
  tests.integration.test_ansys_vertical_flap_policy_reports `
  tests.integration.test_refactoring_docs_are_nonempty `
  tests.integration.test_ansys_vertical_flap_real_fluent_import_gate_doc `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic `
  tests.integration.test_ansys_vertical_flap_generic_solver_artifacts `
  tests.contracts.test_generic_fsi_solver_architecture `
  tests.solvers.test_pressure_sample_pair_provider_contract `
  tests.tools.test_validation_artifact_hygiene
```

Result: `PASSED_LOCAL`
Baseline focused evidence retained from the reviewed branch: `53 tests OK`.
Closure focused evidence: `80 tests OK`.
This closure adds manifest provenance, report, import-gate, and non-empty-doc guards.

### Policy And Hygiene

```powershell
python validation_runs\ansys_vertical_flap_fsi\scripts\check_fluent_artifact_policy.py --write-report validation_runs\ansys_vertical_flap_fsi\policy_reports\fluent_artifact_policy_report.json
python scripts\check_validation_artifact_hygiene.py --write-report validation_runs\ansys_vertical_flap_fsi\policy_reports\validation_artifact_hygiene_report.json
```

Result: `PASSED_LOCAL`
- Fluent artifact policy checker: `PASSED_LOCAL`
- Validation artifact hygiene checker: `PASSED_LOCAL`
- Policy reports:
  - `validation_runs/ansys_vertical_flap_fsi/policy_reports/fluent_artifact_policy_report.json`
  - `validation_runs/ansys_vertical_flap_fsi/policy_reports/validation_artifact_hygiene_report.json`

### Diff And Secret Scan

```powershell
git diff --check
git diff --cached --check
Select-String -Path <changed-files> -Pattern 'api[_-]?key','password','secret','token'
```

- `git diff --check` result: `PASSED_LOCAL`
- `git diff --cached --check` result: `PASSED_LOCAL`
- Secret scan result: `PASSED_LOCAL_DOCS_AND_RULE_TEXT_ONLY_NO_CREDENTIALS`

### Artifact Checksums

- Fluent reference collection `CHECKSUMS.sha256`: `PASSED_LOCAL`
- Fluent parity diagnostics `CHECKSUMS.sha256`: `PASSED_LOCAL`
- `ARTIFACT_MANIFEST.json` outputs match checksums: `PASSED_LOCAL`
- Collection manifest provenance:
  - `generated_from_commit=c94332888fe09d792a119086a4969f78b03bb134`
  - `generated_from_ref=solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
  - `artifact_generation_source_commit=c94332888fe09d792a119086a4969f78b03bb134`
  - `artifact_generation_source_ref=solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
  - `artifact_committed_in_review_head=25b8c60074f3cbcda4f24c611b97e2cf7fca6dc9`
- Parity manifest provenance:
  - `generated_from_commit=c94332888fe09d792a119086a4969f78b03bb134`
  - `generated_from_ref=solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
  - `artifact_generation_source_commit=c94332888fe09d792a119086a4969f78b03bb134`
  - `artifact_generation_source_ref=solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
  - `artifact_committed_in_review_head=25b8c60074f3cbcda4f24c611b97e2cf7fca6dc9`

## Reviewer Sign-Off

- [x] Claim boundary reviewed.
- [x] Generated artifacts reviewed.
- [x] Synthetic-only data did not enter real artifact roots.
- [x] Workflow includes focused tests and policy checks.
- [x] GitHub Actions run URL / run id is explicitly blocked pending manual GitHub Actions verification.
- [x] Ready to merge or split according to `BRANCH_REVIEW_MAP_2026-06-29.md`, subject to manual remote CI confirmation.
