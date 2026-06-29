# ANSYS Vertical-Flap Provenance Wording Closure Goal - 2026-06-29

## Source Review

- Target branch: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Current reviewed branch HEAD before this closure: `25b8c60074f3cbcda4f24c611b97e2cf7fca6dc9`
- Previous artifact generation source commit: `c94332888fe09d792a119086a4969f78b03bb134`
- Remote CI state from review attachment: GitHub Actions evidence is still unavailable through the connector; keep `BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK` and do not claim remote CI green.

## Objective

Make one small final closure commit that clarifies commit provenance wording for merge readiness and generated artifact manifests without changing solver behavior, Fluent parity logic, reference collection logic, or claim boundaries.

The current branch is already close to review/merge readiness. The remaining issue is semantic ambiguity: the branch HEAD is `25b8c60074f3cbcda4f24c611b97e2cf7fca6dc9`, while checklist and artifact manifests still record `c94332888fe09d792a119086a4969f78b03bb134` under broad wording such as `Commit SHA` or `generated_from_commit`. That can mislead reviewers into thinking the artifact generation source commit is the final reviewed HEAD.

## Non-Goals

- Do not modify solver physics.
- Do not modify parity pass/fail logic.
- Do not claim `fluent_parity_validated`.
- Do not promote real Fluent reference contracts.
- Do not replace the remote CI blocked marker with a fabricated status.
- Do not add new validation features beyond provenance wording and tests.

## Required Claim Boundaries

The following must remain true after this closure:

- `fluent_parity_claimed = false`
- parity artifact `candidate_status = fluent_parity_blocked_reference_incomplete`
- parity artifact `reference_contract_status = fluent_reference_incomplete`
- collection artifact remains reference incomplete / pending
- real Fluent import gate remains required before any parity claim
- synthetic / EasyFsi / HIBM-MPM data is not promoted as real Fluent truth

## Phase 1 - Checklist Wording Closure

Target file:

- `docs/refactoring/ANSYS_VERTICAL_FLAP_BRANCH_MERGE_CHECKLIST_2026-06-29.md`

Required edits:

- Replace broad `Commit SHA` wording with explicit roles:
  - `Reviewed HEAD commit at checklist update: 25b8c60074f3cbcda4f24c611b97e2cf7fca6dc9`
  - `Artifact generation source commit: c94332888fe09d792a119086a4969f78b03bb134`
- Preserve:
  - `GitHub Actions run URL / run id: BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK`
  - `Remote CI source: NOT_AVAILABLE_CONNECTOR_EMPTY`
- Fix the Py Compile section so it says `Result: PASSED_LOCAL`, not `Result: PENDING`.
- Keep local evidence including `53 tests OK`, `80 tests OK`, policy/hygiene passed, diff checks passed, and docs/rules-only secret scan.

Required tests:

- Update `tests/integration/test_ansys_vertical_flap_branch_review_docs.py` to assert:
  - reviewed HEAD commit phrase exists,
  - artifact generation source commit phrase exists,
  - old broad `Commit SHA` wording is absent or no longer used as the primary field,
  - no `Result: PENDING` remains,
  - remote CI blocked marker remains.

## Phase 2 - Artifact Manifest Provenance Field Closure

Target generated artifact files:

- `validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics/ARTIFACT_MANIFEST.json`
- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/ARTIFACT_MANIFEST.json`

Target generator files:

- `validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py`
- `validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_selected_formulation_fluent_parity.py`

Required edits:

- Preserve existing `generated_from_commit` / `generated_from_ref` for compatibility.
- Add semantically explicit fields:
  - `artifact_generation_source_commit`
  - `artifact_generation_source_ref`
  - `artifact_committed_in_review_head`
- Use existing environment/default values for generation source:
  - source commit default: `c94332888fe09d792a119086a4969f78b03bb134`
  - source ref default: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Add a stable default for committed/reviewed HEAD:
  - `25b8c60074f3cbcda4f24c611b97e2cf7fca6dc9`
- Allow `EASYFSI_VALIDATION_REVIEW_HEAD` to override `artifact_committed_in_review_head` when regenerating future manifests.

Required tests:

- Update collection and parity artifact tests to assert:
  - `generated_from_commit` still equals the source commit,
  - `generated_from_ref` still equals the source ref,
  - `artifact_generation_source_commit` equals the source commit,
  - `artifact_generation_source_ref` equals the source ref,
  - `artifact_committed_in_review_head` equals `25b8c60074f3cbcda4f24c611b97e2cf7fca6dc9`,
  - all commit-like fields are 40-character lowercase hexadecimal strings.

## Phase 3 - Regenerate Only Required Artifacts

Regenerate the two manifest-containing artifact roots using explicit environment variables:

- `EASYFSI_VALIDATION_COMMIT=c94332888fe09d792a119086a4969f78b03bb134`
- `EASYFSI_VALIDATION_REF=solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- `EASYFSI_VALIDATION_REVIEW_HEAD=25b8c60074f3cbcda4f24c611b97e2cf7fca6dc9`

Expected artifact states after regeneration:

- collection runner writes `fluent_reference_collection_pending`
- parity runner writes `fluent_parity_blocked_reference_incomplete`
- policy report remains passed
- hygiene report remains passed

## Phase 4 - Verification

Run at minimum:

```powershell
python -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py `
  tests\integration\test_ansys_vertical_flap_branch_review_docs.py `
  tests\integration\test_ansys_vertical_flap_fluent_reference_collection_artifacts.py `
  tests\integration\test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts.py
```

```powershell
python -m unittest -v `
  tests.integration.test_ansys_vertical_flap_branch_review_docs `
  tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts
```

Also run:

- `python validation_runs\ansys_vertical_flap_fsi\scripts\check_fluent_artifact_policy.py --write-report validation_runs\ansys_vertical_flap_fsi\policy_reports\fluent_artifact_policy_report.json`
- `python scripts\check_validation_artifact_hygiene.py --write-report validation_runs\ansys_vertical_flap_fsi\policy_reports\validation_artifact_hygiene_report.json`
- `git diff --check`
- after staging, `git diff --cached --check`
- changed-file secret keyword scan, expecting docs/rules-only matches and no credential material

## Completion Criteria

- Checklist distinguishes reviewed HEAD from artifact generation source commit.
- Py Compile checklist result no longer contains `PENDING`.
- Artifact manifests contain both compatibility provenance fields and semantically explicit provenance fields.
- Tests pass locally.
- Policy/hygiene reports remain passed.
- Worktree is clean after commit.
- Branch is pushed to `origin/solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.
- Final response reports final commit hash, pushed branch, local validation results, and the remaining remote CI blocker.
