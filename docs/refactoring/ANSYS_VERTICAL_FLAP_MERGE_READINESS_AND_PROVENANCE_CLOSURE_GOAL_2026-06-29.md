# ANSYS Vertical Flap Merge Readiness And Provenance Closure Goal - 2026-06-29

## Source Branch And Commit

- Repository branch: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Review baseline commit: `c94332888fe09d792a119086a4969f78b03bb134`
- Baseline status: local branch is expected to match the reviewed remote branch before implementation starts.
- Remote CI status: do not claim GitHub Actions green unless a concrete run URL/status is available. If connector/API evidence remains empty, record the CI evidence as blocked and pending manual GitHub Actions verification.

## Primary Objective

Close the merge-readiness and provenance gaps for the ANSYS vertical flap validation branch without weakening the current fail-closed Fluent reference posture.

The implementation must:

1. Convert the branch merge checklist from a template-like document into an evidence-bearing readiness record.
2. Add commit/ref provenance to generated Fluent collection and parity artifact manifests.
3. Add machine-readable policy and hygiene checker reports.
4. Add tests that prevent future placeholder readiness claims, unknown manifest provenance, empty refactoring docs, and undocumented real-Fluent promotion gates.
5. Add a PR split strategy that keeps layout/package movement, benchmark namespace migration, ANSYS diagnostics, generic solver boundary work, and Fluent reference gates reviewable.
6. Regenerate committed artifacts and checksums after code changes.
7. Run focused local verification and push the completed branch only after verification passes.

## Non-Negotiable Claim Boundaries

- Real Fluent parity remains blocked until complete real Fluent source exports exist.
- Synthetic dry-runs may prove pipeline mechanics only; they must not be described as real Fluent parity.
- Any artifact that lacks real Fluent reference completeness must preserve fail-closed markers such as:
  - `fluent_parity_claimed = false`
  - `candidate_status = fluent_parity_blocked_reference_incomplete`
  - `reference_contract_status = fluent_reference_incomplete`
- Missing GitHub Actions evidence must be recorded as blocked/pending manual verification, not inferred from local test success.
- Source export schema-only or synthetic inputs must remain clearly separated from real solver evidence.

## Phase 1 - Merge Checklist Evidence

Target file:

- `docs/refactoring/ANSYS_VERTICAL_FLAP_BRANCH_MERGE_CHECKLIST_2026-06-29.md`

Required edits:

- Replace pending placeholders with concrete local evidence for review baseline commit `c94332888fe09d792a119086a4969f78b03bb134`.
- Record the remote CI evidence as unavailable when GitHub Actions run data is empty:
  - `Remote CI evidence: BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK`
  - `Remote CI source: NOT_AVAILABLE_CONNECTOR_EMPTY`
- Record local evidence for:
  - Python compile check passed.
  - Artifact regeneration passed.
  - Focused unittest suite passed, including the existing 53-test focused result where applicable.
  - Fluent artifact policy checker passed.
  - Validation artifact hygiene checker passed.
  - `git diff --cached --check` passed after staging.
  - Secret keyword scan found only documentation/rule references and no credential material.
- Preserve explicit claim boundaries for real Fluent parity and CI status.

Required tests:

- Extend `tests/integration/test_ansys_vertical_flap_branch_review_docs.py` to assert:
  - the baseline commit SHA appears in the checklist,
  - real Fluent claim boundaries are present,
  - the local focused test result is represented,
  - policy and hygiene checks are represented,
  - GitHub Actions URL/status is not fabricated,
  - the blocked manual CI marker is present.

## Phase 2 - Artifact Manifest Commit And Ref Provenance

Target files:

- `validation_runs/ansys_vertical_flap_fsi/scripts/run_fluent_reference_collection_validation.py`
- `validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_selected_formulation_fluent_parity.py`

Required edits:

- Replace any generated manifest value equivalent to `"generated_from_commit": "unknown"` with a concrete generated commit value.
- Add `"generated_from_ref"` to generated artifact manifests.
- Read provenance from environment variables:
  - `EASYFSI_VALIDATION_COMMIT`
  - `EASYFSI_VALIDATION_REF`
- Use conservative defaults when those variables are not set:
  - commit default: `c94332888fe09d792a119086a4969f78b03bb134`
  - ref default: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Regenerate collection and parity artifacts with those environment variables explicitly set.

Required tests:

- Extend artifact tests to assert:
  - `generated_from_commit` exists,
  - it is not `unknown`,
  - it is a 40-character lowercase hexadecimal commit id,
  - `generated_from_ref` matches the target branch,
  - generated outputs retain SHA/checksum metadata.

## Phase 3 - Policy And Hygiene Reports

Target files:

- `validation_runs/ansys_vertical_flap_fsi/scripts/check_fluent_artifact_policy.py`
- `scripts/check_validation_artifact_hygiene.py`

Required edits:

- Add an optional `--write-report <path>` argument to each checker.
- The default behavior must remain compatible with existing CLI use.
- Reports must be JSON, deterministic enough for committed artifact review, and include at least:
  - `status`
  - `checked_file_count`
  - `violations`
  - policy/check identifiers sufficient for tests and review.

Target generated reports:

- `validation_runs/ansys_vertical_flap_fsi/policy_reports/fluent_artifact_policy_report.json`
- `validation_runs/ansys_vertical_flap_fsi/policy_reports/validation_artifact_hygiene_report.json`

Required tests:

- Add `tests/integration/test_ansys_vertical_flap_policy_reports.py`.
- Assert both committed reports exist and include:
  - passed status,
  - `checked_file_count > 0`,
  - `violations == []`,
  - expected policy/check ids.

## Phase 4 - PR Split Strategy

Target files:

- `docs/refactoring/ANSYS_VERTICAL_FLAP_PR_SPLIT_STRATEGY_2026-06-29.md`
- `docs/refactoring/BRANCH_REVIEW_MAP_2026-06-29.md`

Required edits:

- Add a split strategy with five review slices:
  1. Layout and package relocation.
  2. `benchmarks.official` namespace migration.
  3. ANSYS selected-formulation diagnostics and artifacts.
  4. `generic_fsi_solver` plus runtime pressure-pair audit.
  5. Fluent reference schema gate and merge readiness closure.
- For each slice include:
  - Scope
  - Files
  - Risk
  - Tests
  - Artifacts
  - Do-not-merge-with guidance
- Include a single-PR fallback review order if reviewers decide not to split.
- Update the branch review map to link:
  - merge checklist,
  - PR split strategy,
  - collection artifact manifest,
  - parity artifact manifest,
  - real Fluent import gate documentation.

## Phase 5 - Non-Empty Refactoring Docs Guard

Target file:

- `tests/integration/test_refactoring_docs_are_nonempty.py`

Required behavior:

- Every `docs/refactoring/*.md` file must contain meaningful text.
- A reserved placeholder is allowed only when it contains:
  - `# Reserved`
  - a clear reason explaining why the file is intentionally reserved.
- If empty docs exist, fill them with reserved content or meaningful review content instead of deleting them unless deletion is clearly correct and scoped.

## Phase 6 - Validation Tools Package Goal Clarification

Target file:

- `docs/refactoring/ANSYS_VERTICAL_FLAP_VALIDATION_TOOLS_PACKAGE_GOAL_2026-06-30.md`

Required edits:

- Ensure the validation tools package goal mentions:
  - Fluent artifact policy checker,
  - validation artifact hygiene checker,
  - generated policy reports,
  - wrapper constraints that keep validation tools importable and runnable without making synthetic data look like real Fluent evidence.

## Phase 7 - Real Fluent Import Gate Documentation

Target files:

- `docs/validation/ANSYS_VERTICAL_FLAP_REAL_FLUENT_IMPORT_GATE_2026-06-29.md`
- `tests/integration/test_ansys_vertical_flap_real_fluent_import_gate_doc.py`

Required documentation content:

- Four real Fluent CSV source exports are required.
- All four CSVs must represent `step = 50`.
- Metadata must be complete, including source document, run id, author, and date.
- Source columns must not contain EasyFsi or HIBM-MPM placeholders.
- The collection validator must report complete real reference coverage.
- The active manifest may be promoted only after the collection gate passes.
- The parity runner may compare candidates against references, but it cannot claim Fluent parity until the gates pass.

Required tests:

- Assert the gate document exists and contains the real Fluent CSV count, step requirement, metadata requirements, source-column prohibition, collection validator requirement, manifest promotion rule, and parity-claim prohibition.

## Phase 8 - Regeneration And Verification

Required local commands or equivalent:

1. Compile changed Python scripts.
2. Regenerate Fluent reference collection artifacts.
3. Regenerate selected-formulation Fluent parity artifacts.
4. Run Fluent artifact policy checker and write its JSON report.
5. Run validation artifact hygiene checker and write its JSON report.
6. Run focused integration/unit tests covering:
   - branch review docs,
   - artifact provenance,
   - policy reports,
   - non-empty refactoring docs,
   - real Fluent import gate docs,
   - generic solver architecture,
   - runtime pressure-pair contract,
   - existing ANSYS vertical flap validation artifact gates.
7. Run `git diff --check`.
8. Stage intended changes.
9. Run `git diff --cached --check`.
10. Run a conservative secret keyword scan and confirm no credentials are present.

## Completion Criteria

- All new/updated tests pass locally.
- Generated artifact manifests include concrete commit/ref provenance.
- Policy and hygiene reports are generated, committed, and test-covered.
- Merge checklist is evidence-bearing and explicitly blocks missing remote CI evidence.
- Real Fluent parity remains fail-closed until real references are complete.
- Worktree is clean after commit.
- Branch is pushed to the configured remote after local verification passes.
- Final response reports:
  - final commit hash,
  - pushed branch,
  - verification commands/results,
  - remaining remote CI limitation if GitHub Actions evidence is still unavailable.
