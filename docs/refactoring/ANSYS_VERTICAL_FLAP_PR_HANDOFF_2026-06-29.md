# ANSYS Vertical-Flap PR Handoff 2026-06-29

## Current PR Head

- Current PR head: `97de386279dcaa9e00693b8344d082a21a0114f9`
- Branch: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Suggested PR title: `Validation: fail-closed ANSYS vertical-flap generic and Fluent reference gates`

## Branch Freeze

This branch is now in PR review mode. Do not add new validation features, solver behavior, artifact semantics, or parity logic unless CI or reviewer feedback requires a scoped fix.

Do not keep updating repository files just to chase the latest final commit hash. The PR description should state the current PR head. GitHub Actions evidence should be added to the PR description or a PR comment unless a reviewer explicitly asks for a committed checklist update.

## PR Body Draft

```text
Current PR head: 97de386279dcaa9e00693b8344d082a21a0114f9
Branch: solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25

Scope:
- ANSYS vertical-flap validation artifacts and fail-closed Fluent reference gate
- runtime pressure-pair audit and generic solver boundary
- no Fluent parity claim

Claim boundary:
- fluent_parity_claimed=false
- real Fluent contract remains fluent_reference_incomplete
- no_fluent_parity_claim remains active
- public ANSYS tutorial evidence is metadata_only_not_parity_truth
- synthetic fixtures are temp-only and do not enter real artifact roots
- EasyFsi and HIBM-MPM outputs are not promoted as real Fluent truth

Key artifacts:
- validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/
- validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics/
- validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/
- validation_runs/ansys_vertical_flap_fsi/policy_reports/

Local evidence:
- py_compile passed
- collection/parity artifacts regenerated
- policy checker passed
- hygiene checker passed
- focused suite: 80 tests OK
- final guard subset: 17 tests OK
- git diff/check passed
- secret scan docs/rules only, no credentials

Remote CI:
- BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK
- connector returned no workflow run and no combined status
```

## Review Guide

- `docs/refactoring/BRANCH_REVIEW_MAP_2026-06-29.md`
- `docs/refactoring/ANSYS_VERTICAL_FLAP_BRANCH_MERGE_CHECKLIST_2026-06-29.md`
- `docs/refactoring/ANSYS_VERTICAL_FLAP_PR_SPLIT_STRATEGY_2026-06-29.md`
- `docs/validation/ANSYS_VERTICAL_FLAP_REAL_FLUENT_IMPORT_GATE_2026-06-29.md`

Review order for a single PR:

1. Claim boundary docs and merge checklist.
2. Fluent reference schema and source export validators.
3. Artifact generator manifests and checksums.
4. Policy/hygiene checkers and reports.
5. Synthetic pipeline tests.
6. Generic solver and runtime pressure-pair contracts.
7. Workflow coverage and final diff scan.

If reviewers ask to split the PR, use `docs/refactoring/ANSYS_VERTICAL_FLAP_PR_SPLIT_STRATEGY_2026-06-29.md`.

## Remote CI Handling

Remote CI remains the only external blocker.

1. Open the GitHub Actions page for `lizhuoh9/EasyFsi`.
2. Find the branch or PR run for `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.
3. If no run exists, use `workflow_dispatch` or create a PR to `main` to trigger the `pull_request` workflow.
4. Wait for the workflow to complete.
5. Record the run URL and status in the PR description or PR comment.
6. Do not replace `BLOCKED_PENDING_MANUAL_GITHUB_ACTIONS_CHECK` with a green status unless a real run URL/status exists.

Suggested PR comment when CI passes:

```text
Remote CI evidence:
- GitHub Actions run: <url>
- status: passed
```

## CI Failure Response

- py_compile/import failure: fix only the failing import or workflow path.
- artifact checksum failure: regenerate only the affected artifact root and `CHECKSUMS.sha256`.
- policy checker failure: verify no `synthetic-test-only` content entered real artifact roots and no parity claim was made with incomplete references.
- hygiene failure: check for local absolute paths, checksum mismatch, or secret-like text in generated artifacts.
- CRLF/whitespace failure: fix formatting only.
- synthetic fixture leakage: keep synthetic dry-runs temp-only and outside real artifact roots.

Do not change solver behavior or claim boundaries as part of a CI fix unless the failing test directly proves that boundary is wrong.

## Post-Merge Next Line

After this PR is reviewed and merged, start a new branch for:

`ANSYS_VERTICAL_FLAP_VALIDATION_TOOLS_PACKAGE_GOAL_2026-06-30`

That next line should migrate reusable validators/checkers from `validation_runs/.../scripts` into `tools/validation/ansys_vertical_flap/` while keeping the current script paths as thin wrappers, preserving artifacts, and keeping Fluent parity fail-closed.
