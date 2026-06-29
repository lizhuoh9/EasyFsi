# ANSYS Vertical Flap Validation Tools CI Regression Test Goal - 2026-06-30

## Source Context

- Repository: `lizhuoh9/EasyFsi`
- Working directory:
  `D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM\refactored`
- Active branch:
  `codex/ansys-vertical-flap-validation-tools-package-2026-06-30`
- Current remote HEAD reviewed in the source note:
  `533065621ee91e1bc7b7c6991280fb9380b154f7`
- Current remote commit subject:
  `ci: enable ansys validation stacked pr checks`
- Source review artifact:
  `C:\Users\lizhu\.codex\attachments\f03572c1-bd3a-490f-8ad8-2309042730c4\pasted-text.txt`

The source review concluded that the validation-tools package migration branch
is PR-ready with pending remote CI. It confirmed:

- the reusable package exists under `tools/validation/ansys_vertical_flap/`;
- legacy command paths remain as wrappers;
- wrappers own CLI parsing, printing, and exit codes;
- package modules own reusable business logic;
- the package migration guard is compiled and run by the workflow;
- `pull_request.branches` now includes both `main` and `solver/**`;
- generated artifact roots did not change;
- Fluent parity remains fail-closed;
- no GitHub Actions green claim exists because no run URL/status is available.

The review also said not to keep expanding this branch unless CI fails or a
reviewer requests a scoped fix. This goal therefore adds only regression test
coverage for the CI/PR routing behavior that was just fixed. It must not expand
validation logic.

## Objective

Add a focused local regression test that protects the stacked PR CI routing and
package-migration workflow coverage described in the source review.

The test should prove:

- `.github/workflows/ansys-vertical-flap-validation.yml` keeps
  `pull_request.branches` coverage for `solver/**`, so the recommended stacked
  PR base is eligible for the workflow;
- the workflow still keeps push coverage for `main`, `solver/**`, and
  `codex/**`;
- the workflow still compiles
  `tests\tools\test_ansys_vertical_flap_validation_package.py`;
- the workflow still has a dedicated step running
  `tests.tools.test_ansys_vertical_flap_validation_package`.

If the workflow already satisfies these requirements, only add the regression
test. If the test exposes a gap, make the smallest workflow fix needed to pass
it.

## Required Scope

Modify only test/workflow surfaces that are directly related to the CI routing
contract:

```text
tests/tools/test_ansys_vertical_flap_validation_package.py
.github/workflows/ansys-vertical-flap-validation.yml
```

Expected implementation:

- add a workflow path constant in the existing migration-package test module;
- parse or inspect the workflow text without adding external dependencies;
- assert the relevant branch filters and migration-test workflow entries;
- keep the test deterministic and repository-local.

Do not add PyYAML or any new dependency just to parse this workflow. A small
text-based assertion is acceptable because the workflow shape is intentionally
simple and this is a regression guard.

## Explicit Non-Goals

- Do not modify solver physics.
- Do not modify `simulation_core/`.
- Do not modify validation-tools package implementation under
  `tools/validation/ansys_vertical_flap/`.
- Do not modify wrapper CLI behavior under
  `validation_runs/ansys_vertical_flap_fsi/scripts/` or `scripts/`.
- Do not regenerate generated validation artifacts.
- Do not change Fluent reference contract semantics.
- Do not change Fluent parity blocker strings or claim policy.
- Do not add new validation features.
- Do not create or retarget a PR unless separately requested.
- Do not claim GitHub Actions green without a real workflow run URL/status.

## Required Validation

Run syntax check for the changed test:

```powershell
python -m py_compile tests\tools\test_ansys_vertical_flap_validation_package.py
```

Run the direct regression test module:

```powershell
python -m unittest -v tests.tools.test_ansys_vertical_flap_validation_package
```

Run the focused validation-tools package suite:

```powershell
python -m unittest -v `
  tests.tools.test_ansys_vertical_flap_validation_package `
  tests.integration.test_ansys_vertical_flap_fluent_reference_contract_schema `
  tests.integration.test_ansys_vertical_flap_fluent_source_export_schema `
  tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts `
  tests.integration.test_ansys_vertical_flap_fluent_artifact_policy `
  tests.integration.test_ansys_vertical_flap_policy_reports `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic `
  tests.tools.test_validation_artifact_hygiene
```

Run artifact-diff guards:

```powershell
git diff --name-only validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics
git diff --name-only validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics
```

Expected output: no files.

Run diff hygiene:

```powershell
git diff --check
```

CRLF warnings are acceptable only if there are no whitespace errors.

## Push Contract

The user approved pushing after the scoped change and verification are complete.
Before pushing:

1. inspect `git status --short --branch`;
2. inspect `git remote -v`;
3. stage only this goal file and the direct test/workflow change;
4. commit with a conventional commit message;
5. push the current branch to `origin`;
6. verify the remote ref points at the new commit;
7. report the final commit hash and remote branch.

## PR Notes To Preserve

When a PR is later opened, use:

```text
base: solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25
head: codex/ansys-vertical-flap-validation-tools-package-2026-06-30
```

Recommended PR title:

```text
Refactor: package ANSYS vertical-flap validation tools
```

Keep the PR evidence honest:

```text
Remote CI:
- pending GitHub Actions run URL / status
```

Do not write `green` until a real run exists.

## Done Criteria

- This detailed goal file is committed.
- A short Codex goal references this file.
- Local tests protect `solver/**` pull-request workflow coverage.
- Local tests protect the migration package py_compile and unittest workflow
  entries.
- No package, solver, wrapper, artifact, or parity-semantics change is made
  unless a real test failure forces a narrowly scoped workflow/test fix.
- No generated artifact roots change.
- Focused validation passes locally.
- `git diff --check` has no errors.
- The current branch is pushed to `origin`.
- Final report includes commit hash, branch name, validation results, and an
  explicit note that GitHub Actions green is still not claimed without run
  evidence.
