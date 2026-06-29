# ANSYS Vertical Flap Validation Tools Stacked PR CI Goal - 2026-06-30

## Source Context

- Repository: `lizhuoh9/EasyFsi`
- Working directory:
  `D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM\refactored`
- Active branch:
  `codex/ansys-vertical-flap-validation-tools-package-2026-06-30`
- Current remote HEAD reviewed in the source note:
  `8ea98746f48bc8c4a35907552e9b8f1c3aeb28ea`
- Current remote commit subject:
  `test: harden ansys validation tools migration`
- Stacked PR base recommended by the source note:
  `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Source review artifact:
  `C:\Users\lizhu\.codex\attachments\3d942753-6294-40e4-8460-824b336d8735\pasted-text.txt`

The source review concluded that the validation-tools package migration is
already code-ready:

- reusable package exists under `tools/validation/ansys_vertical_flap/`;
- legacy command paths remain as wrappers;
- wrappers now own CLI parsing, printing, and exit codes;
- package modules now own reusable business logic;
- migration guard test is in the workflow;
- generated artifact roots did not change;
- Fluent parity remains fail-closed;
- no GitHub Actions green claim exists because no run URL/status is available.

The next required hardening is PR/CI readiness for the recommended stacked PR
base. The source note says to open the PR as:

```text
base: solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25
head: codex/ansys-vertical-flap-validation-tools-package-2026-06-30
```

The current workflow already runs on pushes to `codex/**` and `solver/**`, but
its `pull_request` branch filter only targets `main`. A stacked PR with a
`solver/**` base should be covered explicitly by the workflow so the PR surface
matches the source note's review plan.

## Objective

Make the smallest PR/CI workflow change needed to support the recommended
stacked PR path for the already-complete ANSYS vertical-flap validation-tools
package migration.

The expected implementation is limited to:

```text
.github/workflows/ansys-vertical-flap-validation.yml
```

and should add the frozen validation branch pattern to the `pull_request`
branch filter, so pull requests targeting `solver/**` are eligible for the same
ANSYS vertical-flap validation contract workflow.

This is a CI routing hardening task, not a validation-tools feature task.

## Required Change

Modify the workflow trigger:

```yaml
on:
  pull_request:
    branches:
      - main
```

so it also includes:

```yaml
      - solver/**
```

The push trigger already includes:

```yaml
  push:
    branches:
      - main
      - solver/**
      - codex/**
```

Do not remove or weaken any existing push coverage, workflow steps, tests,
policy scans, artifact scans, or Fluent parity overclaim guards.

## Explicit Non-Goals

- Do not modify solver physics.
- Do not modify `simulation_core/`.
- Do not modify package implementation under
  `tools/validation/ansys_vertical_flap/` unless a local verification failure
  proves the current pushed code is broken.
- Do not modify wrapper behavior under
  `validation_runs/ansys_vertical_flap_fsi/scripts/` unless a local verification
  failure proves the current pushed code is broken.
- Do not regenerate generated validation artifacts.
- Do not change Fluent reference contract semantics.
- Do not change Fluent parity blocker strings or claim policy.
- Do not add new validation features, new artifact semantics, or new PR-scope
  documentation beyond this user-requested goal file.
- Do not create or retarget a PR unless separately requested.
- Do not claim GitHub Actions green without a real workflow run URL/status.

## Required Validation

Because this change touches CI routing only, run a focused local validation
slice that proves the existing package migration remains intact:

```powershell
python -m py_compile `
  .github\workflows\ansys-vertical-flap-validation.yml `
  tests\tools\test_ansys_vertical_flap_validation_package.py
```

YAML is not a Python file, so if `py_compile` is not applicable to the workflow,
validate it by careful diff inspection instead and run the Python syntax check
for the migration test:

```powershell
python -m py_compile tests\tools\test_ansys_vertical_flap_validation_package.py
```

Run the direct migration guard:

```powershell
python -m unittest -v tests.tools.test_ansys_vertical_flap_validation_package
```

Run the package migration focused suite if time allows:

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

Run whitespace/diff hygiene:

```powershell
git diff --check
```

CRLF warnings are acceptable only if there are no whitespace errors.

## Push Contract

The user approved pushing after the scoped change and verification are complete.
Before pushing:

1. inspect `git status --short --branch`;
2. inspect `git remote -v`;
3. stage only this goal file and the workflow change;
4. commit with a conventional commit message;
5. push the current branch to `origin`;
6. verify the remote ref points at the new commit;
7. report the final commit hash and remote branch.

## PR Notes To Preserve

When the user later asks to open the PR, use:

```text
base: solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25
head: codex/ansys-vertical-flap-validation-tools-package-2026-06-30
```

Recommended PR title:

```text
Refactor: package ANSYS vertical-flap validation tools
```

Recommended PR description points:

```text
Scope:
- Move reusable ANSYS vertical-flap validation helpers into tools/validation/ansys_vertical_flap/
- Preserve validation_runs/.../scripts command paths as thin wrappers
- Preserve root policy/hygiene wrapper path
- Move CLI main/argparse/print/exit ownership into wrappers
- Keep package modules reusable and side-effect-light
- No solver behavior changes
- No generated artifact changes
- Fluent parity remains fail-closed

Local evidence:
- py_compile passed
- focused suite: 57 tests OK
- check_fluent_artifact_policy.py returned status: passed
- check_validation_artifact_hygiene.py returned status: passed
- generated artifact roots have no diff
- git diff --check passed

Remote CI:
- pending GitHub Actions run URL / status
```

## Done Criteria

- This detailed goal file is committed.
- A short Codex goal references this file.
- `.github/workflows/ansys-vertical-flap-validation.yml` includes `solver/**`
  under `pull_request.branches`.
- No validation-tools package logic changes are made unless required by a real
  local failure.
- No generated artifact roots change.
- Focused migration validation passes locally.
- `git diff --check` has no errors.
- The current branch is pushed to `origin`.
- Final report includes commit hash, branch name, validation results, and an
  explicit note that GitHub Actions green is still not claimed without run
  evidence.
