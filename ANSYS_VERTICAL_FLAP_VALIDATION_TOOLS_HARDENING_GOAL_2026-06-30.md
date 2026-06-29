# ANSYS Vertical Flap Validation Tools Hardening Goal - 2026-06-30

## Source Context

- Repository: `lizhuoh9/EasyFsi`
- Working directory:
  `D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM\refactored`
- Active branch:
  `codex/ansys-vertical-flap-validation-tools-package-2026-06-30`
- Attachment-reviewed HEAD:
  `67ee6af586c1e0c57d939454a1e9df8d5e01c6b0`
- Attachment-reviewed commit subject:
  `refactor: package ansys vertical flap validation tools`
- Source review artifact:
  `C:\Users\lizhu\.codex\attachments\d26ab26a-b826-49e2-9553-48ff36a77935\pasted-text.txt`

The source review concluded that the package-migration direction is correct:
reusable ANSYS vertical-flap validation logic has been moved from
`validation_runs/.../scripts/` into
`tools/validation/ansys_vertical_flap/`, while the old command paths remain as
compatibility wrappers. The review also identified two hardening gaps before
the branch should be pushed as the final follow-up:

1. the new package-migration guard test is not explicitly run by the workflow;
2. package modules still contain some CLI entrypoint code, which is a stricter
   architecture concern but must only be changed if behavior and artifacts stay
   unchanged.

Remote CI evidence is currently absent for this branch. Do not claim GitHub
Actions green unless an actual workflow run exists for the pushed SHA.

## Objective

Harden the ANSYS vertical-flap validation-tools package migration without
changing solver behavior, generated artifact semantics, Fluent reference truth
boundaries, or Fluent parity claim policy.

The minimum required code change is to make the workflow explicitly protect the
new migration guard:

```text
tests.tools.test_ansys_vertical_flap_validation_package
```

Then evaluate the wrapper-purity concern. If the CLI cleanup can be made as a
small behavior-preserving refactor with focused tests, move parsing, printing,
and process-exit behavior out of package modules and into wrapper scripts. If
that refactor expands beyond the safe migration scope, leave package CLI cleanup
as a documented follow-up instead of forcing it into this branch.

The final pushed branch must remain a code-organization hardening branch, not a
physics, runtime, artifact-regeneration, or parity-validation branch.

## Required Scope

### Phase 1: Add CI Coverage For The Migration Guard

Modify:

```text
.github/workflows/ansys-vertical-flap-validation.yml
```

Required workflow changes:

- add `tests\tools\test_ansys_vertical_flap_validation_package.py` to the
  workflow `py_compile` coverage;
- add a dedicated Windows PowerShell step near the Fluent policy/hygiene tests:

```powershell
python -m unittest `
  tests.tools.test_ansys_vertical_flap_validation_package `
  -v
```

The step name should make the migration boundary explicit, for example:

```text
Run ANSYS vertical-flap validation package migration test
```

This phase is mandatory.

### Phase 2: Wrapper-Purity Audit And Bounded Cleanup

Audit these package modules:

```text
tools/validation/ansys_vertical_flap/fluent_artifact_policy.py
tools/validation/ansys_vertical_flap/validation_artifact_hygiene.py
tools/validation/ansys_vertical_flap/fluent_reference_collection.py
tools/validation/ansys_vertical_flap/fluent_parity.py
```

The target architecture is:

- package modules expose reusable functions with explicit inputs and structured
  results;
- wrapper scripts own CLI parsing, printing, filesystem defaults, and exit
  codes;
- legacy command paths continue to work;
- no package refactor is allowed to change policy outcomes, report JSON
  schema, blocker strings, artifact paths, or parity-claim behavior.

If implemented in this branch, prioritize the policy and hygiene CLI wrappers:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/check_fluent_artifact_policy.py
validation_runs/ansys_vertical_flap_fsi/scripts/check_validation_artifact_hygiene.py
scripts/check_validation_artifact_hygiene.py
```

For `fluent_reference_collection.py` and `fluent_parity.py`, keep the cleanup
out of this branch unless it is clearly small and behavior-preserving. They may
remain as a follow-up if moving their `main()` functions would risk touching
artifact or parity semantics.

### Phase 3: Tests For Migration And Wrapper Boundaries

Update or extend:

```text
tests/tools/test_ansys_vertical_flap_validation_package.py
```

Required coverage:

- package imports still work from `tools.validation.ansys_vertical_flap`;
- legacy wrapper imports and command paths still delegate to package code;
- wrappers do not redefine core business functions;
- if CLI parsing is moved out of package modules, the package modules no longer
  contain `argparse.ArgumentParser`;
- wrappers that own CLI options contain the parser and preserve exit behavior;
- policy/hygiene wrapper smoke calls still return success for the existing
  pass-state fixture or current committed reports.

Do not weaken existing tests to make the migration pass. Fix the implementation
or the workflow wiring instead.

### Phase 4: Artifact And Semantics Protection

This goal must not regenerate or accept changes under generated validation
artifact roots. After implementation, explicitly confirm that these diffs are
empty:

```powershell
git diff --name-only validation_runs/ansys_vertical_flap_fsi/fluent_reference/validation_diagnostics
git diff --name-only validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics
```

If either command prints artifact files, stop and investigate. Do not accept
artifact churn as part of this goal.

## Explicit Non-Goals

- Do not modify solver physics.
- Do not modify `simulation_core/` for this hardening task.
- Do not change ANSYS vertical-flap case constants, material parameters,
  tolerances, damping, support radii, source profiles, or runner behavior.
- Do not regenerate, overwrite, or backfill generated validation artifacts.
- Do not change Fluent reference contract semantics.
- Do not change Fluent parity blocker strings:
  `fluent_reference_incomplete` and `no_fluent_parity_claim`.
- Do not claim Fluent parity.
- Do not claim GitHub Actions green unless a real workflow run exists.
- Do not create synthetic dry-run helpers outside test-only scope.

## Required Local Validation

Run syntax checks:

```powershell
python -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\fluent_reference_contract_schema.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\fluent_source_export_schema.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\check_fluent_artifact_policy.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\check_validation_artifact_hygiene.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_fluent_reference_collection_validation.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py `
  scripts\check_validation_artifact_hygiene.py `
  tools\validation\ansys_vertical_flap\fluent_reference_contract_schema.py `
  tools\validation\ansys_vertical_flap\fluent_source_export_schema.py `
  tools\validation\ansys_vertical_flap\fluent_reference_collection.py `
  tools\validation\ansys_vertical_flap\fluent_parity.py `
  tools\validation\ansys_vertical_flap\fluent_artifact_policy.py `
  tools\validation\ansys_vertical_flap\validation_artifact_hygiene.py `
  tools\validation\ansys_vertical_flap\policy_report_writer.py `
  tests\tools\test_ansys_vertical_flap_validation_package.py
```

Run the focused unit/integration suite:

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

Run the policy and hygiene CLI wrappers:

```powershell
python validation_runs\ansys_vertical_flap_fsi\scripts\check_fluent_artifact_policy.py
python scripts\check_validation_artifact_hygiene.py
```

Run diff hygiene:

```powershell
git diff --check
```

If the default `python` is not the intended interpreter on this Windows machine,
use the validated local interpreter path and report that substitution in the
final summary.

## Push Contract

The user has approved pushing after the implementation is complete and the
required local validation has passed. Before pushing:

1. inspect `git status --short --branch`;
2. inspect `git remote -v`;
3. stage only relevant goal, workflow, wrapper, and test changes;
4. commit with a conventional commit message;
5. push the current branch to its configured remote;
6. report the final commit hash and remote branch.

Do not create or retarget a PR unless separately requested.

## PR Notes To Preserve If A PR Is Later Opened

Recommended base and head from the source review:

```text
base: solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25
head: codex/ansys-vertical-flap-validation-tools-package-2026-06-30
```

Recommended title:

```text
Refactor: package ANSYS vertical-flap validation tools
```

Recommended body points:

```text
Scope:
- Move reusable ANSYS vertical-flap validation helpers into tools/validation/ansys_vertical_flap/
- Preserve validation_runs/.../scripts command paths as thin wrappers
- Preserve policy/hygiene CLI wrappers
- No solver behavior changes
- No generated artifact changes
- Fluent parity remains fail-closed

Known note:
- Remote CI evidence pending until GitHub Actions run is available.
```

## Done Criteria

- This detailed goal file is committed.
- A short Codex goal references this file.
- Workflow explicitly compiles and runs
  `tests.tools.test_ansys_vertical_flap_validation_package`.
- Any wrapper-purity cleanup performed in this branch is behavior-preserving and
  covered by tests.
- No generated artifact roots change.
- Required focused validation passes locally.
- `git diff --check` passes.
- The current branch is committed and pushed to `origin`.
- Final report includes commit hash, branch name, validation results, and an
  explicit note that no GitHub Actions green claim is made without run evidence.
