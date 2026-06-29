# ANSYS Vertical-Flap Validation Tools Package Goal 2026-06-30

## Short Goal Reference

Use this file as the detailed goal for the implementation turn:

`docs/refactoring/ANSYS_VERTICAL_FLAP_VALIDATION_TOOLS_PACKAGE_GOAL_2026-06-30.md`

Short `/goal` text:

```text
Implement docs/refactoring/ANSYS_VERTICAL_FLAP_VALIDATION_TOOLS_PACKAGE_GOAL_2026-06-30.md on a new follow-up branch: migrate reusable ANSYS vertical-flap validators/checkers from validation_runs/.../scripts into tools/validation/ansys_vertical_flap/ with thin script wrappers, unchanged artifacts, unchanged solver behavior, and fail-closed Fluent parity.
```

## Objective

Migrate reusable ANSYS vertical-flap Fluent reference validation logic out of
`validation_runs/ansys_vertical_flap_fsi/scripts/` into a stable importable
package at `tools/validation/ansys_vertical_flap/`, while preserving every
existing command path as a thin wrapper.

The migration must make validator/checker code reusable by tests, review tools,
and future validation workflows without changing generated artifact semantics,
solver behavior, Fluent reference truth boundaries, or parity-claim policy.

## Source Context

This goal is the post-merge follow-up named by
`docs/refactoring/ANSYS_VERTICAL_FLAP_PR_HANDOFF_2026-06-29.md`.

The previous branch was frozen for PR review. This follow-up must be implemented
on a new branch and must not append feature commits to
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.

The prior handoff's active claim boundary remains binding:

- `fluent_parity_claimed=false`.
- The real Fluent contract remains `fluent_reference_incomplete`.
- `no_fluent_parity_claim` remains active.
- Public ANSYS tutorial evidence remains `metadata_only_not_parity_truth`.
- Synthetic fixtures are temp-only test fixtures.
- EasyFsi and HIBM-MPM outputs are not real Fluent truth.

## Target Package

Create or extend this package:

`tools/validation/ansys_vertical_flap/`

Target reusable modules:

- `fluent_reference_contract_schema.py`
- `fluent_source_export_schema.py`
- `fluent_reference_collection.py`
- `fluent_parity.py`
- `fluent_artifact_policy.py`
- `validation_artifact_hygiene.py`
- `policy_report_writer.py`

The package should expose reusable functions with explicit inputs and returned
structured results. CLI parsing, printing, process exit codes, and filesystem
path defaults belong in wrapper scripts, not in the package business logic.

## Wrapper Commands To Preserve

Keep existing script entry points under
`validation_runs/ansys_vertical_flap_fsi/scripts/`.

The highest-priority wrappers are:

- `fluent_reference_contract_schema.py`
- `fluent_source_export_schema.py`
- `run_fluent_reference_collection_validation.py`
- `run_traction_selected_formulation_fluent_parity.py`
- `check_fluent_artifact_policy.py`
- `check_validation_artifact_hygiene.py`
- `build_synthetic_fluent_reference_fixture.py`

Wrapper scripts may:

- Parse CLI arguments.
- Resolve repository-relative paths.
- Call package functions.
- Print human-readable summaries.
- Preserve the existing process exit behavior.

Wrapper scripts must not:

- Contain Fluent truth or parity-claim business logic.
- Change generated artifact content unless the existing command already does.
- Hide missing reference data.
- Promote synthetic data into real artifact roots.
- Change solver, FSI, pressure, traction, or runtime behavior.

## Implementation Phases

### Phase 1: Inventory And Tests

Identify reusable logic currently embedded in the validation scripts and add
focused tests before moving it.

The first red tests should prove:

- The new package import path exists.
- The target wrapper commands delegate to package functions.
- The public CLI behavior stays stable for policy/hygiene/report commands.
- Existing fail-closed markers remain unchanged when Fluent references are
  incomplete.

### Phase 2: Extract Pure Validation Logic

Move reusable validation logic into `tools/validation/ansys_vertical_flap/`.

Use small modules with cohesive responsibilities:

- Contract schema validation.
- Fluent source export schema validation.
- Reference collection matrix validation.
- Parity matrix validation and fail-closed result construction.
- Artifact policy scanning.
- Artifact hygiene scanning.
- Deterministic report writing.

Keep data transformations explicit. Prefer dictionaries/dataclasses and
structured return values over ad hoc string parsing where possible.

### Phase 3: Thin Wrappers

Replace moved script internals with wrappers that call the package modules.

Preserve command names, arguments, defaults, output paths, report filenames,
and nonzero exit behavior unless a test proves the old behavior was wrong.

### Phase 4: Artifact And Claim Boundary Verification

Verify that artifacts remain unchanged or intentionally regenerated only where
the goal requires it. This migration should not require artifact rewrites.

If a checksum changes unexpectedly, stop and diagnose whether the wrapper changed
semantics. Do not paper over semantic drift by regenerating artifacts.

### Phase 5: Final Review And Push

After implementation and verification:

- Run focused tests covering the migrated package and wrappers.
- Run the branch review docs test from the prior handoff if still relevant.
- Run `git diff --check`.
- Review the diff for unintended solver/artifact/parity semantics changes.
- Commit and push the new branch only after the checks pass.

## Constraints

- Do not edit solver physics or core FSI behavior.
- Do not change artifact semantics during the migration.
- Do not move generated artifacts.
- Do not use this migration to claim Fluent parity.
- Keep Fluent parity fail-closed.
- Keep `check_fluent_artifact_policy.py` available as a wrapper command.
- Keep `check_validation_artifact_hygiene.py` available as a wrapper command.
- Preserve `--write-report` behavior for policy reports.
- Preserve deterministic report output.
- Preserve existing workflow paths.
- Keep synthetic dry-run helpers test-only.
- Do not make synthetic fixtures importable as real Fluent evidence providers.
- Do not add new top-level docs unless an existing docs location is unsuitable.
- Do not update the frozen PR handoff merely to chase a new commit hash.

## Acceptance Criteria

The implementation is complete only when all of these are true:

- `tools/validation/ansys_vertical_flap/` contains reusable validation modules.
- Existing CLI commands under `validation_runs/ansys_vertical_flap_fsi/scripts/`
  still work.
- Wrapper scripts contain no business logic beyond argument parsing, path
  resolution, delegation, printing, and exit handling.
- The reusable package has direct unit coverage.
- Policy and hygiene report generation remains deterministic and test-covered.
- Existing fail-closed Fluent reference and parity claim boundaries are
  preserved.
- No solver behavior changed.
- No generated artifact roots were moved.
- No real Fluent parity claim is introduced.
- Focused tests pass.
- `git diff --check` passes.
- The final commit and pushed branch are reported with exact hashes.

## Suggested Verification Commands

Use the available local Python interpreter first. If the default `python` is not
the project interpreter, use the known working interpreter from this environment.

```powershell
python -m py_compile validation_runs\ansys_vertical_flap_fsi\scripts\fluent_reference_contract_schema.py
python -m py_compile validation_runs\ansys_vertical_flap_fsi\scripts\fluent_source_export_schema.py
python -m py_compile validation_runs\ansys_vertical_flap_fsi\scripts\check_fluent_artifact_policy.py
python -m py_compile validation_runs\ansys_vertical_flap_fsi\scripts\check_validation_artifact_hygiene.py
python -m unittest -v tests.integration.test_ansys_vertical_flap_fluent_reference_contract_schema
python -m unittest -v tests.integration.test_ansys_vertical_flap_fluent_source_export_schema
python -m unittest -v tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts
python -m unittest -v tests.integration.test_ansys_vertical_flap_fluent_artifact_policy
python -m unittest -v tests.integration.test_ansys_vertical_flap_policy_reports
git diff --check
```

Broaden the test set if extraction touches shared helpers or wrapper behavior
used by additional ANSYS vertical-flap validation commands.
