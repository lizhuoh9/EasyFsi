# ANSYS Vertical-Flap Real Fluent Atomic Import Goal - 2026-07-01

## Source

This goal is derived from the attached remote-branch review for:

- Repository: `lizhuoh9/EasyFsi`
- Branch: `codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`
- Reviewed remote HEAD: `972a55905dd50dc489f1aaa21eb950baf5d90873`
- Reviewed commit subject: `fix: reject ambiguous Fluent importer validation flag`
- Attachment path supplied by the user:
  `C:\Users\lizhu\.codex\attachments\f4b2be75-4f90-434d-93e1-cea4336fe83c\pasted-text.txt`

## Current Accepted State

The branch has already completed the previous importer CLI hardening stage:

1. CLI default mode is no-copy preflight.
2. CLI copy/import requires explicit `--commit-import`.
3. `--run-collection-validator` without `--commit-import` fails fast with JSON stderr.
4. CLI JSON contracts are covered by tests for preflight success, commit-import success, and failure.
5. CI workflow compiles `import_real_fluent_source_exports.py`.
6. CI workflow explicitly runs the real Fluent importer CLI integration test with the adjacent source-export artifact test.
7. Committed `source_exports` remain schema-only / provenance-missing placeholders, not fake real Fluent data.
8. `fluent_parity_claimed` remains false.
9. No remote GitHub Actions green result is available from the attachment, so remote CI must not be claimed passed.

## Problems To Fix In This Stage

### Problem 1: Destination Import Is Not Atomic

`import_real_fluent_source_exports(...)` currently copies required files directly into the requested destination directory. If a future real Fluent promotion writes the official committed `source_exports` destination and an exception happens during copy or post-copy validation, the destination can be left half-updated.

This is acceptable for temporary smoke destinations, but it is too risky for the future provenance-backed real Fluent source-export promotion path.

### Problem 2: Official `SOURCE_EXPORTS_ROOT` Can Be Written Without Destination Collection Validation

The CLI currently allows:

```powershell
--commit-import
```

without:

```powershell
--run-collection-validator
```

For a temporary destination this can remain useful. For the official committed `SOURCE_EXPORTS_ROOT`, it is too weak: writing the formal source-export directory must require the collection validator.

## Required Behavior

### Atomic Import Behavior

When `import_real_fluent_source_exports(...)` copies a validated bundle to any destination, it must:

1. Keep the existing destination unchanged until the replacement payload has been fully copied into a sibling staging directory.
2. Ensure the public evidence map exists in the staging destination.
3. If `run_collection_validator=True`, run a staged collection validator against the staging destination before replacing the live destination.
4. Replace the live destination only after staged copy and staged validation succeed.
5. If the destination already exists, keep a backup long enough to restore it if a later destination validation failure occurs.
6. If the destination does not already exist and a later failure occurs, remove the newly installed destination.
7. If final destination validation fails after replacement, restore the previous destination state before raising the failure.
8. Return `copied_files` as final destination paths, not temporary staging paths.
9. Preserve the existing success summary and ready-gate semantics.
10. Preserve `fluent_parity_claimed: false`.

### Official Source-Exports CLI Guard

When CLI `--commit-import` targets the default official `SOURCE_EXPORTS_ROOT`, the command must require `--run-collection-validator`.

This command must fail fast:

```powershell
python validation_runs\ansys_vertical_flap_fsi\scripts\import_real_fluent_source_exports.py `
  --input-dir <candidate-real-fluent-bundle> `
  --commit-import
```

The failure JSON must:

1. Be printed to stderr.
2. Return a nonzero exit code.
3. Include `mode: "commit_import"`.
4. Include `ready: false`.
5. Include blocker `source_exports_commit_requires_collection_validator`.
6. Include `copied_files: []`.
7. Include `copied_file_count: 0`.
8. Include `destination_dir`.
9. Not validate the input bundle as a success path.
10. Not create or modify the official `SOURCE_EXPORTS_ROOT`.

The CLI must still allow `--commit-import` without `--run-collection-validator` when the operator provides an explicit temporary destination different from the official `SOURCE_EXPORTS_ROOT`.

## Non-Goals

Do not:

1. Commit real Fluent source-export data rows.
2. Replace the committed schema-only `source_exports` directory with synthetic or fake real data.
3. Regenerate Fluent validation diagnostics, active manifests, or checksums.
4. Modify solver, FSI, pressure, velocity, displacement, force, or parity logic.
5. Claim Fluent parity.
6. Set `fluent_parity_claimed=true`.
7. Claim GitHub Actions green unless a live run is visible.
8. Change the no-copy preflight default.
9. Remove the existing `--run-collection-validator` without `--commit-import` fail-fast guard.

## TDD Plan

1. Add failing integration tests before production-code changes:
   - `test_commit_import_failure_keeps_existing_destination_unchanged`
   - `test_commit_import_replaces_destination_only_after_staged_ready`
   - `test_commit_import_atomic_replace_drops_stale_files`
   - `test_commit_import_preserves_public_evidence_map_after_atomic_replace`
   - `test_cli_rejects_default_source_exports_commit_without_collection_validator`
   - `test_cli_allows_temp_destination_commit_without_collection_validator`
2. Run the focused importer integration test and confirm RED is caused by the direct-copy behavior and missing official-destination guard.
3. Implement the smallest atomic import layer:
   - copy required files to a sibling staging directory;
   - validate staged source exports when requested;
   - move existing destination to backup;
   - move staging into final destination;
   - run final destination validator when requested;
   - restore backup or remove new destination on failure.
4. Add the official `SOURCE_EXPORTS_ROOT` CLI guard.
5. Re-run the focused importer integration test and confirm GREEN.
6. Run the adjacent real Fluent source-export artifact test.
7. Run the Fluent reference/schema/policy/parity comparison logic test group from the previous handoff.
8. Run `py_compile` for the importer and updated integration test.
9. Run `git diff --check`.

## Acceptance Criteria

This goal is complete only when:

1. This detailed goal file exists in the repository and the active Codex goal references it.
2. Destination import no longer writes directly into the live destination before staging succeeds.
3. Existing destination content is preserved when staged copy, staged validation, or final destination validation fails.
4. Successful import replaces stale destination files with the staged complete payload.
5. Returned `copied_files` point to final destination paths.
6. Public evidence map is present after atomic replacement.
7. CLI refuses default official `SOURCE_EXPORTS_ROOT` commit-import without `--run-collection-validator`.
8. CLI still allows temporary destination commit-import without `--run-collection-validator`.
9. Focused RED was observed before implementation.
10. Focused GREEN and adjacent verification pass after implementation.
11. No solver/parity/source-export data files are changed.
12. The verified change is committed and pushed to the remote branch.

