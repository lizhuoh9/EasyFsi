# ANSYS Vertical-Flap Real Fluent Import CI And CLI Flag Goal - 2026-07-01

## Source

This goal is derived from the attached remote review for:

- Repository: `lizhuoh9/EasyFsi`
- Branch: `codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`
- Reviewed remote HEAD: `3479e962cada0bc1cd25297a9f265017192cde5d`
- Reviewed commit subject: `fix: require explicit Fluent source import commit`
- Attachment path supplied by the user:
  `C:\Users\lizhu\.codex\attachments\1a349fb9-8e99-4ba3-9a80-9036a38385a1\pasted-text.txt`

## Current Accepted State

The previous change correctly made the real Fluent source-export importer safer:

1. CLI default mode is no-copy preflight.
2. CLI copy/import requires explicit `--commit-import`.
3. CLI preflight success reports `mode: "preflight"`, `copied_files: []`, and `copied_file_count: 0`.
4. CLI commit import success reports `mode: "commit_import"`.
5. Schema-only committed `source_exports` still fail closed.
6. The committed source-export CSVs and metadata were not replaced by fake real-Fluent data.
7. `fluent_parity_claimed` remains false.
8. No GitHub Actions success was visible in the attachment, so remote CI must not be claimed green until a real run is visible.

## Problems To Fix In This Follow-Up

### Problem 1: CI Does Not Cover The New Importer CLI Test

The Windows ANSYS vertical-flap validation workflow runs many Fluent reference/export and parity-boundary tests, but it does not yet explicitly run:

```text
tests.integration.test_ansys_vertical_flap_real_fluent_source_export_import
```

That means a remote workflow run cannot prove the CLI mode split. The workflow should compile the importer script and run the importer CLI integration test alongside the adjacent real Fluent source-export artifact test.

### Problem 2: `--run-collection-validator` Is Silently Ignored Without `--commit-import`

The current CLI dispatch only passes `run_collection_validator` into `import_real_fluent_source_exports(...)` inside the `--commit-import` branch. In default preflight mode, `--run-collection-validator` is accepted but ignored.

This is easy for an operator to misread as "the destination collection validator was run." The CLI should fail fast for this flag combination.

## Required Behavior

### CI Coverage

The workflow `.github/workflows/ansys-vertical-flap-validation.yml` must:

1. Include `validation_runs\ansys_vertical_flap_fsi\scripts\import_real_fluent_source_exports.py` in the ANSYS validation `py_compile` list.
2. Add a focused workflow step that runs:

   ```powershell
   python -m unittest `
     tests.integration.test_ansys_vertical_flap_real_fluent_source_export_import `
     tests.integration.test_ansys_vertical_flap_real_fluent_source_export_artifacts `
     -v
   ```

3. Place the new step near the Fluent source-export/reference workflow steps so the CI evidence is easy to audit.
4. Not claim GitHub Actions green unless a live remote run is actually visible later.

### CLI Fail-Fast Flag Semantics

Running:

```powershell
python validation_runs\ansys_vertical_flap_fsi\scripts\import_real_fluent_source_exports.py `
  --input-dir <bundle> `
  --run-collection-validator
```

without `--commit-import` must:

1. Return a nonzero exit code.
2. Print machine-readable JSON to stderr.
3. Include `mode: "preflight"`.
4. Include `ready: false`.
5. Include blocker `collection_validator_requires_commit_import`.
6. Include `copied_files: []`.
7. Include `copied_file_count: 0`.
8. Include the resolved `destination_dir`.
9. Not validate the input bundle as a success path.
10. Not create or modify the destination directory.
11. Not regenerate diagnostics or write an active manifest.

### JSON Contract Stability

Add focused assertions so downstream scripts can rely on the CLI JSON shape.

Default preflight success must include at least:

```text
mode
ready
input_dir
destination_dir
required_files
blockers
source_checks
metadata_check
real_fluent_import_gate
copied_files
copied_file_count
collection
```

Commit-import success must include at least:

```text
mode
ready
input_dir
destination_dir
copied_files
copied_file_count
preflight
collection
```

CLI failure JSON must include at least:

```text
mode
ready
blockers
copied_file_count
copied_files
destination_dir
```

## Non-Goals

Do not:

1. Commit real Fluent source-export CSV rows.
2. Replace the committed schema-only `source_exports`.
3. Generate or regenerate Fluent validation diagnostics.
4. Change solver, FSI, pressure, velocity, displacement, force, or parity logic.
5. Set `fluent_parity_claimed=true`.
6. Weaken provenance, missing-file, schema-only, or staged collection checks.
7. Implement atomic destination replacement in this follow-up; keep that as a later, separately reviewed import hardening task.
8. Claim full local test-suite success or remote CI success unless directly verified.

## TDD Plan

1. Add a RED CLI integration test named `test_cli_rejects_collection_validator_without_commit_import`.
2. Add JSON key contract assertions to the existing CLI success/failure tests.
3. Run the focused importer integration test and confirm RED comes from the intended silently ignored flag combination.
4. Implement the smallest CLI dispatch change:
   - after parsing args, if `args.run_collection_validator and not args.commit_import`, print JSON failure to stderr and return `1`;
   - do not call `validate_import_bundle(...)` for that invalid flag combination.
5. Re-run the focused importer integration test and confirm GREEN.
6. Run the adjacent real Fluent source-export artifact test with the importer test.
7. Run `py_compile` for the importer and workflow-referenced scripts touched by this change.
8. Run `git diff --check`.

## Acceptance Criteria

This goal is complete only when:

1. This detailed goal file exists in the repository and the active Codex goal references it.
2. CI workflow explicitly compiles `import_real_fluent_source_exports.py`.
3. CI workflow explicitly runs `test_ansys_vertical_flap_real_fluent_source_export_import` with the adjacent source-export artifact test.
4. `--run-collection-validator` without `--commit-import` fails fast with JSON stderr and does not copy or create the destination.
5. CLI JSON contract assertions are present for preflight success, commit-import success, and failure.
6. Focused RED was observed before production code changes.
7. Focused GREEN and adjacent integration tests pass after implementation.
8. No solver/parity/source-export data files are changed.
9. The verified change is committed and pushed to the remote branch.

