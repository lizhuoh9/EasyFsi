# ANSYS Vertical-Flap Real Fluent Import Preflight CLI Goal - 2026-07-01

## Source

This goal is derived from the attached branch review for:

- Branch: `codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`
- Reviewed remote HEAD: `b8f3f99051de13cfc73d97530d3eeea340656d37`
- Reviewed commit subject: `validation: add real fluent import executor`
- Attachment path supplied by the user:
  `C:\Users\lizhu\.codex\attachments\59eb1173-d5b6-4f75-ad31-906eb3368b20\pasted-text.txt`

## Current Accepted State

The current branch already has the important implementation boundaries in place:

1. A real Fluent source-export import executor exists.
2. Import preflight rejects schema-only committed exports.
3. Import preflight rejects missing required CSV or metadata files before copy.
4. Import preflight rejects disallowed provenance before copy.
5. A complete temporary real-Fluent-like bundle can pass the collection readiness gate in tests.
6. Solver logic, parity diagnostics, and generated collection artifacts were not changed.
7. `real_fluent_import_gate.fluent_parity_claimed` must remain `false`.
8. GitHub Actions status was not available in the attachment (`workflow_runs: []`), so no remote CI success may be claimed from that review.

## Problem To Fix

The attachment's recommended next operator step is to run the importer as a preflight against a real Fluent bundle before any committed `source_exports` replacement. The documented command shape is centered on `--input-dir` first.

The current CLI implementation calls `import_real_fluent_source_exports(...)` unconditionally from `main(...)`. That means a successful CLI invocation with only `--input-dir` also copies the required files into the destination directory. This is too aggressive for the next-step workflow because it makes the first operator command a write/import operation instead of a pure preflight.

The fix is not another data guard and not another schema validator. The missing behavior is a safer CLI mode split:

- Default CLI mode: validate the input bundle only and emit JSON preflight status.
- Explicit import mode: copy validated files only when the operator passes an intentional commit/import flag.

## Non-Goals

Do not:

1. Create fake real Fluent CSVs in the committed `source_exports` directory.
2. Modify committed source-export CSV values or metadata as a substitute for a real Fluent run.
3. Regenerate collection diagnostics from synthetic data.
4. Change solver logic, FSI coupling logic, pressure/velocity/displacement formulas, or parity calculations.
5. Claim Fluent parity.
6. Set `fluent_parity_claimed=true`.
7. Claim GitHub Actions passed unless a live check actually proves it.
8. Remove existing provenance, schema, missing-file, or staged-collection checks.
9. Hide failures by weakening tests or accepting schema-only artifacts.

## Required Behavior

### Default CLI Preflight

Running:

```powershell
python validation_runs\ansys_vertical_flap_fsi\scripts\import_real_fluent_source_exports.py `
  --input-dir <candidate-real-fluent-bundle>
```

must:

1. Validate the candidate input bundle in a temporary staging area.
2. Return exit code `0` only when the candidate bundle is ready.
3. Print machine-readable JSON to stdout on success.
4. Include a mode marker such as `mode: "preflight"`.
5. Include `ready: true` for a complete bundle.
6. Include `copied_file_count: 0`.
7. Not create or modify the destination `source_exports` directory.
8. Not regenerate collection diagnostics.
9. Not write an active contract manifest.

### Default CLI Failure

Running the same default preflight against committed schema-only exports must:

1. Return a nonzero exit code.
2. Print machine-readable JSON to stderr.
3. Include `ready: false`.
4. Include blockers containing `schema_only`.
5. Not create or modify any requested destination directory.

### Explicit Commit Import

Copying validated real Fluent source exports must require an explicit flag, named `--commit-import` unless the surrounding code strongly suggests a better local name.

Running with `--commit-import` and `--run-collection-validator` against a complete candidate bundle must:

1. Run the same preflight validation first.
2. Run the staged collection validator before copying.
3. Copy only the required files into the destination directory after validation passes.
4. Ensure the public evidence map is present in the destination.
5. Run the collection validator on the destination when requested.
6. Print JSON to stdout with a mode marker such as `mode: "commit_import"`.
7. Preserve `fluent_parity_claimed: false`.
8. Preserve the existing ready gate semantics.

## Test Plan

Use red-to-green TDD.

1. Add a failing CLI integration test proving default preflight with a complete temporary bundle does not copy to the destination.
2. Add a failing CLI integration test proving `--commit-import --run-collection-validator` performs the copy and keeps the ready gate honest.
3. Add a failing CLI integration test proving schema-only default preflight fails with JSON blockers and no destination write.
4. Run only the focused importer integration test first and observe the expected RED failure.
5. Implement the smallest CLI change needed to pass the tests.
6. Re-run the focused importer integration test and observe GREEN.
7. Run the adjacent artifact/readiness integration tests to ensure no regression.

## Implementation Plan

1. Keep the public Python function `import_real_fluent_source_exports(...)` as the explicit import/copy API for callers that already chose to import.
2. Keep `validate_import_bundle(...)` as the no-copy preflight API.
3. Change `main(...)` so it dispatches by CLI intent:
   - no `--commit-import`: call `validate_import_bundle(...)`, enrich the summary with CLI preflight fields, and return without copying;
   - with `--commit-import`: call `import_real_fluent_source_exports(...)` using existing destination and collection options.
4. Add a CLI-only success summary shape that is easy for operators and scripts to inspect:
   - `mode`
   - `ready`
   - `input_dir`
   - `destination_dir`
   - `copied_files`
   - `copied_file_count`
   - existing preflight checks/blockers/gate fields
5. Keep failures flowing through `ImportPreflightError` so stderr JSON behavior stays consistent.

## Acceptance Criteria

This goal is complete only when:

1. The detailed goal file exists in the repository and is referenced by the active Codex goal.
2. CLI default mode is preflight-only and no-copy.
3. Explicit `--commit-import` is required for copy/import from the CLI.
4. Focused RED test was observed before the implementation change.
5. Focused GREEN test was observed after the implementation change.
6. Adjacent importer/artifact integration tests pass.
7. The diff does not modify solver/parity logic or committed Fluent source-export data.
8. The final commit is pushed to the configured remote branch after verification.

