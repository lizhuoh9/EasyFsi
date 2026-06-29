# ANSYS Vertical-Flap Real Solver Validation Goal - 2026-07-01

## Source

This goal is derived from the attached remote-branch review and execution plan:

- Repository: `lizhuoh9/EasyFsi`
- Branch: `codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`
- Reviewed remote HEAD: `15b66e573e83801f98699871fcf4b630018bf144`
- Reviewed commit subject: `fix: make Fluent source import atomic`
- Attachment path supplied by the user:
  `C:\Users\lizhu\.codex\attachments\9ebe1da2-fc73-46a0-8db3-6c59ae35f7ac\pasted-text.txt`

## Current Accepted State

The importer layer is now considered sufficiently guarded for the next validation stage:

1. CLI default mode is no-copy preflight.
2. Copy/import requires explicit `--commit-import`.
3. `--run-collection-validator` without `--commit-import` fails fast.
4. Official `SOURCE_EXPORTS_ROOT` writes require `--run-collection-validator`.
5. Commit-import uses staging, backup, replace, and rollback behavior.
6. Importer tests cover rollback, stale-file removal, public evidence map preservation, JSON shape, and official-destination guard behavior.
7. Committed `source_exports` still contain schema-only placeholders, not fake real Fluent rows.
8. `fluent_parity_claimed` remains false.
9. GitHub Actions green is not visible and must not be claimed.

## Core Direction

Do not keep adding importer guard code unless a real solver or Fluent validation run exposes a concrete defect.

The next work must move to real validation evidence:

1. Run the EasyFsi ANSYS vertical-flap selected formulation coupled step-50 validation.
2. Run the EasyFsi generic solver selected formulation vertical-flap validation.
3. Verify the generated EasyFsi artifacts with existing artifact tests.
4. If a real ANSYS Fluent export bundle is available, run no-copy preflight, temporary import, official import, collection diagnostics, and parity comparison.
5. If a real Fluent bundle or Fluent executable/license is not available in this workspace, record that as an explicit blocker and do not fabricate data.

## Non-Goals

Do not:

1. Add more importer guard code as speculative hardening.
2. Modify committed Fluent `source_exports` unless a provenance-backed real Fluent bundle passes no-copy preflight and temporary import first.
3. Generate fake real Fluent CSV rows.
4. Edit Fluent metadata to bypass provenance or schema checks.
5. Modify solver tolerances to force a parity result.
6. Claim Fluent parity without real Fluent reference data and policy-backed comparison.
7. Set `fluent_parity_claimed=true`.
8. Claim GitHub Actions green without a visible remote run.
9. Hide solver failures by updating tests to accept bad physics.

## Required Local EasyFsi Solver Validation

Use the project validation interpreter where possible:

```text
D:\working\taichi\env\python.exe
```

First compile the relevant scripts:

```powershell
python -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_coupled_step50.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_ansys_vertical_flap_generic_solver.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\import_real_fluent_source_exports.py
```

Then run selected formulation coupled validation:

```powershell
python validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_coupled_step50.py
```

Then run generic solver selected formulation through `run()`:

```powershell
@'
import importlib.util
from pathlib import Path

script = Path("validation_runs/ansys_vertical_flap_fsi/scripts/run_ansys_vertical_flap_generic_solver.py")
spec = importlib.util.spec_from_file_location("easyfsi_generic_vertical_flap", script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

payload = module.run()
row = payload["rows"][0]
print("candidate_status =", payload["candidate_status"])
print("completed_step_count =", row["completed_step_count"])
print("max_displacement_m =", row["max_displacement_m"])
print("max_pressure_abs_pa =", row["max_pressure_abs_pa"])
print("max_velocity_mps =", row["max_velocity_mps"])
print("fluent_parity_claimed =", payload["fluent_parity_claimed"])
'@ | python -
```

Verify with:

```powershell
python -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_coupled_step50_artifacts `
  tests.integration.test_ansys_vertical_flap_generic_solver_artifacts `
  -v
```

## Required Real Fluent Handling

If a real Fluent export bundle is available, process it only in this sequence:

1. No-copy preflight.
2. Temporary destination commit-import with `--run-collection-validator`.
3. Official `source_exports` commit-import with `--run-collection-validator`.
4. Strict real Fluent source-export artifact test.
5. Collection diagnostics regeneration.
6. Solver-vs-Fluent parity comparison.

If the bundle is unavailable, do not write `source_exports`; instead, add a clear validation status artifact that says the Fluent side is blocked by missing external Fluent exports.

## Tests And Evidence

Existing artifact tests are the primary verification for this stage:

1. `tests.integration.test_ansys_vertical_flap_traction_selected_formulation_coupled_step50_artifacts`
2. `tests.integration.test_ansys_vertical_flap_generic_solver_artifacts`
3. If Fluent source exports become available:
   - `tests.integration.test_ansys_vertical_flap_real_fluent_source_export_import`
   - `tests.integration.test_ansys_vertical_flap_real_fluent_source_export_artifacts`
   - `tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts`
   - `tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic`
   - `tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts`

If the EasyFsi solver rerun changes artifacts, commit the changed artifacts with an evidence-focused message. If it fails, fix the solver or artifact contract only when the failure identifies a real repo bug.

## Acceptance Criteria

This goal is complete only when:

1. This detailed goal file exists in the repository and the active Codex goal references it.
2. The selected formulation coupled step-50 runner has been attempted.
3. The generic solver selected formulation runner has been attempted.
4. The corresponding artifact tests have been run.
5. Any changed EasyFsi validation artifacts are either committed with evidence or explained as intentionally unchanged.
6. The workflow does not add speculative importer guards.
7. No fake Fluent source-export data is committed.
8. If Fluent exports are unavailable, the blocker is documented explicitly.
9. No solver/parity success is overstated beyond the artifacts.
10. Verification commands and outcomes are recorded in the final response.
11. The final verified changes are committed and pushed to the current remote branch.

