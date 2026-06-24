# ANSYS Vertical Flap Runtime Diagnosis Goal - 2026-06-25

## Purpose

Build the next validation-tooling step for the ANSYS vertical flap case before any solver fix work continues.

The previous validation tooling branch established report-file parsing and stage-level diagnostics. This step must expose the current known red condition more directly: the EasyFsi solid tip streamwise history can rebound against the expected monotone downstream response, but the existing diagnostics do not print that failure explicitly.

This branch is diagnostic-only. It must make the current solver/case state easier to judge from artifacts, without changing the solver, physics, case parameters, reference values, or acceptance tolerances.

## Branch And Base

- Work branch: `validation/ansys-vertical-flap-runtime-diagnosis-2026-06-24`
- Base branch: `validation/ansys-vertical-flap-fsi-2026-06-24`
- Base commit observed before this goal was written: `4149187`

## Hard Boundaries

Do not modify any solver or case physics behavior in this step.

Forbidden paths and behaviors:

- No edits under `simulation_core/`.
- No edits under `cases/ansys_vertical_flap_fsi/` unless a test-only fixture proves it is strictly necessary. The expected implementation should not need this.
- No edits under `benchmarks/`.
- No changes to pressure projection, IBM/HIBM coupling, structural update logic, time integration, damping, loads, material parameters, or boundary conditions.
- No changes to ANSYS reference values, validation tolerances, or pass/fail thresholds except adding the new diagnostic status classification.
- No generated long-run `validation_runs/` outputs should be committed.
- Do not hide or downgrade the known red condition.

Allowed paths:

- `tools/validation/print_ansys_vertical_flap_diagnostics.py`
- `tests/tools/test_ansys_vertical_flap_diagnostics.py`
- `docs/validation/ANSYS_VERTICAL_FLAP_RUNTIME_DIAGNOSIS_GOAL_2026-06-25.md`
- Existing validation documentation only if needed to keep usage notes accurate.

## Required Diagnostic Behavior

The diagnostics must add solid tip history health metrics to the summary row printed by `tools/validation/print_ansys_vertical_flap_diagnostics.py`.

Add these summary columns:

- `tip_dz_final_m`
- `tip_dz_min_m`
- `tip_dz_max_m`
- `tip_dz_monotonic_violation_count`
- `first_tip_dz_violation_step`
- `max_tip_dz_rebound_m`
- `tip_dz_sign_violation_count`

Definitions:

- Read `dz` from `history[*].tip_mean_displacement_m[2]`.
- Treat negative `dz` as the expected streamwise direction for this case.
- `tip_dz_final_m` is the last available history `dz`.
- `tip_dz_min_m` and `tip_dz_max_m` are computed over all available history `dz` values.
- `tip_dz_sign_violation_count` counts history entries where `dz > 0.0`.
- `tip_dz_monotonic_violation_count` counts consecutive history pairs where `later_dz > earlier_dz + 1e-8`.
- `first_tip_dz_violation_step` is the `step` value of the later history entry for the first monotonic violation.
- `max_tip_dz_rebound_m` is the maximum positive rebound amount `later_dz - earlier_dz` among monotonic violations.

Missing or malformed history should not crash diagnostics:

- Numeric extrema/final/rebound fields should be blank when no valid history `dz` exists.
- Violation counts should be zero when no valid history `dz` exists.
- `first_tip_dz_violation_step` should be blank when no monotonic violation exists.

## Required Status Behavior

Add a new status:

- `FAIL_SOLID_HISTORY`

The status ordering must be:

1. Existing input/reference/load/file failures.
2. Existing fluid failures.
3. Existing solid sign failure: `FAIL_SOLID_SIGN`.
4. New solid history failure: `FAIL_SOLID_HISTORY`.
5. Existing magnitude failure: `FAIL_MAGNITUDE`.
6. Existing pass or unknown states.

The new logic must be:

- If any valid tip history `dz` is positive, classify as `FAIL_SOLID_SIGN`.
- Else if any consecutive valid tip history pair has `later_dz > earlier_dz + 1e-8`, classify as `FAIL_SOLID_HISTORY`.
- Else continue to existing magnitude classification.

The history failure must win over magnitude failure. A report that has both a monotonic rebound and displacement magnitude error must return `FAIL_SOLID_HISTORY`.

## Required Stage Check Behavior

The generated `stage_check.md` text must expose the same history health information under a solid-response or solid-history section.

It should report at least:

- `tip_dz_final_m`
- `tip_dz_min_m`
- `tip_dz_max_m`
- `tip_dz_monotonic_violation_count`
- `first_tip_dz_violation_step`
- `max_tip_dz_rebound_m`
- `tip_dz_sign_violation_count`
- the status or diagnosis text for `FAIL_SOLID_HISTORY`

The FSI feedback section must explicitly report:

- `fluid_recomputed_after_feedback`
- `feedback_closure_status`

If the report has not recomputed fluid after feedback, `feedback_closure_status` must be:

- `OPEN_LOOP_LOAD_REUSE`

If a future report contains evidence that fluid was recomputed after solid feedback, the status may become:

- `CLOSED_LOOP_RECOMPUTED_FLOW`

This branch is expected to reveal open-loop load reuse for the current validation tooling base, not to repair it.

## Required Tests

Add focused unit tests in `tests/tools/test_ansys_vertical_flap_diagnostics.py`.

Required test names:

- `test_summary_records_tip_history_monotonic_violation`
- `test_status_returns_fail_solid_history_before_fail_magnitude`
- `test_stage_check_reports_open_loop_load_reuse`
- `test_history_health_metrics_are_blank_or_zero_for_missing_history`

Required fixture behavior:

- Use history `dz` values `-2e-5`, `-4e-5`, `-3e-5`.
- This has one monotonic violation at the third sample.
- Expected monotonic violation count: `1`.
- Expected first violation step: `3`.
- Expected max rebound: `1e-5`.
- Expected status: `FAIL_SOLID_HISTORY`.

## Required Verification

Run focused validation after implementation:

- `D:\working\taichi\env\python.exe -m py_compile tools\validation\print_ansys_vertical_flap_diagnostics.py tests\tools\test_ansys_vertical_flap_diagnostics.py`
- `D:\working\taichi\env\python.exe -m unittest tests.tools.test_ansys_vertical_flap_diagnostics -v`
- `D:\working\taichi\env\python.exe -m unittest discover -s tests\tools -p "test_*.py" -v`
- `D:\working\taichi\env\python.exe scripts\validate_structure.py`
- `git diff --check`

Optional if runtime cost is acceptable:

- `D:\working\taichi\env\python.exe -m unittest tests.cases.test_ansys_vertical_flap_fsi -v`

If the optional ANSYS vertical flap case remains physically red, do not patch around it in this branch. The correct outcome is that diagnostics now identify the red condition more precisely.

## Completion Criteria

This step is complete when:

- The goal file is committed.
- The new summary fields are present.
- `FAIL_SOLID_HISTORY` is implemented with the required ordering.
- `stage_check.md` output contains solid-history and FSI feedback closure diagnostics.
- Required tests are present and passing.
- Required verification commands pass, except any explicitly documented optional known-red runtime case.
- The branch is pushed to GitHub.
