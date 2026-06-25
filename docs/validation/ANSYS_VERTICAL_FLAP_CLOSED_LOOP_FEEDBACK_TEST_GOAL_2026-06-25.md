# ANSYS Vertical Flap Closed-Loop Feedback Test Goal - 2026-06-25

## Purpose

Add focused tests for the next ANSYS vertical-flap solver repair before changing solver code.

The previous official-web baseline showed:

- `status = FAIL_SOLID_HISTORY`
- `feedback_closure_status = OPEN_LOOP_LOAD_REUSE`
- `tip_dz_monotonic_violation_count = 23`
- final displacement relative error vs the ANSYS web-published displacement scale is about `0.46994`

The next solver branch must repair closed-loop FSI behavior. This test-only branch defines the expected contract first, without implementing the solver fix.

## Branch And Base

- Work branch: `test/ansys-vertical-flap-closed-loop-feedback-2026-06-25`
- Base branch: `validation/ansys-vertical-flap-official-web-baseline-2026-06-25`
- Baseline commit observed before this goal was written: `07bc780`

## Hard Boundaries

This branch is test-only plus goal documentation.

Do not modify solver behavior.

Forbidden:

- No edits under `simulation_core/`.
- No edits under `benchmarks/official/solid_mpm_fsi_runner.py`.
- No edits under `cases/ansys_vertical_flap_fsi.py`.
- No edits to solver formulas, feedback-loop logic, pressure projection, stress sampling, scatter, solid integration, damping, support radius, materials, boundary conditions, or tolerances.
- No new runtime artifact generation should be required by the tests.
- Do not rerun the 50-step case as part of these tests.

Allowed:

- Add this detailed goal file under `docs/validation/`.
- Add a focused test file under `tests/integration/`.
- Add package marker files if needed for unittest discovery.

## Test Strategy

The tests must capture the solver repair target without making the current baseline branch fail globally.

Use two layers:

1. Passing report-shape tests that validate a synthetic closed-loop report can be diagnosed as closed-loop by the existing diagnostics tool.
2. `unittest.expectedFailure` contract tests that encode the current real EasyFsi baseline gap:
   - current artifact still reports `OPEN_LOOP_LOAD_REUSE`
   - current artifact still reports `FAIL_SOLID_HISTORY`
   - current artifact still has nonzero history rebound count

This keeps the branch green while documenting exactly what the upcoming solver branch must make pass. When the solver is fixed, remove `expectedFailure` from the real-artifact contract tests.

## Required Tests

Add:

```text
tests/integration/test_ansys_vertical_flap_closed_loop_feedback.py
```

Required test names:

- `test_diagnostics_accept_closed_loop_report_contract`
- `test_current_web_baseline_requires_closed_loop_feedback`
- `test_current_web_baseline_requires_no_solid_history_rebound`
- `test_current_web_baseline_targets_twenty_percent_displacement_error`

### `test_diagnostics_accept_closed_loop_report_contract`

This must be a passing synthetic-report test.

Build a minimal in-memory report with:

- `fluid_recomputed_after_feedback = True`
- `feedback_closure_status = CLOSED_LOOP_RECOMPUTED_FLOW` implied through `build_stage_check`
- monotone negative tip `dz` history
- no sign violations
- no scatter/interface failures
- displacement error within tolerance

Expected:

- `build_summary_row(report)["status"] == "PASS_SMOKE"`
- `build_summary_row(report)["tip_dz_monotonic_violation_count"] == 0`
- `build_stage_check(... )` contains:
  - `fluid_recomputed_after_feedback = true`
  - `feedback_closure_status = CLOSED_LOOP_RECOMPUTED_FLOW`

### `test_current_web_baseline_requires_closed_loop_feedback`

This must be an `expectedFailure` test against:

```text
validation_runs/ansys_vertical_flap_fsi/compare/stage_check.md
```

Expected future contract:

- stage check contains `feedback_closure_status = CLOSED_LOOP_RECOMPUTED_FLOW`
- stage check contains `fluid_recomputed_after_feedback = true`

Current known result:

- `OPEN_LOOP_LOAD_REUSE`
- `fluid_recomputed_after_feedback = false`

### `test_current_web_baseline_requires_no_solid_history_rebound`

This must be an `expectedFailure` test against:

```text
validation_runs/ansys_vertical_flap_fsi/compare/easyfsi_summary.json
```

Expected future contract:

- `status != FAIL_SOLID_HISTORY`
- `tip_dz_monotonic_violation_count == 0`

Current known result:

- `status = FAIL_SOLID_HISTORY`
- `tip_dz_monotonic_violation_count = 23`

### `test_current_web_baseline_targets_twenty_percent_displacement_error`

This must be an `expectedFailure` test against:

```text
validation_runs/ansys_vertical_flap_fsi/compare/displacement_compare.csv
```

Expected first solver target:

- final displacement relative error vs the ANSYS web-published displacement scale is `<= 0.20`

Current known result:

- final relative error is about `0.46994`

## Required Verification

Run:

```powershell
D:\working\taichi\env\python.exe -m py_compile tests\integration\test_ansys_vertical_flap_closed_loop_feedback.py
D:\working\taichi\env\python.exe -m unittest tests.integration.test_ansys_vertical_flap_closed_loop_feedback -v
D:\working\taichi\env\python.exe -m unittest discover -s tests -p "test_*.py" -v
D:\working\taichi\env\python.exe scripts\validate_structure.py
git diff --check
```

Expected verification behavior:

- The new integration test module must pass overall.
- The real-baseline contract tests should be reported as expected failures until solver repair lands.
- Existing tool and structure tests must remain green.

## Completion Criteria

This goal is complete when:

- The detailed goal file is committed.
- The closed-loop feedback integration test file is committed.
- The tests document the upcoming solver target without changing solver behavior.
- Required verification commands pass.
- The branch is pushed to GitHub.
