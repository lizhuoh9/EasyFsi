# ANSYS Vertical Flap Runner Loop Contract Test Goal - 2026-06-25

## Purpose

Add a focused test layer that prevents the next solver repair from becoming a fake closed-loop fix.

The previous closed-loop feedback contract tests pinned the artifact-level targets:

- `OPEN_LOOP_LOAD_REUSE -> CLOSED_LOOP_RECOMPUTED_FLOW`
- `FAIL_SOLID_HISTORY -> no solid history rebound`
- final displacement relative error `<= 0.20`

This follow-up test branch must pin the runner-level implementation contract from the review note: it is not enough to set report flags. The ANSYS vertical-flap runner must actually recompute/project the fluid field inside the FSI loop after marker/solid feedback is available and before the next stress sample.

## Branch And Base

- Work branch: `test/ansys-vertical-flap-runner-loop-contract-2026-06-25`
- Base branch: `test/ansys-vertical-flap-closed-loop-feedback-2026-06-25`
- Base commit observed before this goal was written: `ac6069e`

For PR hygiene, this PR must be opened with:

- base: `test/ansys-vertical-flap-closed-loop-feedback-2026-06-25`
- head: `test/ansys-vertical-flap-runner-loop-contract-2026-06-25`

That keeps the diff limited to this goal file plus the new runner-loop contract tests.

## Hard Boundaries

This branch is test-only plus goal documentation.

Do not implement the solver repair here.

Forbidden:

- No edits under `simulation_core/`.
- No edits under `benchmarks/official/solid_mpm_fsi_runner.py`.
- No edits under `cases/ansys_vertical_flap_fsi.py`.
- No changes to case parameters, reference values, tolerances, material constants, solver formulas, time integration, pressure projection kernels, stress sampling implementation, scatter implementation, damping, support radius, or runtime artifacts.
- Do not regenerate `validation_runs/`.
- Do not remove `expectedFailure` markers from existing artifact-level contract tests in this branch.

Allowed:

- Add this goal file under `docs/validation/`.
- Add focused source/contract tests under `tests/integration/` or `tests/contracts/`.
- Read current source and artifacts to define tests.

## Required Test Intent

The new tests must encode the implementation contract for the upcoming solver branch:

1. The current ANSYS vertical-flap runner must still be recognized as open-loop before the solver fix.
2. A future solver implementation must prove it does more than set:
   - `fluid_recomputed_after_feedback = True`
   - `feedback_closure_status = CLOSED_LOOP_RECOMPUTED_FLOW`
3. The runner source must eventually contain an explicit per-step flow recomputation/projection path inside the FSI loop.
4. The runner report must eventually expose:
   - `fluid_recomputed_after_feedback`
   - `feedback_closure_status`
   - `fluid_recompute_count`
5. Each history entry must eventually expose:
   - `fluid_recomputed`
   - `local_velocity_peak_mps`
   - `pressure_min_pa`
   - `pressure_max_pa`
   - `flow_projection_report`

## Required Tests

Add:

```text
tests/integration/test_ansys_vertical_flap_runner_loop_contract.py
```

Required test names:

- `test_current_runner_solves_computed_flow_before_fsi_loop`
- `test_closed_loop_solver_must_report_fluid_recompute_count`
- `test_closed_loop_solver_must_record_per_step_flow_recompute_fields`
- `test_closed_loop_solver_must_project_fluid_inside_fsi_loop`

### `test_current_runner_solves_computed_flow_before_fsi_loop`

This must be a passing test documenting the current open-loop structure.

It should inspect `benchmarks/official/solid_mpm_fsi_runner.py` source and verify the current baseline structure still contains:

- a call to `_solve_computed_flow(fluid, config)`
- a later `for step_index in range(config.step_count):` FSI loop
- the `_solve_computed_flow(fluid, config)` call appears before the FSI loop

This test documents why the current baseline reports `OPEN_LOOP_LOAD_REUSE`.

### `test_closed_loop_solver_must_report_fluid_recompute_count`

This must be an `expectedFailure` source contract test.

Expected future contract:

- runner source contains `fluid_recompute_count`
- the final report includes `fluid_recomputed_after_feedback`
- the final report includes `feedback_closure_status`
- the final report exposes `CLOSED_LOOP_RECOMPUTED_FLOW`

Current known state may lack one or more of these fields, so this test should remain expected-failing until the solver branch is implemented.

### `test_closed_loop_solver_must_record_per_step_flow_recompute_fields`

This must be an `expectedFailure` source contract test.

Expected future contract:

- runner source records `fluid_recomputed` in each history entry
- runner source records `local_velocity_peak_mps` in each history entry
- runner source records `pressure_min_pa` in each history entry
- runner source records `pressure_max_pa` in each history entry
- runner source records `flow_projection_report` in each history entry

### `test_closed_loop_solver_must_project_fluid_inside_fsi_loop`

This must be an `expectedFailure` source contract test.

Expected future contract:

- inside the FSI loop body, the runner must call a fluid projection/recompute helper before stress sampling.
- the test should not merely accept a report flag.
- acceptable future source evidence can include:
  - `fluid.project(...)` inside the FSI loop body
  - `_project_current_flow(...)` inside the FSI loop body
  - another clearly named per-step flow recompute helper inside the FSI loop body

This is intentionally a source-level guard because it catches the common AI regression where the report fields are toggled without moving the actual solver computation.

## Required Verification

Run:

```powershell
D:\working\taichi\env\python.exe -m py_compile tests\integration\test_ansys_vertical_flap_runner_loop_contract.py
D:\working\taichi\env\python.exe -m unittest tests.integration.test_ansys_vertical_flap_runner_loop_contract -v
D:\working\taichi\env\python.exe -m unittest discover -s tests\integration -p "test_*.py" -v
D:\working\taichi\env\python.exe -m unittest discover -s tests\tools -p "test_*.py" -v
D:\working\taichi\env\python.exe scripts\validate_structure.py
git diff --check
```

Do not use full:

```powershell
unittest discover -s tests -p "test_*.py" -v
```

as this checkout already enters heavy Taichi solver tests, times out, and exposes existing discovery/import issues unrelated to this test-only branch.

## Completion Criteria

This goal is complete when:

- This detailed goal file is committed.
- The runner-loop contract test file is committed.
- The current open-loop source structure is documented by a passing test.
- Future closed-loop source requirements are documented by expected-failure tests.
- Focused integration/tools/structure validation passes.
- No solver, case, benchmark runner, or runtime artifact files are modified.
- The branch is pushed to GitHub.
