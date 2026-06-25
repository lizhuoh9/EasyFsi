# ANSYS Vertical Flap Three-Step Feedback Clear Smoke Goal - 2026-06-25

## Source Branch And Baseline

- Repository: `lizhuoh9/EasyFsi`
- Working directory: `D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM\refactored`
- Baseline branch: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Baseline HEAD: `8e17e08e517396f1598d32680014c1169b8e5d5a`
- New implementation branch: `solver/ansys-vertical-flap-feedback-three-step-clear-smoke-2026-06-25`

## Objective

Add the missing runtime proof that the ANSYS vertical-flap feedback projection
guard actually clears previous marker-owned fluid constraints during a real
runner execution. The previous branch added runner-local feedback-owned cell
tracking and a 2-step runtime smoke proving feedback consumption on step 2. This
branch must extend that runtime proof to 3 steps so step 3 exercises the
stale-clear path for constraints written during step 2.

This branch is a runtime-test strengthening branch. It must not change solver
physics, case constants, tolerances, material parameters, damping, support
radius, or committed 50-step artifacts.

## Required Behavior

The runtime smoke must execute `run_vertical_flap_fsi_smoke(...)` with a small
ANSYS vertical-flap configuration:

- `step_count=3`
- low `flow_projection_iterations` suitable for a contract smoke
- low `solid_substeps` suitable for a contract smoke

It must assert the real runner history shows:

- step 1: `fluid_projection_consumed_feedback is False`
- step 2: `fluid_projection_consumed_feedback is True`
- step 3: `fluid_projection_consumed_feedback is True`
- step 2: `fluid_feedback_constraint_active_cell_count > 0`
- step 3: `fluid_feedback_constraint_cleared_cell_count > 0`
- step 3: `fluid_feedback_constraint_cleared_cell_count <= step 2 active cell count`
- top-level `fluid_projection_consumed_feedback_count == step_count - 1`
- step 3 target residual and projected residual are finite
- step 3 projection-participating count remains present and non-negative

This proves the stale-clear guard runs in an actual execution, not only in
source-level contract tests.

## Test-Driven Workflow

1. Add the 3-step runtime assertion before production changes.
2. Run the runtime test and confirm RED if the new behavior is not already
   proven by the existing code.
3. If the code is already correct and the new test passes, keep this as a
   test-only branch and do not invent production edits.
4. Commit the test/goal as the task evidence.
5. Re-run the focused feedback projection guard target and the ANSYS vertical
   flap slice.
6. Run syntax and whitespace checks.
7. Push the new branch.

## Required Validation Commands

Use the trusted interpreter for this workspace:

```powershell
& 'D:\working\taichi\env\python.exe' -m pytest tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection_runtime.py -q
```

Run the focused guard target:

```powershell
& 'D:\working\taichi\env\python.exe' -m pytest tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection.py tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection_runtime.py tests\tools\test_ansys_vertical_flap_diagnostics.py -q
```

Run the existing ANSYS vertical-flap slice:

```powershell
& 'D:\working\taichi\env\python.exe' -m pytest tests\cases\test_ansys_vertical_flap_fsi.py tests\integration\test_ansys_vertical_flap_runner_loop_contract.py tests\tools\test_ansys_vertical_flap_diagnostics.py tests\integration\test_ansys_vertical_flap_closed_loop_feedback.py tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection.py tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection_runtime.py -k "not matches_reference_displacement_tolerance" -q
```

Run syntax and whitespace checks:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection_runtime.py
git diff --check
```

## Explicit Non-Goals

- Do not modify `simulation_core/`.
- Do not modify `benchmarks/official/solid_mpm_fsi_runner.py` unless the 3-step
  runtime test exposes an actual implementation failure.
- Do not change ANSYS case constants, tolerances, material parameters, damping,
  support radius, or reference values.
- Do not regenerate or overwrite old 50-step artifacts.
- Do not claim GitHub Actions passed unless a workflow run actually exists for
  this SHA.
- Do not claim ANSYS physical validation is fixed. This branch only proves the
  stale-clear runtime guard is exercised.

## Done Criteria

- This detailed goal file is committed.
- A short `/goal`-style objective references this file.
- The 3-step runtime smoke is committed.
- If no production change is needed, the branch remains test-only.
- Focused feedback guard tests, ANSYS vertical-flap slice, `py_compile`, and
  `git diff --check` pass locally.
- The branch is pushed to `origin`.
- Final report includes branch name, final commit hash, local validation
  results, and artifact-honesty note.
