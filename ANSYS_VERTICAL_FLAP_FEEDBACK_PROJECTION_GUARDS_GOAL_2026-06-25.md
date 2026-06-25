# ANSYS Vertical Flap Feedback Projection Guards Goal - 2026-06-25

## Source Branch And Baseline

- Repository: `lizhuoh9/EasyFsi`
- Working directory: `D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM\refactored`
- Baseline branch: `solver/ansys-vertical-flap-feedback-conditioned-fluid-projection-2026-06-25`
- Baseline HEAD: `1d3b5052b7de5e43904bdb72db155d7fdefb46b5`
- New implementation branch: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`

## Objective

Add the smallest runtime and diagnostic guards needed before any new 50-step
ANSYS vertical-flap feedback-conditioned artifact is generated. The previous
branch proved that marker feedback is written into fluid-side velocity
Dirichlet arrays before projection. This branch must prevent stale marker-owned
constraints from accumulating, distinguish target-assembly residuals from
post-projection residuals, and add a short runtime smoke proving that a real
runner execution consumes feedback on step 2+.

This is still a plumbing and diagnostic guard branch. It must not claim that the
ANSYS validation is physically passing.

## Required Code Changes

### 1. Clear Marker-Owned Constraints Before Reassembly

- Keep a runner-local set of feedback-owned grid cells.
- Pass that set into `_apply_marker_feedback_to_fluid(...)`.
- Before writing current marker feedback constraints, clear the previous
  feedback-owned cells from:
  - `fluid.velocity_dirichlet_boundary_active`
  - `fluid.velocity_dirichlet_boundary_value_mps`
  - `fluid.velocity_dirichlet_boundary_projection_weight`
- Preserve non-marker constraints, especially inlet or pre-existing case
  constraints. The guard must only clear cells previously owned by marker
  feedback, not arbitrary active cells.
- Return the newly owned feedback cell set from the adapter and keep it in the
  runner loop for the next step.
- Report a count for stale marker-owned cells cleared on each step.

### 2. Preserve Existing Field Semantics And Add Clearer Residual Fields

- Keep existing fields for backward compatibility:
  - `no_slip_residual_before_mps`
  - `no_slip_residual_after_mps`
- Treat `no_slip_residual_after_mps` as the target-assembly residual and expose
  the clearer alias:
  - `no_slip_target_residual_after_assembly_mps`
- After `_project_current_flow(...)`, compute the actual marker-mapped fluid
  velocity residual against marker velocity and expose:
  - `no_slip_projected_residual_after_projection_mps`
- The projected residual must be computed from `fluid.velocity` after
  projection, not from the Dirichlet target array.
- The projected residual must be finite and report `0.0` when no feedback was
  consumed.

### 3. Add Constraint Cell Diagnostics

- Add per-step and top-level report fields:
  - `fluid_feedback_constraint_cleared_cell_count`
  - `fluid_feedback_constraint_obstacle_cell_count`
  - `fluid_feedback_constraint_non_obstacle_cell_count`
  - `fluid_feedback_constraint_projection_participating_cell_count`
  - `no_slip_target_residual_after_assembly_mps`
  - `no_slip_projected_residual_after_projection_mps`
- The projection-participating count may initially be the non-obstacle count if
  the runner cannot cheaply distinguish a narrower pressure-operator
  participation set. It must be explicitly named and deterministic.
- Do not reinterpret `fluid_feedback_constraint_active_cell_count`; it remains
  the number of marker-feedback target cells written by the adapter.

### 4. Diagnostics Tool Propagation

Update `tools/validation/print_ansys_vertical_flap_diagnostics.py` so that:

- history CSV includes the new guard fields.
- displacement compare CSV includes the new guard fields.
- stage_check `[FSI_FEEDBACK]` prints the stale-clear count, obstacle/non-obstacle
  counts, projection-participating count, target residual, and post-projection
  residual.
- Missing fields in old committed artifacts remain tolerated. Existing old
  artifacts must not be rewritten just to add blank columns.

### 5. Runtime Smoke

Add a short runtime smoke that does not generate or overwrite committed
50-step artifacts. The smoke must prove:

- step 1 reports `fluid_projection_consumed_feedback is False`.
- step 2 reports `fluid_projection_consumed_feedback is True`.
- `fluid_projection_consumed_feedback_count == step_count - 1` for a 2- or
  3-step run.
- step 2 marker and active-cell counts are positive.
- target-assembly residual and projected residual fields are finite.
- stale-clear count is present and non-negative.

If a real CUDA smoke is too expensive for the normal focused suite, keep it as a
small explicit integration test target, not as a fake fixture-only proof.

## Required Tests

Follow RED -> GREEN:

1. Add source-level tests before implementation for:
   - `_apply_marker_feedback_to_fluid(...)` receives previous feedback-owned
     cells.
   - previous feedback-owned cells are cleared before current marker cells are
     written.
   - the runner keeps the returned feedback-owned cell set between steps.
   - target residual and projected residual are distinct named report fields.
2. Add diagnostics tests before implementation for:
   - history rows and CSVs contain all new guard fields.
   - stage_check prints all new guard fields.
3. Add runtime smoke before implementation for:
   - real 2- or 3-step execution consumes feedback on step 2+.
   - counts and residual fields are finite and report-compatible.

## Validation Commands

Use the trusted interpreter for this workspace:

```powershell
& 'D:\working\taichi\env\python.exe' -m pytest tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection.py tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection_runtime.py tests\tools\test_ansys_vertical_flap_diagnostics.py -q
```

Also run the existing ANSYS vertical-flap slice:

```powershell
& 'D:\working\taichi\env\python.exe' -m pytest tests\cases\test_ansys_vertical_flap_fsi.py tests\integration\test_ansys_vertical_flap_runner_loop_contract.py tests\tools\test_ansys_vertical_flap_diagnostics.py tests\integration\test_ansys_vertical_flap_closed_loop_feedback.py tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection.py tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection_runtime.py -k "not matches_reference_displacement_tolerance" -q
```

Run syntax and whitespace checks:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile benchmarks\official\solid_mpm_fsi_runner.py tools\validation\print_ansys_vertical_flap_diagnostics.py tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection.py tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection_runtime.py tests\tools\test_ansys_vertical_flap_diagnostics.py
git diff --check
```

## Explicit Non-Goals

- Do not modify `simulation_core/`.
- Do not change ANSYS case constants, material parameters, damping, support
  radius, tolerances, or reference values.
- Do not regenerate, overwrite, or backfill old 50-step artifacts.
- Do not claim CI passed unless a real GitHub Actions run exists for the pushed
  SHA.
- Do not claim ANSYS physical validation is fixed. This branch only strengthens
  the feedback-conditioned projection guard and diagnostics.

## Done Criteria

- The detailed goal file is committed.
- A short `/goal`-style objective references this file.
- RED tests are committed before production implementation.
- GREEN implementation is committed separately.
- Focused tests, ANSYS vertical-flap slice, `py_compile`, and `git diff --check`
  pass locally.
- The branch is pushed to `origin`.
- Final report includes the branch name, final commit hash, local validation
  commands/results, and a clear artifact-honesty note.
