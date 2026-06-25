# ANSYS Vertical-Flap Feedback-Conditioned Projection Goal - 2026-06-25

## Source Context

This goal follows the PR-ready branch
`solver/ansys-vertical-flap-closed-loop-runner-2026-06-25`, whose latest checked
HEAD was `bfc96fa3d2787660d7d19bafe74a65aed687eb42`.

That branch established structural closed-loop behavior and honest diagnostics:

- Fluid projection now occurs inside the FSI loop before stress sampling.
- Step 1 is reported as pre-feedback projection.
- Step 2 and later projections are reported as after-feedback projection only
  when previous marker feedback exists.
- `easyfsi_history.csv`, `displacement_compare.csv`, and `stage_check.md` now
  carry flow/projection diagnostics.
- Blank `projection_final_residual =` lines are omitted when the current
  projection report does not provide a residual key.
- The artifact still honestly reports `FAIL_FLOW`.

The remaining technical gap is that after-feedback projection is still only a
structural loop claim. It proves that projection occurs after marker feedback
was updated, but it does not prove that updated marker surface state is consumed
by the fluid projection as a no-slip or immersed-boundary constraint.

## Primary Objective

Convert the ANSYS vertical-flap runner from structural after-feedback
reprojection to a minimal feedback-conditioned fluid projection contract.

The first implementation must be deliberately conservative:

- Add a runner-level `_apply_marker_feedback_to_fluid(...)` adapter before each
  `_project_current_flow(...)`.
- Step 1 must report no consumed feedback.
- Step 2 and later must consume previous marker feedback and expose nonzero
  marker/cell constraint counts when feedback is available.
- The adapter must use existing fluid-side fields where possible, especially
  `velocity_dirichlet_boundary_active`,
  `velocity_dirichlet_boundary_value_mps`, and
  `velocity_dirichlet_boundary_projection_weight`.
- The adapter must not modify `simulation_core/` in this goal.

This goal is about making the coupling contract physically explicit and
diagnosable. It is not a claim that the ANSYS physical validation now passes.

## In-Scope Files

- `benchmarks/official/solid_mpm_fsi_runner.py`
- `tools/validation/print_ansys_vertical_flap_diagnostics.py`
- `tests/integration/test_ansys_vertical_flap_feedback_conditioned_projection.py`
- Existing ANSYS runner/diagnostics tests if they need field-name updates
- Regenerated ANSYS compare artifacts only if the committed fixture/report
  surface changes

## Out-of-Scope Files And Behavior

- No `simulation_core/` edits.
- No case constant changes.
- No tolerance changes.
- No material, damping, support radius, or step-count tuning.
- No displacement-target tuning.
- No GitHub CI claim unless a workflow run actually exists for the pushed SHA.
- No claim that Fluent/ANSYS physical parity has been achieved.
- No removal of expected failures that still represent real physical failures.

## Required Runner Contract

Inside `run_rectangular_solid_marker_mpm_fsi_smoke(...)`, the FSI loop must have
this effective order:

```text
feedback_available_before_projection = feedback_available_for_projection
feedback_constraint_report = _apply_marker_feedback_to_fluid(...)
latest_flow_report = _project_current_flow(...)
latest_stress_report = _sample_stress_to_marker_forces(...)
...
latest_feedback_report = markers.update_surface_feedback_from_mpm_surface_particles(...)
feedback_available_for_projection = True
```

The source-level contract must make it impossible to confuse these states:

- `fluid_projection_after_feedback_count > 0`
- `fluid_projection_consumed_feedback_count > 0`

After-feedback projection means only that feedback existed before projection.
Consumed-feedback projection means the runner applied marker-derived fluid-side
constraints before projection.

## Required Report Fields

Top-level report must include:

```text
fluid_projection_consumed_feedback_count
fluid_feedback_constraint_marker_count
fluid_feedback_constraint_active_cell_count
fluid_projection_consumed_feedback
no_slip_residual_before_mps
no_slip_residual_after_mps
```

Per-step `history` must include:

```text
fluid_projection_consumed_feedback
fluid_feedback_constraint_marker_count
fluid_feedback_constraint_active_cell_count
no_slip_residual_before_mps
no_slip_residual_after_mps
```

Step 1 requirements:

```text
feedback_available_before_projection == False
fluid_projection_consumed_feedback == False
fluid_feedback_constraint_marker_count == 0
fluid_feedback_constraint_active_cell_count == 0
```

Step 2+ requirements:

```text
feedback_available_before_projection == True
fluid_projection_consumed_feedback == True
fluid_feedback_constraint_marker_count > 0
fluid_feedback_constraint_active_cell_count > 0
```

## Minimal Adapter Behavior

The adapter may stay runner-local for this goal:

```python
def _apply_marker_feedback_to_fluid(
    markers,
    fluid,
    config,
    *,
    feedback_available: bool,
) -> dict[str, object]:
    ...
```

When `feedback_available` is false, it must return a zero-count report without
changing the fluid constraint fields.

When `feedback_available` is true, it must:

1. Read current marker position and velocity from `markers.x_gamma_m` and
   `markers.v_gamma_mps`.
2. Map each marker to an existing fluid cell using the domain bounds and grid
   shape.
3. Activate velocity Dirichlet constraints on the mapped cells.
4. Set those Dirichlet values from the marker velocity, producing a no-slip
   style fluid-side target.
5. Preserve the inlet plane constraints that `_initialize_inlet_flow(...)`
   already installed.
6. Return marker count, active-cell count, and before/after no-slip residuals.

The adapter can be approximate in this first PR. It must be explicit,
deterministic, testable, and visible in reports.

## Diagnostics Requirements

`tools/validation/print_ansys_vertical_flap_diagnostics.py` should propagate the
new per-step feedback-conditioned projection fields into:

- `easyfsi_history.csv`
- `displacement_compare.csv`

`stage_check.md` should show whether the final projection consumed feedback and
the final constraint counts/residuals.

## Test-First Plan

1. Add a source-level integration test file:

```text
tests/integration/test_ansys_vertical_flap_feedback_conditioned_projection.py
```

2. RED source test requirements:

- `_apply_marker_feedback_to_fluid(` exists.
- `_apply_marker_feedback_to_fluid(` occurs before `_project_current_flow(` in
  the FSI loop.
- `_project_current_flow(` occurs before `_sample_stress_to_marker_forces(`.
- The runner tracks `fluid_projection_consumed_feedback_count`.
- The runner writes the required per-step and top-level report fields.

3. RED diagnostics test requirements:

- A fixture report with the new fields produces `easyfsi_history.csv` columns.
- A fixture report with the new fields produces `displacement_compare.csv`
  columns.
- `stage_check.md` includes consumed-feedback status and constraint counts.

4. Confirm RED before modifying production code.

5. Implement the smallest runner/report/diagnostics changes needed to pass.

6. Run the focused tests:

```powershell
& 'D:\working\taichi\env\python.exe' -m pytest tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection.py tests\tools\test_ansys_vertical_flap_diagnostics.py -q
```

7. Run the ANSYS slice used by the previous goal, excluding only the long
50-step expected-failure physical run:

```powershell
& 'D:\working\taichi\env\python.exe' -m pytest tests\cases\test_ansys_vertical_flap_fsi.py tests\integration\test_ansys_vertical_flap_runner_loop_contract.py tests\tools\test_ansys_vertical_flap_diagnostics.py tests\integration\test_ansys_vertical_flap_closed_loop_feedback.py tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection.py -k "not matches_reference_displacement_tolerance" -q
```

## Physical Honesty Requirements

This branch may improve the coupling contract without passing `FAIL_FLOW`.

Final reporting must continue to state:

- This is feedback-conditioned projection plumbing and diagnostics.
- It does not prove ANSYS physical validation has passed.
- Current known artifact status from the previous branch was `FAIL_FLOW`.
- Flow gate remains the first physical gate to fix before tuning solid response.

## Done Criteria

- Detailed goal file exists and is referenced by the short goal.
- RED test proves the missing adapter/report contract.
- GREEN implementation adds the minimal adapter before projection.
- New report fields exist at top level and per step.
- Diagnostics propagate the new fields.
- Focused tests pass.
- Relevant ANSYS slice passes with expected failures preserved.
- No `simulation_core/`, constants, or tolerances are changed.
- Branch is committed and pushed to
  `origin/solver/ansys-vertical-flap-feedback-conditioned-fluid-projection-2026-06-25`.
