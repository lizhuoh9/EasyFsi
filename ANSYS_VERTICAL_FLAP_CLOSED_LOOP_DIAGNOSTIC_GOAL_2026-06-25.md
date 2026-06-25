# ANSYS Vertical-Flap Closed-Loop Diagnostic Goal - 2026-06-25

## Source Context

This goal is based on the review note for `lizhuoh9/EasyFsi` commit
`2e5370686cd011e90a0f4bcf48fc362410ddeb50`.

The review concluded that the current branch has already moved the ANSYS
vertical-flap runner from one-shot open-loop flow reuse to per-step structural
closed-loop projection before stress sampling. That structural direction is
accepted. The remaining problem is diagnostic and semantic honesty:

- GitHub Actions did not provide an independent workflow run for that SHA.
- Existing test results are local validation results, not CI proof.
- The current physical artifact still fails the flow gate with `FAIL_FLOW`.
- The velocity peak is about `10.417 m/s`, below the official `20-29 m/s`
  range.
- The final displacement relative error remains about `0.80566`.
- Tip displacement history still has rebound and sign violations.

## Primary Objective

Implement and test the two small pre-PR corrections recommended by the review
note before pushing this branch:

1. Fix the closed-loop report semantics so `fluid_recomputed_after_feedback`
   means a projection occurred after prior marker feedback was available, not
   merely that at least one projection happened.
2. Add flow/projection diagnostics to the compare history artifacts so the next
   physical flow repair can be diagnosed from CSV and stage-check outputs.

This is a diagnostic and contract-hardening goal. It must not claim that ANSYS
vertical-flap physical validation is fixed.

## Scope

In scope:

- `benchmarks/official/solid_mpm_fsi_runner.py`
- `tools/validation/print_ansys_vertical_flap_diagnostics.py`
- Focused tests under `tests/integration` and `tests/tools`
- Existing ANSYS vertical-flap validation artifacts when they can be refreshed
  with the local environment
- A final commit and push to the configured remote after validation

Out of scope:

- Changing ANSYS case constants
- Changing tolerance thresholds
- Tuning material, damping, support radius, or displacement parameters
- Hiding `FAIL_FLOW`
- Removing expected failures for artifact-level physical targets
- Rewriting `simulation_core/` for this goal
- Claiming Fluent or official ANSYS parity

## Required Semantics

The runner must distinguish these counts:

- `fluid_projection_count`: total per-step fluid projections.
- `fluid_projection_after_feedback_count`: projections that occurred only after
  feedback from a previous step was available.

For a one-step run:

- `fluid_projection_count` must be `1`.
- `fluid_projection_after_feedback_count` must be `0`.
- `fluid_recomputed_after_feedback` must be `False`.
- `feedback_closure_status` must not claim after-feedback closure.

For a multi-step run:

- Step 1 projection is pre-feedback.
- Step 2 and later projections are after-feedback projections, provided marker
  feedback was updated after the previous solid step.
- `fluid_recomputed_after_feedback` must be true only when
  `fluid_projection_after_feedback_count > 0`.

Recommended top-level contract:

```python
"fluid_projection_count": fluid_projection_count,
"fluid_projection_after_feedback_count": fluid_projection_after_feedback_count,
"fluid_recomputed_after_feedback": fluid_projection_after_feedback_count > 0,
"feedback_closure_status": (
    "CLOSED_LOOP_RECOMPUTED_AFTER_FEEDBACK"
    if fluid_projection_after_feedback_count > 0
    else "OPEN_LOOP_OR_PREFEEDBACK_ONLY"
),
```

The old `fluid_recompute_count` field may be retained as a compatibility alias,
but it must not be the source of truth for the after-feedback claim.

Recommended per-step history contract:

```python
"fluid_recomputed": True,
"fluid_recomputed_after_feedback": feedback_available_before_projection,
"feedback_available_before_projection": feedback_available_before_projection,
```

## Required Flow Diagnostics

`easyfsi_history.csv` and any compare history rows should expose flow and
projection diagnostics that already exist in the JSON history entries:

- `local_velocity_peak_mps`
- `pressure_min_pa`
- `pressure_max_pa`
- `projection_l2`
- `projection_max_abs`
- `pre_projection_l2`
- `post_boundary_l2`
- `velocity_dirichlet_boundary_max_delta_mps`

The stage-check output must stop relying only on generic missing keys like
`final_residual` when the projection report contains the more relevant keys
above. It should print the available real projection diagnostic values instead
of leaving the residual line blank when useful keys exist.

## Test-First Plan

Follow a focused TDD loop:

1. Add or update source-level contract tests that fail on the current runner:
   - The loop must track `fluid_projection_count`.
   - The loop must track `fluid_projection_after_feedback_count`.
   - The loop must track `feedback_available_for_projection`.
   - Top-level `fluid_recomputed_after_feedback` must derive from
     `fluid_projection_after_feedback_count > 0`.
   - Per-step history must include `fluid_recomputed_after_feedback` and
     `feedback_available_before_projection`.
2. Add or update diagnostics tests that fail on current CSV behavior:
   - `build_history_rows()` must output the flow/projection diagnostic columns.
   - `write_diagnostics()` must write those columns into `easyfsi_history.csv`.
   - `build_displacement_compare_rows()` must carry flow/projection diagnostics
     into `displacement_compare.csv`.
   - `build_stage_check()` must include real projection keys such as
     `projection_l2` and `projection_max_abs` when present.
3. Run the focused tests and confirm they fail for the intended reason before
   production code changes.
4. Implement the minimum code changes needed to make the focused tests pass.
5. Run the focused tests again.
6. Run the broader relevant test slice for ANSYS vertical-flap runner,
   diagnostics, and closed-loop feedback contracts.

## Verification Commands

Prefer the reliable local Python interpreter if available:

```powershell
& 'D:\TOOL\Anaconda\python.exe' -m pytest tests\integration\test_ansys_vertical_flap_runner_loop_contract.py tests\tools\test_ansys_vertical_flap_diagnostics.py tests\integration\test_ansys_vertical_flap_closed_loop_feedback.py -q
```

If that interpreter is unavailable, use the active environment's `python -m
pytest` only after confirming it can import project modules.

If full-suite runtime is reasonable, run:

```powershell
& 'D:\TOOL\Anaconda\python.exe' -m pytest -q
```

## Artifact Honesty Requirements

Reports, tests, and final summary must say:

- Structural closed-loop recomputation is present.
- Physical validation still fails unless a newly generated artifact proves
  otherwise.
- Current known failure is `FAIL_FLOW`.
- Expected-failure tests for no-rebound and displacement magnitude should stay
  expected failures unless the artifacts truly pass.
- Local test results must not be described as GitHub CI results.

## PR / Push Criteria

The branch can be pushed when all of these are true:

- The detailed goal file is committed.
- Focused RED/GREEN evidence exists from tests.
- Runner closed-loop semantics are corrected.
- History/compare/stage diagnostics include flow projection fields.
- No case constants, tolerances, or `simulation_core/` solver behavior were
  changed for this goal.
- Relevant tests pass locally.
- The final response reports the pushed branch and commit hash.

## Future Work Explicitly Not Included

The next solver branch should convert the current structural recomputation into
physically feedback-conditioned fluid projection by applying updated marker
surface state to fluid-side no-slip or immersed-boundary constraints before
each projection.

That future branch should add fields such as:

- `fluid_feedback_constraint_active_cell_count`
- `fluid_feedback_constraint_marker_count`
- `no_slip_residual_before_mps`
- `no_slip_residual_after_mps`

That work is intentionally not part of this goal.
