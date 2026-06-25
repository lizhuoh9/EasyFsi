# ANSYS Vertical-Flap Stage-Check Residual Goal - 2026-06-25

## Source Context

This goal is based on the follow-up review of branch
`solver/ansys-vertical-flap-closed-loop-runner-2026-06-25` at HEAD
`44a231da61c77f9ffd72a349ac891c2a9484215d`.

The review concluded that the previous diagnostic and semantic correction is
basically ready for PR:

- `fluid_projection_count` and `fluid_projection_after_feedback_count` are now
  reported separately.
- Step 1 is correctly treated as pre-feedback projection.
- Step 2 and later are correctly treated as after-feedback projection.
- `history` contains `fluid_recomputed_after_feedback` and
  `feedback_available_before_projection`.
- `easyfsi_history.csv`, `displacement_compare.csv`, and `stage_check.md`
  expose flow/projection diagnostics.
- Artifact status remains honestly `FAIL_FLOW`.

The only PR-blocking polish item is that `stage_check.md` still prints an empty
line:

```text
projection_final_residual =
```

even though the report now prints real diagnostics such as `projection_l2` and
`projection_max_abs` immediately below it.

## Primary Objective

Remove the misleading blank `projection_final_residual =` line from generated
stage-check output when the current projection report does not actually provide
a final-residual field.

This is a report-quality patch only. It must not alter solver behavior,
physical validation thresholds, case constants, or `simulation_core/`.

## Required Behavior

When `_projection_residual(projection)` returns a real value:

- `stage_check.md` should include:

```text
projection_final_residual = <value>
```

When `_projection_residual(projection)` returns an empty value:

- `stage_check.md` must omit the `projection_final_residual =` line entirely.
- `stage_check.md` must still include the real projection diagnostics that are
  present in the projection report, including `projection_l2` and
  `projection_max_abs`.

The preferred implementation is to build the `[FLOW_ONLY]` section as a list of
lines and append `projection_final_residual` only when `_projection_residual()`
returns a nonblank value.

## Test-First Plan

1. Add a diagnostics test that fails against current behavior:
   - Create or reuse a fixture report whose `flow_projection_report` contains
     `projection_l2` and `projection_max_abs` but no `final_residual`.
   - Generate `stage_check`.
   - Assert that `projection_final_residual =` is not present.
   - Assert that `projection_l2 =` and `projection_max_abs =` are present.
2. Run the focused tools diagnostics test and confirm the failure is caused by
   the blank residual line.
3. Patch `tools/validation/print_ansys_vertical_flap_diagnostics.py`.
4. Regenerate committed ANSYS compare artifacts from the existing
   `easyfsi_step050.json`.
5. Run focused tests and the ANSYS slice used in the previous goal.

## Validation Commands

Use the known working interpreter:

```powershell
& 'D:\working\taichi\env\python.exe' -m pytest tests\tools\test_ansys_vertical_flap_diagnostics.py -q
```

Then run the relevant ANSYS slice, excluding only the long 50-step physical
expected-failure run:

```powershell
& 'D:\working\taichi\env\python.exe' -m pytest tests\cases\test_ansys_vertical_flap_fsi.py tests\integration\test_ansys_vertical_flap_runner_loop_contract.py tests\tools\test_ansys_vertical_flap_diagnostics.py tests\integration\test_ansys_vertical_flap_closed_loop_feedback.py -k "not matches_reference_displacement_tolerance" -q
```

## Scope Boundaries

In scope:

- `tools/validation/print_ansys_vertical_flap_diagnostics.py`
- `tests/tools/test_ansys_vertical_flap_diagnostics.py`
- Regenerated files under `validation_runs/ansys_vertical_flap_fsi/compare`
- This goal file

Out of scope:

- Creating the PR in this turn unless explicitly requested later.
- Changing runner feedback semantics again.
- Adding feedback-conditioned fluid projection.
- Changing solver formulas.
- Changing case constants, tolerances, material, damping, or support radius.
- Editing `simulation_core/`.
- Claiming GitHub CI passed when no workflow run exists for the pushed SHA.
- Claiming physical validation passed while the artifact remains `FAIL_FLOW`.

## Done Criteria

- A RED test proves the blank residual line exists before the fix.
- The diagnostics tool omits blank `projection_final_residual` lines.
- Regenerated `stage_check.md` contains no blank residual line.
- `stage_check.md` still contains `projection_l2`, `projection_max_abs`,
  `pre_projection_l2`, `post_boundary_l2`, and
  `velocity_dirichlet_boundary_max_delta_mps`.
- Focused diagnostics tests pass.
- Relevant ANSYS test slice passes with expected failures preserved.
- Work is committed and pushed to
  `origin/solver/ansys-vertical-flap-closed-loop-runner-2026-06-25`.
