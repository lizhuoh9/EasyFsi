# ANSYS Vertical Flap Guard PR Integration Goal - 2026-06-25

## Source Branches

- Repository: `lizhuoh9/EasyFsi`
- Working directory: `D:\working\squid robot\simulation\src\reference\papers\HIBM-MPM\refactored`
- Guard implementation branch: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Guard implementation HEAD before integration: `8e17e08e517396f1598d32680014c1169b8e5d5a`
- Runtime proof branch: `solver/ansys-vertical-flap-feedback-three-step-clear-smoke-2026-06-25`
- Runtime proof HEAD before integration: `faff9746cae370c6dc379b7502eb970c76149a93`
- Final PR-ready branch to push: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`

## Objective

Prepare the ANSYS vertical-flap feedback projection guard work for a single
reviewable PR by integrating the test-only 3-step runtime stale-clear proof into
the guard implementation branch. The final pushed guard branch must contain:

- runner-level feedback-conditioned projection guard implementation,
- stale marker-owned velocity Dirichlet constraint clearing,
- target-assembly vs post-projection no-slip residual diagnostics,
- feedback constraint obstacle/non-obstacle/projection-participating counts,
- diagnostics propagation through history CSV, displacement compare CSV, and
  `stage_check.md`,
- real 2-step runtime smoke proving step-2 feedback consumption,
- real 3-step runtime smoke proving step-3 clearing of constraints written on
  step 2.

This task is an integration and verification task, not a new physics or artifact
generation task.

## Required Git Shape

Use the attachment's recommended strategy A:

1. Keep the test-only 3-step runtime proof commit.
2. Fast-forward or otherwise integrate it into
   `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.
3. Push that guard branch to `origin`.
4. Do not rewrite already-pushed public history.
5. Do not squash away the RED/GREEN trail from the guard branch.

The final review branch should be:

```text
base: solver/ansys-vertical-flap-feedback-conditioned-fluid-projection-2026-06-25
head: solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25
```

The already-created stacked runtime branch may remain on the remote, but the
preferred PR should be the integrated guard branch.

## Explicit Non-Goals

- Do not modify `simulation_core/`.
- Do not modify production runner code unless a verification run exposes a real
  regression after integration.
- Do not change ANSYS case constants, tolerances, material parameters, damping,
  support radius, or reference values.
- Do not regenerate, overwrite, or backfill old 50-step artifacts.
- Do not create new feedback-conditioned artifacts in this task.
- Do not claim GitHub Actions passed unless a real workflow run exists for the
  pushed SHA.
- Do not claim ANSYS physical validation is fixed. The known previous physical
  status remains `FAIL_FLOW` until a new artifact branch proves otherwise.

## Required Validation

Run the focused feedback guard target locally:

```powershell
& 'D:\working\taichi\env\python.exe' -m pytest tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection.py tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection_runtime.py tests\tools\test_ansys_vertical_flap_diagnostics.py -q
```

Run the ANSYS vertical-flap slice locally:

```powershell
& 'D:\working\taichi\env\python.exe' -m pytest tests\cases\test_ansys_vertical_flap_fsi.py tests\integration\test_ansys_vertical_flap_runner_loop_contract.py tests\tools\test_ansys_vertical_flap_diagnostics.py tests\integration\test_ansys_vertical_flap_closed_loop_feedback.py tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection.py tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection_runtime.py -k "not matches_reference_displacement_tolerance" -q
```

Run syntax and whitespace checks:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile tests\integration\test_ansys_vertical_flap_feedback_conditioned_projection_runtime.py benchmarks\official\solid_mpm_fsi_runner.py tools\validation\print_ansys_vertical_flap_diagnostics.py
git diff --check
```

## PR Body Notes To Preserve

The eventual PR body should describe validation as local only:

```text
Local validation:
- runtime smoke: 2 passed
- focused guard target: 19 passed
- ANSYS vertical-flap slice: 35 passed, 1 deselected, 2 xfailed
- py_compile passed
- git diff --check passed
- only pytest cache permission warning observed
```

It should also say:

```text
No GitHub Actions workflow run exists for this SHA.
No old 50-step artifacts were regenerated or overwritten.
This PR proves feedback projection guards and runtime clearing behavior; it
does not claim ANSYS physical validation passed.
```

## Done Criteria

- This detailed goal file is committed.
- A short goal references this file.
- The guard branch contains the 3-step runtime proof commit.
- The guard branch contains this integration goal commit.
- Required local validation passes.
- The integrated guard branch is pushed to `origin`.
- Final report includes the final guard-branch SHA, branch name, validation
  results, and artifact-honesty note.
