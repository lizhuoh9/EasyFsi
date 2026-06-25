# Verification after half-domain repair commit 10bdb7c8

Date: 2026-06-25

Commit reviewed before this verification:
`10bdb7c8e2600c1f772c12509ffef3e2e4b5fa46`

Branch:
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`

Interpreter:
`D:\working\taichi\env\python.exe`

## Why this record exists

The commit `10bdb7c8...` repaired the official half-domain archive schema,
formal runner full-span geometry, dual streamwise marker faces, and solid CFL
substep reporting. GitHub did not show a workflow run for that commit, so the
evidence was local-only. This record preserves the local verification facts and
the first post-repair physical-validation run results in the repository.

## Prior local verification reported for 10bdb7c8

These were the local checks reported immediately after the commit was created:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile cases\ansys_vertical_flap_fsi.py benchmarks\official\solid_mpm_fsi_runner.py validation\ansys-fluent-official-half-domain-hibm-mpm-2026-06-25\scripts\run_official_fluent_half_domain_hibm_mpm_4x320x640.py validation\ansys-fluent-official-half-domain-hibm-mpm-2026-06-25\scripts\render_official_half_domain_mirrored_pipe_style.py tests\cases\test_ansys_vertical_flap_fsi.py tests\integration\test_ansys_official_half_domain_archive_consistency.py
& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_official_half_domain_archive_consistency -v
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_case_metadata_matches_ansys_tutorial_boundaries_and_targets tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_formal_runner_uses_official_full_span_flap_box tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_formal_runner_places_both_streamwise_marker_faces tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_solid_substep_cfl_report_preserves_explicit_higher_count tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_thin_wall_probe_reach_tracks_refined_streamwise_spacing
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.integration.test_ansys_official_half_domain_archive_consistency tests.integration.test_ansys_vertical_flap_runner_loop_contract
git diff --check
```

Observed outcomes:

- archive consistency tests: `4/4` passed.
- focused ANSYS case/source tests: `5/5` passed.
- archive consistency plus runner loop contract: `8/8` passed.
- full `tests.cases.test_ansys_vertical_flap_fsi` run timed out at 200 seconds
  after reaching the existing runtime-heavy expected-failure 50-step physical
  test.
- integration discovery for `*ansys*vertical*flap*.py` timed out at 200 seconds
  after reaching the existing two-step runtime test.
- `git diff --check` emitted Windows line-ending warnings only; no whitespace
  error lines were reported.
- local pre-push hook output on push:
  `[ECC pre-push] No supported checks found in this repository. Skipping.`

## New CI/reviewability change

This verification stage adds:

```text
.github/workflows/ansys-vertical-flap-validation.yml
```

The workflow intentionally runs source-level and archive-level ANSYS checks and
does not require a CUDA GPU for a 50-step runtime solve.

## Repaired 50-step coarse smoke

Command surface:

```powershell
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_step050_after_halfdomain_repair.py
```

Artifacts:

- `validation_runs/ansys_vertical_flap_fsi/easyfsi/easyfsi_step050_after_halfdomain_repair.json`
- `validation_runs/ansys_vertical_flap_fsi/easyfsi/easyfsi_step050_after_halfdomain_repair_process.json`
- `validation_runs/ansys_vertical_flap_fsi/compare_after_halfdomain_repair/easyfsi_summary.json`
- `validation_runs/ansys_vertical_flap_fsi/compare_after_halfdomain_repair/easyfsi_history.csv`
- `validation_runs/ansys_vertical_flap_fsi/compare_after_halfdomain_repair/stage_check.md`
- `validation_runs/ansys_vertical_flap_fsi/compare_after_halfdomain_repair/displacement_compare.csv`

Process summary:

```json
{
  "status": "completed",
  "history_rows": 50,
  "elapsed_s": 229.60347210001783,
  "local_velocity_peak_mps": 10.68154239654541,
  "fluid_speed_p999_mps": 10.1898175573349,
  "max_displacement_m": 1.1592121154535562e-05,
  "solid_substeps_selected": 1600,
  "solid_estimated_cfl": 0.03064175637198511
}
```

Diagnostic summary:

```json
{
  "status": "FAIL_FLOW",
  "steps": 50,
  "velocity_peak_mps": 10.68154239654541,
  "velocity_p999_mps": 10.1898175573349,
  "velocity_peak_relerr": 0.6198739360659996,
  "marker_force_z_N": -7.83371102100297e-05,
  "scatter_action_reaction_residual_N": 1.8253906948535865e-12,
  "tip_dz_final_m": -4.257075488567352e-06,
  "tip_dz_sign_violation_count": 8,
  "tip_dz_monotonic_violation_count": 23,
  "disp_relerr": 0.7727035067738125
}
```

Interpretation:

- The repaired formal runner completed 50 coarse steps.
- Interface/scatter/root gates stayed structurally valid:
  stress invalid `0`, scatter invalid `0`, feedback invalid `0`, root
  displacement `0`, streamwise marker force negative.
- Flow remains the active failing gate. Velocity peak and p999 remain far below
  the official web range `20-29 m/s`.
- The next solver investigation should prioritize flow solver, boundary
  condition, obstacle/outlet, and pressure projection behavior before changing
  the solid model.

## Preflow smoke

Command surface:

```powershell
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_step001_preflow001_after_halfdomain_repair.py
& 'D:\working\taichi\env\python.exe' -m tools.validation.print_ansys_vertical_flap_diagnostics --easyfsi-json validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step001_preflow001_after_halfdomain_repair.json --output-dir validation_runs\ansys_vertical_flap_fsi\compare_preflow001_smoke
```

Artifacts:

- `validation_runs/ansys_vertical_flap_fsi/easyfsi/easyfsi_step001_preflow001_after_halfdomain_repair.json`
- `validation_runs/ansys_vertical_flap_fsi/compare_preflow001_smoke/easyfsi_summary.json`
- `validation_runs/ansys_vertical_flap_fsi/compare_preflow001_smoke/easyfsi_history.csv`
- `validation_runs/ansys_vertical_flap_fsi/compare_preflow001_smoke/stage_check.md`

Observed outcome:

- `preflow_steps_requested = 1`
- `preflow_steps_completed = 1`
- `preflow_history` rows: `1`
- preflow row records `solid_fixed = true` and `solid_advanced = false`
- subsequent FSI history rows: `1`
- `solid_substeps_selected = 1600`

This validates the preflow reporting path. It is not a full preflow sweep.

## Local verification for this change

Commands run from the repository root:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile cases\ansys_vertical_flap_fsi.py benchmarks\official\solid_mpm_fsi_runner.py tools\validation\print_ansys_vertical_flap_diagnostics.py tests\cases\test_ansys_vertical_flap_fsi.py tests\integration\test_ansys_official_half_domain_archive_consistency.py tests\integration\test_ansys_vertical_flap_postrepair_artifacts.py tests\tools\test_ansys_vertical_flap_diagnostics.py validation_runs\ansys_vertical_flap_fsi\scripts\run_step050_after_halfdomain_repair.py validation_runs\ansys_vertical_flap_fsi\scripts\run_step001_preflow001_after_halfdomain_repair.py
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_case_metadata_matches_ansys_tutorial_boundaries_and_targets tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_formal_runner_uses_official_full_span_flap_box tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_formal_runner_places_both_streamwise_marker_faces tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_solid_substep_cfl_report_preserves_explicit_higher_count tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_preflow_controls_are_exposed_without_changing_default_smoke tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_fixed_solid_preflow_reports_diagnostics_without_mpm_advance tests.integration.test_ansys_official_half_domain_archive_consistency tests.integration.test_ansys_vertical_flap_runner_loop_contract tests.integration.test_ansys_vertical_flap_postrepair_artifacts tests.tools.test_ansys_vertical_flap_diagnostics
git diff --check
```

Observed outcomes:

- `py_compile`: passed.
- focused ANSYS validation tests: `28/28` passed.
- `git diff --check`: Windows line-ending warnings only; no whitespace errors.

## Remaining scope

This record does not claim Fluent parity. It establishes a reviewable baseline:

- CI/source/archive checks exist.
- repaired 50-step coarse run completed.
- repaired 50-step coarse run still fails at the flow gate.
- fixed-solid preflow control and diagnostics are present and smoke-tested.

Future work should inspect pressure projection, boundary condition, obstacle,
and outlet behavior before spending time on high-resolution L3 50-step runs.
