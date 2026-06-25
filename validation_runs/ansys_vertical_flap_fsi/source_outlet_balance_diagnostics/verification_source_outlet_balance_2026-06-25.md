# ANSYS Vertical-Flap Source/Outlet Balance Verification

Date: 2026-06-25

This EasyFsi diagnostic calibrates source strength and outlet balance for the ANSYS vertical-flap formal runner. It does not run 50 steps and does not claim Fluent parity.

## Goal Reference

`docs/refactoring/ANSYS_VERTICAL_FLAP_SOURCE_OUTLET_BALANCE_GOAL_2026-06-25.md`

## Commands Run

```powershell
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_source_outlet_balance_matrix.py
& 'D:\working\taichi\env\python.exe' -m py_compile cases\ansys_vertical_flap_fsi.py benchmarks\official\solid_mpm_fsi_runner.py tools\validation\print_ansys_vertical_flap_diagnostics.py validation_runs\ansys_vertical_flap_fsi\scripts\run_source_outlet_balance_matrix.py tests\cases\test_ansys_vertical_flap_fsi.py tests\tools\test_ansys_vertical_flap_diagnostics.py tests\integration\test_ansys_vertical_flap_source_outlet_balance_artifacts.py
& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_official_half_domain_archive_consistency tests.integration.test_ansys_vertical_flap_postrepair_artifacts tests.integration.test_ansys_vertical_flap_flow_collapse_artifacts tests.integration.test_ansys_vertical_flap_sustained_flow_driver_artifacts tests.integration.test_ansys_vertical_flap_source_outlet_balance_artifacts -v
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_case_metadata_matches_ansys_tutorial_boundaries_and_targets tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_formal_runner_uses_official_full_span_flap_box tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_formal_runner_places_both_streamwise_marker_faces tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_solid_substep_cfl_report_preserves_explicit_higher_count tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_preflow_controls_are_exposed_without_changing_default_smoke tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_fixed_solid_preflow_reports_diagnostics_without_mpm_advance tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_sustained_flow_driver_modes_are_explicit_and_default_safe tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_source_strength_factor_supports_constant_and_ramp_profiles tests.integration.test_ansys_vertical_flap_runner_loop_contract
& 'D:\working\taichi\env\python.exe' -m unittest tests.tools.test_ansys_vertical_flap_diagnostics -v
git diff --check
```

## Local Verification Status

- Matrix generation completed and wrote source/outlet artifacts.
- `py_compile` passed.
- Archive/artifact consistency tests passed: 13 tests.
- Source-level runner contract tests passed: 12 tests.
- Diagnostics unit tests passed: 11 tests.
- Focused source/outlet/parser/diagnostics tests passed: 16 tests.
- `git diff --check` passed with Windows LF-to-CRLF warnings only.
- Changed-file credential scan found no API key, password, private-key, or GitHub token patterns.

## Remote CI Query Status

The implementation commit `02723dd` was pushed to
`origin/solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.

Remote GitHub Actions status could not be queried from this workstation because
GitHub CLI is not authenticated:

```text
To get started with GitHub CLI, please run:  gh auth login
Alternatively, populate the GH_TOKEN environment variable with a GitHub API authentication token.
```

## Result

best_candidate = source_strength_0p75_step10

candidate_status = candidate_found

primary_observation = source_strength range 0.2-1.0; best_candidate=source_strength_0p75_step10; p999 range 10.023739116668702-31.845914394379385 m/s; peak range 10.116854667663574-41.35007095336914 m/s

current_best_hypothesis = a non-full-reset source strength candidate satisfies the 10-step flow gate

next_action = run a 20-step candidate check before any 50-step run

## Outlet Balance

outlet_primary_observation = source_strength range 0.75-1.0; best_candidate=selected_source_strength_step10; p999 range 10.311924845695513-25.899071310043922 m/s; peak range 10.901412963867188-33.63189697265625 m/s

Both pressure outlet flux and velocity outlet flux are recorded. They must not be conflated when judging mass balance.

## Scope Limits

- No 50-step run was performed.
- No solid parameters were tuned.
- No Fluent parity claim is made.
- Full-field reinitialize rows are diagnostic-only and excluded from candidate selection.
