# ANSYS Vertical-Flap Source Candidate STEP20 Verification

Date: 2026-06-25

This EasyFsi diagnostic checks whether previous final-row STEP20 source candidates also satisfy a temporal gate over their per-step history. It does not run 50 steps and does not claim Fluent parity.

## Goal Reference

`docs/refactoring/ANSYS_VERTICAL_FLAP_SOURCE_CANDIDATE_STEP20_GOAL_2026-06-25.md`

`docs/refactoring/ANSYS_VERTICAL_FLAP_SOURCE_CANDIDATE_TEMPORAL_GATE_GOAL_2026-06-25.md`

## Prior Remote State

Remote branch HEAD observed by GitHub connector before this goal:

```text
4d3a2c0966d0b5360a915e297e7a4ee50f583802
```

Implementation commit for the prior source/outlet balance step:

```text
02723dd54643f79da5fda6e3b9ed559eee22e993
```

STEP20 implementation commit before this temporal-gate pass:

```text
21d1eb1f4de1f6196af715c799222b1ce5c26d14
```

Pre-goal remote HEAD before this temporal-gate pass:

```text
d7f7e84b696c9390f45c1f9bf34a8efbfb7a3b42
```

## Commands Run

```powershell
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_source_candidate_step20_matrix.py --reclassify-existing
& 'D:\working\taichi\env\python.exe' -m py_compile validation_runs\ansys_vertical_flap_fsi\scripts\run_source_candidate_step20_matrix.py tests\integration\test_ansys_vertical_flap_source_candidate_step20_artifacts.py tests\integration\test_ansys_vertical_flap_source_candidate_temporal_gate_artifacts.py
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.integration.test_ansys_vertical_flap_source_candidate_step20_artifacts tests.integration.test_ansys_vertical_flap_source_candidate_temporal_gate_artifacts
& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_official_half_domain_archive_consistency tests.integration.test_ansys_vertical_flap_postrepair_artifacts tests.integration.test_ansys_vertical_flap_flow_collapse_artifacts tests.integration.test_ansys_vertical_flap_sustained_flow_driver_artifacts tests.integration.test_ansys_vertical_flap_source_outlet_balance_artifacts tests.integration.test_ansys_vertical_flap_source_candidate_step20_artifacts tests.integration.test_ansys_vertical_flap_source_candidate_temporal_gate_artifacts -v
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_case_metadata_matches_ansys_tutorial_boundaries_and_targets tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_formal_runner_uses_official_full_span_flap_box tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_formal_runner_places_both_streamwise_marker_faces tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_solid_substep_cfl_report_preserves_explicit_higher_count tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_preflow_controls_are_exposed_without_changing_default_smoke tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_fixed_solid_preflow_reports_diagnostics_without_mpm_advance tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_sustained_flow_driver_modes_are_explicit_and_default_safe tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_source_strength_factor_supports_constant_and_ramp_profiles tests.integration.test_ansys_vertical_flap_runner_loop_contract
& 'D:\working\taichi\env\python.exe' -m unittest tests.tools.test_ansys_vertical_flap_diagnostics -v
git diff --check
```

## Local Verification Status

- STEP20 existing artifacts were reclassified with temporal gates.
- No solver rerun was performed during temporal reclassification.
- No 50-step run was performed.
- `py_compile` passed for the STEP20 matrix script and STEP20 artifact tests.
- STEP20 artifact and temporal-gate contract tests passed: 4 tests.
- Archive/artifact consistency tests passed: 17 tests.
- Source-level runner contract tests passed: 12 tests.
- Diagnostics unit tests passed: 11 tests.
- `git diff --check` passed with Windows LF-to-CRLF warnings only.
- Changed-file credential scan found no sensitive credential values.

## Remote CI Query Status

The temporal-gate implementation commit was pushed to
`origin/solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`:

```text
b6da51f9bbbdf498ccc5ab56ca45ed2ff4e93ed9
```

Remote GitHub Actions status could not be queried from this workstation because
GitHub CLI is not authenticated:

```text
To get started with GitHub CLI, please run:  gh auth login
Alternatively, populate the GH_TOKEN environment variable with a GitHub API authentication token.
```

## Result

best_candidate = source_0p75_ramp5_step20

best_final_gate_candidate = source_0p75_ramp5_step20

best_temporal_candidate = none

nearest_non_candidate = source_0p70_constant_step20

candidate_status = no_temporal_candidate

temporal_candidate_status = no_temporal_candidate

temporal_best_candidate_status = none

temporal_candidate_count = 0

best_candidate_history_csv = validation_runs/ansys_vertical_flap_fsi/source_candidate_step20_diagnostics/best_candidate_step20_history.csv

mass_balance_primary_metric = velocity_outlet_flux_ratio

pressure_outlet_flux_interpretation = diagnostic_only_until_pressure_outlet_model_reviewed

primary_observation = best_candidate=source_0p75_ramp5_step20; p999 range 11.175716391563588-25.300280584335876 m/s; peak range 12.719998359680176-32.88761520385742 m/s; velocity_outlet_flux_ratio range 0.0-1.0982373626194526

current_best_hypothesis = final-row source candidates exist, but none satisfies the STEP20 temporal gate

next_action = stop before 50-step; refine source/outlet model or run STEP30 temporal matrix

## Scope Limits

- No 50-step run was performed.
- No solid parameters were tuned.
- No Fluent parity claim is made.
- Full-field reinitialize rows are diagnostic-only and excluded from candidate selection.
- `sustained_inlet_predictor` is not treated as a real predictor path.
- A passing STEP20 temporal gate still requires STEP30 review before any coarse 50-step flow-gate run.
