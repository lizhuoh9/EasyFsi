# ANSYS Vertical-Flap Source Candidate STEP20 Verification

Date: 2026-06-25

This EasyFsi diagnostic checks whether previous final-row STEP20 source candidates also satisfy a temporal gate over their per-step history. It does not run 50 steps and does not claim Fluent parity.

## Goal Reference

`docs/refactoring/ANSYS_VERTICAL_FLAP_PREFLOW_RELEASE_COUPLING_GATE_GOAL_2026-06-25.md`

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
& 'D:\working\taichi\env\python.exe' -m unittest tests.tools.test_ansys_vertical_flap_temporal_gate -v
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.integration.test_ansys_vertical_flap_source_candidate_step20_artifacts tests.integration.test_ansys_vertical_flap_source_candidate_temporal_gate_artifacts
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_fixed_solid_source_temporal_matrix.py
git diff --check
```

## Local Verification Status

- STEP20 existing artifacts were reclassified with separated flow, coupling, and promotion gates.
- `py_compile` passed.
- STEP20 artifact and temporal-gate contract tests passed.
- Fixed-solid STEP30 flow temporal diagnostics were generated.
- Archive/artifact consistency tests passed.
- Source-level runner contract tests passed: 12 tests.
- Diagnostics unit tests passed: 11 tests.
- `git diff --check` passed with Windows LF-to-CRLF warnings only.
- Changed-file credential scan found no sensitive credential values.

## Result

best_candidate = source_0p75_ramp5_step20

best_final_gate_candidate = source_0p75_ramp5_step20

best_temporal_candidate = none

best_flow_temporal_candidate = source_0p80_ramp2_step20

best_combined_temporal_candidate = none

promotion_candidate = none

promotion_candidate_status = no_promotion_candidate

diagnostic_fallback_candidate = source_0p75_ramp5_step20

nearest_non_candidate = source_0p70_constant_step20

candidate_status = no_temporal_candidate

temporal_candidate_status = no_temporal_candidate

temporal_best_candidate_status = none

temporal_candidate_count = 0

flow_temporal_candidate_count = 4

best_candidate_history_csv = validation_runs/ansys_vertical_flap_fsi/source_candidate_step20_diagnostics/best_candidate_step20_history.csv

mass_balance_primary_metric = velocity_outlet_flux_ratio

pressure_outlet_flux_interpretation = diagnostic_only_until_pressure_outlet_model_reviewed

primary_observation = best_candidate=source_0p75_ramp5_step20; p999 range 11.175716391563588-25.300280584335876 m/s; peak range 12.719998359680176-32.88761520385742 m/s; velocity_outlet_flux_ratio range 0.0-1.0982373626194526

current_best_hypothesis = source/outlet flow can satisfy the STEP20 flow temporal gate, but coupled force/tip settling still blocks promotion

next_action = stop before 50-step; run fixed-solid STEP30 flow diagnostics and coupling-settling review

## Scope Limits

- No 50-step run was performed.
- No solid parameters were tuned.
- No Fluent parity claim is made.
- No promotion-ready combined temporal candidate is claimed unless `promotion_candidate` is not `none`.
- Full-field reinitialize rows are diagnostic-only and excluded from candidate selection.
- `sustained_inlet_predictor` is not treated as a real predictor path.
- A passing STEP20 temporal gate still requires STEP30 review before any coarse 50-step flow-gate run.
