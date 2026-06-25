# ANSYS Vertical-Flap Flow-Collapse Diagnostics Verification

Date: 2026-06-25

This record documents the post-repair EasyFsi diagnostic run for the ANSYS
vertical-flap case. The goal is not to claim Fluent parity. The goal is to
separate projection-only flow collapse from marker-feedback or solid-coupled
effects using committed scripts and committed data.

## Goal Reference

Detailed goal:

`docs/refactoring/ANSYS_VERTICAL_FLAP_FLOW_COLLAPSE_DIAGNOSTIC_GOAL_2026-06-25.md`

## Commands Run

```powershell
& 'D:\working\taichi\env\python.exe' -m tools.validation.print_ansys_vertical_flap_diagnostics --easyfsi-json validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step050_after_halfdomain_repair.json --fluent-tip-csv validation_runs\ansys_vertical_flap_fsi\official_web\fluent_tip_displacement_web_final.csv --output-dir validation_runs\ansys_vertical_flap_fsi\compare_after_halfdomain_repair
```

```powershell
& 'D:\working\taichi\env\python.exe' -m tools.validation.print_ansys_vertical_flap_diagnostics --easyfsi-json validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step001_preflow001_after_halfdomain_repair.json --output-dir validation_runs\ansys_vertical_flap_fsi\compare_preflow001_smoke
```

```powershell
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_preflow_only_sweep_after_halfdomain_repair.py
```

```powershell
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_flow_collapse_diagnostic_matrix.py
```

## Artifacts

Preflow-only sweep:

- `validation_runs/ansys_vertical_flap_fsi/flow_collapse_diagnostics/preflow_only_sweep/preflow_only_sweep.json`
- `validation_runs/ansys_vertical_flap_fsi/flow_collapse_diagnostics/preflow_only_sweep/preflow_only_sweep.csv`

Diagnostic matrix:

- `validation_runs/ansys_vertical_flap_fsi/flow_collapse_diagnostics/diagnostic_matrix/flow_collapse_diagnostic_matrix.json`
- `validation_runs/ansys_vertical_flap_fsi/flow_collapse_diagnostics/diagnostic_matrix/flow_collapse_diagnostic_matrix.csv`
- `validation_runs/ansys_vertical_flap_fsi/flow_collapse_diagnostics/diagnostic_matrix/flow_collapse_diagnostic_matrix_summary.md`

Regenerated existing summaries:

- `validation_runs/ansys_vertical_flap_fsi/compare_after_halfdomain_repair/easyfsi_summary.json`
- `validation_runs/ansys_vertical_flap_fsi/compare_after_halfdomain_repair/easyfsi_summary.csv`
- `validation_runs/ansys_vertical_flap_fsi/compare_after_halfdomain_repair/stage_check.md`
- `validation_runs/ansys_vertical_flap_fsi/compare_preflow001_smoke/easyfsi_summary.json`
- `validation_runs/ansys_vertical_flap_fsi/compare_preflow001_smoke/easyfsi_summary.csv`
- `validation_runs/ansys_vertical_flap_fsi/compare_preflow001_smoke/stage_check.md`

## Reporting Fixes

- `markers` now reports the actual visible marker total when available.
- `markers_per_face` is reported separately from the actual total.
- `markers_actual` is included in summary rows.
- `preflow_status` is reported separately from `preflow_converged`.
- A run with `preflow_steps_requested = 0` reports `preflow_status = not_requested`.

## Preflow-Only Sweep Result

The preflow-only sweep used a fixed solid with `step_count = 0`, so it does not
advance MPM and does not apply marker feedback. The p999 velocity still drops
after the first two projection steps:

| scenario | peak m/s | p999 m/s | projection max abs | status |
|---|---:|---:|---:|---|
| preflow_only_01 | 28.14044952392578 | 22.381813049316406 | 6400.0 | max_steps |
| preflow_only_02 | 31.581707000732422 | 25.043361543655397 | 6400.0 | max_steps |
| preflow_only_05 | 15.937021255493164 | 12.645165263175965 | 6400.0 | max_steps |
| preflow_only_10 | 10.596521377563477 | 10.29703426361084 | 6400.0 | max_steps |
| preflow_only_20 | 12.280338287353516 | 11.010656296730042 | 6400.0 | max_steps |

Primary observation:

`preflow-only p999 changed from 22.381813049316406 to 11.010656296730042 m/s`

Current best hypothesis:

`projection-only flow can collapse without solid advance or feedback`

## Diagnostic Matrix Result

The 10-step matrix compares feedback, projection solver choice, pressure reset,
and inlet reinitialization.

| scenario | final peak m/s | final p999 m/s | max p999 m/s | p999 ratio | projection max abs | flow status |
|---|---:|---:|---:|---:|---:|---|
| feedback_on_step10 | 10.901386260986328 | 10.311923771858233 | 24.666577596664457 | 0.4180524732889035 | 6400.0 | collapsed_after_initial_acceleration |
| feedback_off_step10 | 10.596521377563477 | 10.29703426361084 | 25.043361543655397 | 0.4111682150042489 | 6400.0 | collapsed_after_initial_acceleration |
| reset_pressure_every_step_step10 | 23.119972229003906 | 17.799482637405614 | 23.82972980308536 | 0.7469443751351734 | 6400.0 | collapsed_after_initial_acceleration |
| reinitialize_inlet_each_step_step10 | 29.445302963256836 | 22.98385433959964 | 22.98582481193546 | 0.9999142744560207 | 7967.42724609375 | within_official_range |
| solver_fv_cg_1080_step10 | 10.0 | 10.0 | 10.0 | 1.0 | 6400.0009765625 | below_official_range |
| solver_fv_cg_4096_step10 | 10.0 | 10.0 | 10.0 | 1.0 | 6400.0009765625 | below_official_range |

Primary observation:

`feedback_on final p999=10.311923771858233 m/s; feedback_off final p999=10.29703426361084 m/s`

Current best hypothesis:

`projection-only flow path is the primary suspect for flow collapse`

Next action:

`prioritize flow predictor, inlet driving, outlet, and projection solver path`

## Interpretation

The collapse is visible with no solid advance and no marker feedback, and the
feedback-on and feedback-off 10-step runs collapse to nearly the same p999
velocity. That rules against marker feedback as the primary cause in this
baseline.

Resetting pressure every step partially improves the final p999 velocity but
does not keep the flow in the official-like range. Reinitializing the inlet each
step keeps the final p999 velocity near 22.98 m/s and the peak velocity near
29.45 m/s, which points to missing sustained inlet/predictor driving or a related
flow-boundary/projection path issue.

## Scope Limits

- These runs use the EasyFsi solver path.
- These runs do not establish full ANSYS Fluent parity.
- These runs should not be used as an L3 50-step validation claim.
- The next implementation step should focus on the sustained flow predictor,
  inlet driving, outlet behavior, and projection path before tuning solid
  properties.
