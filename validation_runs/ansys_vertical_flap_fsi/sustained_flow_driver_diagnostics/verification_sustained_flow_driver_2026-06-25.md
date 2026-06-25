# ANSYS Vertical-Flap Sustained Flow Driver Verification

Date: 2026-06-25

This record documents the EasyFsi 10-step sustained-flow driver diagnostic for
the ANSYS vertical-flap formal runner. It is not a Fluent parity claim and it is
not an L3 50-step run.

## Goal Reference

Detailed goal:

`docs/refactoring/ANSYS_VERTICAL_FLAP_SUSTAINED_FLOW_DRIVER_GOAL_2026-06-25.md`

## Commands Run

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile cases\ansys_vertical_flap_fsi.py benchmarks\official\solid_mpm_fsi_runner.py tools\validation\print_ansys_vertical_flap_diagnostics.py validation_runs\ansys_vertical_flap_fsi\scripts\run_sustained_flow_driver_matrix.py tests\cases\test_ansys_vertical_flap_fsi.py tests\tools\test_ansys_vertical_flap_diagnostics.py tests\integration\test_ansys_vertical_flap_sustained_flow_driver_artifacts.py
```

```powershell
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_preflow_controls_are_exposed_without_changing_default_smoke tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_fixed_solid_preflow_reports_diagnostics_without_mpm_advance tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_diagnostic_flow_controls_are_explicit_and_default_safe tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_sustained_flow_driver_modes_are_explicit_and_default_safe tests.tools.test_ansys_vertical_flap_diagnostics
```

```powershell
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_sustained_flow_driver_matrix.py
```

Runtime for the full 7-scenario matrix was about 994 seconds.

## Artifacts

- `validation_runs/ansys_vertical_flap_fsi/sustained_flow_driver_diagnostics/sustained_flow_driver_matrix.json`
- `validation_runs/ansys_vertical_flap_fsi/sustained_flow_driver_diagnostics/sustained_flow_driver_matrix.csv`
- `validation_runs/ansys_vertical_flap_fsi/sustained_flow_driver_diagnostics/sustained_flow_driver_matrix_summary.md`

## Matrix Result

Primary observation:

```text
projection_only final p999=10.311924845695513 m/s; diagnostic_reinitialize final p999=22.983848978042634 m/s; best_sustained=sustained_inlet_predictor_feedback_off_step10 final p999=32.11051559448242 m/s
```

Current best hypothesis:

```text
sustained flow driver restores p999 but over-accelerates; refine source strength, outlet compatibility, and predictor coupling before any 50-step run
```

Next action:

```text
refine source strength, outlet compatibility, and predictor coupling before any 50-step run
```

## Key Rows

| scenario | mode | final peak m/s | final p999 m/s | source flux m3/s | projection max abs | flow status |
|---|---|---:|---:|---:|---:|---|
| projection_only_step10 | projection_only | 10.901419639587402 | 10.311924845695513 | 0.0 | 6400.0 | collapsed_after_initial_acceleration |
| reinitialize_inlet_each_step_step10 | reinitialize_inlet_each_step_diagnostic | 29.44529914855957 | 22.983848978042634 | 0.0 | 7967.439453125 | within_official_range |
| sustained_boundary_inlet_step10 | sustained_boundary_inlet | 10.901429176330566 | 10.311926753044146 | 0.0 | 6400.0 | collapsed_after_initial_acceleration |
| sustained_volume_source_inlet_step10 | sustained_volume_source_inlet | 41.35008239746094 | 31.845914274216423 | 0.0006000002613291144 | 814.5744018554688 | above_official_range |
| sustained_inlet_predictor_step10 | sustained_inlet_predictor | 41.369354248046875 | 31.853287534714475 | 0.0006000002613291144 | 1298.1138916015625 | above_official_range |
| sustained_inlet_predictor_feedback_off_step10 | sustained_inlet_predictor | 40.45790481567383 | 32.11051559448242 | 0.0006000002613291144 | 17.003982543945312 | above_official_range |
| reset_pressure_every_step_step10 | projection_only | 23.119979858398438 | 17.799482412338477 | 0.0 | 6400.0 | collapsed_after_initial_acceleration |

## Interpretation

The new non-full-reset source/predictor driver changes the failure mode. It no
longer collapses to the inlet-scale `10 m/s` p999 velocity: p999 reaches
`31-32 m/s`. That supports the prior diagnosis that the first failure gate is
missing sustained inlet/source/predictor driving rather than solid parameters or
marker feedback.

However, the current source strength over-accelerates the flow. The final peak
velocity reaches about `40-41 m/s`, above the intended coarse official-web
sanity range. The sustained driver therefore is useful as a diagnostic
implementation, but it is not yet a validated physical driver for a 50-step run.

Refreshing only the zmax Dirichlet boundary does not fix collapse. It remains
near projection-only behavior, with final p999 about `10.31 m/s`. That indicates
that a persistent velocity boundary alone is insufficient in the current formal
runner; a source/outlet/predictor-compatible path is needed.

The feedback-off sustained predictor row remains high, with final p999 about
`32.11 m/s`, so marker feedback is still not the primary explanation for the
flow recovery or over-acceleration.

## Errors And Issues Found

- `fluid.project(...)` did not include source/outlet flux diagnostics in the
  returned projection report. The runner now explicitly merges
  `fluid.pressure_outlet_fv_flux_report(dt_s=config.dt_s)` into the formal
  runner projection report.
- The first matrix summary wording would have treated p999 recovery alone as a
  success. The matrix decision logic was tightened to require the sustained row
  to stay in the intended velocity range before recommending any 50-step run.

## Scope Limits

- These artifacts use the EasyFsi formal runner.
- The diagnostic full-field reinitialize path remains diagnostic-only.
- The sustained source/predictor path does not reset the full velocity field.
- No solid material, damping, marker count, support radius, or feedback weight
  was tuned.
- No Fluent parity claim is made.
- No 50-step or L-level validation claim is made.

## Remote CI Evidence

After pushing commit `6e5e337e36962fc541acb887537129a703de578d`, I attempted to
query GitHub Actions with:

```powershell
gh run list --repo lizhuoh9/EasyFsi --branch solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25 --limit 10 --json databaseId,headSha,status,conclusion,url,workflowName,createdAt,updatedAt
```

The query did not return run evidence because GitHub CLI is not authenticated in
this environment:

```text
To get started with GitHub CLI, please run: gh auth login
Alternatively, populate the GH_TOKEN environment variable with a GitHub API authentication token.
```

Therefore the committed local verification commands and artifacts above remain
the current evidence basis from this environment. The workflow is configured to
run on `solver/**` branch pushes and also supports `workflow_dispatch`.
