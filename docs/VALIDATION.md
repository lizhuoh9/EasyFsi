# Validation Matrix

Run these commands from the repository root with the trusted Taichi
interpreter on this machine:

```powershell
$python = 'D:\working\taichi\env\python.exe'
```

## Structure-Only Gating

```powershell
$compileTargets = @('scripts\validate_structure.py')
$compileTargets += Get-ChildItem -Path `
  simulation_core, `
  simulation_core\fluids, `
  simulation_core\coupling, `
  simulation_core\coupling\hibm_mpm, `
  simulation_core\solids, `
  simulation_core\solids\mooney_shell, `
  simulation_core\geometry_tools, `
  simulation_core\materials, `
  simulation_core\diagnostics `
  -Filter *.py -File | Select-Object -ExpandProperty FullName
& $python -m py_compile @compileTargets
& $python scripts\validate_structure.py
& $python -m unittest tests.solvers.test_simulation_core_facades -v
& $python -m unittest discover -s tests\contracts -p "test_*.py" -v
& $python -m unittest discover -s tests\integration -p "test_*.py" -v
& $python -m unittest discover -s tests\tools -p "test_*.py" -v
& $python -m unittest tests.cases.test_squid_latest_core_config tests.cases.test_squid_package_exports -v
```

## Optional Light Solver Checks

```powershell
& $python -m unittest tests.solvers.test_validation -v
& $python -m unittest tests.solvers.test_time_stepping -v
& $python -m unittest tests.solvers.test_hyperelastic_ecoflex -v
& $python -m unittest tests.solvers.test_fluid -v
```

## ANSYS Vertical-flap Solver-validation Diagnostics

Use this after writing an EasyFsi JSON report with
`run_simulation.py ansys-vertical-flap-fsi --steps N --json`. The diagnostic
script records the current flow/interface/scatter/solid/feedback layer status
without changing solver behavior:

```powershell
& $python -m tools.validation.print_ansys_vertical_flap_diagnostics `
  --easyfsi-json validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step050.json `
  --output-dir validation_runs\ansys_vertical_flap_fsi\compare
```

If a Fluent tip-displacement report file is available, add:

```powershell
--fluent-tip-csv validation_runs\ansys_vertical_flap_fsi\fluent\fluent_tip_displacement.csv
```

For the post-half-domain-repair baseline, run the committed coarse 50-step
smoke wrapper and diagnostics:

```powershell
& $python validation_runs\ansys_vertical_flap_fsi\scripts\run_step050_after_halfdomain_repair.py
& $python -m tools.validation.print_ansys_vertical_flap_diagnostics `
  --easyfsi-json validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step050_after_halfdomain_repair.json `
  --fluent-tip-csv validation_runs\ansys_vertical_flap_fsi\official_web\fluent_tip_displacement_web_final.csv `
  --output-dir validation_runs\ansys_vertical_flap_fsi\compare_after_halfdomain_repair
```

The case also exposes fixed-solid preflow controls for diagnosing whether a
flow field is established before the MPM body advances:

```powershell
& $python validation_runs\ansys_vertical_flap_fsi\scripts\run_step001_preflow001_after_halfdomain_repair.py
& $python -m tools.validation.print_ansys_vertical_flap_diagnostics `
  --easyfsi-json validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step001_preflow001_after_halfdomain_repair.json `
  --output-dir validation_runs\ansys_vertical_flap_fsi\compare_preflow001_smoke
```

The reviewable CI/local gate for this ANSYS validation surface lives in
`.github\workflows\ansys-vertical-flap-validation.yml`.

For the coarse flow-collapse diagnostic baseline, run:

```powershell
& $python validation_runs\ansys_vertical_flap_fsi\scripts\run_preflow_only_sweep_after_halfdomain_repair.py
& $python validation_runs\ansys_vertical_flap_fsi\scripts\run_flow_collapse_diagnostic_matrix.py
```

These write preflow-only and 10-step matrix artifacts under:

```text
validation_runs\ansys_vertical_flap_fsi\flow_collapse_diagnostics\
```

The diagnostic matrix is intended to answer whether the coarse velocity decay
is projection-only / inlet-driving behavior or marker-feedback behavior. It is
not an L-level Fluent parity run.

For the sustained-flow driver diagnostic baseline, run:

```powershell
& $python validation_runs\ansys_vertical_flap_fsi\scripts\run_sustained_flow_driver_matrix.py
```

This writes explicit flow-driver-mode artifacts under:

```text
validation_runs\ansys_vertical_flap_fsi\sustained_flow_driver_diagnostics\
```

The sustained-flow matrix compares the projection-only collapse baseline, the
diagnostic full-field reinitialize upper bound, and non-full-reset sustained
boundary/source/predictor drivers. It is a 10-step coarse EasyFsi diagnostic and
does not claim Fluent parity.

For the source/outlet balance diagnostic baseline, run:

```powershell
& $python validation_runs\ansys_vertical_flap_fsi\scripts\run_source_outlet_balance_matrix.py
```

This writes source-strength and outlet-balance artifacts under:

```text
validation_runs\ansys_vertical_flap_fsi\source_outlet_balance_diagnostics\
```

The source/outlet balance matrix parameterizes the non-full-reset sustained
inlet source strength and records pressure-outlet and velocity-outlet flux
ratios separately. It is a 10-step calibration diagnostic only; it does not run
50 steps and does not claim Fluent parity.

## Known Non-Gating Historical Failures

- ANSYS vertical-flap displacement tolerance smoke.
- Mooney/Neo-Hookean secondary shell region behavior.
- Long-running `test_core_fluid` or `test_hibm` timeouts or existing count
  failures.

These failures are not structure-refactor gates. Fix them on solver-specific
branches with physical validation evidence.
