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

## Known Non-Gating Historical Failures

- ANSYS vertical-flap displacement tolerance smoke.
- Mooney/Neo-Hookean secondary shell region behavior.
- Long-running `test_core_fluid` or `test_hibm` timeouts or existing count
  failures.

These failures are not structure-refactor gates. Fix them on solver-specific
branches with physical validation evidence.
