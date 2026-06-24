# Post-refactor Baseline

Date: 2026-06-24

Purpose:
Freeze the integration and validation baseline after the ten-step structure
refactor. This document is a boundary marker for future work, not a solver-fix
plan and not a physics-change authorization.

## Refactor Closure

- Final structure-refactor branch: `refactor/final-layout-closure-step10`
- Final structure-refactor commit:
  `56d5167331a1f2d07dd32c8943c27d9bc1bf430f`
- Baseline tag name: `structure-refactor-2026-06-24`

The tag name is recorded here as the post-refactor baseline label. This
docs-only branch does not create or move the tag.

## Structure Gates

The Step 10 structure baseline is defined by the structure-only gates in
`docs/VALIDATION.md`:

```powershell
$python = 'D:\working\taichi\env\python.exe'
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

The optional light solver checks remain informative only:

```powershell
& $python -m unittest tests.solvers.test_validation -v
& $python -m unittest tests.solvers.test_time_stepping -v
& $python -m unittest tests.solvers.test_hyperelastic_ecoflex -v
& $python -m unittest tests.solvers.test_fluid -v
```

## Known Non-gating Failures

These items are explicitly outside the structure-refactor gate:

- ANSYS vertical-flap displacement tolerance.
- Mooney/Neo-Hookean secondary shell region behavior.
- Long-running or historical `test_core_fluid` failures/timeouts.
- Long-running or historical `test_hibm` failures/timeouts.

They are known solver-validation work, not layout-cleanup work.

## Future Work Policy

Do not fix the known non-gating failures in structure-only PRs. Future work for
each item must use a solver-specific branch with:

- A focused reproduction or failing artifact.
- A scoped solver change.
- Physical validation evidence.
- Clear separation from package layout, import cleanup, and documentation-only
  baseline changes.

Baseline, merge, tag, and docs branches may record current status only.
