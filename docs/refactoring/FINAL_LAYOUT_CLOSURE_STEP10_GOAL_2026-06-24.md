# Step 10 Goal: Final layout closure

Objective:
Finish the ten-step structure refactor by closing documentation, compatibility
shim policy, root public API import direction, architecture contracts, and the
lightweight validation matrix.

Base branch:
`refactor/solver-support-packages-step9`

Working branch:
`refactor/final-layout-closure-step10`

Hard boundaries:
- Do not change solver physics.
- Do not change Taichi kernel math.
- Do not change fluid projection formulas.
- Do not change HIBM/MPM coupling behavior.
- Do not change solid MPM formulas.
- Do not change material formulas.
- Do not change geometry or CAD behavior.
- Do not change benchmark formulas.
- Do not change case defaults.
- Do not change CLI defaults.
- Do not change `history.csv` fields.
- Do not change summary or report keys.
- Do not fix known solver red lights in this step.

Known non-gating solver red lights:
- ANSYS vertical-flap displacement tolerance.
- Mooney/Neo-Hookean secondary region behavior.
- Long-running or historical `test_core_fluid` failures/timeouts.
- Long-running or historical `test_hibm` failures/timeouts.

Allowed primary edit surface:
- `README.md`
- `SIMULATION_CORE_USAGE.md`
- `REFACTORING_NOTES.md`
- `ARCHITECTURE.md`
- `docs/`
- `scripts/`
- `simulation_core/__init__.py`
- `tests/contracts/`
- `tests/solvers/test_simulation_core_facades.py`
- `tests/integration/`
- `tests/tools/`

Optional low-risk import cleanup surface:
- `cases/`
- `benchmarks/`
- `tests/`

Do not force a repo-wide import rewrite. Legacy shims remain supported.

Required tasks:
1. Move root Step 4-9 goal files into `docs/refactoring/`.
2. Add `docs/refactoring/README.md`.
3. Add `ARCHITECTURE.md` with runtime layers, dependency direction, and legacy
   compatibility policy.
4. Route root `simulation_core.__init__` public API imports through layered
   packages/facades instead of the `fluid` and `hibm_mpm` legacy shims.
5. Encode unified legacy shim policy in `tests/contracts/`.
6. Encode package implementation presence contracts in `tests/contracts/`.
7. Add root/facade/legacy import identity coverage in
   `tests/solvers/test_simulation_core_facades.py`.
8. Add `scripts/validate_structure.py` and `scripts/__init__.py`.
9. Add `docs/VALIDATION.md` with structure-only gating, optional light solver
   checks, and known non-gating failures.
10. Update README, Simulation Core usage docs, and refactoring notes.

Required validation:
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

Optional validation:
```powershell
& 'D:\working\taichi\env\python.exe' -m unittest tests.solvers.test_validation -v
& 'D:\working\taichi\env\python.exe' -m unittest tests.solvers.test_time_stepping -v
& 'D:\working\taichi\env\python.exe' -m unittest tests.solvers.test_hyperelastic_ecoflex -v
& 'D:\working\taichi\env\python.exe' -m unittest tests.solvers.test_fluid -v
```

Residual checks:
- `git diff --check` passes.
- Root `*GOAL*` files are gone.
- Legacy shims contain no implementation kernels/classes.
- Package implementation modules do not import legacy shim paths.
- Forbidden solver/case/benchmark behavior surfaces remain unchanged.

Acceptance:
- Step goal files are archived under `docs/refactoring/`.
- `ARCHITECTURE.md` and `docs/VALIDATION.md` exist.
- Root public API imports route through package facades, not `fluid` or
  `hibm_mpm` shims.
- Legacy shim policy and implementation locations are covered by contracts.
- Root, facade, package, and legacy import identities pass.
- Structure validation script passes.
- Focused structure tests pass.
- No solver physics, kernels, formulas, defaults, report fields, case behavior,
  or benchmark formulas are changed.
