# Step 7 Goal: Move fluid solver implementation into simulation_core.fluids

Objective:
Move the `simulation_core/fluid.py` implementation into the `simulation_core/fluids/`
package while preserving legacy imports.

Base branch:
`refactor/simulation-core-layered-facades-step6`

Working branch:
`refactor/fluid-package-split-step7`

Hard boundaries:
- Do not change fluid solver numerical formulas.
- Do not change Taichi kernel math.
- Do not change pressure projection, Jacobi, CG, or multigrid logic.
- Do not change HIBM obstacle, pressure outlet, or divergence cleanup logic.
- Do not change force spreading logic.
- Do not change public class names, function names, report keys, or default parameters.
- Do not fix known solver red lights in this step.
- Keep `simulation_core.fluid` as a compatibility shim.

Allowed code surface:
- `simulation_core/fluid.py`
- `simulation_core/fluids/*`
- `tests/solvers/test_simulation_core_facades.py`
- `tests/contracts/test_architecture_boundaries.py`
- `tests/contracts/test_source_static_contracts.py`
- `tests/solvers/test_fluid.py`
- `tests/solvers/test_core_fluid.py`
- `README.md`
- `SIMULATION_CORE_USAGE.md`
- `REFACTORING_NOTES.md`

Forbidden code surface:
- `simulation_core/hibm_mpm.py`
- `simulation_core/mooney_shell_mpm.py`
- `simulation_core/neo_hookean_mpm.py`
- `simulation_core/tri_surface.py`
- `simulation_core/projected_ibm.py`
- `simulation_core/fsi_coupling.py`
- `benchmarks/`
- `cases/`

Target package layout:
```text
simulation_core/fluids/
|-- __init__.py
|-- constants.py
|-- grid.py
|-- spec.py
|-- reports.py
|-- pressure_outlet.py
`-- solver.py
```

Done criteria:
- `simulation_core/fluids/solver.py` owns `CartesianFluidSolver`.
- `simulation_core/fluid.py` is a compatibility shim.
- `CartesianGrid`, `GradedGridSpec`, `RefinementRegion`, and `build_graded_grid`
  live in `simulation_core/fluids/grid.py`.
- `FluidDomainSpec` lives in `simulation_core/fluids/spec.py`.
- Fluid report dataclasses live in `simulation_core/fluids/reports.py`.
- Fluid constants live in `simulation_core/fluids/constants.py`.
- `pressure_outlet_cleanup_iteration_budget` lives in
  `simulation_core/fluids/pressure_outlet.py`.
- `simulation_core.fluids` re-exports public fluid APIs from the new modules.
- Old and new imports are identity-compatible.
- Fluid support modules do not import `simulation_core.fluids.solver`.

Required validation:
```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  simulation_core\fluid.py `
  simulation_core\fluids\__init__.py `
  simulation_core\fluids\constants.py `
  simulation_core\fluids\grid.py `
  simulation_core\fluids\spec.py `
  simulation_core\fluids\reports.py `
  simulation_core\fluids\pressure_outlet.py `
  simulation_core\fluids\solver.py

& 'D:\working\taichi\env\python.exe' -m unittest tests.solvers.test_simulation_core_facades -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\contracts -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\integration -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\tools -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\cases -p "test_squid*.py" -v
```

Optional validation:
```powershell
& 'D:\working\taichi\env\python.exe' -m unittest tests.solvers.test_core_fluid -v
```

Residual checks:
- `simulation_core/fluid.py` contains no `class CartesianFluidSolver`.
- `simulation_core/fluid.py` contains no `@ti.kernel`.
- `simulation_core/fluids/constants.py`, `grid.py`, `spec.py`, `reports.py`, and
  `pressure_outlet.py` do not import `simulation_core.fluids.solver`.
- Forbidden solver/case/benchmark files remain unchanged.
- `git diff --check` passes apart from repository line-ending normalization warnings.
