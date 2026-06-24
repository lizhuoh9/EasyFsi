# Step 9 Goal: Split solids, geometry, materials, and diagnostics packages

Objective:
Move remaining solver support implementations into layered packages while
preserving legacy imports.

Base branch:
`refactor/hibm-mpm-package-split-step8`

Working branch:
`refactor/solver-support-packages-step9`

Packages:
- `simulation_core.solids`
- `simulation_core.geometry_tools`
- `simulation_core.materials`
- `simulation_core.diagnostics`

Hard boundaries:
- Do not change Neo-Hookean MPM numerical logic.
- Do not change Mooney shell MPM numerical logic.
- Do not change Taichi kernel math.
- Do not change material formulas.
- Do not change geometry generation formulas.
- Do not change CAD parse or tessellation behavior.
- Do not change validation semantics.
- Do not change time-stepping semantics.
- Do not change public class names or function names.
- Do not change default parameters.
- Do not change report field names.
- Do not fix known solver red lights in this step.
- Keep legacy module paths as compatibility shims.

Forbidden code surface:
- `simulation_core/fluids/*`
- `simulation_core/coupling/hibm_mpm/*`
- `simulation_core/fluid.py`
- `simulation_core/hibm_mpm.py`
- `benchmarks/`
- `cases/`
- `run_simulation.py`

Allowed code surface:
- `simulation_core/neo_hookean_mpm.py`
- `simulation_core/mooney_shell_mpm.py`
- `simulation_core/geometry.py`
- `simulation_core/hyperelastic.py`
- `simulation_core/validation.py`
- `simulation_core/time_stepping.py`
- `simulation_core/coordinate_models.py`
- `simulation_core/fluid_domain.py`
- `simulation_core/cad_import.py`
- `simulation_core/cad_tessellation.py`
- `simulation_core/solids/*`
- `simulation_core/solids/mooney_shell/*`
- `simulation_core/geometry_tools/*`
- `simulation_core/materials/*`
- `simulation_core/diagnostics/*`
- `tests/solvers/*`
- `tests/contracts/*`
- `README.md`
- `SIMULATION_CORE_USAGE.md`
- `REFACTORING_NOTES.md`

Target layout:
```text
simulation_core/
|-- neo_hookean_mpm.py
|-- mooney_shell_mpm.py
|-- geometry.py
|-- hyperelastic.py
|-- validation.py
|-- time_stepping.py
|-- coordinate_models.py
|-- fluid_domain.py
|-- cad_import.py
|-- cad_tessellation.py
|-- solids/
|   |-- __init__.py
|   |-- neo_hookean_mpm.py
|   `-- mooney_shell/
|       |-- __init__.py
|       |-- reports.py
|       `-- core.py
|-- geometry_tools/
|   |-- __init__.py
|   |-- surface_mesh.py
|   |-- coordinate_models.py
|   |-- fluid_domain.py
|   |-- cad_import.py
|   `-- cad_tessellation.py
|-- materials/
|   |-- __init__.py
|   `-- hyperelastic.py
`-- diagnostics/
    |-- __init__.py
    |-- validation.py
    `-- time_stepping.py
```

Done criteria:
- Neo-Hookean MPM implementation lives under
  `simulation_core.solids.neo_hookean_mpm`.
- Mooney shell MPM implementation lives under
  `simulation_core.solids.mooney_shell.core`.
- Mooney report dataclasses live under
  `simulation_core.solids.mooney_shell.reports`.
- Surface mesh helpers live under `simulation_core.geometry_tools.surface_mesh`.
- Coordinate/domain/CAD helpers live under `simulation_core.geometry_tools`.
- Hyperelastic material helpers live under
  `simulation_core.materials.hyperelastic`.
- Validation and time-stepping helpers live under `simulation_core.diagnostics`.
- Legacy modules remain compatibility shims.
- Old legacy imports, package imports, and facade imports are identity-compatible.
- Support modules do not import their implementation core modules.
- `simulation_core.__init__` keeps the existing public API.

Required validation:
```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  simulation_core\neo_hookean_mpm.py `
  simulation_core\mooney_shell_mpm.py `
  simulation_core\geometry.py `
  simulation_core\hyperelastic.py `
  simulation_core\validation.py `
  simulation_core\time_stepping.py `
  simulation_core\coordinate_models.py `
  simulation_core\fluid_domain.py `
  simulation_core\cad_import.py `
  simulation_core\cad_tessellation.py `
  simulation_core\solids\__init__.py `
  simulation_core\solids\neo_hookean_mpm.py `
  simulation_core\solids\mooney_shell\__init__.py `
  simulation_core\solids\mooney_shell\core.py `
  simulation_core\solids\mooney_shell\reports.py `
  simulation_core\geometry_tools\__init__.py `
  simulation_core\geometry_tools\surface_mesh.py `
  simulation_core\geometry_tools\coordinate_models.py `
  simulation_core\geometry_tools\fluid_domain.py `
  simulation_core\geometry_tools\cad_import.py `
  simulation_core\geometry_tools\cad_tessellation.py `
  simulation_core\materials\__init__.py `
  simulation_core\materials\hyperelastic.py `
  simulation_core\diagnostics\__init__.py `
  simulation_core\diagnostics\validation.py `
  simulation_core\diagnostics\time_stepping.py

& 'D:\working\taichi\env\python.exe' -m unittest tests.solvers.test_simulation_core_facades -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\contracts -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\integration -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\tools -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest tests.cases.test_squid_latest_core_config tests.cases.test_squid_package_exports -v
```

Optional validation:
```powershell
& 'D:\working\taichi\env\python.exe' -m unittest tests.solvers.test_validation -v
& 'D:\working\taichi\env\python.exe' -m unittest tests.solvers.test_time_stepping -v
& 'D:\working\taichi\env\python.exe' -m unittest tests.solvers.test_hyperelastic_ecoflex -v
```

Residual checks:
- Legacy Neo-Hookean and Mooney modules contain no implementation class bodies.
- Legacy Neo-Hookean and Mooney modules contain no `@ti.kernel`.
- Mooney report support module does not import `core.py`.
- Forbidden solver, coupling, benchmark, and case files remain unchanged except
  source-path test/documentation updates.
- `git diff --check` passes apart from repository line-ending normalization
  warnings.
