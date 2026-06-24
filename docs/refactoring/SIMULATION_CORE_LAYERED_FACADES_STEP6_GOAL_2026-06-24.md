# Step 6 Goal: Add `simulation_core` Layered Facades

## Objective

Add low-risk layered import facades under `simulation_core/` without moving solver
implementations or changing numerical behavior.

Base branch:

```text
refactor/squid-step-loop-split-step5
```

Target branch:

```text
refactor/simulation-core-layered-facades-step6
```

## New Facade Packages

Create:

```text
simulation_core/fluids/__init__.py
simulation_core/solids/__init__.py
simulation_core/coupling/__init__.py
simulation_core/geometry_tools/__init__.py
simulation_core/materials/__init__.py
simulation_core/diagnostics/__init__.py
```

These packages should only re-export existing APIs from legacy modules.

## Legacy Paths Must Stay Supported

Keep these existing modules and imports working:

```text
simulation_core.fluid
simulation_core.hibm_mpm
simulation_core.mooney_shell_mpm
simulation_core.neo_hookean_mpm
simulation_core.geometry
simulation_core.hyperelastic
simulation_core.validation
simulation_core.fsi_coupling
simulation_core.projected_ibm
simulation_core.tri_surface
```

## Hard Boundaries

Do not edit solver implementation files:

```text
simulation_core/fluid.py
simulation_core/hibm_mpm.py
simulation_core/mooney_shell_mpm.py
simulation_core/neo_hookean_mpm.py
simulation_core/tri_surface.py
simulation_core/projected_ibm.py
simulation_core/fsi_coupling.py
simulation_core/hyperelastic.py
```

Do not change:

```text
Taichi kernels
solver physics
pressure projection formulas
HIBM/MPM logic
FSI coupling formulas
benchmark formulas
case defaults
history.csv fields
CLI defaults
known solver red lights
```

## Allowed Paths

Primary allowed paths:

```text
simulation_core/fluids/*
simulation_core/solids/*
simulation_core/coupling/*
simulation_core/geometry_tools/*
simulation_core/materials/*
simulation_core/diagnostics/*
tests/contracts/*
tests/solvers/test_simulation_core_facades.py
README.md
SIMULATION_CORE_USAGE.md
REFACTORING_NOTES.md
SIMULATION_CORE_LAYERED_FACADES_STEP6_GOAL_2026-06-24.md
```

## Facade Intent

Recommended new imports:

```python
from simulation_core.fluids import CartesianFluidSolver, FluidDomainSpec
from simulation_core.solids import NeoHookeanMpmState, TriMooneyShellMpmState
from simulation_core.coupling import HibmMpmSharpCouplingState, ProjectedIbmRegionPairStepConfig
from simulation_core.geometry_tools import SurfaceMesh
from simulation_core.materials import NeoHookeanMaterial
from simulation_core.diagnostics import ReferenceCurve
```

Use only symbols that actually exist in the current legacy modules. Do not invent
facade exports.

## Tests

Add `tests/solvers/test_simulation_core_facades.py` covering:

```text
simulation_core.fluids exports existing fluid APIs by identity
simulation_core.solids exports existing solid APIs by identity
simulation_core.coupling exports existing coupling APIs by identity
simulation_core.geometry_tools/materials/diagnostics import and expose expected names
legacy modules still exist during migration
facade package directories exist
```

Update `tests/contracts/test_architecture_boundaries.py` with static facade
existence checks. Do not add tests that forbid old imports yet.

## Documentation

Update docs lightly:

```text
README.md
SIMULATION_CORE_USAGE.md
REFACTORING_NOTES.md
```

Document that layered facades are preferred for new imports while legacy module
paths remain supported during migration.

## Validation Commands

Use `D:\working\taichi\env\python.exe`.

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile simulation_core\fluids\__init__.py simulation_core\solids\__init__.py simulation_core\coupling\__init__.py simulation_core\geometry_tools\__init__.py simulation_core\materials\__init__.py simulation_core\diagnostics\__init__.py
& 'D:\working\taichi\env\python.exe' -m unittest tests.solvers.test_simulation_core_facades -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\contracts -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\integration -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\tools -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests\cases -p "test_squid*.py" -v
```

Residual check:

```powershell
git diff --name-only -- simulation_core/fluid.py simulation_core/hibm_mpm.py simulation_core/mooney_shell_mpm.py simulation_core/neo_hookean_mpm.py simulation_core/tri_surface.py simulation_core/projected_ibm.py simulation_core/fsi_coupling.py simulation_core/hyperelastic.py
```

Expected: empty.

## Done Criteria

1. All six new facade packages exist.
2. New facade imports work.
3. Legacy imports still work.
4. Listed solver implementation files remain untouched.
5. No numerical logic, Taichi kernels, benchmark formulas, case defaults, history
   fields, or CLI defaults change.
6. Focused facade tests, contracts, integration, tools, and squid case tests pass.
7. Changes are committed and pushed to GitHub.

## Commit Message

```text
refactor: add simulation core layered facades
```
