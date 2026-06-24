# Step 4 Goal: Slim Squid Runner Orchestration

## Objective

Continue the structural cleanup of `cases/squid_soft_robot/` by moving low-risk orchestration code out of `runner.py` while preserving solver behavior exactly.

This step starts from `refactor/squid-case-package-split` and should land on:

```text
refactor/squid-runner-slimming-step4
```

## Required Scope

Create or populate these modules:

```text
cases/squid_soft_robot/runtime_state.py
cases/squid_soft_robot/setup.py
cases/squid_soft_robot/summary.py
cases/squid_soft_robot/rows.py
cases/squid_soft_robot/step_context.py
cases/squid_soft_robot/coupling_common.py
cases/squid_soft_robot/coupling_legacy.py
cases/squid_soft_robot/coupling_sharp.py
```

Move these responsibilities out of `runner.py` where doing so is mechanically safe:

```text
ReducedSquidFSI Taichi runtime state class
final summary/report aggregation
per-step row augmentation helpers
case setup/build helpers
pure Python coupling helpers
small step-context dataclasses
```

The main step loop and nonlocal-heavy sharp/legacy trial replay logic may remain in `runner.py` for a later step.

## Hard Boundaries

Do not change:

```text
simulation_core/
benchmarks/
Taichi kernel math
HIBM/MPM/Fluid solver physics
FSI coupling formulas
benchmark formulas
CLI parameter names or defaults
history.csv field names
summary/report field names
case default parameters
run_simulation.py dispatch behavior
```

Do not fix known solver red lights in this structural branch:

```text
ANSYS displacement tolerance numeric failure
simulation_core_package numpy/static contract failures
Mooney/NeoHookean secondary shell region errors
core_fluid/tri_surface timeout or local failures
```

Allowed primary paths:

```text
cases/squid_soft_robot/
tests/cases/test_squid*.py
tests/contracts/test_*.py
```

## Compatibility Requirements

`cases/squid_soft_robot/__init__.py` must keep package-root compatibility exports for old imports, including `ReducedSquidFSI`, while allowing new code to import explicit submodules.

Submodules must not import `runner.py`; `runner.py` is the orchestrator that imports the split modules.

## Required Test Updates

Extend package export tests so explicit submodules import successfully and `ReducedSquidFSI` is still exported from `cases.squid_soft_robot`.

Extend architecture boundary tests so:

```text
runner.py no longer defines class ReducedSquidFSI
runner.py no longer holds final summary bulk tokens:
  final_pressure_outlet_velocity_to_source_ratio =
  max_velocity_constraint_equivalent_force_norm_n =
```

When static tests only check token presence, read all `cases/squid_soft_robot/*.py`. Keep `SQUID_RUNNER_SOURCE` only for tests that truly inspect runner-local control flow.

## Validation Commands

Use `D:\working\taichi\env\python.exe` for Python validation in this checkout.

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile run_simulation.py
& 'D:\working\taichi\env\python.exe' -m py_compile cases\squid_soft_robot\*.py
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests/cases -p "test_squid*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests/contracts -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests/integration -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests/tools -p "test_*.py" -v
```

Run import smoke for package compatibility:

```powershell
@'
import importlib

pkg = importlib.import_module("cases.squid_soft_robot")
runner = importlib.import_module("cases.squid_soft_robot.runner")
runtime_state = importlib.import_module("cases.squid_soft_robot.runtime_state")
summary = importlib.import_module("cases.squid_soft_robot.summary")

assert callable(pkg.main)
assert callable(runner.main)
assert hasattr(runtime_state, "ReducedSquidFSI")
assert hasattr(pkg, "ReducedSquidFSI")
assert callable(pkg._cell_indices_for_points)

print("squid case package compatibility OK")
'@ | & 'D:\working\taichi\env\python.exe' -
```

Run residual and diff checks:

```powershell
Select-String -Path cases\squid_soft_robot\*.py -Pattern 'sys.modules\[__name__\]'
Select-String -Path cases\squid_soft_robot\runner.py -Pattern 'argparse.ArgumentParser\('
Select-String -Path cases\squid_soft_robot\runner.py -Pattern 'class ReducedSquidFSI'
Select-String -Path cases\squid_soft_robot\runner.py -Pattern 'final_pressure_outlet_velocity_to_source_ratio ='
Select-String -Path cases\squid_soft_robot\runner.py -Pattern 'max_velocity_constraint_equivalent_force_norm_n ='
git diff --check
git diff --name-only -- simulation_core benchmarks
```

Expected residual checks: no matches.

## Done Criteria

1. `runtime_state.py` exists and defines `ReducedSquidFSI`.
2. `runner.py` no longer defines `ReducedSquidFSI`.
3. `summary.py` owns final report/summary assembly.
4. `runner.py` no longer contains the large final/max/total summary aggregation tokens listed above.
5. `rows.py`, `setup.py`, `step_context.py`, and `coupling_*.py` each own a coherent piece of low-risk helper logic.
6. Package-root legacy imports still work.
7. `run_simulation.py` squid-soft-robot dispatch is unchanged.
8. Focused squid, contract, integration, and tools tests pass.
9. No changes land under `simulation_core/` or `benchmarks/`.
10. No CLI default semantics, `history.csv` fields, summary keys, solver physics, or benchmark formulas change.

## Commit Message

```text
refactor: slim squid runner orchestration modules
```
