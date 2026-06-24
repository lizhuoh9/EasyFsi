# Step 5 Goal: Split Squid Step Loop and Coupling Outer Paths

## Objective

Move squid step-loop orchestration and coupling outer paths out of `runner.py` while preserving behavior.

Base branch:

```text
refactor/squid-runner-slimming-step4
```

Target branch:

```text
refactor/squid-step-loop-split-step5
```

## Required Modules

Create:

```text
cases/squid_soft_robot/step_loop.py
cases/squid_soft_robot/trial_replay.py
cases/squid_soft_robot/solid_step.py
cases/squid_soft_robot/fluid_step.py
```

Continue refining:

```text
cases/squid_soft_robot/step_context.py
cases/squid_soft_robot/coupling_common.py
cases/squid_soft_robot/coupling_legacy.py
cases/squid_soft_robot/coupling_sharp.py
cases/squid_soft_robot/runner.py
cases/squid_soft_robot/__init__.py
```

## Scope

Move or wrap:

```text
main step-loop shell
legacy projected-IBM coupling outer helpers
sharp HIBM-MPM coupling outer helpers
accepted trial payload and replay helpers
solid advancement outer helpers
fluid advancement outer helpers
step-loop context dataclasses
```

The preferred end state is that `runner.py` owns only `run()`, `main()`, top-level configuration/setup glue, and final report calls.

## Hard Boundaries

Do not change:

```text
simulation_core/
benchmarks/
run_simulation.py
Taichi kernel math
HIBM / MPM / Fluid solver physics
FSI coupling formulas
pressure projection / pressure outlet / HIBM pressure Neumann logic
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
SQUID_*_GOAL_*.md
```

## Step Context Requirements

Extend `step_context.py` with explicit dataclasses for loop state, FSI step control, accepted trial replay state, and step execution results. Use these to reduce naked state where practical, but do not rewrite solver logic just to force full context adoption.

## Extraction Requirements

`trial_replay.py` should own low-risk accepted payload validation/reuse helpers.

`solid_step.py` should own low-risk solid substep helpers and solid advancement wrappers where parameters can be passed explicitly.

`fluid_step.py` should own low-risk fluid-step vector/config helpers and projected-IBM config assembly where parameters can be passed explicitly.

`coupling_legacy.py` and `coupling_sharp.py` should gain outer-path helpers where doing so is mechanical. Do not change fixed-point iteration math, accepted/rejected trial selection, all-trials-rejected semantics, or zero-force-commit semantics.

`step_loop.py` should own the main `for step in range(...)` loop or, if a fully mechanical migration is too risky, should own a visible step-loop shell with `runner.py` delegating step-loop execution to it. The target is that `runner.py` no longer directly contains `for step in range(first_step, step_count + 1):`.

## Compatibility

Update `cases/squid_soft_robot/__init__.py` so package-root compatibility exports include:

```text
step_loop
trial_replay
solid_step
fluid_step
```

Submodules must not import `runner.py`.

## Test Updates

Extend `tests/cases/test_squid_package_exports.py` so explicit imports include:

```text
cases.squid_soft_robot.step_loop
cases.squid_soft_robot.trial_replay
cases.squid_soft_robot.solid_step
cases.squid_soft_robot.fluid_step
```

Extend `tests/contracts/test_architecture_boundaries.py` to check:

```text
runner.py does not directly hold for step in range(first_step, step_count + 1)
step_loop.py owns a for step in range(...) loop
```

Only add a runner-not-defining-sharp-trial-closure test if `advance_sharp_trial_once` is fully moved. Otherwise leave it for the next step.

For `tests/cases/test_squid_latest_core_config.py`:

```text
token-existence tests read all squid package sources
runner-local control-flow tests read runner.py
step-loop control-flow tests read step_loop.py
```

Add `SQUID_STEP_LOOP_SOURCE = SQUID_CASE_ROOT / "step_loop.py"` when useful.

## Validation Commands

Use `D:\working\taichi\env\python.exe`.

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile run_simulation.py @(Get-ChildItem -Path cases\squid_soft_robot -Filter *.py | ForEach-Object { $_.FullName })
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests/cases -p "test_squid*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests/contracts -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests/integration -p "test_*.py" -v
& 'D:\working\taichi\env\python.exe' -m unittest discover -s tests/tools -p "test_*.py" -v
```

Import smoke:

```powershell
@'
import importlib

pkg = importlib.import_module("cases.squid_soft_robot")
runner = importlib.import_module("cases.squid_soft_robot.runner")
step_loop = importlib.import_module("cases.squid_soft_robot.step_loop")
trial_replay = importlib.import_module("cases.squid_soft_robot.trial_replay")
solid_step = importlib.import_module("cases.squid_soft_robot.solid_step")
fluid_step = importlib.import_module("cases.squid_soft_robot.fluid_step")

assert callable(pkg.main)
assert callable(runner.main)
assert step_loop is not None
assert trial_replay is not None
assert solid_step is not None
assert fluid_step is not None

print("squid step loop split import compatibility OK")
'@ | & 'D:\working\taichi\env\python.exe' -
```

Residual checks:

```powershell
Select-String -Path cases\squid_soft_robot\*.py -Pattern 'sys.modules\[__name__\]'
Select-String -Path cases\squid_soft_robot\runner.py -Pattern 'argparse.ArgumentParser\('
Select-String -Path cases\squid_soft_robot\runner.py -Pattern 'class ReducedSquidFSI'
Select-String -Path cases\squid_soft_robot\runner.py -Pattern 'final_pressure_outlet_velocity_to_source_ratio ='
Select-String -Path cases\squid_soft_robot\runner.py -Pattern 'for step in range\(first_step, step_count \+ 1\):'
Select-String -Path cases\squid_soft_robot\*.py -Pattern 'from \. import runner|from cases\.squid_soft_robot import runner'
git diff --check
git diff --name-only -- simulation_core benchmarks
```

Expected:

```text
no sys.modules[__name__]
no runner argparse bulk
no runner ReducedSquidFSI
no runner final summary bulk token
no runner direct main step loop
no submodule imports runner, except __init__.py transitional compatibility import
no simulation_core/ or benchmarks/ diff
```

## Done Criteria

1. `step_loop.py` exists and owns the main step-loop shell.
2. `runner.py` no longer directly contains `for step in range(first_step, step_count + 1):`.
3. `trial_replay.py` exists and owns accepted trial payload/replay helpers.
4. `solid_step.py` exists and owns solid advancement helper logic.
5. `fluid_step.py` exists and owns fluid advancement helper logic.
6. `coupling_legacy.py` / `coupling_sharp.py` own more of their outer-path helpers without physics changes.
7. `runner.py` is further shortened, target below 2500 lines if practical.
8. Package-root legacy exports still work.
9. `run_simulation.py` squid-soft-robot dispatch is unchanged.
10. Focused squid, contracts, integration, and tools tests pass.
11. No changes land under `simulation_core/` or `benchmarks/`.
12. No solver physics, Taichi kernel math, CLI defaults, `history.csv` fields, summary keys, or benchmark formulas change.

## Commit Message

```text
refactor: split squid step loop orchestration
```
