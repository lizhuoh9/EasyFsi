# ANSYS Vertical Flap Closed-Loop Runner Solver Goal - 2026-06-25

## Objective

Change the ANSYS vertical-flap benchmark runner from open-loop load reuse to a real per-step closed-loop runner.

The runner must project or recompute the fluid field inside the FSI loop before stress sampling. It must not merely set `fluid_recomputed_after_feedback = True` or `feedback_closure_status = CLOSED_LOOP_RECOMPUTED_FLOW` in the report.

## Branch And Base

- Work branch: `solver/ansys-vertical-flap-closed-loop-runner-2026-06-25`
- Base branch: `test/ansys-vertical-flap-runner-loop-contract-2026-06-25`
- Base commit observed before this goal was written: `b12c07e`

## Current Baseline

From committed official-web baseline artifacts:

- `status = FAIL_SOLID_HISTORY`
- `feedback_closure_status = OPEN_LOOP_LOAD_REUSE`
- `fluid_recomputed_after_feedback = false`
- `tip_dz_monotonic_violation_count = 23`
- `first_tip_dz_violation_step = 5`
- `max_tip_dz_rebound_m = 5.472451448440552e-06`
- `final displacement relative error = 0.4699417795678494`
- `velocity_peak_mps = 28.15654945373535`, inside the official web `20 to 29 m/s` range

Flow, interface, scatter, and root gates already pass in the baseline. The first solver target is the runner loop, not case tuning.

## Hard Boundaries

Do not change:

- `simulation_core/`
- `cases/ansys_vertical_flap_fsi.py`
- ANSYS geometry, material, boundary-condition, reference, or tolerance values
- `pressure_scale`, `pressure_jump_pa`, or any equivalent pressure shortcut
- material constants
- support radius defaults
- solver kernel formulas

Allowed changes:

- `benchmarks/official/solid_mpm_fsi_runner.py`
- `tests/cases/test_ansys_vertical_flap_fsi.py`, only to keep the 50-step
  physical-displacement smoke gate aligned with the regenerated artifact gates
  while physical targets remain red
- `tests/integration/test_ansys_vertical_flap_closed_loop_feedback.py`
- `tests/integration/test_ansys_vertical_flap_runner_loop_contract.py`
- validation artifacts under `validation_runs/ansys_vertical_flap_fsi/compare/`
- validation conclusion docs under `docs/validation/`

Only commit regenerated artifacts that are small and directly support the solver comparison. Prefer committing compare outputs; commit the full JSON only if needed for test inputs or reproducibility.

## Required Implementation

### 1. Split one-time flow initialization from per-step projection

Replace the current one-shot pattern:

```python
fluid = _build_fluid(config, runtime)
flow_report = _solve_computed_flow(fluid, config)
...
for step_index in range(config.step_count):
    latest_stress_report = _sample_stress_to_marker_forces(markers, fluid)
```

with explicit initialization plus per-step projection:

```python
fluid = _build_fluid(config, runtime)
_initialize_computed_flow(fluid, config)
...
for step_index in range(config.step_count):
    latest_flow_report = _project_current_flow(...)
    latest_stress_report = _sample_stress_to_marker_forces(markers, fluid)
```

The per-step projection helper must be inside `for step_index in range(config.step_count):` and must appear before `_sample_stress_to_marker_forces(markers, fluid)`.

### 2. Preserve initialization semantics

Do not call `_initialize_inlet_flow` inside every FSI step. It may reset the flow state and erase feedback effects.

Use a one-time helper such as:

```python
def _initialize_computed_flow(fluid, config) -> np.ndarray:
    return _initialize_inlet_flow(fluid, config)
```

Then use a per-step helper such as:

```python
def _project_current_flow(fluid, config, *, reset_pressure: bool) -> dict[str, object]:
    projection_report = fluid.project(...)
    return _flow_state_report(fluid, projection_report)
```

### 3. Report top-level closed-loop evidence

Add final report fields:

```python
"fluid_recomputed_after_feedback": True,
"feedback_closure_status": "CLOSED_LOOP_RECOMPUTED_FLOW",
"fluid_recompute_count": fluid_recompute_count,
```

`fluid_recompute_count` must be derived from actual per-step projection calls, not hardcoded.

### 4. Report per-step flow evidence

Each history entry must include:

```python
"fluid_recomputed": True,
"local_velocity_peak_mps": latest_flow_report["local_velocity_peak_mps"],
"pressure_min_pa": latest_flow_report["pressure_min_pa"],
"pressure_max_pa": latest_flow_report["pressure_max_pa"],
"flow_projection_report": latest_flow_report["projection_report"],
```

### 5. Update tests incrementally

Update `tests/integration/test_ansys_vertical_flap_runner_loop_contract.py` so source-level closed-loop tests become normal passing tests once the implementation exists.

Update `tests/integration/test_ansys_vertical_flap_closed_loop_feedback.py` only for artifact-level contracts that the regenerated artifacts genuinely satisfy.

Do not remove `expectedFailure` from:

- no-history-rebound contract
- 20 percent displacement error contract

unless the regenerated 50-step artifacts actually satisfy them.

If `tests/cases/test_ansys_vertical_flap_fsi.py` still fails only on the same
official-web physical displacement targets, keep that case-level smoke as an
`expectedFailure` and document the artifact-backed gap. Do not weaken lower-level
flow/interface/scatter/root assertions.

## Required Runtime Regeneration

After implementation, regenerate the official-web comparison artifacts:

```powershell
$python = 'D:\working\taichi\env\python.exe'

& $python run_simulation.py ansys-vertical-flap-fsi --steps 50 --json `
  > validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step050.json

& $python -m tools.validation.print_ansys_vertical_flap_diagnostics `
  --easyfsi-json validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step050.json `
  --fluent-tip-csv validation_runs\ansys_vertical_flap_fsi\official_web\fluent_tip_displacement_web_final.csv `
  --output-dir validation_runs\ansys_vertical_flap_fsi\compare
```

Normalize the redirected JSON to valid UTF-8 if PowerShell emits NUL bytes.

## Acceptance

Source-level acceptance:

- runner-loop contract tests pass without expected failures for closed-loop implementation requirements
- `solid_mpm_fsi_runner.py` projects/recomputes flow inside the FSI loop before stress sampling
- report fields exist and are tied to actual recompute count
- history entries include per-step flow diagnostics

Artifact-level first target:

- `stage_check.md` reports `fluid_recomputed_after_feedback = true`
- `stage_check.md` reports `feedback_closure_status = CLOSED_LOOP_RECOMPUTED_FLOW`
- flow/interface/scatter/root gates remain passing

Artifact-level physical targets:

- `tip_dz_monotonic_violation_count = 0`
- `status != FAIL_SOLID_HISTORY`
- final displacement relative error `<= 0.20`

If the closed-loop fields pass but the physical targets remain red, keep the corresponding artifact-level tests marked `expectedFailure` and document the remaining gap with the regenerated numbers.

## Required Verification

Run:

```powershell
& $python -m py_compile `
  benchmarks\official\solid_mpm_fsi_runner.py `
  tests\integration\test_ansys_vertical_flap_closed_loop_feedback.py `
  tests\integration\test_ansys_vertical_flap_runner_loop_contract.py `
  tools\validation\print_ansys_vertical_flap_diagnostics.py

& $python -m unittest tests.integration.test_ansys_vertical_flap_runner_loop_contract -v
& $python -m unittest tests.integration.test_ansys_vertical_flap_closed_loop_feedback -v
& $python -m unittest discover -s tests\integration -p "test_*.py" -v
& $python -m unittest discover -s tests\tools -p "test_*.py" -v
& $python -m unittest tests.cases.test_ansys_vertical_flap_fsi -v
& $python scripts\validate_structure.py
git diff --check
```

Do not use full:

```powershell
unittest discover -s tests -p "test_*.py" -v
```

as this checkout enters heavy Taichi solver tests, times out, and exposes existing discovery/import issues unrelated to this branch.

## Completion Criteria

This goal is complete when:

- the goal file is committed
- closed-loop runner implementation is committed
- relevant tests are updated and pass as described
- regenerated comparison artifacts are committed if small and necessary
- no forbidden files or parameters changed
- branch is pushed to GitHub
