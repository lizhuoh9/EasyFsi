# ANSYS Vertical-Flap Real Solver Validation Status - 2026-07-01

## Scope

This status records the first post-importer validation pass for branch
`codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`
after commit `15b66e573e83801f98699871fcf4b630018bf144`.

The importer layer was not changed in this pass. The objective was to move from
importer hardening to actual EasyFsi solver evidence and to keep the Fluent side
blocked unless provenance-backed Fluent exports are available.

## EasyFsi Solver Runs

### Generic Solver Selected Formulation

Command:

```powershell
@'
import importlib.util
from pathlib import Path
script = Path("validation_runs/ansys_vertical_flap_fsi/scripts/run_ansys_vertical_flap_generic_solver.py")
spec = importlib.util.spec_from_file_location("easyfsi_generic_vertical_flap", script)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)
payload = module.run()
row = payload["rows"][0]
print("candidate_status =", payload["candidate_status"])
print("completed_step_count =", row["completed_step_count"])
print("max_displacement_m =", row["max_displacement_m"])
print("max_pressure_abs_pa =", row["max_pressure_abs_pa"])
print("max_velocity_mps =", row["max_velocity_mps"])
print("fluent_parity_claimed =", payload["fluent_parity_claimed"])
'@ | & 'D:\working\taichi\env\python.exe' -
```

Observed result:

```text
candidate_status = generic_solver_selected_formulation_step50_passed
completed_step_count = 50
max_displacement_m = 2.2977244952926412e-05
max_pressure_abs_pa = 488.50444533121583
max_velocity_mps = 31.76406478881836
fluent_parity_claimed = False
```

Generated/updated evidence:

```text
validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/
```

Tip displacement export correction:

```text
tip_displacement_export_status = runtime_vector_mapped
tip_displacement_source_field = tip_mean_displacement_m
tip_displacement_columns = tip_displacement_x_m, tip_displacement_y_m, tip_displacement_z_m, tip_displacement_norm_m
```

The generic solver runtime history stores `tip_mean_displacement_m` as a
three-component displacement vector. The export now maps that vector directly
into `easyfsi_tip_displacement_history.csv` and computes
`tip_displacement_norm_m` from the vector components. `max_displacement_m`
remains the whole-field displacement envelope and is still recorded separately.
This EasyFsi export correction does not make a Fluent parity claim.

### Selected Formulation Coupled Step50

Command:

```powershell
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_coupled_step50.py
```

Observed result:

```text
selected_formulation_coupled_step50_timeout
timeout_seconds = 600
```

The process was still running after timeout and had partially rewritten the
step50 diagnostics directory. The process was stopped and the partial artifacts
were restored to the previous committed state. This run is therefore recorded
as attempted but blocked by local runtime duration, not as fresh step50
completion evidence.

## Artifact Tests

Command:

```powershell
& 'D:\working\taichi\env\python.exe' -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_coupled_step50_artifacts `
  tests.integration.test_ansys_vertical_flap_generic_solver_artifacts `
  tests.integration.test_ansys_vertical_flap_real_solver_validation_status `
  -v
```

Observed result:

```text
13 tests passed
```

Interpretation:

```text
generic_solver_selected_formulation_step50_passed
selected formulation step50 committed artifacts still satisfy their artifact contract
```

Only the generic solver artifacts are fresh from this pass. The selected
formulation step50 artifacts are not fresh because the rerun timed out.

## Fluent Reference Status

Checks:

```powershell
Get-Command fluent -ErrorAction SilentlyContinue
Test-Path 'D:\working\fluent_exports\ansys_vertical_flap_2026_07_01_real_bundle'
```

Observed result:

```text
fluent_command_available = false
real_fluent_bundle_available = false
real_fluent_bundle_unavailable
```

No no-copy preflight was run because the real Fluent export bundle is not
available in this workspace. No source_exports promotion was performed.

## Claim Boundary

```text
fluent_parity_claimed: false
fluent_parity_status: blocked_reference_incomplete
```

This pass does not validate solver-vs-Fluent parity. It only records a fresh
EasyFsi generic solver run and the current blocker for the real Fluent side.
