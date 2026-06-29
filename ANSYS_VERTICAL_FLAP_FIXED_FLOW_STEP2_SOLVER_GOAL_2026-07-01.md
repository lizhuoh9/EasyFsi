# ANSYS Vertical Flap Fixed-Flow Step 2 Solver Goal - 2026-07-01

## Source Request

Implement the Step 2 solver plan described in the user-provided review file:

`C:\Users\lizhu\.codex\attachments\ec4dd664-58b2-45a2-b9d4-b5f33aef2ebf\pasted-text.txt`

Current branch:

`codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`

Starting HEAD:

`b79dd411cc403f1b860277a72c8885aba411ec16`

## Objective

Add a standalone fixed-flap incompressible projection solver for the ANSYS vertical-flap validation case. This is Step 2 after the existing Step 1 preprocessing contract.

The solver must consume only the Step 1 geometry and boundary-condition artifacts:

- `validation_runs/ansys_vertical_flap_fixed_flow/preprocess/geometry_mask.npz`
- `validation_runs/ansys_vertical_flap_fixed_flow/preprocess/bc_map.npz`

The solver must produce a real fixed-flap solver result bundle:

- `validation_runs/ansys_vertical_flap_fixed_flow/fields/final_fields.npz`
- `validation_runs/ansys_vertical_flap_fixed_flow/logs/solver_history.csv`
- `validation_runs/ansys_vertical_flap_fixed_flow/logs/mass_balance.csv`
- `validation_runs/ansys_vertical_flap_fixed_flow/case_manifest_step2.json`

This step must not touch the old FSI traction-snapshot path and must not claim Fluent parity.

## Scope Boundary

Step 2 is a fixed-flap flow solver step only.

Allowed claims:

- Fixed-flap incompressible solver result.
- Solver consumed Step 1 geometry and BC artifacts.
- The result is not sourced from shared FSI diagnostics.

Forbidden claims:

- Fluent parity.
- FSI coupling success.
- Official Fluent velocity-contour equivalence.
- Any velocity field sourced from `traction_shared_snapshot_diagnostics`.

Forbidden input path:

`validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/step020_fields.npz`

Forbidden unrelated commit path:

`validation_runs/ansys_vertical_flap_fsi/rendered_results/`

## Variable And Sign Convention

The Step 1 coordinate convention remains authoritative:

- Arrays use shape `(ny, ns)`.
- Axis 0 is physical/display vertical coordinate `y`.
- Axis 1 is displayed streamwise coordinate `s`.
- Physical solver coordinate is `z = -s`.
- Left-to-right displayed flow means `Uz < 0`.

Step 2 solver internals must use display-plane variables:

- `u = streamwise_minus_Uz = -Uz`
- `v = Uy`
- `speed = sqrt(u^2 + v^2)`

Therefore:

- `u > 0` means displayed left-to-right flow.
- Output must include both `u` and `Uz = -u`.
- Output must include `streamwise_minus_Uz = u`.

## Numerical Method

Implement a deterministic 2-D incompressible projection solver in the `(s, y)` plane:

```text
div(U) = du/ds + dv/dy = 0

du/dt + u du/ds + v du/dy = -1/rho dp/ds + nu laplacian(u)
dv/dt + u dv/ds + v dv/dy = -1/rho dp/dy + nu laplacian(v)
```

Use a fractional-step / projection workflow:

1. Read `geometry_mask.npz` and `bc_map.npz`.
2. Infer `ds` and `dy` from `geometry["s"]` and `geometry["y"]`.
3. Build fluid, solid, inlet, outlet, wall, flap, and near-solid masks.
4. Initialize `u = -inlet_Uz`, `v = inlet_Uy`, `p = 0` on fluid.
5. At every pseudo-time step:
   - Apply velocity BCs.
   - Compute upwind advection.
   - Compute diffusion.
   - Predict `u_star`, `v_star`.
   - Solve pressure Poisson equation.
   - Project velocity.
   - Reapply velocity BCs.
   - Record residuals, divergence, mass balance, velocity statistics, and Poisson diagnostics.
6. Stop on `max_steps` or steady tolerance.
7. Write final fields, CSV logs, and Step 2 manifest.

## Boundary Conditions

Inlet:

- `u = -inlet_Uz = 7.0 m/s`
- `v = inlet_Uy = 0.0`

Outlet:

- `p = outlet_pressure = 0.0`
- `du/ds = 0`
- `dv/ds = 0`

No-slip walls and flap solids:

- `u = 0`
- `v = 0`

Solid cells:

- `u = 0`
- `v = 0`
- `speed = 0`
- Excluded from divergence, centerline, max-speed, and mass-balance statistics.

## Required New Source Files

Add:

- `src/refactored/validation/ansys_vertical_flap_fixed/operators.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/poisson.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/projection_solver.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/solver_diagnostics.py`
- `validation_cases/ansys_vertical_flap_fixed_flow/run_fixed_flap_solver.py`
- `tests/integration/test_ansys_vertical_flap_fixed_flow_step2_solver.py`

Update:

- `src/refactored/validation/ansys_vertical_flap_fixed/__init__.py`

## Operators Contract

`operators.py` must provide:

- `infer_spacing(s, y) -> tuple[float, float]`
- `compute_divergence(u, v, fluid_mask, ds, dy) -> np.ndarray`
- `laplacian(phi, fluid_mask, solid_mask, ds, dy) -> np.ndarray`
- `upwind_advection_u(u, v, fluid_mask, ds, dy) -> np.ndarray`
- `upwind_advection_v(u, v, fluid_mask, ds, dy) -> np.ndarray`
- `compute_pressure_gradient(p, fluid_mask, ds, dy) -> tuple[np.ndarray, np.ndarray]`
- `apply_velocity_bc(u, v, masks, bc_values) -> tuple[np.ndarray, np.ndarray]`

Operator requirements:

- Outputs preserve shape `(128, 360)` for the default case.
- Solid cells always output zero velocity and zero operator values.
- Inlet cells enforce `u = 7`, `v = 0`.
- Outlet cells use zero-gradient velocity from the nearest left fluid neighbor.
- Spacing must come from the arrays, not from config shortcuts.
- Solid-side neighbors must be handled as no-slip boundaries, not as fluid values.

## Poisson Contract

`poisson.py` must provide:

```python
solve_pressure_poisson(
    rhs,
    p_initial,
    fluid_mask,
    solid_mask,
    outlet_mask,
    pressure_reference_value,
    ds,
    dy,
    max_iters,
    tolerance,
    omega,
) -> tuple[np.ndarray, dict]
```

Use NumPy only. Do not add SciPy or external solver dependencies.

Boundary handling:

- Outlet pressure is fixed at `p = 0`.
- Solid/no-slip, inlet, and wall boundaries use zero-normal-gradient pressure behavior.

Return diagnostics:

- `poisson_iters`
- `poisson_residual_linf`
- `poisson_residual_l2`

## Projection Solver Contract

`projection_solver.py` must provide:

```python
run_projection_solver(
    geometry_path: str | Path,
    bc_path: str | Path,
    output_root: str | Path,
    config: SolverConfig | dict | None = None,
) -> dict
```

Default solver config:

```yaml
solver:
  max_steps: 1200
  cfl: 0.35
  steady_tolerance: 1.0e-5
  divergence_tolerance: 1.0e-3
  poisson_max_iters: 400
  poisson_tolerance: 1.0e-5
  poisson_omega: 1.5
  history_interval: 10
  write_checkpoints: false
```

Time step:

```python
dt_adv = cfl * min(ds, dy) / max(max(abs(u)), max(abs(v)), 1e-12)
dt_diff = 0.25 * min(ds, dy)**2 / nu
dt = min(dt_adv, dt_diff)
```

The default runner must remain fast enough for local validation in `D:\working\taichi\env\python.exe`.

## Final Fields Contract

`final_fields.npz` must contain:

- `s`
- `y`
- `S`
- `Y`
- `Z`
- `u`
- `v`
- `Uz`
- `Uy`
- `p`
- `speed`
- `streamwise_minus_Uz`
- `fluid_mask`
- `solid_mask`
- `inlet_mask`
- `outlet_mask`
- `wall_noslip_mask`
- `flap_noslip_mask`
- `near_solid_mask`
- `metadata_json`

## Diagnostics Contract

`solver_diagnostics.py` must provide:

- `compute_mass_balance(u, v, inlet_mask, outlet_mask, ds, dy) -> dict`
- `compute_velocity_stats(u, v, fluid_mask, near_solid_mask) -> dict`
- `compute_centerline_profile(u, y, s, fluid_mask) -> dict`
- `write_solver_history_csv(path, rows) -> None`
- `write_mass_balance_csv(path, rows) -> None`
- `build_step2_manifest(...) -> dict`

The solver history rows must include:

- `step`
- `dt`
- `max_u`
- `max_abs_v`
- `max_speed`
- `interior_max_speed_excluding_near_solid`
- `p99_speed`
- `divergence_linf`
- `divergence_l2`
- `inlet_flux`
- `outlet_flux`
- `mass_imbalance_rel`
- `velocity_change_l2_rel`
- `poisson_iters`
- `poisson_residual_linf`

Keep `interior_max_speed_excluding_near_solid` separate from `max_speed` so that flap-tip peaks cannot be confused with a downstream centerline jet.

## Runner Contract

Add command:

`D:\working\taichi\env\python.exe validation_cases\ansys_vertical_flap_fixed_flow\run_fixed_flap_solver.py`

Default runner behavior:

1. If Step 1 artifacts do not exist, call `run_preprocess`.
2. Read Step 1 `geometry_mask.npz` and `bc_map.npz`.
3. Run the Step 2 projection solver.
4. Write final fields, solver history, mass balance, and Step 2 manifest.
5. Print a JSON summary containing case, step, output paths, and non-claims.

## Required Tests

Use test-first development.

Add `tests/integration/test_ansys_vertical_flap_fixed_flow_step2_solver.py` before implementation and run it to confirm RED.

The tests must verify:

1. The Step 1 contract can be consumed.
2. `run_projection_solver` can be imported.
3. Geometry and BC arrays preserve shape `(128, 360)`.
4. `ds` and `dy` are inferred from `s` and `y` and are positive.
5. Solver variables use `u = -Uz`, so inlet flow is positive.
6. A short solver run writes `final_fields.npz`.
7. Solid velocities and solid speed are exactly zero.
8. Inlet `u = 7.0` and inlet `v = 0.0` remain enforced.
9. No output field contains NaN or Inf.
10. `speed >= 0`.
11. Gap/throat streamwise velocity accelerates above inlet velocity.
12. Downstream centerline contains a streamwise jet above inlet velocity.
13. `max_abs_v > 0`, proving the contraction/expansion creates transverse motion.
14. `solver_history.csv` exists and has at least two rows.
15. `mass_balance.csv` exists and has at least two rows.
16. `case_manifest_step2.json` exists.
17. Manifest states no Fluent parity claim and no FSI claim.
18. Manifest states sources are Step 1 geometry/BC artifacts and not `traction_shared_snapshot_diagnostics`.
19. Step 1 tests still pass.

Do not force a Step 2 maximum speed of `28.1 m/s`. Step 2 must establish a real solver result and jet structure; Fluent-style contour parity belongs to Step 3.

## Verification Commands

RED:

`& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_flow_step2_solver -v`

GREEN:

`& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_flow_step1 -v`

`& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_flow_step2_solver -v`

`& 'D:\working\taichi\env\python.exe' validation_cases\ansys_vertical_flap_fixed_flow\run_fixed_flap_solver.py`

`git diff --check`

## Commit And Push Contract

Use the same branch:

`codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`

Required commit sequence:

1. `test: reproduce fixed flap projection solver contract`
2. `validation: add fixed flap projection solver and artifacts`

Push after implementation and verification succeed.

## Completion Criteria

The goal is complete only when:

- The detailed goal file exists and is referenced by the active Codex goal.
- The RED Step 2 test is committed before implementation.
- The Step 2 solver, runner, diagnostics, and tests exist.
- The Step 2 default artifact bundle exists.
- Step 1 tests pass.
- Step 2 tests pass.
- The Step 2 runner succeeds.
- `git diff --check` passes.
- No old FSI rendered-results or traction snapshot artifacts are committed.
- The branch is pushed to `origin`.
