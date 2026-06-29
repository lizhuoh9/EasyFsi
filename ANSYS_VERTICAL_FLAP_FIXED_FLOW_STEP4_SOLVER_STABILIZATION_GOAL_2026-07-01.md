# ANSYS Vertical Flap Fixed Flow Step 4 Solver Stabilization Goal

## Objective

Implement Step 4 for the ANSYS vertical flap fixed-flow validation chain. Step 4 must upgrade the fixed-flap projection solver from a diagnostic Step 2 artifact into a numerical candidate artifact by improving pressure Poisson convergence diagnostics, divergence projection quality, initialization sensitivity evidence, mass-flux accounting, and stabilized postprocessing artifacts.

Step 4 is still not a Fluent parity step. It must keep all Fluent-parity and FSI claims explicitly disabled.

## Required Scope

Step 4 modifies the fixed-flap projection solver and diagnostics only.

Allowed source areas:

- `src/refactored/validation/ansys_vertical_flap_fixed/poisson.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/operators.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/projection_solver.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/solver_diagnostics.py`
- `src/refactored/validation/ansys_vertical_flap_fixed/postprocess_fluent_style.py` only if needed to reuse Step 3 postprocessing with Step 4 sources honestly
- `validation_cases/ansys_vertical_flap_fixed_flow/run_fixed_flap_stabilized_solver.py`
- `tests/integration/test_ansys_vertical_flap_fixed_flow_step4_solver_stabilization.py`

Forbidden behavior:

- Do not read or depend on `validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics`.
- Do not commit `validation_runs/ansys_vertical_flap_fsi/rendered_results`.
- Do not claim Fluent parity.
- Do not claim FSI validation.
- Do not tune the solver to match a screenshot maximum speed such as `28.1 m/s`.
- Do not hide raw mass imbalance or divergence problems behind correction-only metrics.
- Do not overwrite the existing Step 2 default artifact bundle as the only output.

## Inputs

Primary Step 4 inputs:

- `validation_runs/ansys_vertical_flap_fixed_flow/preprocess/geometry_mask.npz`
- `validation_runs/ansys_vertical_flap_fixed_flow/preprocess/bc_map.npz`

Baseline comparison inputs:

- `validation_runs/ansys_vertical_flap_fixed_flow/fields/final_fields.npz`
- `validation_runs/ansys_vertical_flap_fixed_flow/logs/solver_history.csv`
- `validation_runs/ansys_vertical_flap_fixed_flow/logs/mass_balance.csv`
- `validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step3_fluent_style/case_manifest_step3.json`

Baseline artifacts may be used only for comparison and reporting. They must not be treated as the default initial field source unless a mode is explicitly named `continuation`.

## Solver Requirements

### Pressure Poisson

Add a masked SOR pressure Poisson solver:

```python
def solve_pressure_poisson_sor(
    rhs,
    p_initial,
    fluid_mask,
    solid_mask,
    outlet_mask,
    pressure_reference_value,
    ds,
    dy,
    max_iters,
    tolerance_abs,
    tolerance_rel,
    omega,
    compatibility_correction=True,
    check_interval=25,
) -> tuple[np.ndarray, dict]:
    ...
```

Requirements:

- Permit `omega` in the range `0.1 <= omega <= 1.95`.
- Keep outlet pressure as Dirichlet: `p = pressure_reference_value`.
- Treat solid, wall, and flap sides as zero-normal-gradient pressure boundaries.
- Use per-cell fluid-neighbor coefficients for masked domains.
- Support optional compatibility correction by subtracting the mean RHS over active non-outlet fluid cells.
- Compute absolute and relative residual diagnostics.
- Converge when either absolute tolerance or relative tolerance is satisfied.
- Return diagnostics:
  - `poisson_iters`
  - `poisson_residual_linf`
  - `poisson_residual_l2`
  - `poisson_residual_linf_initial`
  - `poisson_residual_linf_relative`
  - `rhs_linf`
  - `rhs_l2`
  - `compatibility_correction_applied`
  - `converged`
  - `method`

Acceptance target:

- `poisson_residual_linf_relative < 1e-3`, or residual reduction factor greater than `1e3` for the masked-domain solver test.

### Finite-Volume Projection Operators

Add finite-volume-consistent projection diagnostics:

```python
def compute_fv_divergence(u, v, fluid_mask, solid_mask, ds, dy) -> np.ndarray:
    ...

def compute_fv_pressure_gradient(p, fluid_mask, solid_mask, outlet_mask, ds, dy) -> tuple[np.ndarray, np.ndarray]:
    ...

def compute_interior_divergence_metrics(divergence, fluid_mask, near_solid_mask) -> dict:
    ...
```

Metrics must include:

- `divergence_linf`
- `divergence_l2`
- `divergence_linf_excluding_near_solid`
- `divergence_l2_excluding_near_solid`

The near-solid-excluded metrics are required because flap-corner and wall-adjacent stencil artifacts can dominate full-domain `linf`.

### Initialization Modes

Support:

- `uniform`: `u = inlet_u` and `v = 0` for all fluid cells, then apply boundary conditions.
- `structured_jet`: the current Step 2 jet-shaped initialization, preserved for regression comparison.
- `continuation`: explicit continuation from a provided `final_fields.npz`.

Default stabilized run must use:

```text
initialization_mode = uniform
```

Step 4 must demonstrate that the central gap acceleration is not pre-seeded by a `37 m/s` initial centerline jet.

### Outlet Flux Correction

Add optional outlet flux correction:

```python
def apply_outlet_flux_correction(u, inlet_mask, outlet_mask, dy) -> tuple[np.ndarray, dict]:
    ...
```

The correction may adjust outlet velocity to close mass balance, but logs must preserve raw and corrected values:

- `raw_outlet_flux`
- `corrected_outlet_flux`
- `flux_correction_delta`
- `mass_imbalance_rel_raw`
- `mass_imbalance_rel_corrected`

This is a diagnostic accounting correction and must not hide the solver's raw behavior.

## Stabilized Solver Configuration

The Step 4 runner should use a conservative default configuration:

```yaml
solver:
  max_steps: 800
  cfl: 0.20
  steady_tolerance: 1.0e-5
  poisson_method: sor
  poisson_max_iters: 1200
  poisson_tolerance_abs: 1.0e-4
  poisson_tolerance_rel: 1.0e-3
  poisson_omega: 1.65
  poisson_check_interval: 25
  poisson_compatibility_correction: true
  initialization_mode: uniform
  outlet_flux_correction: true
  history_interval: 10
```

Implementation may reduce `max_steps` in test-specific configs to keep tests fast, but the default runner must still produce the required artifact bundle.

## Required Solver Artifacts

Write stabilized solver outputs under:

```text
validation_runs/ansys_vertical_flap_fixed_flow/stabilized_solver/
```

Required files:

- `fields/final_fields_stabilized.npz`
- `logs/solver_history_stabilized.csv`
- `logs/mass_balance_stabilized.csv`
- `logs/poisson_history_stabilized.csv`
- `diagnostics/quality_comparison_step2_vs_stabilized.json`
- `diagnostics/initialization_sensitivity.csv`
- `case_manifest_step4_solver_stabilization.json`

## Required Stabilized Postprocess Artifacts

Reuse the Step 3 Fluent-style postprocessing chain and write stabilized rendered outputs under:

```text
validation_runs/ansys_vertical_flap_fixed_flow/rendered_results/step4_stabilized_fluent_style/
```

Required files:

- `speed_full_fluent_scale_0_28p1.png`
- `speed_full_autoscale.png`
- `streamwise_minus_Uz_fluent_scale_0_28p1.png`
- `streamwise_minus_Uz_autoscale.png`
- `Uy_full.png`
- `pressure_full.png`
- `geometry_overlay.png`
- `solver_history_plot.png`
- `mass_balance_plot.png`
- `centerline_streamwise_minus_Uz.csv`
- `throat_profile_streamwise_minus_Uz.csv`
- `downstream_profiles_streamwise_minus_Uz.csv`
- `validation_report.md`
- `case_manifest_step3.json`

The report and manifest must still state:

- `fluent_parity = not_claimed`
- `fsi = not_claimed`
- `traction_shared_snapshot_diagnostics = not_used`
- `No Fluent parity claim`
- `No FSI claim`
- `diagnostic_only_not_parity` or `candidate_not_parity`, never parity achieved

## Runner

Add:

```text
validation_cases/ansys_vertical_flap_fixed_flow/run_fixed_flap_stabilized_solver.py
```

Runner behavior:

1. Confirm Step 1 geometry and BC artifacts exist; generate them if missing.
2. Run the stabilized projection solver with default `uniform` initialization.
3. Write stabilized solver artifacts.
4. Run an initialization sensitivity comparison for `uniform` vs `structured_jet`.
5. Write quality comparison against the Step 2 baseline artifacts if present.
6. Call Step 3 postprocess API on stabilized fields/logs and write Step 4 rendered artifacts.
7. Print a JSON summary containing paths, quality, claims, and the status string.

Required JSON summary fields:

```json
{
  "case": "ansys_vertical_flap_fixed_flow",
  "step": "step4_solver_stabilization",
  "initialization_mode": "uniform",
  "stabilized_fields": ".../final_fields_stabilized.npz",
  "stabilized_history": ".../solver_history_stabilized.csv",
  "stabilized_report": ".../validation_report.md",
  "quality": {
    "mass_quality": "...",
    "incompressibility_quality": "...",
    "overall_status": "candidate_not_parity"
  },
  "claims": {
    "fluent_parity": "not_claimed",
    "fsi": "not_claimed"
  }
}
```

## Tests

Add:

```text
tests/integration/test_ansys_vertical_flap_fixed_flow_step4_solver_stabilization.py
```

Required test coverage:

1. Poisson solver residual reduction:
   - `poisson_residual_linf_relative < 1e-3`, or residual reduction factor `> 1e3`.
   - `converged == True`.
   - `poisson_iters <= max_iters`.

2. Projection reduces divergence:
   - Projected divergence is lower than predicted velocity divergence.
   - Near-solid-excluded divergence is reported.
   - No NaN or Inf appears.
   - Solid velocities remain zero.
   - Inlet `u=7` and `v=0` remain enforced.

3. Uniform initialization sensitivity:
   - Uniform initial field does not pre-seed a `37 m/s` centerline jet.
   - Uniform final field still forms gap acceleration.
   - `structured_jet` and `uniform` final centerline peaks are comparable or documented in `initialization_sensitivity.csv`.
   - Neither mode claims Fluent parity.

4. Stabilized default artifacts:
   - All stabilized solver files exist.
   - All Step 4 postprocess files exist.
   - PNG files have valid PNG magic bytes and non-trivial size.
   - CSV files have headers and data rows.

5. Quality improves versus Step 2 baseline:
   - `abs(mass_imbalance_rel_corrected) < 0.02`.
   - `poisson_residual_linf_relative < 1e-3`.
   - `divergence_l2_excluding_near_solid <= baseline_divergence_l2`.
   - Stabilized report contains `No Fluent parity claim`.
   - Stabilized manifest keeps `forbidden_sources.traction_shared_snapshot_diagnostics == not_used`.

## Verification Commands

Run with the trusted local interpreter:

```powershell
& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_flow_step1 -v
& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_flow_step2_solver -v
& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_flow_step3_postprocess -v
& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_flow_step4_solver_stabilization -v
& 'D:\working\taichi\env\python.exe' validation_cases\ansys_vertical_flap_fixed_flow\run_fixed_flap_stabilized_solver.py
git diff --check
git status --short
```

## Commit Plan

Use RED/GREEN checkpoints:

1. `test: reproduce fixed flap solver stabilization contract`
2. `solver: stabilize fixed flap projection and Poisson diagnostics`
3. `validation: add fixed flap stabilized solver artifacts`

The exact split may be adjusted if implementation and artifact generation are inseparable, but a RED test commit and a GREEN implementation/artifact commit are mandatory.

## Done Criteria

Step 4 is done only when:

- A detailed Step 4 goal file is checked in.
- Active goal references this file.
- RED Step 4 tests fail for missing stabilization implementation before production edits.
- GREEN Step 4 tests pass after implementation.
- Step 1, Step 2, Step 3, and Step 4 focused tests pass.
- Stabilized solver runner succeeds.
- Stabilized solver artifacts exist.
- Stabilized Step 3-style postprocess artifacts exist.
- Uniform initialization evidence proves no pre-seeded `37 m/s` jet.
- Corrected mass imbalance is under `2%`.
- Relative Poisson residual is under `1e-3` or documented as a residual reduction factor above `1e3` in the solver test.
- Divergence metrics are reported for both full domain and excluding near-solid cells.
- Report and manifest still make no Fluent parity claim and no FSI claim.
- `validation_runs/ansys_vertical_flap_fsi/rendered_results` is not committed.
- Work is committed and pushed to `origin/codex/ansys-vertical-flap-official-fluent-solver-evaluation-2026-07-01`.
