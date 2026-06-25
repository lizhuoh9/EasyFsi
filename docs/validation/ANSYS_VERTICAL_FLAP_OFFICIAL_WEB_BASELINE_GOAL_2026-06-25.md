# ANSYS Vertical Flap Official Web Baseline Goal - 2026-06-25

## Purpose

Create the first EasyFsi-vs-ANSYS-official-web baseline for the ANSYS vertical-flap FSI case.

This step does not download a Fluent case and does not run Fluent locally. It converts the public ANSYS Fluent v242 tutorial result scale into a local EasyFsi validation reference, runs the EasyFsi ANSYS vertical-flap case, and writes a diagnostic report that clearly identifies which validation layer currently fails.

The comparison target is intentionally limited:

- It is a comparison against the public ANSYS tutorial web-published contour scales.
- It is not a point-by-point comparison against a Fluent-exported time-history report.
- It should be used as the first official-web baseline before opening a solver-fix branch.

## Branch And Base

- Work branch: `validation/ansys-vertical-flap-official-web-baseline-2026-06-25`
- Base branch: `validation/ansys-vertical-flap-runtime-diagnosis-2026-06-24`
- Base commit observed before this goal was written: `b71746c`

## Hard Boundaries

Do not modify solver behavior in this step.

Forbidden:

- No solver changes.
- No physics changes.
- No changes under `simulation_core/`.
- No changes under `cases/ansys_vertical_flap_fsi.py`.
- No changes under `benchmarks/`.
- No changes to ANSYS reference values, tolerances, pressure/velocity/solid formulas, material constants, boundary conditions, damping, support radius, substeps, or feedback-loop logic.
- Do not claim Fluent time-history parity.
- Do not claim the solver is fixed if diagnostics remain red.

Allowed:

- Add this detailed goal file under `docs/validation/`.
- Add a small checked-in official-web reference CSV under `docs/validation/`.
- Generate local runtime artifacts under `validation_runs/ansys_vertical_flap_fsi/`.
- Add a concise conclusion document under `docs/validation/`.
- Commit generated comparison artifacts only if they are reasonably small and are needed to support the conclusion.

## Official Web Reference

Use the ANSYS Fluent v242 tutorial "Modeling Two-Way Fluid-Structure Interaction (FSI) Within Fluent" as the public source.

Record the following official web-published values:

- `duct_length_m = 0.10`
- `duct_height_m = 0.04`
- `flap_height_m = 0.01`
- `flap_thickness_m = 0.003`
- `solid_material = silicone rubber`
- `density_kgm3 = 1600`
- `young_modulus_pa = 1.0e6`
- `poisson_ratio = 0.47`
- `inlet_velocity_mps = 10.0`
- `outlet = pressure outlet`
- `modeled_domain = lower half by symmetry`
- `dt_s = 0.0005`
- `step_count = 50`
- `final_time_s = 0.025`
- `displacement contour range = 0 to 5.1e-05 m`
- `velocity magnitude contour range = 20 to 29 m/s`

Create a small checked-in reference CSV:

```text
docs/validation/ansys_vertical_flap_official_web_reference_2026-06-25.csv
```

Required CSV content:

```csv
step,time_s,tip_total_displacement_m,velocity_min_mps,velocity_max_mps,source
50,0.025,5.1e-05,20.0,29.0,ANSYS Fluent v242 tutorial contour ranges
```

Also create a runtime-compatible local CSV for the current diagnostics tool:

```text
validation_runs/ansys_vertical_flap_fsi/official_web/fluent_tip_displacement_web_final.csv
```

Required CSV content:

```csv
step,time_s,tip_total_displacement_m,tip_x_displacement_m,tip_y_displacement_m
50,0.025,5.1e-05,,
```

## Required EasyFsi Run

Use the trusted interpreter:

```powershell
$python = 'D:\working\taichi\env\python.exe'
```

Prepare local artifact directories:

```powershell
New-Item -ItemType Directory -Force validation_runs\ansys_vertical_flap_fsi\easyfsi
New-Item -ItemType Directory -Force validation_runs\ansys_vertical_flap_fsi\official_web
New-Item -ItemType Directory -Force validation_runs\ansys_vertical_flap_fsi\compare
```

Run EasyFsi:

```powershell
& $python run_simulation.py ansys-vertical-flap-fsi --steps 50 --json `
  > validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step050.json
```

Required output:

```text
validation_runs/ansys_vertical_flap_fsi/easyfsi/easyfsi_step050.json
```

The file must be non-empty and must contain the EasyFsi JSON report.

## Required Diagnostics

Run the diagnostics against the EasyFsi JSON and web-final Fluent-compatible CSV:

```powershell
& $python -m tools.validation.print_ansys_vertical_flap_diagnostics `
  --easyfsi-json validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step050.json `
  --fluent-tip-csv validation_runs\ansys_vertical_flap_fsi\official_web\fluent_tip_displacement_web_final.csv `
  --output-dir validation_runs\ansys_vertical_flap_fsi\compare
```

Required outputs:

```text
validation_runs/ansys_vertical_flap_fsi/compare/easyfsi_summary.csv
validation_runs/ansys_vertical_flap_fsi/compare/stage_check.md
validation_runs/ansys_vertical_flap_fsi/compare/displacement_compare.csv
```

The summary and stage-check outputs must expose:

- `status`
- `velocity_peak_mps`
- `velocity_peak_relerr`
- `max_disp_m`
- `ref_max_disp_m`
- `disp_relerr`
- `tip_dz_monotonic_violation_count`
- `first_tip_dz_violation_step`
- `max_tip_dz_rebound_m`
- `fluid_recomputed_after_feedback`
- `feedback_closure_status`

The displacement comparison must expose:

- `fluent_tip_total_m`
- `easyfsi_tip_total_m`
- `abs_error`
- `rel_error`
- `easyfsi_tip_streamwise_m`
- `easyfsi_tip_vertical_m`

## Required Conclusion Document

Create:

```text
docs/validation/ANSYS_VERTICAL_FLAP_OFFICIAL_WEB_BASELINE_2026-06-25.md
```

The document must include:

- Source and scope.
- Official web-published reference values.
- Exact EasyFsi run command.
- Exact diagnostics command.
- Paths to generated artifacts.
- The final measured values from `easyfsi_summary.csv`, `stage_check.md`, and `displacement_compare.csv`.
- The explicit statement:

```text
Compared against ANSYS official tutorial web-published displacement scale.
Not yet compared against a Fluent exported time-history report.
```

The document must fill in:

- `EasyFsi status`
- `EasyFsi max_disp_m`
- `Official web displacement scale = 5.1e-05`
- `Final relative error`
- `tip_dz_monotonic_violation_count`
- `feedback_closure_status`
- `velocity_peak_mps`
- `official velocity range = 20~29`

## First Decision Rules

After the run, interpret in this order:

1. If `status = FAIL_FLOW`, the next solver target is fluid / inlet / outlet / projection / obstacle.
2. If `status = FAIL_SOLID_HISTORY`, the next solver target is time integration / load persistence / feedback loop.
3. If `feedback_closure_status = OPEN_LOOP_LOAD_REUSE`, the next solver target is closed-loop fluid feedback, where each FSI step's solid/marker feedback affects the next fluid solve.
4. If `status = FAIL_MAGNITUDE` but history does not rebound, inspect material model, support radius, solid substeps, damping, and marker area.
5. If `status = PASS_SMOKE` but displacement relative error remains large, inspect Fluent Linear Elasticity vs EasyFsi Neo-Hookean/MPM model differences.

The expected current result is likely:

```text
FAIL_SOLID_HISTORY
feedback_closure_status = OPEN_LOOP_LOAD_REUSE
```

If the actual result differs, document the actual artifact-backed result instead of forcing the expectation.

## Required Verification

Run:

```powershell
& $python -m py_compile tools\validation\print_ansys_vertical_flap_diagnostics.py
& $python -m unittest tests.tools.test_ansys_vertical_flap_diagnostics -v
& $python -m unittest discover -s tests\tools -p "test_*.py" -v
& $python scripts\validate_structure.py
git diff --check
```

Also inspect the generated artifacts directly:

```powershell
Get-Item validation_runs\ansys_vertical_flap_fsi\easyfsi\easyfsi_step050.json
Get-Content validation_runs\ansys_vertical_flap_fsi\compare\easyfsi_summary.csv
Get-Content validation_runs\ansys_vertical_flap_fsi\compare\stage_check.md
Get-Content validation_runs\ansys_vertical_flap_fsi\compare\displacement_compare.csv
```

## Completion Criteria

This goal is complete when:

- The detailed goal file is committed.
- The official-web reference CSV is committed under `docs/validation/`.
- EasyFsi has been run for 50 steps.
- The required diagnostics artifacts exist locally.
- The conclusion document is filled with artifact-backed numbers.
- No solver/case/benchmark physics files were modified.
- Required validation commands pass.
- The branch is committed and pushed to GitHub.
