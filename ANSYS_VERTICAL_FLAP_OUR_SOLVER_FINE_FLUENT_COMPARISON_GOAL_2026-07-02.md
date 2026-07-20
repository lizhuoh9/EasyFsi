# Goal: Run Our Solver on the Official Fluent Two-Way FSI Case With a Very Fine Grid and Compare Honestly

## Objective

Use the current repaired in-repo solver to run the official ANSYS/Fluent `fsi_2way` vertical flap two-way fluid-structure interaction case, using a very fine grid, then compare the result against the saved local Fluent reference.

This goal starts only after user review and approval.

## Locked Fluent Reference

Use the already generated local Fluent reference as the primary comparison target:

- Reference directory: `validation_runs/ansys_vertical_flap_fsi/official_fluent_fine_mesh_steady_2026-07-01/fsi_50step_serial_from_adapt_cycle3_mesh/`
- Final case: `fine_fsi_50step_final.cas.h5`
- Final data: `fine_fsi_50step_final.dat.h5`
- Summary: `fine_fsi_50step_summary.json`
- Time: `t = 0.025 s`
- Time step: `dt = 5.0e-4 s`
- Steps: `50`
- Fluent fluid cells: `78,888`
- Fluent total cells: `79,086`
- Velocity magnitude range: `0.012004324942535071` to `30.0203478570059 m/s`
- Pressure range: `-43.9929563093399` to `467.82098578487745 Pa`

The 276k-cell Fluent steady preflow artifacts are available in the parent directory, but they are not the primary transient FSI reference because Fluent transient FSI on that mesh failed in the parallel compute path with `Stream removed` / `mpt_accept` disconnect. Preserve that failure evidence; do not treat the 276k steady run as a completed transient FSI reference.

## Hard Constraints

1. Do not tune parameters to match Fluent.
2. Do not introduce a special-purpose Fluent-matching solver path.
3. Do not change the Fluent reference metrics or rerender it as if it were a different run.
4. Do not hide numerical failures behind smoothing, clipping, masking, or cherry-picked plots.
5. If the current repaired solver still fails, stop and report the actual blocker with artifacts.
6. Keep solver changes, if any are needed, inside the correct `simulation_core` functional modules.
7. Preserve the distinction between:
   - completed Fluent transient FSI reference;
   - completed Fluent fine steady preflow;
   - failed Fluent transient attempts on too-fine/parallel meshes;
   - our solver results.

## Case Definition to Match

Use the official vertical flap setup:

- Domain: `x = 0..0.10 m`, `y = 0..0.02 m`
- Inlet velocity: `10 m/s`
- Fluid density: `1.2 kg/m^3`
- Fluid dynamic viscosity: `1.8e-5 Pa*s`
- Solid density: `1600 kg/m^3`
- Solid Young's modulus: `1.0e6 Pa`
- Solid Poisson ratio: `0.47`
- Final comparison time: `t = 0.025 s`
- Primary field comparison: velocity magnitude, pressure, and flap/interface geometry at final time

If the in-repo solver is strictly 3D while Fluent reference is 2D, run a documented quasi-2D slab:

- Use the same `x-y` geometry.
- Add a thin, explicitly recorded `z` thickness.
- Use enough `z` cells to avoid a single-cell artifact if the solver needs 3D stencils.
- Compare Fluent 2D fields to the mid-plane or thickness-average field, and state which one is used.

## Our Solver Grid Requirement

Use a very fine grid for our solver:

- Minimum target: at least comparable to the completed Fluent transient reference, i.e. roughly `>= 80k` 2D-equivalent fluid cells or the closest 3D slab equivalent.
- Preferred target: push finer than the Fluent transient reference if memory/runtime allows.
- Do a short preflight run first only to confirm the repaired solver and output pipeline work; do not present the preflight as the final result.
- The production run must save its actual grid dimensions, physical cell sizes, particle counts, marker counts, and CFL/step diagnostics.

## Execution Plan After Approval

1. Inspect current bug-fix state and identify the intended in-repo runner/config for the ANSYS vertical flap FSI case.
2. Freeze a run manifest under:
   `validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/`
3. Build or select the very fine solver grid and save the exact config.
4. Run a short smoke/preflight case to catch import/config/output failures.
5. Run the production fine-grid solver to `t = 0.025 s`.
6. Export final fields and histories:
   - velocity components and velocity magnitude;
   - pressure;
   - solid/flap geometry or particle state;
   - interface markers / HIBM-MPM diagnostics if present;
   - mass balance, CFL, force, and displacement histories if available.
7. Interpolate our solver field and Fluent field onto a common comparison grid.
8. Generate comparison figures with identical axes and color scales:
   - Fluent velocity magnitude, fixed scale `0..28.1 m/s`;
   - our solver velocity magnitude, same fixed scale;
   - absolute difference;
   - relative difference where meaningful;
   - pressure comparison;
   - centerline and downstream profile CSV/plots;
   - flap/interface overlay.
9. Write an honest comparison report.

## Required Artifacts

All outputs must be saved under:

`validation_runs/ansys_vertical_flap_fsi/our_solver_fine_vs_fluent_2026-07-02/`

Required files:

- `run_manifest.json`
- `our_solver_config.*`
- `our_solver_summary.json`
- `our_solver_history.csv`
- `our_solver_final_fields.*`
- `comparison_metrics.json`
- `comparison_report.md`
- `figures/fluent_velocity_t0p025.png`
- `figures/our_solver_velocity_t0p025.png`
- `figures/velocity_abs_difference_t0p025.png`
- `figures/pressure_comparison_t0p025.png`
- `figures/flap_overlay_t0p025.png`
- `profiles/*.csv`

## Comparison Metrics

At minimum compute:

- velocity magnitude min/max/mean for Fluent and our solver;
- pressure min/max/mean for Fluent and our solver;
- L1/L2/Linf velocity magnitude error on common grid;
- signed and absolute pressure error on common grid;
- throat/downstream profile mismatch;
- mass balance or inlet/outlet flux mismatch if the solver exposes it;
- flap tip or interface displacement mismatch if available.

## Acceptance Criteria

1. Our solver production run reaches `t = 0.025 s` on the agreed fine grid.
2. The run saves enough artifacts to reproduce the exact grid, parameters, and postprocessing.
3. The comparison uses the locked Fluent transient FSI reference, not a stale or failed artifact.
4. Plots use the same geometry, axes, and color scale where direct visual comparison is claimed.
5. Any mismatch is reported as a mismatch, not tuned away.
6. If the run fails, the failure is captured with logs, last valid fields, diagnostics, and a concrete module-level bug hypothesis.

## Non-Goals

- No Fluent rerun unless the user explicitly asks for a new reference.
- No solver parameter tuning for visual agreement.
- No replacing HIBM-MPM or FSI coupling with a dedicated case-specific shortcut.
- No claims of Fluent parity unless supported by saved metrics and plots.
