# ANSYS Vertical-Flap Fixed-Solid Source Temporal Verification

Date: 2026-06-25

This diagnostic runs STEP30 fixed-solid/preflow-only source scenarios with `step_count=0` and `preflow_steps=30`. The MPM solid is not advanced and marker feedback is not applied, so these artifacts only test the source/outlet flow path before coupled release.

## Result

best_fixed_solid_flow_candidate = fixed_source_0p75_constant_step30

candidate_status = candidate_found

fixed_solid_flow_candidate_count = 4

## Runtime Note

The matrix runner executes each scenario in a separate Python worker process. A single-process trial exited after writing two scenario histories, while the same next scenario completed when run by itself; the worker isolation keeps the generated data tied to real EasyFsi solver runs without depending on Taichi/CUDA multi-run lifecycle behavior. Each worker has timeout_s = 900 and records return code, timeout status, elapsed time, stdout log, and stderr log in the matrix row.

## Schedule Note

The fixed-solid histories record phase-local, global, and source schedule indices. The ramp5 scenario now advances schedule indices 0, 1, 2, 3, 4 with source factors 0.15, 0.30, 0.45, 0.60, 0.75; it does not skip to 0, 2, 4 during preflow.

## Scope Limits

- No coupled FSI step was run.
- These artifacts are not coupled FSI validation.
- No Fluent parity claim is made.
- Solid material, damping, and promotion gates were not tuned.
- Diagnostic full-field reinitialize rows are not fixed-solid source candidates.
