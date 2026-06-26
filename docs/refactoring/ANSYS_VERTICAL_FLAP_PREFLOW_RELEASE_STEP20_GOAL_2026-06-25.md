# ANSYS Vertical Flap Preflow Release STEP20 Goal - 2026-06-25

## Background

Remote commit `e8feca95cfa68b533cce74e0f39ecac040f47631` already separated the ANSYS vertical flap STEP20 diagnostics into flow temporal, coupling settling, and promotion gates. That work correctly preserves:

- `best_flow_temporal_candidate = source_0p80_ramp2_step20`
- `best_combined_temporal_candidate = none`
- `promotion_candidate = none`
- `diagnostic_fallback_candidate = source_0p75_ramp5_step20`

It also generated fixed-solid STEP30 source temporal evidence showing that source/outlet flow can pass temporal gates when the solid is fixed and marker feedback is disabled. The remaining issue is coupled release after preflow.

The last implementation also exposed a schedule-indexing bug. The default `global` source schedule was intended to prevent source ramp restart after preflow, but the preflow phase currently computes the schedule index from `len(preflow_history) + step_index`. During preflow step `k`, both values equal `k`, so global indices become `0, 2, 4, ...` instead of `0, 1, 2, ...`. This makes ramp5 scenarios physically wrong and invalidates all checked-in fixed-solid global-index history fields.

## Primary Objective

Correct ANSYS vertical flap source schedule indexing, regenerate fixed-solid STEP30 artifacts with correct indices, extract reusable temporal-gate logic, then run a real coupled preflow-release STEP20 matrix to determine whether pre-established source/outlet flow reduces or removes the force/tip settling delay.

The goal must remain physically honest:

- no 50-step run
- no L2/L3 matrix
- no Fluent parity claim
- no full-field reinitialize candidate
- no solid material, damping, support-radius, or gate-threshold tuning
- no relaxation or coupling-subiteration implementation unless the preflow-release matrix proves that source/outlet flow is not the remaining blocker

## Phase 1 - Correct Source Schedule Indexing

Replace implicit schedule-index inference with explicit local/global/schedule indices.

Required runner API change:

```python
_flow_advance_current_step(
    ...,
    flow_phase="preflow" | "fsi",
    step_index_local=...,
    step_index_global=...,
)
```

Required calling rules:

```text
preflow:
    local = preflow_index
    global = preflow_index

FSI:
    local = fsi_step_index
    global = completed_preflow_steps + fsi_step_index
```

Required schedule-index rule:

```text
global scope:
    schedule_index = global_index

phase_local scope:
    schedule_index = local_index
```

Required report fields:

- `flow_phase`
- `flow_step_index_local`
- `flow_step_index_global`
- `flow_source_schedule_step_index`
- `flow_source_schedule_scope`
- `flow_source_ramp_restarted_after_preflow`

Expected behavior after fix:

```text
preflow ramp5 factors:
    0.15, 0.30, 0.45, 0.60, 0.75

preflow local index:
    0, 1, 2, 3, 4

preflow global index:
    0, 1, 2, 3, 4

global-scope first FSI:
    local = 0
    global = 5
    schedule = 5
    factor = 0.75
    restart = false

phase-local first FSI:
    local = 0
    global = 5
    schedule = 0
    factor = 0.15
    restart = true
```

## Phase 2 - Regenerate Fixed-Solid STEP30 Artifacts

Regenerate:

```text
validation_runs/ansys_vertical_flap_fsi/fixed_solid_source_temporal_diagnostics/
```

Required fixed-solid scenarios:

- `fixed_source_0p75_constant_step30`
- `fixed_source_0p80_constant_step30`
- `fixed_source_0p75_ramp2_step30`
- `fixed_source_0p80_ramp2_step30`
- `fixed_source_0p75_ramp5_step30`
- `projection_only_step30_baseline`
- `diagnostic_reinitialize_step30_upper_bound`

Required run settings:

- `step_count = 0`
- `preflow_steps = 30`
- solid fixed
- marker feedback disabled / not applied
- every scenario isolated in a worker subprocess

Required artifact assertions:

- every 30-step history has `flow_step_index_global == 0..29`
- every 30-step history has `flow_step_index_local == 0..29`
- every 30-step history has `flow_source_schedule_step_index == 0..29` for global scope
- ramp5 first five source factors are exactly `0.15, 0.30, 0.45, 0.60, 0.75`
- no global index gap, duplicate, or rollback exists
- full-field reinitialize row remains diagnostic-only / not applicable

## Phase 3 - Extract Shared Temporal Gate Logic

Move duplicated temporal classification logic into:

```text
tools/validation/ansys_vertical_flap_temporal_gates.py
```

Required pure functions:

- `classify_flow_temporal(...)`
- `classify_coupling_settling(...)`
- `classify_combined_temporal(...)`
- `select_flow_candidate(...)`
- `select_promotion_candidate(...)`

Required gate profiles:

- `STEP20_COUPLED_PROFILE`
- `STEP30_FIXED_SOLID_PROFILE`
- `STEP20_PREFLOW_RELEASE_PROFILE`

Shared baseline rules:

- post-warmup p999 in `[20, 29]`
- peak velocity `<= 40`
- post-warmup outlet ratio in `[0.75, 1.25]`
- last-window outlet ratio in `[0.80, 1.20]`
- invalid counts equal `0`

Existing temporal-gate unit tests must move from testing a matrix script import to testing this shared module directly.

## Phase 4 - Run Coupled Preflow-Release STEP20 Matrix

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_preflow_release_step20_matrix.py
```

Output directory:

```text
validation_runs/ansys_vertical_flap_fsi/preflow_release_coupling_diagnostics/
```

Required scenarios:

- `no_preflow_release20_source_0p80_ramp2`
- `preflow10_release20_source_0p80_ramp2`
- `preflow20_release20_source_0p80_ramp2`
- `preflow30_release20_source_0p80_ramp2`
- `preflow20_release20_source_0p75_constant`
- `preflow30_release20_source_0p75_constant`
- `preflow20_release20_source_0p75_ramp2`
- `preflow20_release20_source_0p80_ramp2_feedback_off`
- `preflow20_release20_source_0p80_ramp2_phase_local`

Each scenario must run in an isolated Python worker subprocess because Taichi/CUDA multi-scenario lifecycle instability has already been observed.

The matrix must answer:

- whether preflow reduces coupling settling time
- whether 10/20/30 preflow steps are sufficient
- whether fixed-solid best flow and coupled STEP20 best flow differ under release
- whether feedback drives release oscillation
- whether phase-local ramp restart worsens coupling

## Phase 5 - Required Transition Metrics

Each scenario must save both preflow and release histories, not final rows only.

Required history fields:

- `flow_phase`
- `phase_step`
- `global_step`
- `source_schedule_step`
- `source_factor`
- `velocity_peak_mps`
- `velocity_p999_mps`
- `velocity_outlet_flux_ratio`
- `pressure_outlet_flux_ratio`
- `pressure_min_pa`
- `pressure_max_pa`
- `projection_l2`
- `projection_max_abs`
- `marker_force_z_N`
- `mpm_external_force_z_N`
- `tip_dz_m`
- `root_max_displacement_m`
- `scatter_action_reaction_residual_N`
- `stress_invalid_marker_count`
- `scatter_invalid_marker_count`
- `feedback_invalid_marker_count`
- `fluid_projection_consumed_feedback`
- `no_slip_projected_residual_mps`

Required transition summary fields:

- `preflow_final_p999_mps`
- `preflow_final_outlet_ratio`
- `preflow_final_marker_force_z_N`
- `release_step1_marker_force_z_N`
- `release_step1_tip_dz_m`
- `release_step1_force_jump_N`
- `release_step1_force_ratio`
- `release_first_permanently_negative_force_step`
- `release_first_permanently_negative_tip_step`
- `release_first_permanently_valid_step`
- `release_longest_consecutive_pass_steps`
- `release_last10_min_p999_mps`
- `release_last10_mean_outlet_ratio`
- `release_last10_force_sign_ok`
- `release_last10_tip_sign_ok`

Required state-continuity checks:

- last preflow global index + 1 equals first FSI global index
- last preflow source factor equals first FSI source factor for global scope
- first FSI pressure reset is false after preflow
- first FSI full-field reinitialize is false
- `flow_source_ramp_restarted_after_preflow = false` for global scope
- `flow_source_ramp_restarted_after_preflow = true` for the phase-local control

## Phase 6 - Promotion Gate

Do not loosen existing gates.

Preflow flow gate:

- preflow final/last-window p999 in `[20, 29]`
- peak velocity `<= 40`
- outlet ratio in `[0.80, 1.20]`
- invalid counts equal `0`
- flow trend sufficiently settled

Release flow gate:

- release flow temporal strict/soft
- last-10 p999 `>= 20`
- last-10 outlet ratio in `[0.80, 1.20]`
- peak velocity `<= 40`
- invalid counts equal `0`

Release coupling gate:

- last-10 marker force z `< 0`
- last-10 tip dz `< 0`
- root displacement `<= 1e-8`
- action-reaction residual within tolerance

Promotion condition:

- preflow flow ready
- release flow temporal pass
- release combined temporal strict/soft
- no full-field reset
- no phase-local restart

## Phase 7 - Worker Reliability

Add worker provenance to fixed-solid and preflow-release matrix rows:

- `worker_mode = isolated_subprocess`
- `worker_returncode`
- `worker_timed_out`
- `worker_elapsed_s`
- `worker_stdout_log`
- `worker_stderr_log`

Use:

```python
WORKER_TIMEOUT_S = 900
```

If a worker times out or fails, keep exact stdout/stderr evidence in a `failures/` subdirectory and keep the matrix row reviewable.

## Required Tests

Add or update tests for:

- preflow ramp5 source factors: `0.15, 0.30, 0.45, 0.60, 0.75`
- preflow local/global/schedule indices: `0..4`
- global first FSI schedule continuity after five preflow steps
- phase-local first FSI restart flag and source factor
- fixed-solid histories have global indices `0..29`
- fixed-solid ramp5 first five source factors are correct
- fixed-solid rows include worker provenance
- shared temporal gate module covers strict, soft, failed, last-window failed, missing history, ramp warmup, flow strict plus coupling unsettled, flow strict plus coupling settled, and idempotence
- preflow-release matrix artifacts include required scenarios, preflow/release histories, state-continuity fields, promotion fields, and scope-limit documentation

## Verification Commands

Use the trusted Taichi Python on this machine:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile cases\ansys_vertical_flap_fsi.py benchmarks\official\solid_mpm_fsi_runner.py tools\validation\ansys_vertical_flap_temporal_gates.py validation_runs\ansys_vertical_flap_fsi\scripts\run_fixed_solid_source_temporal_matrix.py validation_runs\ansys_vertical_flap_fsi\scripts\run_preflow_release_step20_matrix.py
& 'D:\working\taichi\env\python.exe' -m unittest tests.tools.test_ansys_vertical_flap_temporal_gate -v
& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_fixed_solid_source_temporal_artifacts -v
& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_vertical_flap_preflow_release_step20_artifacts -v
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_fixed_solid_source_temporal_matrix.py
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_preflow_release_step20_matrix.py
git diff --check
```

## Completion Criteria

The goal is complete only when:

1. This detailed goal file exists and the active goal references it.
2. The source schedule no longer double-counts preflow history.
3. Fixed-solid STEP30 artifacts are regenerated with correct local/global/schedule indices.
4. The fixed-solid ramp5 history uses the correct ramp factors.
5. Shared temporal gate logic exists under `tools/validation/`.
6. STEP20 source-candidate and fixed-solid scripts use shared temporal gate functions where practical.
7. The preflow-release STEP20 matrix script exists.
8. Real preflow-release STEP20 artifacts are generated and reviewable.
9. Promotion remains blocked unless the strict combined criteria truly pass.
10. Documentation records what was run, what failed, what passed, and what cannot be claimed.
11. Focused compile/tests pass, or any failure is recorded with exact command output and reason.
12. The branch is committed and pushed to GitHub.
