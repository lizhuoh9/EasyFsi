# ANSYS Vertical Flap Preflow Release Coupling Gate Goal - 2026-06-25

## Background

The latest remote branch state has already recorded the STEP20 temporal-gate query status. The current combined temporal gate honestly reports `no_temporal_candidate`. That result must be preserved because no STEP20 source scenario is currently proven as a promotion-ready coupled FSI candidate.

The diagnosis also shows that this combined result is too coarse for the next engineering step. The source/outlet flow can be temporally usable while the solid force/tip coupling is still settling. In particular, `source_0p80_ramp2_step20` has a stable last-window source-driven flow and outlet balance, but it is not allowed to become a promotion candidate because the combined coupled temporal gate still fails due early force/tip sign transients.

This goal separates those meanings in code, tests, artifacts, and documentation without changing the physical solver model or loosening the existing promotion criteria.

## Primary Objective

Implement an honest ANSYS vertical flap validation classification layer that reports three separate outcomes:

1. Flow temporal gate: whether a source-driven flow scenario sustains the desired flow/outlet behavior after warmup.
2. Coupling settling gate: whether the coupled solid response has settled into physically expected force/tip signs.
3. Promotion gate: whether a scenario satisfies the original combined temporal criteria and can be treated as a promotion-ready candidate.

The implementation must preserve the current conclusion that there is no combined temporal promotion candidate. It must also expose `source_0p80_ramp2_step20` as the best flow-temporal diagnostic candidate and `source_0p75_ramp5_step20` only as the diagnostic fallback/final-gate candidate, not as a promotion candidate.

## Required Code Changes

### STEP20 source candidate matrix classification

Update `validation_runs/ansys_vertical_flap_fsi/scripts/run_source_candidate_step20_matrix.py` so each matrix row includes separate flow and coupling classification fields in addition to the existing combined temporal fields.

Required flow fields:

- `flow_temporal_status`
- `flow_temporal_fail_reasons`
- `flow_post_warmup_failed_step_count`
- `flow_last_window_failed_step_count`
- `flow_last_window_min_p999_mps`
- `flow_last_window_mean_outlet_ratio`

Required coupling fields:

- `coupling_settling_status`
- `coupling_first_permanently_negative_force_step`
- `coupling_first_permanently_negative_tip_step`
- `coupling_first_permanently_valid_step`
- `coupling_longest_consecutive_pass_steps`
- `coupling_last_window_force_sign_ok`
- `coupling_last_window_tip_sign_ok`

Required promotion/fallback fields:

- `promotion_candidate_status`
- `promotion_candidate`
- `diagnostic_fallback_candidate`

Required top-level candidate semantics:

- `best_final_gate_candidate = source_0p75_ramp5_step20`
- `best_flow_temporal_candidate = source_0p80_ramp2_step20`
- `best_combined_temporal_candidate = none`
- `promotion_candidate = none`
- `diagnostic_fallback_candidate = source_0p75_ramp5_step20`

Compatibility is allowed, but compatibility fields must be clearly described:

- `best_candidate_step20_history.csv` may remain as a compatibility fallback artifact.
- The summary must explicitly state that this compatibility history is a diagnostic fallback, not a promotion-ready candidate.

### Candidate history artifacts

Write or preserve the following history artifacts when reclassifying or regenerating the STEP20 matrix:

- `best_final_gate_candidate_history.csv`
- `best_flow_temporal_candidate_history.csv`
- `best_combined_temporal_candidate_history.csv`
- `best_candidate_step20_history.csv`

If there is no combined temporal candidate, `best_combined_temporal_candidate_history.csv` must be empty or explicitly header-only. It must not silently contain fallback data.

### Source schedule continuity

Fix the source ramp schedule so source ramping is continuous from preflow into FSI by default.

Current behavior to fix:

- Preflow advances with `_flow_advance_current_step(..., step_index=preflow_index)`.
- The FSI loop restarts at `step_index=0`.
- That causes the source ramp to restart immediately after preflow.

Required behavior:

- Add `flow_inlet_source_schedule_scope: str = "global"` to the vertical flap configuration.
- Supported values:
  - `global`: FSI source factor uses `global_flow_step_index = len(preflow_history) + step_index`.
  - `phase_local`: preserve the old phase-local behavior for compatibility.
- Report the following fields in per-step flow diagnostics:
  - `flow_step_index_local`
  - `flow_step_index_global`
  - `flow_source_schedule_scope`
  - `flow_source_ramp_restarted_after_preflow`

Required test case:

- With `preflow_steps=5`, `flow_inlet_source_ramp_steps=5`, and `flow_inlet_source_schedule_scope="global"`, the first FSI source factor must remain at the target factor instead of restarting to the first ramp value. For a `0.75` target this means first FSI factor `0.75`, not `0.15`.

### Fixed-solid flow temporal matrix

Add a fixed-solid source temporal matrix script:

`validation_runs/ansys_vertical_flap_fsi/scripts/run_fixed_solid_source_temporal_matrix.py`

The script must run fixed-solid/preflow-only diagnostics and write artifacts under:

`validation_runs/ansys_vertical_flap_fsi/fixed_solid_source_temporal_diagnostics/`

Required scenarios:

- `fixed_source_0p75_constant_step30`
- `fixed_source_0p80_constant_step30`
- `fixed_source_0p75_ramp2_step30`
- `fixed_source_0p80_ramp2_step30`
- `fixed_source_0p75_ramp5_step30`
- `projection_only_step30_baseline`
- `diagnostic_reinitialize_step30_upper_bound`

Run settings:

- `step_count = 0`
- `preflow_steps = 30`
- fixed solid, no solid advancement
- feedback disabled or demonstrably not applied

Fixed-solid flow gate:

- Post-warmup `p999` must be in `[20, 29]`.
- Peak velocity must be `<= 40`.
- `velocity_outlet_flux_ratio` must be in `[0.75, 1.25]`.
- Stress invalid count must be `0`.
- Last-10 `p999` must be `>= 20`.
- Last-10 outlet ratio must be in `[0.80, 1.20]`.

The fixed-solid matrix is diagnostic evidence only. It must not be used to claim coupled FSI validation.

## Required Tests

Add logic-level tests in:

`tests/tools/test_ansys_vertical_flap_temporal_gate.py`

The tests must cover at least:

- strict flow temporal pass
- soft flow temporal pass with exactly 2 post-warmup failures
- failed flow temporal pass with 3 post-warmup failures
- last-window flow failure
- missing history / not applicable classification
- ramp warmup behavior
- flow strict plus coupling unsettled remains non-promotable
- flow strict plus coupling settled becomes eligible for promotion when the combined temporal criteria also pass
- idempotent `--reclassify-existing` behavior at the logic level

Add or update tests for source schedule continuity so the global preflow-to-FSI source ramp does not restart.

Add or update artifact tests so they assert the separated candidate semantics:

- best flow temporal candidate is `source_0p80_ramp2_step20`
- best combined temporal candidate is absent/none
- promotion candidate is absent/none
- diagnostic fallback candidate is `source_0p75_ramp5_step20`

## Required Documentation

Update the validation documentation and archived run notes so a reviewer can see:

- what was simulated
- which solver path was used
- which artifacts were generated
- why `source_0p80_ramp2_step20` is a flow-temporal candidate only
- why `source_0p75_ramp5_step20` is a diagnostic fallback only
- why there is still no combined/promotion candidate
- any errors or limitations discovered while running or reclassifying the diagnostics

Do not claim Fluent parity, full physical validation, or promotion readiness unless the combined temporal gate actually passes.

## Non-Goals

Do not do any of the following in this goal:

- Do not run a coupled 50-step matrix.
- Do not tune solid material, damping, forcing, or gate thresholds to force a green result.
- Do not add pressure/velocity shortcuts that fake a jet or hide solver instability.
- Do not hard-reset flow state after preflow to make a graph look better.
- Do not replace the current solver with Fluent or any external CFD tool.
- Do not report the diagnostic fallback as a promotion candidate.

## Verification Commands

Run focused unit and artifact checks after implementation:

```powershell
python -m unittest tests.tools.test_ansys_vertical_flap_temporal_gate -v
python -m unittest tests.tools.test_ansys_vertical_flap_source_candidate_temporal_gate_artifacts -v
python validation_runs/ansys_vertical_flap_fsi/scripts/run_source_candidate_step20_matrix.py --reclassify-existing
python validation_runs/ansys_vertical_flap_fsi/scripts/run_fixed_solid_source_temporal_matrix.py
```

If the local Python shim is unreliable, use the repository's validated explicit interpreter path for this environment.

## Completion Criteria

The goal is complete only when:

1. The detailed goal file exists in `docs/refactoring/`.
2. The STEP20 matrix artifacts expose separated flow, coupling, and promotion gates.
3. `best_flow_temporal_candidate` is `source_0p80_ramp2_step20`.
4. `best_combined_temporal_candidate` and `promotion_candidate` remain none.
5. `diagnostic_fallback_candidate` is `source_0p75_ramp5_step20`.
6. The global source schedule continuity bug is fixed and tested.
7. The fixed-solid STEP30 source temporal matrix script exists and has generated artifacts.
8. Documentation explains the diagnostic scope and limitations without overclaiming.
9. Focused tests pass, or any failure is documented with exact command output and reason.
10. The branch is committed with `fix: separate ANSYS flow and coupling temporal gates` and pushed to GitHub after verification.
