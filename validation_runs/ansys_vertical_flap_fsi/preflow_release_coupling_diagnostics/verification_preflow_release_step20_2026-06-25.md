# ANSYS Vertical-Flap Preflow-Release STEP20 Verification

Date: 2026-06-25

This EasyFsi diagnostic runs coupled STEP20 release scenarios after 0/10/20/30 fixed-solid preflow steps. It uses isolated worker subprocesses with timeout_s = 900 because Taichi/CUDA multi-scenario lifecycle instability was observed in the fixed-solid matrix. It does not run 50 steps and does not claim Fluent parity.

## Result

best_preflow_release_candidate = none

best_release_flow_candidate = preflow20_release20_source_0p80_ramp2

best_release_coupling_candidate = none

best_release_promotion_candidate = none

promotion_candidate_count = 0

candidate_status = no_promotion_candidate

## Findings

- Source schedule indexing is recorded separately as local, global, and schedule indices. Global-scope release rows continue after preflow; the phase-local scenario intentionally restarts and is diagnostic-only.
- The shared temporal gate treats any last-window failure as a non-strict result. A run can no longer report strict status while also reporting last-window failures.
- The STEP20 release flow is stable across the matrix, but every coupled release still fails the combined temporal/coupling gate; the remaining issue is force/tip settling after MPM release, not source/outlet flow establishment.

## Scope Limits

- No 50-step run was performed.
- No L2/L3 matrix was run.
- No Fluent parity claim is made.
- No solid material, damping, support-radius, or gate threshold was tuned.
- Full-field reinitialize and phase-local restart controls are diagnostic only.
