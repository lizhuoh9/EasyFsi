# ANSYS Vertical-Flap Fixed-Solid Load Temporal Verification

Date: 2026-06-26

This EasyFsi diagnostic runs fixed-solid STEP60 load scenarios with `step_count=0` and `preflow_steps=60`. It records face-resolved marker force, hydrodynamic load sign statistics, and marker/scatter action-reaction residuals. The MPM solid is not advanced and no coupled release or 50-step Fluent-parity run is performed.

## Result

best_fixed_solid_load_candidate = fixed_load_0p80_ramp2_step60

fixed_solid_load_candidate_count = 1

candidate_status = candidate_found

Candidate rows must be completed, non-diagnostic, `flow_temporal_strict`, and `load_temporal_strict`; full-field or inlet-reinitialize diagnostic upper-bound rows are never release candidates.

## Runtime Finding

The first matrix attempt exposed a fixed-solid preflow reporting bug: the runner emitted scatter residuals but not the scatter marker-count columns expected by the matrix summary path. The archived rerun uses real solver histories after that reporting gap was fixed.

## Scope Limits

- No coupled FSI release was run.
- No 50-step run was performed.
- No Fluent parity claim is made.
- No solid material, damping, support-radius, or gate threshold was tuned.
- Full-field reinitialize rows are diagnostic only.
