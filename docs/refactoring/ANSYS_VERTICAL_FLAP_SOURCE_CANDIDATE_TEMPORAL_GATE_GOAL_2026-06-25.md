# ANSYS Vertical-Flap Source Candidate Temporal Gate Goal - 2026-06-25

## Goal Summary

Upgrade the ANSYS vertical-flap STEP20 source-candidate artifacts from a
final-row candidate gate to a temporal candidate gate. The prior STEP20 matrix
proved that several non-full-reset rows can end inside the 20-step flow gate,
but per-step history shows that the selected `source_0p75_ramp5_step20` row is
not uniformly stable through the full interval. This goal must prevent final
recovery from being treated as long-run stability.

This goal is a reclassification and artifact-contract hardening pass. It does
not run 50 steps, does not tune solid parameters, and does not claim Fluent
parity.

## Starting Evidence

Current remote branch:

```text
solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25
```

Remote HEAD observed before this goal:

```text
d7f7e84b696c9390f45c1f9bf34a8efbfb7a3b42
```

STEP20 implementation commit:

```text
21d1eb1f4de1f6196af715c799222b1ce5c26d14
```

Prior STEP20 final-row result:

```text
candidate_status = candidate_found
best_candidate = source_0p75_ramp5_step20
best_candidate final p999 = 24.7323 m/s
best_candidate final peak = 32.0971 m/s
best_candidate max peak = 32.1574 m/s
best_candidate final velocity_outlet_flux_ratio = 1.0453
```

Reviewer concern:

```text
candidate_found is based primarily on final-row gates.
source_0p75_ramp5_step20 has intermediate p999 values below 20 m/s.
The existing source_strength_0p75_step20_history.csv points to
source_0p75_constant_step20, not the best candidate.
```

Therefore the next step must harden candidate classification with per-step
history checks and repair the history artifact naming.

## Non-Goals

Do not run a 50-step simulation in this goal.

Do not rename this work as Fluent validation or Fluent parity.

Do not tune solid parameters, material properties, grid resolution, support
radius, marker geometry, or official boundary metadata.

Do not treat `sustained_inlet_predictor` as a physical predictor/advection path.

Do not delete prior STEP20 raw results. Reclassify them honestly and preserve
evidence.

Do not make remote CI pass claims unless a real run URL/status/head SHA is
available.

## Required Code Changes

Update:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_source_candidate_step20_matrix.py
```

The script must add temporal analysis over per-step histories and write those
results into the matrix rows, summary, and verification.

The default behavior may still run the solver, but the script must also support
a deterministic artifact-only reclassification mode that reads the committed
STEP20 matrix and history artifacts and rewrites the derived temporal fields
without rerunning the expensive solver. The intended flag is:

```powershell
& $python validation_runs\ansys_vertical_flap_fsi\scripts\run_source_candidate_step20_matrix.py --reclassify-existing
```

This mode must not fabricate runtime data; it may only recompute derived
classification fields from existing `source_candidate_step20_matrix.json` and
`source_candidate_step20_history.json`.

## Required Temporal Gate

For each non-full-reset completed row, compute:

```text
temporal_warmup_steps = max(source_ramp_steps + 2, 5)
temporal_evaluation_start_step = temporal_warmup_steps + 1
temporal_last_window_steps = 5
```

Strict temporal pass requires all post-warmup steps to satisfy:

```text
velocity_p999_mps >= 20
velocity_peak_mps <= 40
0.75 <= velocity_outlet_flux_ratio <= 1.25
marker_force_z_N < 0
tip_dz_m < 0
stress_invalid_marker_count = 0
scatter_invalid_marker_count = 0
feedback_invalid_marker_count = 0
```

Soft temporal pass allows up to two post-warmup transition failures, but the
last five steps must all satisfy the same temporal gate.

Rows should receive:

```text
temporal_candidate_status = temporal_strict | temporal_soft |
                            temporal_failed | temporal_not_applicable
temporal_fail_reasons = [...]
temporal_post_warmup_failed_step_count
temporal_last_window_failed_step_count
temporal_last_window_min_p999_mps
temporal_last_window_mean_velocity_outlet_flux_ratio
temporal_last_window_force_sign_ok
temporal_last_window_tip_sign_ok
```

Full-field reinitialize rows remain diagnostic-only and must not become
temporal candidates.

## Required Candidate Ranking

The top-level candidate selection must no longer use final-row `candidate`
status alone.

Ranking order:

```text
1. temporal_strict rows
2. temporal_soft rows
3. final-row candidates with temporal_failed only as diagnostic fallback
```

Top-level fields must include:

```text
best_candidate
best_final_gate_candidate
best_temporal_candidate
candidate_status
temporal_candidate_status
temporal_best_candidate_status
temporal_candidate_count
```

If no temporal strict/soft row exists, then:

```text
candidate_status = no_temporal_candidate
next_action = stop before 50-step; refine source/outlet model or run STEP30
              temporal matrix
```

If a temporal strict/soft row exists, then:

```text
candidate_status = temporal_candidate_found
next_action = run STEP30 temporal matrix before any 50-step run
```

Even when a temporal candidate exists, the summary must still state that 50-step
is not approved by this goal.

## Required History Artifacts

The existing `source_strength_0p75_step20_history.csv` name is misleading
because it points to `source_0p75_constant_step20`, not necessarily the best
candidate. Keep it only as a compatibility artifact if needed, but add explicit
history outputs:

```text
validation_runs/ansys_vertical_flap_fsi/source_candidate_step20_diagnostics/histories/
  source_0p75_constant_step20_history.csv
  source_0p80_constant_step20_history.csv
  source_0p75_ramp2_step20_history.csv
  source_0p80_ramp2_step20_history.csv
  source_0p75_ramp5_step20_history.csv
best_candidate_step20_history.csv
all_candidate_step20_histories.csv
```

`best_candidate_step20_history.csv` must match the top-level `best_candidate`.
If no temporal candidate exists, it may point to the best final-gate diagnostic
fallback, but the summary must say so.

## Required Documentation Updates

Update:

```text
docs/VALIDATION.md
```

The validation docs must say that STEP20 now has both final-row and temporal
classification, and that the temporal gate is required before any STEP30/STEP50
promotion.

Update the STEP20 verification artifact:

```text
validation_runs/ansys_vertical_flap_fsi/source_candidate_step20_diagnostics/verification_source_candidate_step20_2026-06-25.md
```

It must record:

```text
implementation commit = 21d1eb1f4de1f6196af715c799222b1ce5c26d14
pre-goal remote HEAD = d7f7e84b696c9390f45c1f9bf34a8efbfb7a3b42
local gh unauthenticated if remote Actions still cannot be queried
```

## Required Tests

Add:

```text
tests/integration/test_ansys_vertical_flap_source_candidate_temporal_gate_artifacts.py
```

The tests must verify:

```text
matrix JSON has temporal top-level fields
each completed non-diagnostic row has temporal fields
full-field reinitialize rows have temporal_not_applicable or remain excluded
candidate_status is not candidate_found based only on final row
best_candidate history CSV exists
best_candidate_step20_history.csv scenario matches top-level best_candidate
per-scenario history CSVs exist for all final-row source candidates
all_candidate_step20_histories.csv exists and contains multiple scenarios
summary includes temporal_candidate_status and best_candidate_history_csv
verification says no 50-step run and no Fluent parity
```

Existing STEP20 artifact tests must be updated if their expected status values
are too narrow.

## Verification Commands

Run:

```powershell
& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_source_candidate_step20_matrix.py --reclassify-existing
& 'D:\working\taichi\env\python.exe' -m py_compile validation_runs\ansys_vertical_flap_fsi\scripts\run_source_candidate_step20_matrix.py tests\integration\test_ansys_vertical_flap_source_candidate_step20_artifacts.py tests\integration\test_ansys_vertical_flap_source_candidate_temporal_gate_artifacts.py
& 'D:\working\taichi\env\python.exe' -m unittest -v tests.integration.test_ansys_vertical_flap_source_candidate_step20_artifacts tests.integration.test_ansys_vertical_flap_source_candidate_temporal_gate_artifacts
& 'D:\working\taichi\env\python.exe' -m unittest tests.integration.test_ansys_official_half_domain_archive_consistency tests.integration.test_ansys_vertical_flap_postrepair_artifacts tests.integration.test_ansys_vertical_flap_flow_collapse_artifacts tests.integration.test_ansys_vertical_flap_sustained_flow_driver_artifacts tests.integration.test_ansys_vertical_flap_source_outlet_balance_artifacts tests.integration.test_ansys_vertical_flap_source_candidate_step20_artifacts tests.integration.test_ansys_vertical_flap_source_candidate_temporal_gate_artifacts -v
git diff --check
```

Also run a changed-file credential scan before commit.

## Git And Push Requirements

Commit message:

```text
fix: add temporal gate for ANSYS source candidates
```

Push to the current tracked GitHub branch:

```text
solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25
```

After push, attempt to query GitHub Actions. If local `gh` is unauthenticated,
record that exact limitation in the verification artifact and push the
documentation update.
