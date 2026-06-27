# ANSYS Vertical-Flap Selected Formulation Coupled 5-Step Goal

Date: 2026-06-27

Source checkpoint: remote branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
at commit `b0f423d723f128e9a3771e02f01f48f4611c14d6`.

Short goal reference for Codex `/goal`:

```text
Implement the detailed ANSYS vertical-flap selected-formulation coupled
5-step smoke contract in
docs/refactoring/ANSYS_VERTICAL_FLAP_SELECTED_FORMULATION_COUPLED_5STEP_GOAL_2026-06-27.md.
First harden the current 1-step anchor-injected checkpoint, then extend the
selected formulation coupled-smoke artifact to retain the 1-step row and add a
true requested 5-step row. Keep all gates fail-closed, add step-history
reducers and artifact tests, regenerate committed artifacts, run focused local
verification, commit, push, and verify GitHub Actions. Do not claim 50-step
validation or Fluent parity.
```

## Current Checkpoint

The latest committed checkpoint correctly completed the anchor-injection
stage. The selected formulation coupled-smoke path now opens the real coupled
runner, installs the fixed-solid selected anchor map into live markers, and
removes the previous stress-sampling failure.

Current artifact-backed state:

```text
candidate_status = selected_formulation_coupled_smoke_pending
smoke_status = blocked_requested_5step_not_completed
requested_step_count = 5
completed_step_count = 1
invalid_marker_count_max = 0.0
pressure_pair_anchor_active_marker_count_min = 24
anchor_selected_marker_count_min = 24
anchor_fallback_marker_count_max = 0.0
one_sided_marker_count_min = 24
one_sided_anchor_fallback_marker_count_max = 0.0
```

The previous blocker:

```text
blocked_invalid_marker_sampling
```

has been retired for the current artifact. The remaining blocker is only that
the requested 5-step smoke has not actually completed.

The current technical risk has moved from:

```text
sampling infrastructure / anchor injection failure
```

to:

```text
multi-step coupled stability
```

This goal must keep that distinction explicit.

## Objective

Advance the ANSYS vertical-flap selected-formulation coupled smoke from a
1-step anchor-injected preflight to a true requested 5-step coupled smoke while
preserving the current 1-step evidence as its own artifact row.

The preferred artifact shape is two rows:

```text
selected_formulation_coupled_preflight_1step
selected_formulation_coupled_smoke_5step
```

The first row proves the anchor-injected preflight still works. The second row
is the new requested smoke completion candidate.

If the 5-step row passes all gates, the artifact may move to:

```text
candidate_status = selected_formulation_coupled_smoke_passed
```

If the 5-step row fails any gate, the artifact must remain pending and record
the exact first failing step and first failing gate.

## Non-Goals

Do not implement or claim:

- 10-step, 30-step, 50-step, or longer coupled validation
- Fluent parity
- final displacement parity
- material parameter tuning
- geometry tuning
- selected formulation candidate changes
- pressure, velocity, force, displacement, or marker-result hardcoding
- marker, anchor, one-sided, residual, velocity, pressure, or finite-field gate
  relaxation
- a pass status when the 5-step row does not satisfy every gate
- heavy coupled runner execution in GitHub Actions

Keep `no_fluent_parity_claim` active in every passing or pending state.

## Phase 0 - Harden the Current 1-Step Checkpoint

Tighten:

```text
tests/integration/test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts.py
```

The artifact test should no longer broadly accept `blocked_invalid_marker_sampling`
or `not_run` as equivalent to the current checkpoint. The current 1-step row
must be locked as:

```text
candidate_status = selected_formulation_coupled_smoke_pending
smoke_status = blocked_requested_5step_not_completed
requested_step_count = 5
completed_step_count = 1
invalid_marker_count_max = 0
pressure_pair_anchor_active_marker_count_min >= 24
anchor_selected_marker_count_min >= 24
anchor_fallback_marker_count_max = 0
one_sided_marker_count_min >= 24
one_sided_anchor_fallback_marker_count_max = 0
```

The acceptance object must record:

```text
finite_fields = true
no_marker_invalid = true
anchor_selected_all = true
anchor_fallback_zero = true
one_sided_complete = true
one_sided_fallback_zero = true
residual_within_tolerance = true
completed_requested_steps = false
accepted = false
```

The scenario diagnostics/history must also show:

```text
stress_invalid_marker_count = 0
stress_valid_marker_count = 24
primary_face_invalid_marker_count = 0
secondary_face_invalid_marker_count = 0
one_sided_pressure_marker_count = 24
surface_feedback_updated_marker_count = 24
```

## Phase 1 - Extend the Artifact Runner to Two Rows

Update:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_selected_formulation_coupled_smoke.py
```

The script currently uses:

```text
REQUESTED_STEP_COUNT = 5
DIAGNOSTIC_STEP_COUNT = 1
SCENARIO = selected_formulation_coupled_smoke_5step
```

Change it so it defines two scenarios:

```text
PREFLIGHT_SCENARIO = selected_formulation_coupled_preflight_1step
SMOKE_SCENARIO = selected_formulation_coupled_smoke_5step
PREFLIGHT_STEP_COUNT = 1
REQUESTED_STEP_COUNT = 5
```

Both rows must use the same selected formulation:

```text
reference_formulation_candidate = anchored_dual_face_pressure_pair_with_per_face_one_sided
pressure_pair_policy_candidate = baseline_anchored_cell_pair
one_sided_pressure_policy_candidate = per_face_mirrored
selected_anchor_markers_source = validation_runs/ansys_vertical_flap_fsi/traction_fixed_solid_selected_formulation_diagnostics/marker_diagnostics/fixed_solid_selected_per_face_one_sided_probe0p51_markers.json
```

Both rows must keep source provenance:

```text
reference_selection_source
reference_selection_source_sha256
fixed_solid_selected_formulation_source
fixed_solid_selected_formulation_source_sha256
selected_anchor_markers_source
selected_anchor_markers_source_sha256
shared_snapshot_manifest
shared_snapshot_sha256
pressure_pair_anchor_map_sha256
pressure_pair_anchor_source_flow_snapshot_sha256
pressure_pair_anchor_source_marker_geometry_sha256
pressure_pair_anchor_current_marker_geometry_sha256
```

## Phase 2 - Add Step-History Reducers

For each row, derive explicit per-step arrays from `report["history"]`:

```text
invalid_marker_count_by_step
one_sided_marker_count_by_step
anchor_selected_marker_count_by_step
anchor_fallback_marker_count_by_step
one_sided_anchor_fallback_marker_count_by_step
force_action_reaction_residual_by_step
max_velocity_by_step
max_pressure_abs_by_step
max_displacement_by_step
```

Also derive:

```text
first_failed_step
first_failed_gate
first_failed_gate_value
completed_step_count
```

The reducers should make a failed 5-step run diagnosable from the matrix and
summary without inspecting thousands of lines of scenario JSON.

## Phase 3 - 5-Step Acceptance Gate

For the 5-step row, pass only when all of the following hold:

```text
completed_step_count == requested_step_count == 5
fluid_finite == true
pressure_finite == true
solid_position_finite == true
invalid_marker_count_max == 0
pressure_pair_anchor_active_marker_count_min >= 24
anchor_selected_marker_count_min >= 24
anchor_fallback_marker_count_max == 0
one_sided_marker_count_min >= 24
one_sided_anchor_fallback_marker_count_max == 0
force_action_reaction_residual_max_n <= 1e-8
max_velocity_mps <= 1e6
max_pressure_pa <= 1e9
max_displacement_m is finite
```

If the 5-step row passes:

```text
candidate_status = selected_formulation_coupled_smoke_passed
historical_blockers_retired includes coupled_fsi_validation_pending
candidate_blockers = long_coupled_validation_pending, no_fluent_parity_claim
```

If it fails:

```text
candidate_status = selected_formulation_coupled_smoke_pending
candidate_blockers includes coupled_fsi_validation_pending
candidate_blockers includes no_fluent_parity_claim
candidate_blockers includes the exact 5-step smoke_status
```

Valid fail-closed smoke statuses include:

```text
blocked_nan_or_inf
blocked_invalid_marker_sampling
blocked_anchor_fallback
blocked_one_sided_incomplete
blocked_force_residual
blocked_velocity_threshold
blocked_pressure_threshold
blocked_solid_displacement_threshold
blocked_requested_5step_not_completed
not_run
```

Do not invent a pass by dropping failed steps or weakening thresholds.

## Phase 4 - Summary and History Output

Update:

```text
traction_selected_formulation_coupled_smoke_matrix.json
traction_selected_formulation_coupled_smoke_matrix.csv
traction_selected_formulation_coupled_smoke_history.json
traction_selected_formulation_coupled_smoke_summary.md
scenario_diagnostics/*.json
CHECKSUMS.sha256
```

The summary must include:

```text
candidate_status
preflight row smoke_status
5-step row smoke_status
requested_step_count
completed_step_count for both rows
invalid_marker_count_max for both rows
active blockers
no 50-step validation claim
no Fluent parity claim
```

The summary should include a compact per-step table for the 5-step row:

```text
step | invalid | one-sided | anchor selected | anchor fallback | force residual | max velocity | max pressure | max displacement
```

## Phase 5 - Artifact Tests

The artifact tests must verify:

```text
scenario_count == 2
rows include selected_formulation_coupled_preflight_1step
rows include selected_formulation_coupled_smoke_5step
source reference-selection SHA matches file
source fixed-solid selected SHA matches file
selected anchor marker source SHA matches file
selected formulation candidate exact
preflight row remains a 1-step anchor-injection pass / requested-5 pending checkpoint
5-step row either passes every gate or fails closed with exact first_failed_step and first_failed_gate
candidate_status exact for pass or pending
candidate_blockers exact for pass or pending
historical blockers retired only when 5-step passes
no 50-step claim
no Fluent parity claim
checksum covers matrix, CSV, history, summary, and scenario diagnostics
```

CI should continue to run only cheap checks:

```text
py_compile validation_runs/.../run_traction_selected_formulation_coupled_smoke.py
unittest tests.integration.test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts
```

Do not add the heavy coupled runner itself to GitHub Actions.

## Phase 6 - Local Verification

Use the trusted local interpreter:

```powershell
D:\working\taichi\env\python.exe
```

Run at minimum:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_coupled_smoke.py `
  tests\integration\test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts.py

& 'D:\working\taichi\env\python.exe' validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_coupled_smoke.py

& 'D:\working\taichi\env\python.exe' -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts `
  -v

git diff --check
```

If practical after the focused checks:

```powershell
& 'D:\working\taichi\env\python.exe' -m compileall `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_coupled_smoke.py `
  tests\integration\test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts.py
```

If the 5-step runner takes longer than expected, keep the artifact honest and
record the command duration in the final report.

## Phase 7 - Commit, Push, and Remote Verification

Before commit:

```text
git status --short
git diff --stat
git diff --check
focused tests pass
artifact regenerated from current code
lightweight sensitive-string scan has no hits
```

Commit with a conventional message, for example:

```text
validation: extend ANSYS selected formulation coupled smoke to 5 steps
```

Push to:

```text
origin solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25
```

After push:

```text
git ls-remote origin solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25
```

Verify the GitHub Actions run for the pushed commit. The final report must
include:

```text
commit SHA
remote branch
Actions run id
Actions conclusion
local verification commands and outcomes
final artifact candidate_status
final 5-step row smoke_status
whether the 5-step smoke passed or remained fail-closed pending
```

## Done Criteria

This goal is complete only when:

```text
the detailed goal file is committed
the active goal references this file
the artifact runner has a preserved 1-step row and a true 5-step row
the 5-step row is artifact-backed pass or exact fail-closed pending
step-history reducers identify per-step health and first failure
artifact tests lock the two-row contract and source provenance
artifacts are regenerated and checksummed
focused local verification passes
commit is pushed
remote branch points at the new commit
GitHub Actions for the pushed commit is checked
no 50-step or Fluent parity claim is introduced
```
