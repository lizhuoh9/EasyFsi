# ANSYS Vertical-Flap Selected Formulation Coupled Smoke Goal

Date: 2026-06-27

Source checkpoint: remote branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
at commit `0ef4dc7816d1e4d6e181e7717b4569539378c5cf`.

Short goal reference for Codex `/goal`:

```text
Implement the detailed ANSYS vertical-flap selected-formulation coupled smoke
contract in
docs/refactoring/ANSYS_VERTICAL_FLAP_SELECTED_FORMULATION_COUPLED_SMOKE_GOAL_2026-06-27.md.
First harden fixed-solid selected-formulation semantics. Then add the selected
formulation coupled-smoke runner, committed smoke artifacts, integration tests,
workflow cheap checks, local verification, commit, push, and remote Actions
verification. Do not run or claim 50-step validation or Fluent parity.
```

## Current Checkpoint

The previous checkpoint completed fixed-solid selected-formulation validation.
The selected formulation is:

```text
reference_formulation_candidate =
  anchored_dual_face_pressure_pair_with_per_face_one_sided
pressure_pair_policy_candidate = baseline_anchored_cell_pair
one_sided_pressure_policy_candidate = per_face_mirrored
candidate_status = fixed_solid_selected_formulation_validated
```

The fixed-solid selected-formulation artifact records:

```text
fixed_solid_snapshot_policy = confirmed_shared_fixed_solid_snapshot_reused
fixed_solid_flow_candidate = fixed_source_0p75_constant_step30
fixed_solid_load_candidate = fixed_load_0p80_ramp2_step60
```

This is committed fixed-solid evidence tied to the selected formulation. It is
not a coupled dynamic validation and it is not Fluent parity.

The remaining active blockers are:

```text
coupled_fsi_validation_pending
no_fluent_parity_claim
```

## Objective

Move from selected formulation fixed-solid evidence to a bounded short coupled
smoke contract.

This goal has three required outputs:

1. Harden the fixed-solid selected-formulation artifact test so the artifact
   cannot be misread as a regenerated heavy run, coupled validation, or Fluent
   parity.
2. Add a selected-formulation coupled-smoke runner and committed smoke artifact
   layer that records the selected formulation metadata, source artifact
   digests, requested step count, completion status, finite-state checks,
   marker/anchor/one-sided checks, and residual gates.
3. Add cheap CI coverage that validates committed smoke artifacts and compiles
   the runner without running a heavy coupled simulation in GitHub Actions.

The strongest allowed final claim is:

```text
selected formulation coupled smoke passed
```

Only if the committed smoke artifact proves the requested short smoke completed
and all smoke gates passed.

If the runner cannot produce actual coupled-step evidence in this turn, it must
fail closed with:

```text
candidate_status = selected_formulation_coupled_smoke_pending
```

and must keep `coupled_fsi_validation_pending` active. It must not fake a pass.

## Non-Goals

Do not implement or claim:

- 30-step, 50-step, or longer coupled validation
- Fluent parity
- final displacement parity
- material parameter tuning
- geometry tuning
- source schedule tuning
- pressure or velocity hardcoding
- force, displacement, or marker-result hardcoding
- broad solver physics rewrites
- heavy coupled runner execution in GitHub Actions

Do not retire `no_fluent_parity_claim` in this goal.

## Phase 0 - Harden Fixed-Solid Selected-Formulation Semantics

Update:

```text
tests/integration/test_ansys_vertical_flap_traction_fixed_solid_selected_formulation_artifacts.py
```

Required hardening:

- Assert `fixed_solid_snapshot_policy` exactly equals:

  ```text
  confirmed_shared_fixed_solid_snapshot_reused
  ```

- Assert the summary contains:

  ```text
  does not claim coupled FSI
  does not claim Fluent parity
  ```

- Assert `payload["candidate_status"]` does not contain:

  ```text
  coupled_fsi_validated
  fluent_parity
  ```

- Assert active blockers exactly equal:

  ```text
  coupled_fsi_validation_pending
  no_fluent_parity_claim
  ```

- Assert the artifact does not use wording that implies the fixed-solid
  selected-formulation runner performed a fresh long coupled run.

This phase must not regenerate artifacts unless a wording or metadata field is
needed to make the existing artifact honest.

## Phase 1 - Selected Formulation Coupled-Smoke Runner

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_selected_formulation_coupled_smoke.py
```

Output directory:

```text
validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_smoke_diagnostics/
```

The runner must read:

```text
validation_runs/ansys_vertical_flap_fsi/traction_reference_formulation_selection_diagnostics/traction_reference_formulation_selection_matrix.json
validation_runs/ansys_vertical_flap_fsi/traction_fixed_solid_selected_formulation_diagnostics/traction_fixed_solid_selected_formulation_matrix.json
validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/snapshot_manifest.json
```

The runner must produce:

- `traction_selected_formulation_coupled_smoke_matrix.json`
- `traction_selected_formulation_coupled_smoke_matrix.csv`
- `traction_selected_formulation_coupled_smoke_history.json`
- `traction_selected_formulation_coupled_smoke_summary.md`
- per-scenario diagnostic JSON files when applicable
- `CHECKSUMS.sha256`

The runner must record selected-formulation metadata exactly:

```text
reference_formulation_candidate =
  anchored_dual_face_pressure_pair_with_per_face_one_sided
pressure_pair_policy_candidate = baseline_anchored_cell_pair
one_sided_pressure_policy_candidate = per_face_mirrored
```

It must also record source artifact digests:

```text
reference_selection_source
reference_selection_source_sha256
fixed_solid_selected_formulation_source
fixed_solid_selected_formulation_source_sha256
shared_snapshot_manifest
shared_snapshot_sha256
```

## Phase 2 - Coupled Smoke Scope

Minimum scenarios:

```text
selected_formulation_coupled_smoke_5step
```

Optional scenario if runtime cost is acceptable:

```text
selected_formulation_coupled_smoke_10step
```

Every scenario must record:

```text
requested_step_count
completed_step_count
dt_s
solid_substeps
reference_formulation_candidate
pressure_pair_policy_candidate
one_sided_pressure_policy_candidate
source reference-selection SHA
source fixed-solid-selected SHA
max_velocity_mps
max_pressure_pa
max_displacement_m
tip_displacement_norm_m
fluid_finite
pressure_finite
solid_position_finite
invalid_marker_count_max
pressure_complete_marker_count_min
anchor_selected_marker_count_min
anchor_fallback_marker_count_max
one_sided_marker_count_min
one_sided_anchor_fallback_marker_count_max
force_action_reaction_residual_max_n
```

If the runner is artifact-only or cannot execute coupled steps, the row must
say so explicitly with:

```text
run_status = blocked
smoke_status = not_run
candidate_status = selected_formulation_coupled_smoke_pending
```

and the test must not treat that as a pass.

## Phase 3 - Coupled Smoke Gate

The top-level artifact may report:

```text
candidate_status = selected_formulation_coupled_smoke_passed
```

Only when all gates pass:

- `completed_step_count == requested_step_count`
- no NaN/Inf in fluid velocity, pressure, or solid positions
- max displacement finite
- max velocity finite and below the artifact threshold
- max pressure finite and below the artifact threshold
- invalid marker count is zero or explicitly within a committed tolerance
- anchor fallback count is zero
- one-sided pressure is complete on all required markers
- force/action-reaction residual is within tolerance
- selected formulation metadata matches the fixed-solid selected artifact
- source artifact SHA values match actual committed files

After a smoke pass, active blockers must become:

```text
long_coupled_validation_pending
no_fluent_parity_claim
```

The old broad blocker:

```text
coupled_fsi_validation_pending
```

may be retired only when replaced by `long_coupled_validation_pending`.

## Phase 4 - Artifact Test

Add:

```text
tests/integration/test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts.py
```

The test must assert:

- matrix, history, summary, checksums, and scenario diagnostics exist
- `source_script` is repo-relative
- reference-selection source exists and SHA matches
- fixed-solid selected source exists and SHA matches
- selected formulation candidate and policy names match exactly
- smoke scenarios record requested and completed step counts
- candidate status is either pending with clear blockers or passed with all
  smoke gates true
- if passed, completed steps equal requested steps
- if passed, no NaN/Inf flags are present
- if passed, anchor fallback count is zero
- if passed, one-sided rows are complete
- if passed, force residual is within tolerance
- no Fluent parity claim appears anywhere in summary or top-level status
- long-coupled blocker remains after smoke pass
- checksums match committed artifact bytes

## Phase 5 - Workflow Cheap Checks

Update:

```text
.github/workflows/ansys-vertical-flap-validation.yml
```

Required additions:

- include `run_traction_selected_formulation_coupled_smoke.py` in the
  `py_compile` list
- add an independent unittest step for:

  ```text
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts
  ```

Do not run the coupled smoke runner in CI. CI validates committed artifacts and
source-level contracts only.

## Phase 6 - Local Verification

Use the trusted local interpreter when available:

```text
D:\working\taichi\env\python.exe
```

Minimum verification:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  tests\integration\test_ansys_vertical_flap_traction_fixed_solid_selected_formulation_artifacts.py `
  tests\integration\test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_coupled_smoke.py

& 'D:\working\taichi\env\python.exe' -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_fixed_solid_selected_formulation_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts `
  -v

git diff --check
```

Also run the workflow-equivalent artifact block if the new artifact test can
affect CI ordering.

## Phase 7 - Completion Criteria

This goal is complete only when:

- The detailed goal file is committed.
- Fixed-solid selected-formulation semantic hardening is implemented.
- The selected-formulation coupled-smoke runner exists.
- Coupled-smoke artifacts are committed.
- The coupled-smoke artifact test passes locally.
- Workflow compile and artifact-test coverage is updated.
- The artifact does not claim 50-step validation or Fluent parity.
- The branch is committed and pushed.
- The pushed GitHub Actions run is green, or any remaining red status is
  explicitly diagnosed as unrelated and reported with evidence.
