# ANSYS vertical-flap selected formulation coupled step50 goal

Date: 2026-06-27

Source review anchor: remote `lizhuoh9/EasyFsi` commit `c2d7230f25cca45a324404fae8db0f653011ae4a`, GitHub Actions run `28293055018`.

## Current accepted baseline

The current selected-formulation coupled 5-step smoke is accepted as a real passing gate, not as a pending or fail-closed placeholder. Its artifact boundary is:

- `candidate_status` is exactly `selected_formulation_coupled_smoke_passed`.
- The only historical blocker retired by this stage is `coupled_fsi_validation_pending`.
- Active blockers remain exactly `long_coupled_validation_pending` and `no_fluent_parity_claim`.
- The selected formulation remains `anchored_dual_face_pressure_pair_with_per_face_one_sided`.
- The pressure-pair policy remains `baseline_anchored_cell_pair`.
- The one-sided pressure policy remains `per_face_mirrored`.
- The selected anchor marker source remains `fixed_solid_selected_per_face_one_sided_probe0p51_markers.json`.
- No artifact may claim 50-step validation or Fluent parity until those gates are produced by a dedicated run.

## Objective

Advance the ANSYS vertical-flap selected formulation from the accepted 5-step coupled smoke into staged longer coupled validation with exact artifact evidence for requested step counts 10, 30, and 50.

The implementation must make the validation boundary explicit:

- The existing 5-step smoke artifact test must now require the exact passing state that has already been achieved.
- A new runner must produce a staged long-horizon matrix under `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_step50_diagnostics/`.
- The new matrix must include exactly three requested rows:
  - `selected_formulation_coupled_step10`
  - `selected_formulation_coupled_step30`
  - `selected_formulation_coupled_step50`
- Each row must preserve the exact selected formulation and source checksums from the accepted 5-step smoke chain.
- Candidate status must advance only as far as the completed staged evidence justifies.
- The runner must remain outside GitHub Actions heavy execution; Actions should only compile the runner and run cheap artifact contract tests.

## Non-goals

This goal does not include:

- Fluent parity validation.
- ANSYS Fluent numeric agreement claims.
- Material tuning.
- Geometry tuning.
- Switching away from `anchored_dual_face_pressure_pair_with_per_face_one_sided`.
- Relaxing marker, anchor, one-sided, finite-field, residual, velocity, or pressure gates.
- Treating a 10-step pass as a 30-step or 50-step pass.
- Treating a 50-step selected-formulation pass as Fluent parity.
- Hiding solver behavior inside case-specific scripts.

## Phase 0: harden the existing 5-step positive gate

Update `tests/integration/test_ansys_vertical_flap_traction_selected_formulation_coupled_smoke_artifacts.py` so the accepted 5-step smoke is tested as an exact pass:

- `candidate_status` must equal `selected_formulation_coupled_smoke_passed`.
- `candidate_blockers` must equal exactly:
  - `long_coupled_validation_pending`
  - `no_fluent_parity_claim`
- `historical_blockers_retired` must equal exactly:
  - `coupled_fsi_validation_pending`
- The preflight row must remain a hardened one-step anchor-injected checkpoint:
  - `smoke_status` is `blocked_requested_5step_not_completed`
  - `completed_step_count` is 1
  - requested step count is 5
  - invalid marker count is 0
  - active anchor marker count is at least 24
  - selected anchor marker count is at least 24
  - anchor fallback count is 0
  - one-sided marker count is at least 24
  - one-sided fallback count is 0
- The 5-step smoke row must be exact:
  - `run_status` is `completed`
  - `smoke_status` is `passed`
  - `completed_step_count` is 5
  - `requested_step_count` is 5
  - `first_failed_step`, `first_failed_gate`, and `first_failed_gate_value` are empty
  - finite flags are true
  - invalid marker count is 0
  - active/selected anchor marker count is at least 24
  - anchor fallback count is 0
  - one-sided marker count is at least 24
  - one-sided fallback count is 0
  - force action-reaction residual is at most `1.0e-8`

## Phase 1: add narrow long-validation opt-in

The core runner currently treats the selected-formulation coupled exception as a 5-step smoke pathway guarded to small step counts. To support step30 and step50 without making arbitrary non-default formulations run in coupled mode:

- Add an explicit config flag for selected-formulation long coupled validation.
- Keep the existing smoke flag required.
- Allow the selected formulation coupled exception up to 50 steps only when the long-validation flag is true.
- Preserve the existing failure for a 50-step selected coupled config when the long-validation flag is not present.
- Add a source-level unit test proving both boundaries:
  - 50-step selected coupled without long-validation opt-in still fails as fixed-solid-only.
  - 50-step selected coupled with long-validation opt-in validates.

## Phase 2: add the step10/30/50 runner

Add:

`validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_selected_formulation_coupled_step50.py`

The runner writes:

- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_step50_diagnostics/traction_selected_formulation_coupled_step50_matrix.json`
- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_step50_diagnostics/traction_selected_formulation_coupled_step50_matrix.csv`
- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_step50_diagnostics/traction_selected_formulation_coupled_step50_history.json`
- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_step50_diagnostics/traction_selected_formulation_coupled_step50_summary.md`
- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_step50_diagnostics/scenario_diagnostics/*.json`
- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_step50_diagnostics/CHECKSUMS.sha256`

The runner must include exactly these scenario rows:

- `selected_formulation_coupled_step10`
- `selected_formulation_coupled_step30`
- `selected_formulation_coupled_step50`

The runner should execute staged validation in that order. If an earlier row fails, later rows may be recorded as blocked/not-run, but the matrix must still include all three rows with exact first-failed step/gate details and active blockers that describe the unearned later stage.

## Phase 3: preserve selected-formulation metadata

Every row and the matrix payload must carry these exact metadata values:

- `reference_formulation_candidate`: `anchored_dual_face_pressure_pair_with_per_face_one_sided`
- `pressure_pair_policy_candidate`: `baseline_anchored_cell_pair`
- `one_sided_pressure_policy_candidate`: `per_face_mirrored`
- `selected_anchor_markers_source`: `validation_runs/ansys_vertical_flap_fsi/traction_fixed_solid_selected_formulation_diagnostics/marker_diagnostics/fixed_solid_selected_per_face_one_sided_probe0p51_markers.json`

The payload must retain checksum evidence for:

- Reference formulation selection matrix.
- Fixed-solid selected formulation matrix.
- Selected anchor marker source.
- Shared snapshot manifest.
- Pressure-pair anchor map.
- Pressure-pair anchor source flow snapshot.
- Pressure-pair anchor source marker geometry.
- Pressure-pair anchor current marker geometry.
- Source 5-step coupled smoke matrix.

The step50 runner must explicitly record the source 5-step smoke matrix path and SHA256 so the long-horizon validation is tied to the accepted 5-step gate.

## Phase 4: per-row gates

Each requested step row is accepted only when all row gates pass:

- `completed_step_count == requested_step_count`
- fluid fields are finite
- pressure fields are finite
- solid position/displacement fields are finite
- `invalid_marker_count_max == 0`
- `pressure_pair_anchor_active_marker_count_min >= 24`
- `anchor_selected_marker_count_min >= 24`
- `anchor_fallback_marker_count_max == 0`
- `one_sided_marker_count_min >= 24`
- `one_sided_anchor_fallback_marker_count_max == 0`
- `force_action_reaction_residual_max_n <= 1.0e-8`
- `max_velocity_mps <= 1.0e6`
- `max_pressure_pa <= 1.0e9`
- `max_displacement_m` is finite

The long-horizon matrix must also record report-only trajectory diagnostics:

- `max_velocity_growth_ratio`
- `max_pressure_growth_ratio`
- `max_displacement_growth_ratio`
- `force_sign_flip_count`
- `invalid_marker_count_by_step`
- `one_sided_marker_count_by_step`
- `anchor_selected_marker_count_by_step`
- `anchor_fallback_marker_count_by_step`
- `one_sided_anchor_fallback_marker_count_by_step`
- `force_action_reaction_residual_by_step`
- `max_velocity_by_step`
- `max_pressure_abs_by_step`
- `max_displacement_by_step`

Growth-ratio fields are report-only in this goal. They must not be converted into physical parity claims.

## Phase 5: candidate status and blockers

The matrix-level candidate status must be exact:

- If step10 passes but step30 and step50 are not passed:
  - `candidate_status`: `selected_formulation_coupled_step10_passed`
  - active blockers:
    - `step30_coupled_validation_pending`
    - `step50_coupled_validation_pending`
    - `no_fluent_parity_claim`
- If step10 and step30 pass but step50 is not passed:
  - `candidate_status`: `selected_formulation_coupled_step30_passed`
  - active blockers:
    - `step50_coupled_validation_pending`
    - `no_fluent_parity_claim`
- If step10, step30, and step50 all pass:
  - `candidate_status`: `selected_formulation_coupled_step50_passed`
  - active blockers:
    - `no_fluent_parity_claim`
  - retired blockers must include:
    - `long_coupled_validation_pending`

If step10 does not pass, the matrix must remain fail-closed with a status that does not imply any long-horizon pass, and the first failed row/gate must be explicit.

## Phase 6: artifact tests

Add:

`tests/integration/test_ansys_vertical_flap_traction_selected_formulation_coupled_step50_artifacts.py`

The test must verify:

- Matrix JSON, CSV, history JSON, summary markdown, scenario diagnostics, and checksums exist.
- `source_script` is repo-relative and points to the new runner.
- Source 5-step smoke artifact path exists and its SHA256 matches.
- The exact selected candidate metadata is preserved.
- Rows are exactly step10, step30, and step50.
- Per-row array lengths equal `completed_step_count`.
- Per-row acceptance gates are consistent with `run_status` and `smoke_status`.
- Candidate status and active blockers match the staged status rules exactly.
- Summary never claims Fluent parity.
- Summary never contains `Fluent parity validated`.
- Checksums cover all top-level artifacts and scenario diagnostics.

If step50 is not passed, the test must lock an exact fail-closed status and ensure first-failed row/gate are non-empty. It must not keep a broad pass-or-fail-closed branch after the status is known.

## Phase 7: workflow checks

Update `.github/workflows/ansys-vertical-flap-validation.yml` only for cheap checks:

- Add `python -m py_compile validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_coupled_step50.py`.
- Add `python -m unittest tests.integration.test_ansys_vertical_flap_traction_selected_formulation_coupled_step50_artifacts -v`.

Do not add the heavy step10/30/50 runner execution to GitHub Actions.

## Phase 8: verification and push

Before pushing:

- Run targeted source-level tests for the selected coupled guard boundary.
- Compile the new runner and touched source modules.
- Run the hardened 5-step artifact test.
- Run the new step50 artifact test after producing artifacts.
- Review `git diff`.
- Commit with conventional commit messages.
- Push to the configured remote branch only after the implementation and verification pass.

## Deferred follow-up

Only after `selected_formulation_coupled_step50_passed` is established with artifacts should a later goal enter Fluent parity or ANSYS numeric comparison. That future work must have its own runner, artifact contract, and claim boundary.
