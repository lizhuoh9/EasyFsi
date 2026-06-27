# ANSYS Vertical-Flap Fixed-Solid Selected Formulation Goal

Date: 2026-06-27

Source checkpoint: remote branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
at commit `98d4907f20bfa38d718b22903320c4fffaba20c7`.

Short goal reference for Codex `/goal`:

```text
Implement the detailed ANSYS vertical-flap fixed-solid selected-formulation
validation contract in
docs/refactoring/ANSYS_VERTICAL_FLAP_FIXED_SOLID_SELECTED_FORMULATION_GOAL_2026-06-27.md.
First harden reference-selection source provenance. Then add fixed-solid
regenerated selected-formulation evidence, artifacts, tests, workflow cheap
checks, local verification, commit, and push. Do not claim coupled FSI or
Fluent parity.
```

## Current Checkpoint

The previous stage completed shared-snapshot reference formulation selection.
The selected candidate is:

```text
reference_formulation_candidate =
  anchored_dual_face_pressure_pair_with_per_face_one_sided
pressure_pair_policy_candidate = baseline_anchored_cell_pair
one_sided_pressure_policy_candidate = per_face_mirrored
candidate_status = reference_formulation_candidate_selected
```

The shared-snapshot field artifact remains:

```text
flow_snapshot_sha256 =
  3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968
```

The reference-selection artifact is a completed sampling-only decision. It is
not coupled-FSI evidence and it is not Fluent-parity evidence. The current
active blockers are expected to include:

```text
sampling_only_no_coupled_fsi
no_fluent_parity_claim
fixed_solid_regenerated_validation_pending
coupled_fsi_validation_pending
```

The blockers `dual_face_one_sided_unsupported` and
`reference_selection_deferred` are historical retired blockers after the
selection stage.

## Objective

Move the selected formulation from archived shared-snapshot selection into
fixed-solid regenerated evidence.

This goal has two required outputs:

1. Harden the existing reference-selection artifact test so the selection
   source provenance cannot silently drift.
2. Add a fixed-solid selected-formulation diagnostic artifact layer that
   proves the selected formulation remains valid on a regenerated or explicitly
   confirmed fixed-solid snapshot, with anchor provenance tied to that snapshot
   and marker geometry.

The strongest allowed final claim is:

```text
fixed-solid selected formulation validated
```

The final artifact may retire:

```text
fixed_solid_regenerated_validation_pending
```

Only if the regenerated fixed-solid gates pass.

## Non-Goals

Do not implement or claim any of the following in this goal:

- coupled FSI
- 5-step, 10-step, 50-step, or longer coupled smoke runs
- Fluent parity
- changes to fluid solver physics
- changes to solid solver physics
- changes to force aggregation semantics
- changes to marker geometry generation semantics
- reusing an old anchor map when marker geometry or flow snapshot provenance
  changes
- hiding missing provenance by hardcoding hashes or status strings
- replacing the existing reference-selection artifact instead of building the
  next evidence layer from it
- broad expensive GPU execution inside GitHub Actions

If a runner remains diagnostic-only or artifact-aggregation-only, the summary
must say so explicitly.

## Phase 0 - Harden Reference-Selection Source Provenance

Update:

```text
tests/integration/test_ansys_vertical_flap_traction_reference_formulation_selection_artifacts.py
```

Required hardening:

- Assert `pressure_pair_preselection_source` exists and is the exact committed
  source artifact used by the selected pressure-pair component.
- Assert `per_face_one_sided_source` exists and is the exact committed source
  artifact used by the selected one-sided component.
- Assert every row's `source_artifact_sha256` matches the digest of its
  referenced source artifact.
- Assert every row has `source_history_present == true`.
- Assert every row has a non-empty `source_flow_phase`.
- Assert every `source_marker_diagnostics_json` exists.
- Assert every marker wrapper's `source_marker_diagnostics_sha256` equals the
  actual source marker diagnostics file digest.
- Assert `historical_blockers_retired` exactly contains:

  ```text
  dual_face_one_sided_unsupported
  reference_selection_deferred
  ```

- Assert active blockers exactly equal:

  ```text
  sampling_only_no_coupled_fsi
  no_fluent_parity_claim
  fixed_solid_regenerated_validation_pending
  coupled_fsi_validation_pending
  ```

This phase must not regenerate artifacts unless the existing artifact is
missing fields needed to make provenance auditable.

## Phase 1 - Fixed-Solid Selected-Formulation Runner

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_fixed_solid_selected_formulation_matrix.py
```

Output directory:

```text
validation_runs/ansys_vertical_flap_fsi/traction_fixed_solid_selected_formulation_diagnostics/
```

The runner must read:

```text
validation_runs/ansys_vertical_flap_fsi/traction_reference_formulation_selection_diagnostics/traction_reference_formulation_selection_matrix.json
validation_runs/ansys_vertical_flap_fsi/traction_reference_formulation_selection_diagnostics/traction_reference_formulation_selection_history.json
validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/snapshot_manifest.json
```

It may also read the component artifacts used by the selection artifact when
needed for marker wrapper provenance.

The runner must produce:

- `traction_fixed_solid_selected_formulation_matrix.json`
- `traction_fixed_solid_selected_formulation_matrix.csv`
- `traction_fixed_solid_selected_formulation_history.json`
- `traction_fixed_solid_selected_formulation_summary.md`
- per-row marker diagnostic wrappers
- `CHECKSUMS.sha256`

The runner must either generate a fixed-solid regenerated snapshot or explicitly
record that the current fixed-solid snapshot is confirmed and reused. In either
case, it must record:

```text
fixed_solid_snapshot_policy
new_or_confirmed_flow_snapshot_sha256
marker_geometry_sha256
anchor_map_sha256
anchor_source_marker_geometry_sha256
anchor_source_flow_snapshot_sha256
```

Core rule:

```text
If regenerated snapshot SHA or marker geometry SHA changes, the anchor map must
be derived again for that snapshot/geometry pair.
```

## Phase 2 - Minimum Matrix

The minimum matrix rows are:

```text
fixed_solid_selected_baseline_probe0p51
fixed_solid_selected_anchored_probe0p00
fixed_solid_selected_anchored_probe0p25
fixed_solid_selected_anchored_probe0p51
fixed_solid_selected_anchored_probe0p625
fixed_solid_selected_anchored_probe1p00
fixed_solid_selected_per_face_one_sided_probe0p51
fixed_solid_selected_per_face_one_sided_probe0p625
fixed_solid_selected_per_face_one_sided_probe1p00
```

Every row must preserve:

```text
reference_formulation_candidate =
  anchored_dual_face_pressure_pair_with_per_face_one_sided
pressure_pair_policy_candidate = baseline_anchored_cell_pair
one_sided_pressure_policy_candidate = per_face_mirrored
```

Rows derived from one-sided selected formulation evidence must preserve:

```text
traction_one_sided_pressure_policy = per_face_mirrored
one_sided_marker_count = 24
one_sided_primary_marker_count = 12
one_sided_secondary_marker_count = 12
one_sided_anchor_selected_marker_count = 24
one_sided_anchor_fallback_marker_count = 0
```

## Phase 3 - Fixed-Solid Candidate Gates

The top-level artifact may report:

```text
candidate_status = fixed_solid_selected_formulation_validated
```

Only when all required gates pass:

- same regenerated or confirmed fixed-solid snapshot SHA across completed rows
- anchor map source flow snapshot SHA matches the fixed-solid snapshot SHA
- anchor map source marker geometry SHA matches selected rows
- anchor selected all markers
- anchor fallback marker count is zero
- one-sided markers complete on one-sided rows
- pressure sampling is complete
- invalid marker count is zero
- traction decomposition residual is `<= 1e-8`
- force-ratio span is `<= 0.10`
- absolute baseline bias is `<= 0.01`
- reference formulation candidate name matches exactly
- pressure-pair and one-sided policy candidates match exactly
- the summary contains no coupled-FSI claim
- the summary contains no Fluent-parity claim

If any gate fails, the artifact must fail closed with a diagnostic status and
must keep `fixed_solid_regenerated_validation_pending` active.

## Phase 4 - Artifact Test

Add:

```text
tests/integration/test_ansys_vertical_flap_traction_fixed_solid_selected_formulation_artifacts.py
```

The test must assert:

- matrix, history, summary, checksums, and marker wrapper artifacts exist
- the selected formulation name is exact
- the pressure-pair policy is exact
- the one-sided pressure policy is exact
- the fixed-solid snapshot provenance is present
- anchor provenance uses the fixed-solid snapshot and marker geometry
- completed row count is at least the minimum matrix count
- unsupported row count is zero
- force-ratio span is `<= 0.10`
- absolute baseline bias is `<= 0.01`
- anchor selected all markers
- fallback marker count is zero
- one-sided rows have complete one-sided marker counts
- no coupled-FSI claim appears in the summary
- no Fluent-parity claim appears in the summary
- `fixed_solid_regenerated_validation_pending` is retired only when the gate
  passed
- remaining active blockers are exactly:

  ```text
  coupled_fsi_validation_pending
  no_fluent_parity_claim
  ```

## Phase 5 - Workflow Cheap Checks

Update:

```text
.github/workflows/ansys-vertical-flap-validation.yml
```

Required additions:

- include the new runner in the `py_compile` list
- add an independent unittest step for the new artifact test

The workflow must not execute a GPU-heavy fixed-solid or coupled runner. CI
should validate committed artifacts and source-level contracts only.

## Phase 6 - Local Verification

Use the trusted local interpreter when available:

```text
D:\working\taichi\env\python.exe
```

Minimum verification:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  tests\integration\test_ansys_vertical_flap_traction_reference_formulation_selection_artifacts.py `
  tests\integration\test_ansys_vertical_flap_traction_fixed_solid_selected_formulation_artifacts.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_fixed_solid_selected_formulation_matrix.py

& 'D:\working\taichi\env\python.exe' -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_reference_formulation_selection_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_fixed_solid_selected_formulation_artifacts `
  -v

git diff --check
```

If artifact checksum or checkout newline behavior changes, verify the committed
artifact hashes against the same Windows checkout behavior used by CI.

## Phase 7 - Completion Criteria

This goal is complete only when:

- Phase 0 provenance hardening is implemented and passing.
- The fixed-solid selected-formulation runner exists.
- The fixed-solid selected-formulation artifacts are committed.
- The fixed-solid selected-formulation test passes locally.
- The workflow contains cheap compile and artifact-test coverage.
- The summary keeps coupled FSI and Fluent parity blocked.
- The branch is committed and pushed.
- The pushed GitHub Actions run is green, or any remaining red status is
  explicitly diagnosed as unrelated and reported with evidence.

