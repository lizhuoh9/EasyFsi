# ANSYS Vertical-Flap Reference Formulation Selection Goal

Date: 2026-06-27

Source checkpoint: remote commit
`abf58971fe664df80d5fc8a60ae4326ec7355c8c` on branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.

Short goal reference for Codex `/goal`:

```text
Implement the detailed ANSYS vertical-flap shared-snapshot reference
formulation selection contract in
docs/refactoring/ANSYS_VERTICAL_FLAP_REFERENCE_FORMULATION_SELECTION_GOAL_2026-06-27.md.
Finish the per-face one-sided positive hardening, add the reference formulation
selection runner, artifacts, tests, workflow checks, verification, commit, and
push. Do not claim coupled FSI or Fluent parity.
```

## Source Evidence

The previous checkpoint completed the per-face one-sided pressure support stage
for the ANSYS vertical-flap shared-snapshot traction path:

- `candidate_status = per_face_one_sided_pressure_completed`
- `pressure_pair_policy_candidate = baseline_anchored_cell_pair`
- `one_sided_pressure_policy_candidate = per_face_mirrored`
- `reference_formulation_candidate = None`
- `historical_blockers_retired` includes `dual_face_one_sided_unsupported`
- `completed_formulation_count = 4`
- `unsupported_formulation_count = 0`
- shared snapshot SHA:
  `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`
- per-face rows select 24 one-sided markers, with 12 primary markers and
  12 secondary markers
- all per-face one-sided markers currently select the `outside` side
- all 24 one-sided anchors are selected and no one-sided anchor fallback is used

This goal starts from those completed component candidates. It must not
reinterpret the per-face artifact as coupled-FSI or Fluent-parity evidence.

## Objective

Select a complete shared-snapshot traction reference formulation candidate by
combining the existing component candidates:

```text
pressure_pair_policy_candidate = baseline_anchored_cell_pair
one_sided_pressure_policy_candidate = per_face_mirrored
```

The selected formulation candidate must be:

```text
reference_formulation_candidate =
  anchored_dual_face_pressure_pair_with_per_face_one_sided
```

The selection must be artifact-backed on the same shared snapshot evidence
chain. It must remain a marker-traction sampling-only validation step.

## Non-Goals

Do not implement or claim any of the following in this goal:

- coupled FSI execution
- 50-step coupled FSI
- fixed-solid regenerated preflow evidence
- Fluent parity
- changes to fluid solver physics
- changes to solid solver physics
- changes to marker geometry generation
- changes to force aggregation semantics
- overwriting or redefining previous pressure-pair reference-preselection
  artifacts
- overwriting or redefining previous per-face one-sided pressure artifacts
- hardcoded pressure, force, displacement, flow, or marker results

The strongest allowed claim is:

```text
shared-snapshot traction reference formulation candidate selected
```

## Phase 0 - Harden Per-Face One-Sided Positive Gates

Before adding the selection runner, harden the existing per-face one-sided
artifact tests so the completed input component cannot regress silently.

Update:

```text
tests/integration/test_ansys_vertical_flap_traction_per_face_one_sided_artifacts.py
```

Required top-level artifact assertions:

- `candidate_status == per_face_one_sided_pressure_completed`
- `pressure_pair_policy_candidate == baseline_anchored_cell_pair`
- `one_sided_pressure_policy_candidate == per_face_mirrored`
- `reference_formulation_candidate is None`
- `unsupported_formulation_count == 0`
- `dual_face_one_sided_unsupported` is present in
  `historical_blockers_retired`
- active blockers no longer contain `dual_face_one_sided_unsupported`

Required row-level assertions for every per-face one-sided row:

- `one_sided_marker_count == 24`
- `one_sided_primary_marker_count == 12`
- `one_sided_secondary_marker_count == 12`
- `one_sided_anchor_selected_marker_count == 24`
- `one_sided_anchor_fallback_marker_count == 0`
- `one_sided_side_selection_counts == {"inside": 0, "outside": 24}`
- `primary_fluid_side_normal_sign == 1.0`
- `secondary_fluid_side_normal_sign == 1.0`
- `traction_pressure_pair_policy == baseline_anchored_cell_pair`

Required marker-diagnostic assertions:

- primary-region markers report the expected primary one-sided region id
- secondary-region markers report the expected secondary one-sided region id
- all one-sided markers report
  `one_sided_pressure_pair_policy == baseline_anchored_cell_pair`
- all one-sided markers report `one_sided_anchor_selected == True`
- all one-sided markers report `one_sided_anchor_fallback_used == False`

This phase should not modify solver physics.

## Phase 1 - Reference Formulation Selection Runner

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_reference_formulation_selection_matrix.py
```

Output directory:

```text
validation_runs/ansys_vertical_flap_fsi/traction_reference_formulation_selection_diagnostics/
```

Required inputs:

```text
validation_runs/ansys_vertical_flap_fsi/traction_pressure_pair_reference_preselection_diagnostics/traction_pressure_pair_reference_preselection_matrix.json
validation_runs/ansys_vertical_flap_fsi/traction_per_face_one_sided_diagnostics/traction_per_face_one_sided_matrix.json
validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/
```

The runner must:

1. Load the existing shared snapshot provenance.
2. Load the pressure-pair reference-preselection artifact.
3. Load the per-face one-sided pressure artifact.
4. Verify all selected component candidates use the same shared snapshot SHA.
5. Reuse the existing baseline-anchored pressure-pair component candidate.
6. Reuse the existing per-face mirrored one-sided component candidate.
7. Build a small independently reviewable selection matrix.
8. Write matrix, history, summary, marker diagnostics, and checksums.
9. Avoid advancing fluid, solid, or coupled FSI state.

The runner may reuse committed row evidence from the component artifacts when
that evidence is immutable and checksum-backed. Any resampling must use the same
shared snapshot and must remain diagnostic-only.

## Phase 2 - Selection Matrix

The minimum expected selection scenarios are:

```text
reference_baseline_anchored_two_sided_probe0p51
reference_anchored_two_sided_probe0p00
reference_anchored_two_sided_probe0p25
reference_anchored_two_sided_probe0p375
reference_anchored_two_sided_probe0p625
reference_anchored_two_sided_probe1p00
reference_per_face_one_sided_probe0p51
reference_per_face_one_sided_probe0p625
reference_per_face_one_sided_probe1p00
```

Rows derived from pressure-pair preselection must preserve the pressure-pair
policy:

```text
traction_pressure_pair_policy = baseline_anchored_cell_pair
```

Rows derived from per-face one-sided pressure must preserve:

```text
traction_one_sided_pressure_policy = per_face_mirrored
traction_pressure_pair_policy = baseline_anchored_cell_pair
```

The matrix must record the source artifact and source scenario for each row.

## Phase 3 - Candidate Schema

The top-level selection artifact must report:

```text
candidate_status = reference_formulation_candidate_selected
reference_formulation_candidate =
  anchored_dual_face_pressure_pair_with_per_face_one_sided
pressure_pair_policy_candidate = baseline_anchored_cell_pair
one_sided_pressure_policy_candidate = per_face_mirrored
```

It must also record:

- `marker_layout = dual_physical_faces`
- `pressure_sampling_mode = one_sided_surface_pressure_supported`
- `shared_snapshot_sha256`
- `pressure_pair_preselection_source`
- `per_face_one_sided_source`
- completed row count
- unsupported row count
- max traction decomposition residual
- active candidate blockers
- retired blockers
- scope text saying this is shared-snapshot selection only

## Phase 4 - Selection Gates

The selected candidate is valid only if all gates pass:

- pressure-pair preselection candidate exists
- `pressure_pair_policy_candidate == baseline_anchored_cell_pair`
- absolute baseline bias is `<= 0.01`
- anchored force-ratio span is `<= 0.10`
- per-face one-sided pressure stage is complete
- `one_sided_pressure_policy_candidate == per_face_mirrored`
- active blockers do not contain `dual_face_one_sided_unsupported`
- all required selection rows are completed
- all completed rows use the same shared snapshot SHA
- all required markers have anchors selected
- all anchor fallback counts are zero
- all pressure completeness flags pass
- all invalid marker counts are zero
- max traction decomposition residual is `<= 1.0e-8`
- no coupled FSI was advanced
- no Fluent parity is claimed

Allowed active blockers after selection:

```text
sampling_only_no_coupled_fsi
no_fluent_parity_claim
fixed_solid_regenerated_validation_pending
coupled_fsi_validation_pending
```

Forbidden active blockers after selection:

```text
dual_face_one_sided_unsupported
reference_selection_deferred
```

## Phase 5 - Artifact Tests

Add:

```text
tests/integration/test_ansys_vertical_flap_traction_reference_formulation_selection_artifacts.py
```

The test must verify:

- matrix JSON exists and is a JSON object
- history JSON exists and is a JSON object or list of row histories
- summary markdown exists and states the shared-snapshot-only scope
- checksums exist and match generated files
- marker diagnostics exist for completed rows
- `source_script` is repo-relative
- all rows use the same shared snapshot SHA
- the expected reference formulation candidate is non-`None`
- pressure-pair and one-sided policy candidates match the expected names
- `dual_face_one_sided_unsupported` is retired, not active
- `reference_selection_deferred` is not active
- all required rows are completed
- unsupported row count is zero
- anchor selected counts cover every marker
- fallback counts are zero
- pressure completeness flags pass
- invalid marker counts are zero
- residuals are within tolerance
- no coupled-FSI claim is present
- no Fluent-parity claim is present

## Phase 6 - Workflow Integration

Update:

```text
.github/workflows/ansys-vertical-flap-validation.yml
```

Required cheap checks:

- add the new runner to the `py_compile` list
- add the new artifact test to the archive/artifact consistency block

Do not add a CI step that runs GPU, long coupled FSI, or regenerated fixed-solid
simulations.

## Phase 7 - Documentation Decision

If a nearby docs status section already summarizes ANSYS vertical-flap
validation status, update it with this exact scope:

```text
ANSYS vertical-flap traction reference formulation candidate has been selected
on a shared snapshot only; fixed-solid and coupled validations remain pending.
```

If no local status section exists, do not create unrelated top-level docs.
The selection goal and artifact summary are sufficient.

## Deferred Follow-Up Goals

After this goal is complete, the next goals are:

1. fixed-solid regenerated validation for the selected formulation
2. short coupled smoke, 5 to 10 steps
3. 50-step coupled FSI
4. Fluent parity comparison

Those follow-ups are explicitly out of scope for this goal.

## Verification Plan

Use the repository's reliable Taichi Python environment when available:

```powershell
& 'D:\working\taichi\env\python.exe' -m py_compile `
  tests\integration\test_ansys_vertical_flap_traction_per_face_one_sided_artifacts.py `
  tests\integration\test_ansys_vertical_flap_traction_reference_formulation_selection_artifacts.py `
  validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_reference_formulation_selection_matrix.py

& 'D:\working\taichi\env\python.exe' -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_per_face_one_sided_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_reference_formulation_selection_artifacts `
  -v

& 'D:\working\taichi\env\python.exe' -m unittest `
  tests.integration.test_ansys_vertical_flap_traction_pressure_pair_reference_preselection_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_per_face_one_sided_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_reference_formulation_selection_artifacts `
  -v

git diff --check
```

If the workflow artifact block is touched, also run the full local equivalent
of the ANSYS archive/artifact consistency group before pushing.

## Completion Criteria

This goal is complete when:

- the detailed goal file is committed
- per-face one-sided positive gates are hardened
- the reference formulation selection runner is committed
- the selection artifacts are committed
- the selection artifact test is committed and passing locally
- workflow cheap checks include the new runner and artifact test
- the worktree is clean after verification
- the final commit is pushed to the configured remote branch
