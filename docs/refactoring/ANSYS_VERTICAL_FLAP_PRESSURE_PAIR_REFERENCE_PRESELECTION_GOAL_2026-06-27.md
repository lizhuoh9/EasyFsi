# ANSYS Vertical-Flap Pressure-Pair Reference Preselection Goal

Date: 2026-06-27

Source request: advance from the positive anchor-map hardening checkpoint at
commit `ec0568bdb37ae31b1f7923b326275176f4881be3`.

## Objective

Promote `baseline_anchored_cell_pair` from an isolated positive
anchor-map diagnostic result into a pressure-pair policy component candidate for
the next formulation stage, while keeping full reference-formulation selection
explicitly deferred.

This is still a shared-snapshot, marker-traction-sampling-only validation step.
It must not run coupled FSI, claim Fluent parity, alter fluid/solid physics, or
set `reference_formulation_candidate` to anything other than `None`.

The required outcome is a new reference-preselection artifact that says:

- `pressure_pair_policy_candidate = baseline_anchored_cell_pair`
- `reference_formulation_candidate = None`
- `candidate_status = pressure_pair_policy_preselection_candidate_found`
- the completed anchored rows reuse the existing shared snapshot
  `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`
- the anchor-map policy stays stable under the probe-origin sweep
- the absolute force bias relative to the independent-ladder baseline is
  measured and bounded
- dual-face one-sided pressure remains an explicit unsupported blocker

## Current Evidence To Preserve

The previous anchor-map diagnostic already established:

- `candidate_status = pressure_pair_anchor_map_stable_candidate_found`
- `stable_pressure_pair_policy = baseline_anchored_cell_pair`
- anchored force-ratio relative span is `0.0`
- 8 anchored rows select 24/24 anchors
- anchor fallback marker count is `0`
- `reference_formulation_candidate` remains `None`

The current task must build on that evidence, not overwrite it.

## Scope

Expected new files:

- `docs/refactoring/ANSYS_VERTICAL_FLAP_PRESSURE_PAIR_REFERENCE_PRESELECTION_GOAL_2026-06-27.md`
- `validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_pressure_pair_reference_preselection_matrix.py`
- `validation_runs/ansys_vertical_flap_fsi/traction_pressure_pair_reference_preselection_diagnostics/`
- `tests/integration/test_ansys_vertical_flap_traction_pressure_pair_reference_preselection_artifacts.py`

Expected modified files:

- `.github/workflows/ansys-vertical-flap-validation.yml`

No solver-core file should change for this task unless a red test proves the
existing `baseline_anchored_cell_pair` public behavior is insufficient. The
expected route is to reuse existing core and runner APIs.

## Non-Goals

Do not implement or claim any of the following in this task:

- per-face one-sided pressure support
- a completed dual-face one-sided formulation row
- complete reference formulation selection
- any non-`None` `reference_formulation_candidate`
- coupled FSI execution
- fixed-solid regenerated preflow execution
- Fluent parity
- changes to force aggregation
- changes to marker geometry generation
- changes to fluid or solid solver physics
- changes to the existing anchor-map positive artifact contents

The unsupported dual-face one-sided row must stay unsupported and must remain a
blocker in the preselection artifact.

## Runner Contract

Add:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_pressure_pair_reference_preselection_matrix.py
```

The runner must:

1. Load the existing shared snapshot from
   `validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/`.
2. Reuse the snapshot without advancing fluid, solid, or coupled FSI state.
3. Build the independent-ladder baseline row at marker/probe offset `0.51`.
4. Derive the pressure-pair anchor map from that baseline row's
   `inside_probe_nearest_cell` and `outside_probe_nearest_cell` fields.
5. Reuse the same anchor map for all anchored probe-origin offsets.
6. Record a dual-face one-sided row as explicitly unsupported rather than
   pretending it completed.
7. Write matrix, history, summary, marker diagnostics, and checksums.

The minimum scenario set is:

```text
baseline_independent_ladder_probe0p51
anchored_pair_dual_faces_probe0p00
anchored_pair_dual_faces_probe0p25
anchored_pair_dual_faces_probe0p375
anchored_pair_dual_faces_probe0p51
anchored_pair_dual_faces_probe0p625
anchored_pair_dual_faces_probe0p75
anchored_pair_dual_faces_probe1p00
anchored_pair_dual_faces_probe1p50
dual_one_sided_offset0p51_pressure_only_unsupported_confirmed
```

Completed anchored rows must use:

```text
traction_marker_face_offset_cells = 0.51
traction_pressure_probe_origin_mode = physical_face_offset
traction_pressure_pair_policy = baseline_anchored_cell_pair
```

The unsupported one-sided row must include:

```text
run_status = unsupported
formulation_status = unsupported
unsupported_reason = dual-face one-sided pressure needs per-face one-sided region support
```

It must not have marker diagnostics or force values that imply completion.

## Preselection Gate

The new artifact must compute and report:

```text
candidate_status = pressure_pair_policy_preselection_candidate_found
pressure_pair_policy_candidate = baseline_anchored_cell_pair
reference_formulation_candidate = None
```

The completed-row gate requires:

- anchored force-ratio relative span `<= 0.10`
- anchored force-ratio relative span expected to be `0.0` for the committed
  shared snapshot
- all anchored rows have anchors selected for every marker
- all anchored rows have zero anchor fallback markers
- all completed rows have primary and secondary pressure complete
- all completed rows have zero primary and secondary invalid marker counts
- max traction-decomposition residual `<= 1.0e-8`
- every completed row uses the same shared snapshot SHA
- no coupled FSI was advanced
- no marker feedback was applied
- scope text says this is shared-snapshot sampling-only and does not claim
  Fluent parity

The new absolute-bias gate requires:

```text
absolute_baseline_bias = abs(anchor_ratio_at_probe0p51 - 1.0)
absolute_baseline_bias <= 0.01
```

The current artifact is expected to pass with an absolute baseline bias around
`0.003727`.

The blocker gate requires the unsupported row to stay present:

```text
dual_one_sided_offset0p51_pressure_only_unsupported_confirmed
```

and the candidate blockers must include:

- `dual_face_one_sided_unsupported`
- `sampling_only_no_coupled_fsi`
- `no_fluent_parity_claim`
- `reference_selection_deferred`

## Anchor-Map Provenance

The preselection artifact must record anchor-map provenance strongly enough for
later regenerated-snapshot work to decide whether the anchor map is still
applicable:

```text
anchor_source_scenario
anchor_source_policy
anchor_source_probe_origin_offset_cells
anchor_source_marker_face_offset_cells
anchor_source_flow_snapshot_sha256
anchor_source_marker_geometry_sha256
anchor_source_pressure_probe_origin_sha256
anchor_map_sha256
```

Each anchored row should repeat or reference this provenance and should carry
the same `anchor_map_sha256`.

## Artifact Test Contract

Add:

```text
tests/integration/test_ansys_vertical_flap_traction_pressure_pair_reference_preselection_artifacts.py
```

The tests must assert:

- matrix, CSV, history, summary, marker diagnostics, and checksums exist
- `source_script` is repo-relative and not an absolute local path
- every completed row uses the expected shared snapshot SHA
- completed rows are sampling-only: no solid advancement, no feedback
- `candidate_status == pressure_pair_policy_preselection_candidate_found`
- `pressure_pair_policy_candidate == baseline_anchored_cell_pair`
- `reference_formulation_candidate is None`
- `absolute_baseline_bias <= 0.01`
- anchored force span is within gate
- all anchored rows have anchor selected for all markers
- all anchored rows have zero anchor fallback markers
- all anchored rows share one anchor-map SHA
- the one-sided row is present and unsupported
- candidate blockers include the expected deferred/unsupported/non-claim reasons
- marker diagnostics include anchor fields and pressure-pair fields
- summary and checksums match the generated artifacts

The test should not run the runner. It should verify the committed artifacts.

## Workflow Contract

Update:

```text
.github/workflows/ansys-vertical-flap-validation.yml
```

Only add cheap checks:

- include `run_traction_pressure_pair_reference_preselection_matrix.py` in the
  `py_compile` list
- include
  `tests.integration.test_ansys_vertical_flap_traction_pressure_pair_reference_preselection_artifacts`
  in the artifact consistency unittest block

Do not make GitHub Actions execute the runner.

## Validation Plan

Use the reliable local interpreter:

```powershell
& "D:\working\taichi\env\python.exe" -m py_compile validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_pressure_pair_reference_preselection_matrix.py tests\integration\test_ansys_vertical_flap_traction_pressure_pair_reference_preselection_artifacts.py
& "D:\working\taichi\env\python.exe" validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_pressure_pair_reference_preselection_matrix.py
& "D:\working\taichi\env\python.exe" -m unittest tests.integration.test_ansys_vertical_flap_traction_pressure_pair_reference_preselection_artifacts -v
& "D:\working\taichi\env\python.exe" -m unittest tests.integration.test_ansys_vertical_flap_traction_pressure_pair_anchor_map_artifacts tests.integration.test_ansys_vertical_flap_traction_pressure_pair_reference_preselection_artifacts -v
git diff --check
```

If runtime cost is manageable, also run the workflow artifact consistency block
after adding the new test.

## Git And Push Requirements

After validation passes:

1. Confirm `git status --short`.
2. Stage only files belonging to this task.
3. Commit with a conventional message, expected:
   `validation: add ANSYS pressure-pair reference preselection`.
4. Push the current branch to `origin`.
5. Verify the remote branch with `git ls-remote`.

The final report must include:

- commit hash
- remote branch
- validation commands and outcomes
- whether README was checked and updated
- whether push succeeded
- remote-ref verification

## Done Criteria

This task is complete only when:

- this detailed goal file exists
- the active goal references this file
- the new preselection runner exists and generates artifacts
- the preselection artifact records `baseline_anchored_cell_pair` as the
  pressure-pair policy candidate
- `reference_formulation_candidate` remains `None`
- the dual-face one-sided row remains explicitly unsupported
- the artifact tests pass
- workflow cheap checks include the new runner and artifact test
- README has been checked for contract drift
- the commit is pushed and the remote branch is verified
