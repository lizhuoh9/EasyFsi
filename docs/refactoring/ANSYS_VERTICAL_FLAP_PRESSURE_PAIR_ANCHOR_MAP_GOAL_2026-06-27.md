# ANSYS Vertical-Flap Pressure Pair Anchor Map Goal - 2026-06-27

## Source checkpoint

This goal starts from remote commit
`c09eed6de30d4422c47731e9aad0d48460592f05` on branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.

That checkpoint is a valid honest-negative symmetric pressure-pair stage. It
implemented `independent_ladder` and `symmetric_cell_pair`, added marker
pair-selection diagnostics, generated a shared-snapshot matrix, and proved:

- all symmetric-pair rows selected pressure pairs for every marker;
- fallback was zero;
- pair residuals were small;
- force-ratio span was still far above the acceptance gate;
- no reference formulation, coupled FSI, or Fluent parity claim was made.

The next diagnostic must stop tuning a moving probe-origin pair and test whether
fixed pressure-pair cells, anchored from a baseline row, remove the force-span
pathology.

## Objective

Harden the committed symmetric-pair negative evidence, then implement a
diagnostic-only baseline-anchored pressure-pair policy. Use the archived shared
snapshot to answer one question:

If every marker reuses the same inside/outside pressure cells captured from the
baseline probe-origin row, does the ANSYS vertical-flap force ratio become
stable across probe-origin offsets?

## Non-goals

- Do not implement per-face one-sided pressure.
- Do not select a reference formulation.
- Do not run coupled FSI.
- Do not claim Fluent parity.
- Do not change marker force aggregation.
- Do not change fluid or solid physics.
- Do not overwrite previous ladder-control or symmetric-pair artifacts.
- Do not hardcode pressure, force, displacement, flow, or marker outputs to
  manufacture a stable result.

## Phase 0 - Harden symmetric-pair negative evidence

Update
`tests/integration/test_ansys_vertical_flap_traction_symmetric_pressure_pair_artifacts.py`
so the current negative conclusion cannot silently drift.

The test must assert:

- `candidate_status == "symmetric_pressure_pair_no_stable_candidate"`;
- `stable_symmetric_pressure_pair_candidate is None`;
- `symmetric_pair_acceptance["accepted"] is False`;
- `force_ratio_relative_span > 40.0`;
- `pair_selected_all_markers is True`;
- `pair_fallback_zero is True`.

It must also lock representative row behavior:

- `symmetric_pair_probe0p51` force ratio is `1.0`;
- `symmetric_pair_probe0p625` force ratio is below `0.1`;
- `symmetric_pair_probe0p375` force ratio is above `1.9`.

This phase changes only artifact tests and does not modify physics.

## Phase 1 - Add anchor-map config surface

Extend `cases/ansys_vertical_flap_fsi.VerticalFlapFsiConfig` and the official
runner to support a new diagnostic-only pair policy:

`baseline_anchored_cell_pair`

Supported policies after this goal:

- `independent_ladder`;
- `symmetric_cell_pair`;
- `baseline_anchored_cell_pair`.

Default behavior must remain unchanged. A default config must still be the
default traction formulation.

Runner validation must:

- reject unsupported pair policies;
- reject malformed anchor policy values;
- continue rejecting non-default pair policies when `step_count > 0`;
- allow non-default pair policies for `step_count == 0`, `preflow_steps > 0`
  shared-snapshot diagnostics.

## Phase 2 - Add core anchor-cell storage and API

Extend `simulation_core/coupling/hibm_mpm/core.py` with per-marker pressure-pair
anchor storage:

- active flag;
- inside anchor cell;
- outside anchor cell.

Expose a public API:

`set_pressure_pair_anchor_cells(inside_cells, outside_cells)`

Required behavior:

- input lengths must equal the loaded marker count;
- every cell must have exactly three finite integer components;
- negative indices are rejected at API entry;
- reset/load paths must leave inactive markers on unset sentinels;
- anchor storage is Taichi-resident and sampled in the pressure-only kernel.

## Phase 3 - Implement baseline anchored pair sampling

Extend `sample_fluid_stress_to_marker_tractions()` so
`pressure_pair_policy="baseline_anchored_cell_pair"` is pressure-only,
two-sided, diagnostic-only.

For each marker:

- require anchor active;
- read inside pressure directly from the marker's anchored inside cell;
- read outside pressure directly from the marker's anchored outside cell;
- use the same pressure jump and traction equations:
  `pressure_jump = inside_pressure - outside_pressure`;
  `traction = pressure_jump * normal`;
- record inside/outside pressure found, nearest cells, pair selected, and no
  fallback;
- if anchors are missing or invalid, fail closed with two-sided pressure
  missing diagnostics.

Do not fall back silently to independent ladder or symmetric pair selection.

## Phase 4 - Add anchor diagnostics

Every marker diagnostic must include:

- `pressure_pair_anchor_active`;
- `pressure_pair_anchor_inside_cell`;
- `pressure_pair_anchor_outside_cell`;
- `pressure_pair_anchor_source`;
- `pressure_pair_anchor_fallback_used`.

For `baseline_anchored_cell_pair`, every accepted marker must show active
anchors and fallback false. Existing pair fields must continue to report policy,
selected/fallback, inside/outside pair cells, cell delta, and residual.

## Phase 5 - Add solver-level tests

Add:

`tests/solvers/test_hibm_traction_pressure_pair_anchor_map.py`

Required tests:

1. Default independent ladder behavior remains unchanged.
2. `baseline_anchored_cell_pair` reads pressures from explicitly configured
   inside/outside cells.
3. Changing pressure probe origin does not change anchored cells or pressure
   jump.
4. Missing anchors under anchored policy fail closed without fallback.
5. Invalid anchor shape/count/index values fail fast.
6. Positive-step coupled config rejects the anchored policy; `step_count=0`,
   `preflow_steps>0` diagnostics allow it.

These tests must remain fixed-solid / pressure-only and must not run coupled
FSI.

## Phase 6 - Add shared-snapshot anchor-map runner

Add:

`validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_pressure_pair_anchor_map_matrix.py`

Output artifacts under:

`validation_runs/ansys_vertical_flap_fsi/traction_pressure_pair_anchor_map_diagnostics/`

The runner must use:

- shared snapshot SHA
  `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`;
- source commit `8488848d9302f7c05ffb8fd59342aec9d0a7e36f`;
- `marker_face_offset_cells = 0.51`;
- baseline probe-origin offset `0.51`;
- physical-face-offset probe origins;
- two-sided pressure;
- pressure-only sampling;
- no flow advance;
- no solid advance;
- no coupled FSI.

Minimum rows:

- `baseline_independent_probe0p51`;
- `anchored_from_baseline_probe0p00`;
- `anchored_from_baseline_probe0p25`;
- `anchored_from_baseline_probe0p375`;
- `anchored_from_baseline_probe0p51`;
- `anchored_from_baseline_probe0p625`;
- `anchored_from_baseline_probe0p75`;
- `anchored_from_baseline_probe1p00`;
- `anchored_from_baseline_probe1p50`.

The runner must:

- run the baseline row first;
- derive each marker's anchor map from that baseline row's inside/outside
  nearest cells;
- reuse the exact anchor cells for all anchored rows;
- write matrix JSON/CSV, history JSON, marker diagnostics, summary Markdown,
  and checksums;
- keep every emitted path repo-relative.

## Phase 7 - Anchor-map acceptance gate

The artifact may report:

`pressure_pair_anchor_map_stable_candidate_found`

only if the anchored policy satisfies all gates:

- one shared snapshot SHA across all completed rows;
- all required rows completed;
- anchor selected marker count equals marker count;
- anchor fallback marker count is zero;
- pressure-complete counts equal marker counts on both faces;
- invalid marker count is zero;
- force-ratio relative span is `<= 0.10`;
- traction decomposition residual is `<= 1e-8`;
- no coupled FSI claim;
- no Fluent parity claim;
- `reference_formulation_candidate is None`.

If any gate fails, report:

`pressure_pair_anchor_map_no_stable_candidate`

and keep `reference_formulation_candidate = None`.

Even if the anchor map is stable, this goal must not select a reference
formulation. It only proves or disproves the cell-selection instability
hypothesis.

## Phase 8 - Add artifact tests and workflow cheap checks

Add:

`tests/integration/test_ansys_vertical_flap_traction_pressure_pair_anchor_map_artifacts.py`

The test must verify:

- matrix, CSV, history, summary, checksums, and marker diagnostics exist;
- `source_script` is repo-relative;
- shared snapshot SHA matches the manifest;
- row scope is sampling-only;
- no solid advance and no feedback application;
- candidate status is honest;
- `reference_formulation_candidate is None`;
- blockers include no one-sided, no coupled FSI, no Fluent parity, and no
  reference selection;
- marker diagnostics include all anchor fields;
- if a stable candidate is present, every numeric gate is satisfied;
- if no stable candidate exists, no accepted strategy is reported.

Wire only cheap checks into
`.github/workflows/ansys-vertical-flap-validation.yml`:

- `py_compile` for the new runner;
- `unittest` for the new artifact test;
- add `tests.solvers.test_hibm_traction_symmetric_pressure_pair` and the new
  anchor-map solver test to the CUDA-gated solver block without enabling CUDA by
  default.

Do not run GPU artifact-generation runners in CI.

## Phase 9 - Stop condition

Stop after the anchor-map diagnostic matrix and artifacts are committed and
pushed. Do not proceed to one-sided pressure or reference selection in this
goal.

## Verification requirements

Use:

`D:\working\taichi\env\python.exe`

Minimum verification:

1. Run the new/updated tests before implementation and observe the expected
   failure.
2. Implement the minimum core/config/runner changes.
3. Run `py_compile` over changed source, runner, and tests.
4. Run the symmetric-pair artifact hardening test.
5. Run the new anchor-map solver tests.
6. Run the new anchor-map runner to generate artifacts.
7. Run the new anchor-map artifact test.
8. Run the workflow-equivalent artifact consistency block.
9. Run relevant existing traction probe tests.
10. Run `git diff --check`.
11. Verify no new artifact contains local absolute paths.
12. Commit only files belonging to this goal.
13. Push to the current GitHub branch.
14. Verify the remote ref points at the pushed commit.

## Done criteria

This goal is complete only when:

- this goal file is committed;
- the active Codex goal references this file;
- symmetric-pair negative evidence is hardened;
- default independent ladder behavior is preserved;
- `baseline_anchored_cell_pair` exists as a diagnostic-only pressure-pair
  policy;
- per-marker anchor-cell API and diagnostics exist;
- shared-snapshot anchor-map artifacts are committed;
- artifact tests protect candidate gates and non-claims;
- workflow cheap checks include the new runner/test and CUDA-gated solver test
  names but do not run GPU artifact generation;
- validation passes locally;
- the commit is pushed to GitHub and remote ref verification succeeds.
