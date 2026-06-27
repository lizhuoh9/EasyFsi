# ANSYS Vertical-Flap Pressure Probe Ladder Start Control Goal - 2026-06-27

## Source checkpoint

This goal starts from remote commit
`f5633b81c5d271ba2adfe1ac9cf96b4e6e32b947` on branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.

That checkpoint already added the pressure-probe ladder stability diagnostic
around one archived shared preflow snapshot. The result localized the dominant
pathology to a probe ladder / origin classification cliff between the
`0.51` and `0.625` probe-origin offsets. The branch must not jump directly to
one-sided pressure, reference formulation selection, coupled FSI, or Fluent
parity claims until the ladder controls are separated and tested.

## Objective

Make the probe ladder classification finding durable, then add explicit
diagnostic controls that separate:

- pressure probe origin,
- pressure ladder start offset,
- ladder spacing,
- ladder rung count,
- and ladder mode.

The default path must preserve the current behavior exactly. Explicit ladder
controls are diagnostic-only and must be allowed only in fixed-solid /
sampling-only validation contexts until a later goal promotes them.

## Non-goals

- Do not implement per-face one-sided pressure.
- Do not select a reference formulation.
- Do not run or claim coupled FSI readiness.
- Do not claim Fluent parity.
- Do not change marker force aggregation.
- Do not change fluid or solid physics.
- Do not overwrite the existing ladder stability artifacts except when
  regenerating them with strictly stronger transition-map summary fields.
- Do not mask the current offset pathology by hardcoding forces, pressures,
  displacements, flow fields, or marker classifications.

## Phase 0 - Harden the existing ladder stability evidence

Strengthen
`tests/integration/test_ansys_vertical_flap_traction_probe_ladder_stability_artifacts.py`
so it locks the key finding from the `f5633b...` artifact:

- `probe_origin_force_ratio_span.relative_span > 40.0`.
- `first_force_collapse_offset_cells == 0.625`.
- `collapse_0p51_to_1p00_has_probe_classification_change is True`.
- `probe_offset0p375.force_ratio_to_baseline > 1.9`.
- `probe_offset0p625.force_ratio_to_baseline < 0.1`.

Update
`validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_probe_ladder_stability_matrix.py`
to add directional high-side transition fields without deleting the existing
fields:

- `first_high_side_force_collapse_offset_cells`.
- `first_high_side_primary_nearest_cell_transition_offset_cells`.
- `first_high_side_secondary_nearest_cell_transition_offset_cells`.
- `nearest_below_baseline_force_amplification_offset_cells`.
- `nearest_below_baseline_force_ratio_to_baseline`.
- `nearest_above_baseline_force_ratio_to_baseline`.

Regenerate the committed ladder stability artifacts and checksums so the
artifact test protects the improved transition semantics.

## Phase 1 - Add explicit ladder controls with default compatibility

Extend `cases/ansys_vertical_flap_fsi.VerticalFlapFsiConfig` with:

- `traction_pressure_probe_start_offset_cells: float | None = None`.
- `traction_pressure_probe_ladder_spacing_cells: float = 0.5`.
- `traction_pressure_probe_ladder_rung_count: int = 5`.
- `traction_pressure_probe_ladder_mode: str = "current_normal_cell_ladder"`.

The public config defaults must be non-disruptive. A `None` start offset means
the runner must reproduce the existing current pressure-only ladder behavior
for the active code path. Explicit settings may alter the sampled ladder
positions and must be reflected in marker diagnostics.

Add helper accessors and validation in
`benchmarks/official/solid_mpm_fsi_runner.py`:

- finite, non-negative start offset when provided,
- finite, positive spacing,
- positive rung count,
- supported ladder mode only,
- non-default ladder controls are fixed-solid diagnostics only when
  `step_count > 0`,
- `step_count == 0` with preflow/shared snapshot diagnostics remains allowed.

Pass the ladder controls into
`HibmMpmSurfaceMarkers.sample_fluid_stress_to_marker_tractions`.

## Phase 2 - Implement the pressure-only ladder control path

Extend `simulation_core/coupling/hibm_mpm/core.py` so the pressure-only
two-sided sampling path can use explicit ladder start, spacing, and rung count.

Compatibility rule:

- unset start offset keeps the current integer ladder behavior unchanged;
- explicit start offset uses
  `start + spacing * rung_index` cell-distance multipliers;
- diagnostic fields must continue to report the selected rung, multiplier,
  probe distance, grid coordinate, nearest cell, and ladder mode.

This phase must not alter force aggregation, marker placement, fluid state,
solid state, or coupled-loop ordering.

## Phase 3 - Add solver-level tests

Add or extend solver tests to cover:

- default ladder controls preserve the existing pressure jump, nearest cells,
  rung diagnostics, and marker position behavior;
- explicit ladder start changes sampling classification without moving the
  marker or changing the pressure probe origin;
- invalid controls fail fast with clear `ValueError` messages;
- non-default ladder controls are rejected for positive-step coupled runs;
- preflow/shared snapshot diagnostic mode remains allowed.

The tests should stay focused and not require a long coupled simulation.

## Phase 4 - Add the ladder-control diagnostic matrix

Add:

`validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_probe_ladder_control_matrix.py`

and write artifacts under:

`validation_runs/ansys_vertical_flap_fsi/traction_probe_ladder_control_diagnostics/`

The runner must:

- reuse the archived shared snapshot with SHA
  `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`;
- keep `marker_face_offset_cells = 0.51`;
- use `physical_face_offset` probe origins;
- use two-sided pressure jump sampling;
- be pressure-only and sampling-only;
- never advance fluid, solid, or coupled FSI state.

Include candidate strategies that isolate ladder controls, for example:

- current behavior / current control baseline;
- origin `0.51`, start `1.00`, spacing `0.50`;
- origin `0.51`, start `0.75`, spacing `0.50`;
- origin `0.51`, start `0.625`, spacing `0.25`;
- origin `0.51`, start `0.51`, spacing `0.25`;
- a symmetric-cell-pair policy placeholder that remains report-only unless
  implemented with artifact proof.

The runner must record:

- force ratio span,
- pressure-complete marker counts,
- nearest-cell transition counts,
- rung transition counts,
- primary and secondary mean pressure jump,
- traction decomposition residuals,
- marker diagnostics,
- history JSON,
- summary Markdown,
- checksums.

## Phase 5 - Acceptance gate for stable ladder candidates

The ladder-control artifact may mark
`probe_ladder_control_stable_candidate_found` only if at least one explicit
strategy satisfies all of these conditions:

- all rows completed;
- one shared snapshot SHA across all completed rows;
- all primary and secondary pressure-complete marker counts equal marker count;
- all invalid marker counts are zero;
- force ratio relative span for that strategy is `<= 0.10`;
- traction decomposition residual is `<= 1e-8`;
- no coupled FSI claim;
- no Fluent parity claim;
- `reference_formulation_candidate is None`.

If no strategy satisfies this, the artifact must honestly report:

`probe_ladder_control_no_stable_candidate`

and still keep `reference_formulation_candidate = None`.

## Phase 6 - Artifact tests and workflow wiring

Add:

`tests/integration/test_ansys_vertical_flap_traction_probe_ladder_control_artifacts.py`

The test must check:

- matrix, CSV, history, summary, checksums, and marker diagnostics exist;
- `source_script` is repo-relative;
- shared snapshot SHA matches the manifest;
- rows are sampling-only with no solid advance and no feedback application;
- candidate status is diagnostic-only or no-stable-candidate/stable-candidate
  according to the gate;
- `reference_formulation_candidate is None`;
- blockers include no one-sided, no coupled FSI, no Fluent parity, and no
  reference selection;
- transition metrics are recorded;
- if a stable candidate is present, every numeric acceptance gate is satisfied.

Wire only cheap checks into
`.github/workflows/ansys-vertical-flap-validation.yml`:

- `py_compile` for the new runner;
- `unittest` for the new artifact test.

Do not run the GPU artifact-generation runner in CI.

## Verification requirements

Use the trusted local interpreter:

`D:\working\taichi\env\python.exe`

Minimum verification before push:

1. `py_compile` the changed Python modules and tests.
2. Run the strengthened ladder stability artifact test.
3. Run the new ladder-control runner to generate artifacts.
4. Run the new ladder-control artifact test.
5. Run the affected solver tests.
6. Run the workflow-equivalent artifact consistency slice.
7. Run `git diff --check`.
8. Review `git status --short` before staging.
9. Stage only files belonging to this goal.
10. Commit and push to the current GitHub branch.
11. Verify the remote ref points at the pushed commit.

## Done criteria

This goal is complete only when:

- the detailed goal file is committed;
- the active Codex goal references this file;
- the transition-map conclusions are hardened in tests;
- explicit ladder controls exist and default compatibility is tested;
- the ladder-control runner and committed artifacts exist;
- artifact tests and workflow cheap checks are wired;
- local verification passes;
- the commit is pushed to GitHub;
- the final response reports the commit hash, branch, remote verification,
  key artifact conclusions, and exact validation commands/results.
