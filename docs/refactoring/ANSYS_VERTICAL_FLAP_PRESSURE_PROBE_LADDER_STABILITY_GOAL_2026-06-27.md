# ANSYS Vertical Flap Pressure Probe Ladder Stability Goal - 2026-06-27

## Source Context

- Repository: `lizhuoh9/EasyFsi`
- Branch at goal creation:
  `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Starting checkpoint:
  `46db03206ef33ee68fdf2c367c95c4beb21dc821`
- Prior checkpoint status: phase 0 probe-offset hardening is complete.
- Existing probe-offset hardening goal:
  `docs/refactoring/ANSYS_VERTICAL_FLAP_PROBE_OFFSET_HARDENING_GOAL_2026-06-27.md`
- Existing probe-offset decoupling runner:
  `validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_probe_offset_decoupling_matrix.py`
- Existing probe-offset decoupling artifacts:
  `validation_runs/ansys_vertical_flap_fsi/traction_probe_offset_decoupling_diagnostics/`
- Existing probe-offset decoupling artifact test:
  `tests/integration/test_ansys_vertical_flap_traction_probe_offset_decoupling_artifacts.py`
- Shared snapshot SHA-256:
  `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`
- Shared snapshot source commit:
  `8488848d9302f7c05ffb8fd59342aec9d0a7e36f`

The previous probe-offset decoupling matrix established the key diagnosis:

- fixed force-marker geometry plus swept pressure-probe origin has relative
  force-ratio span `27.89320486200173`;
- fixed pressure-probe origin plus swept force-marker geometry has relative
  force-ratio span `0.0`;
- therefore the current offset pathology is dominated by pressure-probe origin
  and ladder placement, not by force marker geometry itself.

## Objective

Build the next diagnostic stage: pressure-probe ladder stability.

The goal is to keep force-marker geometry fixed, reuse the same archived shared
flow snapshot, sweep pressure-probe origin offsets more finely, and produce a
transition map that explains whether the force amplification/collapse is caused
by probe nearest-cell and rung transitions.

This is a diagnostic evidence step only. It must not change solver physics,
select a reference formulation, or claim Fluent parity.

## Explicit Non-Goals

- Do not implement dual-face one-sided pressure support.
- Do not select a reference formulation.
- Do not change `reference_formulation_candidate` from `None`.
- Do not run coupled 50-step FSI.
- Do not run any coupled FSI validation.
- Do not claim Fluent parity.
- Do not change fluid or solid physics.
- Do not change pressure formulas.
- Do not change force aggregation formulas.
- Do not change ANSYS material constants, geometry, grid dimensions, source
  schedule, support radii, damping, or official-web targets.
- Do not overwrite existing
  `traction_probe_offset_decoupling_diagnostics/` artifacts.
- Do not introduce probe ladder start/spacing controls in this first diagnostic
  commit.
- Do not add expensive matrix execution to the default GitHub workflow.

## Required Output Files

Create the detailed diagnostic runner:

```text
validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_probe_ladder_stability_matrix.py
```

Create a new artifact directory:

```text
validation_runs/ansys_vertical_flap_fsi/traction_probe_ladder_stability_diagnostics/
```

The artifact directory must include:

- `traction_probe_ladder_stability_matrix.json`
- `traction_probe_ladder_stability_matrix.csv`
- `traction_probe_ladder_stability_history.json`
- `traction_probe_ladder_transition_map.json`
- `traction_probe_ladder_stability_summary.md`
- `CHECKSUMS.sha256`
- marker diagnostics under `marker_diagnostics/`

Create a committed artifact test:

```text
tests/integration/test_ansys_vertical_flap_traction_probe_ladder_stability_artifacts.py
```

Update the ANSYS workflow:

```text
.github/workflows/ansys-vertical-flap-validation.yml
```

## Phase 0 - Active Goal

Create this detailed markdown file first and then set a short active goal that
references it.

Required active-goal shape:

```text
Implement and verify docs/refactoring/ANSYS_VERTICAL_FLAP_PRESSURE_PROBE_LADDER_STABILITY_GOAL_2026-06-27.md: shared-snapshot ladder stability runner, transition-map artifacts, artifact tests, workflow cheap checks, commit, and push.
```

## Phase 1 - Shared-Snapshot Ladder Stability Runner

The new runner must be read-only with respect to solver physics.

Fixed configuration:

- `marker_face_offset_cells = 0.51`
- `traction_pressure_probe_origin_mode = physical_face_offset`
- `traction_pressure_sampling_mode = two_sided_pressure_jump`
- `traction_include_viscous = False`
- shared snapshot SHA
  `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`
- no flow advance
- no solid advance
- no coupled FSI loop

Probe-origin offsets to sweep:

```text
0.00
0.125
0.25
0.375
0.51
0.625
0.75
0.875
1.00
1.25
1.50
```

The baseline scenario must be the `0.51` row, matching the prior decoupling
baseline.

Each row must include at least:

- `scenario`
- `run_status`
- `formulation_status`
- `probe_origin_offset_cells`
- `marker_face_offset_cells`
- `total_force_z_N`
- `force_ratio_to_baseline`
- `primary_face_mean_pressure_jump_pa`
- `secondary_face_mean_pressure_jump_pa`
- `primary_face_inside_probe_rung_histogram`
- `primary_face_outside_probe_rung_histogram`
- `secondary_face_inside_probe_rung_histogram`
- `secondary_face_outside_probe_rung_histogram`
- `primary_face_inside_unique_nearest_cell_count`
- `primary_face_outside_unique_nearest_cell_count`
- `secondary_face_inside_unique_nearest_cell_count`
- `secondary_face_outside_unique_nearest_cell_count`
- `primary_face_pressure_complete_marker_count`
- `secondary_face_pressure_complete_marker_count`
- `marker_geometry_sha256`
- `pressure_probe_origin_sha256`
- `marker_diagnostics_json`
- `flow_snapshot_sha256`
- `flow_snapshot_source_commit`
- `scope_limit`

Rows may include additional face diagnostics from the existing public
`stress_face_diagnostics` route.

## Phase 2 - Transition Map Artifact

Create:

```text
traction_probe_ladder_transition_map.json
```

The transition map must include one entry per probe-origin offset with:

- `offset_cells`
- `scenario`
- `force_ratio_to_baseline`
- `total_force_z_N`
- `primary_inside_nearest_cell_histogram`
- `primary_outside_nearest_cell_histogram`
- `secondary_inside_nearest_cell_histogram`
- `secondary_outside_nearest_cell_histogram`
- `primary_inside_rung_histogram`
- `primary_outside_rung_histogram`
- `secondary_inside_rung_histogram`
- `secondary_outside_rung_histogram`
- `primary_mean_pressure_jump_pa`
- `secondary_mean_pressure_jump_pa`
- `primary_pressure_complete_marker_count`
- `secondary_pressure_complete_marker_count`

The top-level transition map must also summarize:

- first offset where force ratio exceeds `1.5`;
- first offset where force ratio drops below `0.1`;
- first offset where a primary-face nearest-cell histogram changes relative to
  baseline;
- first offset where a secondary-face nearest-cell histogram changes relative
  to baseline;
- whether the observed force collapse from `0.51` to `1.00` is accompanied by
  nearest-cell or rung histogram changes.

## Phase 3 - Summary Markdown

Create:

```text
traction_probe_ladder_stability_summary.md
```

The summary must explicitly state:

- the artifact reuses one archived shared preflow snapshot;
- it only re-runs marker traction sampling;
- it does not advance flow;
- it does not advance solid;
- it does not run coupled FSI;
- it does not claim Fluent parity;
- it does not select a reference formulation;
- it explains which probe offsets amplify or collapse the force ratio;
- it reports whether transition-map evidence points to nearest-cell/rung
  classification changes.

## Phase 4 - Artifact Tests

Create:

```text
tests/integration/test_ansys_vertical_flap_traction_probe_ladder_stability_artifacts.py
```

The test must not run the GPU runner. It must validate committed artifacts only.

Required artifact contract:

- matrix JSON, CSV, history JSON, transition map JSON, summary MD, checksums,
  and marker diagnostics exist;
- all rows completed;
- all rows share the exact shared snapshot SHA;
- `source_script` is repo-relative;
- every row and the top-level payload say `shared snapshot`, `sampling-only`,
  and `does not claim Fluent parity`;
- `candidate_status` is `probe_ladder_stability_diagnostic_only`;
- `reference_formulation_candidate` is `None`;
- candidate blockers include no-reference, no-coupled-FSI, no-Fluent, and
  one-sided unsupported blockers;
- all expected probe offsets are present;
- transition-map entries cover every offset;
- transition-map checksums match;
- marker diagnostics expose pressure-probe origin and probe ladder fields;
- checksums cover matrix, CSV, history, summary, transition map, and marker
  diagnostics.

Required numeric gates for the current ladder behavior:

- `offset0p25.force_ratio_to_baseline > 1.5`;
- `offset1p00.force_ratio_to_baseline < 0.1`;
- relative force-ratio span across the sweep remains greater than `20.0`;
- baseline `offset0p51.force_ratio_to_baseline` is approximately `1.0`.

These gates preserve the diagnosis. They are not acceptance criteria for a
stable ladder strategy.

## Phase 5 - Workflow Cheap Checks

Update:

```text
.github/workflows/ansys-vertical-flap-validation.yml
```

Required workflow changes:

- add `run_traction_probe_ladder_stability_matrix.py` to the compile block;
- add
  `tests.integration.test_ansys_vertical_flap_traction_probe_ladder_stability_artifacts`
  to the artifact consistency unittest block;
- do not run the ladder stability runner in CI.

## Phase 6 - Validation Commands

Use the repository Python environment available in the current checkout.
Local execution may use an explicit interpreter path if that is the reliable
environment, but committed docs and workflow content should remain path-neutral.

Expected validation sequence:

```powershell
python -m py_compile validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_probe_ladder_stability_matrix.py tests/integration/test_ansys_vertical_flap_traction_probe_ladder_stability_artifacts.py
```

```powershell
python -m unittest tests.integration.test_ansys_vertical_flap_traction_probe_ladder_stability_artifacts -v
```

The first artifact test run should fail before artifacts exist or before the
runner is executed. After generating artifacts, it must pass.

Generate the artifacts:

```powershell
python validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_probe_ladder_stability_matrix.py
```

Run the artifact test again:

```powershell
python -m unittest tests.integration.test_ansys_vertical_flap_traction_probe_ladder_stability_artifacts -v
```

Run the relevant workflow slice:

```powershell
python -m unittest tests.integration.test_ansys_vertical_flap_traction_probe_ladder_stability_artifacts tests.integration.test_ansys_vertical_flap_traction_probe_offset_decoupling_artifacts tests.integration.test_ansys_vertical_flap_traction_snapshot_resampling_artifacts tests.integration.test_ansys_vertical_flap_traction_shared_snapshot_artifacts -v
```

Run whitespace verification:

```powershell
git diff --check
```

## Phase 7 - Commit and Push

After implementation and verification:

- inspect `git status --short`;
- inspect staged file list before commit;
- commit with a conventional message, preferably:

```text
validation: add ANSYS pressure probe ladder stability matrix
```

- push the current branch to `origin`;
- verify the remote branch ref points to the new commit;
- report the final commit hash, branch, validation commands, key artifact
  numbers, and any benign pre-push messages.

## Done Criteria

This goal is complete only when:

- the detailed goal exists and is referenced by the active short goal;
- the ladder stability runner exists;
- the new artifact directory is generated and committed;
- the transition map exists and covers all probe-origin offsets;
- the summary states the sampling-only/no-coupled/no-Fluent/no-reference
  boundaries;
- artifact tests pass and validate both fields and current numeric pathology;
- the workflow compiles the new runner and runs the artifact test;
- `git diff --check` passes;
- the work is committed and pushed to the configured GitHub remote.

## Deferred Follow-Up

Only after the transition map is reviewed should a later goal add configurable
probe ladder start/spacing controls, such as:

```python
traction_pressure_probe_start_offset_cells: float | None = None
traction_pressure_probe_ladder_spacing_cells: float = 0.5
traction_pressure_probe_ladder_rung_count: int = 5
traction_pressure_probe_ladder_mode: str = "cell_normal_ladder"
```

That later work must remain diagnostic-only until a stable ladder candidate is
proven. It is intentionally not part of this goal.
