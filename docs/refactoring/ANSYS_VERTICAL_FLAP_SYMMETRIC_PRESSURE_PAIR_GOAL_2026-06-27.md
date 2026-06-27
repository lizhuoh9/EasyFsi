# ANSYS Vertical-Flap Symmetric Pressure Pair Goal - 2026-06-27

## Source checkpoint

This goal starts from remote commit
`bd896aa8a5daddcc36e1da3688c55cdfb1e0ff9f` on branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`.

That checkpoint added default-compatible pressure-probe ladder controls and a
shared-snapshot ladder-control diagnostic matrix. The matrix correctly reported
no stable ladder candidate: ordinary start/spacing strategies reduced the
force-ratio span from catastrophic values to roughly `0.828`, but this still
misses the `<= 0.10` stability gate. The next step must change the pressure
cell-selection policy rather than continue broad start/spacing sweeps.

## Objective

Implement a diagnostic-only symmetric pressure cell-pair policy for two-sided
pressure sampling, then use the archived shared snapshot to test whether that
policy stabilizes the ANSYS vertical-flap pressure traction under probe-origin
offset sweeps.

The new policy must keep the pressure formula and force aggregation unchanged:

`pressure_jump = inside_pressure - outside_pressure`

`traction = pressure_jump * normal`

Only the inside/outside pressure sample selection mechanism may change.

## Non-goals

- Do not implement per-face one-sided pressure.
- Do not select a reference formulation.
- Do not run coupled FSI.
- Do not claim Fluent parity.
- Do not change marker force aggregation.
- Do not change fluid or solid physics.
- Do not overwrite existing ladder-control artifacts.
- Do not mask pressure, force, displacement, flow, or marker classification
  values with hardcoded shortcuts.

## Phase 0 - Preserve current checkpoint boundaries

Keep the current ladder-control matrix as an honest checkpoint. Do not expand
ordinary start/spacing sweeps as the main fix path. The existing artifact shows
ordinary ladder controls do not satisfy the stability gate.

The new work must produce separate artifacts under:

`validation_runs/ansys_vertical_flap_fsi/traction_symmetric_pressure_pair_diagnostics/`

and must leave the previous ladder-control artifact directory intact.

## Phase 1 - Add explicit pressure pair config

Extend `cases/ansys_vertical_flap_fsi.VerticalFlapFsiConfig` with:

- `traction_pressure_pair_policy: str = "independent_ladder"`.
- `traction_pressure_pair_max_cell_delta: int = 1`.
- `traction_pressure_pair_require_opposite_sides: bool = True`.

Supported policies:

- `independent_ladder`: current default behavior.
- `symmetric_cell_pair`: diagnostic-only paired pressure selection.

Default behavior must be unchanged. A default config must still be considered
the default traction formulation.

Runner validation must:

- reject unsupported policies;
- reject negative pair max cell delta;
- allow `symmetric_cell_pair` only for fixed-solid / sampling-only diagnostics;
- reject `step_count > 0` when the pair policy is non-default;
- allow `step_count == 0` with preflow/shared-snapshot diagnostics.

## Phase 2 - Implement symmetric pressure pair sampling

Extend `simulation_core/coupling/hibm_mpm/core.py` pressure-only two-sided
sampling so `symmetric_cell_pair` selects inside and outside pressure samples
as one pair rather than independently.

Minimum required behavior:

- `independent_ladder` preserves current output exactly.
- `symmetric_cell_pair` walks the same normal-cell ladder but accepts a rung
  only when both inside and outside samples are valid on the same rung.
- The pressure jump and traction equations remain unchanged.
- If no pair is found, the marker must fail closed for two-sided pressure; no
  silent fallback to independent ladder.
- The implementation must remain pressure-only diagnostic scoped. If a caller
  combines explicit pair policy with unsupported non-pressure-only paths, fail
  fast.

## Phase 3 - Add marker diagnostics for pair selection

Every marker diagnostic must expose:

- `pressure_pair_policy`.
- `pressure_pair_selected`.
- `pressure_pair_fallback_used`.
- `pressure_pair_inside_cell`.
- `pressure_pair_outside_cell`.
- `pressure_pair_cell_delta`.
- `pressure_pair_symmetry_residual_cells`.

For `independent_ladder`, `pressure_pair_selected` is false and fallback is
false. For `symmetric_cell_pair`, selected markers must show the actual cell
pair and residual. The artifact must be able to prove whether the new policy
really selected paired samples for every marker.

## Phase 4 - Add solver-level tests

Add:

`tests/solvers/test_hibm_traction_symmetric_pressure_pair.py`

Required tests:

1. Default `independent_ladder` preserves current output and diagnostics.
2. `symmetric_cell_pair` selects a same-rung inside/outside pair in a step
   pressure field.
3. Changing pressure probe origin preserves selected pair behavior within the
   configured diagnostic tolerance when a symmetric pair exists.
4. Invalid pair policy and malformed config values fail fast.
5. The official runner rejects non-default pair policy for positive-step
   coupled configs and allows it for `step_count=0`, `preflow_steps>0`.

These tests must remain focused and must not run a coupled FSI case.

## Phase 5 - Add shared-snapshot diagnostic runner

Add:

`validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_symmetric_pressure_pair_matrix.py`

Output artifacts under:

`validation_runs/ansys_vertical_flap_fsi/traction_symmetric_pressure_pair_diagnostics/`

The runner must use:

- shared snapshot SHA
  `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`;
- `marker_face_offset_cells = 0.51`;
- physical-face-offset probe origins;
- two-sided pressure;
- pressure-only sampling;
- no flow advance;
- no solid advance;
- no coupled FSI.

Minimum rows:

- `independent_ladder_baseline_probe0p51`.
- `independent_ladder_baseline_probe0p625`.
- `independent_ladder_baseline_probe1p00`.
- `symmetric_pair_probe0p51`.
- `symmetric_pair_probe0p625`.
- `symmetric_pair_probe1p00`.
- `symmetric_pair_probe0p00`.
- `symmetric_pair_probe0p25`.
- `symmetric_pair_probe0p375`.
- `symmetric_pair_probe0p75`.
- `symmetric_pair_probe1p50`.

The runner must write matrix JSON/CSV, history JSON, marker diagnostics,
summary Markdown, and checksums.

## Phase 6 - Stable candidate gate

The symmetric pair artifact may report:

`symmetric_pressure_pair_stable_candidate_found`

only if the symmetric policy satisfies all gates:

- one shared snapshot SHA across all completed rows;
- all required rows completed;
- all primary and secondary pressure-complete marker counts equal marker count;
- invalid marker count is zero;
- pair selected count equals marker count;
- pair fallback count is zero;
- force-ratio relative span is `<= 0.10`;
- traction decomposition residual is `<= 1e-8`;
- no coupled FSI claim;
- no Fluent parity claim;
- `reference_formulation_candidate is None`.

If any gate fails, report:

`symmetric_pressure_pair_no_stable_candidate`

and keep `reference_formulation_candidate = None`.

## Phase 7 - Add artifact tests and workflow cheap checks

Add:

`tests/integration/test_ansys_vertical_flap_traction_symmetric_pressure_pair_artifacts.py`

The test must verify:

- matrix, CSV, history, summary, checksums, and marker diagnostics exist;
- `source_script` is repo-relative;
- shared snapshot SHA matches the manifest;
- row scope is sampling-only;
- no solid advance and no feedback application;
- candidate status is honest;
- `reference_formulation_candidate is None`;
- blocker list includes no one-sided, no coupled FSI, no Fluent parity, and no
  reference selection;
- marker diagnostics include all pair-policy fields;
- if a stable candidate is present, every numeric acceptance gate is satisfied;
- if no stable candidate exists, no accepted strategy is reported.

Wire only cheap checks into
`.github/workflows/ansys-vertical-flap-validation.yml`:

- `py_compile` for the new runner;
- `unittest` for the new artifact test.

Do not run the GPU artifact-generation runner in CI.

## Phase 8 - Stop condition

This goal stops after the symmetric-pair diagnostic matrix and artifacts are
committed and pushed. It must not continue into one-sided pressure or reference
selection even if a stable symmetric pair is found.

## Verification requirements

Use:

`D:\working\taichi\env\python.exe`

Minimum verification:

1. Run the new solver tests and confirm they fail before implementation.
2. Implement the minimum core/config/runner changes.
3. Run `py_compile` over changed source, runner, and tests.
4. Run the new solver tests.
5. Run the new symmetric-pair runner to generate artifacts.
6. Run the new artifact test.
7. Run the workflow-equivalent artifact consistency block.
8. Run relevant existing traction probe tests.
9. Run `git diff --check`.
10. Verify no new artifact contains local absolute paths.
11. Commit only files belonging to this goal.
12. Push to the current GitHub branch.
13. Verify the remote ref points at the pushed commit.

## Done criteria

This goal is complete only when:

- this goal file is committed;
- the active Codex goal references this file;
- default independent ladder behavior is preserved;
- `symmetric_cell_pair` exists as diagnostic-only pressure pair policy;
- marker diagnostics expose pair-selection evidence;
- shared-snapshot symmetric-pair artifacts are committed;
- artifact tests protect the candidate gate and non-claims;
- workflow cheap checks include the new runner/test but do not run the GPU
  runner;
- validation passes locally;
- the commit is pushed to GitHub and remote ref verification succeeds.
