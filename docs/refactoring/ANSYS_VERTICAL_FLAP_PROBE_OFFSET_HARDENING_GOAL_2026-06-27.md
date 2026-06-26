# ANSYS Vertical Flap Probe Offset Hardening Goal - 2026-06-27

## Source Context

- Repository: `lizhuoh9/EasyFsi`
- Branch at goal creation:
  `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Starting checkpoint:
  `a12d4a4b19a50936e47050bfc369066b9b30e7f9`
- Starting checkpoint status: probe-offset decoupling implementation and
  artifact chain are valid.
- Existing decoupling artifact directory:
  `validation_runs/ansys_vertical_flap_fsi/traction_probe_offset_decoupling_diagnostics/`
- Existing decoupling runner:
  `validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_probe_offset_decoupling_matrix.py`
- Existing decoupling artifact test:
  `tests/integration/test_ansys_vertical_flap_traction_probe_offset_decoupling_artifacts.py`
- Shared snapshot SHA-256:
  `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`
- Shared snapshot source commit:
  `8488848d9302f7c05ffb8fd59342aec9d0a7e36f`

The previous decoupling artifacts showed the important diagnosis:

- fixed force-marker geometry plus swept pressure-probe origin has relative
  force-ratio span `27.89320486200173`;
- fixed pressure-probe origin plus swept force-marker geometry has relative
  force-ratio span `0.0`;
- therefore the current offset pathology is dominated by pressure-probe origin
  and ladder placement, not by force marker geometry itself.

## Objective

Make the completed probe-offset decoupling work harder to regress before moving
to pressure-probe ladder stability or dual-face one-sided pressure work.

This goal must implement the immediate hardening stage only:

1. wire the new decoupling runner and artifact test into the existing ANSYS
   vertical-flap GitHub workflow;
2. make the coupled-run gate reject non-default probe-origin modes;
3. keep fixed-solid diagnostic usage of explicit probe origins legal;
4. strengthen the decoupling artifact test so it protects the numeric diagnosis,
   not just the presence of fields.

## Explicit Non-Goals

- Do not implement the pressure-probe ladder stability matrix.
- Do not create `traction_probe_ladder_stability_diagnostics/`.
- Do not add probe ladder start/spacing config fields in this commit.
- Do not implement dual-face one-sided pressure support.
- Do not select or change `reference_formulation_candidate`.
- Do not run coupled FSI for validation.
- Do not claim Fluent parity.
- Do not change pressure formulas.
- Do not change force aggregation formulas.
- Do not change fluid or solid physics.
- Do not regenerate existing probe-offset decoupling artifacts unless a test
  proves the current artifact content is stale or inconsistent.
- Do not add the GPU-only traction probe solver tests to the default workflow.
  The workflow hardening should be limited to compile and artifact/source-level
  contracts.

## Phase 0 - Goal and Scope Anchor

Create this file as the detailed source-of-truth goal and use a short active
goal that references this path. The active goal should explicitly say the task
is the immediate probe-offset decoupling hardening stage and should not include
the later ladder stability matrix implementation.

Required active-goal shape:

```text
Implement and verify docs/refactoring/ANSYS_VERTICAL_FLAP_PROBE_OFFSET_HARDENING_GOAL_2026-06-27.md: CI wiring, coupled-run probe-origin gate, diagnostic allowance, numeric artifact gates, commit, and push.
```

## Phase 1 - Workflow Coverage

Update:

```text
.github/workflows/ansys-vertical-flap-validation.yml
```

Required compile wiring:

- Add
  `validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_probe_offset_decoupling_matrix.py`
  to the existing `python -m py_compile` block.
- Keep the workflow compile-only for this runner; do not run the runner in CI.
- Keep the workflow on Windows and preserve the current minimal dependency set.

Required artifact-test wiring:

- Add
  `tests.integration.test_ansys_vertical_flap_traction_probe_offset_decoupling_artifacts`
  to the existing archive/artifact consistency unittest block.
- Keep the order near the other traction artifact contracts so the workflow
  remains readable.
- Do not add long GPU solver tests to the default CI workflow.

Acceptance criteria:

- The workflow compiles the decoupling runner by default.
- The workflow runs the decoupling artifact consistency test by default.
- No existing workflow test module is removed.
- No expensive matrix runner is executed by the workflow.

## Phase 2 - Coupled-Run Probe-Origin Gate

Update:

```text
benchmarks/official/solid_mpm_fsi_runner.py
```

The current `_is_default_traction_formulation(config)` must treat probe-origin
configuration as part of the default formulation identity.

Required behavior:

- Default formulation remains true only when:
  - marker layout is `dual_physical_faces`;
  - pressure sampling mode is `two_sided_pressure_jump`;
  - marker face offset is `0.51` cells;
  - viscous contribution is off;
  - probe-origin mode is `marker_position`;
  - probe-origin offset is unset (`None`).
- If `step_count > 0` and the probe-origin mode is
  `physical_face_offset`, `_validate_rectangular_solid_config()` must reject the
  configuration through the existing "fixed-solid diagnostics only" guard.
- If `step_count == 0` and `preflow_steps > 0`, the same
  `physical_face_offset` configuration must remain valid. This is needed for
  explicit diagnostic runners that resample or preflow without advancing
  coupled FSI.
- Existing validation of probe-origin mode, probe-origin offset finiteness,
  non-negativity, diagnostic range, and required offset for
  `physical_face_offset` must remain intact.

Acceptance criteria:

- Default configuration remains valid.
- Fixed-solid diagnostic explicit probe-origin configuration remains valid.
- Coupled non-default probe-origin configuration fails fast.
- Existing one-sided fail-closed behavior is unchanged.

## Phase 3 - Source-Level Tests

Update:

```text
tests/cases/test_ansys_vertical_flap_fsi.py
```

Add source-level coverage near the existing traction formulation control tests.

Required tests:

1. `step_count > 0` plus
   `traction_pressure_probe_origin_mode="physical_face_offset"` plus
   `traction_pressure_probe_origin_offset_cells=0.51` must raise `ValueError`
   with the existing "fixed-solid diagnostics only" message.
2. `step_count=0`, `preflow_steps>0`, and the same explicit probe-origin config
   must pass `_validate_rectangular_solid_config()`.
3. The default config should still be considered default by the guard.
4. A config with explicit `physical_face_offset` should not be considered
   default, even if marker layout, pressure sampling mode, marker offset, and
   viscous settings are otherwise default.

Acceptance criteria:

- Tests prove diagnostic-only probe-origin usage remains possible.
- Tests prove coupled FSI cannot silently use the diagnostic probe-origin mode.
- Tests do not require CUDA.
- Tests do not run the full FSI smoke.

## Phase 4 - Numeric Artifact Gates

Update:

```text
tests/integration/test_ansys_vertical_flap_traction_probe_offset_decoupling_artifacts.py
```

Strengthen the existing hash-decoupling test so it protects the key diagnosis.

Required numeric gates:

- `fixed_marker_probe_origin_ratio_span.relative_span` must be greater than
  `20.0`.
- `fixed_probe_marker_ratio_span.relative_span` must be less than or equal to
  `1.0e-12` in absolute value.
- Every `fixed_probe` row must have `force_ratio_to_group_baseline` equal to
  `1.0` within tight numerical tolerance.
- The fixed-marker low-probe and high-probe rows must continue to capture the
  previously observed pressure-ladder pathology:
  - `fixed_marker0p51_probe0p25.force_ratio_to_group_baseline` must remain
    greater than `1.5`;
  - `fixed_marker0p51_probe1p00.force_ratio_to_group_baseline` must remain
    less than `0.1`.

Acceptance criteria:

- The artifact test fails if a future code change removes the evidence that
  probe-origin placement dominates the offset pathology.
- The artifact test still checks fields, source-script portability, shared
  snapshot identity, checksums, and diagnostic-only non-claims.
- The test relies only on checked-in artifacts and does not regenerate the
  matrix.

## Phase 5 - Validation Commands

Use the repository Python environment available in the current checkout.

Compile checks:

```powershell
python -m py_compile benchmarks/official/solid_mpm_fsi_runner.py validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_probe_offset_decoupling_matrix.py tests/integration/test_ansys_vertical_flap_traction_probe_offset_decoupling_artifacts.py tests/cases/test_ansys_vertical_flap_fsi.py
```

Focused tests:

```powershell
python -m unittest tests.integration.test_ansys_vertical_flap_traction_probe_offset_decoupling_artifacts tests.cases.test_ansys_vertical_flap_fsi -v
```

Workflow syntax/source smoke:

```powershell
python -m py_compile cases/ansys_vertical_flap_fsi.py benchmarks/official/solid_mpm_fsi_runner.py
```

Whitespace verification:

```powershell
git diff --check
```

Local execution may use an explicit interpreter path if that is the reliable
environment for this checkout, but repo documentation and workflow content
should remain path-neutral.

## Phase 6 - Commit and Push

After implementation and verification:

- inspect `git status --short`;
- inspect staged file list before commit;
- commit with a conventional message, preferably:

```text
test: wire ANSYS probe offset decoupling contracts into CI
```

- push the current branch to `origin`;
- verify the remote branch ref points to the new commit;
- report the final commit hash, branch, validation commands, and any benign
  pre-push messages.

## Done Criteria

This goal is complete only when all of the following are true:

- `.github/workflows/ansys-vertical-flap-validation.yml` compiles the decoupling
  runner and runs the decoupling artifact test.
- `_is_default_traction_formulation()` rejects non-default probe-origin config
  as non-default.
- `_validate_rectangular_solid_config()` rejects coupled
  `physical_face_offset` probe origins and allows fixed-solid diagnostic
  `physical_face_offset` probe origins.
- Source-level tests cover both paths.
- Artifact tests lock the key numeric conclusion:
  probe-origin sweep span > 20 and marker sweep span <= 1e-12 on the checked-in
  shared-snapshot artifact.
- All required validation commands pass.
- The working tree is committed and pushed to the configured GitHub remote.

## Deferred Follow-Up

After this hardening commit lands, the next separate goal should be:

```text
docs/refactoring/ANSYS_VERTICAL_FLAP_PRESSURE_PROBE_LADDER_STABILITY_GOAL_2026-06-27.md
```

That later goal should introduce a new
`traction_probe_ladder_stability_diagnostics/` artifact set and investigate the
nearest-cell/rung transition map. It is intentionally not part of this
hardening goal.
