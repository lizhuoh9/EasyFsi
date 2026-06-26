# ANSYS Vertical Flap Traction Snapshot Resampling Contract Goal - 2026-06-27

## Source Context

- Repository: `lizhuoh9/EasyFsi`
- Working tree: refactored EasyFsi checkout root
- Active branch at goal creation: `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`
- Target runner: `validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_snapshot_resampling_matrix.py`
- Target artifact directory: `validation_runs/ansys_vertical_flap_fsi/traction_snapshot_resampling_diagnostics`
- Target artifact test: `tests/integration/test_ansys_vertical_flap_traction_snapshot_resampling_artifacts.py`
- Shared snapshot manifest: `validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/snapshot_manifest.json`
- Shared snapshot NPZ: `validation_runs/ansys_vertical_flap_fsi/traction_shared_snapshot_diagnostics/step020_fields.npz`

## Objective

Harden the ANSYS vertical-flap shared-snapshot traction resampling evidence chain
so the committed runner, artifacts, tests, and workflow contract are
audit-ready. The implementation direction is already correct: the runner loads
one archived shared preflow snapshot and re-runs only marker stress sampling.
This goal closes the remaining artifact contract drift and reproducibility gaps
without changing the traction physics or making any new Fluent parity claim.

The finished patch must prove that:

- every row-level scope says this is shared-snapshot sampling-only evidence,
- artifacts do not leak local absolute filesystem paths,
- the summary explicitly says no reference formulation is selected,
- every candidate blocker has a non-empty explanatory detail,
- snapshot field shapes are validated before Taichi `from_numpy` assignment,
- committed artifacts are regenerated from the updated runner,
- tests cover these contracts without requiring GPU execution for the contract
  checks,
- the completed branch is committed and pushed to the configured EasyFsi remote.

## Required P0 Fixes

### P0-1 Row-Level Scope Contract

Problem: the matrix top-level `scope_limit` already uses the correct
shared-snapshot resampling wording, but each row still inherits the older
fixed-solid diagnostic wording:

```text
fixed-solid traction formulation diagnostic only; no coupled 50-step or Fluent parity claim
```

Required change:

1. Add a runner-level `RESAMPLING_SCOPE_LIMIT` constant.
2. Use it for the top-level payload `scope_limit`.
3. Override `row["scope_limit"]` in both `_complete_row()` and
   `_unsupported_row()`.
4. Ensure every row says this is shared snapshot, sampling-only evidence and
   does not claim Fluent parity.
5. Ensure no row contains the old fixed-solid wording.

Required test coverage:

- Assert the top-level payload and every row contain `shared snapshot`.
- Assert the top-level payload and every row contain `sampling-only`.
- Assert the top-level payload and every row contain `does not claim Fluent parity`.
- Assert no row contains `fixed-solid traction formulation diagnostic only`.

### P0-2 Repo-Relative Source Script

Problem: `source_script` in the matrix JSON currently stores a machine-local
absolute path, which makes the artifact less portable and exposes private local
directory structure.

Required change:

1. Change `_resampling_payload()` from:

```python
"source_script": str(Path(__file__).resolve())
```

to:

```python
"source_script": _repo_relative(Path(__file__).resolve())
```

Required test coverage:

- Assert `source_script` equals
  `validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_snapshot_resampling_matrix.py`.
- Assert it is not an absolute path.
- Assert it contains no backslashes.
- Assert it contains no local drive prefix or local user/workstation path fragments.

### P0-3 Summary Contract

Problem: the current summary records scope, snapshot identity, counts, offset
ratios, and fail-closed one-sided status, but it does not fully state the
candidate decision contract required by the artifact:

- `reference_formulation_candidate: none`
- `candidate_status: snapshot_resampling_no_reference_selection`
- candidate blocker list
- completed scenario list
- unsupported scenario list
- no Fluent parity
- no coupled 50-step FSI
- next intended step: split marker face offset from pressure-probe start offset

Required change:

1. Extend `_summary_markdown()` with a `Candidate decision` section.
2. List `reference_formulation_candidate: none`.
3. List `candidate_status`.
4. List every candidate blocker by name.
5. Add `Completed scenarios` and `Unsupported scenarios` sections.
6. Add a `Non-claims` section explicitly saying:
   - does not claim Fluent parity,
   - does not run coupled 50-step FSI.
7. Add a `Next step` section saying marker face offset must be split from
   pressure-probe start offset before reference selection is attempted.

Required test coverage:

- Assert summary contains `reference_formulation_candidate`.
- Assert summary contains `none`.
- Assert summary contains `candidate_blockers`.
- Assert summary contains every completed scenario name.
- Assert summary contains the unsupported scenario name.
- Assert summary contains `Fluent parity`.
- Assert summary contains `coupled 50-step FSI`.
- Assert summary contains `split marker offset from pressure-probe`.

### P0-4 Candidate Blocker Detail Upsert

Problem: `_ensure_candidate_blocker()` currently returns as soon as the blocker
already exists. If the existing blocker has an empty `detail`, the runner never
replaces it with the more useful detail text. The current artifact therefore has
empty details for important blockers:

- `required_formulation_unsupported`
- `dual_face_one_sided_unsupported`
- `dual_two_sided_offset_sensitivity_above_tolerance`

Required change:

Implement `_ensure_candidate_blocker()` as an upsert:

```python
def _ensure_candidate_blocker(payload, blocker, detail):
    blockers = payload.setdefault("candidate_blockers", [])
    for item in blockers:
        if item.get("blocker") == blocker:
            if not str(item.get("detail", "")).strip():
                item["detail"] = detail
            return
    blockers.append({"blocker": blocker, "detail": detail})
```

Also call `_ensure_candidate_blocker()` with non-empty detail text for the three
core blockers above.

Required test coverage:

- Assert every item in `candidate_blockers` has a non-empty `detail`.
- Assert at least the three core blockers above have non-empty `detail`.
- Assert `formulation_resampling_only` and `reference_selection_deferred` remain
  present with details.

### P0-5 Snapshot Shape Contract

Problem: `_load_snapshot_fields()` checks required arrays and NPZ hash, but it
does not explicitly validate the pressure, velocity, obstacle, and grid
coordinate shapes against the manifest and baseline config. Shape mismatches are
therefore left to Taichi `from_numpy` failures instead of producing a clear
runner error.

Required change:

1. Add a pure Python `_validate_snapshot_fields(fields, manifest, config)`
   function.
2. Read `nx, ny, nz` from `manifest["grid_nodes"]`.
3. Check `manifest["grid_nodes"]` matches `config.grid_nodes`.
4. Check these exact shapes:

```python
{
    "velocity": (nx, ny, nz, 3),
    "pressure": (nx, ny, nz),
    "obstacle": (nx, ny, nz),
    "cell_face_x_m": (nx + 1,),
    "cell_face_y_m": (ny + 1,),
    "cell_face_z_m": (nz + 1,),
    "cell_center_x_m": (nx,),
    "cell_center_y_m": (ny,),
    "cell_center_z_m": (nz,),
    "cell_width_x_m": (nx,),
    "cell_width_y_m": (ny,),
    "cell_width_z_m": (nz,),
}
```

5. Raise `SnapshotResamplingError` with a clear array name and expected/actual
   shape when a mismatch is found.
6. Call the validator in `run()` after `_load_snapshot_fields()` and before
   `_build_fluid()`.

Required test coverage:

- Add a light synthetic unit-style test in
  `tests/integration/test_ansys_vertical_flap_traction_snapshot_resampling_artifacts.py`.
- The test must not initialize Taichi.
- It must call `_validate_snapshot_fields()` directly with synthetic arrays.
- It must prove valid synthetic fields pass.
- It must prove a shape mismatch raises `SnapshotResamplingError` and includes
  the offending array name.
- It must prove manifest/config grid mismatch raises `SnapshotResamplingError`.

## Artifact Regeneration Requirements

After code and tests are updated, regenerate the snapshot-resampling artifacts by
running:

```powershell
python validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_snapshot_resampling_matrix.py
```

Regenerated artifacts must include:

- `traction_snapshot_resampling_matrix.json`
- `traction_snapshot_resampling_matrix.csv`
- `traction_snapshot_resampling_history.json`
- `traction_snapshot_resampling_summary.md`
- `verification_snapshot_resampling_2026-06-26.md`
- `CHECKSUMS.sha256`
- completed marker diagnostics under `marker_diagnostics/`

Acceptance constraints:

- Still 5 completed formulations.
- Still 1 unsupported formulation.
- Shared snapshot SHA remains
  `3ea3f6e95ec1a43ddf9556785a87d423d25f68d59ce61696a00627786a8ea968`.
- Shared snapshot source commit remains
  `8488848d9302f7c05ffb8fd59342aec9d0a7e36f`.
- Offset 0.25 and 1.00 ratios should remain materially consistent with the
  current artifact unless the difference is explained as formatting-only or
  elapsed-time-only output churn.
- Checksums must be refreshed after artifact writes.

## Required Validation

Run syntax checks:

```powershell
python -m py_compile validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_snapshot_resampling_matrix.py tests/integration/test_ansys_vertical_flap_traction_snapshot_resampling_artifacts.py
```

Run the focused artifact contract test:

```powershell
python -m unittest tests.integration.test_ansys_vertical_flap_traction_snapshot_resampling_artifacts -v
```

Run the related ANSYS traction artifact slice:

```powershell
python -m unittest tests.integration.test_ansys_vertical_flap_traction_snapshot_resampling_artifacts tests.integration.test_ansys_vertical_flap_traction_shared_snapshot_artifacts tests.integration.test_ansys_vertical_flap_traction_probe_observability_artifacts tests.integration.test_ansys_vertical_flap_traction_formulation_artifacts -v
```

Run whitespace verification:

```powershell
git diff --check
```

If GPU/CUDA execution is unavailable, stop before artifact regeneration and
report the exact blocker. Do not fake regenerated artifacts.

## Non-Goals

- Do not change traction formulas.
- Do not change pressure sampling formulas.
- Do not add one-sided dual-face support in this patch.
- Do not change material parameters, geometry constants, grid dimensions, source
  schedules, support radii, or tolerance gates.
- Do not run or claim coupled 50-step FSI.
- Do not claim Fluent parity.
- Do not select a reference formulation.
- Do not hide a failing one-sided scenario by changing the scenario matrix.
- Do not broaden the patch into unrelated feedback projection guard work.

## Commit And Push Requirements

The user approved push after the modification is complete. Do not push before:

1. this goal file exists and is referenced by the short active goal,
2. code changes are implemented,
3. artifacts are regenerated or an honest GPU blocker is reported,
4. required tests/checks have been run,
5. `git diff --check` passes,
6. changed files are reviewed for unrelated modifications.

Use a conventional commit message, preferably:

```text
test: harden ANSYS traction snapshot resampling contracts
```

Push the completed branch to `origin`. The final report must include:

- final branch name,
- final commit hash,
- push target,
- validation commands and results,
- whether artifacts were regenerated,
- an explicit note that the patch remains sampling-only and does not claim
  Fluent parity or coupled FSI validation.

## Done Criteria

- Detailed goal file is committed.
- Short active goal references this file.
- `RESAMPLING_SCOPE_LIMIT` drives top-level and row-level scope.
- `source_script` is repo-relative.
- Summary contains candidate decision, blockers, scenario lists, non-claims,
  and next step.
- Candidate blocker details are non-empty.
- Snapshot shapes are validated before fluid restoration.
- Focused artifact tests cover every new contract.
- Regenerated artifacts pass contract tests.
- Branch is committed and pushed to `origin`.
