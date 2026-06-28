# ANSYS vertical-flap selected formulation Fluent parity goal

Date: 2026-06-27

Source review anchor: remote `lizhuoh9/EasyFsi` branch `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25` at commit `330bd83a9798a011e7c141d5b5062c0ffa008271`.

## Current accepted baseline

The selected-formulation coupled validation chain has advanced to a real 50-step staged pass:

- Shared snapshot selection has passed.
- Fixed-solid selected formulation has passed.
- Selected formulation coupled 5-step smoke has passed.
- Selected formulation coupled step10, step30, and step50 have passed.
- `candidate_status` in the step50 artifact is exactly `selected_formulation_coupled_step50_passed`.
- The only active blocker remaining is `no_fluent_parity_claim`.
- `long_coupled_validation_pending` is retired.

This goal starts the independent Fluent parity phase. It must not reinterpret the step50 coupled pass as Fluent parity.

## Objective

Create an independent ANSYS Fluent parity artifact surface for the selected formulation by comparing the committed 50-step coupled evidence against an explicit Fluent reference contract.

The implementation must:

- Harden the step50 artifact test so it explicitly rejects Fluent parity overclaiming.
- Add this detailed Fluent parity goal as the source of truth.
- Add a Fluent reference contract file with clear provenance fields.
- Add a Fluent parity runner that reads committed step50 evidence and the reference contract.
- Add Fluent parity matrix, history, summary, scenario diagnostics, and checksums artifacts.
- Add artifact tests that enforce exact current candidate status and blockers.
- Add workflow cheap checks for py_compile and artifact tests.
- Avoid rerunning heavy coupled simulation in CI or the parity runner.

## Non-goals

This goal does not include:

- Material parameter tuning.
- Geometry tuning.
- Selected formulation replacement.
- Relaxing the step50 coupled validation gates.
- Treating displacement alone as complete Fluent parity.
- Claiming Fluent parity while Fluent reference data is incomplete.
- Running heavy coupled simulation in GitHub Actions.
- Adding more traction or coupling infrastructure.

## Phase 0: harden step50 non-parity semantics

Update `tests/integration/test_ansys_vertical_flap_traction_selected_formulation_coupled_step50_artifacts.py` so the existing step50 pass cannot be mistaken for Fluent parity:

- Summary must contain `does not claim Fluent parity`.
- Summary must not contain `Fluent parity validated`.
- Matrix payload must not contain `fluent_parity_claim`.
- Active blockers must equal exactly `{ "no_fluent_parity_claim" }`.
- Retired blockers must equal exactly `[ "long_coupled_validation_pending" ]`.
- Step50 row must be locked to:
  - `smoke_status == "passed"`
  - `completed_step_count == 50`
  - `invalid_marker_count_max == 0`
  - `one_sided_marker_count_min >= 24`
  - `anchor_selected_marker_count_min >= 24`
  - `anchor_fallback_marker_count_max == 0`

This phase should not rerun the heavy step10/30/50 runner unless the test exposes missing artifact fields.

## Phase 1: Fluent reference contract

Add:

`validation_runs/ansys_vertical_flap_fsi/fluent_reference/fluent_reference_contract_2026-06-27.json`

The contract must be explicit and machine-readable. Required top-level fields:

- `case`
- `source`
- `source_provenance`
- `provenance_status`
- `simulation`
- `step_count`
- `time_step_s`
- `geometry`
- `material`
- `flow`
- `reference_metrics`
- `tolerances`
- `missing_reference_metrics`
- `contract_status`

If Fluent reference metrics are not available with reliable provenance, represent them as `missing`; do not fabricate values. The current expected status is:

- `contract_status == "fluent_reference_incomplete"`
- missing metrics include displacement, force, flow, and pressure reference data unless real sourced values are present.

The `source_provenance` object must make missing provenance explicit with
fields for `document`, `run_id`, `author`, `date`, and `status`. The
`simulation` object must mirror the ANSYS vertical-flap comparison horizon:
`step_count == 50`, `time_step_s == 0.0005`, and `total_time_s == 0.025`.

## Phase 2: Fluent parity runner

Add:

`validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_selected_formulation_fluent_parity.py`

The runner writes:

- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/traction_selected_formulation_fluent_parity_matrix.json`
- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/traction_selected_formulation_fluent_parity_matrix.csv`
- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/traction_selected_formulation_fluent_parity_history.json`
- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/traction_selected_formulation_fluent_parity_summary.md`
- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/scenario_diagnostics/selected_formulation_fluent_parity.json`
- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/CHECKSUMS.sha256`

The runner must read, not regenerate:

- `validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_coupled_step50_diagnostics/traction_selected_formulation_coupled_step50_matrix.json`
- `validation_runs/ansys_vertical_flap_fsi/fluent_reference/fluent_reference_contract_2026-06-27.json`

The runner must fail closed when the reference contract is incomplete:

- `candidate_status == "fluent_parity_blocked_reference_incomplete"`
- active blockers include:
  - `fluent_reference_incomplete`
  - `no_fluent_parity_claim`
- `historical_blockers_retired == []`

If future sourced Fluent reference data completes the contract and all gates pass, only then may a later artifact set:

- `candidate_status == "fluent_parity_validated"`
- `historical_blockers_retired == [ "no_fluent_parity_claim" ]`

## Phase 3: parity metrics

The parity artifact must include a structured parity metric surface even while blocked by missing reference data:

### Displacement parity

- step50 max displacement from selected formulation source artifact.
- step50 tip displacement if present in source history.
- Fluent reference displacement target or `missing`.
- relative error if both values exist, otherwise `null`.
- gate status, expected initially `blocked_reference_missing`.

### Force parity

- marker force z history from the step50 source artifact when present.
- step50 marker force z.
- primary/secondary face force z if present in source history.
- force sign flip count from source row.
- Fluent force reference target or `missing`.
- relative error if both values exist, otherwise `null`.
- gate status, expected initially `blocked_reference_missing`.

### Flow/outlet parity

- source flow finite status.
- available source flow/outlet balance fields if present.
- Fluent flow reference target or `missing`.
- relative error if both values exist, otherwise `null`.
- gate status, expected initially `blocked_reference_missing`.

### Pressure sanity

- step50 pressure min/max or max absolute pressure from source artifact.
- pressure growth ratio from source row.
- pressure finite status.
- Fluent pressure sanity target or `missing`.
- gate status, expected initially `blocked_reference_missing`.

### Metadata provenance

- selected formulation candidate.
- source step50 artifact path and SHA256.
- Fluent reference contract path and SHA256.
- geometry/material/time-step/step-count metadata.
- selected anchor marker source and SHA256.
- anchor map SHA.
- source script path.

## Phase 4: artifact tests

Add:

`tests/integration/test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts.py`

The test must verify:

- Matrix JSON, CSV, history JSON, summary markdown, scenario diagnostics, and checksums exist.
- `source_script` is repo-relative.
- Source step50 artifact exists and SHA256 matches.
- Source step50 `candidate_status == "selected_formulation_coupled_step50_passed"`.
- Fluent reference contract exists and SHA256 matches.
- Reference formulation candidate remains exactly `anchored_dual_face_pressure_pair_with_per_face_one_sided`.
- Parity metric groups exist for displacement, force, flow/outlet, pressure, and metadata.
- Current candidate status is exactly `fluent_parity_blocked_reference_incomplete`.
- Active blockers are exactly `{ "fluent_reference_incomplete", "no_fluent_parity_claim" }`.
- Retired blockers are exactly `[]`.
- Summary does not overclaim parity and does not contain `Fluent parity validated`.
- Checksums cover all committed parity artifacts and scenario diagnostics.

## Phase 5: workflow cheap checks

Update `.github/workflows/ansys-vertical-flap-validation.yml`:

- Add py_compile for `validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py`.
- Add unittest for `tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts`.

Do not add the parity runner or any heavy coupled simulation to GitHub Actions execution.

## Phase 6: verification

Use the trusted local interpreter:

`D:\working\taichi\env\python.exe`

Run:

- py_compile for the new parity runner and tests.
- the hardened step50 artifact test.
- the new Fluent parity artifact test after generating parity artifacts.
- `git diff --check`.

The parity runner is lightweight and may be run locally because it reads committed artifacts and writes comparison artifacts only.

## Phase 7: push

Before pushing:

- Confirm the worktree contains only goal, source, test, workflow, reference contract, and generated parity artifact changes.
- Review `git diff --cached --stat`.
- Commit with a conventional commit message.
- Push the current branch to `origin`.

Do not retire `no_fluent_parity_claim` in this goal unless real, complete, provenance-backed Fluent reference data exists and all parity gates pass.
