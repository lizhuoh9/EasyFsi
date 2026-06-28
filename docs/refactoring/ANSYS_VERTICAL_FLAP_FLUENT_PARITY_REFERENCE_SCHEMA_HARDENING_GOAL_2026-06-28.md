# ANSYS vertical-flap Fluent parity reference schema hardening goal

Date: 2026-06-28

Source request: follow-up review of `solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`, written against commit `330bd83a9798a011e7c141d5b5062c0ffa008271`, asking to enter the Fluent parity reference contract / parity artifact phase after `selected_formulation_coupled_step50_passed`.

Current local baseline: commit `eaf4bab22aa4f77398114a02c183b05622a155d6` already added the first Fluent parity fail-closed contract, runner, artifact test, workflow cheap checks, and committed parity artifacts. This goal hardens that implementation against the more specific schema and semantic requirements in the follow-up review.

## Objective

Tighten the existing Fluent parity implementation so the reference contract explicitly exposes the attachment-requested `source_provenance` and `simulation` schema, while preserving the current fail-closed semantics:

- `candidate_status` remains `fluent_parity_blocked_reference_incomplete`.
- active blockers remain exactly `fluent_reference_incomplete` and `no_fluent_parity_claim`.
- `no_fluent_parity_claim` is not retired.
- no Fluent data is fabricated.
- no heavy coupled runner is rerun.
- no solver, traction, or coupling infrastructure is changed.

## Non-goals

This goal does not include:

- Changing material parameters.
- Changing geometry.
- Changing the selected formulation.
- Relaxing selected coupled step50 gates.
- Running the step10/30/50 heavy coupled runner.
- Running heavy simulation in CI.
- Claiming `fluent_parity_validated`.
- Retiring `no_fluent_parity_claim`.
- Inventing Fluent reference values.

## Phase 0: preserve existing goal and artifact boundary

The existing detailed goal remains:

`docs/refactoring/ANSYS_VERTICAL_FLAP_FLUENT_PARITY_SELECTED_FORMULATION_GOAL_2026-06-27.md`

This hardening goal supplements it with schema precision. The implementation should update that existing goal as needed so future readers see the exact reference-contract schema requirements in one place.

## Phase 1: harden step50 non-parity test

Update `tests/integration/test_ansys_vertical_flap_traction_selected_formulation_coupled_step50_artifacts.py` to ensure the step50 pass cannot be interpreted as Fluent parity:

- `candidate_status == "selected_formulation_coupled_step50_passed"`.
- active blockers equal exactly `{ "no_fluent_parity_claim" }`.
- retired blockers equal exactly `[ "long_coupled_validation_pending" ]`.
- summary contains `does not claim Fluent parity`.
- summary does not contain `Fluent parity validated`.
- matrix payload does not contain `fluent_parity_claim`.
- step50 row is locked to:
  - `smoke_status == "passed"`
  - `run_status == "completed"`
  - `completed_step_count == 50`
  - `requested_step_count == 50`
  - `invalid_marker_count_max == 0`
  - `one_sided_marker_count_min >= 24`
  - `anchor_selected_marker_count_min >= 24`
  - `anchor_fallback_marker_count_max == 0`

## Phase 2: harden Fluent reference contract schema

Update:

`validation_runs/ansys_vertical_flap_fsi/fluent_reference/fluent_reference_contract_2026-06-27.json`

Required additions:

- `source_provenance`
- `simulation`

The contract should retain the fields already used by the current runner:

- `provenance_status`
- `step_count`
- `time_step_s`
- `geometry`
- `material`
- `flow`
- `reference_metrics`
- `tolerances`
- `missing_reference_metrics`
- `contract_status`

The new `source_provenance` object must be explicit and should contain:

- `document`
- `run_id`
- `author`
- `date`
- `status`

Because no provenance-backed Fluent reference values are available, these fields should not pretend otherwise. Use empty strings or `missing`/`not_collected` style status values.

The new `simulation` object must contain:

- `step_count == 50`
- `time_step_s == 0.0005`
- `total_time_s == 0.025`

Reference metrics remain missing until real Fluent data is collected. Required missing metrics:

- `tip_displacement_m`
- `max_displacement_m`
- `force_z_N`
- `flow_rate_m3s`
- `pressure_range_pa`

## Phase 3: keep parity runner compatible and explicit

Update `validation_runs/ansys_vertical_flap_fsi/scripts/run_traction_selected_formulation_fluent_parity.py` only if needed so generated parity artifacts carry or expose the new schema fields.

The runner must continue to:

- Read committed step50 matrix/history artifacts.
- Read the Fluent reference contract.
- Avoid rerunning heavy coupled simulation.
- Keep `candidate_status == "fluent_parity_blocked_reference_incomplete"` when `contract_status != "fluent_reference_complete"`.
- Keep `no_fluent_parity_claim` active.
- Keep `historical_blockers_retired == []`.

## Phase 4: harden Fluent parity artifact tests

Update `tests/integration/test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts.py` so it checks:

- Fluent reference contract contains `source_provenance`.
- Fluent reference contract contains `simulation`.
- `simulation.step_count == 50`.
- `simulation.time_step_s == 0.0005`.
- `simulation.total_time_s == 0.025`.
- `source_provenance.status` is not a completed/validated provenance claim.
- `contract_status == "fluent_reference_incomplete"`.
- all missing reference metrics are explicitly missing with null values.
- generated parity matrix records the updated reference contract SHA.
- generated parity metrics still expose displacement, force, flow/outlet, pressure, and metadata groups.
- summary does not overclaim and does not contain `Fluent parity validated`.

## Phase 5: regenerate parity artifacts

Run the lightweight parity runner:

`D:\working\taichi\env\python.exe validation_runs\ansys_vertical_flap_fsi\scripts\run_traction_selected_formulation_fluent_parity.py`

This regenerates only:

`validation_runs/ansys_vertical_flap_fsi/traction_selected_formulation_fluent_parity_diagnostics/`

It must not rerun the heavy coupled step10/30/50 simulation.

## Phase 6: verification

Use:

`D:\working\taichi\env\python.exe`

Run:

- py_compile for the parity runner and both artifact tests.
- `tests.integration.test_ansys_vertical_flap_traction_selected_formulation_coupled_step50_artifacts`.
- `tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts`.
- `git diff --check`.

## Phase 7: commit and push

Before committing:

- Confirm worktree changes are limited to goal docs, reference contract, runner/test updates, and regenerated parity artifacts.
- Confirm no heavy runner outputs changed.
- Confirm `candidate_status` remains `fluent_parity_blocked_reference_incomplete`.

Commit and push to:

`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25`

Do not mark Fluent parity validated in this goal.
