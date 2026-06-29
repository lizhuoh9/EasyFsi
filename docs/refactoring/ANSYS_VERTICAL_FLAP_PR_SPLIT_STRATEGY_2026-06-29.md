# ANSYS Vertical-Flap PR Split Strategy 2026-06-29

## Purpose

This branch can be reviewed as one fail-closed validation package, but it is large enough that reviewers may ask for smaller pull requests. If that happens, split by review concern rather than by file type.

## PR 1 - Layout And Package Relocation

- Scope: repository layout movement and package import stabilization only.
- Files: package `__init__.py` files, import path adapters, layout docs, and tests that prove imports remain stable.
- Risk: accidental import drift or duplicate entry points.
- Tests: import smoke tests, py-compile for moved modules, and existing wrapper-path tests.
- Artifacts: no validation artifacts should change unless a path manifest records the relocation.
- Do-not-merge-with: benchmark namespace changes or runtime solver behavior.

## PR 2 - `benchmarks.official` Namespace Migration

- Scope: official benchmark namespace and runner routing.
- Files: `benchmarks/official/*`, benchmark docs, and runner compatibility tests.
- Risk: changing public benchmark entry points without preserving compatibility.
- Tests: benchmark import tests, runner CLI smoke tests, and any benchmark artifact path assertions.
- Artifacts: benchmark metadata only; no Fluent parity artifact promotion.
- Do-not-merge-with: ANSYS selected-formulation diagnostics or Fluent reference gates.

## PR 3 - ANSYS Selected-Formulation Diagnostics And Artifacts

- Scope: ANSYS vertical-flap selected-formulation diagnostics, provenance, and generated artifact checks.
- Files: `validation_runs/ansys_vertical_flap_fsi/*selected_formulation*`, associated tests, and diagnostics docs.
- Risk: overclaiming short diagnostic runs as full physical validation.
- Tests: selected-formulation artifact tests, comparison-logic tests, policy/hygiene checks.
- Artifacts: selected-formulation diagnostics and checksums only.
- Do-not-merge-with: generic solver boundary refactors or Fluent reference schema promotion.

## PR 4 - `generic_fsi_solver` And Runtime Pressure-Pair Audit

- Scope: generic solver boundary, runtime pressure-pair provider contract, and architecture evidence.
- Files: `simulation_core/generic_fsi_solver.py`, `simulation_core/pressure_sample_pairs.py`, related case adapters, and architecture tests.
- Risk: moving case-specific pressure-pair assumptions into generic solver logic.
- Tests: `tests.contracts.test_generic_fsi_solver_architecture`, `tests.solvers.test_pressure_sample_pair_provider_contract`, and generic solver artifact tests.
- Artifacts: generic solver selected-formulation diagnostics.
- Do-not-merge-with: Fluent reference collection or real Fluent parity claim changes.

## PR 5 - Fluent Reference Schema Gate And Merge Readiness

- Scope: Fluent reference schema validation, source export gate, fail-closed parity runner, policy reports, and merge checklist.
- Files: `validation_runs/ansys_vertical_flap_fsi/scripts/fluent_*`, Fluent reference source exports, artifact manifests, policy reports, branch review docs, and import-gate docs.
- Risk: accidentally treating schema-only or synthetic data as real Fluent truth.
- Tests: Fluent contract schema, source export schema, collection artifacts, parity artifacts, synthetic pipeline, policy reports, branch review docs, real Fluent import gate docs, and artifact hygiene.
- Artifacts: Fluent collection diagnostics, Fluent parity diagnostics, policy reports, and checksums.
- Do-not-merge-with: solver physics changes or benchmark namespace migration.

## Single-PR Fallback Review Order

If reviewers keep the branch as one PR, review in this order:

1. Claim boundary docs and merge checklist.
2. Fluent reference schema and source export validators.
3. Artifact generator manifests and checksums.
4. Policy/hygiene checkers and reports.
5. Synthetic pipeline tests.
6. Generic solver and runtime pressure-pair contracts.
7. Workflow coverage and final diff scan.
