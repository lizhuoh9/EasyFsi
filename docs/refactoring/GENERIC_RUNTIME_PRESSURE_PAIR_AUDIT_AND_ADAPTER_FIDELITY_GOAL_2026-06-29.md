# Generic Runtime Pressure Pair Audit And Adapter Fidelity Goal - 2026-06-29

## Source Context

This goal follows the remote review of branch
`solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25` at commit
`b8344bbe9cab2f1117bc4797d4d347f651a4ece0`.

The previous checkpoint successfully moved the official ANSYS generic selected
formulation runner from a transition-seeded selected-anchor JSON path to
runtime-generated pressure sample pairs:

- `candidate_status = generic_solver_selected_formulation_step50_passed`
- `completed_step_count = 50`
- `pressure_pair_policy.mode = runtime_anchored_cell_pair`
- `pressure_pair_policy.pair_source_status = runtime_generated`
- `pressure_pair_policy.transition_backed = false`
- `pressure_pair_runtime_generation_complete = true`
- `transition_artifact_dependency = false`
- `invalid_marker_count_max = 0`
- `sample_pair_fallback_count_max = 0`
- `one_sided_marker_count_min = 24`
- `force_action_reaction_residual_max_n = 0`

The remaining work is not a new physics claim and not Fluent parity. The next
step is to harden the runtime pressure-pair path so it is CI-protected,
auditable, reproducible, and metadata-consistent with the actual half-domain
runtime used by the lower-level runner.

## Short Active Goal Reference

Use this compact goal text for the active goal entry:

```text
Implement docs/refactoring/GENERIC_RUNTIME_PRESSURE_PAIR_AUDIT_AND_ADAPTER_FIDELITY_GOAL_2026-06-29.md:
add CI execution for the focused runtime pressure-pair and generic architecture
tests; add artifact-only forbidden-state scanning; export a checksummed
pressure_sample_pair_map.json sidecar; record runtime marker geometry
provenance; fix ANSYS generic adapter runtime domain metadata; move runtime
anchored pair generation behind an injectable generic provider contract; add
provider boundary tests; regenerate official artifacts; verify; commit; and
push branch solver/ansys-vertical-flap-feedback-projection-guards-2026-06-25.
```

## Objective

Upgrade the runtime pressure-pair implementation from "passes the official
matrix gate" to "CI-protected, auditable, reproducible, and faithful to the
actual runtime domain metadata" without claiming Fluent parity and without
changing the physical solver behavior beyond the pressure-pair contract surface.

## Scope

### In Scope

- Update GitHub Actions so the focused runtime pressure-pair tests and generic
  architecture tests are executed, not only compiled.
- Add a forbidden-state scan that is scoped to official generated artifacts and
  runtime outputs, so historical goal text and negative test assertions do not
  create false positives.
- Export a runtime pressure pair sidecar artifact:
  `validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/pressure_sample_pair_map.json`.
- Add the sidecar artifact to `CHECKSUMS.sha256`.
- Ensure the sidecar content comes from `PressureSamplePairMap.as_diagnostics()`
  or an equivalent deterministic diagnostics payload.
- Record runtime marker geometry provenance so the official artifact identifies
  which runtime marker geometry produced the pressure pair map.
- Fix `AnsysVerticalFlapProblem.to_fsi_problem()` metadata so the generic
  `FluidDomain` exposes the actual lower-symmetry-half runtime domain used by
  the official runner.
- Expose or record a runtime discretization model that is not misleading for
  3D `grid_nodes=(4, 32, 64)` and half-domain bounds.
- Move runtime anchored pair generation behind an injectable provider contract
  instead of leaving the official implementation only as an ANSYS benchmark
  helper.
- Add provider boundary tests for axis handling, offset validation, domain/grid
  validation, inside/outside separation, deterministic SHA behavior, and
  grid-bound cell generation.
- Regenerate official generic selected formulation artifacts after the code
  changes.

### Out Of Scope

- Do not fabricate Fluent source data.
- Do not mark Fluent reference complete.
- Do not claim Fluent parity.
- Do not alter Fluent parity gates except to keep them fail-closed.
- Do not add heavy coupled runs to CI.
- Do not merge or split the branch into PRs in this task.
- Do not delete historical selected-anchor artifacts; they remain valid replay
  and debug evidence.
- Do not use repository-wide forbidden-string scans without allowlists; docs and
  negative tests may legitimately mention old transition states.

## Required Implementation Details

### 1. CI Execution For Focused Tests

Update `.github/workflows/ansys-vertical-flap-validation.yml` so CI explicitly
runs the focused tests that protect this stage:

```powershell
python -m unittest -v `
  tests.solvers.test_pressure_sample_pair_provider_contract `
  tests.contracts.test_generic_fsi_solver_architecture `
  tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_generic_problem_defaults_to_runtime_pressure_pairs_without_json `
  tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_generic_replay_pressure_pairs_require_anchor_json `
  tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_generic_pressure_pair_provider_mode_is_fail_closed `
  tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_runtime_pressure_pair_mode_ignores_supplied_anchor_json_source `
  tests.integration.test_ansys_vertical_flap_generic_solver_artifacts `
  tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic
```

The workflow may keep py_compile steps, but compile-only coverage is not enough
for the pressure-pair provider and generic architecture contracts.

### 2. Artifact-Only Forbidden-State Scan

Add a lightweight script or workflow step that scans generated official
artifacts and runtime outputs for forbidden transition-state strings:

```text
transition_seeded_from_anchor_artifact
runtime_pressure_pair_generation_pending
```

The scan must cover at least:

```text
validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/
```

It should not fail because goal documents or unit tests mention these strings
as historical context or negative assertions.

### 3. Pressure Pair Sidecar Artifact

Generate:

```text
validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/pressure_sample_pair_map.json
```

The sidecar must contain at least:

```json
{
  "provider_mode": "runtime_anchored_cell_pair",
  "pair_map_sha256": "...",
  "fallback_count": 0,
  "selected_count": 24,
  "pairs": [
    {
      "marker_index": 0,
      "region_id": "101",
      "inside_cell": [0, 0, 0],
      "outside_cell": [0, 0, 0],
      "sample_status": "runtime_generated",
      "fallback_status": "no_fallback",
      "diagnostic_reason": "runtime_anchored_cell_pair"
    }
  ]
}
```

The official integration test must verify:

- `pairs` length is 24 for the official artifact.
- `marker_index` is contiguous from 0 through 23.
- region IDs include the expected primary and secondary marker sets.
- every `inside_cell` and `outside_cell` is inside `grid_nodes=(4, 32, 64)`.
- every `inside_cell` differs from the matching `outside_cell`.
- `fallback_count = 0`.
- `selected_count = 24`.
- `pair_map_sha256` equals the SHA recomputed from the pair rows.
- the sidecar is listed in `CHECKSUMS.sha256`.

### 4. Runtime Marker Geometry Provenance

Record a deterministic runtime marker geometry hash in the install report and
official artifacts. The artifact row should no longer leave the current runtime
marker geometry provenance empty when the source is `runtime_generated`.

Required row-level fields:

```text
pressure_pair_anchor_current_marker_geometry_sha256 != ""
pressure_pair_anchor_map_sha256 != ""
pressure_pair_anchor_source = runtime_generated
```

It is acceptable for legacy replay/source snapshot fields to remain empty for
runtime mode, but the current runtime marker geometry hash must be present.

### 5. Adapter Metadata Fidelity

Fix `AnsysVerticalFlapProblem.to_fsi_problem()` so the generic domain metadata
does not report a full-height domain when the official runner uses the
lower-symmetry-half domain.

Required behavior:

- `FluidDomain.bounds_m[1][1]` should match the runtime half-height
  `0.5 * duct_height_m`, i.e. `0.02` for the current ANSYS vertical flap config.
- `FluidDomain.grid_nodes` remains `(4, 32, 64)`.
- metadata exposes a non-misleading runtime discretization model, such as
  `cartesian-3d-half-domain`, while preserving any conceptual case metadata
  separately if needed.
- tests must assert the generic adapter's domain metadata matches the lower
  runner's runtime domain.

### 6. Injectable Runtime Provider Contract

Move runtime anchored pair generation behind a generic provider implementation:

```python
class RuntimeAnchoredCellPairProvider:
    def compute_pairs(
        self,
        markers,
        fluid_state,
        interface_surface,
    ) -> PressureSamplePairMap:
        ...
```

The official runner may still call a helper, but the helper should delegate to
this provider or use the same provider contract. The contract should remain
case-agnostic and must not depend on ANSYS, Fluent, or vertical-flap names.

The provider should accept the geometry/grid settings it needs explicitly:

- `domain_bounds_m`
- `grid_nodes`
- `anchor_axis`
- `inside_axis_position_m`
- `outside_axis_offset_cells`

The existing function `compute_runtime_anchored_cell_pair_map(...)` may remain
as the pure computation primitive.

### 7. Boundary And Failure Tests

Extend `tests/solvers/test_pressure_sample_pair_provider_contract.py` with
focused tests for:

- `anchor_axis = 0`, `1`, and `2`.
- `outside_axis_offset_cells <= 0` fails closed.
- zero anchor-axis marker normal fails closed.
- non-positive grid nodes fail closed.
- non-positive or non-finite domain spacing fails closed.
- inside/outside cells differ for valid generated pairs.
- positive and negative marker normals produce opposite outside directions.
- generated cells stay inside grid bounds.
- pair-map SHA behavior is explicit and tested.

If marker positions or inside-axis positions outside the domain are clamped, the
tests should state that clearly. If they fail closed instead, the tests should
state that clearly. The behavior must be deterministic either way.

## Official Artifact Requirements

After regeneration, the official matrix must still report:

```text
candidate_status = generic_solver_selected_formulation_step50_passed
pressure_pair_policy.mode = runtime_anchored_cell_pair
pressure_pair_policy.pair_source_status = runtime_generated
pressure_pair_policy.transition_backed = false
pressure_pair_runtime_generation_complete = true
transition_artifact_dependency = false
completed_step_count = 50
invalid_marker_count_max = 0
sample_pair_fallback_count_max = 0
one_sided_marker_count_min >= 24
force_action_reaction_residual_max_n <= 1e-8
fluent_parity_claimed = false
fluent_parity_status = blocked_reference_incomplete
```

The generated official artifact directory must not contain:

```text
transition_seeded_from_anchor_artifact
runtime_pressure_pair_generation_pending
```

## Verification Plan

Use the reliable local interpreter:

```text
D:\working\taichi\env\python.exe
```

Run:

```text
D:\working\taichi\env\python.exe -m py_compile simulation_core/generic_fsi_solver.py simulation_core/pressure_sample_pairs.py cases/ansys_vertical_flap_fsi.py benchmarks/official/solid_mpm_fsi_runner.py validation_runs/ansys_vertical_flap_fsi/scripts/run_ansys_vertical_flap_generic_solver.py tests/contracts/test_generic_fsi_solver_architecture.py tests/integration/test_ansys_vertical_flap_generic_solver_artifacts.py tests/solvers/test_pressure_sample_pair_provider_contract.py tests/cases/test_ansys_vertical_flap_fsi.py
D:\working\taichi\env\python.exe validation_runs/ansys_vertical_flap_fsi/scripts/run_ansys_vertical_flap_generic_solver.py
D:\working\taichi\env\python.exe -m unittest -v tests.solvers.test_pressure_sample_pair_provider_contract tests.contracts.test_generic_fsi_solver_architecture tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_generic_problem_defaults_to_runtime_pressure_pairs_without_json tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_generic_replay_pressure_pairs_require_anchor_json tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_generic_pressure_pair_provider_mode_is_fail_closed tests.cases.test_ansys_vertical_flap_fsi.AnsysVerticalFlapFsiSmokeTests.test_runtime_pressure_pair_mode_ignores_supplied_anchor_json_source tests.integration.test_ansys_vertical_flap_generic_solver_artifacts tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic
D:\working\taichi\env\python.exe -m unittest -v tests.integration.test_ansys_vertical_flap_fluent_reference_collection_artifacts tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_artifacts tests.integration.test_ansys_vertical_flap_traction_selected_formulation_fluent_parity_comparison_logic
git diff --check
```

Also run an artifact-only forbidden-state scan over:

```text
validation_runs/ansys_vertical_flap_fsi/generic_solver_selected_formulation_diagnostics/
```

## Push Condition

Push only after:

- this goal file exists and the short active goal references it;
- focused tests are run in CI workflow, not just compiled;
- artifact-only forbidden-state scan exists and passes locally;
- official generic artifact exports `pressure_sample_pair_map.json`;
- sidecar is listed in `CHECKSUMS.sha256`;
- runtime marker geometry SHA is present in official artifacts;
- generic adapter metadata matches the runtime half-domain;
- provider boundary/failure tests pass;
- Fluent reference remains incomplete and no Fluent parity claim is made;
- staged diff has been reviewed;
- local verification passes.
